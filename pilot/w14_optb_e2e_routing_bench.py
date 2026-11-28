"""Option B Exp-1: True O(1) Stateful Steady-State & Symmetric E2E Routed Crossover Benchmark.

Fixes both Codex B1 & Claude M-D timing/crossover critiques without altering historical runners:
  1. True O(1) prefix-sum lookups `G_cum[:, T] - G_cum[:, b]` and `L_cum[:, T] - L_cum[:, b]`
     precomputed at prefill (no suffix token summation inside timed edit or detector path).
  2. In-place `[p0:b_end)` KV write (`patch_kv_slice`) for fixed-length (`delta_L == 0`) edits
     (zero full-KV `torch.cat` copy), benchmarked side-by-side against variable-length `delta_L != 0`
     Delta-RoPE + `torch.cat` (`apply_delta_rope_to_kv_cache`).
  3. Monotone slow-to-fast (`diff_0 <= 0 and diff_1 > 0` sustained) crossover interpolation plus
     first/last crossing diagnostics (eliminating silent jitter artifacts).
  4. Symmetric request-path E2E wall-clock sweep (`with-H2D` vs `GPU-resident`) with fixed-step
     decode (`eval_fixed_steps_in_place`, zero EOS early-exit), genuine `c_hat_mean` & `gamma_mean`
     features, explicit analytical synthesis, and a genuine conditional live-routed execution arm.

Supports `--synthetic-dry-run` on CPU before running on GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HOME", r"D:\applicationData\HuggingFace")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from w5_9b_e5_runner import (  # noqa: E402
    clone_cache,
    compute_state_delta_fp64,
    count_linear_layers,
    extract_kv_slice,
    get_eos_ids,
    has_logits_to_keep_arg,
    make_position_ids,
    move_cache_device,
    patch_kv_slice,
    run_chunked_prefill,
    snap_states,
)
from w10_p1_3_e4_e7_counterfact import apply_delta_rope_to_kv_cache  # noqa: E402
from w13_fallback_runner import (  # noqa: E402
    FROZEN_LR_TAU_DEV_9B,
    FROZEN_OSC_TAU_DEV_2B,
    freeze_lr_from_dev_split,
    score_frozen_lr,
    score_frozen_osc,
)


def _sync(device: torch.device | str) -> float:
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)
    return time.perf_counter()


def _med_iqr(xs: List[float], include_raw: bool = True) -> Dict[str, Any]:
    s = sorted(xs)
    n = len(s)
    med = float(s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])) if n else 0.0
    q25 = float(s[max(0, min(n - 1, int(0.25 * n)))]) if n else 0.0
    q75 = float(s[max(0, min(n - 1, int(0.75 * n)))]) if n else 0.0
    p95 = float(s[max(0, min(n - 1, int(0.95 * n)))]) if n else 0.0
    out: Dict[str, Any] = {
        "median": round(med, 4),
        "iqr": round(q75 - q25, 4),
        "mean": round(sum(s) / max(1, n), 4),
        "p95": round(p95, 4),
    }
    if include_raw:
        out["raw_ms"] = [round(float(x), 4) for x in xs]
    return out


def eval_fixed_steps_in_place(
    model: Any,
    base_model: Any,
    ctx_cache: Any,
    T_ctx: int,
    ids_q: torch.Tensor,
    use_logits_to_keep: bool = True,
    decode_steps: int = 8,
) -> Dict[str, Any]:
    """Run query prefill + exact `decode_steps - 1` decode tokens without EOS early-exit."""
    q_len = int(ids_q.shape[1])
    q_pos = make_position_ids(T_ctx, q_len, ids_q.device, base_model)
    kwargs: Dict[str, Any] = {
        "input_ids": ids_q,
        "position_ids": q_pos,
        "use_cache": True,
        "past_key_values": ctx_cache,
    }
    if use_logits_to_keep:
        kwargs["logits_to_keep"] = 1
    out = model(**kwargs)
    first_logits = out.logits[:, -1, :].detach().float().clone()
    q_cache = out.past_key_values
    del out

    cur_pos = T_ctx + q_len
    nxt = first_logits.argmax(dim=-1, keepdim=True)
    gen_ids: List[int] = [int(nxt.item())]

    for _ in range(max(0, decode_steps - 1)):
        step_pos = make_position_ids(cur_pos, 1, ids_q.device, base_model)
        step_kwargs: Dict[str, Any] = {
            "input_ids": nxt,
            "position_ids": step_pos,
            "use_cache": True,
            "past_key_values": q_cache,
        }
        if use_logits_to_keep:
            step_kwargs["logits_to_keep"] = 1
        out = model(**step_kwargs)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        q_cache = out.past_key_values
        del out
        cur_pos += 1
        gen_ids.append(int(nxt.item()))

    if len(gen_ids) != decode_steps:
        raise RuntimeError(f"Expected {decode_steps} fixed decode steps, executed {len(gen_ids)}")
    return {"first_logits": first_logits, "gen_ids": gen_ids, "decode_steps_executed": len(gen_ids)}


def capture_prefix_cum_decay(
    model: Any,
    base_model: Any,
    ids_ctx: torch.Tensor,
    n_linear: int,
    use_l2k: bool,
    return_l_cum: bool = False,
) -> Any:
    """Run context prefill while recording per-layer cumulative `G_cum` (and optionally `L_cum`)."""
    g_chunks: Dict[int, List[torch.Tensor]] = {}
    l_chunks: Dict[int, List[torch.Tensor]] = {}
    hooks = []
    text_mod = getattr(base_model, "language_model", base_model)

    def _make_a_hook(mod: Any):
        l_idx = int(mod.layer_idx)

        def _hook(_m, _inp, a_out):
            g = -mod.A_log.double().exp() * F.softplus(a_out.double() + mod.dt_bias.double())
            g_chunks.setdefault(l_idx, []).append(g.detach())

        return _hook

    def _make_b_hook(mod: Any):
        l_idx = int(mod.layer_idx)

        def _hook(_m, _inp, b_out):
            beta = torch.sigmoid(b_out.double())
            l_step = -torch.log((1.0 - beta).clamp(min=1e-12))
            l_chunks.setdefault(l_idx, []).append(l_step.detach())

        return _hook

    for mod in text_mod.modules():
        if hasattr(mod, "in_proj_a") and hasattr(mod, "A_log") and hasattr(mod, "dt_bias") and hasattr(mod, "layer_idx"):
            hooks.append(mod.in_proj_a.register_forward_hook(_make_a_hook(mod)))
            if return_l_cum and hasattr(mod, "in_proj_b"):
                hooks.append(mod.in_proj_b.register_forward_hook(_make_b_hook(mod)))

    try:
        _, c_ctx = run_chunked_prefill(model, base_model, ids_ctx, 0, None, 2048, use_l2k)
    finally:
        for hk in hooks:
            hk.remove()

    g_cum: Dict[int, torch.Tensor] = {}
    for l_idx, chunks in g_chunks.items():
        g_full = torch.cat(chunks, dim=1)  # (1, T, H_v)
        zeros = torch.zeros((1, 1, g_full.shape[-1]), device=g_full.device, dtype=torch.float64)
        g_cum[l_idx] = torch.cat([zeros, g_full.cumsum(dim=1)], dim=1)
    assert len(g_cum) == n_linear, f"Expected {n_linear} hooked linear layers, got {len(g_cum)}"
    if not return_l_cum:
        return c_ctx, g_cum

    l_cum: Dict[int, torch.Tensor] = {}
    for l_idx, chunks in l_chunks.items():
        l_full = torch.cat(chunks, dim=1)
        zeros = torch.zeros((1, 1, l_full.shape[-1]), device=l_full.device, dtype=torch.float64)
        l_cum[l_idx] = torch.cat([zeros, l_full.cumsum(dim=1)], dim=1)
    assert len(l_cum) == n_linear, f"Expected {n_linear} hooked in_proj_b layers, got {len(l_cum)}"
    return c_ctx, g_cum, l_cum


def apply_jlocal_update_o1(
    cache_ctx: Any,
    delta: Dict[str, List[Optional[torch.Tensor]]],
    g_cum: Dict[int, torch.Tensor],
    block_end: int,
    T_ctx: int,
    w_conv: int,
) -> None:
    """True O(1) steady-state update using precomputed cumulative sum lookup G[T] - G[b]."""
    shift = T_ctx - block_end
    for i, layer in enumerate(cache_ctx.layers):
        d_rec = delta["rec"][i] if i < len(delta["rec"]) else None
        if d_rec is not None and i in g_cum:
            g_diff = g_cum[i][:, T_ctx, :] - g_cum[i][:, block_end, :]  # O(1) subtraction
            gamma = torch.exp(g_diff)
            tgt = layer.recurrent_states[0]
            tgt.copy_((tgt.double() + d_rec.double() * gamma[:, :, None, None]).to(tgt.dtype))
        d_conv = delta["conv"][i] if i < len(delta["conv"]) else None
        if d_conv is not None and 0 <= shift < w_conv:
            tgt_c = layer.conv_states[0]
            overlap = min(tgt_c.shape[-1], d_conv.shape[-1]) - shift
            if overlap > 0:
                tgt_c[..., :overlap].copy_(
                    (tgt_c[..., :overlap].double() + d_conv.double()[..., shift : shift + overlap]).to(tgt_c.dtype)
                )


def diagnose_crossings(
    sweep: List[Dict[str, Any]],
    candidate_key: str,
    baseline_key: str = "suffix_recompute_ms",
) -> Dict[str, Any]:
    """Return sustained slow-to-fast monotone crossover and exact sign-flip diagnostics."""
    if not sweep:
        return {"sustained_monotone": None, "first_crossing": None, "last_crossing": None, "is_monotone_after_first": False}
    diffs = [float(r[baseline_key]) - float(r[candidate_key]) for r in sweep]
    lo_lbl = f"<= {sweep[0]['suffix_length']}"
    gt_lbl = f"> {sweep[-1]['suffix_length']}"

    if all(d > 0 for d in diffs):
        return {"sustained_monotone": lo_lbl, "first_crossing": lo_lbl, "last_crossing": lo_lbl, "is_monotone_after_first": True}
    if all(d <= 0 for d in diffs):
        return {"sustained_monotone": gt_lbl, "first_crossing": gt_lbl, "last_crossing": gt_lbl, "is_monotone_after_first": False}

    def _interp_up(idx: int) -> float:
        frac = (-diffs[idx]) / max(1e-9, diffs[idx + 1] - diffs[idx])
        x = sweep[idx]["suffix_length"] + frac * (sweep[idx + 1]["suffix_length"] - sweep[idx]["suffix_length"])
        return round(float(x), 1)

    def _interp_down(idx: int) -> float:
        frac = diffs[idx] / max(1e-9, diffs[idx] - diffs[idx + 1])
        x = sweep[idx]["suffix_length"] + frac * (sweep[idx + 1]["suffix_length"] - sweep[idx]["suffix_length"])
        return round(float(x), 1)

    first_c: Any = lo_lbl if diffs[0] > 0 else None
    last_c: Any = lo_lbl if diffs[0] > 0 else None
    first_up_idx: Optional[int] = None
    sustained_c: Any = None

    for i in range(len(sweep) - 1):
        if diffs[i] <= 0 < diffs[i + 1]:
            xc = _interp_up(i)
            if first_c is None:
                first_c = xc
                first_up_idx = i
            last_c = xc
            if sustained_c is None and all(d > 0 for d in diffs[i + 1 :]):
                sustained_c = xc
        elif diffs[i] > 0 >= diffs[i + 1]:
            xd = _interp_down(i)
            last_c = xd

    if sustained_c is None:
        sustained_c = gt_lbl
        mono_after_first = False
    elif diffs[0] > 0:
        mono_after_first = all(d > 0 for d in diffs)
    else:
        mono_after_first = bool(first_up_idx is not None and all(d > 0 for d in diffs[first_up_idx + 1 :]) and sustained_c == first_c)

    return {
        "sustained_monotone": sustained_c,
        "first_crossing": first_c if first_c is not None else gt_lbl,
        "last_crossing": last_c if last_c is not None else gt_lbl,
        "is_monotone_after_first": bool(mono_after_first),
    }


def interpolate_monotone_crossover(
    sweep: List[Dict[str, Any]],
    candidate_key: str,
    baseline_key: str = "suffix_recompute_ms",
) -> Any:
    """Locate sustained slow-to-fast crossover (`baseline - candidate` turns >= 0 and stays >= 0)."""
    return diagnose_crossings(sweep, candidate_key, baseline_key)["sustained_monotone"]


def run_part_a_o1_e8(
    model: Any,
    base_model: Any,
    n_linear: int,
    w_conv: int,
    device: torch.device,
    use_l2k: bool,
    suffix_lengths: List[int],
    n_repeats: int = 15,
    n_warmup: int = 3,
    p0: int = 2048,
) -> Dict[str, Any]:
    """Fixed-O(1) steady-state timing (delta_L=0 in-place vs delta_L=+4 Delta-RoPE cat) & monotone T*."""
    b_deep = p0 + 3 * max(1, n_linear - 1)
    c_ckpt_pos = max(16, p0 - min(128, p0 // 2))
    max_T = b_deep + max(suffix_lengths) + 32

    torch.manual_seed(42)
    ids_o = torch.randint(100, 2000, (1, max_T), device=device, dtype=torch.long)
    ids_e = ids_o.clone()
    ids_e[:, p0 : p0 + 4] = torch.tensor([[101, 102, 103, 104]], device=device, dtype=torch.long)
    # Varlen edited span (+4 tokens inside [p0:p1+4))
    ids_e_vl = torch.cat(
        [ids_o[:, :p0], torch.tensor([[101, 102, 103, 104, 105, 106, 107, 108]], device=device), ids_o[:, p0 + 4 :]],
        dim=1,
    )
    b_new_vl = b_deep + 4

    with torch.no_grad():
        _, c_1920 = run_chunked_prefill(model, base_model, ids_o[:, :c_ckpt_pos], 0, None, 2048, use_l2k)
        c_1920_cpu = move_cache_device(c_1920, "cpu")
        _, c_p0 = run_chunked_prefill(model, base_model, ids_o[:, c_ckpt_pos:p0], c_ckpt_pos, clone_cache(c_1920), 2048, use_l2k)
        del c_1920
        _, c_ob = run_chunked_prefill(model, base_model, ids_o[:, p0:b_deep], p0, clone_cache(c_p0), 2048, use_l2k)
        s_ob = snap_states(c_ob)
        c_prefix_base = c_ob  # [0, b_deep)

        _, g_cum = capture_prefix_cum_decay(model, base_model, ids_o[:, :max_T], n_linear, use_l2k)

        def _time_blk():
            t0 = _sync(device)
            c_w = clone_cache(c_p0)
            _, c_nb = run_chunked_prefill(model, base_model, ids_e[:, p0:b_deep], p0, c_w, 2048, use_l2k)
            delta = compute_state_delta_fp64(s_ob, snap_states(c_nb))
            kv_sl = extract_kv_slice(c_nb, p0, b_deep)
            dt_ms = (_sync(device) - t0) * 1000.0
            return dt_ms, delta, kv_sl

        # Varlen block forward (+4 tok)
        def _time_blk_vl():
            t0 = _sync(device)
            c_w = clone_cache(c_p0)
            _, c_nb = run_chunked_prefill(model, base_model, ids_e_vl[:, p0:b_new_vl], p0, c_w, 2048, use_l2k)
            delta = compute_state_delta_fp64(s_ob, snap_states(c_nb))
            kv_sl = extract_kv_slice(c_nb, p0, b_new_vl)
            dt_ms = (_sync(device) - t0) * 1000.0
            return dt_ms, delta, kv_sl

        def _time_r128():
            t0 = _sync(device)
            c_g = move_cache_device(c_1920_cpu, str(device))
            _, c_g = run_chunked_prefill(model, base_model, ids_o[:, c_ckpt_pos:p0], c_ckpt_pos, c_g, 2048, use_l2k)
            return (_sync(device) - t0) * 1000.0

        # Interleaved warmup and repeats across fixed-block, varlen-block, and r128 replay to prevent thermal drift
        bf_list: List[float] = []
        bf_vl_list: List[float] = []
        r128_list: List[float] = []
        delta_ref = kv_ref = delta_vl = kv_vl = None
        for rep in range(n_warmup + n_repeats):
            if rep % 2 == 0:
                t_b, delta_ref, kv_ref = _time_blk()
                t_v, delta_vl, kv_vl = _time_blk_vl()
            else:
                t_v, delta_vl, kv_vl = _time_blk_vl()
                t_b, delta_ref, kv_ref = _time_blk()
            t_r = _time_r128()
            if rep >= n_warmup:
                bf_list.append(t_b)
                bf_vl_list.append(t_v)
                r128_list.append(t_r)
        blk_stats = _med_iqr(bf_list)
        blk_vl_stats = _med_iqr(bf_vl_list)
        r128_stats = _med_iqr(r128_list)

        sweep = []
        for l_suf in suffix_lengths:
            t_end = b_deep + l_suf
            c_ctx_T = clone_cache(c_prefix_base)
            _, c_ctx_T = run_chunked_prefill(model, base_model, ids_o[:, b_deep:t_end], b_deep, c_ctx_T, 2048, use_l2k)

            cl_list: List[float] = []
            upd_list: List[float] = []
            kv_list: List[float] = []
            vl_kv_list: List[float] = []
            rec_list: List[float] = []

            # Interleave all per-suffix arms within each repeat to mitigate sequential thermal drift
            for rep in range(n_warmup + n_repeats):
                t0 = _sync(device)
                _ = clone_cache(c_ctx_T)
                c_ms = (_sync(device) - t0) * 1000.0

                c_tmp = clone_cache(c_ctx_T)
                t0 = _sync(device)
                apply_jlocal_update_o1(c_tmp, delta_ref, g_cum, b_deep, t_end, w_conv)
                u_ms = (_sync(device) - t0) * 1000.0
                t1 = _sync(device)
                patch_kv_slice(c_tmp, kv_ref, p0, b_deep)
                k_ms = (_sync(device) - t1) * 1000.0
                del c_tmp

                c_tmp2 = clone_cache(c_ctx_T)
                t0 = _sync(device)
                apply_jlocal_update_o1(c_tmp2, delta_vl, g_cum, b_deep, t_end, w_conv)
                apply_delta_rope_to_kv_cache(c_tmp2, kv_vl, p0, b_deep, b_new_vl, t_end, 4, base_model, apply_delta_rope=True)
                v_ms = (_sync(device) - t0) * 1000.0
                del c_tmp2

                t0 = _sync(device)
                c_s = clone_cache(c_prefix_base)
                _, c_s = run_chunked_prefill(model, base_model, ids_o[:, b_deep:t_end], b_deep, c_s, 2048, use_l2k)
                r_ms = (_sync(device) - t0) * 1000.0
                del c_s

                if rep >= n_warmup:
                    cl_list.append(c_ms)
                    upd_list.append(u_ms)
                    kv_list.append(k_ms)
                    vl_kv_list.append(v_ms)
                    rec_list.append(r_ms)

            cl_stats = _med_iqr(cl_list)
            upd_stats = _med_iqr(upd_list)
            kv_stats = _med_iqr(kv_list)
            vl_kv_stats = _med_iqr(vl_kv_list)
            rec_stats = _med_iqr(rec_list)

            inpl_ms = blk_stats["median"] + upd_stats["median"] + kv_stats["median"]
            clnd_ms = inpl_ms + cl_stats["median"]
            offb_ms = inpl_ms + r128_stats["median"]
            varlen_ms = blk_vl_stats["median"] + vl_kv_stats["median"] + cl_stats["median"]
            suf_ms = rec_stats["median"]

            row = {
                "suffix_length": int(l_suf),
                "total_tokens": int(t_end),
                "suffix_recompute_ms": round(suf_ms, 3),
                "suffix_recompute_iqr": rec_stats["iqr"],
                "jlocal_o1_inplace_ms": round(inpl_ms, 3),
                "jlocal_o1_cloned_ms": round(clnd_ms, 3),
                "jlocal_o1_offboundary_ms": round(offb_ms, 3),
                "jlocal_varlen_dl4_cloned_ms": round(varlen_ms, 3),
                "components_ms": {
                    "block_fwd_med": blk_stats["median"],
                    "block_fwd_iqr": blk_stats["iqr"],
                    "o1_state_inject_med": upd_stats["median"],
                    "inplace_kv_patch_med": kv_stats["median"],
                    "clone_ctx_med": cl_stats["median"],
                    "r128_replay_med": r128_stats["median"],
                    "varlen_inject_cat_med": vl_kv_stats["median"],
                },
                "raw_repeats_ms": {
                    "suffix_recompute": rec_stats["raw_ms"],
                    "clone_ctx": cl_stats["raw_ms"],
                    "o1_state_inject": upd_stats["raw_ms"],
                    "inplace_kv_patch": kv_stats["raw_ms"],
                    "varlen_inject_cat": vl_kv_stats["raw_ms"],
                },
                "speedup_inplace": round(suf_ms / max(1e-6, inpl_ms), 3),
                "speedup_cloned": round(suf_ms / max(1e-6, clnd_ms), 3),
                "speedup_offboundary": round(suf_ms / max(1e-6, offb_ms), 3),
                "speedup_varlen_dl4_cloned": round(suf_ms / max(1e-6, varlen_ms), 3),
            }
            sweep.append(row)
            del c_ctx_T

        del c_1920_cpu, c_p0, c_prefix_base, g_cum

    cross_diag = {
        "inplace_o1": diagnose_crossings(sweep, "jlocal_o1_inplace_ms"),
        "cloned_o1": diagnose_crossings(sweep, "jlocal_o1_cloned_ms"),
        "avg_offboundary_o1": diagnose_crossings(sweep, "jlocal_o1_offboundary_ms"),
        "varlen_dl4_cloned": diagnose_crossings(sweep, "jlocal_varlen_dl4_cloned_ms"),
    }
    cross = {k: v["sustained_monotone"] for k, v in cross_diag.items()}
    return {
        "p0_part_a": int(p0),
        "b_deep": int(b_deep),
        "n_warmup": int(n_warmup),
        "n_repeats": int(n_repeats),
        "fixed_components": {
            "block_forward": blk_stats,
            "block_forward_varlen_dl4": blk_vl_stats,
            "half_chunk_128_replay": r128_stats,
        },
        "monotone_crossover_T_star_suffix_tokens": cross,
        "crossing_diagnostics": cross_diag,
        "sweep": sweep,
    }


def _build_detector_feats(
    s_ob: Dict[str, List[Optional[torch.Tensor]]],
    delta: Dict[str, List[Optional[torch.Tensor]]],
    g_cum: Dict[int, torch.Tensor],
    l_cum: Dict[int, torch.Tensor],
    lin_indices: List[int],
    b_deep: int,
    t_end: int,
    d_k: int,
    lg_jl: torch.Tensor,
    lg_old: torch.Tensor,
) -> Dict[str, float]:
    """Compute all 11 runtime detector features using exact O(1) prefix lookups for c_hat & gamma."""
    p_jl = F.softmax(lg_jl, dim=-1)
    p_old = F.softmax(lg_old, dim=-1)
    top2 = torch.topk(lg_jl[0], 2).values
    old_top1 = int(lg_old.argmax(dim=-1).item())
    rel_ds = [
        float(dr.float().norm().item() / (ro.float().norm().item() + 1e-6))
        for ro, dr in zip(s_ob["rec"], delta["rec"])
        if ro is not None and dr is not None
    ]
    gam_vals = torch.stack([torch.exp(g_cum[li][:, t_end, :] - g_cum[li][:, b_deep, :]).mean() for li in lin_indices])
    chat_vals = torch.stack([((l_cum[li][:, t_end, :] - l_cum[li][:, b_deep, :]) / float(d_k)).mean() for li in lin_indices])
    return {
        "rel_delta_s_mean": sum(rel_ds) / max(1, len(rel_ds)),
        "rel_delta_s_max": max(rel_ds, default=0.0),
        "c_hat_mean": float(chat_vals.mean().item()),
        "gamma_mean": float(gam_vals.mean().item()),
        "abs_delta_L": 0.0,
        "force_no_rope": 0.0,
        "margin_jl": float((top2[0] - top2[1]).item()),
        "entropy_jl": float(-(p_jl * (p_jl + 1e-12).log()).sum().item()),
        "shift_kl_old_vs_jl": float(F.kl_div((p_old + 1e-12).log(), p_jl, reduction="batchmean").item()),
        "top1_flipped_from_old": float(old_top1 != int(lg_jl.argmax(dim=-1).item())),
        "stale_top1_prob_under_jl": float(p_jl[0, old_top1].item()),
    }


def _build_flippable_sniah_part_b(
    tok: Any,
    p0_target: int,
    max_suffix_len: int,
    n_linear: int,
    device: torch.device,
    old_val: str = "3841",
    new_val: str = "7294",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, str, str]:
    """Construct an equal-length flippable benign S-NIAH prompt so unforced OSC has verified ground truth."""
    from w5_9b_e5_runner import FILLERS, SYSTEM_PROMPT, apply_chat_no_think

    fact_o = f"Audit log: Node-Alpha holds vault key {old_val}."
    fact_e = f"Audit log: Node-Alpha holds vault key {new_val}."
    q_str = "Question: What is the assigned vault key for Node-Alpha? Reply with 4 digits."

    # Build prefix filler hitting ~p0_target tokens and suffix filler exceeding max_suffix_len + b_deep
    pre_parts: List[str] = []
    step = 0
    while len(tok.encode(" ".join(pre_parts), add_special_tokens=False)) < max(32, p0_target - 32):
        pre_parts.append(FILLERS[step % len(FILLERS)])
        step += 1
    pre_txt = " ".join(pre_parts)

    suf_need = max_suffix_len + 3 * max(1, n_linear - 1) + 96
    base_suf = " ".join(FILLERS[(step + j) % len(FILLERS)] for j in range(24))
    base_suf_len = max(1, len(tok.encode(base_suf, add_special_tokens=False)))
    reps = (suf_need // base_suf_len) + 2
    suf_txt = " ".join([base_suf] * reps)

    u_o = f"{pre_txt} {fact_o} {suf_txt}\n\n{q_str}"
    u_e = f"{pre_txt} {fact_e} {suf_txt}\n\n{q_str}"
    full_o = apply_chat_no_think(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": u_o}])
    full_e = apply_chat_no_think(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": u_e}])
    qo = full_o.rfind(q_str)
    qe = full_e.rfind(q_str)
    assert qo > 0 and qe > 0, "Failed to locate query string in chat prompt"
    ctx_o, ctx_e, q_tail = full_o[:qo], full_e[:qe], full_o[qo:]
    ids_o_l = tok.encode(ctx_o, add_special_tokens=False)
    ids_e_l = tok.encode(ctx_e, add_special_tokens=False)
    ids_q_l = tok.encode(q_tail, add_special_tokens=False)
    assert len(ids_o_l) == len(ids_e_l), f"Expected equal length edit, got {len(ids_o_l)} vs {len(ids_e_l)}"
    diff_idx = [i for i, (a, b) in enumerate(zip(ids_o_l, ids_e_l)) if a != b]
    assert diff_idx, "No differing token positions between old and edited prompts"
    p0 = diff_idx[0]
    p1 = diff_idx[-1] + 1
    b_deep = p1 + 3 * max(1, n_linear - 1)
    max_T = b_deep + max_suffix_len + 16
    assert len(ids_o_l) >= max_T, f"Constructed context {len(ids_o_l)} < needed {max_T}"
    return (
        torch.tensor([ids_o_l[:max_T]], device=device, dtype=torch.long),
        torch.tensor([ids_e_l[:max_T]], device=device, dtype=torch.long),
        torch.tensor([ids_q_l], device=device, dtype=torch.long),
        int(p0),
        int(b_deep),
        old_val,
        new_val,
    )


def run_part_b_symmetric_e2e_routing(
    model: Any,
    base_model: Any,
    tok: Any,
    eos_ids: List[int],
    n_linear: int,
    w_conv: int,
    device: torch.device,
    use_l2k: bool,
    detector_mode: str,
    lr_blob: Optional[Dict[str, Any]],
    suffix_lengths: List[int],
    fail_rates: List[float],
    n_repeats: int = 15,
    p0: int = 1280,
    d_k: int = 128,
    decode_steps: int = 8,
    n_warmup: int = 5,
    n_live_trials: int = 40,
) -> Dict[str, Any]:
    """Matched symmetric E2E routing sweep with fixed decode steps, ground-truth S-NIAH prompt, & live conditional arm."""
    ids_o, ids_e, q_ids, p0_real, b_deep, old_val, new_val = _build_flippable_sniah_part_b(
        tok, p0_target=p0, max_suffix_len=max(suffix_lengths), n_linear=n_linear, device=device
    )
    p0 = p0_real
    max_T = int(ids_o.shape[1])

    def _score_det(feats: Dict[str, float]) -> Tuple[float, int]:
        if detector_mode == "lr" and lr_blob is not None:
            sc, fl = score_frozen_lr(lr_blob, feats)
            return float(sc), int(fl)
        sc, fl, _ = score_frozen_osc(feats, FROZEN_OSC_TAU_DEV_2B)
        return float(sc), int(fl)

    with torch.no_grad():
        _, c_pre = run_chunked_prefill(model, base_model, ids_o[:, :p0], 0, None, 2048, use_l2k)
        c_pre_cpu = move_cache_device(c_pre, "cpu")
        _, c_old_b = run_chunked_prefill(model, base_model, ids_o[:, p0:b_deep], p0, clone_cache(c_pre), 2048, use_l2k)
        s_ob = snap_states(c_old_b)
        _, g_cum, l_cum = capture_prefix_cum_decay(model, base_model, ids_o[:, :max_T], n_linear, use_l2k, return_l_cum=True)
        lin_indices = sorted(g_cum.keys())

        rows: List[Dict[str, Any]] = []
        for l_suf in suffix_lengths:
            t_end = b_deep + l_suf
            c_ctx_old = clone_cache(c_old_b)
            _, c_ctx_old = run_chunked_prefill(model, base_model, ids_o[:, b_deep:t_end], b_deep, c_ctx_old, 2048, use_l2k)
            c_ctx_old_cpu = move_cache_device(c_ctx_old, "cpu")
            ev_old = eval_fixed_steps_in_place(model, base_model, clone_cache(c_ctx_old), t_end, q_ids, use_l2k, decode_steps=1)
            lg_old = ev_old["first_logits"].float()

            # Compute exact ground truth (oracle vs J-local) on this benign S-NIAH prompt at l_suf
            c_ora_gt = clone_cache(c_pre)
            _, c_ora_gt = run_chunked_prefill(model, base_model, ids_e[:, p0:t_end], p0, c_ora_gt, 2048, use_l2k)
            ev_ora_gt = eval_fixed_steps_in_place(model, base_model, c_ora_gt, t_end, q_ids, use_l2k, decode_steps=decode_steps)
            ora_txt_gt = tok.decode(ev_ora_gt["gen_ids"], skip_special_tokens=True)
            ora_top1_gt = int(ev_ora_gt["first_logits"].argmax(dim=-1).item())
            del c_ora_gt

            c_jl_gt = clone_cache(c_ctx_old)
            _, c_nb_gt = run_chunked_prefill(model, base_model, ids_e[:, p0:b_deep], p0, clone_cache(c_pre), 2048, use_l2k)
            d_gt = compute_state_delta_fp64(s_ob, snap_states(c_nb_gt))
            apply_jlocal_update_o1(c_jl_gt, d_gt, g_cum, b_deep, t_end, w_conv)
            patch_kv_slice(c_jl_gt, extract_kv_slice(c_nb_gt, p0, b_deep), p0, b_deep)
            ev_jl_gt = eval_fixed_steps_in_place(model, base_model, c_jl_gt, t_end, q_ids, use_l2k, decode_steps=decode_steps)
            jl_txt_gt = tok.decode(ev_jl_gt["gen_ids"], skip_special_tokens=True)
            jl_top1_gt = int(ev_jl_gt["first_logits"].argmax(dim=-1).item())
            del c_jl_gt, c_nb_gt

            gt_ora_cn = bool(new_val in ora_txt_gt)
            gt_jl_cn = bool(new_val in jl_txt_gt)
            gt_jl_leak = bool((old_val in jl_txt_gt) and not gt_jl_cn)
            gt_is_failure = bool((not gt_jl_cn) or gt_jl_leak)
            prompt_gt = {
                "prompt_type": "flippable_benign_sniah",
                "old_val": old_val,
                "new_val": new_val,
                "oracle_text": ora_txt_gt,
                "jlocal_text": jl_txt_gt,
                "oracle_contains_new": gt_ora_cn,
                "jlocal_contains_new": gt_jl_cn,
                "jlocal_old_leak": gt_jl_leak,
                "jlocal_top1_eq_oracle": bool(jl_top1_gt == ora_top1_gt),
                "ground_truth_jlocal_failure": gt_is_failure,
            }

            jl_gpu_inpl, jl_gpu_clnd, det_ms_l, fb_ms_l, suf_gpu_l = [], [], [], [], []
            jl_h2d_pre_l, jl_h2d_all_l, suf_h2d_l = [], [], []
            observed_det_flags: List[int] = []
            last_feats: Dict[str, float] = {}

            for rep in range(n_warmup + n_repeats):
                # 1. GPU-resident J-local (pre-cloned working copy)
                c_work = clone_cache(c_ctx_old)
                t0 = _sync(device)
                c_pre_w = clone_cache(c_pre)
                _, c_nb = run_chunked_prefill(model, base_model, ids_e[:, p0:b_deep], p0, c_pre_w, 2048, use_l2k)
                delta = compute_state_delta_fp64(s_ob, snap_states(c_nb))
                kv_sl = extract_kv_slice(c_nb, p0, b_deep)
                apply_jlocal_update_o1(c_work, delta, g_cum, b_deep, t_end, w_conv)
                patch_kv_slice(c_work, kv_sl, p0, b_deep)
                ev_jl = eval_fixed_steps_in_place(model, base_model, c_work, t_end, q_ids, use_l2k, decode_steps=decode_steps)
                t_jl_inpl = (_sync(device) - t0) * 1000.0
                assert ev_jl["decode_steps_executed"] == decode_steps
                del c_work, c_pre_w

                # 2. GPU-resident J-local + clone(c_ctx_old) inside timer
                t0 = _sync(device)
                c_w2 = clone_cache(c_ctx_old)
                c_p2 = clone_cache(c_pre)
                _, c_nb2 = run_chunked_prefill(model, base_model, ids_e[:, p0:b_deep], p0, c_p2, 2048, use_l2k)
                delta2 = compute_state_delta_fp64(s_ob, snap_states(c_nb2))
                kv_sl2 = extract_kv_slice(c_nb2, p0, b_deep)
                apply_jlocal_update_o1(c_w2, delta2, g_cum, b_deep, t_end, w_conv)
                patch_kv_slice(c_w2, kv_sl2, p0, b_deep)
                ev_cl = eval_fixed_steps_in_place(model, base_model, c_w2, t_end, q_ids, use_l2k, decode_steps=decode_steps)
                t_jl_clnd = (_sync(device) - t0) * 1000.0
                assert ev_cl["decode_steps_executed"] == decode_steps
                del c_w2, c_p2, c_nb2

                # 3. Detector feature extraction & scoring on GPU using genuine O(1) c_hat & gamma
                t0 = _sync(device)
                feats = _build_detector_feats(
                    s_ob, delta, g_cum, l_cum, lin_indices, b_deep, t_end, d_k, ev_jl["first_logits"].float(), lg_old
                )
                _, det_flag = _score_det(feats)
                t_det = (_sync(device) - t0) * 1000.0

                # 4. Fallback tail [b_deep:t_end) + fixed decode from c_nb
                t0 = _sync(device)
                c_fb = clone_cache(c_nb)
                _, c_fb = run_chunked_prefill(model, base_model, ids_e[:, b_deep:t_end], b_deep, c_fb, 2048, use_l2k)
                ev_fb = eval_fixed_steps_in_place(model, base_model, c_fb, t_end, q_ids, use_l2k, decode_steps=decode_steps)
                t_fb = (_sync(device) - t0) * 1000.0
                assert ev_fb["decode_steps_executed"] == decode_steps
                del c_fb, c_nb

                # 5. GPU-resident always_suffix from GPU c_pre
                t0 = _sync(device)
                c_sf = clone_cache(c_pre)
                _, c_sf = run_chunked_prefill(model, base_model, ids_e[:, p0:t_end], p0, c_sf, 2048, use_l2k)
                ev_sfg = eval_fixed_steps_in_place(model, base_model, c_sf, t_end, q_ids, use_l2k, decode_steps=decode_steps)
                t_suf_gpu = (_sync(device) - t0) * 1000.0
                assert ev_sfg["decode_steps_executed"] == decode_steps
                del c_sf

                # 6. Symmetric H2D: always_suffix paying move_cache_device(c_pre_cpu) INSIDE timer
                t0 = _sync(device)
                c_sf_h = move_cache_device(c_pre_cpu, str(device))
                _, c_sf_h = run_chunked_prefill(model, base_model, ids_e[:, p0:t_end], p0, c_sf_h, 2048, use_l2k)
                ev_sfh = eval_fixed_steps_in_place(model, base_model, c_sf_h, t_end, q_ids, use_l2k, decode_steps=decode_steps)
                t_suf_h2d = (_sync(device) - t0) * 1000.0
                assert ev_sfh["decode_steps_executed"] == decode_steps
                del c_sf_h

                # 7. Symmetric H2D: O_local paying H2D(c_pre_cpu) + clone(GPU c_ctx_old) INSIDE timer
                t0 = _sync(device)
                c_p_h = move_cache_device(c_pre_cpu, str(device))
                c_ctx_w = clone_cache(c_ctx_old)
                _, c_nb_h = run_chunked_prefill(model, base_model, ids_e[:, p0:b_deep], p0, c_p_h, 2048, use_l2k)
                delta_h = compute_state_delta_fp64(s_ob, snap_states(c_nb_h))
                kv_sl_h = extract_kv_slice(c_nb_h, p0, b_deep)
                apply_jlocal_update_o1(c_ctx_w, delta_h, g_cum, b_deep, t_end, w_conv)
                patch_kv_slice(c_ctx_w, kv_sl_h, p0, b_deep)
                ev_pre = eval_fixed_steps_in_place(model, base_model, c_ctx_w, t_end, q_ids, use_l2k, decode_steps=decode_steps)
                t_jl_h2d_pre = (_sync(device) - t0) * 1000.0
                assert ev_pre["decode_steps_executed"] == decode_steps
                del c_ctx_w, c_nb_h

                # 8. Symmetric H2D Full Swap: O_local paying H2D(c_pre_cpu) + H2D(c_ctx_old_cpu) INSIDE timer
                t0 = _sync(device)
                c_p_f = move_cache_device(c_pre_cpu, str(device))
                c_ctx_f = move_cache_device(c_ctx_old_cpu, str(device))
                _, c_nb_f = run_chunked_prefill(model, base_model, ids_e[:, p0:b_deep], p0, c_p_f, 2048, use_l2k)
                delta_f = compute_state_delta_fp64(s_ob, snap_states(c_nb_f))
                kv_sl_f = extract_kv_slice(c_nb_f, p0, b_deep)
                apply_jlocal_update_o1(c_ctx_f, delta_f, g_cum, b_deep, t_end, w_conv)
                patch_kv_slice(c_ctx_f, kv_sl_f, p0, b_deep)
                ev_all = eval_fixed_steps_in_place(model, base_model, c_ctx_f, t_end, q_ids, use_l2k, decode_steps=decode_steps)
                t_jl_h2d_all = (_sync(device) - t0) * 1000.0
                assert ev_all["decode_steps_executed"] == decode_steps
                del c_ctx_f, c_nb_f

                if rep >= n_warmup:
                    jl_gpu_inpl.append(t_jl_inpl)
                    jl_gpu_clnd.append(t_jl_clnd)
                    det_ms_l.append(t_det)
                    fb_ms_l.append(t_fb)
                    suf_gpu_l.append(t_suf_gpu)
                    suf_h2d_l.append(t_suf_h2d)
                    jl_h2d_pre_l.append(t_jl_h2d_pre)
                    jl_h2d_all_l.append(t_jl_h2d_all)
                    observed_det_flags.append(det_flag)
                    last_feats = feats

            # 9. Genuine single-timer conditional E2E routed execution arm (sym_h2d_full_swap)
            def _run_one_conditional_full_swap(force_fallback: Optional[bool] = None) -> Tuple[float, int]:
                t0 = _sync(device)
                c_p_c = move_cache_device(c_pre_cpu, str(device))
                c_ctx_c = move_cache_device(c_ctx_old_cpu, str(device))
                _, c_nb_c = run_chunked_prefill(model, base_model, ids_e[:, p0:b_deep], p0, c_p_c, 2048, use_l2k)
                delta_c = compute_state_delta_fp64(s_ob, snap_states(c_nb_c))
                kv_sl_c = extract_kv_slice(c_nb_c, p0, b_deep)
                apply_jlocal_update_o1(c_ctx_c, delta_c, g_cum, b_deep, t_end, w_conv)
                patch_kv_slice(c_ctx_c, kv_sl_c, p0, b_deep)
                ev_c = eval_fixed_steps_in_place(model, base_model, c_ctx_c, t_end, q_ids, use_l2k, decode_steps=decode_steps)
                feats_c = _build_detector_feats(
                    s_ob, delta_c, g_cum, l_cum, lin_indices, b_deep, t_end, d_k, ev_c["first_logits"].float(), lg_old
                )
                _, det_trig = _score_det(feats_c)
                del c_ctx_c
                take_fb = bool(force_fallback) if force_fallback is not None else bool(det_trig)
                if take_fb:
                    c_fbc = c_nb_c
                    _, c_fbc = run_chunked_prefill(model, base_model, ids_e[:, b_deep:t_end], b_deep, c_fbc, 2048, use_l2k)
                    _ = eval_fixed_steps_in_place(model, base_model, c_fbc, t_end, q_ids, use_l2k, decode_steps=decode_steps)
                    del c_fbc
                else:
                    del c_nb_c
                return (_sync(device) - t0) * 1000.0, int(det_trig)

            # Warmup conditional arm twice per suffix
            _, _ = _run_one_conditional_full_swap(force_fallback=False)
            _, _ = _run_one_conditional_full_swap(force_fallback=True)
            m_suf_h2d_val = statistics.median(suf_h2d_l)
            suf_h2d_mean_val = sum(suf_h2d_l) / max(1, len(suf_h2d_l))

            # 9a. Genuine UNFORCED detector-driven live branching arm (force_fallback=None)
            n_tr = max(4, n_live_trials)
            nat_trials: List[float] = []
            nat_triggers = 0
            for _ in range(n_tr):
                dt_ms, nat_f = _run_one_conditional_full_swap(force_fallback=None)
                nat_trials.append(dt_ms)
                nat_triggers += int(nat_f)
            st_nat = _med_iqr(nat_trials)
            live_detector_driven = {
                "arm_role": "unforced_natural_detector_live_branching",
                "detector_mode": detector_mode,
                "ground_truth_jlocal_failure": gt_is_failure,
                "natural_trigger_count": int(nat_triggers),
                "natural_trigger_rate": round(nat_triggers / max(1, n_tr), 4),
                "unforced_false_trigger_count": int(nat_triggers if not gt_is_failure else 0),
                "unforced_true_trigger_count": int(nat_triggers if gt_is_failure else 0),
                "n_trials": int(n_tr),
                "mean_ms": round(st_nat["mean"], 3),
                "median_ms": round(st_nat["median"], 3),
                "p95_ms": round(st_nat["p95"], 3),
                "iqr_ms": round(st_nat["iqr"], 3),
                "speedup_vs_suf_h2d_mean": round(suf_h2d_mean_val / max(1e-6, st_nat["mean"]), 3),
                "speedup_vs_suf_h2d_median": round(m_suf_h2d_val / max(1e-6, st_nat["median"]), 3),
                "raw_trials_ms": st_nat["raw_ms"],
            }

            # 9b. Exogenous controlled fail-rate stress sweep (shuffled force_fallback schedule)
            live_conditional_by_r: Dict[str, Any] = {}
            for r_fail in fail_rates:
                k_fail = min(n_tr, int(round(r_fail * n_tr)))
                trig_sched = [True] * k_fail + [False] * (n_tr - k_fail)
                random.Random(202614 + l_suf * 100 + k_fail).shuffle(trig_sched)
                lat_trials: List[float] = []
                natural_hits = 0
                for is_fb in trig_sched:
                    dt_ms, nat_f = _run_one_conditional_full_swap(force_fallback=is_fb)
                    lat_trials.append(dt_ms)
                    natural_hits += nat_f
                st_live = _med_iqr(lat_trials)
                live_conditional_by_r[f"r_{r_fail:.2f}"] = {
                    "arm_role": "exogenous_forced_fallback_stress_sweep",
                    "effective_fail_rate": round(k_fail / max(1, n_tr), 4),
                    "n_trials": int(n_tr),
                    "mean_ms": round(st_live["mean"], 3),
                    "median_ms": round(st_live["median"], 3),
                    "p95_ms": round(st_live["p95"], 3),
                    "iqr_ms": round(st_live["iqr"], 3),
                    "speedup_vs_suf_h2d_mean": round(suf_h2d_mean_val / max(1e-6, st_live["mean"]), 3),
                    "speedup_vs_suf_h2d_median": round(m_suf_h2d_val / max(1e-6, st_live["median"]), 3),
                    "natural_detector_triggers": int(natural_hits),
                    "raw_trials_ms": st_live["raw_ms"],
                }

            st_gpu_inpl = _med_iqr(jl_gpu_inpl)
            st_gpu_clnd = _med_iqr(jl_gpu_clnd)
            st_det = _med_iqr(det_ms_l)
            st_fb = _med_iqr(fb_ms_l)
            st_suf_gpu = _med_iqr(suf_gpu_l)
            st_suf_h2d = _med_iqr(suf_h2d_l)
            st_jl_h2d_pre = _med_iqr(jl_h2d_pre_l)
            st_jl_h2d_all = _med_iqr(jl_h2d_all_l)

            m_gpu_clnd = st_gpu_clnd["median"]
            m_det = st_det["median"]
            m_fb = st_fb["median"]
            m_suf_gpu = st_suf_gpu["median"]
            m_suf_h2d = st_suf_h2d["median"]
            m_jl_h2d_pre = st_jl_h2d_pre["median"]
            m_jl_h2d_all = st_jl_h2d_all["median"]

            routed_by_r: Dict[str, Any] = {}
            for r_fail in fail_rates:
                rt_gpu_clnd = m_gpu_clnd + m_det + r_fail * m_fb
                rt_h2d_pre = m_jl_h2d_pre + m_det + r_fail * m_fb
                rt_h2d_all = m_jl_h2d_all + m_det + r_fail * m_fb
                routed_by_r[f"r_{r_fail:.2f}"] = {
                    "gpu_resident_cloned_ms": round(rt_gpu_clnd, 3),
                    "gpu_resident_speedup": round(m_suf_gpu / max(1e-6, rt_gpu_clnd), 3),
                    "sym_h2d_pre_ms": round(rt_h2d_pre, 3),
                    "sym_h2d_pre_speedup": round(m_suf_h2d / max(1e-6, rt_h2d_pre), 3),
                    "sym_h2d_full_swap_ms": round(rt_h2d_all, 3),
                    "sym_h2d_full_swap_speedup": round(m_suf_h2d / max(1e-6, rt_h2d_all), 3),
                }

            rows.append({
                "suffix_length": int(l_suf),
                "total_tokens": int(t_end),
                "decode_steps_executed": int(decode_steps),
                "prompt_ground_truth": prompt_gt,
                "detector_sample_feats": {k: round(float(v), 6) for k, v in last_feats.items()},
                "detector_observed_flag_rate": round(sum(observed_det_flags) / max(1, len(observed_det_flags)), 4),
                "gpu_resident_ms": {
                    "always_jlocal_inplace": round(st_gpu_inpl["median"], 3),
                    "always_jlocal_cloned": round(m_gpu_clnd, 3),
                    "detector": round(m_det, 3),
                    "fallback_tail": round(m_fb, 3),
                    "always_suffix": round(m_suf_gpu, 3),
                    "mean_ms": {
                        "always_jlocal_inplace": round(st_gpu_inpl["mean"], 3),
                        "always_jlocal_cloned": round(st_gpu_clnd["mean"], 3),
                        "detector": round(st_det["mean"], 3),
                        "fallback_tail": round(st_fb["mean"], 3),
                        "always_suffix": round(st_suf_gpu["mean"], 3),
                    },
                    "p95_ms": {
                        "always_jlocal_inplace": round(st_gpu_inpl["p95"], 3),
                        "always_jlocal_cloned": round(st_gpu_clnd["p95"], 3),
                        "detector": round(st_det["p95"], 3),
                        "fallback_tail": round(st_fb["p95"], 3),
                        "always_suffix": round(st_suf_gpu["p95"], 3),
                    },
                    "iqr": {
                        "always_jlocal_inplace": st_gpu_inpl["iqr"],
                        "always_jlocal_cloned": st_gpu_clnd["iqr"],
                        "fallback_tail": st_fb["iqr"],
                        "always_suffix": st_suf_gpu["iqr"],
                    },
                    "raw_repeats_ms": {
                        "always_jlocal_inplace": st_gpu_inpl["raw_ms"],
                        "always_jlocal_cloned": st_gpu_clnd["raw_ms"],
                        "detector": st_det["raw_ms"],
                        "fallback_tail": st_fb["raw_ms"],
                        "always_suffix": st_suf_gpu["raw_ms"],
                    },
                },
                "symmetric_h2d_ms": {
                    "always_jlocal_h2d_pre_only": round(m_jl_h2d_pre, 3),
                    "always_jlocal_h2d_full_swap": round(m_jl_h2d_all, 3),
                    "always_suffix_h2d": round(m_suf_h2d, 3),
                    "mean_ms": {
                        "always_jlocal_h2d_pre_only": round(st_jl_h2d_pre["mean"], 3),
                        "always_jlocal_h2d_full_swap": round(st_jl_h2d_all["mean"], 3),
                        "always_suffix_h2d": round(st_suf_h2d["mean"], 3),
                    },
                    "p95_ms": {
                        "always_jlocal_h2d_pre_only": round(st_jl_h2d_pre["p95"], 3),
                        "always_jlocal_h2d_full_swap": round(st_jl_h2d_all["p95"], 3),
                        "always_suffix_h2d": round(st_suf_h2d["p95"], 3),
                    },
                    "iqr": {
                        "always_jlocal_h2d_pre_only": st_jl_h2d_pre["iqr"],
                        "always_jlocal_h2d_full_swap": st_jl_h2d_all["iqr"],
                        "always_suffix_h2d": st_suf_h2d["iqr"],
                    },
                    "raw_repeats_ms": {
                        "always_jlocal_h2d_pre_only": st_jl_h2d_pre["raw_ms"],
                        "always_jlocal_h2d_full_swap": st_jl_h2d_all["raw_ms"],
                        "always_suffix_h2d": st_suf_h2d["raw_ms"],
                    },
                },
                "routed_e2e_by_fail_rate": routed_by_r,
                "live_detector_driven_full_swap": live_detector_driven,
                "live_conditional_full_swap_by_fail_rate": live_conditional_by_r,
            })
            print(
                f"  [Part B] l_suf={l_suf} done: gt_fail={gt_is_failure}, nat_trig={nat_triggers}/{n_tr}, "
                f"nat_spd={live_detector_driven['speedup_vs_suf_h2d_mean']}x, "
                f"r05_p95={live_conditional_by_r.get('r_0.05', {}).get('p95_ms')} vs suf_p95={round(st_suf_h2d['p95'], 3)}",
                flush=True,
            )
            del c_ctx_old, c_ctx_old_cpu

        del c_pre, c_pre_cpu, c_old_b, g_cum, l_cum

    crossovers: Dict[str, Any] = {}
    cross_diag_b: Dict[str, Any] = {}
    for r_fail in fail_rates:
        rk = f"r_{r_fail:.2f}"
        s_gpu = [{"suffix_length": r["suffix_length"], "cand": r["routed_e2e_by_fail_rate"][rk]["gpu_resident_cloned_ms"], "base": r["gpu_resident_ms"]["always_suffix"]} for r in rows]
        s_h2d_p = [{"suffix_length": r["suffix_length"], "cand": r["routed_e2e_by_fail_rate"][rk]["sym_h2d_pre_ms"], "base": r["symmetric_h2d_ms"]["always_suffix_h2d"]} for r in rows]
        s_h2d_f = [{"suffix_length": r["suffix_length"], "cand": r["routed_e2e_by_fail_rate"][rk]["sym_h2d_full_swap_ms"], "base": r["symmetric_h2d_ms"]["always_suffix_h2d"]} for r in rows]
        s_live_mean = [{"suffix_length": r["suffix_length"], "cand": r["live_conditional_full_swap_by_fail_rate"][rk]["mean_ms"], "base": r["symmetric_h2d_ms"]["mean_ms"]["always_suffix_h2d"]} for r in rows]
        s_live_med = [{"suffix_length": r["suffix_length"], "cand": r["live_conditional_full_swap_by_fail_rate"][rk]["median_ms"], "base": r["symmetric_h2d_ms"]["always_suffix_h2d"]} for r in rows]
        d_gpu = diagnose_crossings(s_gpu, "cand", "base")
        d_hp = diagnose_crossings(s_h2d_p, "cand", "base")
        d_hf = diagnose_crossings(s_h2d_f, "cand", "base")
        d_lv_mean = diagnose_crossings(s_live_mean, "cand", "base")
        d_lv_med = diagnose_crossings(s_live_med, "cand", "base")
        crossovers[rk] = {
            "T_star_gpu_resident_cloned": d_gpu["sustained_monotone"],
            "T_star_sym_h2d_pre_only": d_hp["sustained_monotone"],
            "T_star_sym_h2d_full_swap": d_hf["sustained_monotone"],
            "T_star_live_conditional_full_swap_mean": d_lv_mean["sustained_monotone"],
            "T_star_live_conditional_full_swap_median": d_lv_med["sustained_monotone"],
        }
        cross_diag_b[rk] = {
            "gpu_resident_cloned": d_gpu,
            "sym_h2d_pre_only": d_hp,
            "sym_h2d_full_swap": d_hf,
            "live_conditional_full_swap_mean": d_lv_mean,
            "live_conditional_full_swap_median": d_lv_med,
        }
    s_nat_mean = [{"suffix_length": r["suffix_length"], "cand": r["live_detector_driven_full_swap"]["mean_ms"], "base": r["symmetric_h2d_ms"]["mean_ms"]["always_suffix_h2d"]} for r in rows]
    s_nat_med = [{"suffix_length": r["suffix_length"], "cand": r["live_detector_driven_full_swap"]["median_ms"], "base": r["symmetric_h2d_ms"]["always_suffix_h2d"]} for r in rows]
    d_nat_mean = diagnose_crossings(s_nat_mean, "cand", "base")
    d_nat_med = diagnose_crossings(s_nat_med, "cand", "base")
    crossovers["unforced_detector_driven"] = {
        "T_star_live_detector_driven_full_swap_mean": d_nat_mean["sustained_monotone"],
        "T_star_live_detector_driven_full_swap_median": d_nat_med["sustained_monotone"],
    }
    cross_diag_b["unforced_detector_driven"] = {
        "live_detector_driven_full_swap_mean": d_nat_mean,
        "live_detector_driven_full_swap_median": d_nat_med,
    }
    return {
        "p0_part_b": int(p0),
        "b_deep": int(b_deep),
        "decode_steps": int(decode_steps),
        "eos_early_stop_disabled": True,
        "n_warmup": int(n_warmup),
        "n_repeats": int(n_repeats),
        "n_live_trials": int(n_live_trials),
        "fail_rates": fail_rates,
        "monotone_routed_crossovers": crossovers,
        "crossing_diagnostics": cross_diag_b,
        "rows": rows,
    }


def run_synthetic_dry_run(out_dir: Path) -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
    if hasattr(qm.torch_chunk_gated_delta_rule, "__wrapped__"):
        qm.torch_chunk_gated_delta_rule = qm.torch_chunk_gated_delta_rule.__wrapped__
    if hasattr(qm.torch_recurrent_gated_delta_rule, "__wrapped__"):
        qm.torch_recurrent_gated_delta_rule = qm.torch_recurrent_gated_delta_rule.__wrapped__

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True, local_files_only=True)
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B", local_files_only=True)
    tc = getattr(cfg, "text_config", cfg)
    tc.num_hidden_layers = 4
    tc.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    tc.vocab_size = len(tok)
    tc.hidden_size = 64
    tc.intermediate_size = 128
    tc.num_attention_heads = 2
    tc.num_key_value_heads = 1
    tc.head_dim = 256
    tc.linear_num_key_heads = 2
    tc.linear_num_value_heads = 2
    tc.linear_key_head_dim = 32
    tc.linear_value_head_dim = 32
    tc.mrope_section = [11, 11, 10]
    if hasattr(tc, "rope_parameters") and isinstance(tc.rope_parameters, dict):
        tc.rope_parameters["mrope_section"] = [11, 11, 10]
        tc.rope_parameters["partial_rotary_factor"] = 0.25

    model = Qwen3_5ForCausalLM(tc).eval()
    base_model = getattr(model, "model", model)
    eos_ids = get_eos_ids(model, tok)
    n_linear = count_linear_layers(tc)
    lr_blob = freeze_lr_from_dev_split(out_dir / "w11_detector_v2_clean_9b_records.jsonl")

    # Unit checks for diagnose_crossings boundary patterns ([-,+,-] & [+,-,-])
    dc_mpm = diagnose_crossings(
        [{"suffix_length": 128, "cand": 110.0, "base": 100.0}, {"suffix_length": 256, "cand": 90.0, "base": 100.0}, {"suffix_length": 512, "cand": 105.0, "base": 100.0}],
        "cand", "base",
    )
    assert dc_mpm["sustained_monotone"] == "> 512" and dc_mpm["is_monotone_after_first"] is False, f"[-,+,-] failed: {dc_mpm}"
    dc_pmm = diagnose_crossings(
        [{"suffix_length": 128, "cand": 90.0, "base": 100.0}, {"suffix_length": 256, "cand": 110.0, "base": 100.0}, {"suffix_length": 512, "cand": 120.0, "base": 100.0}],
        "cand", "base",
    )
    assert dc_pmm["sustained_monotone"] == "> 512" and dc_pmm["is_monotone_after_first"] is False, f"[+,-,-] failed: {dc_pmm}"
    dc_pmp = diagnose_crossings(
        [{"suffix_length": 128, "cand": 90.0, "base": 100.0}, {"suffix_length": 256, "cand": 110.0, "base": 100.0}, {"suffix_length": 512, "cand": 90.0, "base": 100.0}],
        "cand", "base",
    )
    assert dc_pmp["first_crossing"] == "<= 128" and dc_pmp["sustained_monotone"] == 384.0 and dc_pmp["is_monotone_after_first"] is False, f"[+,-,+] failed: {dc_pmp}"

    res_a = run_part_a_o1_e8(model, base_model, n_linear, 4, torch.device("cpu"), False, [32, 64, 128], n_repeats=2, n_warmup=1, p0=192)
    res_b = run_part_b_symmetric_e2e_routing(
        model, base_model, tok, eos_ids, n_linear, 4, torch.device("cpu"), False,
        "osc", None, [32, 64, 128], [0.0, 0.05, 0.20], n_repeats=2, p0=160, d_k=32, decode_steps=8, n_warmup=1, n_live_trials=4,
    )
    assert len(res_a["sweep"]) == 3 and len(res_b["rows"]) == 3
    for rw in res_b["rows"]:
        assert rw["decode_steps_executed"] == 8
        assert "prompt_ground_truth" in rw
        assert rw["detector_sample_feats"]["c_hat_mean"] != 0.1
        assert rw["detector_sample_feats"]["gamma_mean"] != 0.5
        assert "r_0.20" in rw["live_conditional_full_swap_by_fail_rate"]
        assert "live_detector_driven_full_swap" in rw
        assert "raw_repeats_ms" in rw["symmetric_h2d_ms"] and "mean_ms" in rw["symmetric_h2d_ms"]
    print("[DRY-RUN PASS] w14_optb_e2e_routing_bench.py crossings edge cases, interleaved Part A, ground-truth S-NIAH Part B & E2E OK.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--tag", type=str, default="2b_4060")
    parser.add_argument("--detector", type=str, default="osc", choices=["osc", "lr"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default=str(SCRIPT_DIR))
    parser.add_argument("--e8-suffixes", type=str, default="64,128,192,224,256,288,320,384,512,768,1024,1536,2048,3072,4096")
    parser.add_argument("--e2e-suffixes", type=str, default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--n-repeats-e8", type=int, default=20)
    parser.add_argument("--n-repeats-e2e", type=int, default=15)
    parser.add_argument("--n-warmup-e2e", type=int, default=5)
    parser.add_argument("--n-live-trials", type=int, default=40)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--synthetic-dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.synthetic_dry_run:
        run_synthetic_dry_run(out_dir)
        return

    e8_sufs = [int(x) for x in args.e8_suffixes.split(",") if x.strip()]
    e2e_sufs = [int(x) for x in args.e2e_suffixes.split(",") if x.strip()]
    if args.smoke:
        e8_sufs = [128, 256, 512]
        e2e_sufs = [128, 512]

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    t_cfg = getattr(cfg, "text_config", cfg)
    n_linear = count_linear_layers(t_cfg)
    d_k = int(getattr(t_cfg, "linear_key_head_dim", 128))
    w_conv = int(getattr(t_cfg, "linear_conv_kernel_dim", 4))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True, local_files_only=args.local_files_only,
    ).eval()
    base_model = getattr(model.model, "language_model", model.model)
    eos_ids = get_eos_ids(model, tok)
    use_l2k = has_logits_to_keep_arg(model)

    lr_blob = None
    if args.detector == "lr":
        dev_p = out_dir / "w11_detector_v2_clean_9b_records.jsonl"
        if not dev_p.exists():
            dev_p = SCRIPT_DIR / "w11_detector_v2_clean_9b_records.jsonl"
        lr_blob = freeze_lr_from_dev_split(dev_p, tau_lr=FROZEN_LR_TAU_DEV_9B)

    print(f"[OptB-1] Running Part A O(1) E8 on {args.model} ({torch.cuda.get_device_name(device)})...", flush=True)
    part_a = run_part_a_o1_e8(
        model, base_model, n_linear, w_conv, device, use_l2k,
        e8_sufs, n_repeats=(3 if args.smoke else args.n_repeats_e8), n_warmup=(1 if args.smoke else 3),
    )
    print("[OptB-1] Part A monotone crossovers:", json.dumps(part_a["monotone_crossover_T_star_suffix_tokens"]), flush=True)

    print(f"[OptB-1] Running Part B Symmetric E2E Routing Sweep on {args.model}...", flush=True)
    part_b = run_part_b_symmetric_e2e_routing(
        model, base_model, tok, eos_ids, n_linear, w_conv, device, use_l2k,
        args.detector, lr_blob, e2e_sufs, [0.00, 0.05, 0.10, 0.20],
        n_repeats=(2 if args.smoke else args.n_repeats_e2e),
        d_k=d_k,
        decode_steps=args.decode_steps,
        n_warmup=(1 if args.smoke else args.n_warmup_e2e),
        n_live_trials=(4 if args.smoke else args.n_live_trials),
    )
    print("[OptB-1] Part B routed crossovers:", json.dumps(part_b["monotone_routed_crossovers"]), flush=True)

    import transformers as _tf

    payload = {
        "runner": "pilot/w14_optb_e2e_routing_bench.py",
        "model": args.model,
        "tag": args.tag,
        "detector": args.detector,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "transformers_version": _tf.__version__,
        "timing_design": {
            "p0_part_a": int(part_a["p0_part_a"]),
            "p0_part_b": int(part_b["p0_part_b"]),
            "decode_steps": int(args.decode_steps),
            "eos_early_stop_disabled": True,
            "n_warmup_e2e": int(args.n_warmup_e2e),
            "n_live_trials": int(args.n_live_trials),
            "part_a_interleaved_repeats": True,
            "part_b_prompt_mode": "flippable_benign_sniah_with_oracle_ground_truth",
            "included_inside_single_timer": [
                "move_cache_device(c_pre_cpu)",
                "move_cache_device(c_ctx_old_cpu)",
                "block_forward",
                "o1_state_update_and_inplace_kv_patch",
                "fixed_8_step_decode",
                "detector_feature_extraction_and_scoring",
                "conditional_suffix_fallback_and_decode",
            ],
            "precomputed_outside_timer": [
                "g_cum_prefix_table",
                "l_cum_prefix_table",
                "lg_old_query_logits",
                "s_ob_old_block_state_snapshot",
            ],
            "o1_decay_lookup": "precomputed cumulative G_cum[:, T] - G_cum[:, b] & L_cum[:, T] - L_cum[:, b] (O(1) inside edit & detector timers)",
            "delta_L_0_kv_write": "strictly in-place [p0:b_end) patch_kv_slice (NO torch.cat)",
            "delta_L_pos4_varlen": "explicitly benchmarked separately with apply_delta_rope_to_kv_cache (torch.cat + suffix key rotation)",
            "symmetric_h2d_scope": "both routed O_local and always_suffix include move_cache_device(c_pre_cpu) INSIDE timers; full_swap also includes move_cache_device(c_ctx_old_cpu)",
            "routed_aggregation_arms": "reports both analytical formula synthesis (`routed_e2e_by_fail_rate`) and true single-timer conditional E2E execution (`live_conditional_full_swap_by_fail_rate`)",
            "crossover_rule": "strictly monotone slow-to-fast crossing + explicit first/last crossing diagnostics",
        },
        "part_a_o1_e8": part_a,
        "part_b_symmetric_e2e_routing": part_b,
    }
    out_p = out_dir / f"w14_optb_e2e_routing_{args.tag}_summary.json"
    out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Wrote {out_p}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
