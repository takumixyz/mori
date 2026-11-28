"""Stage-C Wave 11-2: Third Hybrid Architecture Family (`AI21-Jamba-Reasoning-3B`, Mamba-1 + Sparse NoPE Attention).

Strictly follows third-family scoping rules:
  - Excludes `Nemotron-H` (Mamba-2 + Attention -> 3rd model of 2nd family, not a 3rd family).
  - Excludes `LFM2.5` (short conv + Attention -> lacks recurrent state; O_local state surgery not directly applicable).
  - Evaluates `jamba3b` (`ai21labs/AI21-Jamba-Reasoning-3B`): Mamba-1 selective SSM (26 layers, `d_conv=4, d_state=16`)
    + Sparse NoPE Full Attention (2 layers `[7, 21]` of 28).
  - Supports `--one-sample-gate` (mandatory 1-sample O_local feasibility gate after 0-CU 3-check)
    and `--synthetic-dry-run` on CPU (`CUDA_VISIBLE_DEVICES=""`).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HOME", r"D:\applicationData\HuggingFace")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    JambaConfig,
    JambaForCausalLM,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from w2_crossfamily_e0c import (  # noqa: E402
    build_fact_prompt,
    check_oracle_clean,
    clone_cache,
    get_eos_ids,
    kl_div_from_logits,
    make_position_ids,
    manual_greedy,
    overwrite_ssm_states,
    patch_kv_range,
    run_segment as _orig_run_segment,
)


def run_segment(
    model: Any,
    cache: DynamicCache,
    ids_seg: torch.LongTensor,
    pos_start: int,
    max_chunk: int = 128,
) -> torch.Tensor:
    """Bounded-chunk wrapper to prevent pure-PyTorch Mamba-1 scan OOM on L4."""
    return _orig_run_segment(model, cache, ids_seg, pos_start, max_chunk=max_chunk)
from w2_crossfamily_e2_e3_e6 import (  # noqa: E402
    append_jsonl,
    build_multifact_prompt,
    eval_suffix_condition,
    load_completed_keys,
    mean_ans_kl,
    patch_conv_tail,
    sync_time,
    wilson_ci,
)
from w5_9b_e5_runner import wilcoxon_signed_rank_two_sided_p  # noqa: E402
from w11_crossfamily_scaleup import (  # noqa: E402
    CANDIDATE_WORDS,
    SCALED_NAMES,
    attach_kveraser_lora,
    build_case_pools,
    build_scaled_e3_configs,
    build_single_token_pairs,
    collect_shared_single_tokens,
    compute_hypic_vs_olocal_memory,
    detach_kveraser_lora,
    generate_disjoint_digit4_pairs,
    rotate_half,
    set_lora_enabled,
    train_kveraser_lora,
)


FAMILY_REGISTRY = {
    "jamba3b": {
        "repo_id": "ai21labs/AI21-Jamba-Reasoning-3B",
        "arch": "jamba",
        "has_recurrent": True,
        "has_rope": False,  # JambaAttention is NoPE
    },
}


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) * (0.5 ** n) for i in range(0, k + 1)) * 2.0
    return min(1.0, float(p))


def wilcoxon_p(a: List[float], b: List[float]) -> float:
    import numpy as np
    diffs = np.asarray([x - y for x, y in zip(a, b)], dtype=np.float64)
    if not np.any(np.abs(diffs) > 1e-18):
        return 1.0
    return float(wilcoxon_signed_rank_two_sided_p(diffs))


def extract_mixers(model: Any, arch: str) -> List[Tuple[int, Any]]:
    mixers = []
    for idx, layer in enumerate(model.model.layers):
        if arch == "jamba" and hasattr(layer, "mamba"):
            mixers.append((idx, layer.mamba))
    return mixers


def resolve_third_family_offset(arch: str, n_mixers: int, w_conv: int, horizon: str = "deep") -> int:
    if horizon == "span_only":
        return 0
    if horizon == "single":
        return max(1, w_conv - 1)
    # Jamba Mamba-1 has d_conv=4 across 26 layers
    return int((w_conv - 1) * max(1, n_mixers - 1))


def run_segment_with_jamba_decay(
    model: Any,
    cache: DynamicCache,
    ids_seg: torch.LongTensor,
    pos_start: int,
    mixers: List[Tuple[int, Any]],
    max_chunk: int = 128,
) -> Tuple[Optional[torch.Tensor], Dict[int, torch.Tensor]]:
    """Run ids_seg while accumulating exact fp64 Mamba-1 diagonal decay sum G_t = sum(dt * A)."""
    seq_len = int(ids_seg.shape[1])
    if seq_len == 0:
        empty = {
            li: torch.zeros((1, mix.intermediate_size, mix.ssm_state_size), device=ids_seg.device, dtype=torch.float64)
            for li, mix in mixers
        }
        return None, empty

    g_sums: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int, mixer: Any):
        def _hook(_mod, _inp, out):
            # out is output of dt_layernorm: (B, L, time_step_rank)
            if hasattr(mixer.config, "_is_quantized"):
                dt_pre = (mixer.dt_proj(out) - mixer.dt_proj.bias)
            else:
                dt_pre = F.linear(out, mixer.dt_proj.weight)
            if mixer.dt_proj.bias is not None:
                dt_pre = dt_pre + mixer.dt_proj.bias
            dt = F.softplus(dt_pre.float())  # (1, L, intermediate_size)
            A = -torch.exp(mixer.A_log.float())  # (intermediate_size, d_state)
            dt_sum = dt.sum(dim=1, dtype=torch.float64).squeeze(0).unsqueeze(-1)  # (D, 1)
            chunk_g = (dt_sum * A.double()).unsqueeze(0)  # (1, D, N)
            g_sums[layer_idx] = g_sums[layer_idx] + chunk_g if layer_idx in g_sums else chunk_g

        return _hook

    for li, mix in mixers:
        hooks.append(mix.dt_layernorm.register_forward_hook(make_hook(li, mix)))

    last_logits = None
    try:
        for off in range(0, seq_len, max_chunk):
            sub = ids_seg[:, off:off + max_chunk]
            pos = make_position_ids(pos_start + off, int(sub.shape[1]), ids_seg.device)
            out = model(input_ids=sub, position_ids=pos, past_key_values=cache, use_cache=True)
            last_logits = out.logits
    finally:
        for hk in hooks:
            hk.remove()
    return last_logits, g_sums


def compute_state_delta_fp64(cache_b_old: DynamicCache, cache_b_new: DynamicCache) -> Dict[int, torch.Tensor]:
    deltas: Dict[int, torch.Tensor] = {}
    for li, (l_old, l_new) in enumerate(zip(cache_b_old.layers, cache_b_new.layers)):
        if hasattr(l_old, "recurrent_states") and l_old.recurrent_states.get(0) is not None:
            deltas[li] = l_new.recurrent_states[0].double() - l_old.recurrent_states[0].double()
    return deltas


def apply_third_family_jlocal_update(
    cache_old_ctx: DynamicCache,
    cache_new_blk: DynamicCache,
    delta_b: Dict[int, torch.Tensor],
    g_sums: Dict[int, torch.Tensor],
    start_pos: int,
    block_end: int,
    T_ctx: int,
    w_conv: int,
    has_recurrent: bool,
    use_decay: bool = True,
    patch_state: bool = True,
    patch_kv: bool = True,
) -> DynamicCache:
    c_out = clone_cache(cache_old_ctx)
    if patch_state and has_recurrent:
        for li, d_s in delta_b.items():
            layer = c_out.layers[li]
            orig_dt = layer.recurrent_states[0].dtype
            if use_decay and li in g_sums:
                decay = torch.exp(g_sums[li])
                new_s = layer.recurrent_states[0].double() + decay * d_s
            else:
                new_s = layer.recurrent_states[0].double() + d_s
            layer.recurrent_states[0].copy_(new_s.to(orig_dt))
    if patch_state:
        patch_conv_tail(c_out, cache_new_blk, block_end, T_ctx, w_conv=w_conv)
    if patch_kv:
        patch_kv_range(c_out, cache_new_blk, start_pos, block_end)
    return c_out


def prepare_bckpt_cache(
    model: Any,
    cfg: Any,
    ids_o: torch.LongTensor,
    p0: int,
    chunk_size: int = 128,
) -> Tuple[DynamicCache, int]:
    c_pre = DynamicCache(config=cfg)
    c_pos = (p0 // chunk_size) * chunk_size
    if c_pos > 0:
        _ = run_segment(model, c_pre, ids_o[:, :c_pos], 0, max_chunk=128)
    if p0 > c_pos:
        _ = run_segment(model, c_pre, ids_o[:, c_pos:p0], c_pos, max_chunk=128)
    return c_pre, c_pos


def build_delete_and_shift_generic(
    model: Any,
    c_old_ctx: DynamicCache,
    p0: int,
    p1: int,
    T_ctx: int,
    has_rope: bool,
) -> Tuple[DynamicCache, int]:
    """Delete [p0, p1) from KV; apply RoPE shift (-delta) if model uses RoPE on full attention."""
    delta = p1 - p0
    c_ds = clone_cache(c_old_ctx)
    if has_rope and delta > 0:
        rotary = model.model.rotary_emb
        suf_len = T_ctx - p1
        if suf_len > 0:
            dev_m = next(model.parameters()).device
            dt_m = next(model.parameters()).dtype
            dummy = torch.empty((1, 1, suf_len, 1), device=dev_m, dtype=dt_m)
            pos_old = torch.arange(p1, T_ctx, device=dev_m).unsqueeze(0)
            pos_new = torch.arange(p0, T_ctx - delta, device=dev_m).unsqueeze(0)
            cos_o, sin_o = rotary(dummy, pos_old)
            cos_n, sin_n = rotary(dummy, pos_new)
    for layer in c_ds.layers:
        if getattr(layer, "keys", None) is None or getattr(layer, "values", None) is None:
            continue
        k = layer.keys
        v = layer.values
        k_pre, v_pre = k[:, :, :p0, :], v[:, :, :p0, :]
        k_suf, v_suf = k[:, :, p1:T_ctx, :], v[:, :, p1:T_ctx, :]
        if k_suf.shape[2] > 0 and has_rope and delta > 0:
            d_rot = cos_o.shape[-1]
            # Unrotate old pos then rotate new pos
            c_o, s_o = cos_o.unsqueeze(1)[..., :d_rot], sin_o.unsqueeze(1)[..., :d_rot]
            c_n, s_n = cos_n.unsqueeze(1)[..., :d_rot], sin_n.unsqueeze(1)[..., :d_rot]
            k_rot = k_suf[..., :d_rot]
            k_base = k_rot * c_o + rotate_half(k_rot) * (-s_o)
            k_new_rot = k_base * c_n + rotate_half(k_base) * s_n
            if d_rot < k_suf.shape[-1]:
                k_suf = torch.cat([k_new_rot, k_suf[..., d_rot:]], dim=-1)
            else:
                k_suf = k_new_rot
        layer.keys = torch.cat([k_pre, k_suf], dim=2).contiguous()
        layer.values = torch.cat([v_pre, v_suf], dim=2).contiguous()
    return c_ds, T_ctx - delta


def build_kveraser_surrogate(
    c_old_ctx: DynamicCache,
    c_lora: DynamicCache,
    p0: int,
    p1: int,
) -> DynamicCache:
    c_out = clone_cache(c_old_ctx)
    for l_dst, l_src in zip(c_out.layers, c_lora.layers):
        if getattr(l_dst, "keys", None) is not None and getattr(l_src, "keys", None) is not None:
            l_dst.keys[:, :, p0:p1, :].copy_(l_src.keys[:, :, p0:p1, :])
            l_dst.values[:, :, p0:p1, :].copy_(l_src.values[:, :, p0:p1, :])
    return c_out


def run_e0c_suite(
    model: Any, tok: Any, cfg: Any, mixers: List[Tuple[int, Any]],
    arch: str, has_recurrent: bool, device: torch.device, eos_ids: List[int],
    e2_cases: List[Dict[str, Any]], w_conv: int, max_n: int = 10,
) -> Dict[str, Any]:
    """Verify G1a (<=1e-5), G1b (<=1e-4), manual==generate, and conv-spillover horizon."""
    n_mix = len(mixers)
    b_deep = resolve_third_family_offset(arch, n_mix, w_conv, "deep")
    n_check = min(max_n, len(e2_cases))
    g1a_diffs, g1b_kls, g1b_diffs = [], [], []
    span_p1_kls, deep_kls = [], []
    manual_matches = 0

    for idx in range(n_check):
        case = e2_cases[idx]
        ids_o = torch.tensor([case["ids_o"]], device=device, dtype=torch.long)
        ids_e = torch.tensor([case["ids_n"]], device=device, dtype=torch.long)
        p0, p1, T_ctx = case["p0"], case["p1"], case["T_ctx"]
        ids_suf = ids_e[:, T_ctx:]
        b_end = min(T_ctx, p1 + b_deep)

        with torch.no_grad():
            # Oneshot oracle
            c_one = DynamicCache(config=cfg)
            lg_one_all = run_segment(model, c_one, ids_e, 0)
            gen_one = manual_greedy(model, clone_cache(c_one), lg_one_all[:, -1:, :], int(ids_e.shape[1]), 6, eos_ids)

            # Generate parity
            gen_hf = model.generate(ids_e, max_new_tokens=6, do_sample=False, pad_token_id=eos_ids[0] if eos_ids else 0)
            gen_hf_ids = gen_hf[0, int(ids_e.shape[1]):].tolist()
            if gen_hf_ids[:len(gen_one)] == gen_one:
                manual_matches += 1

            # Split oracle [0..b_end) -> [b_end..T_ctx)
            c_sp = DynamicCache(config=cfg)
            _ = run_segment(model, c_sp, ids_e[:, :b_end], 0)
            _ = run_segment(model, c_sp, ids_e[:, b_end:T_ctx], b_end)
            ev_sp = eval_suffix_condition(model, tok, c_sp, ids_suf, T_ctx, case["new"], case["old"], case["format"], 6, eos_ids)

            # J-full: forward [0..b_end) on ids_e, overwrite states into old_b + [b_end..T_ctx)
            c_old_ctx = DynamicCache(config=cfg)
            _ = run_segment(model, c_old_ctx, ids_o[:, :b_end], 0)
            c_new_b = DynamicCache(config=cfg)
            _ = run_segment(model, c_new_b, ids_e[:, :b_end], 0)
            c_jfull = clone_cache(c_old_ctx)
            for ld, ls in zip(c_jfull.layers, c_new_b.layers):
                if hasattr(ls, "conv_states") and ls.conv_states.get(0) is not None:
                    ld.conv_states[0].copy_(ls.conv_states[0])
                if has_recurrent and hasattr(ls, "recurrent_states") and ls.recurrent_states.get(0) is not None:
                    ld.recurrent_states[0].copy_(ls.recurrent_states[0])
                if getattr(ls, "keys", None) is not None and getattr(ld, "keys", None) is not None:
                    ld.keys[:, :, :b_end, :].copy_(ls.keys[:, :, :b_end, :])
                    ld.values[:, :, :b_end, :].copy_(ls.values[:, :, :b_end, :])
            _ = run_segment(model, c_jfull, ids_o[:, b_end:T_ctx], b_end)
            ev_jf = eval_suffix_condition(model, tok, c_jfull, ids_suf, T_ctx, case["new"], case["old"], case["format"], 6, eos_ids)

            # Compare oneshot at query boundary
            c_one_ctx = DynamicCache(config=cfg)
            _ = run_segment(model, c_one_ctx, ids_e[:, :T_ctx], 0)
            ev_one = eval_suffix_condition(model, tok, c_one_ctx, ids_suf, T_ctx, case["new"], case["old"], case["format"], 6, eos_ids)

            g1a_diffs.append(float((ev_jf["last_logits"].float() - ev_sp["last_logits"].float()).abs().max().item()))
            g1b_kls.append(kl_div_from_logits(ev_one["last_logits"], ev_jf["last_logits"]))
            g1b_diffs.append(float((ev_jf["last_logits"].float() - ev_one["last_logits"].float()).abs().max().item()))

            # Compare span_only (b=p1) vs deep (b=p1+b_deep) J-local on conv/state spillover
            c_pre, _ = prepare_bckpt_cache(model, cfg, ids_o, p0, chunk_size=128)
            c_ctx_full = DynamicCache(config=cfg)
            _ = run_segment(model, c_ctx_full, ids_o[:, :T_ctx], 0)

            # Span-only (b=p1)
            c_old_p1 = clone_cache(c_pre)
            _ = run_segment(model, c_old_p1, ids_o[:, p0:p1], p0)
            c_new_p1 = clone_cache(c_pre)
            _ = run_segment(model, c_new_p1, ids_e[:, p0:p1], p0)
            if has_recurrent:
                c_tmp1 = clone_cache(c_old_p1)
                _, g1 = run_segment_with_jamba_decay(model, c_tmp1, ids_o[:, p1:T_ctx], p1, mixers)
                del c_tmp1
            else:
                g1 = {}
            d1 = compute_state_delta_fp64(c_old_p1, c_new_p1)
            c_jl_p1 = apply_third_family_jlocal_update(
                c_ctx_full, c_new_p1, d1, g1, p0, p1, T_ctx, w_conv, has_recurrent, True, True, True
            )
            ev_p1 = eval_suffix_condition(model, tok, c_jl_p1, ids_suf, T_ctx, case["new"], case["old"], case["format"], 6, eos_ids)
            span_p1_kls.append(kl_div_from_logits(ev_one["last_logits"], ev_p1["last_logits"]))

            # Deep horizon
            c_old_d = clone_cache(c_pre)
            _ = run_segment(model, c_old_d, ids_o[:, p0:b_end], p0)
            c_new_d = clone_cache(c_pre)
            _ = run_segment(model, c_new_d, ids_e[:, p0:b_end], p0)
            if has_recurrent:
                c_tmpd = clone_cache(c_old_d)
                _, gd = run_segment_with_jamba_decay(model, c_tmpd, ids_o[:, b_end:T_ctx], b_end, mixers)
                del c_tmpd
            else:
                gd = {}
            dd = compute_state_delta_fp64(c_old_d, c_new_d)
            c_jl_d = apply_third_family_jlocal_update(
                c_ctx_full, c_new_d, dd, gd, p0, b_end, T_ctx, w_conv, has_recurrent, True, True, True
            )
            ev_d = eval_suffix_condition(model, tok, c_jl_d, ids_suf, T_ctx, case["new"], case["old"], case["format"], 6, eos_ids)
            deep_kls.append(kl_div_from_logits(ev_one["last_logits"], ev_d["last_logits"]))

            del c_one, c_sp, c_old_ctx, c_new_b, c_jfull, c_one_ctx, c_pre, c_old_p1, c_new_p1, c_jl_p1, c_old_d, c_new_d, c_jl_d, c_ctx_full

    return {
        "n": n_check,
        "b_deep_offset": b_deep,
        "manual_eq_generate": f"{manual_matches}/{n_check}",
        "g1a_pass": sum(1 for x in g1a_diffs if x <= 1e-5),
        "g1a_max_diff": max(g1a_diffs),
        "g1b_strict_pass": sum(1 for x in g1b_kls if x <= 1e-4),
        "g1b_mean_kl": sum(g1b_kls) / max(1, len(g1b_kls)),
        "g1b_max_diff": max(g1b_diffs),
        "span_p1_mean_kl": sum(span_p1_kls) / max(1, len(span_p1_kls)),
        "deep_jlocal_mean_kl": sum(deep_kls) / max(1, len(deep_kls)),
    }


def run_third_family_e2(
    model_key: str, model: Any, tok: Any, cfg: Any, mixers: List[Tuple[int, Any]],
    arch: str, has_recurrent: bool, device: torch.device, eos_ids: List[int],
    cases: List[Dict[str, Any]], w_conv: int, jsonl_path: Path,
) -> Dict[str, Any]:
    n_mix = len(mixers)
    b_offset = resolve_third_family_offset(arch, n_mix, w_conv, "deep")
    done = load_completed_keys(jsonl_path)

    conds = ["old", "oracle", "J-local", "kv_only_deep", "kv_only_span_p1", "J-local-evict", "J-control"]
    if has_recurrent:
        conds.extend(["J-local-stale", "J-local-recompute", "state_only"])

    for idx, case in enumerate(cases):
        rk = f"{model_key}:e2:{idx}"
        if rk in done:
            continue
        ids_o = torch.tensor([case["ids_o"]], device=device, dtype=torch.long)
        ids_e = torch.tensor([case["ids_n"]], device=device, dtype=torch.long)
        p0, p1, T_ctx = case["p0"], case["p1"], case["T_ctx"]
        block_end = min(T_ctx, p1 + b_offset)
        ids_suf = ids_e[:, T_ctx:]
        fmt = case["format"]

        with torch.no_grad():
            c_pre, _ = prepare_bckpt_cache(model, cfg, ids_o, p0, chunk_size=128)
            c_old_b = clone_cache(c_pre)
            _ = run_segment(model, c_old_b, ids_o[:, p0:block_end], p0)
            c_old_ctx = clone_cache(c_old_b)
            if has_recurrent:
                _, g_sums = run_segment_with_jamba_decay(model, c_old_ctx, ids_o[:, block_end:T_ctx], block_end, mixers)
            else:
                if T_ctx > block_end:
                    _ = run_segment(model, c_old_ctx, ids_o[:, block_end:T_ctx], block_end)
                g_sums = {}

            t0 = sync_time(device)
            c_new_b = clone_cache(c_pre)
            _ = run_segment(model, c_new_b, ids_e[:, p0:block_end], p0)
            delta_b = compute_state_delta_fp64(c_old_b, c_new_b)
            c_jlocal = apply_third_family_jlocal_update(
                c_old_ctx, c_new_b, delta_b, g_sums, p0, block_end, T_ctx, w_conv, has_recurrent, True, True, True
            )
            t_jlocal_ms = (sync_time(device) - t0) * 1000.0

            c_oracle = DynamicCache(config=cfg)
            _ = run_segment(model, c_oracle, ids_e[:, :T_ctx], 0)

            t0s = sync_time(device)
            c_suf_rec = clone_cache(c_pre)
            _ = run_segment(model, c_suf_rec, ids_e[:, p0:T_ctx], p0)
            t_suf_ms = (sync_time(device) - t0s) * 1000.0
            del c_suf_rec

            # KV-only deep [p0, block_end)
            c_kv_deep = apply_third_family_jlocal_update(
                c_old_ctx, c_new_b, delta_b, g_sums, p0, block_end, T_ctx, w_conv, has_recurrent, True, False, True
            )
            # KV-only span [p0, p1) without conv/block horizon closure
            c_new_p1 = clone_cache(c_pre)
            _ = run_segment(model, c_new_p1, ids_e[:, p0:p1], p0)
            c_kv_p1 = apply_third_family_jlocal_update(
                c_old_ctx, c_new_p1, {}, {}, p0, p1, T_ctx, w_conv, False, False, False, True
            )
            del c_new_p1

            # J-control: old KV at b + new SSM/conv at b, recompute [b, T_ctx)
            c_ctrl = clone_cache(c_old_b)
            overwrite_ssm_states(c_ctrl, c_new_b)
            if T_ctx > block_end:
                _ = run_segment(model, c_ctrl, ids_o[:, block_end:T_ctx], block_end)

            # J-local-evict: hard attention mask over [p0, p1) via eval_suffix_condition(..., evict_span=(p0, p1))
            c_evict_base = apply_third_family_jlocal_update(
                c_old_ctx, c_new_b, delta_b, g_sums, p0, block_end, T_ctx, w_conv, has_recurrent, True, True, False
            )

            cond_caches: Dict[str, Tuple[DynamicCache, Optional[Tuple[int, int]]]] = {
                "old": (c_old_ctx, None),
                "oracle": (c_oracle, None),
                "J-local": (c_jlocal, None),
                "kv_only_deep": (c_kv_deep, None),
                "kv_only_span_p1": (c_kv_p1, None),
                "J-local-evict": (c_evict_base, (p0, p1)),
                "J-control": (c_ctrl, None),
            }
            if has_recurrent:
                c_stale = apply_third_family_jlocal_update(
                    c_old_ctx, c_new_b, delta_b, g_sums, p0, block_end, T_ctx, w_conv, True, False, True, True
                )
                c_state_only = apply_third_family_jlocal_update(
                    c_old_ctx, c_new_b, delta_b, g_sums, p0, block_end, T_ctx, w_conv, True, True, True, False
                )
                c_recomp = clone_cache(c_new_b)
                if T_ctx > block_end:
                    _ = run_segment(model, c_recomp, ids_o[:, block_end:T_ctx], block_end)
                cond_caches["J-local-stale"] = (c_stale, None)
                cond_caches["J-local-recompute"] = (c_recomp, None)
                cond_caches["state_only"] = (c_state_only, None)

            evals = {
                cn: eval_suffix_condition(
                    model, tok, cc, ids_suf, T_ctx, case["new"], case["old"], fmt, 8, eos_ids, evict_span=ev_sp
                )
                for cn, (cc, ev_sp) in cond_caches.items()
            }

        lg_ora = evals["oracle"]["last_logits"]
        ans_ora = evals["oracle"]["ans_logits_new"]
        out_c = {}
        for cn in conds:
            ev = evals[cn]
            out_c[cn] = {
                "acc_new": ev["acc_new"],
                "acc_old": ev["acc_old"],
                "top1_eq_oracle": ev["top1_id"] == evals["oracle"]["top1_id"],
                "kl_vs_oracle": kl_div_from_logits(lg_ora, ev["last_logits"]),
                "ans_mean_kl": mean_ans_kl(ans_ora, ev["ans_logits_new"]),
                "gen_text": ev.get("gen_text", ""),
            }

        decay_max = float(max(torch.exp(v).max().item() for v in g_sums.values())) if g_sums else None
        delta_max = float(max(v.abs().max().item() for v in delta_b.values())) if delta_b else None
        append_jsonl(jsonl_path, {
            "record_key": rk, "suite": "E2", "model": model_key, "sample_idx": idx,
            "T_ctx": T_ctx, "p0": p0, "b_end": block_end,
            "jlocal_ms": t_jlocal_ms, "suffix_recompute_ms": t_suf_ms,
            "decay_max": decay_max, "delta_max": delta_max,
            "conditions": out_c,
        })
        done.add(rk)
        for cc, _ in cond_caches.values():
            del cc
        del c_pre, c_old_b, c_new_b
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l).get("suite") == "E2" and json.loads(l).get("model") == model_key]
    n = len(rows)
    agg = {}
    for cn in conds:
        k_new = sum(1 for r in rows if r["conditions"][cn]["acc_new"])
        k_old = sum(1 for r in rows if r["conditions"][cn]["acc_old"])
        k_t1 = sum(1 for r in rows if r["conditions"][cn]["top1_eq_oracle"])
        kls = [r["conditions"][cn]["kl_vs_oracle"] for r in rows]
        agg[cn] = {
            "n": n,
            "acc_new": k_new / max(1, n),
            "acc_new_ci95": wilson_ci(k_new, n),
            "acc_old": k_old / max(1, n),
            "top1_eq_oracle": k_t1 / max(1, n),
            "mean_kl_vs_oracle": sum(kls) / max(1, n),
        }

    # Exact paired tests vs J-local
    jl_em = [bool(r["conditions"]["J-local"]["acc_new"]) for r in rows]
    jl_kl = [float(r["conditions"]["J-local"]["kl_vs_oracle"]) for r in rows]
    tests = {}
    for comp in [c for c in ("kv_only_deep", "kv_only_span_p1", "state_only", "J-local-stale") if c in conds]:
        cp_em = [bool(r["conditions"][comp]["acc_new"]) for r in rows]
        cp_kl = [float(r["conditions"][comp]["kl_vs_oracle"]) for r in rows]
        b_cnt = sum(1 for a, b in zip(jl_em, cp_em) if a and not b)
        c_cnt = sum(1 for a, b in zip(jl_em, cp_em) if not a and b)
        tests[f"J-local_vs_{comp}"] = {
            "mcnemar_b_jlocal_only": b_cnt, "mcnemar_c_comp_only": c_cnt,
            "mcnemar_p": exact_mcnemar_p(b_cnt, c_cnt),
            "wilcoxon_kl_p": wilcoxon_p(jl_kl, cp_kl),
            "kl_ratio_comp_over_jlocal": (sum(cp_kl) / max(1, n)) / max(1e-12, sum(jl_kl) / max(1, n)),
        }

    return {
        "suite": "E2", "n": n, "b_offset": b_offset,
        "mean_jlocal_ms": sum(r["jlocal_ms"] for r in rows) / max(1, n),
        "mean_suffix_recompute_ms": sum(r["suffix_recompute_ms"] for r in rows) / max(1, n),
        "conditions": agg, "paired_tests": tests,
    }


def run_third_family_e3(
    model_key: str, model: Any, tok: Any, cfg: Any, mixers: List[Tuple[int, Any]],
    arch: str, has_recurrent: bool, device: torch.device, eos_ids: List[int],
    configs: List[Dict[str, Any]], w_conv: int, jsonl_path: Path,
) -> Dict[str, Any]:
    b_offset = resolve_third_family_offset(arch, len(mixers), w_conv, "deep")
    done = load_completed_keys(jsonl_path)

    for cfg_e3 in configs:
        set_id = cfg_e3["set_id"]
        mf = cfg_e3["mf"]
        edit_indices = cfg_e3["edit_indices"]
        edit_set = set(edit_indices)
        ids_base_list = mf["ids_ctx_o"]
        ids_base = torch.tensor([ids_base_list], device=device, dtype=torch.long)
        T_ctx = int(mf["T_ctx"])

        for p_idx, perm in enumerate(cfg_e3["perms"]):
            rk = f"{model_key}:e3:{set_id}:p{p_idx}"
            if rk in done:
                continue

            cur_ids = list(ids_base_list)
            with torch.no_grad():
                c_ctx = DynamicCache(config=cfg)
                _ = run_segment(model, c_ctx, ids_base[:, :T_ctx], 0)
                c_kv_ctx = clone_cache(c_ctx)

                for fact_idx in perm:
                    fm = mf["facts"][fact_idx]
                    p0, p1 = int(fm["p0"]), int(fm["p1"])
                    b_end = min(T_ctx, p1 + b_offset)
                    new_toks = list(fm["new_span_ids"])
                    if len(new_toks) != (p1 - p0):
                        raise ValueError(f"Expected equal-length E3 edit span at fact {fact_idx}: {len(new_toks)} != {p1 - p0}")

                    # Pre-fact state along cur_ids[:p0]
                    c_pre, _ = prepare_bckpt_cache(model, cfg, torch.tensor([cur_ids], device=device), p0, chunk_size=128)
                    c_old_b = clone_cache(c_pre)
                    _ = run_segment(model, c_old_b, torch.tensor([cur_ids[p0:b_end]], device=device), p0)

                    next_ids = list(cur_ids)
                    next_ids[p0:p1] = new_toks
                    c_new_b = clone_cache(c_pre)
                    _ = run_segment(model, c_new_b, torch.tensor([next_ids[p0:b_end]], device=device), p0)

                    if has_recurrent:
                        c_tmp = clone_cache(c_old_b)
                        _, g_sums = run_segment_with_jamba_decay(
                            model, c_tmp, torch.tensor([cur_ids[b_end:T_ctx]], device=device), b_end, mixers
                        )
                        del c_tmp
                    else:
                        g_sums = {}
                    delta_b = compute_state_delta_fp64(c_old_b, c_new_b)
                    c_ctx = apply_third_family_jlocal_update(
                        c_ctx, c_new_b, delta_b, g_sums, p0, b_end, T_ctx, w_conv, has_recurrent, True, True, True
                    )
                    c_kv_ctx = apply_third_family_jlocal_update(
                        c_kv_ctx, c_new_b, {}, {}, p0, b_end, T_ctx, w_conv, False, False, False, True
                    )
                    cur_ids = next_ids
                    del c_pre, c_old_b, c_new_b

                # Oracle on final edited context
                c_ora = DynamicCache(config=cfg)
                _ = run_segment(model, c_ora, torch.tensor([cur_ids[:T_ctx]], device=device), 0)

                ed_jl, ed_kv, ed_ora, col_jl, col_ora = [], [], [], [], []
                for fm in mf["facts"]:
                    is_edited = int(fm["fact_idx"]) in edit_set
                    expected = fm["new"] if is_edited else fm["old"]
                    other = fm["old"] if is_edited else fm["new"]
                    ids_q = torch.tensor([fm["query_suffix_ids"]], device=device, dtype=torch.long)
                    ev_jl = eval_suffix_condition(model, tok, c_ctx, ids_q, T_ctx, expected, other, fm["format"], 6, eos_ids)
                    ev_kv = eval_suffix_condition(model, tok, c_kv_ctx, ids_q, T_ctx, expected, other, fm["format"], 6, eos_ids)
                    ev_or = eval_suffix_condition(model, tok, c_ora, ids_q, T_ctx, expected, other, fm["format"], 6, eos_ids)
                    if is_edited:
                        ed_jl.append(bool(ev_jl["acc_new"]))
                        ed_kv.append(bool(ev_kv["acc_new"]))
                        ed_ora.append(bool(ev_or["acc_new"]))
                    else:
                        col_jl.append(bool(ev_jl["acc_new"]))
                        col_ora.append(bool(ev_or["acc_new"]))

                del c_ctx, c_kv_ctx, c_ora
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            append_jsonl(jsonl_path, {
                "record_key": rk, "suite": "E3", "model": model_key,
                "cell": "M4_k2" if set_id.startswith("M4_k2") else "M8_k4",
                "set_id": set_id, "perm_idx": p_idx,
                "edited_jlocal": sum(ed_jl), "edited_kv_only": sum(ed_kv), "edited_oracle": sum(ed_ora), "n_edited": len(ed_jl),
                "collat_jlocal": sum(col_jl), "collat_oracle": sum(col_ora), "n_collat": len(col_jl),
            })
            done.add(rk)

    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l).get("suite") == "E3" and json.loads(l).get("model") == model_key]
    agg = {}
    for cell in ("M4_k2", "M8_k4"):
        sub = [r for r in rows if r["cell"] == cell]
        if not sub:
            continue
        tot_e = sum(r["n_edited"] for r in sub)
        tot_c = sum(r["n_collat"] for r in sub)
        agg[cell] = {
            "runs": len(sub),
            "jlocal_edited_acc": sum(r["edited_jlocal"] for r in sub) / max(1, tot_e),
            "kv_only_edited_acc": sum(r["edited_kv_only"] for r in sub) / max(1, tot_e),
            "oracle_edited_acc": sum(r["edited_oracle"] for r in sub) / max(1, tot_e),
            "jlocal_collat_acc": sum(r["collat_jlocal"] for r in sub) / max(1, tot_c),
            "oracle_collat_acc": sum(r["collat_oracle"] for r in sub) / max(1, tot_c),
        }
    return agg


def run_third_family_sb1(
    model_key: str, model: Any, tok: Any, cfg: Any, mixers: List[Tuple[int, Any]],
    arch: str, has_recurrent: bool, has_rope: bool,
    device: torch.device, eos_ids: List[int],
    train_cases: List[Dict[str, Any]], test_cases: List[Dict[str, Any]],
    w_conv: int, jsonl_path: Path, c_seg: int = 128, w_seam: int = 32, lora_epochs: int = 3,
) -> Dict[str, Any]:
    b_offset = resolve_third_family_offset(arch, len(mixers), w_conv, "deep")
    done = load_completed_keys(jsonl_path)
    pending = [i for i in range(len(test_cases)) if f"{model_key}:sb1:{i}" not in done]

    adapters, lora_params, total_params = attach_kveraser_lora(model, rank=16, alpha=32.0)
    if pending:
        lora_stats = train_kveraser_lora(model, tok, cfg, adapters, train_cases, device, epochs=lora_epochs, lr=2e-4)
    else:
        lora_stats = {"skipped": True}
        set_lora_enabled(adapters, False)

    conds = ["old", "oracle", "J-local", "kv_only", "kveraser_lora_steer", "kveraser_delete_shift", "hypic_segmented_seam32", "hypic_chunk_zero_init"]

    for s_idx, case in enumerate(test_cases):
        rk = f"{model_key}:sb1:{s_idx}"
        if rk in done:
            continue
        ids_o = torch.tensor([case["ids_o"]], device=device, dtype=torch.long)
        ids_e = torch.tensor([case["ids_n"]], device=device, dtype=torch.long)
        p0, p1, T_ctx = case["p0"], case["p1"], case["T_ctx"]
        block_end = min(T_ctx, p1 + b_offset)
        ids_suf = ids_e[:, T_ctx:]
        fmt = case["format"]
        s0 = (p0 // c_seg) * c_seg
        s1 = min(T_ctx, s0 + c_seg)
        seam_end = min(T_ctx, s1 + w_seam)

        with torch.no_grad():
            set_lora_enabled(adapters, False)
            c_pre, _ = prepare_bckpt_cache(model, cfg, ids_o, p0, chunk_size=128)
            c_old_b = clone_cache(c_pre)
            _ = run_segment(model, c_old_b, ids_o[:, p0:block_end], p0)
            c_old_ctx = clone_cache(c_old_b)
            if has_recurrent:
                _, g_sums = run_segment_with_jamba_decay(model, c_old_ctx, ids_o[:, block_end:T_ctx], block_end, mixers)
            else:
                if T_ctx > block_end:
                    _ = run_segment(model, c_old_ctx, ids_o[:, block_end:T_ctx], block_end)
                g_sums = {}

            t0 = sync_time(device)
            c_new_b = clone_cache(c_pre)
            _ = run_segment(model, c_new_b, ids_e[:, p0:block_end], p0)
            delta_b = compute_state_delta_fp64(c_old_b, c_new_b)
            c_jlocal = apply_third_family_jlocal_update(
                c_old_ctx, c_new_b, delta_b, g_sums, p0, block_end, T_ctx, w_conv, has_recurrent, True, True, True
            )
            t_jl_ms = (sync_time(device) - t0) * 1000.0

            c_oracle = DynamicCache(config=cfg)
            _ = run_segment(model, c_oracle, ids_e[:, :T_ctx], 0)
            c_kv = apply_third_family_jlocal_update(
                c_old_ctx, c_new_b, {}, {}, p0, block_end, T_ctx, w_conv, False, False, False, True
            )

            # KVEraser LoRA steer
            t0 = sync_time(device)
            set_lora_enabled(adapters, True)
            c_lora = clone_cache(c_pre)
            _ = run_segment(model, c_lora, ids_e[:, p0:p1], p0)
            set_lora_enabled(adapters, False)
            c_kveraser = build_kveraser_surrogate(c_old_ctx, c_lora, p0, p1)
            t_kv_ms = (sync_time(device) - t0) * 1000.0

            # Delete-and-Shift
            c_del, T_del = build_delete_and_shift_generic(model, c_old_ctx, p0, p1, T_ctx, has_rope=has_rope)

            # HYPIC segmented + seam32
            t0 = sync_time(device)
            c_s0, _ = prepare_bckpt_cache(model, cfg, ids_o, s0, chunk_size=128)
            c_h_old = clone_cache(c_s0)
            _ = run_segment(model, c_h_old, ids_o[:, s0:seam_end], s0)
            c_h_new = clone_cache(c_s0)
            _ = run_segment(model, c_h_new, ids_e[:, s0:seam_end], s0)
            if has_recurrent:
                c_htmp = clone_cache(c_h_old)
                _, g_hyp = run_segment_with_jamba_decay(model, c_htmp, ids_o[:, seam_end:T_ctx], seam_end, mixers)
                del c_htmp
            else:
                g_hyp = {}
            dh = compute_state_delta_fp64(c_h_old, c_h_new)
            c_hyp_seg = apply_third_family_jlocal_update(
                c_old_ctx, c_h_new, dh, g_hyp, s0, seam_end, T_ctx, w_conv, has_recurrent, True, True, True
            )
            t_hyp_ms = (sync_time(device) - t0) * 1000.0

            # HYPIC zero-init chunk [s0, seam_end)
            c_hz_old = DynamicCache(config=cfg)
            _ = run_segment(model, c_hz_old, ids_o[:, s0:seam_end], s0)
            c_hz_new = DynamicCache(config=cfg)
            _ = run_segment(model, c_hz_new, ids_e[:, s0:seam_end], s0)
            dhz = compute_state_delta_fp64(c_hz_old, c_hz_new)
            c_hyp_zero = apply_third_family_jlocal_update(
                c_old_ctx, c_hz_new, dhz, g_hyp, s0, seam_end, T_ctx, w_conv, has_recurrent, True, True, False
            )
            seg_len = seam_end - s0
            for ld, ls in zip(c_hyp_zero.layers, c_hz_new.layers):
                if getattr(ls, "keys", None) is not None and getattr(ld, "keys", None) is not None:
                    ld.keys[:, :, s0:seam_end, :].copy_(ls.keys[:, :, :seg_len, :])
                    ld.values[:, :, s0:seam_end, :].copy_(ls.values[:, :, :seg_len, :])

            cmap = {
                "old": (c_old_ctx, T_ctx), "oracle": (c_oracle, T_ctx),
                "J-local": (c_jlocal, T_ctx), "kv_only": (c_kv, T_ctx),
                "kveraser_lora_steer": (c_kveraser, T_ctx), "kveraser_delete_shift": (c_del, T_del),
                "hypic_segmented_seam32": (c_hyp_seg, T_ctx), "hypic_chunk_zero_init": (c_hyp_zero, T_ctx),
            }
            evals = {cn: eval_suffix_condition(model, tok, cc, ids_suf, tp, case["new"], case["old"], fmt, 8, eos_ids) for cn, (cc, tp) in cmap.items()}

        lg_ora = evals["oracle"]["last_logits"]
        out_c = {}
        for cn in conds:
            ev = evals[cn]
            out_c[cn] = {
                "acc_new": ev["acc_new"], "acc_old": ev["acc_old"],
                "top1_eq_oracle": ev["top1_id"] == evals["oracle"]["top1_id"],
                "kl_vs_oracle": kl_div_from_logits(lg_ora, ev["last_logits"]),
            }
        append_jsonl(jsonl_path, {
            "record_key": rk, "suite": "SB1", "model": model_key, "sample_idx": s_idx, "T_ctx": T_ctx,
            "timing_ms": {"J-local": t_jl_ms, "kveraser_lora_steer": t_kv_ms, "hypic_segmented_seam32": t_hyp_ms},
            "conditions": out_c,
        })
        done.add(rk)
        for cc, _ in cmap.values():
            del cc
        del c_pre, c_old_b, c_new_b, c_lora, c_s0, c_h_old, c_h_new, c_hz_old, c_hz_new
        if device.type == "cuda":
            torch.cuda.empty_cache()

    detach_kveraser_lora(model)
    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l).get("suite") == "SB1" and json.loads(l).get("model") == model_key]
    n = len(rows)
    agg = {}
    for cn in conds:
        k_n = sum(1 for r in rows if r["conditions"][cn]["acc_new"])
        k_o = sum(1 for r in rows if r["conditions"][cn]["acc_old"])
        k_t = sum(1 for r in rows if r["conditions"][cn]["top1_eq_oracle"])
        kls = [r["conditions"][cn]["kl_vs_oracle"] for r in rows]
        agg[cn] = {
            "n": n, "acc_new": k_n / max(1, n), "acc_new_ci95": wilson_ci(k_n, n),
            "acc_old": k_o / max(1, n), "top1_eq_oracle": k_t / max(1, n),
            "mean_kl_vs_oracle": sum(kls) / max(1, n),
        }
    return {
        "suite": "SB1", "n": n, "lora_params": lora_params, "total_params": total_params,
        "lora_train_stats": lora_stats,
        "mean_timing_ms": {
            k: sum(r["timing_ms"][k] for r in rows) / max(1, n)
            for k in ("J-local", "kveraser_lora_steer", "hypic_segmented_seam32")
        },
        "conditions": agg,
    }


def run_synthetic_dry_run(out_dir: Path) -> None:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", local_files_only=True)
    shared = collect_shared_single_tokens([tok])
    pools = build_case_pools(tok, shared, n_e2=4, n_sb1_train=2, n_sb1_test=2, pre_fillers=10, post_fillers=16)
    e3_cfgs = build_scaled_e3_configs(tok, k2_n=1, k4_n=1)
    dry_jsonl = out_dir / "w12_c_third_family_dry.jsonl"
    if dry_jsonl.exists():
        dry_jsonl.unlink()

    # Jamba Mamba-1 + Sparse NoPE Attention CPU synthetic test
    j_cfg = JambaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        attn_layer_offset=1, attn_layer_period=2,
        mamba_d_conv=4, mamba_d_state=8, mamba_expand=2, mamba_dt_rank=4,
        num_experts=1, num_experts_per_tok=1, experts_per_token=1, pad_token_id=0,
    )
    jm = JambaForCausalLM(j_cfg).eval()
    eos_j = get_eos_ids(jm, tok)
    mix_j = extract_mixers(jm, "jamba")
    e0_j = run_e0c_suite(jm, tok, j_cfg, mix_j, "jamba", True, torch.device("cpu"), eos_j, pools["e2_cases"], 4, max_n=2)
    e2_j = run_third_family_e2("dry_jamba", jm, tok, j_cfg, mix_j, "jamba", True, torch.device("cpu"), eos_j, pools["e2_cases"], 4, dry_jsonl)
    e3_j = run_third_family_e3("dry_jamba", jm, tok, j_cfg, mix_j, "jamba", True, torch.device("cpu"), eos_j, e3_cfgs[:1], 4, dry_jsonl)
    sb_j = run_third_family_sb1("dry_jamba", jm, tok, j_cfg, mix_j, "jamba", True, False, torch.device("cpu"), eos_j, pools["sb1_train"], pools["sb1_test"], 4, dry_jsonl, lora_epochs=1)

    dry_jsonl.unlink(missing_ok=True)
    print(
        "[DRY-RUN PASS] w12_c_third_family.py verified Jamba Mamba-1:",
        f"g1a_pass={e0_j['g1a_pass']}/{e0_j['n']}",
        f"e2_acc={e2_j['conditions']['J-local']['acc_new']:.2f}",
        f"e3_done={sum(v['runs'] for v in e3_j.values())}",
        f"sb1_done={sb_j['n']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", type=str, default="jamba3b", choices=list(FAMILY_REGISTRY.keys()))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default=str(SCRIPT_DIR))
    parser.add_argument("--n-e2", type=int, default=100)
    parser.add_argument("--n-sb1-train", type=int, default=40)
    parser.add_argument("--n-sb1-test", type=int, default=60)
    parser.add_argument("--k2-n", type=int, default=10)
    parser.add_argument("--k4-n", type=int, default=5)
    parser.add_argument("--lora-epochs", type=int, default=3)
    parser.add_argument("--one-sample-gate", action="store_true", help="Run 1-sample O_local + E0c feasibility gate before scaling.")
    parser.add_argument("--close-think", action="store_true", help="Close trailing <think> tag with </think>\\n\\n for reasoning templates.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--synthetic-dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.synthetic_dry_run:
        run_synthetic_dry_run(out_dir)
        return

    meta = FAMILY_REGISTRY[args.model_key]
    repo_id = meta["repo_id"]
    arch = meta["arch"]
    has_rec = meta["has_recurrent"]
    has_rope = meta["has_rope"]

    if args.one_sample_gate:
        n_e2, n_tr, n_te, k2_n, k4_n = 4, 0, 0, 0, 0
    else:
        n_e2 = 4 if args.smoke else args.n_e2
        n_tr = 4 if args.smoke else args.n_sb1_train
        n_te = 4 if args.smoke else args.n_sb1_test
        k2_n = 1 if args.smoke else args.k2_n
        k4_n = 1 if args.smoke else args.k4_n
    post_f = 24

    gate_path = out_dir / f"w12_c_{args.model_key}_gate1_summary.json"
    try:
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, local_files_only=args.local_files_only)
        cfg = AutoConfig.from_pretrained(repo_id, trust_remote_code=True, local_files_only=args.local_files_only)
        device = torch.device(args.device)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16, device_map=args.device,
            trust_remote_code=True, local_files_only=args.local_files_only,
        ).eval()
    except (ImportError, ModuleNotFoundError) as exc:
        if args.one_sample_gate:
            gate_path.write_text(json.dumps({
                "model_key": args.model_key, "repo_id": repo_id,
                "gate_verdict": {"all_3_gates_plus_jlocal_pass": False, "failure_category": "DEPENDENCY_MISSING", "error": repr(exc)},
            }, indent=2, ensure_ascii=False), encoding="utf-8")
        raise

    eos_ids = get_eos_ids(model, tok)
    mixers = extract_mixers(model, arch)
    w_conv = int(getattr(cfg, "mamba_d_conv", getattr(cfg, "conv_L_cache", 4)))

    if args.close_think:
        import w2_crossfamily_e0c as _w2_e0c
        _orig_render = _w2_e0c.render_chat_prompt

        def _closed_think_render(t: Any, msg: str) -> str:
            rendered = _orig_render(t, msg)
            return rendered + "</think>\n\n" if rendered.endswith("<think>\n") else rendered

        _w2_e0c.render_chat_prompt = _closed_think_render

    shared_words = collect_shared_single_tokens([tok])
    pools = build_case_pools(
        tok, shared_words,
        n_e2=n_e2, n_sb1_train=n_tr, n_sb1_test=n_te,
        pre_fillers=30, post_fillers=post_f,
    )

    if args.one_sample_gate:
        import traceback
        gate_jsonl = out_dir / f"w12_c_{args.model_key}_gate1_records.jsonl"
        gate_jsonl.unlink(missing_ok=True)
        print(f"[W11-2 Third Family 1-Sample Real-Weight Gate] {args.model_key} ({repo_id}) | mixers={len(mixers)} | w_conv={w_conv}", flush=True)

        num_experts = int(getattr(cfg, "num_experts", 1))
        experts_per_tok = int(getattr(cfg, "num_experts_per_tok", getattr(cfg, "experts_per_token", 1)))
        has_moe_router = any(hasattr(getattr(l, "feed_forward", None), "router") for l in model.model.layers)

        try:
            model.eval()
            c0_case = pools["e2_cases"][0]
            ids0 = torch.tensor([c0_case["ids_o"]], device=device, dtype=torch.long)
            with torch.no_grad():
                c_probe, _ = prepare_bckpt_cache(model, cfg, ids0, c0_case["p0"], chunk_size=128)
            layer_diag = [
                {
                    "idx": i,
                    "cls": type(l).__name__,
                    "has_conv_states": hasattr(l, "conv_states"),
                    "has_rec_states": hasattr(l, "recurrent_states"),
                    "has_kv": getattr(l, "keys", None) is not None,
                    "attrs": [a for a in dir(l) if not a.startswith("_")][:20],
                }
                for i, l in enumerate(c_probe.layers[:10])
            ]
            n_conv_acc = sum(1 for l in c_probe.layers if hasattr(l, "conv_states") and l.conv_states.get(0) is not None)
            n_rec_acc = sum(1 for l in c_probe.layers if hasattr(l, "recurrent_states") and l.recurrent_states.get(0) is not None)
            n_kv_acc = sum(1 for l in c_probe.layers if getattr(l, "keys", None) is not None and getattr(l, "values", None) is not None)
            gate1_ok = bool(len(mixers) > 0 and n_conv_acc == len(mixers) and n_rec_acc == len(mixers) and n_kv_acc > 0)
            del c_probe
            if device.type == "cuda":
                torch.cuda.empty_cache()

            e0c_res = run_e0c_suite(model, tok, cfg, mixers, arch, has_rec, device, eos_ids, pools["e2_cases"], w_conv, max_n=1)
            e2_gate = run_third_family_e2(f"{args.model_key}_gate1", model, tok, cfg, mixers, arch, has_rec, device, eos_ids, pools["e2_cases"][:1], w_conv, gate_jsonl)

            gate2_ok = bool(int(e0c_res.get("g1a_pass", 0)) == 1 and float(e0c_res.get("g1a_max_diff", 9e9)) <= 1e-5)
            conds_g = e2_gate.get("conditions", {})
            ora_ok = float(conds_g.get("oracle", {}).get("acc_new", 0.0)) >= 1.0
            jl_ok = float(conds_g.get("J-local", {}).get("acc_new", 0.0)) >= 1.0
            gate3_ok = bool(ora_ok)
            all_3_ok = bool(gate1_ok and gate2_ok and gate3_ok and jl_ok)

            if all_3_ok:
                fail_cat = "NONE_PASSED"
            elif not gate1_ok:
                fail_cat = "INCONCLUSIVE_CACHE_SCHEMA_UNVERIFIED"
            else:
                fail_cat = "INCONCLUSIVE"

            gate_summ = {
                "model_key": args.model_key,
                "repo_id": repo_id,
                "arch": arch,
                "template_mode": "closed_empty_think_prefill" if args.close_think else "raw_apply_chat_template",
                "tested_claim_scope": (
                    "C2 external validity ONLY (C3 GDN transport-decay inapplicable to Mamba-1 + NoPE; "
                    "this checkpoint is dense num_experts=1 unless num_experts > 1)"
                ),
                "n_layers": int(getattr(cfg, "num_hidden_layers", len(model.model.layers))),
                "n_mixers": len(mixers),
                "w_conv": w_conv,
                "has_recurrent": has_rec,
                "has_rope": has_rope,
                "moe_meta": {
                    "num_experts": num_experts,
                    "num_experts_per_tok": experts_per_tok,
                    "has_moe_router": has_moe_router,
                    "note": "MoE router selection across edited hidden states can perturb boundary equivalence when num_experts > 1.",
                },
                "gate_verdict": {
                    "gate1_state_access_ckpt_ok": gate1_ok,
                    "n_conv_accessible": n_conv_acc,
                    "n_rec_accessible": n_rec_acc,
                    "n_kv_accessible": n_kv_acc,
                    "gate2_g1a_eq_diff_zero_ok": gate2_ok,
                    "g1a_max_diff": e0c_res.get("g1a_max_diff"),
                    "gate3_oracle_prescreen_ok": gate3_ok,
                    "jlocal_1sample_acc": float(conds_g.get("J-local", {}).get("acc_new", 0.0)),
                    "all_3_gates_plus_jlocal_pass": all_3_ok,
                    "verdict": ("PASS" if all_3_ok else fail_cat),
                    "failure_category": fail_cat,
                },
                "layer_diag_sample": layer_diag,
                "gate_e0c_1sample": e0c_res,
                "gate_e2_1sample": e2_gate,
            }
        except (ImportError, ModuleNotFoundError) as exc:
            gate_summ = {
                "model_key": args.model_key, "repo_id": repo_id,
                "gate_verdict": {"all_3_gates_plus_jlocal_pass": False, "failure_category": "DEPENDENCY_MISSING", "error": repr(exc), "trace": traceback.format_exc()},
            }
        except (AttributeError, KeyError) as exc:
            gate_summ = {
                "model_key": args.model_key, "repo_id": repo_id,
                "gate_verdict": {"all_3_gates_plus_jlocal_pass": False, "failure_category": "NAMING_OR_CACHE_SCHEMA_MISMATCH", "error": repr(exc), "trace": traceback.format_exc()},
            }
        except Exception as exc:
            gate_summ = {
                "model_key": args.model_key, "repo_id": repo_id,
                "gate_verdict": {"all_3_gates_plus_jlocal_pass": False, "failure_category": "GENUINE_OR_UNCLASSIFIED_RUNTIME_ERROR", "error": repr(exc), "trace": traceback.format_exc()},
            }
        gate_path.write_text(json.dumps(gate_summ, indent=2, ensure_ascii=False), encoding="utf-8")
        print("  Gate Verdict:", gate_summ["gate_verdict"], flush=True)
        print(f"[GATE1 DONE] Wrote {gate_path}", flush=True)
        return

    e3_cfgs = build_scaled_e3_configs(tok, k2_n=k2_n, k4_n=k4_n)

    jsonl_path = out_dir / f"w12_c_{args.model_key}_records.jsonl"
    print(f"[W11-2 Third Family] {args.model_key} ({repo_id}) | mixers={len(mixers)} | w_conv={w_conv}", flush=True)

    e0c_res = run_e0c_suite(model, tok, cfg, mixers, arch, has_rec, device, eos_ids, pools["e2_cases"], w_conv)
    print("  E0c:", e0c_res, flush=True)
    e2_res = run_third_family_e2(args.model_key, model, tok, cfg, mixers, arch, has_rec, device, eos_ids, pools["e2_cases"], w_conv, jsonl_path)
    print("  E2:", {k: v["acc_new"] for k, v in e2_res["conditions"].items()}, flush=True)
    e3_res = run_third_family_e3(args.model_key, model, tok, cfg, mixers, arch, has_rec, device, eos_ids, e3_cfgs, w_conv, jsonl_path)
    print("  E3:", e3_res, flush=True)
    sb1_res = run_third_family_sb1(args.model_key, model, tok, cfg, mixers, arch, has_rec, has_rope, device, eos_ids, pools["sb1_train"], pools["sb1_test"], w_conv, jsonl_path, lora_epochs=args.lora_epochs)
    print("  SB1:", {k: v["acc_new"] for k, v in sb1_res["conditions"].items()}, flush=True)

    summ = {
        "model_key": args.model_key,
        "repo_id": repo_id,
        "arch": arch,
        "n_layers": int(getattr(cfg, "num_hidden_layers", len(model.model.layers))),
        "n_mixers": len(mixers),
        "w_conv": w_conv,
        "has_recurrent": has_rec,
        "has_rope": has_rope,
        "e0c": e0c_res,
        "e2": e2_res,
        "e3": e3_res,
        "sb1": sb1_res,
    }
    summ_path = out_dir / f"w12_c_{args.model_key}_summary.json"
    summ_path.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Wrote {summ_path}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
