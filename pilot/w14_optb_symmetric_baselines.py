"""Option B Exp-2: Strictly Symmetric HYPIC vs O_local & Single-Edit T5 KVEraser Benchmark.

Closes Codex B2 & Blind-Review Major B across GDN (Qwen3.5-2B/9B) and Mamba-2 (Falcon-H1-1.5B, Zamba2-1.2B):
  1. GDN (`arch == "gdn"`): constructs TRUE input-dependent HYPIC segment operators `Phi_m`
     across [seam_end, T_ctx) using real captured `(k_t, g_t, beta_t)` via `apply_householder_wy_fp64`
     (NOT `torch.eye`), times dense `Phi_m` chain composition vs `O_local`, AND verifies post-edit
     output logit `kl_vs_oracle` & `top1_eq_oracle` for both arms.
  2. Mamba-2 (`falcon_h1` / `zamba2`): captures REAL Mamba-2 `g_step = dt_eff * A` prefixes during
     prefill (no zero proxies), separates **faithful Mamba-2 diagonal segment composition**
     (`hypic_mamba2_diagonal_sym`, zero `d_state^2` quadratic bloat) from a clearly labeled
     **generic dense-Phi memory-stress baseline** (`dense_phi_memory_stress_proxy`), and distinguishes
     `precheck_exceeds_free_88pct` from actual caught `torch.cuda.OutOfMemoryError` during VRAM probes.
  3. KVEraser T5 breakdown: selects non-trivial training edit case when available, logs per-step and
     K-step single-edit gradient latency + surrogate splice, and verifies BOTH pre- vs post-edit
     edited accuracy/KL AND held-out unedited retention/KL drift.

Supports `--synthetic-dry-run` on CPU (`CUDA_VISIBLE_DEVICES=""`).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    FalconH1Config,
    FalconH1ForCausalLM,
    Qwen3_5ForCausalLM,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from w2_crossfamily_e0c import (  # noqa: E402
    MODEL_REGISTRY,
    clone_cache as clone_mamba_cache,
    extract_mamba_mixers,
    get_eos_ids,
    kl_div_from_logits,
    make_position_ids as make_mamba_pos,
    patch_kv_range,
    resolve_block_offset,
    run_segment,
)
from w2_crossfamily_e2_e3_e6 import (  # noqa: E402
    apply_mamba2_jlocal_update,
    compute_state_delta_fp64 as mamba_delta_fp64,
    eval_suffix_condition,
    prepare_bckpt_cache,
)
from w5_9b_e5_runner import (  # noqa: E402
    GDNSuffixHookCollector,
    apply_householder_wy_fp64,
    clone_cache as clone_gdn_cache,
    compute_state_delta_fp64 as gdn_delta_fp64,
    count_linear_layers,
    eval_from_ctx_cache_in_place,
    extract_kv_slice,
    has_logits_to_keep_arg,
    make_position_ids as make_gdn_pos,
    patch_kv_slice,
    run_chunked_prefill,
    snap_states,
)
from w11_crossfamily_scaleup import (  # noqa: E402
    attach_kveraser_lora,
    build_case_pools,
    build_kveraser_surrogate_cache,
    collect_shared_single_tokens,
    detach_kveraser_lora,
    resolve_answer_tokens,
    set_lora_enabled,
)
from w11_close_traps import build_gdn_splits, train_gdn_kveraser_lora  # noqa: E402
from w11_gdn_kveraser_hypic import build_gdn_kveraser_surrogate  # noqa: E402
from w14_optb_e2e_routing_bench import apply_jlocal_update_o1, capture_prefix_cum_decay  # noqa: E402


def _sync(dev: torch.device | str) -> float:
    d = torch.device(dev)
    if d.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(d)
    return time.perf_counter()


def _model_geom(model: Any, arch: str, cfg: Any) -> Dict[str, int]:
    t_cfg = getattr(cfg, "text_config", cfg)
    if arch == "gdn":
        n_lin = count_linear_layers(t_cfg)
        n_heads = int(getattr(t_cfg, "linear_num_value_heads", 16))
        d_k = int(getattr(t_cfg, "linear_key_head_dim", 128))
        d_v = int(getattr(t_cfg, "linear_value_head_dim", 128))
        w_conv = int(getattr(t_cfg, "linear_conv_kernel_dim", 4))
        d_state = d_k
        d_val = d_v
    else:
        mixers = extract_mamba_mixers(model)
        n_lin = len(mixers)
        mix0 = mixers[0][1]
        n_heads = int(mix0.num_heads)
        d_state = int(mix0.ssm_state_size)
        d_val = int(mix0.head_dim)
        w_conv = int(getattr(t_cfg, "mamba_d_conv", getattr(t_cfg, "conv_kernel_size", 4)))
    return {
        "n_linear": n_lin,
        "n_heads": n_heads,
        "d_state": d_state,
        "d_val": d_val,
        "w_conv": w_conv,
    }


def _build_real_gdn_phi_chain(
    hs_phi: Dict[str, Any],
    n_lin: int,
    n_heads: int,
    d_state: int,
    c_seg: int,
    m_rem: int,
    device: torch.device,
) -> torch.Tensor:
    """Build true input-dependent segment matrices `Phi_m = prod exp(g_t)(I - beta_t k_t k_t^T)`."""
    if m_rem <= 0:
        return torch.empty((0, n_lin * n_heads, d_state, d_state), device=device, dtype=torch.bfloat16)
    eye_dk = torch.eye(d_state, device=device, dtype=torch.float64).view(1, 1, d_state, d_state).expand(
        1, n_heads, d_state, d_state
    )
    seg_rows: List[torch.Tensor] = []
    suffix_len = int(hs_phi["suffix_len"])
    for mi in range(m_rem):
        t0 = mi * c_seg
        t1 = min(suffix_len, t0 + c_seg)
        layer_blocks: List[torch.Tensor] = []
        for li in range(n_lin):
            k_seg = hs_phi["k_cpu"][li][:, t0:t1].to(device=device)
            b_seg = hs_phi["beta_cpu"][li][:, t0:t1].to(device=device)
            g_seg = hs_phi["g_cpu"][li][:, t0:t1].to(device=device)
            phi_l = apply_householder_wy_fp64(eye_dk, k_seg, b_seg, g_seg, chunk_size=64).squeeze(0).to(torch.bfloat16)
            layer_blocks.append(phi_l)
        seg_rows.append(torch.cat(layer_blocks, dim=0))
    return torch.stack(seg_rows, dim=0).contiguous()


def _capture_mamba2_g_cum(
    model: Any,
    cfg: Any,
    mixers: List[Tuple[int, Any]],
    ids: torch.Tensor,
    n_heads: int,
    max_chunk: int = 1024,
) -> Tuple[DynamicCache, Dict[int, torch.Tensor]]:
    """Run Mamba-2 prefill while recording real cumulative `G_cum[:, :T+1, H]` per mixer layer."""
    g_chunks: Dict[int, List[torch.Tensor]] = {li: [] for li, _ in mixers}
    hooks = []

    def _mk_hook(layer_idx: int, mixer: Any):
        def _hk(_mod, _inp, out):
            proj = out * mixer.mup_vector if hasattr(mixer, "mup_vector") else out
            dt_raw = proj[..., mixer.intermediate_size + mixer.conv_dim :]
            dt_eff = torch.clamp(
                F.softplus(dt_raw.float() + mixer.dt_bias.float()),
                min=mixer.time_step_limit[0],
                max=mixer.time_step_limit[1],
            )
            a_val = -torch.exp(mixer.A_log.float())
            g_chunks[layer_idx].append((dt_eff * a_val).detach().double())

        return _hk

    for li, mix in mixers:
        hooks.append(mix.in_proj.register_forward_hook(_mk_hook(li, mix)))

    c_full = DynamicCache(config=cfg)
    try:
        _ = run_segment(model, c_full, ids, 0, max_chunk=max_chunk)
    finally:
        for hk in hooks:
            hk.remove()

    g_cum_arr: Dict[int, torch.Tensor] = {}
    for li, chunks in g_chunks.items():
        g_seq = torch.cat(chunks, dim=1)
        z = torch.zeros((1, 1, n_heads), device=g_seq.device, dtype=torch.float64)
        g_cum_arr[li] = torch.cat([z, g_seq.cumsum(dim=1)], dim=1)
    return c_full, g_cum_arr


def measure_hypic_vs_olocal_symmetric(
    model: Any,
    arch: str,
    cfg: Any,
    geom: Dict[str, int],
    device: torch.device,
    t_ctx_list: List[int],
    n_repeats: int = 8,
    c_seg: int = 128,
    w_seam: int = 32,
) -> Dict[str, Any]:
    """Symmetric online edit timing, faithfulness vs oracle, & real CUDA allocation probe."""
    base_model = getattr(getattr(model, "model", model), "language_model", getattr(model, "model", model))
    use_l2k = has_logits_to_keep_arg(model) if arch == "gdn" else False
    n_lin = geom["n_linear"]
    n_heads = geom["n_heads"]
    d_state = geom["d_state"]
    w_conv = geom["w_conv"]
    b_offset = 3 * max(1, n_lin - 1)
    mixers = extract_mamba_mixers(model) if arch != "gdn" else []

    rows: List[Dict[str, Any]] = []
    vram_probe: List[Dict[str, Any]] = []

    for T_ctx in t_ctx_list:
        p0 = max(64, T_ctx // 4)
        b_end = min(T_ctx, p0 + 4 + b_offset)
        s0 = (p0 // c_seg) * c_seg
        seam_end = min(T_ctx, p0 + 4 + w_seam)
        n_segs = max(1, (T_ctx + c_seg - 1) // c_seg)
        m_rem = max(0, (T_ctx - seam_end + c_seg - 1) // c_seg)

        # 1. Measure real PyTorch GPU/CPU allocation
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            b0 = torch.cuda.memory_allocated(device)
            free_vram_now, total_vram_now = torch.cuda.mem_get_info(device)
        else:
            b0 = 0
            free_vram_now, total_vram_now = None, None

        # Dense Phi_m footprint: faithful for GDN rank-1 outer products, stress-only proxy for diagonal Mamba-2
        phi_theo_bytes = int(n_segs * n_lin * n_heads * d_state * d_state * 2)
        diag_phi_theo_bytes = int(n_segs * n_lin * n_heads * 8)
        precheck_88pct = bool(free_vram_now is not None and phi_theo_bytes > int(free_vram_now * 0.88))
        phi_oom = False
        oom_reason = "none"
        phi_meas_bytes: Optional[int] = None

        try:
            phi_alloc = torch.empty((n_segs, n_lin * n_heads, d_state, d_state), device=device, dtype=torch.bfloat16)
            phi_meas_bytes = (
                (torch.cuda.memory_allocated(device) - b0)
                if (torch.cuda.is_available() and device.type == "cuda")
                else phi_theo_bytes
            )
            del phi_alloc
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower():
                phi_meas_bytes = None
                phi_oom = True
                oom_reason = "caught_cuda_oom"
            else:
                raise
        finally:
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()

        b1 = torch.cuda.memory_allocated(device) if (torch.cuda.is_available() and device.type == "cuda") else 0
        g_alloc = torch.empty((1, n_lin, T_ctx + 1, n_heads), device=device, dtype=torch.float64)
        g_theo_bytes = int(g_alloc.numel() * g_alloc.element_size())
        g_meas_bytes = (
            (torch.cuda.memory_allocated(device) - b1)
            if (torch.cuda.is_available() and device.type == "cuda")
            else g_theo_bytes
        )
        g_seg_theo_bytes = int(n_segs * n_lin * n_heads * 8)
        del g_alloc

        cuda_platform = "windows_wddm" if os.name == "nt" else "linux_dedicated_vram"
        wddm_concurrency_risk = bool(os.name == "nt" and free_vram_now is not None and phi_theo_bytes > int(free_vram_now * 0.50))
        vram_probe.append({
            "T_ctx": int(T_ctx),
            "n_segs": int(n_segs),
            "d_state": int(d_state),
            "cuda_platform": cuda_platform,
            "vram_allocation_method": "cuda_memory_allocated_delta_on_torch_empty",
            "windows_wddm_paging_risk": bool(os.name == "nt" and precheck_88pct),
            "windows_wddm_concurrency_risk_50pct": wddm_concurrency_risk,
            "free_vram_before_bytes": free_vram_now,
            "total_vram_bytes": total_vram_now,
            "dense_phi_faithful_for_arch": bool(arch == "gdn"),
            "dense_phi_role": "exact_gdn_segment_operator" if arch == "gdn" else "dense_phi_memory_stress_proxy_only",
            "hypic_dense_phi_theoretical_bytes": phi_theo_bytes,
            "hypic_dense_phi_measured_cuda_bytes": phi_meas_bytes,
            "hypic_mamba2_diagonal_phi_bytes": diag_phi_theo_bytes if arch != "gdn" else None,
            "hypic_phi_oom": phi_oom,
            "oom_reason": oom_reason,
            "precheck_exceeded_free_88pct": precheck_88pct,
            "olocal_gcum_theoretical_bytes": g_theo_bytes,
            "olocal_gcum_measured_cuda_bytes": g_meas_bytes,
            "olocal_gseg_theoretical_bytes": g_seg_theo_bytes,
            "measured_dense_vram_ratio_vs_gcum": round(phi_meas_bytes / max(1, g_meas_bytes), 2) if phi_meas_bytes else None,
            "theoretical_dense_vram_ratio_vs_gseg": round(phi_theo_bytes / max(1, g_seg_theo_bytes), 2),
        })
        if arch == "gdn" and phi_oom:
            continue

        # 2. Check realistic headroom & run timing (only skip oversized dense Phi stress on Mamba-2, never skip diagonal)
        run_dense_stress = bool(arch != "gdn" and (not phi_oom) and (not precheck_88pct) and (not wddm_concurrency_risk))
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_b, _tot_b = torch.cuda.mem_get_info(device)
            t_cfg_inner = getattr(cfg, "text_config", cfg)
            n_tot_l = int(getattr(t_cfg_inner, "num_hidden_layers", n_lin))
            n_attn_l = max(1, n_tot_l - n_lin)
            kv_per_tok = n_attn_l * 4 * 256 * 4
            op_bytes = phi_theo_bytes if arch == "gdn" else diag_phi_theo_bytes
            est_need_b = int(op_bytes * 1.15 + T_ctx * kv_per_tok * 3.0 + 350 * 1024 * 1024)
            if T_ctx > 2048 and free_b < est_need_b:
                continue

        torch.manual_seed(1100 + T_ctx)
        ids_o = torch.randint(100, 1500, (1, T_ctx), device=device, dtype=torch.long)
        ids_e = ids_o.clone()
        ids_e[:, p0 : p0 + 4] = torch.tensor([[111, 222, 333, 444]], device=device, dtype=torch.long)
        ids_q_chk = torch.tensor([[120, 121, 122]], device=device, dtype=torch.long)

        try:
          with torch.no_grad():
            if arch == "gdn":
                if p0 > 0:
                    _, c_pre = run_chunked_prefill(model, base_model, ids_o[:, :p0], 0, None, 2048, use_l2k)
                else:
                    c_pre = None
                _, c_old_b = run_chunked_prefill(model, base_model, ids_o[:, p0:b_end], p0, clone_gdn_cache(c_pre), 2048, use_l2k)
                s_ob = snap_states(c_old_b)
                c_ctx_old, g_cum = capture_prefix_cum_decay(model, base_model, ids_o[:, :T_ctx], n_lin, use_l2k)
                if s0 > 0:
                    _, c_s0 = run_chunked_prefill(model, base_model, ids_o[:, :s0], 0, None, 2048, use_l2k)
                else:
                    c_s0 = None
                _, c_hyp_old_end = run_chunked_prefill(model, base_model, ids_o[:, s0:seam_end], s0, clone_gdn_cache(c_s0), 2048, use_l2k)
                s_hyp_ob = snap_states(c_hyp_old_end)

                # Build TRUE input-dependent segment matrices Phi_m over [seam_end, T_ctx)
                if m_rem > 0 and T_ctx > seam_end:
                    c_phi_tmp = clone_gdn_cache(c_hyp_old_end)
                    with GDNSuffixHookCollector(n_linear=n_lin, d_k=d_state, scalar_only=False) as phi_col:
                        _, c_phi_tmp = run_chunked_prefill(model, base_model, ids_o[:, seam_end:T_ctx], seam_end, c_phi_tmp, 2048, use_l2k)
                    hs_phi = phi_col.finalize(expected_suffix_len=T_ctx - seam_end, device=str(device))
                    phi_chain_mats = _build_real_gdn_phi_chain(hs_phi, n_lin, n_heads, d_state, c_seg, m_rem, device)
                    del c_phi_tmp, hs_phi
                else:
                    phi_chain_mats = torch.empty((0, n_lin * n_heads, d_state, d_state), device=device, dtype=torch.bfloat16)

                # Oracle reference logits on edited prefix
                _, c_ora = run_chunked_prefill(model, base_model, ids_e[:, :T_ctx], 0, None, 2048, use_l2k)
                pos_q = make_gdn_pos(T_ctx, int(ids_q_chk.shape[1]), device, base_model)
                lg_ora = model(input_ids=ids_q_chk, position_ids=pos_q, past_key_values=c_ora, use_cache=True).logits[:, -1:, :].detach().float()
                del c_ora

                jl_times, hyp_times = [], []
                kl_jl_ora: Optional[float] = None
                kl_hyp_ora: Optional[float] = None
                top1_jl_eq: Optional[bool] = None
                top1_hyp_eq: Optional[bool] = None

                for rep in range(2 + n_repeats):
                    is_last = rep == (2 + n_repeats - 1)
                    # O_local online edit ONLY (pre-cached c_pre, G_cum)
                    c_target = clone_gdn_cache(c_ctx_old)
                    t0 = _sync(device)
                    c_work = clone_gdn_cache(c_pre)
                    _, c_nb = run_chunked_prefill(model, base_model, ids_e[:, p0:b_end], p0, c_work, 2048, use_l2k)
                    delta = gdn_delta_fp64(s_ob, snap_states(c_nb))
                    kv_sl = extract_kv_slice(c_nb, p0, b_end)
                    apply_jlocal_update_o1(c_target, delta, g_cum, b_end, T_ctx, w_conv)
                    patch_kv_slice(c_target, kv_sl, p0, b_end)
                    t_jl = (_sync(device) - t0) * 1000.0
                    if is_last:
                        lg_jl = model(input_ids=ids_q_chk, position_ids=pos_q, past_key_values=c_target, use_cache=True).logits[:, -1:, :].detach().float()
                        kl_jl_ora = kl_div_from_logits(lg_ora, lg_jl)
                        top1_jl_eq = bool(int(lg_ora.argmax(-1).item()) == int(lg_jl.argmax(-1).item()))
                    del c_work, c_nb, c_target

                    # Faithful input-driven HYPIC online edit ONLY (pre-cached c_s0, Phi_m dense segment chain)
                    c_htarget = clone_gdn_cache(c_ctx_old)
                    t0 = _sync(device)
                    c_hw = clone_gdn_cache(c_s0)
                    _, c_hnb = run_chunked_prefill(model, base_model, ids_e[:, s0:seam_end], s0, c_hw, 2048, use_l2k)
                    d_hyp = gdn_delta_fp64(s_hyp_ob, snap_states(c_hnb))
                    kv_h = extract_kv_slice(c_hnb, s0, seam_end)
                    if m_rem > 0 and phi_chain_mats.shape[0] > 0:
                        acc_op = phi_chain_mats[0]
                        for mi in range(1, phi_chain_mats.shape[0]):
                            acc_op = torch.bmm(phi_chain_mats[mi], acc_op)
                        lin_pos = 0
                        for li_idx in range(len(c_htarget.layers)):
                            dr = d_hyp["rec"][li_idx]
                            if dr is not None:
                                op_l = acc_op[lin_pos : lin_pos + n_heads].to(torch.float32)
                                lin_pos += n_heads
                                tr = torch.einsum("hij,hjv->hiv", op_l, dr[0].float()).unsqueeze(0)
                                tgt = c_htarget.layers[li_idx].recurrent_states[0]
                                tgt.copy_((tgt.float() + tr).to(tgt.dtype))
                    patch_kv_slice(c_htarget, kv_h, s0, seam_end)
                    t_hyp = (_sync(device) - t0) * 1000.0
                    if is_last:
                        lg_hyp = model(input_ids=ids_q_chk, position_ids=pos_q, past_key_values=c_htarget, use_cache=True).logits[:, -1:, :].detach().float()
                        kl_hyp_ora = kl_div_from_logits(lg_ora, lg_hyp)
                        top1_hyp_eq = bool(int(lg_ora.argmax(-1).item()) == int(lg_hyp.argmax(-1).item()))
                    del c_hw, c_hnb, c_htarget

                    if rep >= 2:
                        jl_times.append(t_jl)
                        hyp_times.append(t_hyp)

                del c_pre, c_old_b, c_ctx_old, c_s0, c_hyp_old_end, g_cum, phi_chain_mats
                hyp_diag_times = None
            else:
                # Mamba-2 (Falcon-H1 / Zamba2): capture REAL G_cum and separate diagonal HYPIC vs dense stress
                c_pre = DynamicCache(config=cfg)
                if p0 > 0:
                    _ = run_segment(model, c_pre, ids_o[:, :p0], 0, max_chunk=1024)
                c_old_b = clone_mamba_cache(c_pre)
                _ = run_segment(model, c_old_b, ids_o[:, p0:b_end], p0, max_chunk=1024)
                c_ctx_old, g_cum_arr = _capture_mamba2_g_cum(model, cfg, mixers, ids_o[:, :T_ctx], n_heads, max_chunk=1024)
                c_s0 = DynamicCache(config=cfg)
                if s0 > 0:
                    _ = run_segment(model, c_s0, ids_o[:, :s0], 0, max_chunk=1024)
                c_hyp_old_end = clone_mamba_cache(c_s0)
                _ = run_segment(model, c_hyp_old_end, ids_o[:, s0:seam_end], s0, max_chunk=1024)

                # Oracle logits
                c_ora = DynamicCache(config=cfg)
                _ = run_segment(model, c_ora, ids_e[:, :T_ctx], 0, max_chunk=1024)
                lg_ora = run_segment(model, c_ora, ids_q_chk, T_ctx, max_chunk=1024)[:, -1:, :].detach().float()
                del c_ora

                # Faithful Mamba-2 diagonal segment decays: exp(g_cum[t1] - g_cum[t0]) per segment
                diag_seg_decays: Dict[int, List[torch.Tensor]] = {li: [] for li, _ in mixers}
                for mi in range(m_rem):
                    t0_s = min(T_ctx, seam_end + mi * c_seg)
                    t1_s = min(T_ctx, t0_s + c_seg)
                    for li, _ in mixers:
                        diag_seg_decays[li].append(torch.exp(g_cum_arr[li][:, t1_s, :] - g_cum_arr[li][:, t0_s, :]))

                # Generic dense Phi stress probe matrix constructed from faithful diagonal factors (when fits VRAM)
                if run_dense_stress:
                    phi_chain_stress = torch.zeros((max(1, m_rem), n_lin * n_heads, d_state, d_state), device=device, dtype=torch.bfloat16)
                    for mi in range(m_rem):
                        for m_pos, (li_m, _) in enumerate(mixers):
                            sc_h = diag_seg_decays[li_m][mi][0].to(torch.bfloat16)  # (H,)
                            eye_n = torch.eye(d_state, device=device, dtype=torch.bfloat16)
                            phi_chain_stress[mi, m_pos * n_heads : (m_pos + 1) * n_heads] = sc_h[:, None, None] * eye_n
                else:
                    phi_chain_stress = None

                jl_times, hyp_times, hyp_diag_times = [], [], []
                kl_jl_ora = kl_hyp_ora = None
                top1_jl_eq = top1_hyp_eq = None

                for rep in range(2 + n_repeats):
                    is_last = rep == (2 + n_repeats - 1)
                    # O_local using real G_cum
                    c_target = clone_mamba_cache(c_ctx_old)
                    t0 = _sync(device)
                    c_work = clone_mamba_cache(c_pre)
                    _ = run_segment(model, c_work, ids_e[:, p0:b_end], p0, max_chunk=1024)
                    delta = mamba_delta_fp64(c_old_b, c_work)
                    for li_j, d_s in delta.items():
                        s_o = c_target.layers[li_j].recurrent_states[0]
                        g_diff = g_cum_arr[li_j][:, T_ctx, :] - g_cum_arr[li_j][:, b_end, :]
                        gam = torch.exp(g_diff)[:, :, None, None]
                        s_o.copy_((s_o.double() + gam * d_s).to(s_o.dtype))
                    patch_kv_range(c_target, c_work, p0, b_end)
                    t_jl = (_sync(device) - t0) * 1000.0
                    if is_last:
                        lg_jl = run_segment(model, c_target, ids_q_chk, T_ctx, max_chunk=1024)[:, -1:, :].detach().float()
                        kl_jl_ora = kl_div_from_logits(lg_ora, lg_jl)
                        top1_jl_eq = bool(int(lg_ora.argmax(-1).item()) == int(lg_jl.argmax(-1).item()))
                    del c_work, c_target

                    # Faithful Mamba-2 diagonal HYPIC composition (native scalar/diagonal complexity)
                    c_hd = clone_mamba_cache(c_ctx_old)
                    t0 = _sync(device)
                    c_hwd = clone_mamba_cache(c_s0)
                    _ = run_segment(model, c_hwd, ids_e[:, s0:seam_end], s0, max_chunk=1024)
                    d_hyp_d = mamba_delta_fp64(c_hyp_old_end, c_hwd)
                    for li_m, _ in mixers:
                        d_s = d_hyp_d[li_m]
                        acc_d = torch.ones((1, n_heads), device=device, dtype=torch.float64)
                        for mi in range(m_rem):
                            acc_d = acc_d * diag_seg_decays[li_m][mi]
                        tgt = c_hd.layers[li_m].recurrent_states[0]
                        tgt.copy_((tgt.double() + acc_d[:, :, None, None] * d_s).to(tgt.dtype))
                    patch_kv_range(c_hd, c_hwd, s0, seam_end)
                    t_hyp_diag = (_sync(device) - t0) * 1000.0
                    if is_last:
                        lg_hd = run_segment(model, c_hd, ids_q_chk, T_ctx, max_chunk=1024)[:, -1:, :].detach().float()
                        kl_hyp_ora = kl_div_from_logits(lg_ora, lg_hd)
                        top1_hyp_eq = bool(int(lg_ora.argmax(-1).item()) == int(lg_hd.argmax(-1).item()))
                    del c_hwd, c_hd

                    # Generic dense Phi stress arm (non-faithful GEMM chain proxy)
                    t_hyp = None
                    if phi_chain_stress is not None:
                        c_htarget = clone_mamba_cache(c_ctx_old)
                        t0 = _sync(device)
                        c_hw = clone_mamba_cache(c_s0)
                        _ = run_segment(model, c_hw, ids_e[:, s0:seam_end], s0, max_chunk=1024)
                        d_hyp = mamba_delta_fp64(c_hyp_old_end, c_hw)
                        if m_rem > 0:
                            acc_op = phi_chain_stress[0]
                            for mi in range(1, m_rem):
                                acc_op = torch.bmm(phi_chain_stress[mi], acc_op)
                            for m_pos, (li_m, _) in enumerate(mixers):
                                d_s = d_hyp[li_m]
                                op_m = acc_op[m_pos * n_heads : (m_pos + 1) * n_heads].to(torch.float32)
                                tr = torch.einsum("hvn,hnm->hvm", d_s[0].float(), op_m).unsqueeze(0)
                                tgt = c_htarget.layers[li_m].recurrent_states[0]
                                tgt.copy_((tgt.float() + tr).to(tgt.dtype))
                        patch_kv_range(c_htarget, c_hw, s0, seam_end)
                        t_hyp = (_sync(device) - t0) * 1000.0
                        del c_hw, c_htarget

                    if rep >= 2:
                        jl_times.append(t_jl)
                        hyp_diag_times.append(t_hyp_diag)
                        if t_hyp is not None:
                            hyp_times.append(t_hyp)

                del c_pre, c_old_b, c_ctx_old, c_s0, c_hyp_old_end, g_cum_arr, phi_chain_stress

        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower():
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()
                rows.append({"T_ctx": int(T_ctx), "timing_skipped_cuda_oom": True, "oom_error": str(exc)[:180]})
                continue
            raise

        # 3. Multi-prompt fidelity verification vs oracle (`n_fid = 5` per T_ctx, explicit per-case records, no silent error swallowing)
        n_fid_prompts = 5
        fid_olocal_kls: List[float] = []
        fid_hypic_kls: List[float] = []
        fid_olocal_t1: List[bool] = []
        fid_hypic_t1: List[bool] = []
        fid_per_case: List[Dict[str, Any]] = []
        fid_oom_error: Optional[str] = None
        try:
          with torch.no_grad():
            for fid_i in range(n_fid_prompts):
              torch.manual_seed(1100 + T_ctx + fid_i * 37)
              ids_o_f = torch.randint(100, 1500, (1, T_ctx), device=device, dtype=torch.long)
              ids_e_f = ids_o_f.clone()
              ids_e_f[:, p0 : p0 + 4] = torch.tensor([[111 + fid_i, 222 + fid_i, 333 + fid_i, 444 + fid_i]], device=device, dtype=torch.long)
              ids_q_f = torch.tensor([[120 + fid_i, 121 + fid_i, 122]], device=device, dtype=torch.long)
              if arch == "gdn":
                  c_pre_f = run_chunked_prefill(model, base_model, ids_o_f[:, :p0], 0, None, 2048, use_l2k)[1] if p0 > 0 else None
                  c_old_b_f = run_chunked_prefill(model, base_model, ids_o_f[:, p0:b_end], p0, clone_gdn_cache(c_pre_f), 2048, use_l2k)[1]
                  s_ob_f = snap_states(c_old_b_f)
                  c_ctx_f, g_cum_f = capture_prefix_cum_decay(model, base_model, ids_o_f[:, :T_ctx], n_lin, use_l2k)
                  c_s0_f = run_chunked_prefill(model, base_model, ids_o_f[:, :s0], 0, None, 2048, use_l2k)[1] if s0 > 0 else None
                  c_hob_f = run_chunked_prefill(model, base_model, ids_o_f[:, s0:seam_end], s0, clone_gdn_cache(c_s0_f), 2048, use_l2k)[1]
                  s_hob_f = snap_states(c_hob_f)
                  if m_rem > 0 and T_ctx > seam_end:
                      c_pt_f = clone_gdn_cache(c_hob_f)
                      with GDNSuffixHookCollector(n_linear=n_lin, d_k=d_state, scalar_only=False) as col_f:
                          _, c_pt_f = run_chunked_prefill(model, base_model, ids_o_f[:, seam_end:T_ctx], seam_end, c_pt_f, 2048, use_l2k)
                      hs_f = col_f.finalize(expected_suffix_len=T_ctx - seam_end, device=str(device))
                      phi_f = _build_real_gdn_phi_chain(hs_f, n_lin, n_heads, d_state, c_seg, m_rem, device)
                      del c_pt_f, hs_f
                  else:
                      phi_f = torch.empty((0, n_lin * n_heads, d_state, d_state), device=device, dtype=torch.bfloat16)
                  _, c_ora_f = run_chunked_prefill(model, base_model, ids_e_f[:, :T_ctx], 0, None, 2048, use_l2k)
                  pos_qf = make_gdn_pos(T_ctx, int(ids_q_f.shape[1]), device, base_model)
                  lg_ora_f = model(input_ids=ids_q_f, position_ids=pos_qf, past_key_values=c_ora_f, use_cache=True).logits[:, -1:, :].detach().float()
                  del c_ora_f

                  c_t_jl = clone_gdn_cache(c_ctx_f)
                  _, c_nb_f = run_chunked_prefill(model, base_model, ids_e_f[:, p0:b_end], p0, clone_gdn_cache(c_pre_f), 2048, use_l2k)
                  apply_jlocal_update_o1(c_t_jl, gdn_delta_fp64(s_ob_f, snap_states(c_nb_f)), g_cum_f, b_end, T_ctx, w_conv)
                  patch_kv_slice(c_t_jl, extract_kv_slice(c_nb_f, p0, b_end), p0, b_end)
                  lg_jl_f = model(input_ids=ids_q_f, position_ids=pos_qf, past_key_values=c_t_jl, use_cache=True).logits[:, -1:, :].detach().float()
                  ol_kl_v = float(kl_div_from_logits(lg_ora_f, lg_jl_f))
                  ol_t1_v = bool(int(lg_ora_f.argmax(-1).item()) == int(lg_jl_f.argmax(-1).item()))
                  fid_olocal_kls.append(ol_kl_v)
                  fid_olocal_t1.append(ol_t1_v)
                  del c_nb_f, c_t_jl

                  c_t_hy = clone_gdn_cache(c_ctx_f)
                  _, c_hnb_f = run_chunked_prefill(model, base_model, ids_e_f[:, s0:seam_end], s0, clone_gdn_cache(c_s0_f), 2048, use_l2k)
                  d_hf = gdn_delta_fp64(s_hob_f, snap_states(c_hnb_f))
                  if m_rem > 0 and phi_f.shape[0] > 0:
                      acc_f = phi_f[0]
                      for mi in range(1, phi_f.shape[0]):
                          acc_f = torch.bmm(phi_f[mi], acc_f)
                      lpos = 0
                      for li_i in range(len(c_t_hy.layers)):
                          dr_f = d_hf["rec"][li_i]
                          if dr_f is not None:
                              op_f = acc_f[lpos : lpos + n_heads].to(torch.float32)
                              lpos += n_heads
                              tr_f = torch.einsum("hij,hjv->hiv", op_f, dr_f[0].float()).unsqueeze(0)
                              tgt_f = c_t_hy.layers[li_i].recurrent_states[0]
                              tgt_f.copy_((tgt_f.float() + tr_f).to(tgt_f.dtype))
                  patch_kv_slice(c_t_hy, extract_kv_slice(c_hnb_f, s0, seam_end), s0, seam_end)
                  lg_hy_f = model(input_ids=ids_q_f, position_ids=pos_qf, past_key_values=c_t_hy, use_cache=True).logits[:, -1:, :].detach().float()
                  hy_kl_v = float(kl_div_from_logits(lg_ora_f, lg_hy_f))
                  hy_t1_v = bool(int(lg_ora_f.argmax(-1).item()) == int(lg_hy_f.argmax(-1).item()))
                  fid_hypic_kls.append(hy_kl_v)
                  fid_hypic_t1.append(hy_t1_v)
                  fid_per_case.append({
                      "case_idx": int(fid_i),
                      "olocal_kl": round(ol_kl_v, 8),
                      "hypic_faithful_kl": round(hy_kl_v, 8),
                      "olocal_top1_eq_oracle": ol_t1_v,
                      "hypic_faithful_top1_eq_oracle": hy_t1_v,
                  })
                  del c_hnb_f, c_t_hy, c_pre_f, c_old_b_f, c_ctx_f, c_s0_f, c_hob_f, g_cum_f, phi_f
              else:
                  c_pre_f = DynamicCache(config=cfg)
                  if p0 > 0:
                      _ = run_segment(model, c_pre_f, ids_o_f[:, :p0], 0, max_chunk=1024)
                  c_ob_f = clone_mamba_cache(c_pre_f)
                  _ = run_segment(model, c_ob_f, ids_o_f[:, p0:b_end], p0, max_chunk=1024)
                  c_ctx_f, gc_f = _capture_mamba2_g_cum(model, cfg, mixers, ids_o_f[:, :T_ctx], n_heads, max_chunk=1024)
                  c_s0_f = DynamicCache(config=cfg)
                  if s0 > 0:
                      _ = run_segment(model, c_s0_f, ids_o_f[:, :s0], 0, max_chunk=1024)
                  c_hoe_f = clone_mamba_cache(c_s0_f)
                  _ = run_segment(model, c_hoe_f, ids_o_f[:, s0:seam_end], s0, max_chunk=1024)
                  c_ora_f = DynamicCache(config=cfg)
                  _ = run_segment(model, c_ora_f, ids_e_f[:, :T_ctx], 0, max_chunk=1024)
                  lg_ora_f = run_segment(model, c_ora_f, ids_q_f, T_ctx, max_chunk=1024)[:, -1:, :].detach().float()
                  del c_ora_f

                  c_t_jl = clone_mamba_cache(c_ctx_f)
                  c_wk_f = clone_mamba_cache(c_pre_f)
                  _ = run_segment(model, c_wk_f, ids_e_f[:, p0:b_end], p0, max_chunk=1024)
                  for li_j, d_s in mamba_delta_fp64(c_ob_f, c_wk_f).items():
                      s_o = c_t_jl.layers[li_j].recurrent_states[0]
                      gam = torch.exp(gc_f[li_j][:, T_ctx, :] - gc_f[li_j][:, b_end, :])[:, :, None, None]
                      s_o.copy_((s_o.double() + gam * d_s).to(s_o.dtype))
                  patch_kv_range(c_t_jl, c_wk_f, p0, b_end)
                  lg_jl_f = run_segment(model, c_t_jl, ids_q_f, T_ctx, max_chunk=1024)[:, -1:, :].detach().float()
                  ol_kl_v = float(kl_div_from_logits(lg_ora_f, lg_jl_f))
                  ol_t1_v = bool(int(lg_ora_f.argmax(-1).item()) == int(lg_jl_f.argmax(-1).item()))
                  fid_olocal_kls.append(ol_kl_v)
                  fid_olocal_t1.append(ol_t1_v)
                  del c_wk_f, c_t_jl

                  c_hd_f = clone_mamba_cache(c_ctx_f)
                  c_hwd_f = clone_mamba_cache(c_s0_f)
                  _ = run_segment(model, c_hwd_f, ids_e_f[:, s0:seam_end], s0, max_chunk=1024)
                  for li_m, d_s in mamba_delta_fp64(c_hoe_f, c_hwd_f).items():
                      acc_d = torch.ones((1, n_heads), device=device, dtype=torch.float64)
                      for mi in range(m_rem):
                          t0_s = min(T_ctx, seam_end + mi * c_seg)
                          t1_s = min(T_ctx, t0_s + c_seg)
                          acc_d = acc_d * torch.exp(gc_f[li_m][:, t1_s, :] - gc_f[li_m][:, t0_s, :])
                      tgt_m = c_hd_f.layers[li_m].recurrent_states[0]
                      tgt_m.copy_((tgt_m.double() + acc_d[:, :, None, None] * d_s).to(tgt_m.dtype))
                  patch_kv_range(c_hd_f, c_hwd_f, s0, seam_end)
                  lg_hy_f = run_segment(model, c_hd_f, ids_q_f, T_ctx, max_chunk=1024)[:, -1:, :].detach().float()
                  hy_kl_v = float(kl_div_from_logits(lg_ora_f, lg_hy_f))
                  hy_t1_v = bool(int(lg_ora_f.argmax(-1).item()) == int(lg_hy_f.argmax(-1).item()))
                  fid_hypic_kls.append(hy_kl_v)
                  fid_hypic_t1.append(hy_t1_v)
                  fid_per_case.append({
                      "case_idx": int(fid_i),
                      "olocal_kl": round(ol_kl_v, 8),
                      "hypic_faithful_kl": round(hy_kl_v, 8),
                      "olocal_top1_eq_oracle": ol_t1_v,
                      "hypic_faithful_top1_eq_oracle": hy_t1_v,
                  })
                  del c_hwd_f, c_hd_f, c_pre_f, c_ob_f, c_ctx_f, c_s0_f, c_hoe_f, gc_f
        except torch.cuda.OutOfMemoryError as exc_oom:
            fid_oom_error = str(exc_oom)[:180]
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()

        m_jl = statistics.median(jl_times)
        m_hyp = statistics.median(hyp_times) if hyp_times else None
        m_hyp_diag = statistics.median(hyp_diag_times) if hyp_diag_times else None
        faithful_op_ms = m_hyp if arch == "gdn" else m_hyp_diag
        nf_ol = len(fid_olocal_kls)
        nf_hy = len(fid_hypic_kls)
        ol_t1_hits = sum(1 for x in fid_olocal_t1 if x)
        hy_t1_hits = sum(1 for x in fid_hypic_t1 if x)
        row: Dict[str, Any] = {
            "T_ctx": int(T_ctx),
            "n_segs_128": int(n_segs),
            "remaining_segs": int(m_rem),
            "jlocal_online_sym_ms": round(m_jl, 3),
            "hypic_phi_chain_sym_ms": round(m_hyp, 3) if m_hyp is not None else None,
            "hypic_mamba2_diagonal_sym_ms": round(m_hyp_diag, 3) if m_hyp_diag is not None else None,
            "faithful_hypic_over_olocal_speedup": round(faithful_op_ms / max(1e-6, m_jl), 3) if faithful_op_ms is not None else None,
            "dense_phi_proxy_over_olocal_speedup": round(m_hyp / max(1e-6, m_jl), 3) if (arch != "gdn" and m_hyp is not None) else None,
            "symmetric_speedup_hypic_over_jlocal": round(m_hyp / max(1e-6, m_jl), 3) if m_hyp is not None else None,
            "jlocal_raw_ms": [round(float(x), 3) for x in jl_times],
            "hypic_phi_raw_ms": [round(float(x), 3) for x in hyp_times] if hyp_times else None,
            "hypic_diag_raw_ms": [round(float(x), 3) for x in hyp_diag_times] if hyp_diag_times else None,
            "fidelity_scope": f"n_ol={nf_ol}_n_hy={nf_hy}_synthetic_prompts_last_token",
            "fidelity_vs_oracle": {
                "n_prompts": int(min(nf_ol, nf_hy)),
                "olocal_n_done": int(nf_ol),
                "hypic_faithful_n_done": int(nf_hy),
                "fidelity_oom_error": fid_oom_error,
                "olocal_kl": round(sum(fid_olocal_kls) / max(1, nf_ol), 8) if nf_ol else None,
                "olocal_max_kl": round(max(fid_olocal_kls), 8) if nf_ol else None,
                "hypic_faithful_kl": round(sum(fid_hypic_kls) / max(1, nf_hy), 8) if nf_hy else None,
                "hypic_faithful_max_kl": round(max(fid_hypic_kls), 8) if nf_hy else None,
                "olocal_top1_eq_count": int(ol_t1_hits),
                "hypic_faithful_top1_eq_count": int(hy_t1_hits),
                "olocal_top1_eq_rate": round(ol_t1_hits / max(1, nf_ol), 4) if nf_ol else None,
                "hypic_faithful_top1_eq_rate": round(hy_t1_hits / max(1, nf_hy), 4) if nf_hy else None,
                "olocal_top1_eq_oracle": bool(nf_ol > 0 and ol_t1_hits == nf_ol),
                "hypic_faithful_top1_eq_oracle": bool(nf_hy > 0 and hy_t1_hits == nf_hy),
                "per_case": fid_per_case,
            },
        }
        rows.append(row)
    return {"timing_rows": rows, "vram_probe": vram_probe}


def _eval_lora_case(
    model: Any,
    base_model: Any,
    tok: Any,
    cfg: Any,
    arch: str,
    case: Dict[str, Any],
    adapters: Any,
    device: torch.device,
    use_l2k: bool,
) -> Dict[str, Any]:
    """Evaluate current LoRA-attached model on a single edit case (train or held-out)."""
    p0, p1, T_ctx = case["p0"], case["p1"], case["T_ctx"]
    ids_o = case["ids_o"] if arch == "gdn" else torch.tensor([case["ids_o"]], device=device, dtype=torch.long)
    ids_e = case["ids_e"] if arch == "gdn" else torch.tensor([case["ids_n"]], device=device, dtype=torch.long)
    ids_suf = case["ids_q"] if arch == "gdn" else ids_e[:, T_ctx:]
    nv_str = case["new_val"] if arch == "gdn" else case["new"]
    if arch == "gdn":
        ans_ids = tok.encode(nv_str, add_special_tokens=False) or [0]
        cand_1st = [int(ans_ids[0])]
    else:
        ans_ids, cand_1st = resolve_answer_tokens(tok, nv_str)
        if not ans_ids:
            ans_ids = tok.encode(nv_str, add_special_tokens=False) or [0]
        if not cand_1st:
            cand_1st = [int(ans_ids[0])]
    ans_t = torch.tensor([ans_ids], device=device, dtype=torch.long)
    ids_suf_tf = torch.cat([ids_suf, ans_t[:, :-1]], dim=1) if ans_t.shape[1] > 1 else ids_suf

    with torch.no_grad():
        set_lora_enabled(adapters, False)
        if arch == "gdn":
            _, c_pre = run_chunked_prefill(model, base_model, ids_o[:, :p0], 0, None, 1024, use_l2k)
            c_old_ctx = clone_gdn_cache(c_pre)
            _, c_old_ctx = run_chunked_prefill(model, base_model, ids_o[:, p0:T_ctx], p0, c_old_ctx, 1024, use_l2k)
            _, c_ora = run_chunked_prefill(model, base_model, ids_e[:, :T_ctx], 0, None, 1024, use_l2k)
            pos_sf = make_gdn_pos(T_ctx, int(ids_suf_tf.shape[1]), device, base_model)
            ora_lg = model(input_ids=ids_suf_tf, position_ids=pos_sf, past_key_values=c_ora, use_cache=True).logits[:, -ans_t.shape[1] :, :]
            set_lora_enabled(adapters, True)
            c_l = clone_gdn_cache(c_pre)
            pos_sp = make_gdn_pos(p0, p1 - p0, device, base_model)
            _ = base_model(input_ids=ids_e[:, p0:p1], position_ids=pos_sp, past_key_values=c_l, use_cache=True)
            c_surr = build_gdn_kveraser_surrogate(c_old_ctx, c_l, p0, p1, T_ctx)
            set_lora_enabled(adapters, False)
            cand_lg = model(input_ids=ids_suf_tf, position_ids=pos_sf, past_key_values=c_surr, use_cache=True).logits[:, -ans_t.shape[1] :, :]
            del c_pre, c_old_ctx, c_ora, c_l, c_surr
        else:
            c_pre = DynamicCache(config=cfg)
            if p0 > 0:
                _ = run_segment(model, c_pre, ids_o[:, :p0], 0, max_chunk=1024)
            c_old_ctx = clone_mamba_cache(c_pre)
            _ = run_segment(model, c_old_ctx, ids_o[:, p0:T_ctx], p0, max_chunk=1024)
            c_ora = DynamicCache(config=cfg)
            _ = run_segment(model, c_ora, ids_e[:, :T_ctx], 0, max_chunk=1024)
            ora_lg = run_segment(model, c_ora, ids_suf_tf, T_ctx, max_chunk=1024)[:, -ans_t.shape[1] :, :]
            set_lora_enabled(adapters, True)
            c_l = clone_mamba_cache(c_pre)
            _ = model(input_ids=ids_e[:, p0:p1], position_ids=make_mamba_pos(p0, p1 - p0, device), past_key_values=c_l, use_cache=True)
            c_surr = build_kveraser_surrogate_cache(c_old_ctx, c_l, p0, p1, T_ctx, retain_grad=False)
            set_lora_enabled(adapters, False)
            cand_lg = model(input_ids=ids_suf_tf, position_ids=make_mamba_pos(T_ctx, int(ids_suf_tf.shape[1]), device), past_key_values=c_surr, use_cache=True).logits[:, -ans_t.shape[1] :, :]
            del c_pre, c_old_ctx, c_ora, c_l, c_surr

        ce = float(F.cross_entropy(cand_lg.reshape(-1, cand_lg.shape[-1]).float(), ans_t.reshape(-1)).item())
        kl = kl_div_from_logits(ora_lg[:, :1, :].float(), cand_lg[:, :1, :].float())
        pred_first = int(cand_lg[:, 0, :].argmax(dim=-1).item())
        ora_first = int(ora_lg[:, 0, :].argmax(dim=-1).item())
        return {
            "loss_ce": round(ce, 4),
            "kl_first_vs_oracle": round(float(kl), 6),
            "top1_hit": bool(pred_first in cand_1st),
            "oracle_top1_hit": bool(ora_first in cand_1st),
            "top1_eq_oracle": bool(pred_first == ora_first),
        }


def _case_identity(case: Dict[str, Any], arch: str) -> Tuple[str, str, Tuple[int, ...]]:
    if arch == "gdn":
        ids_list = tuple(int(x) for x in case["ids_o"][0].tolist())
        return str(case["old_val"]), str(case["new_val"]), ids_list
    ids_list = tuple(int(x) for x in case["ids_o"])
    return str(case["old"]), str(case["new"]), ids_list


def _resolve_cf_path_optb() -> Optional[Path]:
    for cand in (
        SCRIPT_DIR / "w11_cf_subset_for_colab.json",
        Path("/content/pilot/w11_cf_subset_for_colab.json"),
        Path(r"D:\applicationData\HuggingFace\counterfact.json"),
        Path("/content/counterfact.json"),
    ):
        if cand.is_file():
            return cand
    return None


def measure_t5_single_edit_vs_batch(
    model: Any,
    tok: Any,
    cfg: Any,
    arch: str,
    geom: Dict[str, int],
    device: torch.device,
    n_batch_cases: int = 8,
    single_edit_steps: int = 10,
) -> Dict[str, Any]:
    """Select an unconfounded oracle-passing pre-fail single-edit case & 5 strictly disjoint oracle-passing held-out cases."""
    base_model = getattr(getattr(model, "model", model), "language_model", getattr(model, "model", model))
    use_l2k = has_logits_to_keep_arg(model) if arch == "gdn" else False
    mamba2_fillers_used: Optional[Tuple[int, int]] = None

    if arch == "gdn":
        n_req = 2 if n_batch_cases <= 2 else max(12, n_batch_cases * 3)
        cf_p = None if n_batch_cases <= 2 else _resolve_cf_path_optb()
        tr_all, te_all = build_gdn_splits(tok, cf_p, geom["n_linear"], n_req, n_req, 512, str(device))
        cases_tr = [c for c in tr_all if c["regime"] in ("counterfact", "collision", "sniah")]
        cases_te = [c for c in te_all if c["regime"] in ("counterfact", "collision", "sniah")]
    else:
        sw = collect_shared_single_tokens([tok])
        mamba2_fillers_used = (12, 10)
        pools = build_case_pools(tok, sw, n_e2=4, n_sb1_train=20, n_sb1_test=20, pre_fillers=12, post_fillers=10)
        cases_tr = pools["sb1_train"]
        cases_te = pools["sb1_test"]

    adapters, lora_p, tot_p = attach_kveraser_lora(model, rank=16, alpha=32.0)
    trainable = [p for ad in adapters for p in (ad.lora_A, ad.lora_B)]
    opt = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)

    if arch == "gdn":
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
        qm.torch_recurrent_gated_delta_rule = getattr(qm.torch_recurrent_gated_delta_rule, "__wrapped__", qm.torch_recurrent_gated_delta_rule)
        qm.torch_chunk_gated_delta_rule = getattr(qm.torch_chunk_gated_delta_rule, "__wrapped__", qm.torch_chunk_gated_delta_rule)

    # Scan candidate training pool (for Mamba-2, fallback to shorter 4+4 fillers if <6 oracle-passing cases exist)
    def _scan_pool(pool_list: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
        return [(idx_c, _eval_lora_case(model, base_model, tok, cfg, arch, cand, adapters, device, use_l2k)) for idx_c, cand in enumerate(pool_list)]

    tr_evals = _scan_pool(cases_tr)
    te_evals = _scan_pool(cases_te)
    if arch != "gdn":
        n_ora_ok = sum(1 for _, e in tr_evals if e["oracle_top1_hit"]) + sum(1 for _, e in te_evals if e["oracle_top1_hit"])
        if n_ora_ok < 8:
            mamba2_fillers_used = (4, 4)
            pools_s = build_case_pools(tok, sw, n_e2=4, n_sb1_train=24, n_sb1_test=24, pre_fillers=4, post_fillers=4)
            cases_tr = pools_s["sb1_train"]
            cases_te = pools_s["sb1_test"]
            tr_evals = _scan_pool(cases_tr)
            te_evals = _scan_pool(cases_te)

    # Priority 1: oracle_top1_hit == True AND pre-edit fails (`not top1_hit`)
    # Priority 2: oracle_top1_hit == True with highest loss_ce
    # Priority 3: highest loss_ce overall
    p1_cands = [(i, e) for i, e in tr_evals if e["oracle_top1_hit"] and (not e["top1_hit"])]
    p2_cands = [(i, e) for i, e in tr_evals if e["oracle_top1_hit"]]
    if p1_cands:
        selected_idx, _ = max(p1_cands, key=lambda x: x[1]["loss_ce"])
        sel_criterion = "oracle_pass_and_pre_fail"
    elif p2_cands:
        selected_idx, _ = max(p2_cands, key=lambda x: x[1]["loss_ce"])
        sel_criterion = "oracle_pass_highest_ce"
    else:
        selected_idx, _ = max(tr_evals, key=lambda x: x[1]["loss_ce"])
        sel_criterion = "fallback_highest_ce"

    c0 = cases_tr[selected_idx]
    tr_old, tr_new, tr_ids = _case_identity(c0, arch)
    tr_regime = c0.get("regime", arch)

    # Select up to 5 strictly disjoint held-out cases preferring oracle_top1_hit == True and same regime first
    te_ranked = sorted(
        [(cases_te[i], e) for i, e in te_evals],
        key=lambda x: (
            1 if x[1]["oracle_top1_hit"] else 0,
            1 if x[0].get("regime", arch) == tr_regime else 0,
            -x[1]["loss_ce"],
        ),
        reverse=True,
    )
    ho_candidates: List[Dict[str, Any]] = []
    seen_ho_keys: set[Tuple[str, str, Tuple[int, ...]]] = set()
    for cand_ho, _ev_ho in te_ranked:
        cid_key = _case_identity(cand_ho, arch)
        if cid_key not in seen_ho_keys and cid_key[2] != tr_ids and (cid_key[0], cid_key[1]) != (tr_old, tr_new):
            seen_ho_keys.add(cid_key)
            ho_candidates.append(cand_ho)
        if len(ho_candidates) >= 5:
            break
    assert len(ho_candidates) >= 1, f"Failed to find strictly disjoint held-out case for {arch} train={tr_old}->{tr_new}"
    c_ho = ho_candidates[0]
    ho_old, ho_new, ho_ids = _case_identity(c_ho, arch)
    assert ho_ids != tr_ids and (ho_old, ho_new) != (tr_old, tr_new), "Held-out leakage assertion failed"

    train_pre = _eval_lora_case(model, base_model, tok, cfg, arch, c0, adapters, device, use_l2k)
    heldout_pre_list = [_eval_lora_case(model, base_model, tok, cfg, arch, h_c, adapters, device, use_l2k) for h_c in ho_candidates]
    heldout_pre = heldout_pre_list[0]

    p0, p1, T_ctx = c0["p0"], c0["p1"], c0["T_ctx"]
    ids_o = c0["ids_o"] if arch == "gdn" else torch.tensor([c0["ids_o"]], device=device, dtype=torch.long)
    ids_e = c0["ids_e"] if arch == "gdn" else torch.tensor([c0["ids_n"]], device=device, dtype=torch.long)
    ids_suf = c0["ids_q"] if arch == "gdn" else ids_e[:, T_ctx:]
    nv_str = c0["new_val"] if arch == "gdn" else c0["new"]
    if arch == "gdn":
        ans_ids = tok.encode(nv_str, add_special_tokens=False) or [0]
    else:
        ans_ids, _ = resolve_answer_tokens(tok, nv_str)
        if not ans_ids:
            ans_ids = tok.encode(nv_str, add_special_tokens=False) or [0]
    ans_t = torch.tensor([ans_ids], device=device, dtype=torch.long)
    ids_suf_tf = torch.cat([ids_suf, ans_t[:, :-1]], dim=1) if ans_t.shape[1] > 1 else ids_suf

    with torch.no_grad():
        set_lora_enabled(adapters, False)
        if arch == "gdn":
            _, c_pre = run_chunked_prefill(model, base_model, ids_o[:, :p0], 0, None, 1024, use_l2k)
            c_old_ctx = clone_gdn_cache(c_pre)
            _, c_old_ctx = run_chunked_prefill(model, base_model, ids_o[:, p0:T_ctx], p0, c_old_ctx, 1024, use_l2k)
            _, c_ora = run_chunked_prefill(model, base_model, ids_e[:, :T_ctx], 0, None, 1024, use_l2k)
            pos_ora_sf = make_gdn_pos(T_ctx, int(ids_suf_tf.shape[1]), device, base_model)
            ora_suf_lg = model(input_ids=ids_suf_tf, position_ids=pos_ora_sf, past_key_values=c_ora, use_cache=True).logits
            ora_ans_lg = ora_suf_lg[:, -ans_t.shape[1] :, :].detach()
            del c_ora, ora_suf_lg
        else:
            c_pre = DynamicCache(config=cfg)
            if p0 > 0:
                _ = run_segment(model, c_pre, ids_o[:, :p0], 0, max_chunk=1024)
            c_old_ctx = clone_mamba_cache(c_pre)
            _ = run_segment(model, c_old_ctx, ids_o[:, p0:T_ctx], p0, max_chunk=1024)
            c_ora = DynamicCache(config=cfg)
            _ = run_segment(model, c_ora, ids_e[:, :T_ctx], 0, max_chunk=1024)
            ora_suf_lg = run_segment(model, c_ora, ids_suf_tf, T_ctx, max_chunk=1024)
            ora_ans_lg = ora_suf_lg[:, -ans_t.shape[1] :, :].detach()
            del c_ora

    step_times: List[float] = []
    losses_1edit: List[float] = []
    t_all0 = _sync(device)
    for st in range(single_edit_steps):
        t_s0 = _sync(device)
        opt.zero_grad(set_to_none=True)
        with torch.autograd.graph.saved_tensors_hooks(
            lambda t: t.clone() if (torch.is_tensor(t) and t.ndim >= 3) else t,
            lambda t: t,
        ):
            set_lora_enabled(adapters, True)
            if arch == "gdn":
                c_l = clone_gdn_cache(c_pre)
                pos_sp = make_gdn_pos(p0, p1 - p0, device, base_model)
                _ = base_model(input_ids=ids_e[:, p0:p1], position_ids=pos_sp, past_key_values=c_l, use_cache=True)
                c_surr = build_gdn_kveraser_surrogate(c_old_ctx, c_l, p0, p1, T_ctx)
                set_lora_enabled(adapters, False)
                pos_sf = make_gdn_pos(T_ctx, int(ids_suf_tf.shape[1]), device, base_model)
                cand_out = model(input_ids=ids_suf_tf, position_ids=pos_sf, past_key_values=c_surr, use_cache=True).logits
            else:
                c_l = clone_mamba_cache(c_pre)
                pos_sp = make_mamba_pos(p0, p1 - p0, device)
                _ = model(input_ids=ids_e[:, p0:p1], position_ids=pos_sp, past_key_values=c_l, use_cache=True)
                c_surr = build_kveraser_surrogate_cache(c_old_ctx, c_l, p0, p1, T_ctx, retain_grad=True)
                set_lora_enabled(adapters, False)
                pos_sf = make_mamba_pos(T_ctx, int(ids_suf_tf.shape[1]), device)
                cand_out = model(input_ids=ids_suf_tf, position_ids=pos_sf, past_key_values=c_surr, use_cache=True).logits
            cand_ans = cand_out[:, -ans_t.shape[1] :, :]
            ce = F.cross_entropy(cand_ans.reshape(-1, cand_ans.shape[-1]).float(), ans_t.reshape(-1))
            lq = F.log_softmax(cand_ans.float(), dim=-1)
            lp = F.log_softmax(ora_ans_lg.float(), dim=-1)
            kl_l = (lp.exp() * (lp - lq)).sum(dim=-1).mean()
            loss = ce + 0.5 * kl_l
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step_times.append(_sync(device) - t_s0)
        losses_1edit.append(float(loss.item()))
        del c_l, c_surr
    single_k_total_s = _sync(device) - t_all0

    train_post = _eval_lora_case(model, base_model, tok, cfg, arch, c0, adapters, device, use_l2k)
    heldout_post_list = [_eval_lora_case(model, base_model, tok, cfg, arch, h_c, adapters, device, use_l2k) for h_c in ho_candidates]
    heldout_post = heldout_post_list[0]

    # Measure inference-only LoRA surrogate application time
    infer_times: List[float] = []
    with torch.no_grad():
        for rep in range(7):
            t_i0 = _sync(device)
            set_lora_enabled(adapters, True)
            if arch == "gdn":
                c_l = clone_gdn_cache(c_pre)
                _ = base_model(input_ids=ids_e[:, p0:p1], position_ids=make_gdn_pos(p0, p1 - p0, device, base_model), past_key_values=c_l, use_cache=True)
                _ = build_gdn_kveraser_surrogate(c_old_ctx, c_l, p0, p1, T_ctx)
            else:
                c_l = clone_mamba_cache(c_pre)
                _ = model(input_ids=ids_e[:, p0:p1], position_ids=make_mamba_pos(p0, p1 - p0, device), past_key_values=c_l, use_cache=True)
                _ = build_kveraser_surrogate_cache(c_old_ctx, c_l, p0, p1, T_ctx, retain_grad=False)
            set_lora_enabled(adapters, False)
            dt_ms = (_sync(device) - t_i0) * 1000.0
            if rep >= 2:
                infer_times.append(dt_ms)
            del c_l

    del c_pre, c_old_ctx, ora_ans_lg
    detach_kveraser_lora(model)
    return {
        "lora_params": int(lora_p),
        "total_params": int(tot_p),
        "selected_case_idx": int(selected_idx),
        "selected_case_regime": str(tr_regime),
        "selection_criterion_met": sel_criterion,
        "mamba2_prompt_fillers": list(mamba2_fillers_used) if mamba2_fillers_used is not None else None,
        "train_pool_scanned": len(tr_evals),
        "train_pool_oracle_passing_count": len(p2_cands),
        "train_pool_oracle_pass_and_pre_fail_count": len(p1_cands),
        "heldout_pool_scanned": len(te_evals),
        "heldout_pool_oracle_passing_count": sum(1 for _, e in te_evals if e["oracle_top1_hit"]),
        "train_case_id": str(c0.get("case_id", f"{arch}_tr_{selected_idx}")),
        "heldout_case_id": str(c_ho.get("case_id", f"{arch}_te")),
        "train_pair": f"{tr_old}->{tr_new}",
        "heldout_pair": f"{ho_old}->{ho_new}",
        "train_heldout_disjoint_verified": True,
        "single_edit_steps_measured": int(single_edit_steps),
        "single_edit_1step_s": round(step_times[0], 4),
        "single_edit_warm_median_step_s": round(statistics.median(step_times[1:]) if len(step_times) > 1 else step_times[0], 4),
        "single_edit_k_steps_total_s": round(single_k_total_s, 4),
        "single_edit_loss_start": round(losses_1edit[0], 4),
        "single_edit_loss_end": round(losses_1edit[-1], 4),
        "train_pre_eval": train_pre,
        "train_post_eval": train_post,
        "heldout_pre_eval": heldout_pre,
        "heldout_post_eval": heldout_post,
        "heldout_n": len(ho_candidates),
        "heldout_multi_eval": {
            "n_heldout": len(ho_candidates),
            "pre_top1_hits": sum(1 for x in heldout_pre_list if x["top1_hit"]),
            "post_top1_hits": sum(1 for x in heldout_post_list if x["top1_hit"]),
            "oracle_top1_hits": sum(1 for x in heldout_pre_list if x["oracle_top1_hit"]),
            "pre_top1_eq_oracle": sum(1 for x in heldout_pre_list if x["top1_eq_oracle"]),
            "post_top1_eq_oracle": sum(1 for x in heldout_post_list if x["top1_eq_oracle"]),
            "pre_mean_kl_vs_oracle": round(sum(x["kl_first_vs_oracle"] for x in heldout_pre_list) / len(heldout_pre_list), 6),
            "post_mean_kl_vs_oracle": round(sum(x["kl_first_vs_oracle"] for x in heldout_post_list) / len(heldout_post_list), 6),
            "pre_mean_ce": round(sum(x["loss_ce"] for x in heldout_pre_list) / len(heldout_pre_list), 4),
            "post_mean_ce": round(sum(x["loss_ce"] for x in heldout_post_list) / len(heldout_post_list), 4),
            "per_case": [
                {
                    "case_id": str(h_c.get("case_id", f"{arch}_te_{hi}")),
                    "pair": f"{_case_identity(h_c, arch)[0]}->{_case_identity(h_c, arch)[1]}",
                    "pre": heldout_pre_list[hi],
                    "post": heldout_post_list[hi],
                }
                for hi, h_c in enumerate(ho_candidates)
            ],
        },
        "surrogate_infer_only_ms": round(statistics.median(infer_times), 3),
    }


def run_synthetic_dry_run() -> None:
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
    geom = _model_geom(model, "gdn", cfg)
    h_res = measure_hypic_vs_olocal_symmetric(model, "gdn", cfg, geom, torch.device("cpu"), [256, 512], n_repeats=1)
    t_res = measure_t5_single_edit_vs_batch(model, tok, cfg, "gdn", geom, torch.device("cpu"), n_batch_cases=2, single_edit_steps=2)
    assert len(h_res["timing_rows"]) == 2 and t_res["single_edit_steps_measured"] == 2
    assert h_res["timing_rows"][0]["fidelity_vs_oracle"]["hypic_faithful_kl"] is not None
    assert "train_post_eval" in t_res and "heldout_post_eval" in t_res
    print("[DRY-RUN PASS] w14_optb_symmetric_baselines.py true input Phi_m, fidelity checks, & T5 train+heldout eval OK.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="qwen2b,falcon-h1,zamba2")
    parser.add_argument("--tag", type=str, default="local_4060")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default=str(SCRIPT_DIR))
    parser.add_argument("--t-ctxs", type=str, default="512,1024,2048,4096,8192")
    parser.add_argument("--single-edit-steps", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--synthetic-dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.synthetic_dry_run:
        run_synthetic_dry_run()
        return

    model_specs = {
        "qwen2b": ("Qwen/Qwen3.5-2B", "gdn"),
        "qwen9b": ("Qwen/Qwen3.5-9B", "gdn"),
        "falcon-h1": ("tiiuae/Falcon-H1-1.5B-Instruct", "falcon_h1"),
        "zamba2": ("Zyphra/Zamba2-1.2B-Instruct-v2", "zamba2"),
    }
    t_ctxs = [int(x) for x in args.t_ctxs.split(",") if x.strip()]
    if args.smoke:
        t_ctxs = [512, 1024]

    dev = torch.device(args.device)
    out_dir = Path(args.out_dir)
    all_models: Dict[str, Any] = {}

    for mkey in [m.strip() for m in args.models.split(",") if m.strip()]:
        repo_id, arch = model_specs[mkey]
        print(f"\n[OptB-2] Loading {mkey} ({repo_id}, arch={arch})...", flush=True)
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, local_files_only=args.local_files_only)
        cfg = AutoConfig.from_pretrained(repo_id, trust_remote_code=True, local_files_only=args.local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16, device_map=args.device,
            trust_remote_code=True, local_files_only=args.local_files_only,
        ).eval()
        geom = _model_geom(model, arch, cfg)
        hyp = measure_hypic_vs_olocal_symmetric(model, arch, cfg, geom, dev, t_ctxs, n_repeats=(2 if args.smoke else 8))
        t5 = measure_t5_single_edit_vs_batch(model, tok, cfg, arch, geom, dev, n_batch_cases=4, single_edit_steps=(2 if args.smoke else args.single_edit_steps))
        all_models[mkey] = {"repo_id": repo_id, "arch": arch, "geom": geom, "symmetric_hypic_vs_olocal": hyp, "t5_single_edit": t5}
        print(f"  {mkey} timing_rows:", json.dumps(hyp["timing_rows"]), flush=True)
        print(f"  {mkey} t5_single_edit:", json.dumps(t5), flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    import transformers as _tf

    out_p = out_dir / f"w14_optb_symmetric_baselines_{args.tag}_summary.json"
    out_p.write_text(
        json.dumps(
            {
                "runner": "pilot/w14_optb_symmetric_baselines.py",
                "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                "os_name": os.name,
                "cuda_platform": "windows_wddm" if os.name == "nt" else "linux_dedicated_vram",
                "torch_version": torch.__version__,
                "transformers_version": _tf.__version__,
                "models": all_models,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[DONE] Wrote {out_p}", flush=True)


if __name__ == "__main__":
    main()
