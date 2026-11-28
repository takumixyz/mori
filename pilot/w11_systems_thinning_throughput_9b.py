"""Stage-B Wave 11-4: Systems Benchmarks (SB7 Batched Throughput, SB8 Host-RAM Thinning/Quantization/mmap)
and Complete 9B/27B Scans (SB9: 9B E2@4K n=100, E3 multi-fact perms, E8 T* curve, 27B-4bit thinning).

Supports:
  --model Qwen/Qwen3.5-2B | Qwen/Qwen3.5-9B | Qwen/Qwen3.5-27B
  --load-in-4bit (for Qwen3.5-27B on Colab A100-40GB / L4)
  --suites throughput,thinning,e2_4k,e3,e8   (or thinning_27b)
  --synthetic-dry-run (CPU validation of int8 checkpoint quant, batched cache split, E2/E3/E8 logic)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HOME", r"D:\applicationData\HuggingFace")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from w5_9b_e5_runner import (  # noqa: E402
    FILLERS,
    SYSTEM_PROMPT,
    GDNSuffixHookCollector,
    append_record_jsonl,
    apply_chat_no_think,
    apply_jlocal_scalar_update,
    build_digit4_pool,
    build_e5_prompt,
    clone_cache,
    compute_state_delta_fp64,
    count_linear_layers,
    eval_from_ctx_cache_in_place,
    extract_kv_slice,
    find_value_token_span,
    format_fact,
    format_query,
    generate_e5_sample_bank,
    get_eos_ids,
    has_logits_to_keep_arg,
    make_position_ids,
    move_cache_device,
    run_chunked_prefill,
    score_condition,
    snap_states,
)


def sync_dev(device: str) -> float:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def wilson_ci(k: int, n: int, z: float = 1.96) -> List[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def load_done_keys(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    done.add(str(json.loads(line)["record_key"]))
                except Exception:
                    continue
    return done


# ---------------------------------------------------------------------------
# Checkpoint Quantization & Host-RAM / mmap Helpers (SB8)
# ---------------------------------------------------------------------------

def quantize_states_int8(snap: Dict[str, List[Optional[torch.Tensor]]]) -> Tuple[Dict[str, Any], int]:
    """Per-tensor/head symmetric int8 quantization of recurrent & conv checkpoints on CPU."""
    q_conv, q_rec = [], []
    n_bytes = 0
    for c in snap["conv"]:
        if c is None:
            q_conv.append(None)
        else:
            t = c.detach().cpu().float()
            mx = t.abs().amax(dim=(-1,), keepdim=True).clamp(min=1e-8)
            sc = mx / 127.0
            qi = torch.round(t / sc).clamp(-127, 127).to(torch.int8)
            q_conv.append((qi, sc.half(), c.dtype))
            n_bytes += qi.numel() + sc.numel() * 2
    for r in snap["rec"]:
        if r is None:
            q_rec.append(None)
        else:
            t = r.detach().cpu().float()
            mx = t.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
            sc = mx / 127.0
            qi = torch.round(t / sc).clamp(-127, 127).to(torch.int8)
            q_rec.append((qi, sc.half(), r.dtype))
            n_bytes += qi.numel() + sc.numel() * 2
    return {"conv": q_conv, "rec": q_rec}, n_bytes


def dequantize_states_to_device(q_snap: Dict[str, Any], target_device: str) -> Dict[str, List[Optional[torch.Tensor]]]:
    conv_out, rec_out = [], []
    for item in q_snap["conv"]:
        if item is None:
            conv_out.append(None)
        else:
            qi, sc, orig_dt = item
            t = (qi.float() * sc.float()).to(dtype=orig_dt, device=target_device)
            conv_out.append(t)
    for item in q_snap["rec"]:
        if item is None:
            rec_out.append(None)
        else:
            qi, sc, orig_dt = item
            t = (qi.float() * sc.float()).to(dtype=orig_dt, device=target_device)
            rec_out.append(t)
    return {"conv": conv_out, "rec": rec_out}


def convert_states_precision(snap: Dict[str, List[Optional[torch.Tensor]]], dtype_mode: str, device: str) -> Tuple[Any, int]:
    """Pack snapshot at CPU host RAM under fp32_rec, bf16, or int8; report exact bytes."""
    if dtype_mode == "int8":
        return quantize_states_int8(snap)
    tgt_dt = torch.float32 if dtype_mode == "fp32_rec" else torch.bfloat16
    b_per = 4 if dtype_mode == "fp32_rec" else 2
    n_bytes = 0
    c_out, r_out = [], []
    for c in snap["conv"]:
        if c is None:
            c_out.append(None)
        else:
            t = c.detach().cpu().to(dtype=torch.bfloat16)
            c_out.append((t, c.dtype))
            n_bytes += t.numel() * 2
    for r in snap["rec"]:
        if r is None:
            r_out.append(None)
        else:
            t = r.detach().cpu().to(dtype=tgt_dt)
            r_out.append((t, r.dtype))
            n_bytes += t.numel() * b_per
    return {"conv": c_out, "rec": r_out}, n_bytes


def restore_snap_into_cache(cache: Any, packed: Any, dtype_mode: str, target_device: str) -> None:
    if dtype_mode == "int8":
        rest = dequantize_states_to_device(packed, target_device)
    else:
        rest = {
            "conv": [(item[0].to(device=target_device, dtype=item[1]) if item is not None else None) for item in packed["conv"]],
            "rec": [(item[0].to(device=target_device, dtype=item[1]) if item is not None else None) for item in packed["rec"]],
        }
    for i, c in enumerate(rest["conv"]):
        if c is not None:
            cache.layers[i].conv_states[0].copy_(c)
    for i, r in enumerate(rest["rec"]):
        if r is not None:
            cache.layers[i].recurrent_states[0].copy_(r)


def _repeat_cache_batch(cache: Any, batch_size: int) -> Any:
    """Expand B=1 cache to `batch_size` along dim=0 cleanly."""
    if batch_size == 1:
        return clone_cache(cache)
    out = clone_cache(cache)
    for layer in out.layers:
        if getattr(layer, "keys", None) is not None and torch.is_tensor(layer.keys):
            layer.keys = layer.keys.repeat(batch_size, 1, 1, 1).contiguous()
            layer.values = layer.values.repeat(batch_size, 1, 1, 1).contiguous()
        if hasattr(layer, "conv_states") and isinstance(layer.conv_states, dict):
            layer.conv_states = {
                k: (v.repeat(batch_size, *([1] * (v.ndim - 1))).contiguous() if torch.is_tensor(v) else v)
                for k, v in layer.conv_states.items()
            }
        if hasattr(layer, "recurrent_states") and isinstance(layer.recurrent_states, dict):
            layer.recurrent_states = {
                k: (v.repeat(batch_size, *([1] * (v.ndim - 1))).contiguous() if torch.is_tensor(v) else v)
                for k, v in layer.recurrent_states.items()
            }
    return out


def _make_batch_pos(pos_start: int, seq_len: int, batch_size: int, device: str, base_model: Any) -> torch.Tensor:
    pos_1d = torch.arange(pos_start, pos_start + seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len).contiguous()
    if hasattr(base_model, "rotary_emb") and "mrope" in type(base_model.rotary_emb).__name__.lower():
        return pos_1d.unsqueeze(0).expand(3, batch_size, seq_len).contiguous()
    return pos_1d


def _batched_chunk_prefill(model: Any, base_model: Any, input_ids: torch.Tensor, pos_start: int, cache: Any, use_l2k: bool, max_chunk: int = 1024):
    bsz, L = input_ids.shape
    last_lg = None
    for s in range(0, L, max_chunk):
        e = min(L, s + max_chunk)
        p_ids = _make_batch_pos(pos_start + s, e - s, bsz, input_ids.device, base_model)
        kw = {"input_ids": input_ids[:, s:e], "position_ids": p_ids, "use_cache": True, "past_key_values": cache}
        if use_l2k:
            kw["logits_to_keep"] = 1
        out = model(**kw)
        cache = out.past_key_values
        if e == L:
            last_lg = out.logits[:, -1:, :].detach()
        del out
    return last_lg, cache


# ---------------------------------------------------------------------------
# Suite 1: SB7 Batched Serving Throughput (B = 1, 2, 4, 8)
# ---------------------------------------------------------------------------

def run_throughput_suite(
    model: Any, base_model: Any, tok: Any,
    n_linear: int, device: str, use_l2k: bool,
    ctx_lens: List[int], batch_sizes: List[int], out_dir: Path, tag: str,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    bank = generate_e5_sample_bank(n_samples=2, depths=[0.50], seed=911)
    for T_tgt in ctx_lens:
        p = build_e5_prompt(tok, bank[0], target_length=T_tgt, n_linear=n_linear, block_horizon="deep", device=device)
        ids_o, ids_e, ids_q = p["ids_o"], p["ids_e"], p["ids_q"]
        p0, b_end, T_ctx = p["p0"], p["b_end"], p["T_ctx"]

        # Prebuild B=1 caches once
        with torch.no_grad():
            _, c_pre1 = _batched_chunk_prefill(model, base_model, ids_o[:, :p0], 0, None, use_l2k)
            _, c_ctx1 = _batched_chunk_prefill(model, base_model, ids_o[:, p0:T_ctx], p0, clone_cache(c_pre1), use_l2k)
            c_pre1_cpu = move_cache_device(c_pre1, "cpu")
            c_ctx1_cpu = move_cache_device(c_ctx1, "cpu")
            del c_pre1, c_ctx1
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        t_key = f"T{T_tgt}"
        results[t_key] = {"T_ctx": int(T_ctx), "b_len": int(b_end - p0), "suf_len": int(T_ctx - b_end), "batches": {}}

        for B in batch_sizes:
            b_rec: Dict[str, Any] = {"batch_size": B}
            ids_e_B = ids_e.repeat(B, 1)
            ids_q_B = ids_q.repeat(B, 1)
            gamma_dummy = [torch.full((B, int(getattr(model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config, "linear_num_value_heads", 32)), 1, 1), 0.7, device=device, dtype=torch.float64) for _ in range(n_linear)]

            # 1. O_local batched serving path
            try:
                if str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                c_pre_B = _repeat_cache_batch(move_cache_device(c_pre1_cpu, device), B)
                c_ctx_B = _repeat_cache_batch(move_cache_device(c_ctx1_cpu, device), B)
                s_old_B = snap_states(c_ctx_B)

                ol_times = []
                for it in range(4):
                    c_blk_work = clone_cache(c_pre_B)
                    c_out_work = clone_cache(c_ctx_B)
                    t0 = sync_dev(device)
                    with torch.no_grad():
                        _, c_blk_work = _batched_chunk_prefill(model, base_model, ids_e_B[:, p0:b_end], p0, c_blk_work, use_l2k)
                        # Inject batched delta & patch batched KV [p0, b_end) + run batched query
                        li = 0
                        for layer, l_old, l_new in zip(c_out_work.layers, c_ctx_B.layers, c_blk_work.layers):
                            rs_o = getattr(l_old, "recurrent_states", {}).get(0) if hasattr(l_old, "recurrent_states") else None
                            rs_n = getattr(l_new, "recurrent_states", {}).get(0) if hasattr(l_new, "recurrent_states") else None
                            if rs_n is not None and rs_o is not None:
                                dr = rs_n.to(torch.float64) - s_old_B["rec"][c_out_work.layers.index(layer)].to(torch.float64)
                                layer.recurrent_states[0].copy_((rs_o.to(torch.float64) + gamma_dummy[li] * dr).to(rs_o.dtype))
                                li += 1
                            if getattr(layer, "keys", None) is not None:
                                layer.keys[:, :, p0:b_end, :].copy_(l_new.keys[:, :, p0:b_end, :])
                                layer.values[:, :, p0:b_end, :].copy_(l_new.values[:, :, p0:b_end, :])
                        _, _ = _batched_chunk_prefill(model, base_model, ids_q_B, T_ctx, c_out_work, use_l2k)
                    dt_ms = (sync_dev(device) - t0) * 1000.0
                    if it > 0:
                        ol_times.append(dt_ms)
                    del c_blk_work, c_out_work
                peak_ol = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if str(device).startswith("cuda") else 0.0
                del c_pre_B, c_ctx_B
                ol_mean = sum(ol_times) / max(1, len(ol_times))
                ol_s = sorted(ol_times)
                b_rec["olocal"] = {
                    "status": "ok",
                    "mean_ms": ol_mean,
                    "p50_ms": ol_s[len(ol_s) // 2],
                    "p95_ms": ol_s[-1],
                    "req_per_sec": float(B / (ol_mean / 1000.0)),
                    "ctx_tok_per_sec": float((B * T_ctx) / (ol_mean / 1000.0)),
                    "peak_vram_mb": peak_ol,
                }
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    b_rec["olocal"] = {"status": "oom"}
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
                else:
                    raise

            # 2. Batched suffix_recompute baseline ([p0, T_ctx) + query)
            try:
                if str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                c_pre_B = _repeat_cache_batch(move_cache_device(c_pre1_cpu, device), B)
                rc_times = []
                for it in range(3):
                    c_work = clone_cache(c_pre_B)
                    t0 = sync_dev(device)
                    with torch.no_grad():
                        _, c_work = _batched_chunk_prefill(model, base_model, ids_e_B[:, p0:T_ctx], p0, c_work, use_l2k)
                        _, _ = _batched_chunk_prefill(model, base_model, ids_q_B, T_ctx, c_work, use_l2k)
                    dt_ms = (sync_dev(device) - t0) * 1000.0
                    if it > 0:
                        rc_times.append(dt_ms)
                    del c_work
                peak_rc = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if str(device).startswith("cuda") else 0.0
                del c_pre_B
                rc_mean = sum(rc_times) / max(1, len(rc_times))
                b_rec["suffix_recompute"] = {
                    "status": "ok",
                    "mean_ms": rc_mean,
                    "req_per_sec": float(B / (rc_mean / 1000.0)),
                    "ctx_tok_per_sec": float((B * T_ctx) / (rc_mean / 1000.0)),
                    "peak_vram_mb": peak_rc,
                }
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    b_rec["suffix_recompute"] = {"status": "oom"}
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
                else:
                    raise

            if b_rec.get("olocal", {}).get("status") == "ok" and b_rec.get("suffix_recompute", {}).get("status") == "ok":
                b_rec["speedup_olocal_vs_recompute"] = b_rec["suffix_recompute"]["mean_ms"] / max(1e-6, b_rec["olocal"]["mean_ms"])
            results[t_key]["batches"][f"B{B}"] = b_rec
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        del c_pre1_cpu, c_ctx1_cpu

    (out_dir / f"w11_throughput_{tag}_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# Suite 2: SB8 Host-RAM O(T/C) Checkpoint Thinning, Quantization & Substep Profiling
# ---------------------------------------------------------------------------

def run_thinning_suite(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int],
    n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool,
    T_ctx_target: int, strides: List[int], dtypes: List[str], n_samples: int,
    out_dir: Path, tag: str, test_mmap: bool = True,
) -> Dict[str, Any]:
    bank = generate_e5_sample_bank(n_samples=max(n_samples, 10), depths=[0.33, 0.57, 0.72], seed=5544)
    samples = [build_e5_prompt(tok, bank[i], target_length=T_ctx_target, n_linear=n_linear, block_horizon="deep", device=device) for i in range(n_samples)]

    grid: Dict[str, Any] = {}
    t_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_heads = int(getattr(t_cfg, "linear_num_value_heads", 32))
    n_k_heads = int(getattr(t_cfg, "linear_num_key_heads", 16))
    d_v = int(getattr(t_cfg, "linear_value_head_dim", 128))
    conv_dim = n_heads * d_v + 2 * n_k_heads * d_k
    per_ckpt_fp32_bytes = n_linear * n_heads * d_k * d_v * 4 + n_linear * conv_dim * w_conv * 2

    # Analytical Host-RAM matrix across strides & precisions at 128K
    analytical_128k = {}
    for C in strides:
        n_ck = 131072 // C
        analytical_128k[f"C{C}"] = {
            "n_ckpts_at_128k": int(n_ck),
            "fp32_rec_gib": float((n_ck * per_ckpt_fp32_bytes) / (1024 ** 3)),
            "bf16_gib": float((n_ck * per_ckpt_fp32_bytes * 0.5) / (1024 ** 3)),
            "int8_gib": float((n_ck * (n_linear * n_heads * d_k * d_v + n_linear * n_heads * 2 + n_linear * conv_dim * w_conv)) / (1024 ** 3)),
        }

    # Empirical test across (C, dtype)
    for idx_s, smp in enumerate(samples):
        ids_o, ids_e, ids_q = smp["ids_o"], smp["ids_e"], smp["ids_q"]
        p0, b_end, T_ctx = smp["p0"], smp["b_end"], smp["T_ctx"]
        old_val, new_val = bank[idx_s]["old"], bank[idx_s]["new"]
        f_o, f_e = smp["f_o"], smp["f_e"]

        with torch.no_grad():
            _, c_ora = run_chunked_prefill(model, base_model, ids_e, 0, None, 2048, use_l2k)
            ev_ora = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_ora, T_ctx, ids_q, use_l2k)
            lg_ora = ev_ora["first_logits"].detach().clone()
            del c_ora

            # Run full prefix once to record states at every stride boundary AND old suffix context
            _, c_full_old = run_chunked_prefill(model, base_model, ids_o[:, :b_end], 0, None, 2048, use_l2k)
            s_old_b_true = snap_states(c_full_old)
            with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as col:
                _, c_full_old = run_chunked_prefill(model, base_model, ids_o[:, b_end:T_ctx], b_end, c_full_old, 2048, use_l2k)
            hs = col.finalize(expected_suffix_len=T_ctx - b_end, device=device)
            c_ctx_old_cpu = move_cache_device(c_full_old, "cpu")
            del c_full_old

            # Checkpoints along prefix [0, p0) at nearest boundary below p0 for each stride C
            for C in strides:
                c_pos = (p0 // C) * C
                # Build exact state at c_pos (if c_pos == 0, blank state; else run prefix[:c_pos])
                if c_pos > 0:
                    _, c_ck = run_chunked_prefill(model, base_model, ids_o[:, :c_pos], 0, None, 2048, use_l2k)
                    snap_ck = snap_states(c_ck)
                    kv_pre_template_cpu = move_cache_device(c_ck, "cpu")
                    del c_ck
                else:
                    snap_ck = None
                    kv_pre_template_cpu = None

                # Reference KV on [c_pos, p0) is stored/available in host c_ctx_old_cpu; allocate fresh cache for replay!
                for dt_mode in dtypes:
                    cell = f"C{C}_{dt_mode}"
                    if cell not in grid:
                        grid[cell] = {"stride": C, "dtype": dt_mode, "records": []}

                    t0 = sync_dev(device)
                    # Substep 1: host -> device ckpt load + dequant + replay [c_pos, p0)
                    if c_pos > 0 and snap_ck is not None:
                        packed, ckpt_bytes = convert_states_precision(snap_ck, dt_mode, device)
                        # Test mmap round-trip on sample 0 cleanly (view bf16 as uint16 for numpy)
                        mmap_rw_ms = 0.0
                        if test_mmap and idx_s == 0 and C == strides[min(1, len(strides) - 1)]:
                            with tempfile.NamedTemporaryFile(delete=False) as tf:
                                tname = tf.name
                            try:
                                np_parts = [
                                    (x[0].view(torch.uint16).numpy().reshape(-1) if x[0].dtype == torch.bfloat16 else x[0].numpy().reshape(-1))
                                    for x in packed["rec"] if x is not None
                                ]
                                raw_np = np.concatenate(np_parts)
                                tm0 = time.perf_counter()
                                mm = np.memmap(tname, dtype=raw_np.dtype, mode="w+", shape=raw_np.shape)
                                mm[:] = raw_np[:]
                                mm.flush()
                                del mm
                                mm_r = np.memmap(tname, dtype=raw_np.dtype, mode="r", shape=raw_np.shape)
                                _ = mm_r[:].copy()
                                del mm_r
                                mmap_rw_ms = (time.perf_counter() - tm0) * 1000.0
                            finally:
                                try:
                                    os.unlink(tname)
                                except OSError:
                                    pass

                        t_h2d0 = sync_dev(device)
                        c_work = move_cache_device(kv_pre_template_cpu, device)
                        restore_snap_into_cache(c_work, packed, dt_mode, device)
                        h2d_ms = (sync_dev(device) - t_h2d0) * 1000.0
                    else:
                        ckpt_bytes = 0
                        mmap_rw_ms = 0.0
                        h2d_ms = 0.0
                        c_work = None

                    t_rep0 = sync_dev(device)
                    if p0 > c_pos:
                        _, c_work = run_chunked_prefill(model, base_model, ids_o[:, c_pos:p0], c_pos, c_work, 2048, use_l2k)
                    prefix_replay_ms = (sync_dev(device) - t_rep0) * 1000.0

                    # Substep 2: edited block [p0, b_end)
                    t_blk0 = sync_dev(device)
                    _, c_new_b = run_chunked_prefill(model, base_model, ids_e[:, p0:b_end], p0, c_work, 2048, use_l2k)
                    block_fwd_ms = (sync_dev(device) - t_blk0) * 1000.0
                    s_new_b = snap_states(c_new_b)
                    new_blk_kv = extract_kv_slice(c_new_b, p0, b_end)
                    del c_new_b

                    # Note: if checkpoint quantization corrupted state, delta uses old state at b (which assumes clean old state vs replayed old state)
                    delta = compute_state_delta_fp64(s_old_b_true, s_new_b)
                    c_ctx = move_cache_device(c_ctx_old_cpu, device)
                    t_inj0 = sync_dev(device)
                    apply_jlocal_scalar_update(c_ctx, delta, hs["gamma_scalar"], b_end, T_ctx, w_conv, use_decay=True)
                    inject_ms = (sync_dev(device) - t_inj0) * 1000.0

                    t_kv0 = sync_dev(device)
                    for i_l, sl in enumerate(new_blk_kv):
                        if sl is not None:
                            c_ctx.layers[i_l].keys[:, :, p0:b_end, :].copy_(sl[0])
                            c_ctx.layers[i_l].values[:, :, p0:b_end, :].copy_(sl[1])
                    kv_patch_ms = (sync_dev(device) - t_kv0) * 1000.0
                    total_edit_ms = h2d_ms + prefix_replay_ms + block_fwd_ms + inject_ms + kv_patch_ms

                    ev = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_ctx, T_ctx, ids_q, use_l2k)
                    sc = score_condition(ev, new_val, old_val, f_e, f_o, lg_ora)
                    del c_ctx

                    grid[cell]["records"].append({
                        "em": bool(sc["contains_new"]),
                        "kl": float(sc["kl_vs_oneshot"]),
                        "top1_eq_ora": bool(sc["top1_eq_oracle"]),
                        "ckpt_bytes": int(ckpt_bytes),
                        "replay_tokens": int(p0 - c_pos),
                        "substeps_ms": {
                            "h2d": h2d_ms, "prefix_replay": prefix_replay_ms,
                            "block_fwd": block_fwd_ms, "inject": inject_ms, "kv_patch": kv_patch_ms,
                            "total_edit": total_edit_ms, "mmap_roundtrip": mmap_rw_ms,
                        },
                    })
                del kv_pre_template_cpu

    agg_grid = {}
    for cell, blob in grid.items():
        recs = blob["records"]
        n = len(recs)
        agg_grid[cell] = {
            "stride": blob["stride"], "dtype": blob["dtype"], "n": n,
            "em_rate": sum(1 for r in recs if r["em"]) / max(1, n),
            "top1_eq_oracle": sum(1 for r in recs if r["top1_eq_ora"]) / max(1, n),
            "mean_kl": sum(r["kl"] for r in recs) / max(1, n),
            "mean_ckpt_mb": (sum(r["ckpt_bytes"] for r in recs) / max(1, n)) / (1024 ** 2),
            "mean_replay_tokens": sum(r["replay_tokens"] for r in recs) / max(1, n),
            "mean_substeps_ms": {
                k: sum(r["substeps_ms"][k] for r in recs) / max(1, n)
                for k in ("h2d", "prefix_replay", "block_fwd", "inject", "kv_patch", "total_edit", "mmap_roundtrip")
            },
        }

    out = {"model": tag, "T_ctx_target": T_ctx_target, "analytical_128k_host_ram": analytical_128k, "empirical_grid": agg_grid}
    (out_dir / f"w11_thinning_{tag}_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Suite 3: SB9 Remaining 9B Scans (E2@4K n=100, E3 multi-fact perms, E8 T* curve)
# ---------------------------------------------------------------------------

def run_9b_e2_4k_suite(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int],
    n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool,
    n_samples: int, out_dir: Path, tag: str,
) -> Dict[str, Any]:
    jsonl_p = out_dir / f"w11_e2_4k_{tag}_records.jsonl"
    done = load_done_keys(jsonl_p)
    bank = generate_e5_sample_bank(n_samples=n_samples, depths=[0.25, 0.50, 0.75], seed=202699)

    conds_list = ("oneshot_oracle", "jlocal", "jlocal_stale", "state_only", "kv_only_blk", "old")
    for idx, smp_spec in enumerate(bank):
        rk = f"{tag}:e2_4k:{idx}"
        if rk in done:
            continue
        p = build_e5_prompt(tok, smp_spec, target_length=4096, n_linear=n_linear, block_horizon="deep", device=device)
        ids_o, ids_e, ids_q = p["ids_o"], p["ids_e"], p["ids_q"]
        p0, b_end, T_ctx = p["p0"], p["b_end"], p["T_ctx"]
        suf_len = T_ctx - b_end
        old_v, new_v, f_o, f_e = smp_spec["old"], smp_spec["new"], p["f_o"], p["f_e"]

        with torch.no_grad():
            _, c_one = run_chunked_prefill(model, base_model, ids_e, 0, None, 2048, use_l2k)
            ev_one = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_one, T_ctx, ids_q, use_l2k)
            lg_one = ev_one["first_logits"].detach().clone()
            del c_one

            _, c_pre = run_chunked_prefill(model, base_model, ids_o[:, :p0], 0, None, 2048, use_l2k)
            c_pre_cpu = move_cache_device(c_pre, "cpu")
            _, c_old_b = run_chunked_prefill(model, base_model, ids_o[:, p0:b_end], p0, c_pre, 2048, use_l2k)
            del c_pre
            s_old_b = snap_states(c_old_b)
            c_old_b_cpu = move_cache_device(c_old_b, "cpu")
            del c_old_b

            c_pre_gpu = move_cache_device(c_pre_cpu, device)
            del c_pre_cpu
            _, c_new_b = run_chunked_prefill(model, base_model, ids_e[:, p0:b_end], p0, c_pre_gpu, 2048, use_l2k)
            del c_pre_gpu
            delta = compute_state_delta_fp64(s_old_b, snap_states(c_new_b))
            kv_blk = extract_kv_slice(c_new_b, p0, b_end)
            del c_new_b

            c_ctx_old = move_cache_device(c_old_b_cpu, device)
            del c_old_b_cpu
            with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as col:
                if suf_len > 0:
                    _, c_ctx_old = run_chunked_prefill(model, base_model, ids_o[:, b_end:T_ctx], b_end, c_ctx_old, 2048, use_l2k)
            hs = col.finalize(expected_suffix_len=suf_len, device=device)

            scores = {"oneshot_oracle": score_condition(ev_one, new_v, old_v, f_e, f_o, lg_one)}
            ev_old = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_ctx_old), T_ctx, ids_q, use_l2k)
            scores["old"] = score_condition(ev_old, new_v, old_v, f_e, f_o, lg_one)

            # jlocal
            c_w = clone_cache(c_ctx_old)
            apply_jlocal_scalar_update(c_w, delta, hs["gamma_scalar"], b_end, T_ctx, w_conv, use_decay=True)
            for li_k, sl in enumerate(kv_blk):
                if sl is not None:
                    c_w.layers[li_k].keys[:, :, p0:b_end, :].copy_(sl[0])
                    c_w.layers[li_k].values[:, :, p0:b_end, :].copy_(sl[1])
            ev_jl = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_w, T_ctx, ids_q, use_l2k)
            scores["jlocal"] = score_condition(ev_jl, new_v, old_v, f_e, f_o, lg_one)
            del c_w

            # jlocal_stale
            c_w = clone_cache(c_ctx_old)
            apply_jlocal_scalar_update(c_w, delta, hs["gamma_scalar"], b_end, T_ctx, w_conv, use_decay=False)
            for li_k, sl in enumerate(kv_blk):
                if sl is not None:
                    c_w.layers[li_k].keys[:, :, p0:b_end, :].copy_(sl[0])
                    c_w.layers[li_k].values[:, :, p0:b_end, :].copy_(sl[1])
            ev_st = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_w, T_ctx, ids_q, use_l2k)
            scores["jlocal_stale"] = score_condition(ev_st, new_v, old_v, f_e, f_o, lg_one)
            del c_w

            # state_only
            c_w = clone_cache(c_ctx_old)
            apply_jlocal_scalar_update(c_w, delta, hs["gamma_scalar"], b_end, T_ctx, w_conv, use_decay=True)
            ev_so = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_w, T_ctx, ids_q, use_l2k)
            scores["state_only"] = score_condition(ev_so, new_v, old_v, f_e, f_o, lg_one)
            del c_w

            # kv_only_blk
            c_w = clone_cache(c_ctx_old)
            for li_k, sl in enumerate(kv_blk):
                if sl is not None:
                    c_w.layers[li_k].keys[:, :, p0:b_end, :].copy_(sl[0])
                    c_w.layers[li_k].values[:, :, p0:b_end, :].copy_(sl[1])
            ev_kv = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_w, T_ctx, ids_q, use_l2k)
            scores["kv_only_blk"] = score_condition(ev_kv, new_v, old_v, f_e, f_o, lg_one)
            del c_w, c_ctx_old

        append_record_jsonl(jsonl_p, {"record_key": rk, "T_ctx": T_ctx, "depth": smp_spec["depth"], "conditions": scores})

    all_r = [json.loads(l) for l in jsonl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(all_r)
    out = {"suite": "e2_4k", "n": n, "conditions": {}}
    for cn in conds_list:
        kem = sum(1 for r in all_r if r["conditions"][cn]["contains_new"])
        ktop = sum(1 for r in all_r if r["conditions"][cn]["top1_eq_oracle"])
        out["conditions"][cn] = {
            "em_rate": kem / max(1, n), "em_ci95": wilson_ci(kem, n),
            "top1_eq_oracle": ktop / max(1, n),
            "kl_mean": sum(r["conditions"][cn]["kl_vs_oneshot"] for r in all_r) / max(1, n),
        }
    (out_dir / f"w11_e2_4k_{tag}_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_9b_e3_suite(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int],
    n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool,
    k2_n: int, k4_n: int, out_dir: Path, tag: str,
) -> Dict[str, Any]:
    """9B multi-fact sequential O_local permutation invariance & collateral preservation (E3)."""
    jsonl_p = out_dir / f"w11_e3_{tag}_records.jsonl"
    done = load_done_keys(jsonl_p)
    rng = random.Random(202688)
    pool_d = build_digit4_pool()
    names_8 = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]

    configs = []
    for s_idx in range(k2_n):
        configs.append(("M4_k2", s_idx, 4, [1, 2], [[1, 2], [2, 1]]))
    perms4 = [[1, 3, 5, 6], [6, 5, 3, 1], [1, 5, 3, 6], [6, 3, 5, 1], [3, 6, 1, 5], [5, 1, 6, 3]]
    for s_idx in range(k4_n):
        configs.append(("M8_k4", s_idx, 8, [1, 3, 5, 6], perms4))

    for cfg_name, s_idx, m_facts, edit_indices, perms in configs:
        fd = rng.sample([str(d) for d in range(1, 10)], min(9, m_facts + 1))
        pairs = []
        for j in range(m_facts):
            v_o = rng.choice(pool_d[fd[j % len(fd)]])
            v_n = rng.choice(pool_d[fd[(j + 1) % len(fd)]])
            while v_n == v_o:
                v_n = rng.choice(pool_d[str(((j + 3) % 9) + 1)])
            pairs.append((names_8[j], v_o, v_n))

        gap = " ".join(FILLERS[(s_idx + k) % len(FILLERS)] for k in range(6))
        blocks_o = [f"{gap} {format_fact(j % 3, nm, vo)}" for j, (nm, vo, _) in enumerate(pairs)]
        suf = " ".join(FILLERS[(s_idx + 20 + k) % len(FILLERS)] for k in range(10))
        ctx_o = " ".join(blocks_o) + " " + suf

        # Build clean base full_o + ids_o and span offsets for each fact i
        q0_str = format_query(0, pairs[0][0])
        full_o = apply_chat_no_think(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ctx_o + "\n\n" + q0_str}])
        qo = full_o.rfind(q0_str)
        prefix_hdr_o = full_o[:qo]
        enc_o = tok(prefix_hdr_o, return_offsets_mapping=True, add_special_tokens=False)
        ids_o_list = enc_o["input_ids"]
        T_ctx = len(ids_o_list)
        fact_metas = []
        valid = True
        for j, (nm, vo, vn) in enumerate(pairs):
            sent_o = format_fact(j % 3, nm, vo)
            sent_e = format_fact(j % 3, nm, vn)
            c_idx = prefix_hdr_o.find(sent_o)
            pos_o = find_value_token_span(prefix_hdr_o, vo, c_idx + sent_o.rfind(vo), enc_o["offset_mapping"])
            if len(pos_o) != 4:
                valid = False
                break
            p0_j, p1_j = pos_o[0], pos_o[-1] + 1
            vn_ids = tok.encode(vn, add_special_tokens=False)
            if len(vn_ids) != 4 or tok.encode(vo, add_special_tokens=False)[0] == vn_ids[0]:
                valid = False
                break
            q_j = format_query(j % 3, nm)
            q_ids_j = tok(full_o[qo:].replace(q0_str, q_j), add_special_tokens=False)["input_ids"]
            fact_metas.append({
                "j": j, "nm": nm, "vo": vo, "vn": vn, "p0": p0_j, "p1": p1_j,
                "vn_ids": vn_ids, "f_o": tok.encode(vo, add_special_tokens=False)[0], "f_e": vn_ids[0],
                "q_ids": torch.tensor([q_ids_j], device=device),
            })
        if not valid:
            continue

        ids_o_t = torch.tensor([ids_o_list], device=device)
        ids_gold = ids_o_t.clone()
        for ei in edit_indices:
            fm = fact_metas[ei]
            ids_gold[0, fm["p0"]:fm["p1"]] = torch.tensor(fm["vn_ids"], device=device)

        with torch.no_grad():
            _, c_ora = run_chunked_prefill(model, base_model, ids_gold, 0, None, 2048, use_l2k)
            ora_logits = {}
            for fm in fact_metas:
                ev_o = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_ora), T_ctx, fm["q_ids"], use_l2k)
                ora_logits[fm["j"]] = ev_o["first_logits"].detach().clone()
            del c_ora

            _, c_old_full = run_chunked_prefill(model, base_model, ids_o_t, 0, None, 2048, use_l2k)

        for pidx, perm in enumerate(perms):
            rk = f"{tag}:e3:{cfg_name}:{s_idx}:p{pidx}"
            if rk in done:
                continue
            with torch.no_grad():
                c_cur = clone_cache(c_old_full)
                cur_ids = ids_o_t.clone()
                for ei in perm:
                    fm = fact_metas[ei]
                    p0_j, p1_j = fm["p0"], fm["p1"]
                    b_j = min(T_ctx, p1_j + 3 * (n_linear - 1))
                    ids_nxt = cur_ids.clone()
                    ids_nxt[0, p0_j:p1_j] = torch.tensor(fm["vn_ids"], device=device)

                    _, c_p = run_chunked_prefill(model, base_model, cur_ids[:, :p0_j], 0, None, 2048, use_l2k)
                    _, c_ob = run_chunked_prefill(model, base_model, cur_ids[:, p0_j:b_j], p0_j, clone_cache(c_p), 2048, use_l2k)
                    _, c_nb = run_chunked_prefill(model, base_model, ids_nxt[:, p0_j:b_j], p0_j, c_p, 2048, use_l2k)
                    del c_p
                    delta_j = compute_state_delta_fp64(snap_states(c_ob), snap_states(c_nb))
                    kv_j = extract_kv_slice(c_nb, p0_j, b_j)
                    suf_l = T_ctx - b_j
                    with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as col:
                        if suf_l > 0:
                            _, _ = run_chunked_prefill(model, base_model, cur_ids[:, b_j:T_ctx], b_j, c_ob, 2048, use_l2k)
                    hs_j = col.finalize(expected_suffix_len=suf_l, device=device)
                    del c_ob, c_nb

                    apply_jlocal_scalar_update(c_cur, delta_j, hs_j["gamma_scalar"], b_j, T_ctx, w_conv, use_decay=True)
                    for li_k, sl in enumerate(kv_j):
                        if sl is not None:
                            c_cur.layers[li_k].keys[:, :, p0_j:b_j, :].copy_(sl[0])
                            c_cur.layers[li_k].values[:, :, p0_j:b_j, :].copy_(sl[1])
                    cur_ids = ids_nxt

                edited_hits = []
                collat_hits = []
                for fm in fact_metas:
                    ev = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_cur), T_ctx, fm["q_ids"], use_l2k)
                    if fm["j"] in edit_indices:
                        sc = score_condition(ev, fm["vn"], fm["vo"], fm["f_e"], fm["f_o"], ora_logits[fm["j"]])
                        edited_hits.append(bool(sc["contains_new"]))
                    else:
                        sc = score_condition(ev, fm["vo"], fm["vn"], fm["f_o"], fm["f_e"], ora_logits[fm["j"]])
                        collat_hits.append(bool(sc["contains_new"]))
                del c_cur

            append_record_jsonl(jsonl_p, {
                "record_key": rk, "cfg": cfg_name, "set_idx": s_idx, "perm_idx": pidx,
                "edited_em_frac": sum(edited_hits) / max(1, len(edited_hits)),
                "all_edited_em": all(edited_hits),
                "collat_em_frac": sum(collat_hits) / max(1, len(collat_hits)),
            })
        del c_old_full

    all_r = [json.loads(l) for l in jsonl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for cn in ("M4_k2", "M8_k4"):
        sr = [r for r in all_r if r.get("cfg") == cn]
        n = len(sr)
        out[cn] = {
            "n_perm_runs": n,
            "mean_edited_em": sum(r["edited_em_frac"] for r in sr) / max(1, n),
            "all_edited_joint_em": sum(1 for r in sr if r["all_edited_em"]) / max(1, n),
            "mean_collat_retention": sum(r["collat_em_frac"] for r in sr) / max(1, n),
        }
    (out_dir / f"w11_e3_{tag}_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_e8_tstar_curve(
    model: Any, base_model: Any, tok: Any,
    n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool,
    lengths: List[int], n_trials: int, out_dir: Path, tag: str, bckpt_stride: int = 1024,
) -> Dict[str, Any]:
    """Measure clean O_local vs suffix_recompute vs full_prefill latency curve & empirical crossover T*."""
    bank = generate_e5_sample_bank(n_samples=1, depths=[0.50], seed=3301)
    curve: Dict[str, Any] = {}
    for L_tgt in lengths:
        p = build_e5_prompt(tok, bank[0], target_length=L_tgt, n_linear=n_linear, block_horizon="deep", device=device)
        ids_o, ids_e, ids_q = p["ids_o"], p["ids_e"], p["ids_q"]
        p0, b_end, T_ctx = p["p0"], p["b_end"], p["T_ctx"]
        c_pos = (p0 // bckpt_stride) * bckpt_stride
        replay_len = p0 - c_pos

        ol_ms, rec_ms, full_ms = [], [], []
        try:
            with torch.no_grad():
                _, c_ck = run_chunked_prefill(model, base_model, ids_o[:, :c_pos], 0, None, 2048, use_l2k) if c_pos > 0 else (None, None)
                c_ck_cpu = move_cache_device(c_ck, "cpu") if c_ck is not None else None
                del c_ck

                _, c_ctx_base = run_chunked_prefill(model, base_model, ids_o[:, :T_ctx], 0, None, 2048, use_l2k)
                c_ctx_cpu = move_cache_device(c_ctx_base, "cpu")
                del c_ctx_base
                gamma_d = [torch.full((1, int(getattr(model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config, "linear_num_value_heads", 32)), 1, 1), 0.7, device=device, dtype=torch.float64) for _ in range(n_linear)]

                for it in range(n_trials + 1):
                    # 1. O_local steady-state runtime: H2D ckpt + replay [c_pos, p0) + blk [p0, b_end) + inject + KV patch
                    c_ctx = move_cache_device(c_ctx_cpu, device)
                    s_old = snap_states(c_ctx)
                    t0 = sync_dev(device)
                    c_w = move_cache_device(c_ck_cpu, device) if c_ck_cpu is not None else None
                    if replay_len > 0:
                        _, c_w = run_chunked_prefill(model, base_model, ids_o[:, c_pos:p0], c_pos, c_w, 2048, use_l2k)
                    _, c_nb = run_chunked_prefill(model, base_model, ids_e[:, p0:b_end], p0, c_w, 2048, use_l2k)
                    delta = compute_state_delta_fp64(s_old, snap_states(c_nb))
                    kv_b = extract_kv_slice(c_nb, p0, b_end)
                    apply_jlocal_scalar_update(c_ctx, delta, gamma_d, b_end, T_ctx, w_conv, use_decay=True)
                    for li_k, sl in enumerate(kv_b):
                        if sl is not None:
                            c_ctx.layers[li_k].keys[:, :, p0:b_end, :].copy_(sl[0])
                            c_ctx.layers[li_k].values[:, :, p0:b_end, :].copy_(sl[1])
                    dt = (sync_dev(device) - t0) * 1000.0
                    del c_w, c_nb, c_ctx
                    if it > 0:
                        ol_ms.append(dt)

                    # 2. Suffix recompute: H2D ckpt + replay + [p0, T_ctx)
                    t0 = sync_dev(device)
                    c_r = move_cache_device(c_ck_cpu, device) if c_ck_cpu is not None else None
                    if replay_len > 0:
                        _, c_r = run_chunked_prefill(model, base_model, ids_o[:, c_pos:p0], c_pos, c_r, 2048, use_l2k)
                    _, c_r = run_chunked_prefill(model, base_model, ids_e[:, p0:T_ctx], p0, c_r, 2048, use_l2k)
                    dt_r = (sync_dev(device) - t0) * 1000.0
                    del c_r
                    if it > 0:
                        rec_ms.append(dt_r)

                    # 3. Full prefill
                    if it <= 3:
                        t0 = sync_dev(device)
                        _, c_f = run_chunked_prefill(model, base_model, ids_e[:, :T_ctx], 0, None, 2048, use_l2k)
                        dt_f = (sync_dev(device) - t0) * 1000.0
                        del c_f
                        if it > 0:
                            full_ms.append(dt_f)

                del c_ck_cpu, c_ctx_cpu
            om = sum(ol_ms) / max(1, len(ol_ms))
            rm = sum(rec_ms) / max(1, len(rec_ms))
            fm = sum(full_ms) / max(1, len(full_ms))
            curve[f"T{L_tgt}"] = {
                "T_ctx": T_ctx, "replay_len": replay_len, "b_len": b_end - p0, "suf_len": T_ctx - b_end,
                "olocal_mean_ms": om, "suffix_recompute_mean_ms": rm, "full_prefill_mean_ms": fm,
                "speedup_vs_suffix_recompute": rm / max(1e-6, om),
                "speedup_vs_full_prefill": fm / max(1e-6, om),
            }
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                curve[f"T{L_tgt}"] = {"status": "oom"}
            else:
                raise
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    (out_dir / f"w11_e8_{tag}_summary.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
    return curve


def run_synthetic_dry_run(out_dir: Path) -> Dict[str, Any]:
    torch.manual_seed(0)
    fake_snap = {"conv": [torch.randn(1, 32, 4)], "rec": [torch.randn(1, 4, 16, 16)]}
    q_snap, n_bytes = quantize_states_int8(fake_snap)
    deq = dequantize_states_to_device(q_snap, "cpu")
    err = float((deq["rec"][0] - fake_snap["rec"][0]).abs().max().item())
    assert err < 0.1, f"Unexpected large int8 quant error: {err}"
    rep = {"int8_max_abs_err": err, "int8_bytes": n_bytes, "fp32_bytes": 1 * 4 * 16 * 16 * 4 + 1 * 32 * 4 * 4}
    (out_dir / "w11_systems_dryrun.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    p.add_argument("--tag", type=str, default="2b")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--suites", type=str, default="throughput,thinning")
    p.add_argument("--out-dir", type=str, default="pilot")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--synthetic-dry-run", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.synthetic_dry_run:
        rep = run_synthetic_dry_run(out_dir)
        print("W11_SYSTEMS_DRYRUN_OK", json.dumps(rep))
        return

    suites = {s.strip() for s in args.suites.split(",") if s.strip()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    load_kw: Dict[str, Any] = {"trust_remote_code": True, "local_files_only": args.local_files_only}
    if args.load_in_4bit:
        load_kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        load_kw["device_map"] = {"": 0} if str(device).startswith("cuda") else "auto"
    else:
        load_kw["dtype"] = torch.bfloat16
        load_kw["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).eval()
    base_model = getattr(model, "model", model)
    t_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_linear = count_linear_layers(t_cfg)
    d_k = int(getattr(t_cfg, "linear_key_head_dim", 128))
    w_conv = int(getattr(t_cfg, "linear_conv_kernel_dim", 4))
    eos_ids = get_eos_ids(model, tok)
    use_l2k = has_logits_to_keep_arg(model)

    sum_p = out_dir / f"w11_systems_{args.tag}_summary.json"
    summary: Dict[str, Any] = json.loads(sum_p.read_text(encoding="utf-8")) if sum_p.exists() else {"model": args.model, "tag": args.tag}

    if "throughput" in suites:
        print(f"[{args.tag}] Running throughput (SB7)", flush=True)
        summary["throughput"] = run_throughput_suite(model, base_model, tok, n_linear, device, use_l2k, [2048, 4096], [1, 2, 4, 8], out_dir, args.tag)
        sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "thinning" in suites or "thinning_27b" in suites:
        print(f"[{args.tag}] Running thinning (SB8)", flush=True)
        T_th = 4096 if "thinning_27b" in suites else 8192
        strides = [256, 512, 1024, 2048] if "thinning_27b" in suites else [256, 512, 1024, 2048, 4096]
        n_smp = 3 if "thinning_27b" in suites else 5
        summary["thinning"] = run_thinning_suite(
            model, base_model, tok, eos_ids, n_linear, d_k, w_conv, device, use_l2k,
            T_ctx_target=T_th, strides=strides, dtypes=["fp32_rec", "bf16", "int8"],
            n_samples=n_smp, out_dir=out_dir, tag=args.tag,
        )
        sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "e2_4k" in suites:
        print(f"[{args.tag}] Running E2@4K (SB9)", flush=True)
        summary["e2_4k"] = run_9b_e2_4k_suite(model, base_model, tok, eos_ids, n_linear, d_k, w_conv, device, use_l2k, 100, out_dir, args.tag)
        sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "e3" in suites:
        print(f"[{args.tag}] Running E3 (SB9)", flush=True)
        summary["e3"] = run_9b_e3_suite(model, base_model, tok, eos_ids, n_linear, d_k, w_conv, device, use_l2k, 10, 5, out_dir, args.tag)
        sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "e8" in suites:
        print(f"[{args.tag}] Running E8 T* curve (SB9)", flush=True)
        summary["e8"] = run_e8_tstar_curve(model, base_model, tok, n_linear, d_k, w_conv, device, use_l2k, [1024, 2048, 4096, 8192, 16384], 5, out_dir, args.tag)
        sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{args.tag}] W11_SYSTEMS DONE", flush=True)


if __name__ == "__main__":
    main()
