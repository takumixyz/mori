"""Stage-C Wave 12-4: Controlled Micro-Benchmark (Random-Tensor Storage+H2D+Dequant) & Pinned-Memory INT8 Stress Test.

Strictly follows W12-4 controlled two-experiment scope:
  1. `run_nvme_swap_microbench`: synthetic random model-shaped tensor cold/warm file read -> Pinned -> Async H2D
     -> GPU dequant bandwidth micro-benchmark for 2B & 9B checkpoints ONLY (explicitly marked
     `hardware_dependent_non_cross_machine_reproducible: true`; excludes 27B 128K; `posix_fadvise(POSIX_FADV_DONTNEED)`
     records syscall support without OS page-residency verification; file-read means reported separately from totals).
  2. `_eval_ckpt_cell` (`g1b_ckpt_only_stress` / `collision_stress`): pinned in-memory quantized buffer restore across
     C in {512, 1024, 2048} (plus C=512 fp32_rec & C=1024 bf16), restoring at c_pos = (p0 // C) * C and evaluating
     `J-local_fidelity` (<= 1e-4 vs oneshot_oracle, distinct from E5 G1b on J-full) on:
       - `G1b` equivalence stress (T=4096; restore-only `ckpt_bytes > 0` subset excludes `c_pos=0` rows; C2048 d=0.75 duplicates C1024)
       - `entity_collision` stress (all cells `ckpt_bytes=0`, effective unique n=30/20)

Supports --synthetic-dry-run on CPU (`CUDA_VISIBLE_DEVICES=""`).
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from w5_9b_e5_runner import (  # noqa: E402
    FILLERS,
    GDNSuffixHookCollector,
    append_record_jsonl,
    apply_jlocal_scalar_update,
    build_digit4_pool,
    build_e5_prompt,
    clone_cache,
    compute_state_delta_fp64,
    count_linear_layers,
    eval_from_ctx_cache_in_place,
    extract_kv_slice,
    generate_e5_sample_bank,
    get_eos_ids,
    has_logits_to_keep_arg,
    move_cache_device,
    run_chunked_prefill,
    score_condition,
    snap_states,
)
from w11_systems_thinning_throughput_9b import convert_states_precision, sync_dev  # noqa: E402
from w11_varlen_e10_e12_detector import _align_contiguous_edit_span, load_done_keys, wilson_ci  # noqa: E402


NAMES_COLLISION = ["Alice", "Bob", "Carol", "Dave"]


def _pack_to_flat_bytes(packed: Dict[str, Any], dtype_mode: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    parts: List[np.ndarray] = []
    manifest: List[Dict[str, Any]] = []
    offset = 0
    for kind in ("conv", "rec"):
        for idx, item in enumerate(packed[kind]):
            if item is None:
                continue
            if dtype_mode == "int8":
                qi, sc, orig_dt = item
                for sub_k, ten in (("q", qi), ("s", sc)):
                    raw = ten.contiguous().view(torch.uint8).numpy().reshape(-1)
                    parts.append(raw)
                    manifest.append({
                        "kind": kind, "idx": idx, "sub": sub_k,
                        "shape": list(ten.shape), "dtype": str(ten.dtype),
                        "orig_dtype": str(orig_dt), "offset": offset, "nbytes": int(raw.nbytes),
                    })
                    offset += int(raw.nbytes)
            else:
                ten, orig_dt = item
                raw = ten.contiguous().view(torch.uint8).numpy().reshape(-1)
                parts.append(raw)
                manifest.append({
                    "kind": kind, "idx": idx, "sub": "t",
                    "shape": list(ten.shape), "dtype": str(ten.dtype),
                    "orig_dtype": str(orig_dt), "offset": offset, "nbytes": int(raw.nbytes),
                })
                offset += int(raw.nbytes)
    buf = np.concatenate(parts) if parts else np.zeros((0,), dtype=np.uint8)
    return buf, manifest


def _try_drop_os_page_cache(fd: int, nbytes: int) -> bool:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(fd, 0, nbytes, os.POSIX_FADV_DONTNEED)
            return True
        except OSError:
            return False
    return False


def _nvme_read_into_pinned(path: Path, nbytes: int, drop_cache: bool) -> Tuple[torch.Tensor, float, bool]:
    use_pin = torch.cuda.is_available()
    pinned = torch.empty((nbytes,), dtype=torch.uint8, pin_memory=use_pin)
    arr = pinned.numpy()
    dropped = False
    if drop_cache:
        with open(path, "rb") as f0:
            dropped = _try_drop_os_page_cache(f0.fileno(), nbytes)
    t0 = time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        mv = memoryview(arr)
        pos = 0
        while pos < nbytes:
            r = f.readinto(mv[pos:])
            if not r:
                break
            pos += r
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return pinned, dt_ms, dropped


def _async_h2d_and_dequant_gpu(
    pinned_u8: torch.Tensor,
    manifest: List[Dict[str, Any]],
    dtype_mode: str,
    device: str,
    n_layers: int,
) -> Tuple[Dict[str, List[Optional[torch.Tensor]]], float, float]:
    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    stream = torch.cuda.Stream(device=device) if is_cuda else None
    t0_h2d = sync_dev(device)
    if stream is not None:
        with torch.cuda.stream(stream):
            gpu_u8 = pinned_u8.to(device=device, non_blocking=True)
        torch.cuda.current_stream(device).wait_stream(stream)
    else:
        gpu_u8 = pinned_u8.to(device=device)
    h2d_ms = (sync_dev(device) - t0_h2d) * 1000.0

    t0_dq = sync_dev(device)
    conv_out: List[Optional[torch.Tensor]] = [None] * n_layers
    rec_out: List[Optional[torch.Tensor]] = [None] * n_layers
    dtype_map = {
        "torch.int8": torch.int8,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
    }
    if dtype_mode == "int8":
        grouped: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for m in manifest:
            grouped.setdefault((m["kind"], m["idx"]), {})[m["sub"]] = m
        for (kind, idx), pair in grouped.items():
            mq, ms = pair["q"], pair["s"]
            qi = gpu_u8[mq["offset"]:mq["offset"] + mq["nbytes"]].view(torch.int8).view(*mq["shape"])
            sc = gpu_u8[ms["offset"]:ms["offset"] + ms["nbytes"]].view(torch.float16).view(*ms["shape"])
            orig_dt = dtype_map.get(mq["orig_dtype"], torch.bfloat16)
            ten = (qi.float() * sc.float()).to(dtype=orig_dt)
            if kind == "conv":
                conv_out[idx] = ten
            else:
                rec_out[idx] = ten
    else:
        for m in manifest:
            dt = dtype_map[m["dtype"]]
            orig_dt = dtype_map.get(m["orig_dtype"], dt)
            ten = gpu_u8[m["offset"]:m["offset"] + m["nbytes"]].view(dt).view(*m["shape"]).to(dtype=orig_dt)
            if m["kind"] == "conv":
                conv_out[m["idx"]] = ten
            else:
                rec_out[m["idx"]] = ten
    dequant_ms = (sync_dev(device) - t0_dq) * 1000.0
    return {"conv": conv_out, "rec": rec_out}, h2d_ms, dequant_ms


def run_nvme_swap_microbench(
    model_shapes: Dict[str, Dict[str, int]],
    device: str,
    out_dir: Path,
    n_repeats: int = 5,
) -> Dict[str, Any]:
    """Single-checkpoint cold/warm NVMe->Pinned->Async H2D restore bandwidth micro-benchmark (2B/9B only)."""
    results: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(dir=str(out_dir)) as tmpd:
        tmp_path = Path(tmpd)
        for mname, sh in model_shapes.items():
            n_lin, n_layers = sh["n_linear"], sh["n_layers"]
            H_v, H_k, d_k, d_v, w_conv = sh["H_v"], sh["H_k"], sh["d_k"], sh["d_v"], sh["w_conv"]
            conv_dim = H_v * d_v + 2 * H_k * d_k
            snap = {"conv": [], "rec": []}
            for li in range(n_layers):
                if li < n_lin:
                    snap["conv"].append(torch.randn((1, conv_dim, w_conv), dtype=torch.bfloat16))
                    snap["rec"].append(torch.randn((1, H_v, d_k, d_v), dtype=torch.float32))
                else:
                    snap["conv"].append(None)
                    snap["rec"].append(None)

            results[mname] = {"shapes": sh, "modes": {}}
            for dtype_mode in ("fp32_rec", "bf16", "int8"):
                packed, _ = convert_states_precision(snap, dtype_mode, device)
                buf, manifest = _pack_to_flat_bytes(packed, dtype_mode)
                bin_path = tmp_path / f"{mname}_{dtype_mode}.bin"

                t_w0 = time.perf_counter()
                with open(bin_path, "wb") as fw:
                    fw.write(buf.tobytes())
                    fw.flush()
                    os.fsync(fw.fileno())
                write_ms = (time.perf_counter() - t_w0) * 1000.0

                cold_reads, warm_reads, h2ds, dequants = [], [], [], []
                dropped_any = False
                for r in range(n_repeats):
                    pin_c, rd_c, drp = _nvme_read_into_pinned(bin_path, int(buf.nbytes), drop_cache=True)
                    dropped_any = dropped_any or drp
                    if r > 0 or n_repeats == 1:
                        cold_reads.append(rd_c)
                    del pin_c

                    pin_w, rd_w, _ = _nvme_read_into_pinned(bin_path, int(buf.nbytes), drop_cache=False)
                    _, h2d_ms, dq_ms = _async_h2d_and_dequant_gpu(pin_w, manifest, dtype_mode, device, n_layers)
                    if r > 0 or n_repeats == 1:
                        warm_reads.append(rd_w)
                        h2ds.append(h2d_ms)
                        dequants.append(dq_ms)
                    del pin_w

                mean_cold = sum(cold_reads) / len(cold_reads)
                mean_warm = sum(warm_reads) / len(warm_reads)
                mean_h2d = sum(h2ds) / len(h2ds)
                mean_dq = sum(dequants) / len(dequants)
                results[mname]["modes"][dtype_mode] = {
                    "ckpt_bytes": int(buf.nbytes),
                    "ckpt_mb": float(buf.nbytes / (1024 ** 2)),
                    "nvme_write_ms": float(write_ms),
                    "nvme_read_cold_ms": float(mean_cold),
                    "nvme_read_warm_ms": float(mean_warm),
                    "fadvise_dontneed_supported": bool(dropped_any),
                    "cold_read_bw_gb_s": float((buf.nbytes / 1e9) / max(1e-6, mean_cold / 1000.0)),
                    "warm_read_bw_gb_s": float((buf.nbytes / 1e9) / max(1e-6, mean_warm / 1000.0)),
                    "pinned_async_h2d_ms": float(mean_h2d),
                    "h2d_bw_gb_s": float((buf.nbytes / 1e9) / max(1e-6, mean_h2d / 1000.0)),
                    "gpu_dequant_ms": float(mean_dq),
                    "total_swap_in_cold_ms": float(mean_cold + mean_h2d + mean_dq),
                    "total_swap_in_warm_ms": float(mean_warm + mean_h2d + mean_dq),
                }
                bin_path.unlink(missing_ok=True)
    return results


def _build_collision_cases(tok: Any, n_linear: int, n_cases: int) -> List[Dict[str, Any]]:
    rng = random.Random(202644)
    pool_d = build_digit4_pool()
    pre_fillers = " ".join(FILLERS[i % len(FILLERS)] for i in range(10))
    suf_fillers = " ".join(FILLERS[(i + 15) % len(FILLERS)] for i in range(45))
    cases: List[Dict[str, Any]] = []
    cnt = 0
    for idx in range(n_cases * 4):
        if cnt >= n_cases:
            break
        fd = rng.sample([str(d) for d in range(1, 10)], 4)
        c_old = rng.choice(pool_d[fd[0]])
        c_new = rng.choice(pool_d[fd[1]])
        c_sib = c_old[:2] + rng.choice(pool_d[fd[2]])[2:]
        c_oth = rng.choice(pool_d[fd[3]])
        target = NAMES_COLLISION[idx % len(NAMES_COLLISION)]
        other = NAMES_COLLISION[(idx + 1) % len(NAMES_COLLISION)]
        q_s = f"Question: What is the exact security code for {target}? Answer with only the 4-digit code."
        u_o = (
            f"{pre_fillers} Registry: {target} has security code {c_old}; "
            f"{target}-Backup has security code {c_sib}; {other} has security code {c_oth}. "
            f"{suf_fillers}\n\n{q_s}"
        )
        u_e = (
            f"{pre_fillers} Registry: {target} has security code {c_new}; "
            f"{target}-Backup has security code {c_sib}; {other} has security code {c_oth}. "
            f"{suf_fillers}\n\n{q_s}"
        )
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
        if al is None or al["delta_L"] != 0:
            continue
        cases.append({"case_id": cnt, "old_val": c_old, "new_val": c_new, "al": al})
        cnt += 1
    return cases


def _eval_ckpt_cell(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int],
    n_linear: int, d_k: int, w_conv: int, n_layers: int,
    device: str, use_l2k: bool,
    ids_o: torch.Tensor, ids_e: torch.Tensor, ids_q: torch.Tensor,
    p0: int, b_end: int, T_ctx: int, old_val: str, new_val: str, lg_ora: torch.Tensor,
    s_old_b_true: Dict[str, List[Optional[torch.Tensor]]], c_ctx_old_cpu: Any, hs: Dict[str, Any],
    c_pos: int, snap_ck: Optional[Dict[str, List[Optional[torch.Tensor]]]],
    kv_pre_template_cpu: Optional[Any], dt_mode: str, include_kv_only: bool = False,
) -> Dict[str, Any]:
    f_o = tok.encode(old_val, add_special_tokens=False)[0]
    f_e = tok.encode(new_val, add_special_tokens=False)[0]

    with torch.no_grad():
        if c_pos > 0 and snap_ck is not None:
            packed, ckpt_bytes = convert_states_precision(snap_ck, dt_mode, device)
            buf, manifest = _pack_to_flat_bytes(packed, dt_mode)
            pin_u8 = torch.from_numpy(buf).pin_memory() if str(device).startswith("cuda") else torch.from_numpy(buf)
            rest, h2d_ms, dq_ms = _async_h2d_and_dequant_gpu(pin_u8, manifest, dt_mode, device, n_layers)
            c_work = move_cache_device(kv_pre_template_cpu, device)
            for i, c in enumerate(rest["conv"]):
                if c is not None:
                    c_work.layers[i].conv_states[0].copy_(c)
            for i, r in enumerate(rest["rec"]):
                if r is not None:
                    c_work.layers[i].recurrent_states[0].copy_(r)
        else:
            ckpt_bytes, h2d_ms, dq_ms, c_work = 0, 0.0, 0.0, None

        t_rep0 = sync_dev(device)
        if p0 > c_pos:
            _, c_work = run_chunked_prefill(model, base_model, ids_o[:, c_pos:p0], c_pos, c_work, 2048, use_l2k)
        replay_ms = (sync_dev(device) - t_rep0) * 1000.0

        t_blk0 = sync_dev(device)
        c_p0_old = clone_cache(c_work) if c_work is not None else None
        _, c_old_b_ckpt = run_chunked_prefill(model, base_model, ids_o[:, p0:b_end], p0, c_p0_old, 2048, use_l2k)
        s_old_b_ckpt = snap_states(c_old_b_ckpt)
        del c_old_b_ckpt

        _, c_new_b_ckpt = run_chunked_prefill(model, base_model, ids_e[:, p0:b_end], p0, c_work, 2048, use_l2k)
        block_fwd_ms = (sync_dev(device) - t_blk0) * 1000.0
        s_new_b_ckpt = snap_states(c_new_b_ckpt)
        new_blk_kv = extract_kv_slice(c_new_b_ckpt, p0, b_end)
        del c_new_b_ckpt

        num_sq, den_sq = 0.0, 0.0
        for ro_t, ro_c in zip(s_old_b_true["rec"], s_old_b_ckpt["rec"]):
            if ro_t is not None and ro_c is not None:
                diff = ro_c.double() - ro_t.double()
                num_sq += float((diff * diff).sum().item())
                den_sq += float((ro_t.double() * ro_t.double()).sum().item())
        state_quant_rel_err = math.sqrt(num_sq / max(1e-18, den_sq))

        delta = compute_state_delta_fp64(s_old_b_ckpt, s_new_b_ckpt)
        c_ctx = move_cache_device(c_ctx_old_cpu, device)
        t_inj0 = sync_dev(device)
        apply_jlocal_scalar_update(c_ctx, delta, hs["gamma_scalar"], b_end, T_ctx, w_conv, use_decay=True)
        inject_ms = (sync_dev(device) - t_inj0) * 1000.0

        t_kv0 = sync_dev(device)
        for i_l, sl in enumerate(new_blk_kv):
            if sl is not None:
                c_ctx.layers[i_l].keys[:, :, p0:b_end, :].copy_(sl[0])
                c_ctx.layers[i_l].values[:, :, p0:b_end, :].copy_(sl[1])
        kv_ms = (sync_dev(device) - t_kv0) * 1000.0

        ev = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_ctx, T_ctx, ids_q, use_l2k, max_new_tokens=8)
        sc = score_condition(ev, new_val, old_val, f_e, f_o, lg_ora)
        del c_ctx

        kv_only_res = None
        if include_kv_only:
            c_kv = move_cache_device(c_ctx_old_cpu, device)
            for i_l, sl in enumerate(new_blk_kv):
                if sl is not None:
                    c_kv.layers[i_l].keys[:, :, p0:b_end, :].copy_(sl[0])
                    c_kv.layers[i_l].values[:, :, p0:b_end, :].copy_(sl[1])
            ev_kv = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_kv, T_ctx, ids_q, use_l2k, max_new_tokens=8)
            sc_kv = score_condition(ev_kv, new_val, old_val, f_e, f_o, lg_ora)
            del c_kv
            kv_only_res = {
                "em": bool(sc_kv["contains_new"]),
                "old_leak": bool(sc_kv["contains_old"]),
                "kl": float(sc_kv["kl_vs_oneshot"]),
                "max_abs_logit_diff": float(sc_kv["max_logit_diff_vs_oneshot"]),
            }

    ld = float(sc["max_logit_diff_vs_oneshot"])
    kl = float(sc["kl_vs_oneshot"])
    return {
        "em": bool(sc["contains_new"]),
        "old_leak": bool(sc["contains_old"]),
        "top1_eq_ora": bool(sc["top1_eq_oracle"]),
        "kl": kl,
        "max_abs_logit_diff": ld,
        "g1b_strict": bool(kl <= 1e-4),
        "g1b_relaxed": bool((kl <= 1e-4) or (ld <= 0.28125 and kl <= 1.5e-3)),
        "tail_ld_gt_05": bool(ld > 0.5),
        "state_quant_rel_err": float(state_quant_rel_err),
        "ckpt_bytes": int(ckpt_bytes),
        "replay_tokens": int(p0 - c_pos),
        "substeps_ms": {
            "h2d": float(h2d_ms),
            "dequant": float(dq_ms),
            "replay": float(replay_ms),
            "block_fwd": float(block_fwd_ms),
            "inject_kv": float(inject_ms + kv_ms),
            "total_edit": float(h2d_ms + dq_ms + replay_ms + block_fwd_ms + inject_ms + kv_ms),
        },
        "kv_only": kv_only_res,
    }


def _agg_cells(rows: List[Dict[str, Any]], include_g1b: bool) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for cell in sorted({r["cell"] for r in rows}):
        sub = [r for r in rows if r["cell"] == cell]
        n = len(sub)
        entry: Dict[str, Any] = {
            "n": n,
            "stride": sub[0]["stride"],
            "dtype": sub[0]["dtype"],
            "em_rate": sum(1 for r in sub if r["em"]) / n,
            "em_ci95": wilson_ci(sum(1 for r in sub if r["em"]), n),
            "old_leak_rate": sum(1 for r in sub if r["old_leak"]) / n,
            "tail_ld_gt_05_rate": sum(1 for r in sub if r["tail_ld_gt_05"]) / n,
            "mean_kl": sum(r["kl"] for r in sub) / n,
            "median_kl": sorted(r["kl"] for r in sub)[n // 2],
            "max_kl": max(r["kl"] for r in sub),
            "mean_logit_diff": sum(r["max_abs_logit_diff"] for r in sub) / n,
            "max_logit_diff": max(r["max_abs_logit_diff"] for r in sub),
            "mean_state_quant_rel_err": sum(r["state_quant_rel_err"] for r in sub) / n,
            "mean_replay_tokens": sum(r["replay_tokens"] for r in sub) / n,
            "mean_substeps_ms": {
                k: sum(r["substeps_ms"][k] for r in sub) / n
                for k in ("h2d", "dequant", "replay", "block_fwd", "inject_kv", "total_edit")
            },
        }
        if include_g1b:
            entry["g1b_strict_rate"] = sum(1 for r in sub if r["g1b_strict"]) / n
            entry["g1b_strict_ci95"] = wilson_ci(sum(1 for r in sub if r["g1b_strict"]), n)
            entry["g1b_relaxed_rate"] = sum(1 for r in sub if r["g1b_relaxed"]) / n
        agg[cell] = entry
    kv_s = [r["kv_only"] for r in rows if r.get("kv_only") is not None]
    if kv_s:
        agg["kv_only_blk_baseline"] = {
            "n": len(kv_s),
            "em_rate": sum(1 for x in kv_s if x["em"]) / len(kv_s),
            "old_leak_rate": sum(1 for x in kv_s if x["old_leak"]) / len(kv_s),
            "tail_ld_gt_05_rate": sum(1 for x in kv_s if x["max_abs_logit_diff"] > 0.5) / len(kv_s),
            "mean_kl": sum(x["kl"] for x in kv_s) / len(kv_s),
        }
    return agg


def run_g1b_stress_suite(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int],
    n_linear: int, d_k: int, w_conv: int, n_layers: int,
    device: str, use_l2k: bool, n_samples: int, T_tgt: int,
    grid_specs: List[Tuple[int, str]], jsonl_path: Path, tag: str,
) -> Dict[str, Any]:
    bank = generate_e5_sample_bank(n_samples=max(n_samples, 30), depths=[0.25, 0.50, 0.75], seed=202605)
    done = load_done_keys(jsonl_path)

    for idx_s in range(n_samples):
        rk_base = f"{tag}:g1b_stress:{idx_s}"
        if all(f"{rk_base}:C{C}_{dt}" in done for C, dt in grid_specs):
            continue
        smp_meta = bank[idx_s]
        p = build_e5_prompt(tok, smp_meta, target_length=T_tgt, n_linear=n_linear, block_horizon="deep", device=device)
        ids_o, ids_e, ids_q = p["ids_o"], p["ids_e"], p["ids_q"]
        p0, b_end, T_ctx = p["p0"], p["b_end"], p["T_ctx"]

        with torch.no_grad():
            _, c_ora = run_chunked_prefill(model, base_model, ids_e, 0, None, 2048, use_l2k)
            ev_ora = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_ora, T_ctx, ids_q, use_l2k, max_new_tokens=8)
            lg_ora = ev_ora["first_logits"].detach().clone()
            del c_ora

            _, c_full_old = run_chunked_prefill(model, base_model, ids_o[:, :b_end], 0, None, 2048, use_l2k)
            s_old_b_true = snap_states(c_full_old)
            with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as col:
                _, c_full_old = run_chunked_prefill(model, base_model, ids_o[:, b_end:T_ctx], b_end, c_full_old, 2048, use_l2k)
            hs = col.finalize(expected_suffix_len=T_ctx - b_end, device=device)
            c_ctx_old_cpu = move_cache_device(c_full_old, "cpu")
            del c_full_old

        for C in sorted({cx for cx, _ in grid_specs}):
            c_pos = (p0 // C) * C
            if c_pos > 0:
                with torch.no_grad():
                    _, c_ck = run_chunked_prefill(model, base_model, ids_o[:, :c_pos], 0, None, 2048, use_l2k)
                    snap_ck = snap_states(c_ck)
                    kv_pre_cpu = move_cache_device(c_ck, "cpu")
                    del c_ck
            else:
                snap_ck, kv_pre_cpu = None, None

            for C_spec, dt_mode in grid_specs:
                if C_spec != C:
                    continue
                rk = f"{rk_base}:C{C}_{dt_mode}"
                if rk in done:
                    continue
                res = _eval_ckpt_cell(
                    model, base_model, tok, eos_ids, n_linear, d_k, w_conv, n_layers, device, use_l2k,
                    ids_o, ids_e, ids_q, p0, b_end, T_ctx, smp_meta["old"], smp_meta["new"], lg_ora,
                    s_old_b_true, c_ctx_old_cpu, hs, c_pos, snap_ck, kv_pre_cpu, dt_mode,
                )
                res.update({
                    "record_key": rk, "suite": "g1b_stress", "sample_idx": idx_s,
                    "depth": float(smp_meta["depth"]), "tpl_id": int(smp_meta["tpl_id"]),
                    "stride": C, "dtype": dt_mode, "cell": f"C{C}_{dt_mode}",
                })
                append_record_jsonl(jsonl_path, res)
                done.add(rk)
            del kv_pre_cpu
        del c_ctx_old_cpu
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l).get("suite") == "g1b_stress"]
    return _agg_cells(rows, include_g1b=True)


def run_collision_stress_suite(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int],
    n_linear: int, d_k: int, w_conv: int, n_layers: int,
    device: str, use_l2k: bool, n_cases: int,
    grid_specs: List[Tuple[int, str]], jsonl_path: Path, tag: str,
) -> Dict[str, Any]:
    cases = _build_collision_cases(tok, n_linear, n_cases=n_cases)
    done = load_done_keys(jsonl_path)

    for c in cases:
        idx_s = c["case_id"]
        rk_base = f"{tag}:col_stress:{idx_s}"
        if all(f"{rk_base}:C{C}_{dt}" in done for C, dt in grid_specs):
            continue
        al = c["al"]
        ids_o = torch.tensor([al["ids_o_list"]], device=device)
        ids_e = torch.tensor([al["ids_e_list"]], device=device)
        ids_q = torch.tensor([al["q_ids_list"]], device=device)
        p0, b_end, T_ctx = al["p0"], al["b_old"], al["T_old"]

        with torch.no_grad():
            _, c_ora = run_chunked_prefill(model, base_model, ids_e, 0, None, 2048, use_l2k)
            ev_ora = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_ora, T_ctx, ids_q, use_l2k, max_new_tokens=8)
            lg_ora = ev_ora["first_logits"].detach().clone()
            del c_ora

            _, c_full_old = run_chunked_prefill(model, base_model, ids_o[:, :b_end], 0, None, 2048, use_l2k)
            s_old_b_true = snap_states(c_full_old)
            with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as col:
                _, c_full_old = run_chunked_prefill(model, base_model, ids_o[:, b_end:T_ctx], b_end, c_full_old, 2048, use_l2k)
            hs = col.finalize(expected_suffix_len=T_ctx - b_end, device=device)
            c_ctx_old_cpu = move_cache_device(c_full_old, "cpu")
            del c_full_old

        for C in sorted({cx for cx, _ in grid_specs}):
            c_pos = (p0 // C) * C
            if c_pos > 0:
                with torch.no_grad():
                    _, c_ck = run_chunked_prefill(model, base_model, ids_o[:, :c_pos], 0, None, 2048, use_l2k)
                    snap_ck = snap_states(c_ck)
                    kv_pre_cpu = move_cache_device(c_ck, "cpu")
                    del c_ck
            else:
                snap_ck, kv_pre_cpu = None, None

            for C_spec, dt_mode in grid_specs:
                if C_spec != C:
                    continue
                rk = f"{rk_base}:C{C}_{dt_mode}"
                if rk in done:
                    continue
                res = _eval_ckpt_cell(
                    model, base_model, tok, eos_ids, n_linear, d_k, w_conv, n_layers, device, use_l2k,
                    ids_o, ids_e, ids_q, p0, b_end, T_ctx, c["old_val"], c["new_val"], lg_ora,
                    s_old_b_true, c_ctx_old_cpu, hs, c_pos, snap_ck, kv_pre_cpu, dt_mode,
                    include_kv_only=(C == grid_specs[0][0] and dt_mode == grid_specs[0][1]),
                )
                res.update({
                    "record_key": rk, "suite": "collision_stress", "sample_idx": idx_s,
                    "stride": C, "dtype": dt_mode, "cell": f"C{C}_{dt_mode}",
                })
                append_record_jsonl(jsonl_path, res)
                done.add(rk)
            del kv_pre_cpu
        del c_ctx_old_cpu
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l).get("suite") == "collision_stress"]
    return _agg_cells(rows, include_g1b=False)


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
    nvme = run_nvme_swap_microbench(
        {"dry2b": {"n_layers": 4, "n_linear": 3, "H_v": 2, "H_k": 2, "d_k": 32, "d_v": 32, "w_conv": 4}},
        "cpu", out_dir, n_repeats=2,
    )
    dry_jsonl = out_dir / "w12_c_nvme_int8_dry.jsonl"
    if dry_jsonl.exists():
        dry_jsonl.unlink()
    g1b = run_g1b_stress_suite(
        model, base_model, tok, eos_ids, 3, 32, 4, 4, "cpu", False,
        n_samples=1, T_tgt=256, grid_specs=[(64, "fp32_rec"), (64, "int8")], jsonl_path=dry_jsonl, tag="dry",
    )
    col = run_collision_stress_suite(
        model, base_model, tok, eos_ids, 3, 32, 4, 4, "cpu", False,
        n_cases=1, grid_specs=[(64, "fp32_rec"), (64, "int8")], jsonl_path=dry_jsonl, tag="dry",
    )
    dry_jsonl.unlink(missing_ok=True)
    print("[DRY-RUN PASS] w12_c_nvme_int8_stress.py verified:", list(nvme.keys()), list(g1b.keys()), list(col.keys()), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--tag", type=str, default="2b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default=str(SCRIPT_DIR))
    parser.add_argument("--n-g1b", type=int, default=30)
    parser.add_argument("--n-col", type=int, default=30)
    parser.add_argument("--g1b-ctx", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--synthetic-dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.synthetic_dry_run:
        run_synthetic_dry_run(out_dir)
        return

    n_g1b = 2 if args.smoke else args.n_g1b
    n_col = 2 if args.smoke else args.n_col

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    t_cfg = getattr(cfg, "text_config", cfg)
    n_linear = count_linear_layers(t_cfg)
    n_layers = int(getattr(t_cfg, "num_hidden_layers", 24))
    d_k = int(getattr(t_cfg, "linear_key_head_dim", 128))
    w_conv = int(getattr(t_cfg, "linear_conv_kernel_dim", 4))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True, local_files_only=args.local_files_only,
    ).eval()
    base_model = getattr(model.model, "language_model", model.model)
    eos_ids = get_eos_ids(model, tok)
    use_l2k = has_logits_to_keep_arg(model)

    model_shapes = {
        "qwen35_2b": {"n_layers": 24, "n_linear": 18, "H_v": 16, "H_k": 16, "d_k": 128, "d_v": 128, "w_conv": 4},
        "qwen35_9b": {"n_layers": 32, "n_linear": 24, "H_v": 32, "H_k": 16, "d_k": 128, "d_v": 128, "w_conv": 4},
    }
    print(f"[W12-4 Controlled NVMe Micro-Bench (2B/9B only)] Running on {args.device}...", flush=True)
    nvme_bench = run_nvme_swap_microbench(model_shapes, args.device, out_dir, n_repeats=5)
    nvme_meta = {
        "hardware_dependent_non_cross_machine_reproducible": True,
        "scope_note": (
            "硬件相关、不可跨机复现（Hardware-dependent, non-cross-machine reproducible）: "
            "controlled single-checkpoint cold/warm NVMe->Pinned->Async H2D restore bandwidth micro-benchmark "
            "for 2B/9B checkpoints only; 27B 128K full NVMe I/O intentionally excluded."
        ),
        "device_name": torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu",
    }

    grid_g1b = [
        (512, "fp32_rec"),
        (512, "int8"),
        (1024, "bf16"),
        (1024, "int8"),
        (2048, "int8"),
    ]
    grid_col = [
        (512, "fp32_rec"),
        (512, "int8"),
        (1024, "bf16"),
        (1024, "int8"),
        (2048, "int8"),
    ]
    jsonl_path = out_dir / f"w12_c_nvme_int8_{args.tag}_records.jsonl"
    print(f"[W12-4 G1b Stress] model={args.model} n={n_g1b} T={args.g1b_ctx}...", flush=True)
    g1b_res = run_g1b_stress_suite(
        model, base_model, tok, eos_ids, n_linear, d_k, w_conv, n_layers,
        args.device, use_l2k, n_g1b, args.g1b_ctx, grid_g1b, jsonl_path, args.tag,
    )
    print(f"[W12-4 Collision Stress] model={args.model} n={n_col}...", flush=True)
    col_res = run_collision_stress_suite(
        model, base_model, tok, eos_ids, n_linear, d_k, w_conv, n_layers,
        args.device, use_l2k, n_col, grid_col, jsonl_path, args.tag,
    )

    summ = {
        "model": args.model,
        "tag": args.tag,
        "nvme_microbench_meta": nvme_meta,
        "nvme_swap_benchmark": nvme_bench,
        "g1b_ckpt_only_stress": g1b_res,
        "entity_collision_ckpt_only_stress": col_res,
    }
    summ_path = out_dir / f"w12_c_nvme_int8_{args.tag}_summary.json"
    summ_path.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Wrote {summ_path}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
