"""Option B Exp-3: Coupled Cold-Disk Restore (POSIX_FADV_DONTNEED + mincore) & Non-Duplicate C512/1024/2048 Accuracy Suite.

Addresses Stage 2 decoupling, C512==C1024 collision at T=8K+, and genuine single-timer cold accounting:
  1. Uses calibrated prompt target length (`8704` when nominal `--ctx-len 8192`) and calibrated
     depths (`[0.47, 0.72, 0.94]`) so actual `T_ctx >= 8192` AND `1536 <= (p0 % 2048) < 2048` on
     100% of samples — strictly enforcing `0 < C2048 < C1024 < C512 < p0` across ALL pairs.
  2. Couples real cold disk I/O directly to accuracy evaluation:
     - Primary path (when `--measure-prefix-kv-disk-cold` is enabled): evicts BOTH the quantized
       recurrent/conv checkpoint file AND the `[0:c_pos)` full-attention prefix-KV file via
       `POSIX_FADV_DONTNEED`, verifies `libc.mincore() == 0` resident pages on both files before
       the timer starts, and inside ONE continuous single timer cold-reads BOTH files from disk,
       transfers to GPU, wires the disk-loaded GPU prefix-KV views and dequantized states directly
       into a tensor-free `DynamicCache` skeleton (zero RAM prefix-KV copy), runs `[c_pos:p0)`
       replay, `block_fwd`, `O_local` injection, and greedy decode (`coupled_cold_disk_prefix_kv_single_timer_ms`).
     - Paired RAM-prefix-KV control: re-evicts the state checkpoint (`mincore() == 0`) and executes
       a separate continuous single-timer pass loading `[0:c_pos)` prefix KV from host RAM
       (`coupled_cold_ram_prefix_kv_single_timer_ms`), verifying numerical agreement against the
       disk-wired prefix-KV pass with zero post-hoc arithmetic substitution.
  3. Persists raw `generated_text`, `old_val`, `new_val`, `contains_new`, `exact_match`, `old_leak`,
     runtime metadata (`os_name`, `device_name`, `torch_version`, `transformers_version`), and
     filesystem provenance (`df -T`, `findmnt`, `/proc/mounts`, `lsblk`, `backing_is_verified_nvme`).

Supports `--synthetic-dry-run` on CPU (`CUDA_VISIBLE_DEVICES=""`).
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import json
import math
import mmap
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HOME", r"D:\applicationData\HuggingFace")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers as _tf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from w5_9b_e5_runner import (  # noqa: E402
    GDNSuffixHookCollector,
    append_record_jsonl,
    apply_jlocal_scalar_update,
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
    patch_kv_slice,
    run_chunked_prefill,
    score_condition,
    snap_states,
)
from w11_systems_thinning_throughput_9b import convert_states_precision, sync_dev  # noqa: E402
from w11_varlen_e10_e12_detector import load_done_keys, wilson_ci  # noqa: E402
from w12_c_nvme_int8_stress import (  # noqa: E402
    _async_h2d_and_dequant_gpu,
    _nvme_read_into_pinned,
    _pack_to_flat_bytes,
    _try_drop_os_page_cache,
)


def _normalize_exact(text: str) -> str:
    s = (text or "").strip().strip("`\"'.,;:!?()[]{}")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _strict_exact_match(text: str, target: str) -> bool:
    return _normalize_exact(text) == _normalize_exact(target)


def _upper_order_p50_p95(xs: List[float]) -> Dict[str, float]:
    """Upper-order-statistic p50/p95 matching Stage C W13 & posthoc (`sorted[n//2]`, `sorted[min(n-1, int(0.95*n))]`)."""
    if not xs:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(float(x) for x in xs)
    n = len(s)
    return {
        "mean": round(sum(s) / n, 3),
        "p50": round(s[n // 2], 3),
        "p95": round(s[min(n - 1, int(0.95 * n))], 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
    }


def check_mincore_residency(path: Path, nbytes: int) -> Dict[str, Any]:
    """Use POSIX mmap + libc mincore() to measure resident OS page count cleanly without buffer writability errors."""
    if os.name != "posix" or nbytes <= 0:
        return {"mincore_supported": False, "resident_pages": None, "total_pages": None}
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        mincore = libc.mincore
        mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
        mincore.restype = ctypes.c_int
        c_mmap = libc.mmap
        c_mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
        c_mmap.restype = ctypes.c_void_p
        c_munmap = libc.munmap
        c_munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        c_munmap.restype = ctypes.c_int

        page_sz = mmap.PAGESIZE
        n_pages = (nbytes + page_sz - 1) // page_sz
        vec = (ctypes.c_ubyte * n_pages)()
        prot_read = 1
        map_private = 2
        with open(path, "rb") as fh:
            addr = c_mmap(None, ctypes.c_size_t(nbytes), prot_read, map_private, fh.fileno(), 0)
            if not addr or addr == ctypes.c_void_p(-1).value:
                err = ctypes.get_errno()
                return {"mincore_supported": False, "resident_pages": None, "total_pages": n_pages, "mmap_errno": int(err)}
            try:
                rc = mincore(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes), vec)
                err = ctypes.get_errno()
            finally:
                c_munmap(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes))
        if rc != 0:
            return {"mincore_supported": False, "resident_pages": None, "total_pages": n_pages, "errno": int(err)}
        res = sum(1 for v in vec if (v & 1) != 0)
        return {
            "mincore_supported": True,
            "resident_pages": int(res),
            "total_pages": int(n_pages),
            "resident_frac": float(res / max(1, n_pages)),
        }
    except Exception as exc:
        return {"mincore_supported": False, "resident_pages": None, "total_pages": None, "error": repr(exc)}


def get_dir_fs_info(path: Path) -> Dict[str, Any]:
    """Capture filesystem type (`df -T`), `/proc/mounts` entry, overlay upperdir backing device, and `lsblk` metadata."""
    if os.name != "posix":
        return {
            "checkpoint_dir_abs": str(path.resolve()),
            "df_t": "windows_ntfs",
            "findmnt": "windows_ntfs",
            "mount_entry": "windows_ntfs",
            "overlay_upperdir_probe": {"backing_is_verified_nvme": False, "backing_source": "windows_ntfs"},
            "lsblk": "windows_ntfs",
        }
    df_out = "unknown"
    try:
        out = subprocess.check_output(["df", "-T", str(path)], text=True)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if lines:
            df_out = lines[-1]
    except Exception:
        pass
    mount_entry = "unknown"
    try:
        rp = str(path.resolve())
        best_mp = ""
        for line in Path("/proc/mounts").read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 3 and rp.startswith(parts[1]) and len(parts[1]) >= len(best_mp):
                best_mp = parts[1]
                mount_entry = line
    except Exception:
        pass
    lsblk_out = "unknown"
    try:
        lsblk_out = subprocess.check_output(["lsblk", "-no", "NAME,KNAME,PKNAME,TYPE,FSTYPE,MOUNTPOINT,MODEL,TRAN,ROTA"], text=True).strip()
    except Exception:
        try:
            lsblk_out = subprocess.check_output(["lsblk", "-no", "NAME,TYPE,FSTYPE,MOUNTPOINT,ROTA"], text=True).strip()
        except Exception:
            pass
    findmnt_out = "unknown"
    try:
        findmnt_out = subprocess.check_output(["findmnt", "-no", "SOURCE,FSTYPE,TARGET,OPTIONS", "-T", str(path)], text=True).strip()
    except Exception:
        pass

    upperdir_probe: Dict[str, Any] = {"upperdir_path": None, "backing_source": "unknown", "backing_is_verified_nvme": False}
    if "upperdir=" in mount_entry:
        for tok_opt in mount_entry.split():
            if "upperdir=" in tok_opt:
                for kv in tok_opt.split(","):
                    if kv.startswith("upperdir="):
                        upath = kv.split("=", 1)[1]
                        upperdir_probe["upperdir_path"] = upath
                        up = Path(upath)
                        upperdir_probe["upperdir_exists_in_container"] = up.exists()
                        if up.exists():
                            try:
                                upperdir_probe["upperdir_findmnt"] = subprocess.check_output(
                                    ["findmnt", "-no", "SOURCE,FSTYPE,TARGET", "-T", upath], text=True
                                ).strip()
                                upperdir_probe["upperdir_df_t"] = subprocess.check_output(["df", "-T", upath], text=True).strip()
                            except Exception as exc:
                                upperdir_probe["upperdir_probe_error"] = repr(exc)
                        else:
                            upperdir_probe["note"] = "Host containerd overlay upperdir path is outside container mount namespace; physical block backing cannot be proven inside guest container."
    try:
        mounts_nvme = [l for l in Path("/proc/mounts").read_text(encoding="utf-8", errors="ignore").splitlines() if "nvme" in l]
        upperdir_probe["proc_mounts_nvme_entries"] = mounts_nvme
    except Exception:
        pass
    src_str = str(upperdir_probe.get("upperdir_findmnt", ""))
    if "/dev/nvme" in src_str or "/dev/nvme" in findmnt_out:
        upperdir_probe["backing_is_verified_nvme"] = True
        upperdir_probe["backing_source"] = src_str or findmnt_out

    return {
        "checkpoint_dir_abs": str(path.resolve()),
        "df_t": df_out,
        "findmnt": findmnt_out,
        "mount_entry": mount_entry,
        "overlay_upperdir_probe": upperdir_probe,
        "lsblk": lsblk_out,
    }


def _pack_prefix_kv_to_flat_bytes(kv_cache_cpu: Any) -> Tuple[Any, List[Dict[str, Any]]]:
    """Serialize all full-attention key/value tensors from prefix cache `[0:c_pos)` into contiguous uint8 bytes."""
    import numpy as np

    chunks: List[Any] = []
    manifest: List[Dict[str, Any]] = []
    offset = 0
    for li, layer in enumerate(kv_cache_cpu.layers):
        for attr in ("keys", "values"):
            t = getattr(layer, attr, None)
            if torch.is_tensor(t) and t.numel() > 0:
                t_c = t.detach().contiguous().cpu()
                raw = t_c.view(torch.uint8).numpy().reshape(-1)
                nbytes = int(raw.nbytes)
                chunks.append(raw)
                manifest.append({
                    "layer": int(li),
                    "attr": attr,
                    "shape": list(t_c.shape),
                    "dtype": str(t_c.dtype),
                    "offset": int(offset),
                    "nbytes": nbytes,
                })
                offset += nbytes
    flat = np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.uint8)
    return flat, manifest


def _build_cache_skeleton(template_cache: Any) -> Any:
    """Create a tensor-free DynamicCache metadata skeleton (holds zero KV or recurrent/conv tensors)."""
    skel = copy.copy(template_cache)
    skel_layers = []
    for layer in template_cache.layers:
        nl = copy.copy(layer)
        if hasattr(nl, "keys"):
            nl.keys = None
        if hasattr(nl, "values"):
            nl.values = None
        if hasattr(nl, "conv_states") and isinstance(nl.conv_states, dict):
            nl.conv_states = {}
        if hasattr(nl, "recurrent_states") and isinstance(nl.recurrent_states, dict):
            nl.recurrent_states = {}
        for attr in (
            "has_previous_state",
            "is_conv_states_initialized",
            "is_recurrent_states_initialized",
            "conv_kernel_size",
        ):
            if hasattr(nl, attr) and isinstance(getattr(nl, attr), dict):
                setattr(nl, attr, dict(getattr(nl, attr)))
        skel_layers.append(nl)
    skel.layers = skel_layers
    return skel


def _assemble_cache_from_disk_gpu(
    cache_skeleton: Any,
    rest_states: Dict[str, List[Optional[torch.Tensor]]],
    gpu_u8_pkv: Optional[torch.Tensor],
    kv_man: List[Dict[str, Any]],
    device: str,
) -> Any:
    """Wire disk-loaded GPU state tensors and disk-loaded GPU `[0:c_pos)` prefix-KV tensors directly into a live cache."""
    dev_obj = torch.device(device)
    c_out = copy.copy(cache_skeleton)
    new_layers = []
    for i, layer in enumerate(cache_skeleton.layers):
        nl = copy.copy(layer)
        if hasattr(nl, "device"):
            nl.device = dev_obj
        for attr in (
            "has_previous_state",
            "is_conv_states_initialized",
            "is_recurrent_states_initialized",
            "conv_kernel_size",
        ):
            if hasattr(nl, attr) and isinstance(getattr(nl, attr), dict):
                setattr(nl, attr, dict(getattr(nl, attr)))
        if hasattr(nl, "conv_states"):
            c_t = rest_states["conv"][i]
            nl.conv_states = {0: c_t.clone()} if c_t is not None else {}
        if hasattr(nl, "recurrent_states"):
            r_t = rest_states["rec"][i]
            nl.recurrent_states = {0: r_t.clone()} if r_t is not None else {}
        new_layers.append(nl)
    if gpu_u8_pkv is not None and kv_man:
        for ent in kv_man:
            dt = getattr(torch, ent["dtype"].split(".")[-1])
            sl = gpu_u8_pkv[ent["offset"] : ent["offset"] + ent["nbytes"]].view(dt).view(*ent["shape"]).clone()
            setattr(new_layers[ent["layer"]], ent["attr"], sl)
    c_out.layers = new_layers
    return c_out


def _evict_file_and_verify(path: Path, nbytes: int, label: str) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    mc_before = check_mincore_residency(path, nbytes)
    with open(path, "rb") as f_drop:
        dropped = _try_drop_os_page_cache(f_drop.fileno(), nbytes)
    mc_after = check_mincore_residency(path, nbytes)
    if os.name == "posix":
        assert mc_after.get("mincore_supported", False), f"POSIX mincore() unsupported on {label} ({path}): {mc_after}"
        assert mc_after.get("resident_pages", -1) == 0, f"POSIX_FADV_DONTNEED failed to evict {label}: {mc_after}"
    return bool(dropped), mc_before, mc_after


def coupled_eval_cell(
    model: Any,
    base_model: Any,
    tok: Any,
    eos_ids: List[int],
    n_linear: int,
    w_conv: int,
    n_layers: int,
    device: str,
    use_l2k: bool,
    tmp_dir: Path,
    ids_o: torch.Tensor,
    ids_e: torch.Tensor,
    ids_q: torch.Tensor,
    p0: int,
    b_end: int,
    T_ctx: int,
    old_val: str,
    new_val: str,
    lg_ora: torch.Tensor,
    s_old_b_true: Dict[str, List[Optional[torch.Tensor]]],
    c_ctx_old_cpu: Any,
    hs: Dict[str, Any],
    c_pos: int,
    snap_ck: Dict[str, List[Optional[torch.Tensor]]],
    kv_pre_template_cpu: Any,
    dt_mode: str,
    tag_id: str,
    measure_prefix_kv_disk_cold: bool = False,
) -> Dict[str, Any]:
    assert c_pos > 0 and snap_ck is not None, f"Coupled suite requires c_pos > 0, got {c_pos}"
    f_o = tok.encode(old_val, add_special_tokens=False)[0]
    f_e = tok.encode(new_val, add_special_tokens=False)[0]

    packed, ckpt_bytes = convert_states_precision(snap_ck, dt_mode, device)
    buf, manifest = _pack_to_flat_bytes(packed, dt_mode)
    nbytes = int(buf.nbytes)
    bin_path = tmp_dir / f"ckpt_{tag_id}.bin"
    with open(bin_path, "wb") as fw:
        fw.write(buf.tobytes())
        fw.flush()
        os.fsync(fw.fileno())

    kv_path: Optional[Path] = None
    kv_man: List[Dict[str, Any]] = []
    pkv_nbytes = 0
    cache_skeleton: Optional[Any] = None
    pkv_dropped = False
    pkv_mc_before: Dict[str, Any] = {"mincore_supported": False}
    pkv_mc_after: Dict[str, Any] = {"mincore_supported": False}

    if measure_prefix_kv_disk_cold:
        flat_pkv, kv_man = _pack_prefix_kv_to_flat_bytes(kv_pre_template_cpu)
        pkv_nbytes = int(flat_pkv.nbytes)
        cache_skeleton = _build_cache_skeleton(kv_pre_template_cpu)
        if pkv_nbytes > 0:
            kv_path = tmp_dir / f"prefix_kv_{tag_id}.bin"
            with open(kv_path, "wb") as fw:
                fw.write(flat_pkv.tobytes())
                fw.flush()
                os.fsync(fw.fileno())

    def _run_edit_and_decode_from_c_work(c_work: Any) -> Tuple[Dict[str, Any], Dict[str, List[Optional[torch.Tensor]]], Dict[str, float]]:
        t_rep0 = sync_dev(device)
        if p0 > c_pos:
            _, c_work = run_chunked_prefill(model, base_model, ids_o[:, c_pos:p0], c_pos, c_work, 2048, use_l2k)
        replay_ms = (sync_dev(device) - t_rep0) * 1000.0

        t_blk0 = sync_dev(device)
        c_p0_old = clone_cache(c_work)
        _, c_old_b_ckpt = run_chunked_prefill(model, base_model, ids_o[:, p0:b_end], p0, c_p0_old, 2048, use_l2k)
        s_old_b_ckpt = snap_states(c_old_b_ckpt)
        del c_old_b_ckpt

        _, c_new_b_ckpt = run_chunked_prefill(model, base_model, ids_e[:, p0:b_end], p0, c_work, 2048, use_l2k)
        s_new_b_ckpt = snap_states(c_new_b_ckpt)
        new_blk_kv = extract_kv_slice(c_new_b_ckpt, p0, b_end)
        block_fwd_ms = (sync_dev(device) - t_blk0) * 1000.0
        del c_new_b_ckpt, c_work

        t_diff0 = sync_dev(device)
        delta = compute_state_delta_fp64(s_old_b_ckpt, s_new_b_ckpt)
        diff_delta_ms = (sync_dev(device) - t_diff0) * 1000.0

        t_kvh0 = sync_dev(device)
        c_ctx = move_cache_device(c_ctx_old_cpu, device)
        kv_ctx_h2d_ms = (sync_dev(device) - t_kvh0) * 1000.0

        t_inj0 = sync_dev(device)
        apply_jlocal_scalar_update(c_ctx, delta, hs["gamma_scalar"], b_end, T_ctx, w_conv, use_decay=True)
        patch_kv_slice(c_ctx, new_blk_kv, p0, b_end)
        inj_kv_ms = (sync_dev(device) - t_inj0) * 1000.0

        t_dec0 = sync_dev(device)
        ev = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_ctx, T_ctx, ids_q, use_l2k, max_new_tokens=8)
        eval_decode_ms = (sync_dev(device) - t_dec0) * 1000.0
        del c_ctx
        return ev, s_old_b_ckpt, {
            "replay": float(replay_ms),
            "block_fwd": float(block_fwd_ms),
            "diff_delta": float(diff_delta_ms),
            "kv_ctx_h2d": float(kv_ctx_h2d_ms),
            "inject_kv": float(inj_kv_ms),
            "eval_decode": float(eval_decode_ms),
        }

    # --- Pass A: Continuous single-timer Cold-Disk State + RAM `[0:c_pos)` Prefix KV ---
    dropped_ram, mc_before_ram, mc_final_ram = _evict_file_and_verify(bin_path, nbytes, "state_ckpt_ram_pass")
    with torch.no_grad():
        t_ram0 = sync_dev(device)
        pin_u8, disk_cold_read_ms, rd_dropped = _nvme_read_into_pinned(bin_path, nbytes, drop_cache=False)
        rest, h2d_ms, dq_ms = _async_h2d_and_dequant_gpu(pin_u8, manifest, dt_mode, device, n_layers)
        del pin_u8
        t_pkv0 = sync_dev(device)
        c_work_ram = move_cache_device(kv_pre_template_cpu, device)
        for i, c in enumerate(rest["conv"]):
            if c is not None:
                c_work_ram.layers[i].conv_states[0].copy_(c)
        for i, r in enumerate(rest["rec"]):
            if r is not None:
                c_work_ram.layers[i].recurrent_states[0].copy_(r)
        prefix_kv_h2d_ms = (sync_dev(device) - t_pkv0) * 1000.0
        ev_ram, s_old_b_ckpt_ram, mid_ram = _run_edit_and_decode_from_c_work(c_work_ram)
        coupled_cold_ram_single_timer_ms = (sync_dev(device) - t_ram0) * 1000.0
        del rest

    # --- Pass B (when enabled): Continuous single-timer Cold-Disk State + Cold-Disk `[0:c_pos)` Prefix KV ---
    # ZERO post-hoc substitution: disk-loaded prefix-KV GPU views are directly wired into the live decode cache!
    disk_pass_info: Optional[Dict[str, Any]] = None
    ev_primary = ev_ram
    s_old_b_primary = s_old_b_ckpt_ram

    if measure_prefix_kv_disk_cold and kv_path is not None and pkv_nbytes > 0:
        ev_ram_logits_cpu = ev_ram["first_logits"].detach().float().cpu()
        del ev_ram, s_old_b_ckpt_ram
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        dropped_disk_st, mc_before_disk_st, mc_final_disk_st = _evict_file_and_verify(bin_path, nbytes, "state_ckpt_disk_pass")
        pkv_dropped, pkv_mc_before, pkv_mc_after = _evict_file_and_verify(kv_path, pkv_nbytes, "prefix_kv_disk_pass")
        with torch.no_grad():
            t_dsk0 = sync_dev(device)
            pin_st2, st_cold_rd_disk_pass_ms, rd_dr2 = _nvme_read_into_pinned(bin_path, nbytes, drop_cache=False)
            rest2, h2d2_ms, dq2_ms = _async_h2d_and_dequant_gpu(pin_st2, manifest, dt_mode, device, n_layers)
            del pin_st2

            pin_pkv, pkv_cold_rd_ms, _ = _nvme_read_into_pinned(kv_path, pkv_nbytes, drop_cache=False)
            t_pkv_d0 = sync_dev(device)
            gpu_u8_pkv = pin_pkv.to(device=device, non_blocking=True)
            c_work_disk = _assemble_cache_from_disk_gpu(cache_skeleton, rest2, gpu_u8_pkv, kv_man, device)
            del gpu_u8_pkv, pin_pkv, rest2
            pkv_disk_h2d_wire_ms = (sync_dev(device) - t_pkv_d0) * 1000.0

            ev_disk, s_old_b_ckpt_disk, mid_disk = _run_edit_and_decode_from_c_work(c_work_disk)
            coupled_cold_disk_single_timer_ms = (sync_dev(device) - t_dsk0) * 1000.0

        disk_vs_ram_diff = float(
            (ev_disk["first_logits"].detach().float().cpu() - ev_ram_logits_cpu).abs().max().item()
        )
        del ev_ram_logits_cpu
        assert disk_vs_ram_diff <= 1e-4, f"Disk-wired prefix KV diverged from RAM prefix KV: max_diff={disk_vs_ram_diff}"
        ev_primary = ev_disk
        s_old_b_primary = s_old_b_ckpt_disk

        # Warm reads strictly AFTER both cold single-timer passes finish
        pin_pkv_w, pkv_warm_rd_ms, _ = _nvme_read_into_pinned(kv_path, pkv_nbytes, drop_cache=False)
        del pin_pkv_w
        kv_path.unlink(missing_ok=True)

        disk_substeps_sum = float(
            st_cold_rd_disk_pass_ms + h2d2_ms + dq2_ms + pkv_cold_rd_ms + pkv_disk_h2d_wire_ms
            + mid_disk["replay"] + mid_disk["block_fwd"] + mid_disk["diff_delta"]
            + mid_disk["kv_ctx_h2d"] + mid_disk["inject_kv"] + mid_disk["eval_decode"]
        )
        disk_pass_info = {
            "state_fadvise_dropped": bool(dropped_disk_st or rd_dr2),
            "state_mincore_before_drop": mc_before_disk_st,
            "state_mincore_after_drop": mc_final_disk_st,
            "prefix_kv_disk_bytes": int(pkv_nbytes),
            "prefix_kv_fadvise_dropped": bool(pkv_dropped),
            "prefix_kv_mincore_before_drop": pkv_mc_before,
            "prefix_kv_mincore_after_drop": pkv_mc_after,
            "state_cold_read_ms": float(st_cold_rd_disk_pass_ms),
            "state_h2d_ms": float(h2d2_ms),
            "state_dequant_ms": float(dq2_ms),
            "prefix_kv_disk_cold_read_ms": float(pkv_cold_rd_ms),
            "prefix_kv_disk_warm_read_ms": float(pkv_warm_rd_ms),
            "prefix_kv_disk_to_gpu_h2d_and_wire_ms": float(pkv_disk_h2d_wire_ms),
            "substeps_sum_ms": disk_substeps_sum,
            "coupled_cold_disk_prefix_kv_single_timer_ms": float(coupled_cold_disk_single_timer_ms),
            "timer_overhead_gap_disk_ms": float(coupled_cold_disk_single_timer_ms - disk_substeps_sum),
            "disk_vs_ram_prefix_kv_max_logit_diff": disk_vs_ram_diff,
        }

    # Paired warm state read strictly AFTER cold single-timer passes
    pin_warm, disk_warm_read_ms, _ = _nvme_read_into_pinned(bin_path, nbytes, drop_cache=False)
    del pin_warm
    bin_path.unlink(missing_ok=True)

    with torch.no_grad():
        num_sq, den_sq = 0.0, 0.0
        for ro_t, ro_c in zip(s_old_b_true["rec"], s_old_b_primary["rec"]):
            if ro_t is not None and ro_c is not None:
                diff = ro_c.double() - ro_t.double()
                num_sq += float((diff * diff).sum().item())
                den_sq += float((ro_t.double() * ro_t.double()).sum().item())
        state_quant_rel_err = math.sqrt(num_sq / max(1e-18, den_sq))
        sc = score_condition(ev_primary, new_val, old_val, f_e, f_o, lg_ora)

    gen_txt = str(ev_primary.get("text", ""))
    ld = float(sc["max_logit_diff_vs_oneshot"])
    kl = float(sc["kl_vs_oneshot"])
    substeps_linear_only = float(
        disk_cold_read_ms + h2d_ms + dq_ms + mid_ram["replay"] + mid_ram["block_fwd"] + mid_ram["inject_kv"]
    )
    cold_e2e_ram_substeps_sum = float(
        substeps_linear_only + prefix_kv_h2d_ms + mid_ram["diff_delta"] + mid_ram["kv_ctx_h2d"] + mid_ram["eval_decode"]
    )
    timer_gap_ram_ms = float(coupled_cold_ram_single_timer_ms - cold_e2e_ram_substeps_sum)

    substeps_dict: Dict[str, Any] = {
        "disk_cold_read": float(disk_cold_read_ms),
        "disk_warm_read": float(disk_warm_read_ms),
        "h2d": float(h2d_ms),
        "dequant": float(dq_ms),
        "prefix_kv_h2d": float(prefix_kv_h2d_ms),
        "replay": float(mid_ram["replay"]),
        "block_fwd": float(mid_ram["block_fwd"]),
        "diff_delta": float(mid_ram["diff_delta"]),
        "kv_ctx_h2d": float(mid_ram["kv_ctx_h2d"]),
        "inject_kv": float(mid_ram["inject_kv"]),
        "eval_decode": float(mid_ram["eval_decode"]),
        "coupled_cold_substeps_ms": substeps_linear_only,
        "coupled_cold_total_ms": substeps_linear_only,
        "coupled_cold_e2e_substeps_sum_ms": cold_e2e_ram_substeps_sum,
        "coupled_cold_ram_prefix_kv_single_timer_ms": float(coupled_cold_ram_single_timer_ms),
        "coupled_cold_e2e_single_timer_ms": float(coupled_cold_ram_single_timer_ms),
        "timer_overhead_gap_ms": timer_gap_ram_ms,
    }
    if disk_pass_info is not None:
        substeps_dict["prefix_kv_disk_cold_read"] = disk_pass_info["prefix_kv_disk_cold_read_ms"]
        substeps_dict["prefix_kv_disk_warm_read"] = disk_pass_info["prefix_kv_disk_warm_read_ms"]
        substeps_dict["prefix_kv_disk_to_gpu_h2d_and_wire"] = disk_pass_info["prefix_kv_disk_to_gpu_h2d_and_wire_ms"]
        substeps_dict["coupled_cold_disk_prefix_kv_substeps_sum_ms"] = disk_pass_info["substeps_sum_ms"]
        substeps_dict["coupled_cold_disk_prefix_kv_single_timer_ms"] = disk_pass_info["coupled_cold_disk_prefix_kv_single_timer_ms"]
        substeps_dict["timer_overhead_gap_disk_ms"] = disk_pass_info["timer_overhead_gap_disk_ms"]

    out_rec: Dict[str, Any] = {
        "c_pos": int(c_pos),
        "p0": int(p0),
        "T_ctx": int(T_ctx),
        "ckpt_bytes": int(ckpt_bytes),
        "replay_tokens": int(p0 - c_pos),
        "old_val": str(old_val),
        "new_val": str(new_val),
        "generated_text": gen_txt,
        "fadvise_dropped": bool(dropped_ram or rd_dropped),
        "mincore_before_drop": mc_before_ram,
        "mincore_after_drop": mc_final_ram,
        "contains_new": bool(sc["contains_new"]),
        "em": bool(sc["contains_new"]),
        "exact_match": bool(_strict_exact_match(gen_txt, new_val)),
        "old_leak": bool(sc["contains_old"]),
        "top1_eq_ora": bool(sc["top1_eq_oracle"]),
        "kl": kl,
        "max_abs_logit_diff": ld,
        "g1b_strict": bool(kl <= 1e-4),
        "g1b_relaxed": bool((kl <= 1e-4) or (ld <= 0.28125 and kl <= 1.5e-3)),
        "state_quant_rel_err": float(state_quant_rel_err),
        "accuracy_sourced_from_disk_wired_prefix_kv": bool(disk_pass_info is not None),
        "substeps_ms": substeps_dict,
    }
    if disk_pass_info is not None:
        out_rec["prefix_kv_disk_bytes"] = disk_pass_info["prefix_kv_disk_bytes"]
        out_rec["prefix_kv_mincore_before_drop"] = disk_pass_info["prefix_kv_mincore_before_drop"]
        out_rec["prefix_kv_mincore_after_drop"] = disk_pass_info["prefix_kv_mincore_after_drop"]
        out_rec["disk_pass_state_mincore_after_drop"] = disk_pass_info["state_mincore_after_drop"]
        out_rec["disk_vs_ram_prefix_kv_max_logit_diff"] = disk_pass_info["disk_vs_ram_prefix_kv_max_logit_diff"]
    return out_rec


def _aggregate_coupled(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    agg_keys = (
        "disk_cold_read", "disk_warm_read", "h2d", "dequant", "prefix_kv_h2d", "replay", "block_fwd",
        "diff_delta", "kv_ctx_h2d", "inject_kv", "eval_decode",
        "coupled_cold_substeps_ms", "coupled_cold_total_ms", "coupled_cold_e2e_substeps_sum_ms",
        "coupled_cold_ram_prefix_kv_single_timer_ms", "coupled_cold_e2e_single_timer_ms", "timer_overhead_gap_ms",
    )
    disk_keys = (
        "prefix_kv_disk_cold_read",
        "prefix_kv_disk_warm_read",
        "prefix_kv_disk_to_gpu_h2d_and_wire",
        "coupled_cold_disk_prefix_kv_substeps_sum_ms",
        "coupled_cold_disk_prefix_kv_single_timer_ms",
        "timer_overhead_gap_disk_ms",
    )
    for cell in sorted({r["cell"] for r in rows}):
        sub = [r for r in rows if r["cell"] == cell]
        n = len(sub)
        mc_pre = [
            float(r["mincore_before_drop"]["resident_frac"])
            for r in sub
            if isinstance(r.get("mincore_before_drop"), dict) and r["mincore_before_drop"].get("mincore_supported")
        ]
        mc_res = [r["mincore_after_drop"]["resident_pages"] for r in sub if r["mincore_after_drop"].get("mincore_supported")]
        has_pkv_disk = all(k in r["substeps_ms"] for r in sub for k in disk_keys)
        mean_subs = {
            k: round(sum(r["substeps_ms"][k] for r in sub) / n, 3)
            for k in agg_keys
        }
        if has_pkv_disk:
            for k in disk_keys:
                mean_subs[k] = round(sum(r["substeps_ms"][k] for r in sub) / n, 3)

        ram_e2e_stats = _upper_order_p50_p95([r["substeps_ms"]["coupled_cold_ram_prefix_kv_single_timer_ms"] for r in sub])
        cell_out: Dict[str, Any] = {
            "n": n,
            "zero_ckpt_rows": sum(1 for r in sub if r["ckpt_bytes"] <= 0),
            "distinct_c_pos_count": len({r["c_pos"] for r in sub}),
            "contains_new_rate": sum(1 for r in sub if r["em"]) / n,
            "exact_match_rate": sum(1 for r in sub if r.get("exact_match", False)) / n,
            "old_leak_rate": sum(1 for r in sub if r["old_leak"]) / n,
            "g1b_strict_rate": sum(1 for r in sub if r["g1b_strict"]) / n,
            "g1b_strict_ci95": wilson_ci(sum(1 for r in sub if r["g1b_strict"]), n),
            "g1b_relaxed_rate": sum(1 for r in sub if r["g1b_relaxed"]) / n,
            "mean_kl": sum(r["kl"] for r in sub) / n,
            "max_kl": max(r["kl"] for r in sub),
            "mean_state_quant_rel_err": sum(r["state_quant_rel_err"] for r in sub) / n,
            "mincore_before_checked_rows": len(mc_pre),
            "mincore_before_resident_frac_mean": (sum(mc_pre) / len(mc_pre)) if mc_pre else None,
            "mincore_checked_rows": len(mc_res),
            "mincore_zero_resident_rows": sum(1 for x in mc_res if x == 0),
            "timer_overhead_gap_ram_row_max_ms": round(max(r["substeps_ms"]["timer_overhead_gap_ms"] for r in sub), 3),
            "ram_prefix_kv_single_timer_ms": ram_e2e_stats,
            "mean_substeps_ms": mean_subs,
        }
        if has_pkv_disk:
            pkv_mc = [
                r["prefix_kv_mincore_after_drop"]["resident_pages"]
                for r in sub
                if isinstance(r.get("prefix_kv_mincore_after_drop"), dict)
                and r["prefix_kv_mincore_after_drop"].get("mincore_supported")
            ]
            st2_mc = [
                r["disk_pass_state_mincore_after_drop"]["resident_pages"]
                for r in sub
                if isinstance(r.get("disk_pass_state_mincore_after_drop"), dict)
                and r["disk_pass_state_mincore_after_drop"].get("mincore_supported")
            ]
            pkv_bytes = [int(r.get("prefix_kv_disk_bytes", 0)) for r in sub]
            cell_out["prefix_kv_disk_bytes_mean"] = round(sum(pkv_bytes) / n, 1)
            cell_out["prefix_kv_disk_mib_row_min"] = round(min(pkv_bytes) / (1024 * 1024), 3)
            cell_out["prefix_kv_disk_mib_row_max"] = round(max(pkv_bytes) / (1024 * 1024), 3)
            cell_out["prefix_kv_mincore_checked_rows"] = len(pkv_mc)
            cell_out["prefix_kv_mincore_zero_resident_rows"] = sum(1 for x in pkv_mc if x == 0)
            cell_out["disk_pass_state_mincore_checked_rows"] = len(st2_mc)
            cell_out["disk_pass_state_mincore_zero_resident_rows"] = sum(1 for x in st2_mc if x == 0)
            cell_out["disk_vs_ram_prefix_kv_max_logit_diff"] = max(
                float(r.get("disk_vs_ram_prefix_kv_max_logit_diff", 0.0)) for r in sub
            )
            cell_out["timer_overhead_gap_disk_row_max_ms"] = round(
                max(r["substeps_ms"]["timer_overhead_gap_disk_ms"] for r in sub), 3
            )
            cell_out["disk_prefix_kv_single_timer_ms"] = _upper_order_p50_p95(
                [r["substeps_ms"]["coupled_cold_disk_prefix_kv_single_timer_ms"] for r in sub]
            )
        out[cell] = cell_out
    return out


def run_suite(
    model: Any,
    base_model: Any,
    tok: Any,
    eos_ids: List[int],
    n_linear: int,
    d_k: int,
    w_conv: int,
    n_layers: int,
    device: str,
    use_l2k: bool,
    n_samples: int,
    t_tgt: int,
    strides: List[int],
    dtypes: List[str],
    depths: List[float],
    jsonl_path: Path,
    tag: str,
    out_dir: Path,
    measure_prefix_kv_disk_cold: bool = False,
) -> Dict[str, Any]:
    bank = generate_e5_sample_bank(n_samples=max(n_samples, 30), depths=depths, seed=202615)
    done = load_done_keys(jsonl_path)
    grid_specs = [(C, dt) for C in strides for dt in dtypes]

    with tempfile.TemporaryDirectory(dir=str(out_dir)) as tmpd:
        tmp_path = Path(tmpd)
        fs_info = get_dir_fs_info(tmp_path)

        for idx_s in range(n_samples):
            rk_base = f"{tag}:coupled:{idx_s}"
            if all(f"{rk_base}:C{C}_{dt}" in done for C, dt in grid_specs):
                continue
            smp = bank[idx_s]
            p = build_e5_prompt(tok, smp, target_length=t_tgt, n_linear=n_linear, block_horizon="deep", device=device)
            ids_o, ids_e, ids_q = p["ids_o"], p["ids_e"], p["ids_q"]
            p0, b_end, T_ctx = p["p0"], p["b_end"], p["T_ctx"]

            c_positions = {C: (p0 // C) * C for C in strides}
            for C, cp in c_positions.items():
                assert cp > 0, f"Sample {idx_s} depth={smp['depth']} yielded c_pos=0 for stride={C}"
            assert len(set(c_positions.values())) == len(strides), (
                f"Sample {idx_s} depth={smp['depth']} p0={p0} has colliding stride checkpoints {c_positions}"
            )

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

            for C in strides:
                c_pos = c_positions[C]
                with torch.no_grad():
                    _, c_ck = run_chunked_prefill(model, base_model, ids_o[:, :c_pos], 0, None, 2048, use_l2k)
                    snap_ck = snap_states(c_ck)
                    kv_pre_cpu = move_cache_device(c_ck, "cpu")
                    del c_ck

                for dt_mode in dtypes:
                    rk = f"{rk_base}:C{C}_{dt_mode}"
                    if rk in done:
                        continue
                    res = coupled_eval_cell(
                        model, base_model, tok, eos_ids, n_linear, w_conv, n_layers, device, use_l2k,
                        tmp_path, ids_o, ids_e, ids_q, p0, b_end, T_ctx, smp["old"], smp["new"],
                        lg_ora, s_old_b_true, c_ctx_old_cpu, hs, c_pos, snap_ck, kv_pre_cpu,
                        dt_mode, f"{idx_s}_{C}_{dt_mode}",
                        measure_prefix_kv_disk_cold=measure_prefix_kv_disk_cold,
                    )
                    res.update({
                        "record_key": rk, "sample_idx": idx_s, "depth": float(smp["depth"]),
                        "stride": int(C), "dtype": dt_mode, "cell": f"C{C}_{dt_mode}",
                    })
                    append_record_jsonl(jsonl_path, res)
                    done.add(rk)
                    _ram_ms = res["substeps_ms"]["coupled_cold_ram_prefix_kv_single_timer_ms"]
                    _dsk_ms = res["substeps_ms"].get("coupled_cold_disk_prefix_kv_single_timer_ms", -1.0)
                    print(
                        f"[PROGRESS] {tag} sample={idx_s+1}/{n_samples} C={C} dt={dt_mode} "
                        f"ram_ms={_ram_ms:.1f} disk_ms={_dsk_ms:.1f} kl={res['kl']:.2e} em={int(res['em'])}",
                        flush=True,
                    )
                del kv_pre_cpu, snap_ck
            del c_ctx_old_cpu
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_sd: Dict[Tuple[int, str], Dict[int, int]] = {}
    for r in rows:
        by_sd.setdefault((r["sample_idx"], r["dtype"]), {})[r["stride"]] = r["c_pos"]
    s_sorted = sorted(strides)
    c512_eq_c1024 = sum(1 for d in by_sd.values() if len(s_sorted) >= 2 and s_sorted[0] in d and s_sorted[1] in d and d[s_sorted[0]] == d[s_sorted[1]])
    c1024_eq_c2048 = sum(1 for d in by_sd.values() if len(s_sorted) >= 3 and s_sorted[1] in d and s_sorted[2] in d and d[s_sorted[1]] == d[s_sorted[2]])
    c512_eq_c2048 = sum(1 for d in by_sd.values() if len(s_sorted) >= 3 and s_sorted[0] in d and s_sorted[2] in d and d[s_sorted[0]] == d[s_sorted[2]])
    any_collision = sum(1 for d in by_sd.values() if len(set(d.values())) < len(d))
    assert any_collision == 0, f"Detected {any_collision} stride collision sample-dtype pairs!"

    t_ctxs = [int(r["T_ctx"]) for r in rows]
    overall_lat: Dict[str, Any] = {
        "ram_prefix_kv_single_timer_ms": _upper_order_p50_p95(
            [r["substeps_ms"]["coupled_cold_ram_prefix_kv_single_timer_ms"] for r in rows]
        ),
        "timer_overhead_gap_ram_row_max_ms": round(max(r["substeps_ms"]["timer_overhead_gap_ms"] for r in rows), 3),
    }
    if all("coupled_cold_disk_prefix_kv_single_timer_ms" in r["substeps_ms"] for r in rows):
        pkv_all = [int(r.get("prefix_kv_disk_bytes", 0)) for r in rows]
        overall_lat["disk_prefix_kv_single_timer_ms"] = _upper_order_p50_p95(
            [r["substeps_ms"]["coupled_cold_disk_prefix_kv_single_timer_ms"] for r in rows]
        )
        overall_lat["timer_overhead_gap_disk_row_max_ms"] = round(
            max(r["substeps_ms"]["timer_overhead_gap_disk_ms"] for r in rows), 3
        )
        overall_lat["prefix_kv_disk_mib_row_range"] = [
            round(min(pkv_all) / (1024 * 1024), 3),
            round(max(pkv_all) / (1024 * 1024), 3),
        ]
        overall_lat["disk_vs_ram_prefix_kv_max_logit_diff_all_rows"] = max(
            float(r.get("disk_vs_ram_prefix_kv_max_logit_diff", 0.0)) for r in rows
        )

    return {
        "fs_info": fs_info,
        "n_records": len(rows),
        "t_ctx_actual_range": [min(t_ctxs), max(t_ctxs)],
        "c512_eq_c1024_cpos_pairs": int(c512_eq_c1024),
        "c1024_eq_c2048_cpos_pairs": int(c1024_eq_c2048),
        "c512_eq_c2048_cpos_pairs": int(c512_eq_c2048),
        "any_stride_collision_pairs": int(any_collision),
        "overall_latencies_ms": overall_lat,
        "cells": _aggregate_coupled(rows),
    }


def _verify_8k_depths_no_collision(tok: Any, depths: List[float], target_len: int) -> None:
    """Quick CPU tokenizer assertion verifying 100% distinct C512/C1024/C2048 and T_ctx >= 8192."""
    bank = generate_e5_sample_bank(n_samples=18, depths=depths, seed=202615)
    for nl in (18, 24):
        for s in bank:
            p = build_e5_prompt(tok, s, target_length=target_len, n_linear=nl, block_horizon="deep", device="cpu")
            p0, T_ctx = p["p0"], p["T_ctx"]
            c512, c1024, c2048 = (p0 // 512) * 512, (p0 // 1024) * 1024, (p0 // 2048) * 2048
            assert T_ctx >= 8192, f"Expected T_ctx >= 8192, got {T_ctx}"
            assert 0 < c2048 < c1024 < c512 < p0, f"Collision at nl={nl} s={s['sample_index']}: p0={p0}, ({c512},{c1024},{c2048})"


def run_synthetic_dry_run(out_dir: Path) -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
    if hasattr(qm.torch_chunk_gated_delta_rule, "__wrapped__"):
        qm.torch_chunk_gated_delta_rule = qm.torch_chunk_gated_delta_rule.__wrapped__
    if hasattr(qm.torch_recurrent_gated_delta_rule, "__wrapped__"):
        qm.torch_recurrent_gated_delta_rule = qm.torch_recurrent_gated_delta_rule.__wrapped__

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True, local_files_only=True)
    _verify_8k_depths_no_collision(tok, [0.47, 0.72, 0.94], 8704)

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
    dry_jsonl = out_dir / "w14_optb_coupled_nvme_dry.jsonl"
    dry_jsonl.unlink(missing_ok=True)
    res = run_suite(
        model, base_model, tok, eos_ids, 3, 32, 4, 4, "cpu", False,
        n_samples=1, t_tgt=360, strides=[32, 64, 128], dtypes=["fp32_rec", "int8"],
        depths=[0.68], jsonl_path=dry_jsonl, tag="dry", out_dir=out_dir,
        measure_prefix_kv_disk_cold=True,
    )
    rows = [json.loads(l) for l in dry_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    dry_jsonl.unlink(missing_ok=True)
    assert res["any_stride_collision_pairs"] == 0 and res["n_records"] == 6
    cell0 = next(iter(res["cells"].values()))
    assert "coupled_cold_ram_prefix_kv_single_timer_ms" in cell0["mean_substeps_ms"]
    assert "coupled_cold_disk_prefix_kv_single_timer_ms" in cell0["mean_substeps_ms"]
    assert "coupled_cold_e2e_with_prefix_kv_disk_ms" not in cell0["mean_substeps_ms"]
    assert all(r["disk_vs_ram_prefix_kv_max_logit_diff"] <= 1e-4 for r in rows)
    assert all("generated_text" in r and "exact_match" in r for r in rows)
    print(
        "[DRY-RUN PASS] w14_optb_coupled_nvme_eval.py genuine disk-wired prefix-KV single-timer E2E "
        f"(max_disk_vs_ram_diff={res['overall_latencies_ms']['disk_vs_ram_prefix_kv_max_logit_diff_all_rows']:.2e}) OK.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--tag", type=str, default="2b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default=str(SCRIPT_DIR))
    parser.add_argument("--n-samples", type=int, default=18)
    parser.add_argument("--ctx-len", type=int, default=8192)
    parser.add_argument("--prompt-target-len", type=int, default=0, help="Auto 8704 when ctx-len==8192 to yield actual T_ctx>=8192")
    parser.add_argument("--clean-out", action="store_true", help="Remove existing JSONL output before running")
    parser.add_argument("--measure-prefix-kv-disk-cold", action="store_true", help="Also run genuine single-timer cold-disk restore for [0:c_pos) full-attention prefix KV wired directly into decode cache")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--synthetic-dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.synthetic_dry_run:
        run_synthetic_dry_run(out_dir)
        return

    calibrated_depths = [0.47, 0.72, 0.94]
    prompt_target = int(args.prompt_target_len) if args.prompt_target_len > 0 else (8704 if args.ctx_len == 8192 else args.ctx_len)

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

    jsonl_p = out_dir / f"w14_optb_coupled_nvme_{args.tag}_records.jsonl"
    if args.clean_out:
        jsonl_p.unlink(missing_ok=True)
    res = run_suite(
        model, base_model, tok, eos_ids, n_linear, d_k, w_conv, n_layers,
        args.device, use_l2k, (2 if args.smoke else args.n_samples),
        prompt_target, [512, 1024, 2048], ["fp32_rec", "bf16", "int8"], calibrated_depths,
        jsonl_p, args.tag, out_dir,
        measure_prefix_kv_disk_cold=bool(args.measure_prefix_kv_disk_cold),
    )

    summ = {
        "runner": "pilot/w14_optb_coupled_nvme_eval.py",
        "model": args.model,
        "tag": args.tag,
        "device_name": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else "cpu",
        "os_name": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": _tf.__version__,
        "clean_rerun": bool(args.clean_out),
        "measure_prefix_kv_disk_cold": bool(args.measure_prefix_kv_disk_cold),
        "percentile_convention": "upper_order_statistic (sorted[n//2], sorted[min(n-1, int(0.95*n))])",
        "tiered_memory_scope": (
            "Two independent continuous single-timer E2E passes are timed with zero post-hoc arithmetic substitution: "
            "(1) RAM-prefix-KV pass (`coupled_cold_ram_prefix_kv_single_timer_ms`) cold-reads evicted recurrent/conv checkpoints from disk while loading `[0:c_pos)` and `[0:T_ctx)` full-attention KV from host RAM; "
            "(2) Disk-prefix-KV pass (`coupled_cold_disk_prefix_kv_single_timer_ms`, when `--measure-prefix-kv-disk-cold` is set) evicts BOTH recurrent/conv checkpoints AND `[0:c_pos)` prefix KV from OS page cache (`mincore()==0`), cold-reads both inside one continuous timer, wires disk-loaded GPU prefix-KV tensors directly into the live decode cache, while `[0:T_ctx)` full-context KV remains host-RAM-resident (`kv_ctx_h2d`). "
            "On container overlay filesystems (`backing_is_verified_nvme=False`), disk timings reflect Linux file page-cache-evicted I/O rather than verified physical NVMe."
        ),
        "decode_max_new_tokens": 8,
        "decode_eos_can_stop_early": True,
        "ctx_len": args.ctx_len,
        "prompt_target_len": prompt_target,
        "strides": [512, 1024, 2048],
        "dtypes": ["fp32_rec", "bf16", "int8"],
        "depths": calibrated_depths,
        "results": res,
    }
    summ_p = out_dir / f"w14_optb_coupled_nvme_{args.tag}_summary.json"
    summ_p.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Wrote {summ_p}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
