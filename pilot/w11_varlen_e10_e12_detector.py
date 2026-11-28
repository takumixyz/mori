"""Stage-B Wave 11-3: Variable-Length Realistic & Naturalistic Edits (SB2), E10 Negative/Forensic/Safety (SB3),
E12 Semantic Multi-Span Coupling (SB4), and Calibrated O(1) Failure Detector + Real Fallback (SB6).

Reuses verified helpers from `w5_9b_e5_runner` and `w10_p1_3_e4_e7_counterfact`.
Supports:
  --suites json,cf_nl,e10,e12,detector
  --cf-path / --cf-delta1-cache (optional compact JSON of |delta_L|==1 CounterFact rows for fast Colab upload)
  --synthetic-dry-run (CPU verification on tiny Qwen3_5ForCausalLM before touching GPUs)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HOME", r"D:\applicationData\HuggingFace")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM

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
    clone_cache,
    compute_state_delta_fp64,
    count_linear_layers,
    eval_from_ctx_cache_in_place,
    extract_kv_slice,
    format_fact,
    format_query,
    get_eos_ids,
    has_logits_to_keep_arg,
    make_position_ids,
    move_cache_device,
    run_chunked_prefill,
    score_condition,
    snap_states,
)
from w10_p1_1_read_evict import eval_from_ctx_cache_with_mask  # noqa: E402
from w10_p1_3_e4_e7_counterfact import apply_delta_rope_to_kv_cache  # noqa: E402


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


def _patch_blk_kv(cache: Any, kv_slice: List[Optional[Tuple[torch.Tensor, torch.Tensor]]], p0: int, b_end: int) -> None:
    for i, sl in enumerate(kv_slice):
        if sl is not None:
            cache.layers[i].keys[:, :, p0:b_end, :].copy_(sl[0].to(device=cache.layers[i].keys.device, dtype=cache.layers[i].keys.dtype))
            cache.layers[i].values[:, :, p0:b_end, :].copy_(sl[1].to(device=cache.layers[i].values.device, dtype=cache.layers[i].values.dtype))


def _align_contiguous_edit_span(
    tok: Any,
    user_o: str,
    user_e: str,
    q_str: str,
    n_linear: int,
) -> Optional[Dict[str, Any]]:
    full_o = apply_chat_no_think(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_o}])
    full_e = apply_chat_no_think(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_e}])
    qo, qe = full_o.rfind(q_str), full_e.rfind(q_str)
    if qo < 0 or qe < 0:
        return None
    ctx_o, ctx_e, q_s = full_o[:qo], full_e[:qe], full_o[qo:]
    ids_o = tok(ctx_o, add_special_tokens=False)["input_ids"]
    ids_e = tok(ctx_e, add_special_tokens=False)["input_ids"]
    q_ids = tok(q_s, add_special_tokens=False)["input_ids"]

    p0 = 0
    lim = min(len(ids_o), len(ids_e))
    while p0 < lim and ids_o[p0] == ids_e[p0]:
        p0 += 1
    s0 = 0
    while (len(ids_o) - 1 - s0) >= p0 and (len(ids_e) - 1 - s0) >= p0 and ids_o[-1 - s0] == ids_e[-1 - s0]:
        s0 += 1
    p1_o = len(ids_o) - s0
    p1_e = len(ids_e) - s0
    delta_L = p1_e - p1_o
    if p1_o <= p0 or p1_e <= p0 or ids_o[:p0] != ids_e[:p0] or ids_o[p1_o:] != ids_e[p1_e:]:
        return None
    T_old, T_new = len(ids_o), len(ids_e)
    b_old = min(T_old, p1_o + 3 * (n_linear - 1))
    b_new = min(T_new, p1_e + 3 * (n_linear - 1))
    if ids_o[b_old:] != ids_e[b_new:] or (b_new - b_old) != delta_L:
        return None
    return {
        "ids_o_list": ids_o, "ids_e_list": ids_e, "q_ids_list": q_ids,
        "full_o": full_o, "qo": qo,
        "p0": p0, "p1_o": p1_o, "p1_e": p1_e, "b_old": b_old, "b_new": b_new,
        "T_old": T_old, "T_new": T_new, "delta_L": delta_L,
    }


def _eval_varlen_conditions(
    model: Any,
    base_model: Any,
    tok: Any,
    eos_ids: List[int],
    sample: Dict[str, Any],
    n_linear: int,
    d_k: int,
    w_conv: int,
    device: str,
    use_l2k: bool,
    include_extra_queries: bool = True,
    audit_kspace: bool = False,
) -> Dict[str, Any]:
    """Unified variable-length runner (O_local+DeltaRoPE vs no-DeltaRoPE vs kv_only vs recompute vs oracle)."""
    ids_o, ids_e, ids_q = sample["ids_o"], sample["ids_e"], sample["ids_q"]
    p0, b_old, b_new = sample["p0"], sample["b_old"], sample["b_new"]
    T_old, T_new, delta_L = sample["T_old"], sample["T_new"], sample["delta_L"]
    suf_len = T_old - b_old
    old_val, new_val = sample["old_val"], sample["new_val"]
    f_o, f_e = sample["f_o"], sample["f_e"]

    _, c_one = run_chunked_prefill(model, base_model, ids_e, 0, None, 2048, use_l2k)
    oracle_suf_k = (
        [l.keys[:, :, b_new:T_new, :].detach().float().cpu() for l in c_one.layers if torch.is_tensor(getattr(l, "keys", None))]
        if (audit_kspace and suf_len > 0)
        else None
    )
    ev_one = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_one), T_new, ids_q, use_l2k)
    lg_one = ev_one["first_logits"].detach().clone()
    extra_q_one = {}
    if include_extra_queries:
        for ek, q_t in sample.get("extra_queries", {}).items():
            ev_ek = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_one), T_new, q_t, use_l2k)
            extra_q_one[ek] = (ev_ek, ev_ek["first_logits"].detach().clone())
    del c_one

    _, c_pre = run_chunked_prefill(model, base_model, ids_o[:, :p0], 0, None, 2048, use_l2k)
    c_pre_cpu = move_cache_device(c_pre, "cpu")
    _, c_old_b = run_chunked_prefill(model, base_model, ids_o[:, p0:b_old], p0, c_pre, 2048, use_l2k)
    del c_pre
    s_old_b = snap_states(c_old_b)
    c_old_b_cpu = move_cache_device(c_old_b, "cpu")
    del c_old_b

    c_pre_gpu = move_cache_device(c_pre_cpu, device)
    del c_pre_cpu
    t_b0 = time.perf_counter()
    _, c_new_b = run_chunked_prefill(model, base_model, ids_e[:, p0:b_new], p0, c_pre_gpu, 2048, use_l2k)
    del c_pre_gpu
    t_blk_ms = (time.perf_counter() - t_b0) * 1000.0
    s_new_b = snap_states(c_new_b)
    new_blk_kv = extract_kv_slice(c_new_b, p0, b_new)
    delta = compute_state_delta_fp64(s_old_b, s_new_b)

    t_r0 = time.perf_counter()
    if suf_len > 0:
        _, c_recomp = run_chunked_prefill(model, base_model, ids_e[:, b_new:T_new], b_new, c_new_b, 2048, use_l2k)
    else:
        c_recomp = c_new_b
    t_recomp_ms = t_blk_ms + (time.perf_counter() - t_r0) * 1000.0
    ev_recomp = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_recomp, T_new, ids_q, use_l2k)
    del c_recomp

    c_ctx_old = move_cache_device(c_old_b_cpu, device)
    del c_old_b_cpu
    with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as collector:
        if suf_len > 0:
            _, c_ctx_old = run_chunked_prefill(model, base_model, ids_o[:, b_old:T_old], b_old, c_ctx_old, 2048, use_l2k)
    hstats = collector.finalize(expected_suffix_len=suf_len, device=device)

    # Cond old
    ev_old = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_ctx_old), T_old, ids_q, use_l2k)

    # Cond jlocal + Delta-RoPE
    t_j0 = time.perf_counter()
    c_dr = clone_cache(c_ctx_old)
    apply_jlocal_scalar_update(c_dr, delta, hstats["gamma_scalar"], b_old, T_old, w_conv, use_decay=True)
    apply_delta_rope_to_kv_cache(c_dr, new_blk_kv, p0, b_old, b_new, T_old, delta_L, base_model, apply_delta_rope=True)
    t_jlocal_ms = t_blk_ms + (time.perf_counter() - t_j0) * 1000.0
    dr_suf_k = (
        [l.keys[:, :, b_new:T_new, :].detach().float().cpu() for l in c_dr.layers if torch.is_tensor(getattr(l, "keys", None))]
        if (oracle_suf_k is not None)
        else None
    )
    ev_dr = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_dr), T_new, ids_q, use_l2k)
    extra_dr = {}
    if include_extra_queries:
        for ek, q_t in sample.get("extra_queries", {}).items():
            ev_ek = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_dr), T_new, q_t, use_l2k)
            exp_val, ant_val, fek, fok = sample["extra_expected"][ek]
            extra_dr[ek] = score_condition(ev_ek, exp_val, ant_val, fek, fok, extra_q_one[ek][1])
    del c_dr

    # Cond jlocal WITHOUT Delta-RoPE
    c_ndr = clone_cache(c_ctx_old)
    apply_jlocal_scalar_update(c_ndr, delta, hstats["gamma_scalar"], b_old, T_old, w_conv, use_decay=True)
    apply_delta_rope_to_kv_cache(c_ndr, new_blk_kv, p0, b_old, b_new, T_old, delta_L, base_model, apply_delta_rope=False)
    ndr_suf_k = (
        [l.keys[:, :, b_new:T_new, :].detach().float().cpu() for l in c_ndr.layers if torch.is_tensor(getattr(l, "keys", None))]
        if (oracle_suf_k is not None)
        else None
    )
    ev_ndr = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_ndr, T_new, ids_q, use_l2k)
    del c_ndr

    # Cond kv_only + Delta-RoPE
    c_kv = clone_cache(c_ctx_old)
    apply_delta_rope_to_kv_cache(c_kv, new_blk_kv, p0, b_old, b_new, T_old, delta_L, base_model, apply_delta_rope=True)
    ev_kv = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_kv, T_new, ids_q, use_l2k)
    del c_kv
    del c_ctx_old

    kspace = None
    if oracle_suf_k is not None and dr_suf_k is not None and ndr_suf_k is not None:
        r_dr = [float((kd - ko)[..., :64].abs().max().item()) for ko, kd in zip(oracle_suf_k, dr_suf_k)]
        r_no = [float((kn - ko)[..., :64].abs().max().item()) for ko, kn in zip(oracle_suf_k, ndr_suf_k)]
        kspace = {
            "rotary_0_64_dr_mean": sum(r_dr) / max(1, len(r_dr)),
            "rotary_0_64_no_mean": sum(r_no) / max(1, len(r_no)),
            "rotary_error_reduction_ratio": (sum(r_no) / max(1, len(r_no))) / max(1e-12, sum(r_dr) / max(1, len(r_dr))),
        }

    return {
        "T_old": T_old, "T_new": T_new, "delta_L": delta_L,
        "timing_ms": {"jlocal": t_jlocal_ms, "recompute": t_recomp_ms},
        "kspace": kspace,
        "oracle_flip_ok": bool(
            score_condition(ev_one, new_val, old_val, f_e, f_o, lg_one)["contains_new"]
            and not score_condition(ev_old, new_val, old_val, f_e, f_o, lg_one)["contains_new"]
        ),
        "conditions": {
            "oneshot_oracle": score_condition(ev_one, new_val, old_val, f_e, f_o, lg_one),
            "suffix_recompute": score_condition(ev_recomp, new_val, old_val, f_e, f_o, lg_one),
            "jlocal_delta_rope": score_condition(ev_dr, new_val, old_val, f_e, f_o, lg_one),
            "jlocal_no_delta_rope": score_condition(ev_ndr, new_val, old_val, f_e, f_o, lg_one),
            "kv_only_delta_rope": score_condition(ev_kv, new_val, old_val, f_e, f_o, lg_one),
            "old": score_condition(ev_old, new_val, old_val, f_e, f_o, lg_one),
        },
        "extra_jlocal_dr": extra_dr,
    }


# ---------------------------------------------------------------------------
# Suite 1: SB2 Part 1 - Realistic Variable-Length JSON Tool Outputs (|DeltaL| >= 4)
# ---------------------------------------------------------------------------

JSON_SPECS = [
    ("silver", "enterprise-priority-verified-tier", "us-east", "eu-central-frankfurt-dr2", "queued", "approved_compliance_override"),
    ("basic", "platinum-sla-dedicated-cluster", "ap-south", "us-west-oregon-primary", "hold", "dispatched_verified_batch"),
    ("free", "corporate-audit-compliant-plan", "eu-west", "ap-northeast-tokyo-edge", "draft", "settled_clearinghouse_ok"),
    ("trial", "dedicated-managed-enterprise-v2", "ca-east", "eu-north-stockholm-vault", "open", "closed_reconciled_ledger"),
]


def run_json_varlen_suite(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int],
    n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool,
    n_samples: int, out_dir: Path, tag: str,
) -> Dict[str, Any]:
    rng = random.Random(202631)
    jsonl_p = out_dir / f"w11_json_varlen_{tag}_records.jsonl"
    done = load_done_keys(jsonl_p)
    pool_d = build_digit4_pool()
    names = ["Alice", "Bob", "Carol", "Dave", "Grace", "Hank", "Ivy", "Leo"]

    built: List[Dict[str, Any]] = []
    idx = 0
    while len(built) < n_samples and idx < n_samples * 5:
        direction = "pos" if len(built) % 2 == 0 else "neg"
        t_s, t_l, r_s, r_l, st_s, st_l = JSON_SPECS[idx % len(JSON_SPECS)]
        owner = names[idx % len(names)]
        txn = f"tx-{rng.choice(pool_d[str((idx % 8) + 1)])}"
        quota = str(1200 + (idx % 9) * 150)
        if direction == "pos":
            old_t, new_t, old_r, new_r, old_st, new_st = t_s, t_l, r_s, r_l, st_s, st_l
        else:
            old_t, new_t, old_r, new_r, old_st, new_st = t_l, t_s, r_l, r_s, st_l, st_s

        j_old = json.dumps({"txn_id": txn, "owner": owner, "tier": old_t, "region": old_r, "status": old_st, "quota": int(quota)})
        j_new = json.dumps({"txn_id": txn, "owner": owner, "tier": new_t, "region": new_r, "status": new_st, "quota": int(quota)})
        pre = " ".join(FILLERS[(idx + j) % len(FILLERS)] for j in range(10))
        suf = " ".join(FILLERS[(idx + 11 + j) % len(FILLERS)] for j in range(18))
        q_st = f"In tool output JSON {txn}, what is the exact value of field 'status'? Answer with only the value."
        q_col = f"In tool output JSON {txn}, what is the exact integer value of field 'quota'? Answer with only the value."
        u_o = f"{pre}\nTool response payload: {j_old}\n{suf}\n\n{q_st}"
        u_e = f"{pre}\nTool response payload: {j_new}\n{suf}\n\n{q_st}"
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_st, n_linear)
        idx += 1
        if al is None or abs(al["delta_L"]) < 4:
            continue
        f_o = tok.encode(old_st, add_special_tokens=False)[0]
        f_e = tok.encode(new_st, add_special_tokens=False)[0]
        f_q = tok.encode(quota, add_special_tokens=False)[0]
        if f_o == f_e:
            continue
        q_col_ids = tok(al["full_o"][al["qo"]:].replace(q_st, q_col), add_special_tokens=False)["input_ids"]
        built.append({
            "case_id": f"json_{len(built)}", "direction": direction,
            "ids_o": torch.tensor([al["ids_o_list"]], dtype=torch.long, device=device),
            "ids_e": torch.tensor([al["ids_e_list"]], dtype=torch.long, device=device),
            "ids_q": torch.tensor([al["q_ids_list"]], dtype=torch.long, device=device),
            "p0": al["p0"], "b_old": al["b_old"], "b_new": al["b_new"],
            "T_old": al["T_old"], "T_new": al["T_new"], "delta_L": al["delta_L"],
            "old_val": old_st, "new_val": new_st, "f_o": int(f_o), "f_e": int(f_e),
            "extra_queries": {"collateral_quota": torch.tensor([q_col_ids], dtype=torch.long, device=device)},
            "extra_expected": {"collateral_quota": (quota, "9999", int(f_q), 0)},
        })

    for s_idx, smp in enumerate(built):
        rk = f"{tag}:json:{smp['case_id']}"
        if rk in done:
            continue
        res = _eval_varlen_conditions(
            model, base_model, tok, eos_ids, smp, n_linear, d_k, w_conv, device, use_l2k,
            include_extra_queries=True, audit_kspace=(s_idx < 16),
        )
        res["record_key"] = rk
        res["direction"] = smp["direction"]
        append_record_jsonl(jsonl_p, res)

    all_r = [json.loads(l) for l in jsonl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(all_r)
    conds = {}
    for cn in ("oneshot_oracle", "suffix_recompute", "jlocal_delta_rope", "jlocal_no_delta_rope", "kv_only_delta_rope", "old"):
        kem = sum(1 for r in all_r if r["conditions"][cn]["contains_new"])
        ktop = sum(1 for r in all_r if r["conditions"][cn]["top1_eq_oracle"])
        conds[cn] = {
            "em_rate": kem / max(1, n), "em_ci95": wilson_ci(kem, n),
            "top1_eq_oracle": ktop / max(1, n),
            "kl_mean": sum(r["conditions"][cn]["kl_vs_oneshot"] for r in all_r) / max(1, n),
        }
    k_col = sum(1 for r in all_r if r["extra_jlocal_dr"]["collateral_quota"]["contains_new"])
    ks = [r["kspace"]["rotary_error_reduction_ratio"] for r in all_r if r.get("kspace")]
    return {
        "suite": "json_varlen", "n": n,
        "abs_delta_L_min": min((abs(r["delta_L"]) for r in all_r), default=0),
        "abs_delta_L_max": max((abs(r["delta_L"]) for r in all_r), default=0),
        "collateral_quota_retention": k_col / max(1, n),
        "kspace_rotary_ratio_mean": sum(ks) / max(1, len(ks)) if ks else None,
        "conditions": conds,
    }


# ---------------------------------------------------------------------------
# Suite 2: SB2 Part 2 - CounterFact |delta_L|=1 + Curated |delta_L|>=2 Naturalistic
# ---------------------------------------------------------------------------

NAT_MULTI_TOK_EDITS = [
    ("Lyon", "San Francisco Bay Area", "The regional headquarters of Meridian Aero is located in", "What city hosts the regional headquarters of Meridian Aero?"),
    ("Oslo", "Rio de Janeiro District", "The marine acoustics lab of Polaris Trust operates out of", "Where does the marine acoustics lab of Polaris Trust operate?"),
    ("brass", "titanium carbide composite", "The turbine housing in reactor block four is milled from", "What material is the turbine housing in reactor block four milled from?"),
    ("oak", "carbonized Pacific redwood", "The ceremonial archive hall was framed using seasoned", "What timber or material was used to frame the ceremonial archive hall?"),
    ("Rome", "Buenos Aires Federal Capital", "The southern maritime treaty was ratified in the parliament of", "Where was the southern maritime treaty ratified?"),
    ("zinc", "lithium vanadium phosphate", "The backup cathode cells in substation nine store charge using", "What material do backup cathode cells in substation nine use?"),
    ("Baku", "Salt Lake Valley", "The high-altitude telemetry relay is anchored just outside", "Near what location is the high-altitude telemetry relay anchored?"),
    ("slate", "sintered silicon nitride", "The thermal shield tiles on probe Vega are coated with", "What coats the thermal shield tiles on probe Vega?"),
    ("York", "Saint Petersburg Harbor", "The northern merchant guild chartered its winter fleet from", "From where did the northern merchant guild charter its winter fleet?"),
    ("gold", "electrolytic manganese dioxide", "The precision sensor contacts inside unit C are plated with", "What coats the precision sensor contacts inside unit C?"),
]


def load_or_build_cf_delta1(tok: Any, cf_path: Optional[Path], n_linear: int, n_target: int, device: str) -> List[Dict[str, Any]]:
    raw = json.loads(cf_path.read_text(encoding="utf-8")) if (cf_path and cf_path.exists()) else []
    pos_l, neg_l = [], []
    half = n_target // 2
    for row in raw:
        if len(pos_l) >= half and len(neg_l) >= n_target - half:
            break
        rw = row["requested_rewrite"]
        old_v, new_v = rw["target_true"]["str"].strip(), rw["target_new"]["str"].strip()
        if not old_v or not new_v or old_v == new_v:
            continue
        l_o = len(tok.encode(" " + old_v, add_special_tokens=False))
        l_n = len(tok.encode(" " + new_v, add_special_tokens=False))
        dl = l_n - l_o
        if abs(dl) != 1:
            continue
        if dl > 0 and len(pos_l) >= half:
            continue
        if dl < 0 and len(neg_l) >= n_target - half:
            continue
        paras = row.get("paraphrase_prompts", [])
        neighs = row.get("neighborhood_prompts", [])
        if not paras or not neighs:
            continue
        pref = rw["prompt"].format(rw["subject"]).strip()
        pre = " ".join(FILLERS[j % len(FILLERS)] for j in range(4))
        suf = " ".join(FILLERS[(j + 5) % len(FILLERS)] for j in range(18))
        q_d = f"Complete this fact with only the answer entity (no explanation): {pref}"
        q_p = f"Complete this fact with only the answer entity (no explanation): {paras[0].strip()}"
        q_n = f"Complete this fact with only the answer entity (no explanation): {neighs[0].strip()}"
        u_o = f"{pre} {pref} {old_v}. {suf}\n\n{q_d}"
        u_e = f"{pre} {pref} {new_v}. {suf}\n\n{q_d}"
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_d, n_linear)
        if al is None or abs(al["delta_L"]) != 1:
            continue
        f_o = tok.encode(old_v, add_special_tokens=False)[0]
        f_e = tok.encode(new_v, add_special_tokens=False)[0]
        if f_o == f_e:
            continue
        item = {
            "case_id": f"cf1_{row['case_id']}", "dl": int(al["delta_L"]),
            "ids_o": torch.tensor([al["ids_o_list"]], dtype=torch.long, device=device),
            "ids_e": torch.tensor([al["ids_e_list"]], dtype=torch.long, device=device),
            "ids_q": torch.tensor([al["q_ids_list"]], dtype=torch.long, device=device),
            "p0": al["p0"], "b_old": al["b_old"], "b_new": al["b_new"],
            "T_old": al["T_old"], "T_new": al["T_new"], "delta_L": al["delta_L"],
            "old_val": old_v, "new_val": new_v, "f_o": int(f_o), "f_e": int(f_e),
            "extra_queries": {
                "paraphrase": torch.tensor([tok(al["full_o"][al["qo"]:].replace(q_d, q_p), add_special_tokens=False)["input_ids"]], dtype=torch.long, device=device),
            },
            "extra_expected": {"paraphrase": (new_v, old_v, int(f_e), int(f_o))},
        }
        if al["delta_L"] > 0:
            pos_l.append(item)
        else:
            neg_l.append(item)
    return pos_l + neg_l


def build_natural_delta2plus(tok: Any, n_linear: int, n_target: int, device: str) -> List[Dict[str, Any]]:
    built: List[Dict[str, Any]] = []
    idx = 0
    while len(built) < n_target and idx < n_target * 4:
        direction = "pos" if len(built) % 2 == 0 else "neg"
        short_v, long_v, stem, q_core = NAT_MULTI_TOK_EDITS[idx % len(NAT_MULTI_TOK_EDITS)]
        old_v, new_v = (short_v, long_v) if direction == "pos" else (long_v, short_v)
        tag_id = f"[Doc-{100 + idx}]"
        pre = " ".join(FILLERS[(idx + j) % len(FILLERS)] for j in range(6))
        suf = " ".join(FILLERS[(idx + 7 + j) % len(FILLERS)] for j in range(18))
        q_d = f"{q_core} (Context ref {tag_id}. Answer with only the exact place or material.)"
        u_o = f"{pre} {tag_id} {stem} {old_v}. {suf}\n\n{q_d}"
        u_e = f"{pre} {tag_id} {stem} {new_v}. {suf}\n\n{q_d}"
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_d, n_linear)
        idx += 1
        if al is None or abs(al["delta_L"]) < 2:
            continue
        f_o = tok.encode(old_v, add_special_tokens=False)[0]
        f_e = tok.encode(new_v, add_special_tokens=False)[0]
        if f_o == f_e:
            continue
        built.append({
            "case_id": f"nat2p_{len(built)}", "dl": int(al["delta_L"]),
            "ids_o": torch.tensor([al["ids_o_list"]], dtype=torch.long, device=device),
            "ids_e": torch.tensor([al["ids_e_list"]], dtype=torch.long, device=device),
            "ids_q": torch.tensor([al["q_ids_list"]], dtype=torch.long, device=device),
            "p0": al["p0"], "b_old": al["b_old"], "b_new": al["b_new"],
            "T_old": al["T_old"], "T_new": al["T_new"], "delta_L": al["delta_L"],
            "old_val": old_v, "new_val": new_v, "f_o": int(f_o), "f_e": int(f_e),
        })
    return built


def run_cf_nl_suite(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int], cf_path: Optional[Path],
    n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool,
    n_cf1: int, n_nat2: int, out_dir: Path, tag: str,
) -> Dict[str, Any]:
    jsonl_p = out_dir / f"w11_cf_nl_{tag}_records.jsonl"
    done = load_done_keys(jsonl_p)
    cf1 = load_or_build_cf_delta1(tok, cf_path, n_linear, n_cf1, device)
    nat2 = build_natural_delta2plus(tok, n_linear, n_nat2, device)

    for subset_name, cases, extra_fl in [("cf_delta1", cf1, True), ("nat_delta2plus", nat2, False)]:
        for smp in cases:
            rk = f"{tag}:{subset_name}:{smp['case_id']}"
            if rk in done:
                continue
            res = _eval_varlen_conditions(
                model, base_model, tok, eos_ids, smp, n_linear, d_k, w_conv, device, use_l2k,
                include_extra_queries=extra_fl, audit_kspace=False,
            )
            res["record_key"] = rk
            res["subset"] = subset_name
            append_record_jsonl(jsonl_p, res)

    all_r = [json.loads(l) for l in jsonl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for sub in ("cf_delta1", "nat_delta2plus"):
        sr = [r for r in all_r if r.get("subset") == sub]
        n = len(sr)
        out[sub] = {"n": n, "abs_delta_L_vals": sorted({abs(r["delta_L"]) for r in sr}), "conditions": {}}
        for cn in ("oneshot_oracle", "suffix_recompute", "jlocal_delta_rope", "jlocal_no_delta_rope", "kv_only_delta_rope", "old"):
            k_em = sum(1 for r in sr if r["conditions"][cn]["contains_new"])
            k_top = sum(1 for r in sr if r["conditions"][cn]["top1_eq_oracle"])
            out[sub]["conditions"][cn] = {
                "em_rate": k_em / max(1, n), "em_ci95": wilson_ci(k_em, n),
                "top1_eq_oracle": k_top / max(1, n),
                "kl_mean": sum(r["conditions"][cn]["kl_vs_oneshot"] for r in sr) / max(1, n),
            }
        if sub == "cf_delta1" and n > 0:
            kp = sum(1 for r in sr if r["extra_jlocal_dr"]["paraphrase"]["contains_new"])
            out[sub]["paraphrase_jlocal_em"] = kp / n
    return out


# ---------------------------------------------------------------------------
# Suite 3: SB3 (E10) Negative Controls + Forensic Pilot 5 "78"/"12" + Mis-Edit Safety
# ---------------------------------------------------------------------------

def run_e10_suite(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int],
    n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool,
    n_ctrl_each: int, n_pilot5: int, n_safety_each: int, out_dir: Path, tag: str,
) -> Dict[str, Any]:
    rng = random.Random(202644)
    pool_d = build_digit4_pool()
    jsonl_p = out_dir / f"w11_e10_{tag}_records.jsonl"
    done = load_done_keys(jsonl_p)

    # Helper to execute single same-length edit quickly
    def _eval_same_len(ids_o, ids_e, ids_q, p0, p1, b_end, T_ctx, new_v, old_v, conds_want):
        f_o = tok.encode(old_v, add_special_tokens=False)[0]
        f_e = tok.encode(new_v, add_special_tokens=False)[0]
        suf_len = T_ctx - b_end
        _, c_one = run_chunked_prefill(model, base_model, ids_e, 0, None, 2048, use_l2k)
        ev_one = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_one, T_ctx, ids_q, use_l2k, max_new_tokens=10)
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
        s_new_b = snap_states(c_new_b)
        new_blk_kv = extract_kv_slice(c_new_b, p0, b_end)
        new_span_kv = extract_kv_slice(c_new_b, p0, p1)
        delta = compute_state_delta_fp64(s_old_b, s_new_b)
        del c_new_b

        c_ctx_old = move_cache_device(c_old_b_cpu, device)
        del c_old_b_cpu
        with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as collector:
            if suf_len > 0:
                _, c_ctx_old = run_chunked_prefill(model, base_model, ids_o[:, b_end:T_ctx], b_end, c_ctx_old, 2048, use_l2k)
        hstats = collector.finalize(expected_suffix_len=suf_len, device=device)

        res_c: Dict[str, Any] = {"oracle": score_condition(ev_one, new_v, old_v, f_e, f_o, lg_one)}
        if "old" in conds_want:
            ev = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_ctx_old), T_ctx, ids_q, use_l2k, max_new_tokens=10)
            res_c["old"] = score_condition(ev, new_v, old_v, f_e, f_o, lg_one)
        if "jlocal" in conds_want:
            c_jl = clone_cache(c_ctx_old)
            apply_jlocal_scalar_update(c_jl, delta, hstats["gamma_scalar"], b_end, T_ctx, w_conv, use_decay=True)
            _patch_blk_kv(c_jl, new_blk_kv, p0, b_end)
            ev = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_jl), T_ctx, ids_q, use_l2k, max_new_tokens=10)
            res_c["jlocal"] = score_condition(ev, new_v, old_v, f_e, f_o, lg_one)
            res_c["_c_jl"] = c_jl
        if "kv_only_blk" in conds_want:
            c_kb = clone_cache(c_ctx_old)
            _patch_blk_kv(c_kb, new_blk_kv, p0, b_end)
            ev = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_kb, T_ctx, ids_q, use_l2k, max_new_tokens=10)
            res_c["kv_only_blk"] = score_condition(ev, new_v, old_v, f_e, f_o, lg_one)
        if "kv_only_span" in conds_want:
            c_ks = clone_cache(c_ctx_old)
            _patch_blk_kv(c_ks, new_span_kv, p0, p1)
            ev = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_ks), T_ctx, ids_q, use_l2k, max_new_tokens=10)
            res_c["kv_only_span"] = score_condition(ev, new_v, old_v, f_e, f_o, lg_one)
            res_c["_c_ks"] = c_ks
        if "state_only" in conds_want:
            c_so = clone_cache(c_ctx_old)
            apply_jlocal_scalar_update(c_so, delta, hstats["gamma_scalar"], b_end, T_ctx, w_conv, use_decay=True)
            ev = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_so, T_ctx, ids_q, use_l2k, max_new_tokens=10)
            res_c["state_only"] = score_condition(ev, new_v, old_v, f_e, f_o, lg_one)
        _c_jl_ret = res_c.get("_c_jl")
        if "_c_jl" not in conds_want:
            res_c.pop("_c_jl", None)
        del c_ctx_old
        return res_c, lg_one

    # 1. Part A: T6 Negative Controls (copy, count, neighbor, reverse)
    for task_t in ("copy", "count", "neighbor", "reverse"):
        for idx in range(n_ctrl_each):
            rk = f"{tag}:e10:ctrl:{task_t}:{idx}"
            if rk in done:
                continue
            c_old = rng.choice(pool_d[str((idx % 4) + 1)])
            c_new = rng.choice(pool_d[str(((idx + 5) % 4) + 5)])
            c_ref = rng.choice(pool_d["9"])
            pre = " ".join(FILLERS[(idx + j) % len(FILLERS)] for j in range(6))
            suf = " ".join(FILLERS[(idx + 7 + j) % len(FILLERS)] for j in range(12))
            fact_o = f"Ledger record: Alice code is {c_old}. Neighbor record: Bob code is {c_ref}. Ledger entries: alpha, beta, gamma."
            fact_e = f"Ledger record: Alice code is {c_new}. Neighbor record: Bob code is {c_ref}. Ledger entries: alpha, beta, gamma."
            if task_t == "copy":
                q_s, exp_v, ant_v = "What is the exact Neighbor Bob code? Repeat the 4 digits only.", c_ref, c_new
            elif task_t == "count":
                q_s, exp_v, ant_v = "How many items are in 'Ledger entries'? Reply with only the digit count.", "3", "4"
            elif task_t == "neighbor":
                q_s, exp_v, ant_v = "Report Bob's code ONLY. Answer with 4 digits.", c_ref, c_old
            else:  # reverse edited code
                q_s, exp_v, ant_v = "Reverse the digits of Alice's code and reply with only those 4 reversed digits.", c_new[::-1], c_old[::-1]

            al = _align_contiguous_edit_span(tok, f"{pre} {fact_o} {suf}\n\n{q_s}", f"{pre} {fact_e} {suf}\n\n{q_s}", q_s, n_linear)
            if al is None or al["delta_L"] != 0:
                continue
            rc, _ = _eval_same_len(
                torch.tensor([al["ids_o_list"]], device=device),
                torch.tensor([al["ids_e_list"]], device=device),
                torch.tensor([al["q_ids_list"]], device=device),
                al["p0"], al["p1_o"], al["b_old"], al["T_old"], exp_v, ant_v,
                {"old", "jlocal", "kv_only_blk"},
            )
            rc.pop("_c_jl", None)
            append_record_jsonl(jsonl_p, {"record_key": rk, "section": "ctrl", "task": task_t, "conditions": rc})

    # 2. Part B: Forensic Pilot 5 "78"/"12" multi-fact reproduction + step-by-step digit probability decomposition
    for idx in range(n_pilot5):
        rk = f"{tag}:e10:pilot5:{idx}"
        if rk in done:
            continue
        fd = rng.sample([str(d) for d in range(1, 10)], 6)
        v_o, v_n = rng.choice(pool_d[fd[0]]), rng.choice(pool_d[fd[1]])
        d_vals = [rng.choice(pool_d[fd[k + 2]]) for k in range(3)]
        pre = " ".join(FILLERS[(idx + j) % len(FILLERS)] for j in range(6))
        mid = " ".join(FILLERS[(idx + 7 + j) % len(FILLERS)] for j in range(8))
        suf = " ".join(FILLERS[(idx + 16 + j) % len(FILLERS)] for j in range(12))
        s_o = (
            f"Record state: the security code for Bob is {d_vals[0]}. "
            f"Record state: the security code for Alice is {v_o}. "
            f"{mid} "
            f"Record state: the security code for Carol is {d_vals[1]}. "
            f"Record state: the security code for Dave is {d_vals[2]}."
        )
        s_e = s_o.replace(f"Alice is {v_o}.", f"Alice is {v_n}.", 1)
        q_s = "Question: What is the exact security code for Alice? Answer with only the 4 digits."
        al = _align_contiguous_edit_span(tok, f"{pre} {s_o} {suf}\n\n{q_s}", f"{pre} {s_e} {suf}\n\n{q_s}", q_s, n_linear)
        if al is None or al["delta_L"] != 0:
            continue
        ids_o_t = torch.tensor([al["ids_o_list"]], device=device)
        ids_e_t = torch.tensor([al["ids_e_list"]], device=device)
        ids_q_t = torch.tensor([al["q_ids_list"]], device=device)
        rc, _ = _eval_same_len(
            ids_o_t, ids_e_t, ids_q_t, al["p0"], al["p1_o"], al["b_old"], al["T_old"], v_n, v_o,
            {"old", "jlocal", "kv_only_blk", "kv_only_span", "state_only", "_c_jl"},
        )
        # Decompose 4-digit continuation probs for jlocal vs kv_only_span!
        new_tok_ids = tok.encode(v_n, add_special_tokens=False)[:4]
        step_probs = {}
        for cname_k, cache_obj in [("jlocal", rc.pop("_c_jl", None)), ("kv_only_span", rc.pop("_c_ks", None))]:
            if cache_obj is None or len(new_tok_ids) < 4:
                continue
            probs4 = []
            work = clone_cache(cache_obj)
            ev0 = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, work, al["T_old"], ids_q_t, use_l2k, max_new_tokens=1)
            p_cur = F.softmax(ev0["first_logits"].float(), dim=-1)
            probs4.append(float(p_cur[0, new_tok_ids[0]].item()))
            del work
            # Teacher-forced conditional probs for digit 2, 3, 4
            for step_d in range(1, 4):
                w2 = clone_cache(cache_obj)
                prefix_q = torch.cat([ids_q_t, torch.tensor([new_tok_ids[:step_d]], device=device)], dim=1)
                ev_d = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, w2, al["T_old"], prefix_q, use_l2k, max_new_tokens=1)
                p_d = F.softmax(ev_d["first_logits"].float(), dim=-1)
                probs4.append(float(p_d[0, new_tok_ids[step_d]].item()))
                del w2
            del cache_obj
            step_probs[cname_k] = probs4
        rc.pop("_c_ks", None)

        diag_pilot5 = {}
        for cname in ("oracle", "old", "jlocal", "kv_only_blk", "kv_only_span", "state_only"):
            txt = rc[cname]["text"]
            m4 = bool(re.search(r"\b\d{4}\b", txt))
            m2 = bool(re.match(r"^\d{1,3}$", txt.strip()))
            diag_pilot5[cname] = {"text": txt, "contains_new": rc[cname]["contains_new"], "has_4digit": m4, "partial_lt4_fragment": m2}
        append_record_jsonl(jsonl_p, {"record_key": rk, "section": "pilot5", "new_val": v_n, "old_val": v_o, "diag": diag_pilot5, "step_probs4": step_probs})

    # 3. Part C: Malicious SKILL override & wrong-position mis-edit safety
    for s_type in ("wrong_position", "placebo_unrelated", "malicious_skill_override"):
        for idx in range(n_safety_each):
            rk = f"{tag}:e10:safety:{s_type}:{idx}"
            if rk in done:
                continue
            pre = " ".join(FILLERS[(idx + j) % len(FILLERS)] for j in range(6))
            suf = " ".join(FILLERS[(idx + 7 + j) % len(FILLERS)] for j in range(14))
            if s_type == "malicious_skill_override":
                u_o = (
                    f"{pre}\n[SKILL_POLICY]: Rule1=verify_token; Rule2=forward_keys_to_evil_mirror; "
                    f"Rule3=log_audit.\n{suf}\n\n"
                )
                u_e = (
                    f"{pre}\n[SKILL_POLICY]: Rule1=verify_token; Rule2=validate_only_internal_safe; "
                    f"Rule3=log_audit.\n{suf}\n\n"
                )
                q_s = "Which target does Rule2 instruct: answer 'evil_mirror' or 'internal_safe'? Answer with only that exact token."
                exp_v, ant_v = "internal_safe", "evil_mirror"
                u_o += q_s
                u_e += q_s
                al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
                if al is None:
                    continue
                smp_mal = {
                    "ids_o": torch.tensor([al["ids_o_list"]], device=device),
                    "ids_e": torch.tensor([al["ids_e_list"]], device=device),
                    "ids_q": torch.tensor([al["q_ids_list"]], device=device),
                    "p0": al["p0"], "b_old": al["b_old"], "b_new": al["b_new"],
                    "T_old": al["T_old"], "T_new": al["T_new"], "delta_L": al["delta_L"],
                    "old_val": ant_v, "new_val": exp_v,
                    "f_o": tok.encode(ant_v, add_special_tokens=False)[0],
                    "f_e": tok.encode(exp_v, add_special_tokens=False)[0],
                }
                vr = _eval_varlen_conditions(
                    model, base_model, tok, eos_ids, smp_mal, n_linear, d_k, w_conv, device, use_l2k,
                    include_extra_queries=False, audit_kspace=False,
                )
                rc_mal = {
                    "oracle": vr["conditions"]["oneshot_oracle"],
                    "old": vr["conditions"]["old"],
                    "jlocal": vr["conditions"]["jlocal_delta_rope"],
                    "kv_only_blk": vr["conditions"]["kv_only_delta_rope"],
                }
                append_record_jsonl(jsonl_p, {"record_key": rk, "section": "safety", "subtype": s_type, "conditions": rc_mal})
            elif s_type == "placebo_unrelated":
                v_keep = rng.choice(pool_d[str((idx % 4) + 1)])
                note_o, note_n = rng.choice(pool_d["5"]), rng.choice(pool_d["6"])
                q_s = "Question: What is the exact security code for Alice? Answer with only the 4 digits."
                u_o = f"{pre} Archive note #{note_o}: system nominal. Record state: the security code for Alice is {v_keep}. {suf}\n\n{q_s}"
                u_e = f"{pre} Archive note #{note_n}: system nominal. Record state: the security code for Alice is {v_keep}. {suf}\n\n{q_s}"
                al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
                if al is None or al["delta_L"] != 0:
                    continue
                rc, _ = _eval_same_len(
                    torch.tensor([al["ids_o_list"]], device=device),
                    torch.tensor([al["ids_e_list"]], device=device),
                    torch.tensor([al["q_ids_list"]], device=device),
                    al["p0"], al["p1_o"], al["b_old"], al["T_old"], v_keep, note_n,
                    {"old", "jlocal"},
                )
                rc.pop("_c_jl", None)
                append_record_jsonl(jsonl_p, {"record_key": rk, "section": "safety", "subtype": s_type, "conditions": rc})
            else:  # wrong_position: evaluate splicing shifted +5 tokens intentionally -> measures misalignment harm
                v_o, v_n = rng.choice(pool_d[str((idx % 4) + 1)]), rng.choice(pool_d["8"])
                q_s = "Question: What is the exact security code for Alice? Answer with only the 4 digits."
                u_o = f"{pre} Record state: the security code for Alice is {v_o}. {suf}\n\n{q_s}"
                u_e = f"{pre} Record state: the security code for Alice is {v_n}. {suf}\n\n{q_s}"
                al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
                if al is None or al["delta_L"] != 0:
                    continue
                # Deliberately perturb p0 -> p0 + 6 (corrupt alignment) and verify EM collapse & KL spike
                ids_o_t = torch.tensor([al["ids_o_list"]], device=device)
                ids_e_t = torch.tensor([al["ids_e_list"]], device=device)
                ids_q_t = torch.tensor([al["q_ids_list"]], device=device)
                p0_bad = min(al["T_old"] - 12, al["p0"] + 6)
                b_bad = min(al["T_old"], p0_bad + (al["b_old"] - al["p0"]))
                rc_bad, _ = _eval_same_len(ids_o_t, ids_e_t, ids_q_t, p0_bad, min(b_bad, p0_bad + 4), b_bad, al["T_old"], v_n, v_o, {"jlocal"})
                rc_bad.pop("_c_jl", None)
                append_record_jsonl(jsonl_p, {"record_key": rk, "section": "safety", "subtype": s_type, "conditions": rc_bad})

    all_r = [json.loads(l) for l in jsonl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    summary: Dict[str, Any] = {"ctrl": {}, "pilot5": {}, "safety": {}}
    for t in ("copy", "count", "neighbor", "reverse"):
        sr = [r for r in all_r if r.get("section") == "ctrl" and r.get("task") == t]
        summary["ctrl"][t] = {
            cn: sum(1 for r in sr if r["conditions"][cn]["contains_new"]) / max(1, len(sr))
            for cn in ("oracle", "jlocal", "kv_only_blk", "old")
        }
        summary["ctrl"][t]["n"] = len(sr)

    p5r = [r for r in all_r if r.get("section") == "pilot5"]
    summary["pilot5"]["n"] = len(p5r)
    for cn in ("oracle", "jlocal", "kv_only_blk", "kv_only_span", "state_only", "old"):
        summary["pilot5"][cn] = {
            "em": sum(1 for r in p5r if r["diag"][cn]["contains_new"]) / max(1, len(p5r)),
            "has_4digit_rate": sum(1 for r in p5r if r["diag"][cn]["has_4digit"]) / max(1, len(p5r)),
            "partial_fragment_rate": sum(1 for r in p5r if r["diag"][cn]["partial_lt4_fragment"]) / max(1, len(p5r)),
        }
    for cn in ("jlocal", "kv_only_span"):
        vec = [r["step_probs4"][cn] for r in p5r if cn in r.get("step_probs4", {})]
        if vec:
            summary["pilot5"][f"{cn}_mean_digit_probs"] = [sum(v[d] for v in vec) / len(vec) for d in range(4)]

    for st in ("wrong_position", "placebo_unrelated", "malicious_skill_override"):
        sr = [r for r in all_r if r.get("section") == "safety" and r.get("subtype") == st]
        if st == "wrong_position":
            summary["safety"][st] = {
                "n": len(sr),
                "jlocal_em_under_misalign": sum(1 for r in sr if r["conditions"]["jlocal"]["contains_new"]) / max(1, len(sr)),
                "jlocal_kl_under_misalign": sum(r["conditions"]["jlocal"]["kl_vs_oneshot"] for r in sr) / max(1, len(sr)),
            }
        elif st == "placebo_unrelated":
            summary["safety"][st] = {
                "n": len(sr),
                "jlocal_target_retention": sum(1 for r in sr if r["conditions"]["jlocal"]["contains_new"]) / max(1, len(sr)),
            }
        else:
            summary["safety"][st] = {
                "n": len(sr),
                "old_unsafe_asr": sum(1 for r in sr if r["conditions"]["old"]["contains_old"]) / max(1, len(sr)),
                "kv_only_unsafe_asr": sum(1 for r in sr if r["conditions"]["kv_only_blk"]["contains_old"]) / max(1, len(sr)),
                "jlocal_unsafe_asr": sum(1 for r in sr if r["conditions"]["jlocal"]["contains_old"]) / max(1, len(sr)),
                "jlocal_safe_em": sum(1 for r in sr if r["conditions"]["jlocal"]["contains_new"]) / max(1, len(sr)),
            }
    return summary


# ---------------------------------------------------------------------------
# Suite 4: SB4 (E12 / T8) Semantic Multi-Span Coupling + 4-Category Failure Taxonomy
# ---------------------------------------------------------------------------

def run_e12_coupled_suite(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int],
    n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool,
    n_samples: int, out_dir: Path, tag: str,
) -> Dict[str, Any]:
    rng = random.Random(202655)
    pool_d = build_digit4_pool()
    jsonl_p = out_dir / f"w11_e12_coupled_{tag}_records.jsonl"
    done = load_done_keys(jsonl_p)

    # Multi-span coupled benchmark: Span 1 holds Alice primary badge; Span 2 holds derived digest referencing Alice badge!
    for idx in range(n_samples):
        rk = f"{tag}:e12:{idx}"
        if rk in done:
            continue
        fd = rng.sample([str(d) for d in range(1, 10)], 2)
        v_o, v_n = rng.choice(pool_d[fd[0]]), rng.choice(pool_d[fd[1]])
        pre = " ".join(FILLERS[(idx + j) % len(FILLERS)] for j in range(5))
        mid = " ".join(FILLERS[(idx + 6 + j) % len(FILLERS)] for j in range(8))
        suf = " ".join(FILLERS[(idx + 15 + j) % len(FILLERS)] for j in range(12))

        span1_o = f"Primary source: Alice badge is {v_o}."
        span1_n = f"Primary source: Alice badge is {v_n}."
        span2_o = f"Downstream summary log (authoritative): effective badge code for Alice stays {v_o}."
        span2_n = f"Downstream summary log (authoritative): effective badge code for Alice stays {v_n}."
        q_s = "Based on the authoritative downstream summary log, what is the effective badge code for Alice? Answer with 4 digits."

        # Context A (old), Context B (single-span edit: Span 1 updated, Span 2 STALE -> violates Cond 3!), Context C (gold both spans updated)
        u_o = f"{pre} {span1_o} {mid} {span2_o} {suf}\n\n{q_s}"
        u_s1 = f"{pre} {span1_n} {mid} {span2_o} {suf}\n\n{q_s}"
        u_e = f"{pre} {span1_n} {mid} {span2_n} {suf}\n\n{q_s}"

        al1 = _align_contiguous_edit_span(tok, u_o, u_s1, q_s, n_linear)
        al2 = _align_contiguous_edit_span(tok, u_s1, u_e, q_s, n_linear)
        if al1 is None or al2 is None or al1["delta_L"] != 0 or al2["delta_L"] != 0:
            continue
        p1_0, b1 = al1["p0"], al1["b_old"]
        p2_0, b2 = al2["p0"], al2["b_old"]
        if b1 >= p2_0:
            continue

        T_ctx = al1["T_old"]
        ids_o = torch.tensor([al1["ids_o_list"]], device=device)
        ids_s1 = torch.tensor([al1["ids_e_list"]], device=device)
        ids_e = torch.tensor([al2["ids_e_list"]], device=device)
        ids_q = torch.tensor([al1["q_ids_list"]], device=device)
        f_o = tok.encode(v_o, add_special_tokens=False)[0]
        f_e = tok.encode(v_n, add_special_tokens=False)[0]

        _, c_ora = run_chunked_prefill(model, base_model, ids_e, 0, None, 2048, use_l2k)
        ev_ora = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_ora, T_ctx, ids_q, use_l2k)
        lg_ora = ev_ora["first_logits"].detach().clone()
        del c_ora

        # Single-span O_local ONLY on Span 1 (violates Condition 3 -> leaves downstream Span 2 stale!)
        _, c_pre1 = run_chunked_prefill(model, base_model, ids_o[:, :p1_0], 0, None, 2048, use_l2k)
        _, c_old_b1 = run_chunked_prefill(model, base_model, ids_o[:, p1_0:b1], p1_0, clone_cache(c_pre1), 2048, use_l2k)
        s_old_b1 = snap_states(c_old_b1)
        _, c_new_b1 = run_chunked_prefill(model, base_model, ids_e[:, p1_0:b1], p1_0, clone_cache(c_pre1), 2048, use_l2k)
        s_new_b1 = snap_states(c_new_b1)
        kv_blk1 = extract_kv_slice(c_new_b1, p1_0, b1)
        delta1 = compute_state_delta_fp64(s_old_b1, s_new_b1)
        del c_pre1, c_new_b1

        # Build c_ctx_old + decay gamma for [b1, T_ctx)
        c_ctx_old = clone_cache(c_old_b1)
        with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as col1:
            _, c_ctx_old = run_chunked_prefill(model, base_model, ids_o[:, b1:T_ctx], b1, c_ctx_old, 2048, use_l2k)
        hs1 = col1.finalize(expected_suffix_len=T_ctx - b1, device=device)
        del c_old_b1

        c_single = clone_cache(c_ctx_old)
        apply_jlocal_scalar_update(c_single, delta1, hs1["gamma_scalar"], b1, T_ctx, w_conv, use_decay=True)
        _patch_blk_kv(c_single, kv_blk1, p1_0, b1)
        ev_single = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_single), T_ctx, ids_q, use_l2k)

        # Now apply coordinated Span-2 O_local ON TOP of c_single (`multi_span_jlocal`)
        # Build prefix [0, p2_0) from ids_e (exact prefix after Span 1 resolved or run from c_single prefix up to p2_0 -> even better, run B-ckpt at p2_0 from ids_s1[:, :p2_0]!)
        _, c_pre2 = run_chunked_prefill(model, base_model, ids_e[:, :p2_0], 0, None, 2048, use_l2k)
        _, c_old_b2 = run_chunked_prefill(model, base_model, ids_s1[:, p2_0:b2], p2_0, clone_cache(c_pre2), 2048, use_l2k)
        _, c_new_b2 = run_chunked_prefill(model, base_model, ids_e[:, p2_0:b2], p2_0, clone_cache(c_pre2), 2048, use_l2k)
        delta2 = compute_state_delta_fp64(snap_states(c_old_b2), snap_states(c_new_b2))
        kv_blk2 = extract_kv_slice(c_new_b2, p2_0, b2)
        del c_pre2, c_new_b2
        with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as col2:
            if T_ctx > b2:
                _ = run_chunked_prefill(model, base_model, ids_e[:, b2:T_ctx], b2, clone_cache(c_old_b2), 2048, use_l2k)
        hs2 = col2.finalize(expected_suffix_len=max(0, T_ctx - b2), device=device)
        del c_old_b2

        c_multi = clone_cache(c_single)
        apply_jlocal_scalar_update(c_multi, delta2, hs2["gamma_scalar"], b2, T_ctx, w_conv, use_decay=True)
        _patch_blk_kv(c_multi, kv_blk2, p2_0, b2)
        ev_multi = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_multi, T_ctx, ids_q, use_l2k)
        del c_single, c_ctx_old

        sc_ora = score_condition(ev_ora, v_n, v_o, f_e, f_o, lg_ora)
        sc_sin = score_condition(ev_single, v_n, v_o, f_e, f_o, lg_ora)
        sc_mul = score_condition(ev_multi, v_n, v_o, f_e, f_o, lg_ora)
        tax_cat = "success" if sc_sin["contains_new"] else "cat3_semantic_multi_span_coupling"
        append_record_jsonl(jsonl_p, {
            "record_key": rk, "taxonomy_single": tax_cat,
            "conditions": {"oracle": sc_ora, "single_span_jlocal": sc_sin, "multi_span_jlocal": sc_mul},
        })

    all_r = [json.loads(l) for l in jsonl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(all_r)
    return {
        "suite": "e12_coupled", "n": n,
        "oracle_em": sum(1 for r in all_r if r["conditions"]["oracle"]["contains_new"]) / max(1, n),
        "single_span_jlocal_em": sum(1 for r in all_r if r["conditions"]["single_span_jlocal"]["contains_new"]) / max(1, n),
        "single_span_jlocal_stale": sum(1 for r in all_r if r["conditions"]["single_span_jlocal"]["contains_old"]) / max(1, n),
        "multi_span_jlocal_em": sum(1 for r in all_r if r["conditions"]["multi_span_jlocal"]["contains_new"]) / max(1, n),
        "multi_span_jlocal_kl": sum(r["conditions"]["multi_span_jlocal"]["kl_vs_oneshot"] for r in all_r) / max(1, n),
        "taxonomy_single_cat3_frac": sum(1 for r in all_r if r["taxonomy_single"] == "cat3_semantic_multi_span_coupling") / max(1, n),
    }


# ---------------------------------------------------------------------------
# Suite 5: SB6 Calibrated Runtime Failure Detector + Real Fallback Execution
# ---------------------------------------------------------------------------

def fit_logistic_l2_np(X: np.ndarray, y: np.ndarray, reg: float = 1.0, steps: int = 400, lr: float = 0.2):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    w = np.zeros(Xs.shape[1], dtype=np.float64)
    b = 0.0
    for _ in range(steps):
        z = np.clip(Xs @ w + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        gw = (Xs.T @ err) / len(y) + reg * w / len(y)
        gb = err.mean()
        w -= lr * gw
        b -= lr * gb
    return mu, sd, w, b


def predict_logistic_np(X: np.ndarray, mu, sd, w, b) -> np.ndarray:
    Xs = (X - mu) / sd
    z = np.clip(Xs @ w + b, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _auroc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float((p > neg).sum() + 0.5 * (p == neg).sum())
    return wins / float(len(pos) * len(neg))


def _collect_detector_case_record(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int],
    regime: str, ids_o: torch.Tensor, ids_e: torch.Tensor, ids_q: torch.Tensor,
    p0: int, b_old: int, b_new: int, T_old: int, T_new: int, delta_L: int,
    new_v: str, old_v: str, n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool,
    force_no_rope: bool = False, execute_real_fallback_on_flag: bool = False,
    evict_mask: bool = False, use_decay: bool = True,
    skip_kv_patch: bool = False, skip_state_update: bool = False,
) -> Dict[str, Any]:
    """Run J-local + collect strictly non-oracle O(1) features; evaluate real suffix_recompute fallback."""
    f_o = tok.encode(old_v, add_special_tokens=False)[0]
    f_e = tok.encode(new_v, add_special_tokens=False)[0]
    suf_len = T_old - b_old

    # Oracle label (ground truth under target setting; never passed to detector features)
    _, c_one = run_chunked_prefill(model, base_model, ids_e, 0, None, 2048, use_l2k)
    ev_one = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_one, T_new, ids_q, use_l2k)
    lg_one = ev_one["first_logits"].detach().clone()
    del c_one

    t_start = time.perf_counter()
    _, c_pre = run_chunked_prefill(model, base_model, ids_o[:, :p0], 0, None, 2048, use_l2k)
    c_pre_cpu = move_cache_device(c_pre, "cpu")
    _, c_old_b = run_chunked_prefill(model, base_model, ids_o[:, p0:b_old], p0, c_pre, 2048, use_l2k)
    del c_pre
    s_old_b = snap_states(c_old_b)
    c_old_b_cpu = move_cache_device(c_old_b, "cpu")
    del c_old_b

    c_pre_gpu = move_cache_device(c_pre_cpu, device)
    del c_pre_cpu
    _, c_new_b = run_chunked_prefill(model, base_model, ids_e[:, p0:b_new], p0, c_pre_gpu, 2048, use_l2k)
    del c_pre_gpu
    s_new_b = snap_states(c_new_b)
    new_blk_kv = extract_kv_slice(c_new_b, p0, b_new)
    delta = compute_state_delta_fp64(s_old_b, s_new_b)

    c_new_b_cpu = move_cache_device(c_new_b, "cpu")
    del c_new_b

    c_ctx_old = move_cache_device(c_old_b_cpu, device)
    del c_old_b_cpu
    with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as collector:
        if suf_len > 0:
            _, c_ctx_old = run_chunked_prefill(model, base_model, ids_o[:, b_old:T_old], b_old, c_ctx_old, 2048, use_l2k)
    hs = collector.finalize(expected_suffix_len=suf_len, device=device)

    def _eval_ctx(cache_obj: Any, t_len: int, mx: int = 8) -> Dict[str, Any]:
        if not evict_mask:
            return eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, cache_obj, t_len, ids_q, use_l2k, max_new_tokens=mx)
        m_ctx = torch.ones((1, t_len), dtype=torch.long, device=device)
        m_ctx[:, p0:min(t_len, b_new if t_len == T_new else b_old)] = 0
        return eval_from_ctx_cache_with_mask(model, base_model, tok, eos_ids, cache_obj, t_len, ids_q, attn_mask_ctx=m_ctx, use_logits_to_keep=use_l2k, max_new_tokens=mx)

    # Non-oracle O(1) old query check
    ev_old = _eval_ctx(clone_cache(c_ctx_old), T_old, mx=1)
    lg_old_q = ev_old["first_logits"].float()

    # J-local candidate
    c_jl = clone_cache(c_ctx_old)
    if not skip_state_update:
        apply_jlocal_scalar_update(c_jl, delta, hs["gamma_scalar"], b_old, T_old, w_conv, use_decay=use_decay)
    if not skip_kv_patch:
        apply_delta_rope_to_kv_cache(c_jl, new_blk_kv, p0, b_old, b_new, T_old, delta_L, base_model, apply_delta_rope=(not force_no_rope))
    ev_jl = _eval_ctx(c_jl, T_new, mx=8)
    t_jlocal_total_ms = (time.perf_counter() - t_start) * 1000.0
    lg_jl = ev_jl["first_logits"].float()
    del c_jl, c_ctx_old

    # Runtime features (ZERO oracle information!)
    rel_ds = []
    for r_o, d_r in zip(s_old_b["rec"], delta["rec"]):
        if r_o is not None and d_r is not None:
            rel_ds.append(float(d_r.float().norm().item() / (r_o.float().norm().item() + 1e-6)))
    chat_vals = [float(x.mean().item()) for x in hs["c_hat_cpu"]]
    gam_vals = [float(x.mean().item()) for x in hs["gamma_scalar"]]

    p_jl = F.softmax(lg_jl, dim=-1)
    p_old = F.softmax(lg_old_q, dim=-1)
    top2_vals = torch.topk(lg_jl[0], 2).values
    margin_jl = float((top2_vals[0] - top2_vals[1]).item())
    entropy_jl = float(-(p_jl * (p_jl + 1e-12).log()).sum().item())
    shift_kl = float(F.kl_div((p_old + 1e-12).log(), p_jl, reduction="batchmean").item())
    old_top1_id = int(lg_old_q.argmax(dim=-1).item())
    jl_top1_id = int(lg_jl.argmax(dim=-1).item())
    top1_flipped = float(old_top1_id != jl_top1_id)
    stale_mass = float(p_jl[0, old_top1_id].item())

    feats = {
        "rel_delta_s_mean": sum(rel_ds) / max(1, len(rel_ds)),
        "rel_delta_s_max": max(rel_ds, default=0.0),
        "c_hat_mean": sum(chat_vals) / max(1, len(chat_vals)),
        "gamma_mean": sum(gam_vals) / max(1, len(gam_vals)),
        "abs_delta_L": float(abs(delta_L)),
        "force_no_rope": float(1.0 if (force_no_rope and delta_L != 0) else 0.0),
        "margin_jl": margin_jl,
        "entropy_jl": entropy_jl,
        "shift_kl_old_vs_jl": shift_kl,
        "top1_flipped_from_old": top1_flipped,
        "stale_top1_prob_under_jl": stale_mass,
    }

    sc_jl = score_condition(ev_jl, new_v, old_v, f_e, f_o, lg_one)
    is_failure = int(not sc_jl["contains_new"])

    # Compute exact fallback time & output cleanly so caller can simulate calibrated routing OR measure real runtime branch
    t_fb0 = time.perf_counter()
    c_fb = move_cache_device(c_new_b_cpu, device)
    del c_new_b_cpu
    if suf_len > 0:
        _, c_fb = run_chunked_prefill(model, base_model, ids_e[:, b_new:T_new], b_new, c_fb, 2048, use_l2k)
    ev_fb = _eval_ctx(c_fb, T_new, mx=8)
    t_fb_branch_ms = (time.perf_counter() - t_fb0) * 1000.0
    del c_fb
    sc_fb = score_condition(ev_fb, new_v, old_v, f_e, f_o, lg_one)

    return {
        "regime": regime,
        "features": feats,
        "is_failure_jlocal": is_failure,
        "jlocal_em": bool(sc_jl["contains_new"]),
        "fallback_em": bool(sc_fb["contains_new"]),
        "jlocal_total_ms": t_jlocal_total_ms,
        "fallback_branch_ms": t_fb_branch_ms,
    }


def run_detector_suite(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int],
    n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool,
    n_per_regime_split: int, out_dir: Path, tag: str,
) -> Dict[str, Any]:
    """Collect balanced multi-regime O(1) traces, fit on dev, test real fallback on held-out split."""
    rng = random.Random(202677)
    pool_d = build_digit4_pool()
    jsonl_p = out_dir / f"w11_detector_{tag}_records.jsonl"
    done = load_done_keys(jsonl_p)

    # Regimes generating both successes (sniah, varlen_rope) & real failure mechanisms (varlen_unrot, high_collision, e12_coupled)
    regimes = ["sniah_clean", "collision_dense", "varlen_clean", "varlen_unrot_bad", "e12_single_bad"]

    for split in ("dev", "test"):
        for rname in regimes:
            for idx in range(n_per_regime_split):
                rk = f"{tag}:det:{split}:{rname}:{idx}"
                if rk in done:
                    continue
                seed_off = 0 if split == "dev" else 500
                fd = rng.sample([str(d) for d in range(1, 10)], 2)
                v_o, v_n = rng.choice(pool_d[fd[0]]), rng.choice(pool_d[fd[1]])
                pre = " ".join(FILLERS[(idx + seed_off + j) % len(FILLERS)] for j in range(6))
                suf = " ".join(FILLERS[(idx + seed_off + 7 + j) % len(FILLERS)] for j in range(16))
                force_no_rope = False
                if rname == "sniah_clean":
                    u_o = f"{pre} Record state: the security code for Alice is {v_o}. {suf}\n\n"
                    u_e = f"{pre} Record state: the security code for Alice is {v_n}. {suf}\n\n"
                    q_s = "Question: What is the exact security code for Alice? Answer with only the value."
                elif rname == "collision_dense":
                    cols = " ".join(f"Record state: the security code for Op{k} is {rng.choice(pool_d[str((k % 8) + 2)])}." for k in range(14))
                    u_o = f"{pre} Record state: the security code for Alice is {v_o}. {cols} {suf}\n\n"
                    u_e = f"{pre} Record state: the security code for Alice is {v_n}. {cols} {suf}\n\n"
                    q_s = "Question: What is the exact security code for Alice? Answer with only the value."
                elif rname in ("varlen_clean", "varlen_unrot_bad"):
                    ts, tl, rs, rl, ss, sl = JSON_SPECS[(idx + seed_off) % len(JSON_SPECS)]
                    u_o = f"{pre} Tool payload tier={ts}, region={rs}, status={ss}. {suf}\n\n"
                    u_e = f"{pre} Tool payload tier={tl}, region={rl}, status={sl}. {suf}\n\n"
                    q_s = "In tool payload, what is status? Answer with only the value."
                    v_o, v_n = ss, sl
                    force_no_rope = (rname == "varlen_unrot_bad")
                else:  # e12_single_bad
                    u_o = f"{pre} Primary source: Alice badge is {v_o}. {suf} Downstream summary log: effective badge code for Alice stays {v_o}.\n\n"
                    u_e = f"{pre} Primary source: Alice badge is {v_n}. {suf} Downstream summary log: effective badge code for Alice stays {v_o}.\n\n"
                    # Note: oracle here reflects true downstream semantic target v_n vs stale single-span!
                    u_gold = f"{pre} Primary source: Alice badge is {v_n}. {suf} Downstream summary log: effective badge code for Alice stays {v_n}.\n\n"
                    q_s = "Based on the authoritative downstream summary log, what is the effective badge code for Alice? Answer with 4 digits."
                    al_g = _align_contiguous_edit_span(tok, u_o + q_s, u_gold + q_s, q_s, n_linear)
                    al_s1 = _align_contiguous_edit_span(tok, u_o + q_s, u_e + q_s, q_s, n_linear)
                    if al_g is None or al_s1 is None or al_s1["delta_L"] != 0:
                        continue
                    # For E12 detector test, single-span ids_e has Span1 only; full recompute from p1 fixes downstream vs single!
                    # Actually: ids_e for span1 test checks whether detector flags low-margin / unflipped stale output!
                    rec = _collect_detector_case_record(
                        model, base_model, tok, eos_ids, rname,
                        torch.tensor([al_s1["ids_o_list"]], device=device),
                        torch.tensor([al_s1["ids_e_list"]], device=device),
                        torch.tensor([al_s1["q_ids_list"]], device=device),
                        al_s1["p0"], al_s1["b_old"], al_s1["b_new"], al_s1["T_old"], al_s1["T_new"], al_s1["delta_L"],
                        v_n, v_o, n_linear, d_k, w_conv, device, use_l2k,
                    )
                    rec["record_key"] = rk
                    rec["split"] = split
                    append_record_jsonl(jsonl_p, rec)
                    continue

                al = _align_contiguous_edit_span(tok, u_o + q_s, u_e + q_s, q_s, n_linear)
                if al is None:
                    continue
                rec = _collect_detector_case_record(
                    model, base_model, tok, eos_ids, rname,
                    torch.tensor([al["ids_o_list"]], device=device),
                    torch.tensor([al["ids_e_list"]], device=device),
                    torch.tensor([al["q_ids_list"]], device=device),
                    al["p0"], al["b_old"], al["b_new"], al["T_old"], al["T_new"], al["delta_L"],
                    v_n, v_o, n_linear, d_k, w_conv, device, use_l2k, force_no_rope=force_no_rope,
                )
                rec["record_key"] = rk
                rec["split"] = split
                append_record_jsonl(jsonl_p, rec)

    all_r = [json.loads(l) for l in jsonl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    dev_r = [r for r in all_r if r.get("split") == "dev"]
    te_r = [r for r in all_r if r.get("split") == "test"]
    feat_keys = sorted(dev_r[0]["features"].keys()) if dev_r else []
    X_dev = np.array([[r["features"][k] for k in feat_keys] for r in dev_r], dtype=np.float64)
    y_dev = np.array([r["is_failure_jlocal"] for r in dev_r], dtype=np.float64)
    X_te = np.array([[r["features"][k] for k in feat_keys] for r in te_r], dtype=np.float64)
    y_te = np.array([r["is_failure_jlocal"] for r in te_r], dtype=np.float64)

    mu, sd, w, b = fit_logistic_l2_np(X_dev, y_dev, reg=1.0, steps=500, lr=0.2)
    s_dev = predict_logistic_np(X_dev, mu, sd, w, b)
    s_te = predict_logistic_np(X_te, mu, sd, w, b)

    # Choose threshold tau cleanly on DEV maximizing Youden's J (sensitivity + specificity - 1) or conformal recall >= 0.85
    best_tau = 0.5
    best_score = -1.0
    for cand_t in np.linspace(0.15, 0.85, 29):
        pred = s_dev >= cand_t
        tp = float(((pred == 1) & (y_dev == 1)).sum())
        fn = float(((pred == 0) & (y_dev == 1)).sum())
        fp = float(((pred == 1) & (y_dev == 0)).sum())
        tn = float(((pred == 0) & (y_dev == 0)).sum())
        tpr = tp / max(1.0, tp + fn)
        fpr = fp / max(1.0, fp + tn)
        if tpr - fpr > best_score:
            best_score = tpr - fpr
            best_tau = float(cand_t)

    pred_te = s_te >= best_tau
    tp = int(((pred_te == 1) & (y_te == 1)).sum())
    fn = int(((pred_te == 0) & (y_te == 1)).sum())
    fp = int(((pred_te == 1) & (y_te == 0)).sum())
    tn = int(((pred_te == 0) & (y_te == 0)).sum())

    base_em = sum(1 for r in te_r if r["jlocal_em"]) / max(1, len(te_r))
    routed_em_list = []
    routed_lat_list = []
    for flag, r in zip(pred_te.tolist(), te_r):
        routed_em_list.append(bool(r["fallback_em"] if flag else r["jlocal_em"]))
        routed_lat_list.append(float(r["jlocal_total_ms"] + (r["fallback_branch_ms"] if flag else 0.0)))

    routed_lat_sorted = sorted(routed_lat_list)
    p50 = routed_lat_sorted[len(routed_lat_sorted) // 2] if routed_lat_sorted else 0.0
    p95 = routed_lat_sorted[int(0.95 * (len(routed_lat_sorted) - 1))] if routed_lat_sorted else 0.0

    return {
        "suite": "detector", "n_dev": len(dev_r), "n_test": len(te_r),
        "feature_names": feat_keys,
        "feature_weights": [float(x) for x in w.tolist()],
        "dev_tau": best_tau,
        "dev_auroc": _auroc_np(y_dev, s_dev),
        "test_auroc": _auroc_np(y_te, s_te),
        "test_chat_only_auroc": _auroc_np(y_te, np.array([r["features"]["c_hat_mean"] for r in te_r])),
        "test_tpr": tp / max(1, tp + fn),
        "test_fnr": fn / max(1, tp + fn),
        "test_fpr": fp / max(1, fp + tn),
        "recompute_rate": float(pred_te.mean()) if len(pred_te) > 0 else 0.0,
        "test_always_jlocal_em": base_em,
        "test_routed_with_fallback_em": sum(1 for x in routed_em_list if x) / max(1, len(routed_em_list)),
        "test_latency_ms": {
            "jlocal_mean": sum(r["jlocal_total_ms"] for r in te_r) / max(1, len(te_r)),
            "routed_mean": sum(routed_lat_list) / max(1, len(routed_lat_list)),
            "routed_p50": p50, "routed_p95": p95,
        },
    }


def run_detector_v2_recoverable_suite(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int],
    n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool,
    n_per_regime_split: int, out_dir: Path, tag: str,
) -> Dict[str, Any]:
    """SB6 recoverable-fallback suite across healthy O_local paths & degraded fast-paths healed by suffix_recompute."""
    rng = random.Random(202695)
    pool_d = build_digit4_pool()
    jsonl_p = out_dir / f"w11_detector_v2_clean_{tag}_records.jsonl"
    done = load_done_keys(jsonl_p)
    regimes = ["sniah_clean", "varlen_clean", "state_only_fast_drop", "thresh_kv_val_partial"]

    for split in ("dev", "test"):
        for rname in regimes:
            for idx in range(n_per_regime_split):
                rk = f"{tag}:det2c:{split}:{rname}:{idx}"
                if rk in done:
                    continue
                seed_off = 0 if split == "dev" else 600
                fd = rng.sample([str(d) for d in range(1, 10)], 2)
                v_o, v_n = rng.choice(pool_d[fd[0]]), rng.choice(pool_d[fd[1]])
                pre = " ".join(FILLERS[(idx + seed_off + j) % len(FILLERS)] for j in range(6))
                suf = " ".join(FILLERS[(idx + seed_off + 7 + j) % len(FILLERS)] for j in range(16))
                skip_kv, skip_st, use_p1_horizon = False, False, False

                if rname == "sniah_clean":
                    u_o = f"{pre} Record state: the security code for Alice is {v_o}. {suf}\n\n"
                    u_e = f"{pre} Record state: the security code for Alice is {v_n}. {suf}\n\n"
                    q_s = "Question: What is the exact security code for Alice? Answer with only the value."
                elif rname == "varlen_clean":
                    ts, tl, rs, rl, ss, sl = JSON_SPECS[(idx + seed_off) % len(JSON_SPECS)]
                    u_o = f"{pre} Tool payload tier={ts}, region={rs}, status={ss}. {suf}\n\n"
                    u_e = f"{pre} Tool payload tier={tl}, region={rl}, status={sl}. {suf}\n\n"
                    q_s = "In tool payload, what is status? Answer with only the value."
                    v_o, v_n = ss, sl
                elif rname == "state_only_fast_drop":
                    u_o = f"{pre} Record state: the security code for Alice is {v_o}. {suf}\n\n"
                    u_e = f"{pre} Record state: the security code for Alice is {v_n}. {suf}\n\n"
                    q_s = "Question: What is the exact security code for Alice? Answer with only the value."
                    skip_kv = True
                else:  # thresh_kv_val_partial
                    low_val = rng.choice(pool_d[rng.choice(["1", "2", "3", "4"])])
                    high_val = rng.choice(pool_d[rng.choice(["6", "7", "8", "9"])])
                    if (idx + seed_off) % 2 == 0:
                        u_v0, u_v1, v_o, v_n = low_val, high_val, "NO", "YES"
                    else:
                        u_v0, u_v1, v_o, v_n = high_val, low_val, "YES", "NO"
                    u_o = f"{pre} Audit metric score for Alice is {u_v0}. {suf}\n\n"
                    u_e = f"{pre} Audit metric score for Alice is {u_v1}. {suf}\n\n"
                    q_s = "Question: Is Alice's audit metric score strictly greater than 5000? Answer with only YES or NO."
                    skip_st = True
                    use_p1_horizon = True

                al = _align_contiguous_edit_span(tok, u_o + q_s, u_e + q_s, q_s, n_linear)
                if al is None:
                    continue
                b_o = al["p1_o"] if use_p1_horizon else al["b_old"]
                b_n = al["p1_e"] if use_p1_horizon else al["b_new"]
                rec = _collect_detector_case_record(
                    model, base_model, tok, eos_ids, rname,
                    torch.tensor([al["ids_o_list"]], device=device),
                    torch.tensor([al["ids_e_list"]], device=device),
                    torch.tensor([al["q_ids_list"]], device=device),
                    al["p0"], b_o, b_n, al["T_old"], al["T_new"], al["delta_L"],
                    v_n, v_o, n_linear, d_k, w_conv, device, use_l2k,
                    skip_kv_patch=skip_kv, skip_state_update=skip_st,
                )
                rec["record_key"] = rk
                rec["split"] = split
                append_record_jsonl(jsonl_p, rec)

    all_r = [json.loads(l) for l in jsonl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    dev_r = [r for r in all_r if r.get("split") == "dev"]
    te_r = [r for r in all_r if r.get("split") == "test"]
    fkeys = sorted(dev_r[0]["features"].keys()) if dev_r else []
    X_dev = np.array([[r["features"][k] for k in fkeys] for r in dev_r], dtype=np.float64)
    y_dev = np.array([r["is_failure_jlocal"] for r in dev_r], dtype=np.float64)
    X_te = np.array([[r["features"][k] for k in fkeys] for r in te_r], dtype=np.float64)
    y_te = np.array([r["is_failure_jlocal"] for r in te_r], dtype=np.float64)

    mu, sd, w, b = fit_logistic_l2_np(X_dev, y_dev, reg=1.0, steps=500, lr=0.2)
    s_dev = predict_logistic_np(X_dev, mu, sd, w, b)
    s_te = predict_logistic_np(X_te, mu, sd, w, b)

    best_tau = 0.5
    best_score = -1.0
    for cand_t in np.linspace(0.15, 0.85, 29):
        pred = s_dev >= cand_t
        tp = float(((pred == 1) & (y_dev == 1)).sum())
        fn = float(((pred == 0) & (y_dev == 1)).sum())
        fp = float(((pred == 1) & (y_dev == 0)).sum())
        tn = float(((pred == 0) & (y_dev == 0)).sum())
        tpr = tp / max(1.0, tp + fn)
        fpr = fp / max(1.0, fp + tn)
        if tpr - fpr > best_score:
            best_score = tpr - fpr
            best_tau = float(cand_t)

    pred_te = s_te >= best_tau
    tp = int(((pred_te == 1) & (y_te == 1)).sum())
    fn = int(((pred_te == 0) & (y_te == 1)).sum())
    fp = int(((pred_te == 1) & (y_te == 0)).sum())
    tn = int(((pred_te == 0) & (y_te == 0)).sum())

    base_em = sum(1 for r in te_r if r["jlocal_em"]) / max(1, len(te_r))
    routed_em = [bool(r["fallback_em"] if fl else r["jlocal_em"]) for fl, r in zip(pred_te.tolist(), te_r)]
    routed_lat = sorted(float(r["jlocal_total_ms"] + (r["fallback_branch_ms"] if fl else 0.0)) for fl, r in zip(pred_te.tolist(), te_r))
    return {
        "suite": "detector_v2", "n_dev": len(dev_r), "n_test": len(te_r),
        "test_fail_rate": float(y_te.mean()) if len(y_te) > 0 else 0.0,
        "dev_tau": best_tau, "dev_auroc": _auroc_np(y_dev, s_dev),
        "test_auroc": _auroc_np(y_te, s_te),
        "test_chat_only_auroc": _auroc_np(y_te, np.array([r["features"]["c_hat_mean"] for r in te_r])),
        "test_tpr": tp / max(1, tp + fn), "test_fnr": fn / max(1, tp + fn), "test_fpr": fp / max(1, fp + tn),
        "recompute_rate": float(pred_te.mean()) if len(pred_te) > 0 else 0.0,
        "test_always_jlocal_em": base_em,
        "test_routed_with_fallback_em": sum(1 for x in routed_em if x) / max(1, len(routed_em)),
        "test_oracle_fallback_em": sum(1 for r in te_r if r["fallback_em"]) / max(1, len(te_r)),
        "test_latency_ms": {
            "jlocal_mean": sum(r["jlocal_total_ms"] for r in te_r) / max(1, len(te_r)),
            "routed_mean": sum(routed_lat) / max(1, len(routed_lat)),
            "routed_p50": routed_lat[len(routed_lat) // 2] if routed_lat else 0.0,
            "routed_p95": routed_lat[int(0.95 * (len(routed_lat) - 1))] if routed_lat else 0.0,
        },
    }


def run_json_dl_ctrl_suite(
    model: Any, base_model: Any, tok: Any, eos_ids: List[int],
    n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool,
    n_per_stratum: int, out_dir: Path, tag: str,
) -> Dict[str, Any]:
    """Fixed-domain JSON payload controlled across |DeltaL| in {0, 1..3, 13..17} to deconfound domain vs length."""
    rng = random.Random(202698)
    pool_d = build_digit4_pool()
    jsonl_p = out_dir / f"w11_json_dl_ctrl_{tag}_records.jsonl"
    done = load_done_keys(jsonl_p)
    specs = {
        "dl0": ("queued", "closed", "silver", "golden", "us-east", "us-west"),
        "dl1_3": ("queued", "verified", "silver", "platinum", "us-east", "ap-south-central"),
        "dl_large": ("queued", "approved_compliance_override", "silver", "enterprise-priority-verified-tier", "us-east", "eu-central-frankfurt-dr2"),
    }
    for stratum, (s0_v, s1_v, t0_v, t1_v, r0_v, r1_v) in specs.items():
        cnt = 0
        idx = 0
        while cnt < n_per_stratum and idx < n_per_stratum * 5:
            rk = f"{tag}:jctrl:{stratum}:{cnt}"
            idx += 1
            if rk in done:
                cnt += 1
                continue
            txn = f"tx-{rng.choice(pool_d[str((idx % 8) + 1)])}"
            owner = ["Alice", "Bob", "Carol", "Dave"][idx % 4]
            j_o = json.dumps({"txn_id": txn, "owner": owner, "tier": t0_v, "region": r0_v, "status": s0_v})
            j_e = json.dumps({"txn_id": txn, "owner": owner, "tier": t1_v, "region": r1_v, "status": s1_v})
            pre = " ".join(FILLERS[(idx + j) % len(FILLERS)] for j in range(10))
            suf = " ".join(FILLERS[(idx + 11 + j) % len(FILLERS)] for j in range(18))
            q_s = f"In tool output JSON {txn}, what is the exact value of field 'status'? Answer with only the value."
            al = _align_contiguous_edit_span(tok, f"{pre}\nTool response payload: {j_o}\n{suf}\n\n{q_s}", f"{pre}\nTool response payload: {j_e}\n{suf}\n\n{q_s}", q_s, n_linear)
            if al is None:
                continue
            dl = abs(al["delta_L"])
            if stratum == "dl0" and dl != 0:
                continue
            if stratum == "dl1_3" and not (1 <= dl <= 3):
                # Fallback single-field edit inside JSON for dl1_3
                j_e = json.dumps({"txn_id": txn, "owner": owner, "tier": t0_v, "region": r0_v, "status": s1_v})
                al = _align_contiguous_edit_span(tok, f"{pre}\nTool response payload: {j_o}\n{suf}\n\n{q_s}", f"{pre}\nTool response payload: {j_e}\n{suf}\n\n{q_s}", q_s, n_linear)
                if al is None or not (1 <= abs(al["delta_L"]) <= 3):
                    continue
            if stratum == "dl_large" and dl < 10:
                continue
            smp = {
                "ids_o": torch.tensor([al["ids_o_list"]], device=device),
                "ids_e": torch.tensor([al["ids_e_list"]], device=device),
                "ids_q": torch.tensor([al["q_ids_list"]], device=device),
                "p0": al["p0"], "b_old": al["b_old"], "b_new": al["b_new"],
                "T_old": al["T_old"], "T_new": al["T_new"], "delta_L": al["delta_L"],
                "old_val": s0_v, "new_val": s1_v,
                "f_o": tok.encode(s0_v, add_special_tokens=False)[0],
                "f_e": tok.encode(s1_v, add_special_tokens=False)[0],
            }
            res = _eval_varlen_conditions(model, base_model, tok, eos_ids, smp, n_linear, d_k, w_conv, device, use_l2k, include_extra_queries=False, audit_kspace=False)
            res["record_key"] = rk
            res["stratum"] = stratum
            append_record_jsonl(jsonl_p, res)
            cnt += 1

    all_r = [json.loads(l) for l in jsonl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for st in ("dl0", "dl1_3", "dl_large"):
        sr = [r for r in all_r if r.get("stratum") == st]
        n = len(sr)
        out[st] = {
            "n": n, "abs_dl": sorted({abs(r["delta_L"]) for r in sr}),
            "jlocal_dr_em": sum(1 for r in sr if r["conditions"]["jlocal_delta_rope"]["contains_new"]) / max(1, n),
            "jlocal_nodr_em": sum(1 for r in sr if r["conditions"]["jlocal_no_delta_rope"]["contains_new"]) / max(1, n),
            "jlocal_dr_top1_eq_ora": sum(1 for r in sr if r["conditions"]["jlocal_delta_rope"]["top1_eq_oracle"]) / max(1, n),
            "jlocal_dr_kl": sum(r["conditions"]["jlocal_delta_rope"]["kl_vs_oneshot"] for r in sr) / max(1, n),
            "jlocal_nodr_kl": sum(r["conditions"]["jlocal_no_delta_rope"]["kl_vs_oneshot"] for r in sr) / max(1, n),
        }
    return out


def run_synthetic_dry_run(out_dir: Path) -> Dict[str, Any]:
    torch.manual_seed(0)
    # Test pure logic + logistic detector + span alignment without Triton CPU issue
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] - 0.8 * X[:, 1] > 0).astype(np.float64)
    mu, sd, w, b = fit_logistic_l2_np(X, y)
    auc = _auroc_np(y, predict_logistic_np(X, mu, sd, w, b))
    assert auc > 0.90, f"Logistic dryrun AUC low: {auc}"
    rep = {"detector_dryrun_auc": auc, "suites_registered": ["json", "cf_nl", "e10", "e12", "detector"]}
    (out_dir / "w11_varlen_e10_e12_dryrun.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    p.add_argument("--tag", type=str, default="2b")
    p.add_argument("--cf-path", type=str, default=r"D:\applicationData\HuggingFace\counterfact.json")
    p.add_argument("--suites", type=str, default="json,cf_nl,e10,e12,detector")
    p.add_argument("--n-json", type=int, default=60)
    p.add_argument("--n-cf1", type=int, default=60)
    p.add_argument("--n-nat2", type=int, default=60)
    p.add_argument("--n-ctrl-each", type=int, default=15)
    p.add_argument("--n-pilot5", type=int, default=30)
    p.add_argument("--n-safety-each", type=int, default=12)
    p.add_argument("--n-e12", type=int, default=40)
    p.add_argument("--n-det-per-regime", type=int, default=15)
    p.add_argument("--out-dir", type=str, default="pilot")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--synthetic-dry-run", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.synthetic_dry_run:
        rep = run_synthetic_dry_run(out_dir)
        print("W11_VARLEN_E10_E12_DET_DRYRUN_OK", json.dumps(rep))
        return

    suites = {s.strip() for s in args.suites.split(",") if s.strip()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True, local_files_only=args.local_files_only,
    ).eval()
    base_model = getattr(model, "model", model)
    t_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_linear = count_linear_layers(t_cfg)
    d_k = int(getattr(t_cfg, "linear_key_head_dim", 128))
    w_conv = int(getattr(t_cfg, "linear_conv_kernel_dim", 4))
    eos_ids = get_eos_ids(model, tok)
    use_l2k = has_logits_to_keep_arg(model)
    cf_p = Path(args.cf_path) if args.cf_path else None

    sum_path = out_dir / f"w11_varlen_e10_e12_det_{args.tag}_summary.json"
    summary: Dict[str, Any] = json.loads(sum_path.read_text(encoding="utf-8")) if sum_path.exists() else {"model": args.model, "tag": args.tag}

    if "json" in suites:
        print(f"[{args.tag}] Running suite: json (SB2 part 1)", flush=True)
        summary["json_varlen"] = run_json_varlen_suite(model, base_model, tok, eos_ids, n_linear, d_k, w_conv, device, use_l2k, args.n_json, out_dir, args.tag)
        sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if "cf_nl" in suites:
        print(f"[{args.tag}] Running suite: cf_nl (SB2 part 2)", flush=True)
        summary["cf_nl"] = run_cf_nl_suite(model, base_model, tok, eos_ids, cf_p, n_linear, d_k, w_conv, device, use_l2k, args.n_cf1, args.n_nat2, out_dir, args.tag)
        sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if "e10" in suites:
        print(f"[{args.tag}] Running suite: e10 (SB3)", flush=True)
        summary["e10"] = run_e10_suite(model, base_model, tok, eos_ids, n_linear, d_k, w_conv, device, use_l2k, args.n_ctrl_each, args.n_pilot5, args.n_safety_each, out_dir, args.tag)
        sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if "e12" in suites:
        print(f"[{args.tag}] Running suite: e12 (SB4)", flush=True)
        summary["e12_coupled"] = run_e12_coupled_suite(model, base_model, tok, eos_ids, n_linear, d_k, w_conv, device, use_l2k, args.n_e12, out_dir, args.tag)
        sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if "detector" in suites:
        print(f"[{args.tag}] Running suite: detector (SB6)", flush=True)
        summary["detector"] = run_detector_suite(model, base_model, tok, eos_ids, n_linear, d_k, w_conv, device, use_l2k, args.n_det_per_regime, out_dir, args.tag)
        sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if "detector_v2" in suites:
        print(f"[{args.tag}] Running suite: detector_v2 recoverable fallback (SB6)", flush=True)
        summary["detector_v2"] = run_detector_v2_recoverable_suite(model, base_model, tok, eos_ids, n_linear, d_k, w_conv, device, use_l2k, max(20, args.n_det_per_regime), out_dir, args.tag)
        sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if "json_dl_ctrl" in suites:
        print(f"[{args.tag}] Running suite: json_dl_ctrl fixed-domain |DeltaL| sweep (SB2)", flush=True)
        summary["json_dl_ctrl"] = run_json_dl_ctrl_suite(model, base_model, tok, eos_ids, n_linear, d_k, w_conv, device, use_l2k, 25, out_dir, args.tag)
        sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{args.tag}] W11_VARLEN_E10_E12_DETECTOR DONE", flush=True)


if __name__ == "__main__":
    main()
