"""Stage-C Unified Post-Hoc Statistical Aggregator (`w12_c_posthoc_stats.py`).

Collects and verifies all Stage C experimental artifacts strictly respecting audit & Option-A rules:
  - Blocker B1 fix: distinguishes E5 total-context crossover (`T_ctx=242` at grid index 0 where actual
    suffix lengths were 49 tok on 9B and 67 tok on 2B) from E8 standalone edit microbench (`264.7 / 302.7 /
    1219.9` tok on 2B L4), neither measuring a live-routing boundary. Labels W12-2 latencies as timed
    GPU-path means (excluding pre-timer H2D, cache clones, and noting KV concat/RoPE rotation).
  - Segment 1 (`W12-2` + `G4` held-out unseen templates): separates `2B (frozen OSC)` [2K & separate
    16K run on reused 100 case keys] and `9B (frozen LR, exact dev midpoint 0.48272, operating point 0.50)`.
    Audits `unseen_coupled_e12_single` under BOTH `draft_value_em` and `manifest_fidelity_old` (with pooled
    relabeled counts `2B 179->218->224/225, FP 24/179` & `9B 166->213->225/225, FP 11/166`, exact McNemar `p`,
    and `always_suffix` KL), reports `healable_only` benign FP (`9B 1/156 = 0.64%`), offline 9B passive
    `OSC@0.40` replay, frozen-141 vs sweep-union-144 rows, suppresses `pr_auc` on single-class strata, and
    specifies upper-order-statistic quantiles for `live_routed` and `always_suffix`.
  - Segment 2 (`W12-4` controlled microbench & in-memory int8 stress): distinguishes the synthetic
    random-tensor storage+H2D+dequant microbench (`nvme_read_cold/warm` `4.39/2.47 ms` vs summed `8.99/7.08 ms`
    on L4 9B int8; `posix_fadvise` logs syscall without page-residency check) from the pinned in-memory
    buffer checkpoint-restore accuracy stress suite (no NVMe disk reads); recomputes restore-only (`ckpt_bytes>0`)
    `J-local_fidelity` (explicitly distinct from E5 `J-full` G1b), same-stride exact McNemar flip p-values,
    mean/max KL, `9B collision 19/20 per cell >0.5`, and duplicate C2048/collision disclosures.
  - Segment 3 (`W12-5` Exploratory Extension): reads harmonized & task-ID-joined
    `w12_c_offline_pairing_n200_results.json`.
  - Segment 4 (`W11-2` Jamba-3B): records `INCONCLUSIVE` verdict (`num_experts=1` dense non-MoE variant),
    omitted logits/hashes, two-chunked-path `G1a=0` caveat, and closed-template Gate-3 failure.
  - Segment 5 (`M5` Mamba-2 & HYPIC external baselines): preserves Wave 11 scope and timer boundaries.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PILOT_DIR = Path(__file__).resolve().parent


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> List[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) * (0.5 ** n) for i in range(0, k + 1)))


def _audit_w13_records(records_path: Optional[Path], include_passive_osc: bool = False) -> Optional[Dict[str, Any]]:
    if not records_path or not records_path.exists():
        return None
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = len(rows)
    if n == 0:
        return None

    def _quantiles(sub: List[Dict[str, Any]]) -> Dict[str, float]:
        m = len(sub)
        rt = sorted(float(r["timing_ms"]["live_routed_e2e_ms"]) for r in sub)
        sf = sorted(float(r["timing_ms"]["always_suffix_e2e_ms"]) for r in sub)
        return {
            "n": m,
            "live_routed_e2e_mean": float(sum(rt) / m),
            "live_routed_e2e_p50": float(rt[m // 2]),
            "live_routed_e2e_p95": float(rt[min(m - 1, int(0.95 * m))]),
            "always_suffix_e2e_mean": float(sum(sf) / m),
            "always_suffix_e2e_p50": float(sf[m // 2]),
            "always_suffix_e2e_p95": float(sf[min(m - 1, int(0.95 * m))]),
        }

    suites_q: Dict[str, Any] = {}
    for sname in sorted({r["suite"] for r in rows}):
        suites_q[sname] = _quantiles([r for r in rows if r["suite"] == sname])

    e12 = [r for r in rows if r["regime"] == "unseen_coupled_e12_single"]
    non_e12 = [r for r in rows if r["regime"] != "unseen_coupled_e12_single"]
    e12_audit: Optional[Dict[str, Any]] = None
    if e12:
        b_flip = sum(1 for r in e12 if r["always_jlocal"]["em"] and not r["live_routed"]["em"])
        c_flip = sum(1 for r in e12 if not r["always_jlocal"]["em"] and r["live_routed"]["em"])
        relab_j = sum(bool(r["always_jlocal"]["em"]) for r in non_e12) + sum(bool(r["always_jlocal"]["leak"]) for r in e12)
        relab_r = sum(bool(r["live_routed"]["em"]) for r in non_e12) + sum(bool(r["live_routed"]["leak"]) for r in e12)
        relab_s = sum(bool(r["always_suffix"]["em"]) for r in non_e12) + sum(bool(r["always_suffix"]["leak"]) for r in e12)
        relab_fp = (
            sum(1 for r in non_e12 if int(r["is_failure_jlocal"]) == 0 and int(r["detector_flag"]) == 1)
            + sum(1 for r in e12 if bool(r["always_jlocal"]["leak"]) and int(r["detector_flag"]) == 1)
        )
        relab_bn = (
            sum(1 for r in non_e12 if int(r["is_failure_jlocal"]) == 0)
            + sum(1 for r in e12 if bool(r["always_jlocal"]["leak"]))
        )
        e12_audit = {
            "n": len(e12),
            "query_target_note": "Query asks for active permit code per the final signed manifest, which retains old_v; draft_value scorer rewards new_v.",
            "draft_value_em_counts": {
                "always_jlocal": sum(bool(r["always_jlocal"]["em"]) for r in e12),
                "live_routed": sum(bool(r["live_routed"]["em"]) for r in e12),
                "always_suffix": sum(bool(r["always_suffix"]["em"]) for r in e12),
            },
            "manifest_fidelity_old_counts": {
                "always_jlocal": sum(bool(r["always_jlocal"]["leak"]) for r in e12),
                "live_routed": sum(bool(r["live_routed"]["leak"]) for r in e12),
                "always_suffix": sum(bool(r["always_suffix"]["leak"]) for r in e12),
            },
            "pooled_relabeled_to_manifest": {
                "n_total": n,
                "always_jlocal": relab_j,
                "live_routed": relab_r,
                "always_suffix": relab_s,
                "benign_fp": relab_fp,
                "benign_n": relab_bn,
                "benign_fp_rate": float(relab_fp / relab_bn) if relab_bn else None,
            },
            "always_suffix_mean_kl_vs_edited_oracle": float(sum(float(r["always_suffix"]["kl"]) for r in e12) / len(e12)),
            "routed_vs_jlocal_flips": {"jl_only": b_flip, "routed_only": c_flip, "exact_mcnemar_p": _exact_mcnemar_p(b_flip, c_flip)},
        }

    flagged = [r for r in rows if bool(r["live_recomputed"])]
    tau_em_leak_agree = sum(
        1 for r in flagged
        if r["live_routed"]["em"] == r["always_suffix"]["em"] and r["live_routed"]["leak"] == r["always_suffix"]["leak"]
    )
    tau_kl_differ = sum(1 for r in flagged if float(r["live_routed"]["kl"]) != float(r["always_suffix"]["kl"]))

    y_all = [int(r["is_failure_jlocal"]) for r in rows]
    n_pos = sum(y_all)
    n_neg = n - n_pos
    benign_fp = sum(1 for r in rows if int(r["is_failure_jlocal"]) == 0 and int(r["detector_flag"]) == 1)
    heal_benign = [r for r in rows if int(r["is_failure_jlocal"]) == 0 and bool(r.get("healable_by_suffix", False))]
    heal_benign_fp = sum(1 for r in heal_benign if int(r["detector_flag"]) == 1)
    jloc_em_n = sum(1 for r in rows if bool(r["always_jlocal"]["em"]))
    jloc_em_fp = sum(1 for r in rows if bool(r["always_jlocal"]["em"]) and int(r["detector_flag"]) == 1)

    passive_osc_replay: Optional[Dict[str, Any]] = None
    if include_passive_osc:
        tp_o = sum(1 for r in rows if float(r["passive_osc_score"]) >= 0.40 and int(r["is_failure_jlocal"]) == 1)
        fp_o = sum(1 for r in rows if float(r["passive_osc_score"]) >= 0.40 and int(r["is_failure_jlocal"]) == 0)
        em_o = sum(
            1 for r in rows
            if (r["always_suffix"]["em"] if float(r["passive_osc_score"]) >= 0.40 else r["always_jlocal"]["em"])
        )
        passive_osc_replay = {
            "tau_osc": 0.40,
            "tp": tp_o,
            "n_pos": n_pos,
            "fp": fp_o,
            "n_neg": n_neg,
            "routed_contains_new_count": em_o,
            "routed_contains_new_rate": float(em_o / n),
        }

    def _ap_tie_aware(scores_l: List[float], labels_l: List[int]) -> Optional[float]:
        n_p = sum(labels_l)
        if n_p == 0 or n_p == len(labels_l):
            return None
        by_sc: Dict[float, List[int]] = {}
        for s_val, y_val in zip(scores_l, labels_l):
            by_sc.setdefault(float(s_val), []).append(int(y_val))
        cum_tp = 0
        cum_tot = 0
        ap = 0.0
        for s_val in sorted(by_sc.keys(), reverse=True):
            ys = by_sc[s_val]
            tp_g = sum(ys)
            cum_tp += tp_g
            cum_tot += len(ys)
            if tp_g > 0:
                ap += (cum_tp / cum_tot) * (tp_g / n_p)
        return float(ap)

    det_scores = [float(r["detector_score"]) for r in rows] if rows and "detector_score" in rows[0] else None
    ap_tie_det = _ap_tie_aware(det_scores, y_all) if det_scores is not None else None
    ap_tie_osc = (
        _ap_tie_aware([float(r["passive_osc_score"]) for r in rows], y_all)
        if include_passive_osc and rows and "passive_osc_score" in rows[0]
        else None
    )

    return {
        "combined_quantiles_ms": _quantiles(rows),
        "suites_quantiles_ms": suites_q,
        "benign_fp": benign_fp,
        "benign_n": n_neg,
        "benign_fp_wilson95": _wilson_ci(benign_fp, n_neg),
        "healable_only_benign_fp": heal_benign_fp,
        "healable_only_benign_n": len(heal_benign),
        "healable_only_benign_fp_rate": float(heal_benign_fp / len(heal_benign)) if heal_benign else None,
        "healable_only_benign_fp_wilson95": _wilson_ci(heal_benign_fp, len(heal_benign)) if heal_benign else None,
        "jlocal_contains_new_benign_fp_check": {"fp": jloc_em_fp, "n": jloc_em_n},
        "n_flagged": len(flagged),
        "tau_sweep_standalone_check": {
            "flagged_agree_on_em_and_leak": tau_em_leak_agree,
            "flagged_differ_in_kl": tau_kl_differ,
        },
        "e12_dual_label_audit": e12_audit,
        "passive_osc_at_040_replay": passive_osc_replay,
        "pr_auc_tie_aware_detector": ap_tie_det,
        "pr_auc_tie_aware_passive_osc": ap_tie_osc,
    }


def _audit_w12_4_records(records_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Recompute restore-only (ckpt_bytes>0) strict rates, paired flips, max KL, contains_new, and duplicate overlap."""
    if not records_path or not records_path.exists():
        return None
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_sc: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        by_sc.setdefault((r["suite"], r["cell"]), []).append(r)

    jlocal_fidelity_stress: Dict[str, Any] = {}
    collision_stress: Dict[str, Any] = {}
    for (suite, cell), grp in sorted(by_sc.items()):
        rest = [x for x in grp if not x.get("kv_only", False) and float(x["ckpt_bytes"]) > 0]
        kls_all = [float(x["kl"]) for x in grp]
        kls_rest = [float(x["kl"]) for x in rest]
        s_all = sum(bool(x["g1b_strict"]) for x in grp)
        s_rest = sum(bool(x["g1b_strict"]) for x in rest)
        em_all = sum(bool(x["em"]) for x in grp)
        leak_all = sum(bool(x["old_leak"]) for x in grp)
        rec = {
            "n_all": len(grp),
            "n_restore_only": len(rest),
            "jlocal_fidelity_strict_1e4_all_count": s_all,
            "jlocal_fidelity_strict_1e4_all_rate": s_all / len(grp),
            "jlocal_fidelity_strict_1e4_restore_only_count": s_rest if rest else None,
            "jlocal_fidelity_strict_1e4_restore_only_rate": (s_rest / len(rest)) if rest else None,
            "jlocal_fidelity_strict_1e4_restore_only_ci95": _wilson_ci(s_rest, len(rest)) if rest else None,
            "contains_new_rate": em_all / len(grp),
            "old_leak_rate": leak_all / len(grp),
            "mean_kl_all": float(np.mean(kls_all)),
            "max_kl_all": float(np.max(kls_all)),
            "mean_kl_restore_only": float(np.mean(kls_rest)) if rest else None,
            "max_kl_restore_only": float(np.max(kls_rest)) if rest else None,
            "max_abs_logit_diff_gt_05_count": sum(1 for x in grp if float(x.get("max_abs_logit_diff", 0.0)) > 0.5),
            "note": (
                "C2048 depth=0.75 restored rows share exact c_pos=2048 and bit-identical outputs with C1024"
                if cell == "C2048_int8"
                else None
            ),
        }
        if suite == "collision_stress":
            collision_stress[cell] = rec
        else:
            jlocal_fidelity_stress[cell] = rec

    g_rows = [r for r in rows if r["suite"] != "collision_stress"]
    by_cell_key: Dict[str, Dict[Tuple[int, float], Dict[str, Any]]] = {}
    for r in g_rows:
        by_cell_key.setdefault(r["cell"], {})[(int(r["sample_idx"]), float(r["depth"]))] = r
    paired_same_stride: Dict[str, Any] = {}
    for base_c, int8_c in [("C512_fp32_rec", "C512_int8"), ("C1024_bf16", "C1024_int8")]:
        ca, cb = by_cell_key.get(base_c, {}), by_cell_key.get(int8_c, {})
        keys = [k for k in sorted(ca) if k in cb and float(ca[k]["ckpt_bytes"]) > 0 and float(cb[k]["ckpt_bytes"]) > 0]
        p_to_f = sum(1 for k in keys if ca[k]["g1b_strict"] and not cb[k]["g1b_strict"])
        f_to_p = sum(1 for k in keys if not ca[k]["g1b_strict"] and cb[k]["g1b_strict"])
        paired_same_stride[f"{base_c}_vs_{int8_c}"] = {
            "n_restore_matched": len(keys),
            "pass_to_fail_flips": p_to_f,
            "fail_to_pass_flips": f_to_p,
            "exploratory_exact_mcnemar_p": _exact_mcnemar_p(p_to_f, f_to_p),
        }

    c1024_d75 = {(x["sample_idx"], x["depth"]): x for x in g_rows if x["cell"] == "C1024_int8" and x["depth"] == 0.75}
    c2048_d75 = {(x["sample_idx"], x["depth"]): x for x in g_rows if x["cell"] == "C2048_int8" and x["depth"] == 0.75}
    d75_exact = sum(
        a["kl"] == b["kl"] and a["max_abs_logit_diff"] == b["max_abs_logit_diff"] and a["ckpt_bytes"] == b["ckpt_bytes"]
        for k, a in c1024_d75.items()
        for b in [c2048_d75.get(k)]
        if b is not None
    )
    ec_cells = sorted(collision_stress.keys())
    base_ec = [(x["sample_idx"], x["kl"], x["ckpt_bytes"], x["em"]) for x in rows if x["suite"] == "collision_stress" and x["cell"] == ec_cells[0]]
    ec_identical = all(
        [(x["sample_idx"], x["kl"], x["ckpt_bytes"], x["em"]) for x in rows if x["suite"] == "collision_stress" and x["cell"] == c] == base_ec
        for c in ec_cells[1:]
    )
    return {
        "threshold_terminology_note": "Strict <=1e-4 threshold evaluates J-local_fidelity vs oneshot_oracle (distinct from E5 G1b, which compares J-full vs oneshot_oracle).",
        "accuracy_execution_path_note": "Restores quantized buffers from pinned host memory in _eval_ckpt_cell (no NVMe disk reads in accuracy runs).",
        "jlocal_fidelity_restore_only_audit": jlocal_fidelity_stress,
        "paired_same_stride_flips": paired_same_stride,
        "collision_stress_audit": {
            "effective_unique_n": len(base_ec),
            "all_5_cells_identical_ckpt_bytes_0": ec_identical,
            "cells": collision_stress,
        },
        "c2048_depth075_identical_to_c1024_count": f"{d75_exact}/{len(c1024_d75)}",
    }


def _load_opt(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _extract_block(blk: Dict[str, Any]) -> Dict[str, Any]:
    n = int(blk["n"])
    n_fail = blk.get("n_fail_jlocal")
    single_class = (n_fail is not None) and (int(n_fail) == 0 or int(n_fail) == n)
    return {
        "n": n,
        "n_fail_jlocal": n_fail,
        "flag_rate": blk["flag_rate"],
        "recompute_rate": blk["recompute_rate"],
        "recompute_rate_benign_fp": blk["recompute_rate_benign_fp"],
        "always_jlocal_contains_new": blk["always_jlocal"]["em"],
        "live_routed_contains_new": blk["live_routed"]["em"],
        "always_suffix_contains_new": blk["always_suffix"]["em"],
        "healable_only_note": "Excludes unhealable unseen_coupled_e12_single by generator label (by-construction parity where always_suffix=0).",
        "healable_only": blk.get("healable_only"),
        "always_jlocal_leak": blk["always_jlocal"]["leak_rate"],
        "live_routed_leak": blk["live_routed"]["leak_rate"],
        "always_suffix_leak": blk["always_suffix"]["leak_rate"],
        "always_jlocal_kl": blk["always_jlocal"]["mean_kl"],
        "live_routed_kl": blk["live_routed"]["mean_kl"],
        "always_suffix_kl": blk["always_suffix"]["mean_kl"],
        "latency_ms": blk["latency_ms"],
        "tp": blk.get("tp"),
        "fp": blk.get("fp"),
        "fn": blk.get("fn"),
        "tn": blk.get("tn"),
        "tpr": blk.get("tpr"),
        "fpr": blk.get("fpr"),
        "auroc": None if single_class else blk.get("auroc"),
        "pr_auc": None if single_class else blk.get("pr_auc"),
        "pr_auc_note": "Rank-based average precision (_pr_auc_np; returns 0.112407 on tied 16K LR scores vs tie-aware 0.200 in record_level_audit.pr_auc_tie_aware_detector); suppressed on single-class strata.",
    }


def _extract_fb(
    d: Optional[Dict[str, Any]],
    tau_audit: Optional[Dict[str, Any]] = None,
    record_audit: Optional[Dict[str, Any]] = None,
    is_plateau: bool = True,
) -> Optional[Dict[str, Any]]:
    if not d:
        return None
    tau_key = "tau_plateau_audit" if is_plateau else "tau_sensitivity_curve"
    out: Dict[str, Any] = {
        "model": d["model"],
        "detector_mode": d["detector_mode"],
        "frozen_config": d["frozen_config"],
        "total_records": d["total_records"],
        "combined": _extract_block(d["overall"]),
        tau_key: d.get("tau_plateau_audit") or (tau_audit or {}).get("tau_plateau_audit"),
        "record_level_audit": record_audit,
        "suites": {},
    }
    for sname, sdict in d.get("suites", {}).items():
        out["suites"][sname] = {
            "overall": _extract_block(sdict["overall"]),
            "by_regime": {reg: _extract_block(rblk) for reg, rblk in sdict.get("by_regime", {}).items()},
            tau_key: sdict.get("tau_plateau_audit") or (tau_audit or {}).get("suites", {}).get(sname, {}).get("tau_plateau_audit"),
        }
    return out


def _extract_optb_offline_osc_16k(records_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not records_path or not records_path.exists():
        return None
    rows = [json.loads(l) for l in records_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return None
    n = len(rows)
    pos = [r for r in rows if int(r["is_failure_jlocal"]) == 1]
    neg = [r for r in rows if int(r["is_failure_jlocal"]) == 0]
    flags = [float(r["passive_osc_score"]) >= 0.40 for r in rows]
    rt = sorted(
        float(r["timing_ms"]["live_routed_e2e_ms"]) + (float(r["timing_ms"]["always_suffix_e2e_ms"]) if fl else 0.0)
        for r, fl in zip(rows, flags)
    )
    sf = sorted(float(r["timing_ms"]["always_suffix_e2e_ms"]) for r in rows)
    tp = sum(1 for r, fl in zip(rows, flags) if fl and int(r["is_failure_jlocal"]) == 1)
    fp = sum(1 for r, fl in zip(rows, flags) if fl and int(r["is_failure_jlocal"]) == 0)
    em_c = sum(1 for r, fl in zip(rows, flags) if (r["always_suffix"]["em"] if fl else r["always_jlocal"]["em"]))
    return {
        "arm_scope": "offline_counterfactual_replay_summing_live_routed_e2e_ms_and_standalone_always_suffix_e2e_ms_on_flagged_rows",
        "accuracy_metric_note": "contains_new (8-token substring containment), NOT strict exact-string match",
        "n": n,
        "pos_n": len(pos),
        "neg_n": len(neg),
        "pos_lr_min_max": [min(float(r["detector_score"]) for r in pos), max(float(r["detector_score"]) for r in pos)] if pos else None,
        "pos_osc_min_max": [min(float(r["passive_osc_score"]) for r in pos), max(float(r["passive_osc_score"]) for r in pos)] if pos else None,
        "neg_osc_max": max(float(r["passive_osc_score"]) for r in neg) if neg else None,
        "passive_osc_040_tp": tp,
        "passive_osc_040_fp": fp,
        "passive_osc_040_reconstructed_contains_new_count": em_c,
        "passive_osc_040_reconstructed_contains_new_rate": em_c / n,
        "passive_osc_040_reconstructed_em_count": em_c,
        "passive_osc_040_reconstructed_em_rate": em_c / n,
        "passive_osc_040_reconstructed_mean_ms": sum(rt) / n,
        "passive_osc_040_reconstructed_p50_ms": rt[n // 2],
        "passive_osc_040_reconstructed_p95_ms": rt[min(n - 1, int(0.95 * n))],
        "always_suffix_mean_ms": sum(sf) / n,
        "always_suffix_p95_ms": sf[min(n - 1, int(0.95 * n))],
        "reconstructed_speedup_mean": (sum(sf) / n) / (sum(rt) / n),
    }


def _extract_dedup_w13_suite(jsonl_path: Path) -> Optional[Dict[str, Any]]:
    """Deduplicate exact duplicate serving prompts inside a legacy w13_fallback_runner.py JSONL."""
    if not jsonl_path.exists():
        return None
    from scipy.stats import beta as _beta

    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = set()
    dedup = []
    for r in rows:
        sig = (
            r.get("suite"),
            r.get("regime"),
            r.get("T_old"),
            r.get("T_new"),
            r.get("delta_L"),
            tuple(sorted((k, round(float(v), 8)) for k, v in (r.get("features") or {}).items())),
        )
        if sig not in seen:
            seen.add(sig)
            dedup.append(r)
    n_u = len(dedup)
    tp = sum(1 for r in dedup if int(r["is_failure_jlocal"]) == 1 and int(r["detector_flag"]) == 1)
    fp = sum(1 for r in dedup if int(r["is_failure_jlocal"]) == 0 and int(r["detector_flag"]) == 1)
    n_pos = sum(1 for r in dedup if int(r["is_failure_jlocal"]) == 1)
    n_neg = sum(1 for r in dedup if int(r["is_failure_jlocal"]) == 0)
    cn = sum(1 for r in dedup if r["live_routed"]["em"])
    rt = sorted(float(r["timing_ms"]["live_routed_e2e_ms"]) for r in dedup)
    sf = sorted(float(r["timing_ms"]["always_suffix_e2e_ms"]) for r in dedup)
    rt_mean = sum(rt) / max(1, n_u)
    sf_mean = sum(sf) / max(1, n_u)
    tpr_lb = float(_beta.ppf(0.025, tp, n_pos - tp + 1)) if tp > 0 else 0.0
    tnr_lb = float(_beta.ppf(0.025, n_neg - fp, fp + 1)) if (n_neg - fp) > 0 else 0.0
    return {
        "source_jsonl": jsonl_path.name,
        "raw_rows": len(rows),
        "unique_rows": n_u,
        "duplicate_rows_removed": len(rows) - n_u,
        "tp": tp,
        "fp": fp,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "ci_convention": "two_sided_95pct_clopper_pearson_lower_bound (alpha/2=0.025)",
        "tpr_cp95_lb": round(tpr_lb, 5),
        "tnr_cp95_lb": round(tnr_lb, 5),
        "joint_auc_cp95_lb": round(max(0.0, tpr_lb + tnr_lb - 1.0), 5),
        "contains_new_count": cn,
        "contains_new_rate": round(cn / max(1, n_u), 4),
        "live_routed_mean_ms": round(rt_mean, 3),
        "always_suffix_mean_ms": round(sf_mean, 3),
        "live_routed_p50_ms": round(rt[n_u // 2], 3) if n_u else 0.0,
        "live_routed_p95_ms": round(rt[min(n_u - 1, int(0.95 * n_u))], 3) if n_u else 0.0,
        "always_suffix_p50_ms": round(sf[n_u // 2], 3) if n_u else 0.0,
        "always_suffix_p95_ms": round(sf[min(n_u - 1, int(0.95 * n_u))], 3) if n_u else 0.0,
        "speedup_mean": round(sf_mean / max(1e-6, rt_mean), 3),
    }


def _extract_dedup_w13_16k_osc(jsonl_path: Path) -> Optional[Dict[str, Any]]:
    return _extract_dedup_w13_suite(jsonl_path)


def _tost_zero_var_safe_audit() -> Dict[str, Any]:
    """Independent zero-variance-safe & n<2-safe TOST verification (keeps frozen w12_c_offline_pairing_n200.py untouched)."""
    import numpy as _np
    from scipy import stats as _st

    def _safe_tost(a: _np.ndarray, b: _np.ndarray, raw_bound: float = 1e-5, lo_r: float = 0.80, hi_r: float = 1.25) -> Dict[str, Any]:
        n = len(a)
        d = a - b
        md = float(_np.mean(d)) if n > 0 else float("nan")
        lr = _np.log(_np.maximum(a, 1e-12)) - _np.log(_np.maximum(b, 1e-12))
        mlr = float(_np.mean(lr)) if n > 0 else float("nan")
        if n < 2:
            return {
                "n": n,
                "status": "insufficient_n_lt_2",
                "mean_diff": md,
                "gmr": math.exp(mlr) if n == 1 else None,
                "p_raw_tost": None,
                "p_log_tost": None,
            }
        df = n - 1
        se = float(_np.std(d, ddof=1)) / math.sqrt(n)
        if se > 0.0:
            p_raw = float(max(_st.t.sf((md + raw_bound) / se, df), _st.t.sf((raw_bound - md) / se, df)))
        else:
            p_raw = 0.0 if abs(md) < raw_bound else 1.0
        selr = float(_np.std(lr, ddof=1)) / math.sqrt(n)
        lo_b, hi_b = math.log(lo_r), math.log(hi_r)
        if selr > 0.0:
            p_log = float(max(_st.t.sf((mlr - lo_b) / selr, df), _st.t.sf((hi_b - mlr) / selr, df)))
        else:
            p_log = 0.0 if (lo_b < mlr < hi_b) else 1.0
        return {"n": n, "status": "ok", "mean_diff": md, "gmr": math.exp(mlr), "p_raw_tost": p_raw, "p_log_tost": p_log}

    # Verify zero-variance outside-bounds counterexample returns p=1.0 (not 0.0)
    ce_res = _safe_tost(_np.array([3.125e-5] * 4), _np.array([1.5625e-5] * 4))
    assert ce_res["p_raw_tost"] == 1.0 and ce_res["p_log_tost"] == 1.0, f"Zero-var TOST check failed: {ce_res}"
    # Verify n=1 equal-values edge case returns None / insufficient_n_lt_2 (never p=0.0)
    n1_res = _safe_tost(_np.array([1.0e-4]), _np.array([1.0e-4]))
    assert n1_res["p_raw_tost"] is None and n1_res["p_log_tost"] is None and n1_res["status"] == "insufficient_n_lt_2", f"n=1 TOST check failed: {n1_res}"
    return {
        "note": "Independent zero-variance-safe and n<2-safe TOST verifier; historical pilot/w12_c_offline_pairing_n200.py remains frozen/untouched (all empirical Stage C datasets have n>=6 and strictly positive variance).",
        "zero_var_outside_bounds_counterexample": ce_res,
        "n1_equal_values_edge_case": n1_res,
    }


def _extract_optb_bundle() -> Dict[str, Any]:
    e8_2b = _load_opt(PILOT_DIR / "w14_optb_e2e_routing_2b_4060_summary.json")
    e8_9b = _load_opt(PILOT_DIR / "w14_optb_e2e_routing_9b_l4_summary.json")
    sym_4060 = _load_opt(PILOT_DIR / "w14_optb_symmetric_baselines_local_4060_summary.json")
    sym_small3_a100 = _load_opt(PILOT_DIR / "w14_optb_symmetric_baselines_small3_a100_summary.json")
    sym_9b = _load_opt(PILOT_DIR / "w14_optb_symmetric_baselines_9b_l4_summary.json")
    sym_9b_a100 = _load_opt(PILOT_DIR / "w14_optb_symmetric_baselines_9b_a100_summary.json")
    nv_2b_optb = _load_opt(PILOT_DIR / "w14_optb_coupled_nvme_2b_l4_summary.json")
    nv_9b_optb = _load_opt(PILOT_DIR / "w14_optb_coupled_nvme_9b_l4_summary.json")
    jamba_optb = _load_opt(PILOT_DIR / "w14_optb_jamba3b_summary.json")
    jamba_ctrl = _load_opt(PILOT_DIR / "w14_optb_jamba_28l_fp32_control_summary.json")
    osc_9b = _load_opt(PILOT_DIR / "w13_fallback_9b_osc_optb_summary.json")
    osc_9b_aud = _audit_w13_records(PILOT_DIR / "w13_fallback_9b_osc_optb_records.jsonl", include_passive_osc=False)
    lr16_9b = _load_opt(PILOT_DIR / "w13_fallback_9b_16k_lr_optb_summary.json")
    lr16_9b_aud = _audit_w13_records(PILOT_DIR / "w13_fallback_9b_16k_lr_optb_records.jsonl", include_passive_osc=True)
    osc16_9b = _load_opt(PILOT_DIR / "w13_fallback_9b_16k_osc_optb_summary.json")
    osc16_9b_aud = _audit_w13_records(PILOT_DIR / "w13_fallback_9b_16k_osc_optb_records.jsonl", include_passive_osc=False)
    uniq16_9b = _load_opt(PILOT_DIR / "w14_optb_live_16k_unique_9b_l4_summary.json")
    return {
        "exp1_o1_e8_and_symmetric_e2e_routing": {
            "2b_rtx4060": {
                "part_a_crossovers": e8_2b["part_a_o1_e8"]["monotone_crossover_T_star_suffix_tokens"] if e8_2b else None,
                "part_a_diagnostics": e8_2b["part_a_o1_e8"].get("crossing_diagnostics") if e8_2b else None,
                "part_a_4096": e8_2b["part_a_o1_e8"]["sweep"][-1] if e8_2b else None,
                "part_b_crossovers": e8_2b["part_b_symmetric_e2e_routing"]["monotone_routed_crossovers"] if e8_2b else None,
                "part_b_diagnostics": e8_2b["part_b_symmetric_e2e_routing"].get("crossing_diagnostics") if e8_2b else None,
                "part_b_8192": e8_2b["part_b_symmetric_e2e_routing"]["rows"][-1] if e8_2b else None,
            },
            "9b_l4": {
                "part_a_crossovers": e8_9b["part_a_o1_e8"]["monotone_crossover_T_star_suffix_tokens"] if e8_9b else None,
                "part_a_diagnostics": e8_9b["part_a_o1_e8"].get("crossing_diagnostics") if e8_9b else None,
                "part_a_4096": e8_9b["part_a_o1_e8"]["sweep"][-1] if e8_9b else None,
                "part_b_crossovers": e8_9b["part_b_symmetric_e2e_routing"]["monotone_routed_crossovers"] if e8_9b else None,
                "part_b_diagnostics": e8_9b["part_b_symmetric_e2e_routing"].get("crossing_diagnostics") if e8_9b else None,
                "part_b_4096": e8_9b["part_b_symmetric_e2e_routing"]["rows"][-1] if e8_9b else None,
            },
        },
        "exp2_symmetric_hypic_and_single_edit_t5": {
            "local_4060": sym_4060["models"] if sym_4060 else None,
            "small3_a100": sym_small3_a100["models"] if sym_small3_a100 else None,
            "9b_l4": sym_9b["models"] if sym_9b else None,
            "9b_a100": sym_9b_a100["models"] if sym_9b_a100 else None,
        },
        "exp3_coupled_cold_nvme_t8192_mincore": {
            "2b_l4": nv_2b_optb["results"] if nv_2b_optb else None,
            "9b_l4": nv_9b_optb["results"] if nv_9b_optb else None,
        },
        "exp4_jamba3b_stateful_mamba1": jamba_optb,
        "exp4_jamba3b_synthetic_28l_fp32_control": jamba_ctrl,
        "exp5_9b_live_osc_and_16k_serving": {
            "9b_osc_2k_plus_ood_n225": _extract_fb(osc_9b, record_audit=osc_9b_aud, is_plateau=True),
            "9b_16k_serving_lr_vs_osc_n100": _extract_fb(lr16_9b, record_audit=lr16_9b_aud, is_plateau=False),
            "9b_16k_passive_osc_analysis": _extract_optb_offline_osc_16k(PILOT_DIR / "w13_fallback_9b_16k_lr_optb_records.jsonl"),
            "9b_16k_live_osc_optb_n100": _extract_fb(osc16_9b, record_audit=osc16_9b_aud, is_plateau=True),
            "9b_16k_live_osc_optb_dedup_n93": _extract_dedup_w13_16k_osc(PILOT_DIR / "w13_fallback_9b_16k_osc_optb_records.jsonl"),
            "all_6_legacy_w13_suites_dedup_audit": {
                "2b_osc_2k_plus_ood": _extract_dedup_w13_suite(PILOT_DIR / "w13_fallback_2b_osc_records.jsonl"),
                "2b_16k_osc_serving": _extract_dedup_w13_suite(PILOT_DIR / "w13_fallback_2b_16k_osc_records.jsonl"),
                "9b_lr_2k_plus_ood": _extract_dedup_w13_suite(PILOT_DIR / "w13_fallback_9b_lr_records.jsonl"),
                "9b_osc_optb_2k_plus_ood": _extract_dedup_w13_suite(PILOT_DIR / "w13_fallback_9b_osc_optb_records.jsonl"),
                "9b_16k_lr_optb_serving": _extract_dedup_w13_suite(PILOT_DIR / "w13_fallback_9b_16k_lr_optb_records.jsonl"),
                "9b_16k_osc_optb_serving": _extract_dedup_w13_suite(PILOT_DIR / "w13_fallback_9b_16k_osc_optb_records.jsonl"),
            },
            "9b_16k_live_osc_strictly_unique_n100": uniq16_9b,
        },
        "tost_zero_var_safe_verification": _tost_zero_var_safe_audit(),
    }


def main() -> None:
    fb_2b = _load_opt(PILOT_DIR / "w13_fallback_2b_osc_summary.json")
    fb_2b_tp = _load_opt(PILOT_DIR / "w13_fallback_2b_osc_tau_plateau.json")
    fb_2b_16k = _load_opt(PILOT_DIR / "w13_fallback_2b_16k_osc_summary.json")
    fb_9b = _load_opt(PILOT_DIR / "w13_fallback_9b_lr_summary.json")
    fb_9b_tp = _load_opt(PILOT_DIR / "w13_fallback_9b_lr_tau_plateau.json")
    nv_2b = _load_opt(PILOT_DIR / "w12_c_nvme_int8_2b_summary.json")
    nv_9b = _load_opt(PILOT_DIR / "w12_c_nvme_int8_9b_summary.json")
    w12_5 = _load_opt(PILOT_DIR / "w12_c_offline_pairing_n200_results.json")
    j_gate = _load_opt(PILOT_DIR / "w12_c_jamba3b_gate1_summary.json")
    j_full = _load_opt(PILOT_DIR / "w12_c_jamba3b_summary.json")

    audit_2b = _audit_w13_records(PILOT_DIR / "w13_fallback_2b_osc_records.jsonl", include_passive_osc=False)
    audit_2b_16k = _audit_w13_records(PILOT_DIR / "w13_fallback_2b_16k_osc_records.jsonl", include_passive_osc=False)
    audit_9b = _audit_w13_records(PILOT_DIR / "w13_fallback_9b_lr_records.jsonl", include_passive_osc=True)

    summary = {
        "provenance_registry": {
            "note": "Retrospective experiment registry & calibration log (not external pre-registration). Detector thresholds frozen on dev splits before eval.",
            "registry_doc": "refine-logs/STAGE_C_W13_EXPERIMENT_REGISTRY.md",
            "id_mapping": {
                "W12-2_and_G4_fallback": "pilot/w13_fallback_runner.py (separate from unmodified w5_9b_e5_runner.py)",
                "W12-4_nvme_int8_microbench": "pilot/w12_c_nvme_int8_stress.py",
                "W12-5_2B_16K_32K_n200_ext": "pilot/w5_9b_e5_runner.py + pilot/w12_c_offline_pairing_n200.py",
                "W11-2_third_family": "pilot/w12_c_third_family.py",
            },
            "calibration_and_reporting_notes": [
                "Crossover & timing scope (B1 fix): E5 find_crossover returns total context T_ctx=242 at grid index 0 (actual suffix_len=49 tok on 9B at 1.446x and 67 tok on 2B at 1.538x); E8 standalone edit microbench on L4 yields 264.7/302.7/1219.9 tok (2B) and 239.5/260.0/624.3 tok (9B). Neither measures a live-routed crossover boundary. All W12-2 latencies are timed GPU-path means (pre-timer H2D, cache clones, and baseline H2D excluded; apply_delta_rope_to_kv_cache uses full-KV torch.cat + suffix RoPE rotation on delta_L != 0). Short held-out suffixes (2B 94-152, 9B 76-134 tok) show 0.708x/0.753x slowdowns; benign-heavy long-suffix runs reach 1.225x (2K) and 3.553x (2B 16K separate run on reused 100 case keys; p95 4314.3 vs 4204.2 ms).",
                "Quantile & AP conventions: Legacy W13, updated Exp-3, and updated Exp-5 (`w14_optb_live_16k_unique_osc.py`) report medians and p95 latencies using upper-order-statistic indexing (`sorted[n//2]` and `sorted[min(n-1, int(0.95*n))]`). Exp-1 (`w14_optb_e2e_routing_bench.py`) uses standard interpolated median (`0.5*(s[n//2-1]+s[n//2])` for even n) alongside upper-order p95 (`n=40` routed vs `n=15` baseline). `pr_auc` in legacy W13 is rank-based average precision (`_pr_auc_np`, suppressed on single-class strata); tie-grouped AP (`pr_auc_tie_aware`) is additionally computed on 16K runs. Benign FP 0/156 on 2B carries a 2.40% Wilson 95% upper bound; legacy sidecar `w13_fallback_9b_lr_tau_plateau.json:3` retains a frozen cosmetic `model` field string (`Qwen/Qwen3.5-2B`) although its records belong to 9B LR.",
                "LR dev midpoint & cross-model comparison: 9B 11-feature LR dev split extrema are neg_max=0.02435 and pos_min=0.94109, implying exact midpoint 0.48272 while keeping 0.50 as fixed operating point. On the 9B evaluation rows, offline passive OSC@0.40 yields TP=54/55, FP=1/170, contains_new=214/225 (vs LR@0.50 FP=4/170, contains_new=211/225), so 2B vs 9B comparisons do not isolate detector effect from model effect.",
                "Threshold curve & tau sweep notes: 2B OSC forms a flat plateau across tau in {0.2, 0.4, 0.7} (0/156 FP, 0.8667 contains_new; union=63); 9B LR exhibits a sensitivity tradeoff curve across tau in {0.3, 0.5, 0.7} (FP=7/4/1, contains_new=0.9244/0.9378/0.9333; union=61 adding rows [104, 110, 114] vs 58 frozen). Offline tau sweeps substitute standalone always_suffix fields on flagged rows (verified across all 141 frozen flagged rows; union across all swept thresholds spans 144 rows).",
                "unseen_coupled_e12_single dual scoring & healable_only: In e12, the prompt queries the signed manifest (`remains {v_o}`), whereas the draft_value scorer rewards `new_v` (2B 1->1->0/25, 9B LR 14->11->0/25 with 3 flips McNemar p=0.25, 9B OSC 14->14->0/25; always_suffix mean KL=5.94e-5 [2B] / 5.84e-4 [9B] tracks edited-prompt oracle). Relabeling e12 to authoritative manifest fidelity (`old_v`) shifts pooled counts to 2B OSC 179->218->224/225 (benign FP 24/179), 9B LR 166->213->225/225 (benign FP 11/166; e12 paired flip 3:0 draft vs 0:3 manifest, both exact McNemar p=0.25), and 9B OSC 166->210->225/225 (FP 11/166, TP 44/59 vs draft FP 1/170, TP 54/55). Thus pooled 9B OSC advantages (`0.951 vs 0.889`) must be read alongside the draft-value label definition. healable_only (n=200/100) excludes e12_single by construction (`always_suffix=0`), yielding 9B benign FP 1/156 (0.64%) and 1.000 = always_suffix.",
            ],
        },
        "segment_1_w12_2_and_g4_ood": {
            "2b_osc_frozen_2k_plus_ood": _extract_fb(fb_2b, fb_2b_tp, record_audit=audit_2b, is_plateau=True),
            "2b_osc_frozen_16k_serving_reused_keys": _extract_fb(fb_2b_16k, record_audit=audit_2b_16k, is_plateau=True),
            "9b_lr_frozen_2k_plus_ood": _extract_fb(fb_9b, fb_9b_tp, record_audit=audit_9b, is_plateau=False),
        },
        "segment_2_w12_4_nvme_int8_microbench": {
            "hardware_dependent_non_cross_machine_reproducible": True,
            "two_distinct_experiments_note": (
                "(1) run_nvme_swap_microbench times random model-shaped buffers on disk (on L4 qwen35_9b int8, "
                "file-read cold/warm means are 4.39/2.47 ms while total_swap_in 8.99/7.08 ms sums file-read means + warm-buffer H2D [1.49 ms] & dequant [3.11 ms] means, not one contiguous cold swap-in; "
                "posix_fadvise logs at least one syscall return without page-residency verification). "
                "(2) _eval_ckpt_cell restores quantized states from pinned host memory (no NVMe disk reads); 9B collision_stress shows 19/20 per cell (95/100 duplicate rows) with max_abs_logit_diff > 0.5."
            ),
            "accuracy_metric_note": "Accuracy fields represent 8-token substring containment (`contains_new`), NOT strict exact-string match; strict <=1e-4 threshold evaluates `J-local_fidelity` (distinct from E5 G1b on J-full).",
            "nvme_cold_warm_note": "RTX 4060 Windows lacks POSIX_FADV_DONTNEED (fadvise_dontneed_supported=False, 2/6 modes have warm >= cold); only L4 invokes posix_fadvise(POSIX_FADV_DONTNEED).",
            "audited_restore_only": {
                "2b": _audit_w12_4_records(PILOT_DIR / "w12_c_nvme_int8_2b_records.jsonl"),
                "9b": _audit_w12_4_records(PILOT_DIR / "w12_c_nvme_int8_9b_records.jsonl"),
            },
            "2b_raw_summary": nv_2b,
            "9b_raw_summary": nv_9b,
        },
        "segment_3_w12_5_exploratory_extension_audit": {
            "independent_primary_flag_note": "Legacy field independent_primary=true inside w12_c_offline_pairing_n200_results.json identifies primary cohort membership (not statistical task independence: 2B and 9B share 200 E5 task identities across context lengths; k=6 df=5 relies on between-setup correlation). Both primary k=6 df=5 (GMR=0.95762, p=0.06854) and deduplicated tail n=150 (GMR=0.82740, log_tost_p=0.34095) fail [0.80, 1.25] relative TOST; only tail +-1e-5 raw-scale TOST passes (p=0.00010078). Accuracy fields evaluate 8-token contains_new on E5 vs exact_match on 27B.",
        },
        "segment_3_w12_5_exploratory_extension": w12_5,
        "segment_4_w11_2_third_family": {
            "scope_and_gate_constraints": {
                "tested_claim_scope": "C2 external validity ONLY; Jamba is Mamba-1 + NoPE + dense MLP (`num_experts=1`, sparse MoE routing untested; neither GDN nor Mamba-2), so C3 transport-decay is inapplicable and Stage 4 must never be reported as full 3-family C1-C4 validation.",
                "moe_routing_note": "Verified real AI21-Jamba-Reasoning-3B config has num_experts=1, has_moe_router=False (dense variant); multi-expert MoE routing interaction remains untested.",
                "cpu_dry_run_caveat": "CPU --synthetic-dry-run tested only synthetic JambaConfig (not real cached weights); scale-up requires passing all 3 real-weight gates on 1 sample.",
            },
            "excluded_candidates": {
                "Nemotron-H": "Mamba-2 + attention (3rd model of 2nd family, not a new hybrid family).",
                "LFM2.5": "Short-conv + attention (no recurrent state; O_local state surgery inapplicable).",
            },
            "alternative_unimplemented": {
                "inclusionAI/Ling-3.0-tiny": "Backup candidate, NOT implemented in w12_c_third_family.py (static 0-CU config/tokenizer check: BailingMoeV3ForCausalLM, bailing_hybrid, 24 layers, 128 experts, KDA + MLA; left unimplemented).",
            },
            "implemented_candidate_jamba3b_outcome": {
                "repo_id": "ai21labs/AI21-Jamba-Reasoning-3B",
                "verdict": "INCONCLUSIVE",
                "failure_category": "INCONCLUSIVE",
                "raw_runner_outcome_before_audit": "GENUINE_STATE_OR_KERNEL_INFEASIBLE",
                "triage_notes": [
                    "Real HF weights verified on Colab L4 (`28` layers = 26 Mamba-1 + 2 NoPE Attention, `num_experts=1` dense non-MoE variant).",
                    "Gate 1 (26/26 conv/rec + 2/2 KV state access) and Gate 2 (`G1a max_diff=0.0`, comparing two chunked paths that can both lack recurrent carry) passed; `state_only` and `old` share identical scalar fields (note: JSONL omits logits/tensor hashes to prove bitwise equality).",
                    "Gate 3 oracle itself failed (`acc=0.0` under both earlier unarchived raw-template attempt [`g1b_mean_kl=5.17e-4, max_diff=2.3125`] and archived closed-template `--close-think` attempt [`G1b KL=0.059154, max_diff=4.8164, manual_eq_generate=0/1`], generating 'The user provided a long text with a') -- leaving template, chunking, and initial-state explanations open.",
                    "Local Transformers 5.22.2 inspection of `mamba_selective_scan` & `JambaMambaMixer.forward` shows `mamba_selective_scan` has no initial-state parameter/entry point (zero-initializes `ssm_state`) and `JambaMambaMixer` reads `cache_params.recurrent_states` only at `seq_len == 1` (does not pin historical Colab backend).",
                    "Scale-up halted without rewriting O_local or adding external `mamba-ssm` kernels. Machine-readable verdict and failure_category set to INCONCLUSIVE; core evidence covers tested GDN + Mamba-2 models.",
                ],
            },
            "jamba3b_1sample_gate": j_gate,
            "jamba3b_full": j_full,
        },
        "segment_5_w11_crossfamily_mamba2_scope": {
            "source_summary": "pilot/w11_crossfamily_scaleup_summary.json",
            "summary_note": (
                "Falcon-H1 E2 n=100: 93/100 vs kv_only 91/100 (p=0.50, KL 92.49x), E3 n=50 (97.0% run-weighted collateral across 50 permutation runs from 15 sets); "
                "Zamba2 E2 n=100: 86/100 (84/100 strict) vs kv_only 59/100 (p=1.49e-8), E3 n=50 (48.0% run-weighted collateral across 50 permutation runs from 15 sets); "
                "SB1 containment: Falcon 59/60, Zamba2 J-local 53/60 (88.3%, KL=8.94e-4) vs KVEraser LoRA 54/60 (90.0%, KL=0.468). "
                "HYPIC-inspired segmented condition builds prefix checkpoint + runs old suffix inside the timer; wall-clock comparisons apply strictly to that adaptation."
            ),
        },
        "option_b_posthoc_closure": _extract_optb_bundle(),
    }

    out_path = PILOT_DIR / "w12_c_posthoc_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
