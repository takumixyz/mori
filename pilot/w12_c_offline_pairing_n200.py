"""Stage-C W12-5 Offline Exploratory Extension Pairing (`n=200` at 2B 16K/32K).

Strictly obeys exploratory-extension reporting rules:
  - Preserves Primary (`n=50` at 2B 16K & 32K, k=6 N=670) completely intact and separate.
  - Reports Extension (`n=200` at 2B 16K & 32K, k=6 N=970) plus the disjoint tail slice
    (`sample_index >= 50`, n=150) in SEPARATE columns/keys, joined strictly by explicit task ID
    (`(sample_index, depth, tpl_id, old, new)`), never by positional row index.
  - Harmonizes KL endpoints across E5 and 27B (`first_answer_logit_kl` = E5 `kl_vs_oneshot` & 27B
    `kl_first_vs_oracle` as primary harmonized estimand; `answer_mean_kl` = E5 `answer_kl_vs_oneshot`
    & 27B `drift_mean_ans_kl` alongside; legacy mixed endpoint retained solely for audit lineage).
  - Applies UNCHANGED G1a (<=1e-5), G1b strict (<=1e-4), and G1b relaxed
    (`strict or (max_logit_diff <= 0.28125 and kl <= 1.5e-3)`, exact match to `w5_9b_e5_runner.py:1679`).
  - Specifies ONE primary estimand (`equal_weight_6_setup_t_df5` on `first_answer_logit_kl`,
    [0.80, 1.25] log-ratio TOST) and frames C3 strictly as an exploratory, heterogeneous null ablation
    (removes "independent settings", "equivalence established", and "zero gain" headline claims).
  - Notes `EM` fields reflect 8-token substring containment (`contains_new`), not strict exact match.
  - Runs 0-CU offline using numpy + scipy (`D:\\environment\\python\\pyenvs\\minimind\\python.exe`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

PILOT_DIR = Path(__file__).resolve().parent


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> List[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _task_key(r: Dict[str, Any]) -> Tuple[int, float, int, str, str]:
    return (
        int(r["sample_index"]),
        round(float(r["depth"]), 6),
        int(r["tpl_id"]),
        str(r["old"]),
        str(r["new"]),
    )


def extract_pair_arrays(
    rows: List[Dict[str, Any]],
    fmt: str = "e5",
    kl_endpoint: str = "first_token",
) -> Dict[str, np.ndarray]:
    """Return paired (jlocal, stale, state_only, recompute) arrays plus G1a/G1b booleans.

    kl_endpoint:
      - "first_token": E5 `kl_vs_oneshot` and 27B `kl_first_vs_oracle` (harmonized 1st-answer logit KL)
      - "answer_mean": E5 `answer_kl_vs_oneshot` and 27B `drift_mean_ans_kl` (harmonized answer-mean KL)
    """
    if fmt == "e5":
        e5_key = "answer_kl_vs_oneshot" if kl_endpoint == "answer_mean" else "kl_vs_oneshot"
        jl = np.array([float(r["conditions"]["jlocal"][e5_key]) for r in rows], dtype=np.float64)
        st = np.array([float(r["conditions"]["stale"][e5_key]) for r in rows], dtype=np.float64)
        so = np.array([float(r["conditions"]["state_only"][e5_key]) for r in rows], dtype=np.float64)
        rc = np.array([float(r["conditions"]["recompute"][e5_key]) for r in rows], dtype=np.float64)
        g1a = np.array([bool(r["g1a_pass"]) for r in rows], dtype=bool)
        g1b_s = np.array([bool(r["g1b_pass_strict"]) for r in rows], dtype=bool)
        g1b_r = np.array([bool(r["g1b_pass_relaxed"]) for r in rows], dtype=bool)
        jl_em = np.array([bool(r["conditions"]["jlocal"]["contains_new"]) for r in rows], dtype=bool)
        st_em = np.array([bool(r["conditions"]["stale"]["contains_new"]) for r in rows], dtype=bool)
        rc_em = np.array([bool(r["conditions"]["recompute"]["contains_new"]) for r in rows], dtype=bool)
        so_em = np.array([bool(r["conditions"]["state_only"]["contains_new"]) for r in rows], dtype=bool)
        jl_leak = np.array([bool(r["conditions"]["jlocal"]["contains_old"]) for r in rows], dtype=bool)
        so_leak = np.array([bool(r["conditions"]["state_only"]["contains_old"]) for r in rows], dtype=bool)
    elif fmt == "27b":
        rows = [r for r in rows if "conditions" in r and "J_local" in r.get("conditions", {})]
        k27_key = "kl_first_vs_oracle" if kl_endpoint == "first_token" else "drift_mean_ans_kl"
        jl = np.array([float(r["conditions"]["J_local"][k27_key]) for r in rows], dtype=np.float64)
        st = np.array([float(r["conditions"]["J_local_stale"][k27_key]) for r in rows], dtype=np.float64)
        so = np.array([float(r["conditions"]["state_only"][k27_key]) for r in rows], dtype=np.float64)
        rc = np.array([float(r["conditions"]["J_local_recompute"][k27_key]) for r in rows], dtype=np.float64)
        g1a = np.array([bool(r["g1a_pass"]) for r in rows], dtype=bool)
        g1b_s = np.array([bool(r["g1b_strict"]) for r in rows], dtype=bool)
        g1b_r = np.array([bool(r["g1b_pass"]) for r in rows], dtype=bool)
        jl_em = np.array([bool(r["conditions"]["J_local"]["exact_match"]) for r in rows], dtype=bool)
        st_em = np.array([bool(r["conditions"]["J_local_stale"]["exact_match"]) for r in rows], dtype=bool)
        rc_em = np.array([bool(r["conditions"]["J_local_recompute"]["exact_match"]) for r in rows], dtype=bool)
        so_em = np.array([bool(r["conditions"]["state_only"]["exact_match"]) for r in rows], dtype=bool)
        jl_leak = np.array([bool(r["conditions"]["J_local"]["old_leak"]) for r in rows], dtype=bool)
        so_leak = np.array([bool(r["conditions"]["state_only"]["old_leak"]) for r in rows], dtype=bool)
    else:
        raise ValueError(fmt)
    return {
        "jl_kl": jl, "st_kl": st, "so_kl": so, "rc_kl": rc,
        "g1a": g1a, "g1b_strict": g1b_s, "g1b_relaxed": g1b_r,
        "jl_em": jl_em, "st_em": st_em, "rc_em": rc_em, "so_em": so_em,
        "jl_leak": jl_leak, "so_leak": so_leak,
    }


def compute_tost_stats(a: np.ndarray, b: np.ndarray, independent_primary: bool = True) -> Dict[str, Any]:
    """Exact paired raw-KL & log-ratio [0.80, 1.25] TOST matching Wave 9 w9_offline_pairing.py."""
    n = int(len(a))
    d = a - b
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1))
    se_d = sd_d / math.sqrt(n)
    df = n - 1

    t95 = float(stats.t.ppf(0.95, df))  # two-sided 90% CI = one-sided alpha=0.05 TOST
    t_bonf = float(stats.t.ppf(1.0 - 0.00625, df))  # alpha_bonf = 0.05 / 8 = 0.00625
    tight_a05 = abs(mean_d) + t95 * se_d
    tight_bonf = abs(mean_d) + t_bonf * se_d

    # Raw TOST with delta = 1e-5
    t_low_raw = (mean_d + 1e-5) / se_d if se_d > 0 else float("inf")
    t_high_raw = (1e-5 - mean_d) / se_d if se_d > 0 else float("inf")
    p_raw_tost = float(max(stats.t.sf(t_low_raw, df), stats.t.sf(t_high_raw, df)))

    z_pwr = float(stats.norm.ppf(0.80))
    t_two05 = float(stats.t.ppf(0.975, df))
    t_twobonf = float(stats.t.ppf(1.0 - 0.00625 / 2.0, df))
    mde_05 = (t_two05 + z_pwr) * se_d
    mde_bonf = (t_twobonf + z_pwr) * se_d

    # Log-ratio TOST on [log(0.80), log(1.25)]
    eps = 1e-12
    lr = np.log(np.maximum(a, eps)) - np.log(np.maximum(b, eps))
    mean_lr = float(np.mean(lr))
    se_lr = float(np.std(lr, ddof=1)) / math.sqrt(n)
    gmr = math.exp(mean_lr)
    ci90_lo = math.exp(mean_lr - t95 * se_lr)
    ci90_hi = math.exp(mean_lr + t95 * se_lr)
    lo_b, hi_b = math.log(0.80), math.log(1.25)
    t_lo = (mean_lr - lo_b) / se_lr if se_lr > 0 else float("inf")
    t_hi = (hi_b - mean_lr) / se_lr if se_lr > 0 else float("inf")
    log_tost_p = float(max(stats.t.sf(t_lo, df), stats.t.sf(t_hi, df)))

    # Wilcoxon on raw difference
    nz = d[np.abs(d) > 1e-18]
    wilc_p = float(stats.wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0

    return {
        "n": n,
        "mean_diff_jlocal_minus_stale": mean_d,
        "raw_tightest_margin_a05": tight_a05,
        "raw_tightest_margin_bonf00625": tight_bonf,
        "raw_tost_p_1e5": p_raw_tost,
        "mde_80_power_a05": mde_05,
        "mde_80_power_bonf00625": mde_bonf,
        "gmr_jlocal_over_stale": gmr,
        "gmr_ci90": [ci90_lo, ci90_hi],
        "log_tost_p_080_125": log_tost_p,
        "mean_log_ratio": mean_lr,
        "wilcoxon_p_raw": wilc_p,
        "independent_primary": independent_primary,
    }


def summarize_suite_metrics(arr: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n = int(len(arr["jl_kl"]))
    g1a_c = int(arr["g1a"].sum())
    g1b_s_c = int(arr["g1b_strict"].sum())
    g1b_r_c = int(arr["g1b_relaxed"].sum())
    jl_c = int(arr["jl_em"].sum())
    st_c = int(arr["st_em"].sum())
    rc_c = int(arr["rc_em"].sum())
    so_c = int(arr["so_em"].sum())
    return {
        "n": n,
        "g1a_pass": {"count": g1a_c, "rate": g1a_c / n, "ci95": wilson_ci(g1a_c, n)},
        "g1b_strict_pass": {"count": g1b_s_c, "rate": g1b_s_c / n, "ci95": wilson_ci(g1b_s_c, n)},
        "g1b_relaxed_pass": {"count": g1b_r_c, "rate": g1b_r_c / n, "ci95": wilson_ci(g1b_r_c, n)},
        "jlocal_em": {"count": jl_c, "rate": jl_c / n, "ci95": wilson_ci(jl_c, n), "mean_kl": float(np.mean(arr["jl_kl"])), "median_kl": float(np.median(arr["jl_kl"])), "leak_rate": float(np.mean(arr["jl_leak"]))},
        "stale_em": {"count": st_c, "rate": st_c / n, "ci95": wilson_ci(st_c, n), "mean_kl": float(np.mean(arr["st_kl"])), "median_kl": float(np.median(arr["st_kl"]))},
        "recompute_em": {"count": rc_c, "rate": rc_c / n, "ci95": wilson_ci(rc_c, n), "mean_kl": float(np.mean(arr["rc_kl"]))},
        "state_only_em": {"count": so_c, "rate": so_c / n, "ci95": wilson_ci(so_c, n), "mean_kl": float(np.mean(arr["so_kl"])), "leak_rate": float(np.mean(arr["so_leak"]))},
    }


def equal_weight_setup_tost(per_setup_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    lrs = np.array([s["mean_log_ratio"] for s in per_setup_stats], dtype=np.float64)
    k = len(lrs)
    df = k - 1
    mean_lr = float(np.mean(lrs))
    se_lr = float(np.std(lrs, ddof=1)) / math.sqrt(k)
    t95 = float(stats.t.ppf(0.95, df))
    lo_b, hi_b = math.log(0.80), math.log(1.25)
    t_lo = (mean_lr - lo_b) / se_lr
    t_hi = (hi_b - mean_lr) / se_lr
    return {
        "k_setups": k,
        "gmr_equal_weight": math.exp(mean_lr),
        "gmr_ci90_t_df5": [math.exp(mean_lr - t95 * se_lr), math.exp(mean_lr + t95 * se_lr)],
        "log_tost_p_080_125": float(max(stats.t.sf(t_lo, df), stats.t.sf(t_hi, df))),
    }


def _join_by_task_id(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Explicitly join two E5 record lists by `(sample_index, depth, tpl_id, old, new)`."""
    map_a = {_task_key(r): r for r in rows_a}
    map_b = {_task_key(r): r for r in rows_b}
    assert len(map_a) == len(rows_a), "Duplicate task keys in first cohort"
    assert len(map_b) == len(rows_b), "Duplicate task keys in second cohort"
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    assert len(common) == len(map_a) == len(map_b), (
        f"Task-ID mismatch: len_a={len(map_a)}, len_b={len(map_b)}, common={len(common)}"
    )
    return [map_a[k] for k in common], [map_b[k] for k in common]


def _run_c3_estimand(
    kl_endpoint: str,
    estimand_label: str,
    rows_9b_4k: List[Dict[str, Any]],
    rows_9b_16k: List[Dict[str, Any]],
    rows_9b_32k: List[Dict[str, Any]],
    rows_27b: List[Dict[str, Any]],
    rows_2b_16k_p: List[Dict[str, Any]],
    rows_2b_32k_p: List[Dict[str, Any]],
    rows_2b_16k_ext: List[Dict[str, Any]],
    rows_2b_32k_ext: List[Dict[str, Any]],
    tail_16k_joined: List[Dict[str, Any]],
    tail_32k_joined: List[Dict[str, Any]],
) -> Dict[str, Any]:
    arr_9b_4k = extract_pair_arrays(rows_9b_4k, "e5", kl_endpoint)
    arr_9b_16k = extract_pair_arrays(rows_9b_16k, "e5", kl_endpoint)
    arr_9b_32k = extract_pair_arrays(rows_9b_32k, "e5", kl_endpoint)
    arr_27b = extract_pair_arrays(rows_27b, "27b", kl_endpoint)

    arr_2b_16k_p = extract_pair_arrays(rows_2b_16k_p, "e5", kl_endpoint)
    arr_2b_32k_p = extract_pair_arrays(rows_2b_32k_p, "e5", kl_endpoint)
    arr_2b_16k_ext = extract_pair_arrays(rows_2b_16k_ext, "e5", kl_endpoint)
    arr_2b_32k_ext = extract_pair_arrays(rows_2b_32k_ext, "e5", kl_endpoint)
    arr_2b_16k_tail = extract_pair_arrays(tail_16k_joined, "e5", kl_endpoint)
    arr_2b_32k_tail = extract_pair_arrays(tail_32k_joined, "e5", kl_endpoint)

    primary_per_setup = {
        "9B_4K_n200": compute_tost_stats(arr_9b_4k["jl_kl"], arr_9b_4k["st_kl"]),
        "9B_16K_n200": compute_tost_stats(arr_9b_16k["jl_kl"], arr_9b_16k["st_kl"]),
        "9B_32K_n50": compute_tost_stats(arr_9b_32k["jl_kl"], arr_9b_32k["st_kl"]),
        "2B_16K_n50_PRIMARY": compute_tost_stats(arr_2b_16k_p["jl_kl"], arr_2b_16k_p["st_kl"]),
        "2B_32K_n50_PRIMARY": compute_tost_stats(arr_2b_32k_p["jl_kl"], arr_2b_32k_p["st_kl"]),
        "27B_E2_n120": compute_tost_stats(arr_27b["jl_kl"], arr_27b["st_kl"]),
    }

    ext_per_setup = {
        "9B_4K_n200": primary_per_setup["9B_4K_n200"],
        "9B_16K_n200": primary_per_setup["9B_16K_n200"],
        "9B_32K_n50": primary_per_setup["9B_32K_n50"],
        "2B_16K_n200_EXPLORATORY_EXT": compute_tost_stats(arr_2b_16k_ext["jl_kl"], arr_2b_16k_ext["st_kl"], independent_primary=False),
        "2B_32K_n200_EXPLORATORY_EXT": compute_tost_stats(arr_2b_32k_ext["jl_kl"], arr_2b_32k_ext["st_kl"], independent_primary=False),
        "27B_E2_n120": primary_per_setup["27B_E2_n120"],
    }

    pool_prim_jl = np.concatenate([arr_9b_4k["jl_kl"], arr_9b_16k["jl_kl"], arr_9b_32k["jl_kl"], arr_2b_16k_p["jl_kl"], arr_2b_32k_p["jl_kl"], arr_27b["jl_kl"]])
    pool_prim_st = np.concatenate([arr_9b_4k["st_kl"], arr_9b_16k["st_kl"], arr_9b_32k["st_kl"], arr_2b_16k_p["st_kl"], arr_2b_32k_p["st_kl"], arr_27b["st_kl"]])
    pool_ext_jl = np.concatenate([arr_9b_4k["jl_kl"], arr_9b_16k["jl_kl"], arr_9b_32k["jl_kl"], arr_2b_16k_ext["jl_kl"], arr_2b_32k_ext["jl_kl"], arr_27b["jl_kl"]])
    pool_ext_st = np.concatenate([arr_9b_4k["st_kl"], arr_9b_16k["st_kl"], arr_9b_32k["st_kl"], arr_2b_16k_ext["st_kl"], arr_2b_32k_ext["st_kl"], arr_27b["st_kl"]])

    pool_tail_jl = np.concatenate([arr_2b_16k_tail["jl_kl"], arr_2b_32k_tail["jl_kl"]])
    pool_tail_st = np.concatenate([arr_2b_16k_tail["st_kl"], arr_2b_32k_tail["st_kl"]])
    tail_task_jl = 0.5 * (arr_2b_16k_tail["jl_kl"] + arr_2b_32k_tail["jl_kl"])
    tail_task_st = 0.5 * (arr_2b_16k_tail["st_kl"] + arr_2b_32k_tail["st_kl"])

    eps_t = 1e-12
    d_task_tail = 0.5 * ((arr_2b_16k_tail["jl_kl"] - arr_2b_16k_tail["st_kl"]) + (arr_2b_32k_tail["jl_kl"] - arr_2b_32k_tail["st_kl"]))
    lr_task_tail = 0.5 * (
        (np.log(np.maximum(arr_2b_16k_tail["jl_kl"], eps_t)) - np.log(np.maximum(arr_2b_16k_tail["st_kl"], eps_t)))
        + (np.log(np.maximum(arr_2b_32k_tail["jl_kl"], eps_t)) - np.log(np.maximum(arr_2b_32k_tail["st_kl"], eps_t)))
    )
    n_tasks = int(len(d_task_tail))
    df_tasks = n_tasks - 1
    m_raw_t = float(np.mean(d_task_tail))
    se_raw_t = float(np.std(d_task_tail, ddof=1)) / math.sqrt(n_tasks)
    raw_tost_p_task = float(max(stats.t.sf((m_raw_t + 1e-5) / se_raw_t, df_tasks), stats.t.sf((1e-5 - m_raw_t) / se_raw_t, df_tasks)))
    t95_t = float(stats.t.ppf(0.95, df_tasks))
    m_lr_t = float(np.mean(lr_task_tail))
    se_lr_t = float(np.std(lr_task_tail, ddof=1)) / math.sqrt(n_tasks)
    log_tost_p_task = float(max(stats.t.sf((m_lr_t - math.log(0.80)) / se_lr_t, df_tasks), stats.t.sf((math.log(1.25) - m_lr_t) / se_lr_t, df_tasks)))

    return {
        "kl_endpoint": estimand_label,
        "primary_n50_cohort": {
            "per_setup": primary_per_setup,
            "equal_weight_6_setup_t_df5_PRIMARY_ESTIMAND": equal_weight_setup_tost(list(primary_per_setup.values())),
            "pooled_6_sample_weighted_N670_exploratory": compute_tost_stats(pool_prim_jl, pool_prim_st, False),
        },
        "extension_n200_nested_cohort": {
            "per_setup": ext_per_setup,
            "equal_weight_6_setup_t_df5": equal_weight_setup_tost(list(ext_per_setup.values())),
            "pooled_6_sample_weighted_N970_nested": compute_tost_stats(pool_ext_jl, pool_ext_st, False),
            "disjoint_tail_task_id_joined_n150": {
                "join_key": "(sample_index, depth, tpl_id, old, new)",
                "2B_16K_s50_199_n150": compute_tost_stats(arr_2b_16k_tail["jl_kl"], arr_2b_16k_tail["st_kl"], False),
                "2B_32K_s50_199_n150": compute_tost_stats(arr_2b_32k_tail["jl_kl"], arr_2b_32k_tail["st_kl"], False),
                "combined_record_level_N300_non_independent": compute_tost_stats(pool_tail_jl, pool_tail_st, False),
                "task_deduplicated_n150": {
                    "n_unique_tasks": n_tasks,
                    "task_identity_note": "Joined by explicit task key; 16K and 32K tail records share all 150 task identities and all 200 2B tasks overlap with 9B E5 task identities.",
                    "mean_diff_jlocal_minus_stale": m_raw_t,
                    "raw_tost_p_1e5": raw_tost_p_task,
                    "gmr_jlocal_over_stale": math.exp(m_lr_t),
                    "gmr_ci90": [math.exp(m_lr_t - t95_t * se_lr_t), math.exp(m_lr_t + t95_t * se_lr_t)],
                    "log_tost_p_080_125": log_tost_p_task,
                    "mean_kl_avg_pair_check": [float(np.mean(tail_task_jl)), float(np.mean(tail_task_st))],
                },
            },
        },
    }


def main() -> None:
    rows_9b_4k = load_jsonl(PILOT_DIR / "w5_9b_e5_4k_o1_records.jsonl")
    rows_9b_16k = load_jsonl(PILOT_DIR / "w5_9b_e5_16k_o1_records.jsonl")
    rows_9b_32k = load_jsonl(PILOT_DIR / "w6_9b_32k_n50_records.jsonl")
    rows_27b = load_jsonl(PILOT_DIR / "w7_e9_27b_results.jsonl")

    # Primary 2B cohorts (n=50, frozen Wave 8 files - NEVER overwritten)
    rows_2b_16k_p = load_jsonl(PILOT_DIR / "w8_2b_16k_n50_records.jsonl")
    rows_2b_32k_p = load_jsonl(PILOT_DIR / "w8_2b_32k_n50_records.jsonl")
    # Exploratory Extension 2B cohorts (n=200, W12-5 separate files)
    rows_2b_16k_ext = load_jsonl(PILOT_DIR / "w12_2b_16k_n200_records.jsonl")
    rows_2b_32k_ext = load_jsonl(PILOT_DIR / "w12_2b_32k_n200_records.jsonl")

    arr_2b_16k_p = extract_pair_arrays(rows_2b_16k_p, "e5", "first_token")
    arr_2b_32k_p = extract_pair_arrays(rows_2b_32k_p, "e5", "first_token")
    arr_2b_16k_ext = extract_pair_arrays(rows_2b_16k_ext, "e5", "first_token")
    arr_2b_32k_ext = extract_pair_arrays(rows_2b_32k_ext, "e5", "first_token")

    # Disjoint tail (sample_index >= 50, n=150) explicitly joined by task key
    tail_16k_raw = [r for r in rows_2b_16k_ext if int(r["sample_index"]) >= 50]
    tail_32k_raw = [r for r in rows_2b_32k_ext if int(r["sample_index"]) >= 50]
    tail_16k_joined, tail_32k_joined = _join_by_task_id(tail_16k_raw, tail_32k_raw)
    arr_2b_16k_tail = extract_pair_arrays(tail_16k_joined, "e5", "first_token")
    arr_2b_32k_tail = extract_pair_arrays(tail_32k_joined, "e5", "first_token")

    keys_2b_16 = {_task_key(r) for r in rows_2b_16k_ext}
    keys_2b_32 = {_task_key(r) for r in rows_2b_32k_ext}
    keys_9b_4k = {_task_key(r) for r in rows_9b_4k}
    keys_9b_16k = {_task_key(r) for r in rows_9b_16k}
    keys_9b_32k = {_task_key(r) for r in rows_9b_32k}

    harm_first = _run_c3_estimand(
        "first_token",
        "E5 kl_vs_oneshot (1-tok) harmonized with 27B kl_first_vs_oracle (1st answer logit KL)",
        rows_9b_4k, rows_9b_16k, rows_9b_32k, rows_27b,
        rows_2b_16k_p, rows_2b_32k_p, rows_2b_16k_ext, rows_2b_32k_ext,
        tail_16k_joined, tail_32k_joined,
    )
    harm_ansmean = _run_c3_estimand(
        "answer_mean",
        "E5 answer_kl_vs_oneshot (bit-identical to kl_vs_oneshot on 1-tok E5) harmonized with 27B drift_mean_ans_kl (4-tok answer mean)",
        rows_9b_4k, rows_9b_16k, rows_9b_32k, rows_27b,
        rows_2b_16k_p, rows_2b_32k_p, rows_2b_16k_ext, rows_2b_32k_ext,
        tail_16k_joined, tail_32k_joined,
    )

    out = {
        "meta": {
            "generator": "pilot/w12_c_offline_pairing_n200.py",
            "zero_gpu_offline": True,
            "c3_role": "exploratory_heterogeneous_null_ablation",
            "estimand_and_test_family": {
                "primary_harmonized_endpoint": "first_answer_logit_kl (E5 kl_vs_oneshot + 27B kl_first_vs_oracle)",
                "primary_estimand": "equal_weight_6_setup_t_df5 on [0.80, 1.25] log-KL GMR",
                "secondary_endpoint": "answer_mean_kl (E5 answer_kl_vs_oneshot + 27B drift_mean_ans_kl; bit-identical on E5, 4-token mean on 27B)",
                "multiple_testing_family": "Exploratory C3 ablation; 8 GDN setups pairwise sign-test family alpha_bonf=0.00625 and 6-setup family alpha_bonf=0.00833 disclosed for reference; no confirmation/equivalence claimed.",
            },
            "headline_exclusions": "Do NOT claim 'independent settings', 'equivalence established', or 'zero gain' from C3.",
            "task_identity_overlap": {
                "e5_join_key": "(sample_index, depth, tpl_id, old, new)",
                "2b_16k_vs_2b_32k_overlap": f"{len(keys_2b_16 & keys_2b_32)}/{len(keys_2b_16)}",
                "2b_tail_16k_vs_32k_joined": f"{len(tail_16k_joined)}/150",
                "2b_n200_vs_9b_4k_overlap": f"{len(keys_2b_16 & keys_9b_4k)}/200",
                "2b_n200_vs_9b_16k_overlap": f"{len(keys_2b_16 & keys_9b_16k)}/200",
                "2b_n50_vs_9b_32k_overlap": f"{len({_task_key(r) for r in rows_2b_32k_p} & keys_9b_32k)}/50",
                "27b_distinct_e2_tasks": len([r for r in rows_27b if "J_local" in r.get("conditions", {})]),
            },
            "reporting_policy": "n=50 preserved untouched as PRIMARY; n=200 and n=50..199 tail reported in separate columns (never merged silently into Primary).",
            "accuracy_metric_note": "EM fields evaluate 8-token substring containment (`contains_new`), NOT strict exact-string match.",
            "gate_thresholds_unchanged": {
                "source_of_truth": "pilot/w5_9b_e5_runner.py:1678-1679",
                "g1a_max_diff": 1e-5,
                "g1b_strict_kl": 1e-4,
                "g1b_relaxed_rule": "g1b_pass_strict or (max_logit_diff_vs_oneshot <= 0.28125 and kl_vs_oneshot <= 1.5e-3)",
            },
        },
        "gate_and_accuracy_side_by_side": {
            "2B_16K": {
                "primary_n50": summarize_suite_metrics(arr_2b_16k_p),
                "extension_n200": summarize_suite_metrics(arr_2b_16k_ext),
                "pure_extension_slice_s50_199_n150": summarize_suite_metrics(arr_2b_16k_tail),
            },
            "2B_32K": {
                "primary_n50": summarize_suite_metrics(arr_2b_32k_p),
                "extension_n200": summarize_suite_metrics(arr_2b_32k_ext),
                "pure_extension_slice_s50_199_n150": summarize_suite_metrics(arr_2b_32k_tail),
            },
        },
        "c3_harmonized_first_answer_logit_kl": harm_first,
        "c3_harmonized_answer_mean_kl": harm_ansmean,
        # Top-level canonical keys use harmonized first-answer-logit KL (never mixed endpoints):
        "c3_tost_primary_n50_cohort": harm_first["primary_n50_cohort"],
        "c3_tost_extension_n200_cohort": harm_first["extension_n200_nested_cohort"],
    }

    out_path = PILOT_DIR / "w12_c_offline_pairing_n200_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
