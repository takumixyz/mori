"""Stage-C / Wave 13 Standalone Live Online Fallback & Held-Out Unseen Template Runner (`w13_fallback_runner.py`).

Strictly follows W12-2 & G4 execution constraints without modifying `w5_9b_e5_runner.py`:
  1. Frozen Pre-Eval Detectors (ZERO tuning on evaluation / held-out sets):
       - `2B` pipeline (`--detector osc`): uses frozen `OSC` analytical formula from
         `w12_prop1_perturbation_decay.py` with frozen threshold `tau_osc = 0.40`
         (calibrated strictly on `w11_detector_v2_clean_2b` dev split: TPR=1.00, FPR=0.0455 <= 5%),
         plus passive logging of binary `prop1_analytical_flag` (`osc_bin_flag`).
       - `9B` pipeline (`--detector lr`): uses frozen 11-feature L2 Logistic Regression
         fit strictly on `w11_detector_v2_clean_9b` dev split (`n=100`, neg_max=0.02435, pos_min=0.94109,
         exact dev extrema midpoint = 0.48272) with fixed operating threshold `tau_lr = 0.50`.
  2. Full Feature & `hook_stats` Persistence:
       Persists full `c_hat`, `gamma_scalar`, state norms, and all 11 features to JSONL.
  3. Reports Metrics (Overall, Per-Suite, Per-Regime):
       - Latency scope note: `olocal_wall_ms`, `fb_tail_ms`, and `always_suffix_e2e_ms` are
         timed GPU-path means (prefix H2D transfer, pre-edit/fallback `clone_cache()`, and baseline H2D
         occur before their respective timers; KV splice uses `torch.cat(...).contiguous()` plus suffix
         key rotation when `delta_L != 0`).
       - Quantile convention: upper-order-statistic (`idx = n // 2` for p50, `min(n-1, int(0.95 * n))` for p95).
       - `pr_auc`: rank-based average precision (`_pr_auc_np`); suppressed (`None`) on single-class strata.
  4. Strict 2B (OSC) vs 9B (LR) Pipeline Separation:
       Separate run-tagged invocations (note `2b_osc` 2K serving and `2b_16k_osc` 16K serving reuse the
       same 100 deterministic tag-stripped case keys).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
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
    GDNSuffixHookCollector,
    append_record_jsonl,
    apply_jlocal_scalar_update,
    clone_cache,
    compute_state_delta_fp64,
    count_linear_layers,
    eval_from_ctx_cache_in_place,
    extract_kv_slice,
    get_eos_ids,
    has_logits_to_keep_arg,
    move_cache_device,
    run_chunked_prefill,
    score_condition,
    snap_states,
)
from w10_p1_3_e4_e7_counterfact import apply_delta_rope_to_kv_cache  # noqa: E402
from w11_varlen_e10_e12_detector import (  # noqa: E402
    _align_contiguous_edit_span,
    fit_logistic_l2_np,
    load_done_keys,
    predict_logistic_np,
    wilson_ci,
)
from w12_prop1_perturbation_decay import (  # noqa: E402
    _auroc_np,
    _pr_auc_np,
    clopper_pearson_lower,
    prop1_analytical_flag,
    prop1_analytical_risk_score,
)


FROZEN_OSC_TAU_DEV_2B = 0.40  # Selected ONLY on w11_detector_v2_clean_2b dev split (TPR=1.00, FPR=0.0455)
FROZEN_LR_TAU_DEV_9B = 0.50   # Fixed operating point near w11_detector_v2_clean_9b dev extrema midpoint 0.48272 (neg_max=0.02435, pos_min=0.94109)

UNSEEN_FILLERS = [
    "Meteorological station bulletin: barometric pressure remained constant throughout the morning shift.",
    "Logistics manifest summary: container seals on berth seven passed automated barcode inspection.",
    "Botanical conservatory archive: alpine fern humidity sensors logged steady overnight readings.",
    "Observatory dome schedule: meridian shutter alignment completed ahead of twilight calibration.",
    "Hydrology gauge report: upper basin reservoir outflow held within seasonal targets.",
    "Acoustic lab memo: anechoic chamber floor isolators passed quarterly vibration audit.",
    "Maritime buoy telemetry: coastal swell period averaged nine seconds across the channel.",
    "Railroad signaling digest: siding switch indicators verified clear along corridor four.",
    "Cartography survey note: elevation benchmarks matched geodetic reference markers.",
    "Geothermal plant log: secondary heat exchanger loop stabilized at nominal pressure.",
    "Archival bindery ledger: linen folio bindings stored under controlled museum lighting.",
    "Aviation weather brief: high-altitude jet stream shear remained below advisory limits.",
]

UNSEEN_NAMES = [
    "Henrik", "Nadia", "Kaito", "Soren", "Miriam", "Tarek",
    "Elena", "Darius", "Priya", "Lukas", "Camille", "Mateo",
]

UNSEEN_CF_FACTS = [
    ("In the Meridian Ledger, the chief curator of the Hall is {old}.",
     "In the Meridian Ledger, the chief curator of the Hall is {new}.",
     "Question: According to the Meridian Ledger, who is the chief curator of the Hall? Reply with only the name.",
     [("Vance", "Keller"), ("Mercer", "Thorne"), ("Bauer", "Nolan"), ("Sato", "Ibrahim"), ("Reyes", "Lindqvist")]),
    ("Quarterly shipping docket lists the primary port hub as {old}.",
     "Quarterly shipping docket lists the primary port hub as {new}.",
     "Question: What is the primary port hub on the quarterly shipping docket? Reply with only the city.",
     [("Osaka", "Bergen"), ("Lisbon", "Zurich"), ("Vienna", "Ottawa"), ("Prague", "Bilbao"), ("Ghent", "Verona")]),
    ("The alpine research station runs under operating directive {old}.",
     "The alpine research station runs under operating directive {new}.",
     "Question: Which operating directive governs the alpine research station? Reply with only the directive name.",
     [("Cobalt", "Amber"), ("Boreal", "Quartz"), ("Zephyr", "granite"), ("Helios", "Cinder"), ("Vesper", "Summit")]),
    ("Harbor dispatch confirms flagship vessel {old} holds berth priority.",
     "Harbor dispatch confirms flagship vessel {new} holds berth priority.",
     "Question: Which flagship vessel holds berth priority? Reply with only the vessel name.",
     [("Aurora", "Kestrel"), ("Mariner", "Sentinel"), ("Halcyon", "Vanguard"), ("Pioneer", "Compass"), ("Solstice", "Horizon")]),
    ("Regional energy compact sets baseline tariff tier to {old}.",
     "Regional energy compact sets baseline tariff tier to {new}.",
     "Question: What is the baseline tariff tier set by the compact? Reply with only the tier code.",
     [("Silver", "Platinum"), ("Bronze", "Titanium"), ("Alpha", "Omega"), ("Delta", "Sigma"), ("Prime", "Ultra")]),
]

UNSEEN_JSON_SPECS = [
    ("sandbox", "enterprise-dedicated", "ap-south", "eu-central-2", "throttled", "priority-burst"),
    ("basic", "sovereign-shield", "us-east", "ap-northeast-1", "queued", "realtime-bypass"),
    ("trial", "regulated-cluster", "eu-west", "ca-central-1", "limited", "unrestricted"),
    ("starter", "dedicated-enclave", "sa-east", "ap-southeast-2", "paused", "active-stream"),
    ("legacy", "high-throughput-v4", "us-west", "eu-north-1", "degraded", "verified-ok"),
]


def _sync(device: str) -> float:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _pad_to_tokens(tok: Any, target_tokens: int, seed_offset: int) -> str:
    chunks = []
    idx = 0
    while True:
        chunks.append(UNSEEN_FILLERS[(seed_offset + idx) % len(UNSEEN_FILLERS)])
        txt = " ".join(chunks)
        if len(tok.encode(txt, add_special_tokens=False)) >= target_tokens:
            return txt
        idx += 1


def freeze_lr_from_dev_split(dev_jsonl: Path, tau_lr: float = FROZEN_LR_TAU_DEV_9B) -> Dict[str, Any]:
    """Fit and freeze L2 logistic regression strictly on dev split (`w11_detector_v2_clean_9b`)."""
    assert dev_jsonl.exists(), f"Missing dev split file: {dev_jsonl}"
    rows = [json.loads(l) for l in dev_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    dev_rows = [r for r in rows if r.get("split") == "dev"]
    assert len(dev_rows) > 0, f"No split=='dev' rows in {dev_jsonl}"
    fkeys = [
        "rel_delta_s_mean", "rel_delta_s_max", "c_hat_mean", "gamma_mean", "abs_delta_L",
        "force_no_rope", "margin_jl", "entropy_jl", "shift_kl_old_vs_jl",
        "top1_flipped_from_old", "stale_top1_prob_under_jl",
    ]
    X_dev = np.array([[float(r["features"][k]) for k in fkeys] for r in dev_rows], dtype=np.float64)
    y_dev = np.array([int(r["is_failure_jlocal"]) for r in dev_rows], dtype=np.float64)
    mu, sd, w, b = fit_logistic_l2_np(X_dev, y_dev, reg=1.0, steps=600, lr=0.25)
    return {
        "fkeys": fkeys,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "w": w.tolist(),
        "b": float(b),
        "tau": float(tau_lr),
        "n_dev": len(dev_rows),
        "source": dev_jsonl.name,
    }


def score_frozen_lr(lr_blob: Dict[str, Any], feats: Dict[str, Any]) -> Tuple[float, int]:
    x = np.array([[float(feats[k]) for k in lr_blob["fkeys"]]], dtype=np.float64)
    mu = np.asarray(lr_blob["mu"], dtype=np.float64)
    sd = np.asarray(lr_blob["sd"], dtype=np.float64)
    w = np.asarray(lr_blob["w"], dtype=np.float64)
    b = float(lr_blob["b"])
    prob = float(predict_logistic_np(x, mu, sd, w, b)[0])
    return prob, int(prob >= float(lr_blob["tau"]))


def score_frozen_osc(feats: Dict[str, Any], tau_osc: float = FROZEN_OSC_TAU_DEV_2B) -> Tuple[float, int, int]:
    sc = float(prop1_analytical_risk_score(feats))
    fl_tau = int(sc >= tau_osc)
    fl_bin = int(prop1_analytical_flag(feats))
    return sc, fl_tau, fl_bin


def build_g4_unseen_ood_cases(tok: Any, n_linear: int, n_per_regime: int) -> List[Dict[str, Any]]:
    rng = random.Random(20261122)
    cases: List[Dict[str, Any]] = []

    # 1. unseen_cf_news_nl
    cnt = 0
    for idx in range(n_per_regime * 4):
        if cnt >= n_per_regime:
            break
        tpl_o, tpl_n, q_s, pairs = UNSEEN_CF_FACTS[idx % len(UNSEEN_CF_FACTS)]
        old_v, new_v = pairs[(idx // len(UNSEEN_CF_FACTS)) % len(pairs)]
        pre = " ".join(UNSEEN_FILLERS[(idx + j) % len(UNSEEN_FILLERS)] for j in range(5))
        suf = " ".join(UNSEEN_FILLERS[(idx + 6 + j) % len(UNSEEN_FILLERS)] for j in range(10))
        u_o = f"{pre} {tpl_o.format(old=old_v)} {suf}\n\n{q_s}"
        u_e = f"{pre} {tpl_n.format(new=new_v)} {suf}\n\n{q_s}"
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
        if al is None:
            continue
        cases.append({
            "suite": "g4_unseen_ood", "regime": "unseen_cf_news_nl", "idx": cnt,
            "healable": True, "old_v": old_v, "new_v": new_v, "al": al,
        })
        cnt += 1

    # 2. unseen_api_json_varlen
    cnt = 0
    for idx in range(n_per_regime * 4):
        if cnt >= n_per_regime:
            break
        old_t, new_t, old_r, new_r, old_m, new_m = UNSEEN_JSON_SPECS[idx % len(UNSEEN_JSON_SPECS)]
        svc = UNSEEN_NAMES[idx % len(UNSEEN_NAMES)]
        pre = " ".join(UNSEEN_FILLERS[(idx + j) % len(UNSEEN_FILLERS)] for j in range(5))
        suf = " ".join(UNSEEN_FILLERS[(idx + 7 + j) % len(UNSEEN_FILLERS)] for j in range(10))
        q_s = f"Question: For service {svc}, what is the updated deploy_mode string in the API config JSON? Reply with only the exact string."
        u_o = f'{pre}\nAPI Config JSON: {{"service": "{svc}", "tier": "{old_t}", "zone": "{old_r}", "deploy_mode": "{old_m}"}}\n{suf}\n\n{q_s}'
        u_e = f'{pre}\nAPI Config JSON: {{"service": "{svc}", "tier": "{new_t}", "zone": "{new_r}", "deploy_mode": "{new_m}"}}\n{suf}\n\n{q_s}'
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
        if al is None or abs(al["delta_L"]) < 4:
            continue
        cases.append({
            "suite": "g4_unseen_ood", "regime": "unseen_api_json_varlen", "idx": cnt,
            "healable": True, "old_v": old_m, "new_v": new_m, "al": al,
        })
        cnt += 1

    # 3. unseen_entity_collision
    cnt = 0
    for idx in range(n_per_regime * 4):
        if cnt >= n_per_regime:
            break
        n1, n2 = UNSEEN_NAMES[idx % len(UNSEEN_NAMES)], UNSEEN_NAMES[(idx + 1) % len(UNSEEN_NAMES)]
        c_o = str(rng.randint(1000, 4999))
        c_n = str(rng.randint(5000, 9999))
        c_sib = str(int(c_o[:2] + str(rng.randint(10, 99))))
        if c_o[0] == c_n[0] or c_sib == c_o or c_sib == c_n:
            continue
        pre = " ".join(UNSEEN_FILLERS[(idx + j) % len(UNSEEN_FILLERS)] for j in range(4))
        suf = " ".join(UNSEEN_FILLERS[(idx + 5 + j) % len(UNSEEN_FILLERS)] for j in range(10))
        q_s = f"Question: What is the harbor clearance pin for {n1}? Reply with 4 digits."
        u_o = f"{pre} Clearance registry: {n1} uses pin {c_o}; {n1} Jr uses pin {c_sib}; {n2} uses pin 8194. {suf}\n\n{q_s}"
        u_e = f"{pre} Clearance registry: {n1} uses pin {c_n}; {n1} Jr uses pin {c_sib}; {n2} uses pin 8194. {suf}\n\n{q_s}"
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
        if al is None:
            continue
        cases.append({
            "suite": "g4_unseen_ood", "regime": "unseen_entity_collision", "idx": cnt,
            "healable": True, "old_v": c_o, "new_v": c_n, "al": al,
        })
        cnt += 1

    # 4. unseen_twohop_bridge (healable via live suffix_recompute)
    cnt = 0
    for idx in range(n_per_regime * 4):
        if cnt >= n_per_regime:
            break
        tag_a, tag_b = f"VAL-{rng.randint(100,499)}", f"VAL-{rng.randint(500,999)}"
        code_a, code_b = str(rng.randint(1000, 4999)), str(rng.randint(5000, 9999))
        if code_a[0] == code_b[0]:
            continue
        pre = " ".join(UNSEEN_FILLERS[(idx + j) % len(UNSEEN_FILLERS)] for j in range(4))
        suf = " ".join(UNSEEN_FILLERS[(idx + 5 + j) % len(UNSEEN_FILLERS)] for j in range(8))
        q_s = "Question: What is the resolved numerical backup key for Primary-Node? Reply with 4 digits."
        u_o = f"{pre} Primary-Node points to slot {tag_a}. {suf} Lookup table: {tag_a} -> {code_a}; {tag_b} -> {code_b}.\n\n{q_s}"
        u_e = f"{pre} Primary-Node points to slot {tag_b}. {suf} Lookup table: {tag_a} -> {code_a}; {tag_b} -> {code_b}.\n\n{q_s}"
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
        if al is None or al["b_old"] >= al["T_old"] - 10:
            continue
        cases.append({
            "suite": "g4_unseen_ood", "regime": "unseen_twohop_bridge", "idx": cnt,
            "healable": True, "old_v": code_a, "new_v": code_b, "al": al,
        })
        cnt += 1

    # 5. unseen_coupled_e12_single (conflicting authoritative downstream quote -> unhealable single-span)
    cnt = 0
    for idx in range(n_per_regime * 4):
        if cnt >= n_per_regime:
            break
        svc = UNSEEN_NAMES[idx % len(UNSEEN_NAMES)]
        v_o, v_n = str(rng.randint(1000, 4999)), str(rng.randint(5000, 9999))
        if v_o[0] == v_n[0]:
            continue
        pre = " ".join(UNSEEN_FILLERS[(idx + j) % len(UNSEEN_FILLERS)] for j in range(4))
        mid = " ".join(UNSEEN_FILLERS[(idx + 5 + j) % len(UNSEEN_FILLERS)] for j in range(6))
        suf = " ".join(UNSEEN_FILLERS[(idx + 9 + j) % len(UNSEEN_FILLERS)] for j in range(6))
        q_s = f"Question: Per the final signed manifest, what is the active permit code for {svc}? Reply with 4 digits."
        u_o = f"{pre} Draft permit for {svc} is {v_o}. {mid} Final signed manifest (authoritative): active permit code for {svc} remains {v_o}. {suf}\n\n{q_s}"
        u_e = f"{pre} Draft permit for {svc} is {v_n}. {mid} Final signed manifest (authoritative): active permit code for {svc} remains {v_o}. {suf}\n\n{q_s}"
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
        if al is None:
            continue
        cases.append({
            "suite": "g4_unseen_ood", "regime": "unseen_coupled_e12_single", "idx": cnt,
            "healable": False, "old_v": v_o, "new_v": v_n, "al": al,
        })
        cnt += 1

    return cases


def build_w12_2_serving_traffic(
    tok: Any, n_linear: int, n_benign: int, n_coupled: int, target_ctx_tokens: int = 2048
) -> List[Dict[str, Any]]:
    rng = random.Random(20261123)
    cases: List[Dict[str, Any]] = []
    if target_ctx_tokens >= 8192:
        pre_tgt = target_ctx_tokens // 5
        suf_tgt = (target_ctx_tokens * 4) // 5
    else:
        pre_tgt = max(200, target_ctx_tokens // 4)
        suf_tgt = max(400, (target_ctx_tokens * 3) // 5)
    pre_pad = _pad_to_tokens(tok, pre_tgt, seed_offset=1)
    suf_pad = _pad_to_tokens(tok, suf_tgt, seed_offset=5)

    cnt = 0
    for idx in range(n_benign * 3):
        if cnt >= n_benign:
            break
        nm = UNSEEN_NAMES[idx % len(UNSEEN_NAMES)]
        if idx % 3 == 0:
            old_t, new_t, old_r, new_r, old_m, new_m = UNSEEN_JSON_SPECS[idx % len(UNSEEN_JSON_SPECS)]
            q_s = f"Question: What is {nm}'s updated deploy_mode in the config? Reply with only the string."
            u_o = f'{pre_pad}\nConfig: {{"owner": "{nm}", "tier": "{old_t}", "deploy_mode": "{old_m}"}}\n{suf_pad}\n\n{q_s}'
            u_e = f'{pre_pad}\nConfig: {{"owner": "{nm}", "tier": "{new_t}", "deploy_mode": "{new_m}"}}\n{suf_pad}\n\n{q_s}'
            old_v, new_v, reg = old_m, new_m, "benign_varlen_json"
        else:
            v_o, v_n = str(rng.randint(1000, 4999)), str(rng.randint(5000, 9999))
            if v_o[0] == v_n[0]:
                continue
            q_s = f"Question: What is the assigned vault key for {nm}? Reply with 4 digits."
            u_o = f"{pre_pad} Audit log: {nm} holds vault key {v_o}. {suf_pad}\n\n{q_s}"
            u_e = f"{pre_pad} Audit log: {nm} holds vault key {v_n}. {suf_pad}\n\n{q_s}"
            old_v, new_v, reg = v_o, v_n, "benign_clean_fact"
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
        if al is None:
            continue
        cases.append({
            "suite": "w12_2_serving", "regime": reg, "idx": cnt,
            "healable": True, "old_v": old_v, "new_v": new_v, "al": al,
        })
        cnt += 1

    cnt_c = 0
    for idx in range(n_coupled * 4):
        if cnt_c >= n_coupled:
            break
        tag_a, tag_b = f"KEY-{rng.randint(100,499)}", f"KEY-{rng.randint(500,999)}"
        code_a, code_b = str(rng.randint(1000, 4999)), str(rng.randint(5000, 9999))
        if code_a[0] == code_b[0]:
            continue
        q_s = "Question: What is the resolved numerical backup key for Primary-Node? Reply with 4 digits."
        u_o = f"{pre_pad} Primary-Node points to {tag_a}. {suf_pad} Key table: {tag_a} -> {code_a}; {tag_b} -> {code_b}.\n\n{q_s}"
        u_e = f"{pre_pad} Primary-Node points to {tag_b}. {suf_pad} Key table: {tag_a} -> {code_a}; {tag_b} -> {code_b}.\n\n{q_s}"
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
        if al is None:
            continue
        cases.append({
            "suite": "w12_2_serving", "regime": "coupled_twohop_fail", "idx": cnt_c,
            "healable": True, "old_v": code_a, "new_v": code_b, "al": al,
        })
        cnt_c += 1

    return cases


def execute_live_routed_sample(
    model: Any,
    base_model: Any,
    tok: Any,
    eos_ids: List[int],
    detector_mode: str,
    lr_blob: Optional[Dict[str, Any]],
    tau_osc: float,
    suite_name: str,
    regime: str,
    healable_by_suffix: bool,
    ids_o: torch.Tensor,
    ids_e: torch.Tensor,
    ids_q: torch.Tensor,
    p0: int,
    b_old: int,
    b_new: int,
    T_old: int,
    T_new: int,
    delta_L: int,
    new_v: str,
    old_v: str,
    n_linear: int,
    d_k: int,
    w_conv: int,
    device: str,
    use_l2k: bool,
) -> Dict[str, Any]:
    """Execute O_local + persist FULL hook_stats/features + LIVE branch fallback via frozen detector."""
    f_o = tok.encode(old_v, add_special_tokens=False)[0]
    f_e = tok.encode(new_v, add_special_tokens=False)[0]
    suf_len = T_old - b_old

    with torch.no_grad():
        # Ground-truth oneshot oracle logits for KL evaluation (NEVER exposed to detector)
        _, c_one = run_chunked_prefill(model, base_model, ids_e, 0, None, 2048, use_l2k)
        ev_one = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_one, T_new, ids_q, use_l2k, max_new_tokens=8)
        lg_one = ev_one["first_logits"].detach().clone()
        del c_one

        # Prebuild prefix checkpoint c_pre and prefilled old cache c_ctx_old
        _, c_pre = run_chunked_prefill(model, base_model, ids_o[:, :p0], 0, None, 2048, use_l2k)
        c_pre_cpu = move_cache_device(c_pre, "cpu")
        _, c_old_b = run_chunked_prefill(model, base_model, ids_o[:, p0:b_old], p0, c_pre, 2048, use_l2k)
        del c_pre
        s_old_b = snap_states(c_old_b)
        with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as collector:
            if suf_len > 0:
                _, c_ctx_old = run_chunked_prefill(model, base_model, ids_o[:, b_old:T_old], b_old, c_old_b, 2048, use_l2k)
            else:
                c_ctx_old = c_old_b
        hs = collector.finalize(expected_suffix_len=suf_len, device=device)

        # Pre-edit query forward logits (cached Turn-0 reference)
        ev_old = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_ctx_old), T_old, ids_q, use_l2k, max_new_tokens=1)
        lg_old_q = ev_old["first_logits"].float()

        # ------------------------------------------------------------------
        # TIMED LIVE ONLINE ROUTED EXECUTION
        # ------------------------------------------------------------------
        c_pre_live = move_cache_device(c_pre_cpu, device)
        c_jl = clone_cache(c_ctx_old)

        t0_ol = _sync(device)
        _, c_new_b = run_chunked_prefill(model, base_model, ids_e[:, p0:b_new], p0, c_pre_live, 2048, use_l2k)
        s_new_b = snap_states(c_new_b)
        new_blk_kv = extract_kv_slice(c_new_b, p0, b_new)
        delta = compute_state_delta_fp64(s_old_b, s_new_b)

        apply_jlocal_scalar_update(c_jl, delta, hs["gamma_scalar"], b_old, T_old, w_conv, use_decay=True)
        apply_delta_rope_to_kv_cache(c_jl, new_blk_kv, p0, b_old, b_new, T_old, delta_L, base_model, apply_delta_rope=True)
        ev_jl = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_jl, T_new, ids_q, use_l2k, max_new_tokens=8)
        olocal_wall_ms = (_sync(device) - t0_ol) * 1000.0
        lg_jl = ev_jl["first_logits"].float()
        del c_jl, c_pre_live

        # Extract & persist FULL hook_stats + 11 features (never deleted!)
        t0_feat = _sync(device)
        s_old_norms = [float(r_o.float().norm().item()) for r_o in s_old_b["rec"] if r_o is not None]
        delta_norms = [float(d_r.float().norm().item()) for d_r in delta["rec"] if d_r is not None]
        rel_ds = [
            float(d_r.float().norm().item() / (r_o.float().norm().item() + 1e-6))
            for r_o, d_r in zip(s_old_b["rec"], delta["rec"])
            if r_o is not None and d_r is not None
        ]
        chat_layer_means = [float(x.mean().item()) for x in hs["c_hat_cpu"]]
        chat_all = torch.cat([x.reshape(-1) for x in hs["c_hat_cpu"]]) if hs["c_hat_cpu"] else torch.zeros(1)
        gam_layer_means = [float(x.mean().item()) for x in hs["gamma_scalar"]]
        gam_all = torch.cat([x.reshape(-1) for x in hs["gamma_scalar"]]) if hs["gamma_scalar"] else torch.ones(1)

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
            "c_hat_mean": float(chat_all.float().mean().item()),
            "gamma_mean": float(gam_all.float().mean().item()),
            "abs_delta_L": float(abs(delta_L)),
            "force_no_rope": 0.0,
            "margin_jl": margin_jl,
            "entropy_jl": entropy_jl,
            "shift_kl_old_vs_jl": shift_kl,
            "top1_flipped_from_old": top1_flipped,
            "stale_top1_prob_under_jl": stale_mass,
        }
        hook_stats_persisted = {
            "c_hat_mean": float(chat_all.float().mean().item()),
            "c_hat_max": float(chat_all.float().max().item()),
            "c_hat_p95": float(torch.quantile(chat_all.float(), 0.95).item()),
            "c_hat_per_layer": chat_layer_means,
            "gamma_mean": float(gam_all.float().mean().item()),
            "gamma_min": float(gam_all.float().min().item()),
            "gamma_per_layer": gam_layer_means,
            "s_old_frob_mean": sum(s_old_norms) / max(1, len(s_old_norms)),
            "delta_s_frob_mean": sum(delta_norms) / max(1, len(delta_norms)),
        }

        if detector_mode == "osc":
            score_val, active_flag, aux_bin_flag = score_frozen_osc(feats, tau_osc=tau_osc)
            passive_val = None
        else:
            assert lr_blob is not None
            score_val, active_flag = score_frozen_lr(lr_blob, feats)
            aux_bin_flag = int(prop1_analytical_flag(feats))
            passive_val = float(prop1_analytical_risk_score(feats))
        feat_ms = (_sync(device) - t0_feat) * 1000.0

        # TRUE LIVE CONDITIONAL FALLBACK BRANCH ON GPU
        # Only executes suffix_recompute in live path when active_flag == 1;
        # we ALSO separately time standalone always_suffix_ms & shadow fallback_em for audit parity!
        live_recomputed = bool(active_flag == 1)
        fb_tail_ms = 0.0
        ev_routed = ev_jl
        if live_recomputed:
            c_fb = clone_cache(c_new_b)
            t0_fb = _sync(device)
            if suf_len > 0:
                _, c_fb = run_chunked_prefill(model, base_model, ids_e[:, b_new:T_new], b_new, c_fb, 2048, use_l2k)
            ev_fb = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_fb, T_new, ids_q, use_l2k, max_new_tokens=8)
            fb_tail_ms = (_sync(device) - t0_fb) * 1000.0
            ev_routed = ev_fb
            del c_fb

        live_routed_e2e_ms = olocal_wall_ms + feat_ms + fb_tail_ms

        # Standalone always_suffix_recompute baseline from c_pre ([p0, T_new) + query)
        del c_new_b, c_ctx_old
        c_suf = move_cache_device(c_pre_cpu, device)
        del c_pre_cpu
        t0_suf = _sync(device)
        _, c_suf = run_chunked_prefill(model, base_model, ids_e[:, p0:T_new], p0, c_suf, 2048, use_l2k)
        ev_suf = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_suf, T_new, ids_q, use_l2k, max_new_tokens=8)
        always_suffix_e2e_ms = (_sync(device) - t0_suf) * 1000.0
        del c_suf

    sc_jl = score_condition(ev_jl, new_v, old_v, f_e, f_o, lg_one)
    sc_routed = score_condition(ev_routed, new_v, old_v, f_e, f_o, lg_one) if live_recomputed else sc_jl
    sc_suf = score_condition(ev_suf, new_v, old_v, f_e, f_o, lg_one)

    return {
        "suite": suite_name,
        "regime": regime,
        "detector_mode": detector_mode,
        "frozen_threshold": float(tau_osc if detector_mode == "osc" else lr_blob["tau"]),
        "healable_by_suffix": bool(healable_by_suffix),
        "T_old": int(T_old),
        "T_new": int(T_new),
        "p0": int(p0),
        "b_new": int(b_new),
        "suf_len": int(suf_len),
        "delta_L": int(delta_L),
        "hook_stats": hook_stats_persisted,
        "features": feats,
        "detector_score": float(score_val),
        "detector_flag": int(active_flag),
        "live_recomputed": bool(live_recomputed),
        "aux_osc_bin_flag": int(aux_bin_flag),
        "passive_osc_score": passive_val,
        "is_failure_jlocal": int(not sc_jl["contains_new"]),
        "always_jlocal": {
            "em": bool(sc_jl["contains_new"]),
            "leak": bool(sc_jl["contains_old"]),
            "kl": float(sc_jl["kl_vs_oneshot"]),
        },
        "live_routed": {
            "em": bool(sc_routed["contains_new"]),
            "leak": bool(sc_routed["contains_old"]),
            "kl": float(sc_routed["kl_vs_oneshot"]),
        },
        "always_suffix": {
            "em": bool(sc_suf["contains_new"]),
            "leak": bool(sc_suf["contains_old"]),
            "kl": float(sc_suf["kl_vs_oneshot"]),
        },
        "timing_ms": {
            "olocal_wall_ms": float(olocal_wall_ms),
            "detector_feat_ms": float(feat_ms),
            "fb_tail_ms": float(fb_tail_ms),
            "live_routed_e2e_ms": float(live_routed_e2e_ms),
            "always_suffix_e2e_ms": float(always_suffix_e2e_ms),
        },
    }


def _aggregate_block(rows: List[Dict[str, Any]], detector_mode: str) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    y = np.array([int(r["is_failure_jlocal"]) for r in rows], dtype=np.int64)
    s = np.array([float(r["detector_score"]) for r in rows], dtype=np.float64)
    fl = np.array([int(r["detector_flag"]) for r in rows], dtype=np.int64)
    rc = np.array([int(bool(r["live_recomputed"])) for r in rows], dtype=np.int64)

    tp = int(((fl == 1) & (y == 1)).sum())
    fp = int(((fl == 1) & (y == 0)).sum())
    fn = int(((fl == 0) & (y == 1)).sum())
    tn = int(((fl == 0) & (y == 0)).sum())
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    tpr_lb = clopper_pearson_lower(tp, n_pos) if n_pos > 0 else 0.0
    tnr_lb = clopper_pearson_lower(tn, n_neg) if n_neg > 0 else 0.0

    brier, ece = None, None
    if detector_mode == "lr":
        probs = np.clip(s, 0.0, 1.0)
        brier = float(np.mean((probs - y) ** 2))
        bins = np.linspace(0.0, 1.0, 11)
        ece_val = 0.0
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = (probs >= b0) & (probs <= b1 if b1 == 1.0 else probs < b1)
            if np.any(m):
                ece_val += (float(np.sum(m)) / n) * abs(float(np.mean(probs[m])) - float(np.mean(y[m])))
        ece = float(ece_val)

    benign_rows = [r for r in rows if not r["is_failure_jlocal"]]
    benign_rc = sum(1 for r in benign_rows if r["live_recomputed"]) / max(1, len(benign_rows))

    routed_lat = sorted(float(r["timing_ms"]["live_routed_e2e_ms"]) for r in rows)
    suf_lat = sorted(float(r["timing_ms"]["always_suffix_e2e_ms"]) for r in rows)
    ol_lat = sorted(float(r["timing_ms"]["olocal_wall_ms"]) for r in rows)

    def _branch_stats(bkey: str) -> Dict[str, Any]:
        k_em = sum(1 for r in rows if r[bkey]["em"])
        k_lk = sum(1 for r in rows if r[bkey]["leak"])
        kls = [float(r[bkey]["kl"]) for r in rows]
        return {
            "em": float(k_em / n),
            "em_ci95": wilson_ci(k_em, n),
            "leak_rate": float(k_lk / n),
            "mean_kl": float(sum(kls) / n),
            "median_kl": float(sorted(kls)[n // 2]),
        }

    heal_rows = [r for r in rows if r["healable_by_suffix"]]
    heal_summary = {
        "n": len(heal_rows),
        "always_jlocal_em": sum(1 for r in heal_rows if r["always_jlocal"]["em"]) / max(1, len(heal_rows)),
        "live_routed_em": sum(1 for r in heal_rows if r["live_routed"]["em"]) / max(1, len(heal_rows)),
        "always_suffix_em": sum(1 for r in heal_rows if r["always_suffix"]["em"]) / max(1, len(heal_rows)),
    }

    return {
        "n": n,
        "n_fail_jlocal": n_pos,
        # 5 Mandatory Metrics:
        "flag_rate": float(fl.mean()),
        "recompute_rate": float(rc.mean()),
        "recompute_rate_benign_fp": float(benign_rc),
        "always_jlocal": _branch_stats("always_jlocal"),
        "live_routed": _branch_stats("live_routed"),
        "always_suffix": _branch_stats("always_suffix"),
        "healable_only": heal_summary,
        "latency_ms": {
            "timing_scope": "timed_gpu_path_means (pre-timer H2D & cache clones excluded; apply_delta_rope concatenates KV)",
            "quantile_convention": "upper-order-statistic: sorted[n//2] for p50, sorted[min(n-1, int(0.95*n))] for p95",
            "olocal_mean": float(sum(ol_lat) / n),
            "live_routed_e2e_mean": float(sum(routed_lat) / n),
            "live_routed_e2e_p50": float(routed_lat[n // 2]),
            "live_routed_e2e_p95": float(routed_lat[min(n - 1, int(0.95 * n))]),
            "always_suffix_e2e_mean": float(sum(suf_lat) / n),
            "always_suffix_e2e_p50": float(suf_lat[n // 2]),
            "always_suffix_e2e_p95": float(suf_lat[min(n - 1, int(0.95 * n))]),
            "speedup_routed_vs_always_suffix": float((sum(suf_lat) / n) / max(1e-6, sum(routed_lat) / n)),
        },
        # Detection quality (G4):
        "auroc": float(_auroc_np(y, s)) if n_pos > 0 and n_neg > 0 else None,
        "pr_auc": float(_pr_auc_np(y, s)) if n_pos > 0 and n_neg > 0 else None,
        "pr_auc_definition": "rank-based average precision (_pr_auc_np); suppressed when n_pos==0 or n_neg==0",
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tpr": float(tp / max(1, n_pos)),
        "fpr": float(fp / max(1, n_neg)),
        "tpr_cp95_lb": float(tpr_lb),
        "tnr_cp95_lb": float(tnr_lb),
        "joint_auc_cp95_lb": float(max(0.0, tpr_lb + tnr_lb - 1.0)),
        "brier": brier,
        "ece": ece,
    }


def compute_tau_plateau_audit(rows: List[Dict[str, Any]], detector_mode: str, taus: Optional[List[float]] = None) -> Dict[str, Any]:
    """Audit frozen threshold plateau across fixed candidate points without re-running GPU."""
    if not rows:
        return {}
    if taus is None:
        taus = [0.2, 0.4, 0.7] if detector_mode == "osc" else [0.3, 0.5, 0.7]
    out: Dict[str, Any] = {}
    for tau in taus:
        n = len(rows)
        fl = [1 if float(r["detector_score"]) >= tau else 0 for r in rows]
        y = [int(r["is_failure_jlocal"]) for r in rows]
        tp = sum(1 for f, yi in zip(fl, y) if f == 1 and yi == 1)
        fp = sum(1 for f, yi in zip(fl, y) if f == 1 and yi == 0)
        n_pos = sum(y)
        n_neg = n - n_pos
        em_routed = sum(
            1
            for f, r in zip(fl, rows)
            if (r["always_suffix"]["em"] if f == 1 else r["always_jlocal"]["em"])
        )
        lk_routed = sum(
            1
            for f, r in zip(fl, rows)
            if (r["always_suffix"]["leak"] if f == 1 else r["always_jlocal"]["leak"])
        )
        kl_routed = sum(
            float(r["always_suffix"]["kl"] if f == 1 else r["always_jlocal"]["kl"])
            for f, r in zip(fl, rows)
        )
        out[f"tau_{tau:.1f}"] = {
            "tau": float(tau),
            "n": n,
            "flag_rate": float(sum(fl) / n),
            "recompute_rate": float(sum(fl) / n),
            "benign_fp_rate": float(fp / max(1, n_neg)),
            "tpr": float(tp / max(1, n_pos)) if n_pos > 0 else None,
            "fpr": float(fp / max(1, n_neg)) if n_neg > 0 else None,
            "tp": tp,
            "fp": fp,
            "routed_em": float(em_routed / n),
            "routed_leak": float(lk_routed / n),
            "routed_mean_kl": float(kl_routed / n),
            "standalone_substitution_note": "Offline sweep substitutes standalone always_suffix fields on flagged rows.",
        }
    return out


def summarize_jsonl(jsonl_path: Path, model_name: str, detector_mode: str, frozen_cfg: Dict[str, Any]) -> Dict[str, Any]:
    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    suites: Dict[str, Any] = {}
    for sname in sorted({r["suite"] for r in rows}):
        srows = [r for r in rows if r["suite"] == sname]
        by_reg = {}
        for reg in sorted({r["regime"] for r in srows}):
            by_reg[reg] = _aggregate_block([r for r in srows if r["regime"] == reg], detector_mode)
        suites[sname] = {
            "overall": _aggregate_block(srows, detector_mode),
            "by_regime": by_reg,
            "tau_plateau_audit": compute_tau_plateau_audit(srows, detector_mode),
        }
    return {
        "runner": "pilot/w13_fallback_runner.py",
        "model": model_name,
        "detector_mode": detector_mode,
        "frozen_config": frozen_cfg,
        "provenance_note": "Held-out unseen templates (g4_unseen_ood) & serving stream; healable_only excludes unhealable coupled e12 by generator label.",
        "total_records": len(rows),
        "overall": _aggregate_block(rows, detector_mode),
        "tau_plateau_audit": compute_tau_plateau_audit(rows, detector_mode),
        "suites": suites,
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

    cases = build_g4_unseen_ood_cases(tok, n_linear, n_per_regime=1)
    cases += build_w12_2_serving_traffic(tok, n_linear, n_benign=2, n_coupled=1, target_ctx_tokens=256)
    lr_blob = freeze_lr_from_dev_split(out_dir / "w11_detector_v2_clean_9b_records.jsonl")

    for mode, blob in [("osc", None), ("lr", lr_blob)]:
        dry_jsonl = out_dir / f"w13_fallback_dry_{mode}.jsonl"
        if dry_jsonl.exists():
            dry_jsonl.unlink()
        for idx, c in enumerate(cases):
            al = c["al"]
            rec = execute_live_routed_sample(
                model, base_model, tok, eos_ids, mode, blob, FROZEN_OSC_TAU_DEV_2B,
                c["suite"], c["regime"], c["healable"],
                torch.tensor([al["ids_o_list"]]), torch.tensor([al["ids_e_list"]]),
                torch.tensor([al["q_ids_list"]]),
                al["p0"], al["b_old"], al["b_new"], al["T_old"], al["T_new"], al["delta_L"],
                c["new_v"], c["old_v"], n_linear, 32, 4, "cpu", False,
            )
            assert "c_hat_per_layer" in rec["hook_stats"], "hook_stats must be persisted!"
            rec["record_key"] = f"dry:{mode}:{idx}"
            append_record_jsonl(dry_jsonl, rec)
        summ = summarize_jsonl(dry_jsonl, "dry_qwen", mode, {"tau": FROZEN_OSC_TAU_DEV_2B if mode == "osc" else lr_blob["tau"]})
        assert "leak_rate" in summ["overall"]["live_routed"]
        assert "recompute_rate" in summ["overall"]
        dry_jsonl.unlink(missing_ok=True)
    print("[DRY-RUN PASS] w13_fallback_runner.py verified both OSC and LR pipelines cleanly.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--detector", type=str, required=True, choices=["osc", "lr"])
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default=str(SCRIPT_DIR))
    parser.add_argument("--n-per-ood", type=int, default=25)
    parser.add_argument("--n-benign", type=int, default=80)
    parser.add_argument("--n-coupled", type=int, default=20)
    parser.add_argument("--serving-ctx", type=int, default=2048)
    parser.add_argument("--suites", type=str, default="both", choices=["both", "serving", "ood"])
    parser.add_argument("--resummarize-jsonl", type=str, default="", help="Offline 0-CU re-summarize an existing JSONL (including tau_plateau_audit).")
    parser.add_argument("--refresh-summary-jsonl", type=str, default="", help="Offline 0-CU refresh *_summary.json directly from existing JSONL without GPU.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--synthetic-dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.synthetic_dry_run:
        run_synthetic_dry_run(out_dir)
        return

    if args.refresh_summary_jsonl:
        jp = Path(args.refresh_summary_jsonl)
        summ_p = out_dir / (jp.stem.replace("_records", "") + "_summary.json")
        prev = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.exists() else {}
        frozen_cfg = prev.get("frozen_config") or {
            "detector": args.detector,
            "tau": FROZEN_OSC_TAU_DEV_2B if args.detector == "osc" else FROZEN_LR_TAU_DEV_9B,
        }
        model_name = prev.get("model") or args.model
        summ = summarize_jsonl(jp, model_name, args.detector, frozen_cfg)
        summ_p.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[REFRESH SUMMARY DONE] Wrote {summ_p}", flush=True)
        return

    if args.resummarize_jsonl:
        jp = Path(args.resummarize_jsonl)
        frozen_cfg = {
            "detector": args.detector,
            "tau": FROZEN_OSC_TAU_DEV_2B if args.detector == "osc" else FROZEN_LR_TAU_DEV_9B,
            "resummarized_offline": True,
        }
        summ = summarize_jsonl(jp, args.model, args.detector, frozen_cfg)
        out_p = out_dir / (jp.stem.replace("_records", "") + "_tau_plateau.json")
        out_p.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[RESUMMARIZE DONE] Wrote {out_p}", flush=True)
        return

    tag = args.tag or ("2b_osc" if args.detector == "osc" else "9b_lr")
    n_per_ood = 2 if args.smoke else args.n_per_ood
    n_benign = 4 if args.smoke else args.n_benign
    n_coupled = 2 if args.smoke else args.n_coupled

    lr_blob = None
    if args.detector == "lr":
        dev_file = out_dir / "w11_detector_v2_clean_9b_records.jsonl"
        lr_blob = freeze_lr_from_dev_split(dev_file, tau_lr=FROZEN_LR_TAU_DEV_9B)
        frozen_cfg = {"detector": "lr", "tau": lr_blob["tau"], "dev_source": lr_blob["source"], "n_dev": lr_blob["n_dev"]}
    else:
        frozen_cfg = {
            "detector": "osc",
            "formula": "prop1_analytical_risk_score (w12_prop1_perturbation_decay.py)",
            "tau_osc": FROZEN_OSC_TAU_DEV_2B,
            "dev_source": "w11_detector_v2_clean_2b_records.jsonl (split=='dev')",
        }

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

    jsonl_path = out_dir / f"w13_fallback_{tag}_records.jsonl"
    done = load_done_keys(jsonl_path)

    cases: List[Dict[str, Any]] = []
    if args.suites in ("both", "ood"):
        cases += build_g4_unseen_ood_cases(tok, n_linear, n_per_ood)
    if args.suites in ("both", "serving"):
        cases += build_w12_2_serving_traffic(tok, n_linear, n_benign, n_coupled, target_ctx_tokens=args.serving_ctx)

    print(
        f"[W13 Fallback Runner] model={args.model} | detector={args.detector} | "
        f"frozen_cfg={frozen_cfg} | total={len(cases)} | done={len(done)}",
        flush=True,
    )

    for c in cases:
        rk = f"{tag}:{c['suite']}:{c['regime']}:{c['idx']}"
        if rk in done:
            continue
        al = c["al"]
        rec = execute_live_routed_sample(
            model, base_model, tok, eos_ids, args.detector, lr_blob, FROZEN_OSC_TAU_DEV_2B,
            c["suite"], c["regime"], c["healable"],
            torch.tensor([al["ids_o_list"]], device=args.device),
            torch.tensor([al["ids_e_list"]], device=args.device),
            torch.tensor([al["q_ids_list"]], device=args.device),
            al["p0"], al["b_old"], al["b_new"], al["T_old"], al["T_new"], al["delta_L"],
            c["new_v"], c["old_v"], n_linear, d_k, w_conv, args.device, use_l2k,
        )
        rec["record_key"] = rk
        append_record_jsonl(jsonl_path, rec)
        done.add(rk)
        if len(done) % 20 == 0 or c["idx"] == 0:
            print(
                f"  [{rk}] T={rec['T_old']} | flag={rec['detector_flag']} rc={int(rec['live_recomputed'])} | "
                f"jl_em={int(rec['always_jlocal']['em'])} rt_em={int(rec['live_routed']['em'])} leak={int(rec['live_routed']['leak'])} | "
                f"rt_ms={rec['timing_ms']['live_routed_e2e_ms']:.1f} vs suf={rec['timing_ms']['always_suffix_e2e_ms']:.1f}",
                flush=True,
            )

    summ = summarize_jsonl(jsonl_path, args.model, args.detector, frozen_cfg)
    summ_path = out_dir / f"w13_fallback_{tag}_summary.json"
    summ_path.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Wrote {summ_path}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
