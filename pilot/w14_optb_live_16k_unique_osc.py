"""Option B Exp-5 Extension: Strictly-Unique-Entity, Diverse-Context Live 9B 16K OSC@0.40 Serving Benchmark.

Leaves historical `pilot/w13_fallback_runner.py` completely untouched while constructing
an independent, hash-verified 100%-unique-entity 16K serving workload (`n=100`: 40 benign_clean_fact,
40 benign_varlen_json, 20 coupled_twohop_fail) with:
  1. 100 strictly disjoint entity names (zero entity reuse across all 100 cases), 4 distinct templates
     per regime (12 structural templates total, including 4 distinct coupled two-hop templates), and
     per-case non-shared multi-domain ~16K filler contexts (measured pairwise 8-gram Jaccard overlap).
  2. Full persistence of `entity_name`, `template_id`, `prompt_old`, `prompt_new`, `query_text`, and
     `prompt_sha256 = sha256(prompt_old + "\\x00" + prompt_new)` in every JSONL record for independent
     disk-level hash recomputation.
  3. Live conditional GPU-path execution using frozen `OSC@tau` (`--tau-osc`, default `0.40`),
     recording BOTH `contains_new` and strict normalized exact match (`exact_match`), raw 8-token
     decoded strings, `old_leak`, `kl_vs_suffix`, `AUROC`, both rank-based and tie-aware `PR-AUC`,
     two-sided Clopper-Pearson 95% bounds, tau sensitivity plateau `{0.2, 0.4, 0.7}`, upper-order-statistic
     `p50/p95` latencies, and explicit `lg_old` precomputation scope.

Supports `--synthetic-dry-run` on CPU (`CUDA_VISIBLE_DEVICES=""`).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
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
import transformers as _tf
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
    snap_states,
)
from w10_p1_3_e4_e7_counterfact import apply_delta_rope_to_kv_cache  # noqa: E402
from w11_varlen_e10_e12_detector import (  # noqa: E402
    _align_contiguous_edit_span,
    load_done_keys,
    wilson_ci,
)
from w12_prop1_perturbation_decay import (  # noqa: E402
    _auroc_np,
    _pr_auc_np,
    clopper_pearson_lower,
)
from w13_fallback_runner import (  # noqa: E402
    FROZEN_OSC_TAU_DEV_2B,
    score_frozen_osc,
)


def _sync_dev(device: str) -> float:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _kl_logits(lg_ref: torch.Tensor, lg_cand: torch.Tensor) -> float:
    p = F.softmax(lg_ref.float(), dim=-1)
    lp = F.log_softmax(lg_ref.float(), dim=-1)
    lq = F.log_softmax(lg_cand.float(), dim=-1)
    return float((p * (lp - lq)).sum(dim=-1).mean().item())


def _upper_percentile(xs: List[float], pct: float) -> float:
    """Upper-order-statistic indexing matching legacy W13 and w12_c_posthoc_stats (`sorted[n//2]`, `sorted[min(n-1, int(0.95*n))]`)."""
    if not xs:
        return 0.0
    s = sorted(float(x) for x in xs)
    n = len(s)
    if pct == 50.0:
        idx = n // 2
    else:
        idx = min(n - 1, int((pct / 100.0) * n))
    return float(s[max(0, idx)])


def extract_features_on_gpu(
    s_old_b: Dict[str, List[Optional[torch.Tensor]]],
    delta: Dict[str, List[Optional[torch.Tensor]]],
    hs: Dict[str, Any],
    delta_L: int,
    force_no_rope: bool,
    lg_jl: torch.Tensor,
    lg_old_q: torch.Tensor,
) -> Dict[str, float]:
    rel_ds = [
        float(d_r.float().norm().item() / (r_o.float().norm().item() + 1e-6))
        for r_o, d_r in zip(s_old_b["rec"], delta["rec"])
        if r_o is not None and d_r is not None
    ]
    chat_all = torch.cat([x.reshape(-1) for x in hs["c_hat_cpu"]]) if hs["c_hat_cpu"] else torch.zeros(1)
    gam_all = torch.cat([x.reshape(-1) for x in hs["gamma_scalar"]]) if hs["gamma_scalar"] else torch.ones(1)
    p_jl = F.softmax(lg_jl, dim=-1)
    p_old = F.softmax(lg_old_q, dim=-1)
    top2_vals = torch.topk(lg_jl[0], 2).values
    old_top1_id = int(lg_old_q.argmax(dim=-1).item())
    jl_top1_id = int(lg_jl.argmax(dim=-1).item())
    return {
        "rel_delta_s_mean": sum(rel_ds) / max(1, len(rel_ds)),
        "rel_delta_s_max": max(rel_ds, default=0.0),
        "c_hat_mean": float(chat_all.float().mean().item()),
        "gamma_mean": float(gam_all.float().mean().item()),
        "abs_delta_L": float(abs(delta_L)),
        "force_no_rope": 1.0 if force_no_rope else 0.0,
        "margin_jl": float((top2_vals[0] - top2_vals[1]).item()),
        "entropy_jl": float(-(p_jl * (p_jl + 1e-12).log()).sum().item()),
        "shift_kl_old_vs_jl": float(F.kl_div((p_old + 1e-12).log(), p_jl, reduction="batchmean").item()),
        "top1_flipped_from_old": float(old_top1_id != jl_top1_id),
        "stale_top1_prob_under_jl": float(p_jl[0, old_top1_id].item()),
    }


# 120 strictly unique entity identifiers (zero entity reuse across all 100 cases)
UNIQUE_ENTITIES_120 = [
    "Atlas-01", "Boreal-02", "Cinder-03", "Delphic-04", "Echelon-05",
    "Fjord-06", "Galleon-07", "Helios-08", "Ionis-09", "Juniper-10",
    "Kestrel-11", "Linden-12", "Meridian-13", "Nimbus-14", "Orchid-15",
    "Paladin-16", "Quartz-17", "Redwood-18", "Solstice-19", "Thistle-20",
    "Umbra-21", "Vesper-22", "Warden-23", "Xenon-24", "Yarrow-25",
    "Zephyr-26", "Amber-27", "Basalt-28", "Cobalt-29", "Dune-30",
    "Ember-31", "Flint-32", "Granite-33", "Harbor-34", "Ivory-35",
    "Jasper-36", "Krypton-37", "Lantern-38", "Marble-39", "Nickel-40",
    "Onyx-41", "Pebble-42", "Quiver-43", "Ridge-44", "Slate-45",
    "Timber-46", "Ulster-47", "Velvet-48", "Willow-49", "Zircon-50",
    "Acacia-51", "Beryl-52", "Cedar-53", "Draco-54", "Estuary-55",
    "Falconer-56", "Glacier-57", "Hawthorn-58", "Indigo-59", "Javelin-60",
    "Kelp-61", "Laurel-62", "Monsoon-63", "Nautilus-64", "Obsidian-65",
    "Pinnacle-66", "Quarry-67", "Ravine-68", "Saffron-69", "Tundra-70",
    "Upland-71", "Vortex-72", "Wisteria-73", "Xylem-74", "Yucca-75",
    "Zenith-76", "Alder-77", "Birch-78", "Cypress-79", "Dahlia-80",
    "Elmsworth-81", "Fern-82", "Geyser-83", "Heather-84", "Ibis-85",
    "Jackal-86", "Kodiak-87", "Larch-88", "Magpie-89", "Nectar-90",
    "Osprey-91", "Plover-92", "Quail-93", "Rowan-94", "Sequoia-95",
    "Topaz-96", "Urchin-97", "Valerian-98", "Walnut-99", "Zinnia-100",
    "Agate-101", "Bracken-102", "Clover-103", "Dolomite-104", "Equinox-105",
    "Fennel-106", "Garnet-107", "Hemlock-108", "Iris-109", "Jetty-110",
    "Kyanite-111", "Lichen-112", "Malachite-113", "Nettle-114", "Opal-115",
    "Peridot-116", "Rhodonite-117", "Spruce-118", "Tamarack-119", "Zephyrus-120",
]

UNIQUE_JSON_TIERS = ["alpha", "beta", "gamma", "delta", "omega", "prime", "edge", "core"]
UNIQUE_JSON_MODES = [
    ("sync", "rolling-canary"),
    ("manual", "auto-failover"),
    ("batch", "stream-active"),
    ("cold", "hot-standby"),
    ("local", "multi-zone"),
    ("shadow", "blue-green"),
    ("static", "dynamic-mesh"),
    ("single", "quorum-write"),
    ("direct", "buffered-drain"),
    ("strict", "adaptive-burst"),
]

# Six distinct domain sentence banks (72 templates) combined with unique case/line IDs & vocab tokens
DOMAIN_SENTENCE_BANKS: Dict[str, List[str]] = {
    "telemetry": [
        "Telemetry packet {uid} measured {m1} ms jitter across relay {loc} during window {win}.",
        "Gateway probe {uid} verified ingress rate {m1} Mbps on fiber link {loc} ({win}).",
        "Router snapshot {uid} logged zero dropped frames on interface {loc} at epoch {win}.",
        "Switch fabric digest {uid} reported queue depth {m1} entries in rack {loc} ({win}).",
        "Optical transceiver {uid} held signal margin {m1} dB along corridor {loc} ({win}).",
        "Packet capture {uid} confirmed checksum parity across subnet {loc} during {win}.",
        "Beacon watchdog {uid} recorded latency floor {m1} us at repeater {loc} ({win}).",
        "Backbone monitor {uid} balanced egress traffic across ring {loc} in cycle {win}.",
        "Firewall summary {uid} audited {m1} transient sessions on border {loc} ({win}).",
        "Load balancer trace {uid} rotated health checks for pool {loc} during {win}.",
        "DNS resolver log {uid} served {m1} cached lookups from edge {loc} ({win}).",
        "Netflow exporter {uid} aggregated flow records across trunk {loc} at {win}.",
    ],
    "botany": [
        "Field transect {uid} cataloged {m1} lichen colonies along ridge {loc} in season {win}.",
        "Herbarium sheet {uid} documented spruce needle moisture {m1} pct near {loc} ({win}).",
        "Soil core {uid} measured organic carbon ratio {m1} g/kg in plot {loc} ({win}).",
        "Canopy sensor {uid} tracked leaf area index {m1} across watershed {loc} ({win}).",
        "Botanical survey {uid} counted {m1} flowering sedges beside marsh {loc} ({win}).",
        "Pollen trap {uid} captured birch grain density {m1} per slide at {loc} ({win}).",
        "Dendrology note {uid} recorded ring width {m1} mm on cedar stand {loc} ({win}).",
        "Greenhouse trial {uid} logged germination count {m1} for accession {loc} ({win}).",
        "Bryophyte census {uid} mapped moss coverage across boulder field {loc} ({win}).",
        "Phenology diary {uid} noted bud burst day {m1} in valley {loc} ({win}).",
        "Seedbank ledger {uid} verified viability rate {m1} pct for batch {loc} ({win}).",
        "Mycology sample {uid} isolated root symbionts from terrace {loc} during {win}.",
    ],
    "maritime": [
        "Harbor manifest {uid} cleared {m1} ballast tons at berth {loc} on tide {win}.",
        "Buoy station {uid} transmitted swell height {m1} dm off headland {loc} ({win}).",
        "Pilotage log {uid} tracked channel draft {m1} ft past shoal {loc} ({win}).",
        "Lighthouse keeper note {uid} verified visibility {m1} nm across strait {loc} ({win}).",
        "Drydock inspection {uid} measured hull plate thickness {m1} mm on pier {loc} ({win}).",
        "Tidal gauge {uid} logged salinity {m1} psu inside estuary {loc} ({win}).",
        "Vessel traffic report {uid} spaced anchorage intervals in basin {loc} ({win}).",
        "Sounding chart {uid} updated seabed contour {m1} m along channel {loc} ({win}).",
        "Crane telemetry {uid} hoisted {m1} intermodal containers on quay {loc} ({win}).",
        "Breakwater audit {uid} recorded wave pressure {m1} kPa at jetty {loc} ({win}).",
        "Navigation beacon {uid} synchronized flash cadence on reef {loc} ({win}).",
        "Port weather bulletin {uid} noted barometric trend {m1} hPa over bay {loc} ({win}).",
    ],
    "astronomy": [
        "Observatory plate {uid} measured seeing {m1} arcsec at dome {loc} on night {win}.",
        "Spectrograph frame {uid} calibrated thorium line {m1} nm in slit {loc} ({win}).",
        "Photometry run {uid} integrated flux count {m1} ADU toward field {loc} ({win}).",
        "Radio dish {uid} tracked continuum noise {m1} K along azimuth {loc} ({win}).",
        "Astrometry catalog {uid} refined proper motion {m1} mas in sector {loc} ({win}).",
        "Adaptive optics loop {uid} maintained strehl ratio {m1} pct at mount {loc} ({win}).",
        "Ephemeris check {uid} logged transit epoch across meridian {loc} ({win}).",
        "Bolometer array {uid} held cryostat stage {m1} mK inside vault {loc} ({win}).",
        "Coronagraph test {uid} attenuated stray glare on bench {loc} during {win}.",
        "All-sky camera {uid} noted cloud fraction {m1} pct over ridge {loc} ({win}).",
        "Interferometer baseline {uid} locked fringe phase along arm {loc} ({win}).",
        "Transit survey {uid} archived lightcurve cadence {m1} s for tile {loc} ({win}).",
    ],
    "metallurgy": [
        "Foundry heat {uid} recorded melt temperature {m1} C in furnace {loc} ({win}).",
        "Tensile coupon {uid} achieved yield stress {m1} MPa from billet {loc} ({win}).",
        "Annealing cycle {uid} held soak duration {m1} min in kiln {loc} ({win}).",
        "Metallograph slide {uid} measured grain size {m1} um across ingot {loc} ({win}).",
        "Hardness tester {uid} logged Rockwell value {m1} on flange {loc} ({win}).",
        "Ultrasonic scan {uid} cleared weld seam along plate {loc} during {win}.",
        "Spectrometer assay {uid} confirmed chromium fraction {m1} ppm in lot {loc} ({win}).",
        "Quench tank {uid} stabilized polymer bath {m1} C at line {loc} ({win}).",
        "Rolling mill pass {uid} reduced strip gauge to {m1} mm on stand {loc} ({win}).",
        "Fatigue rig {uid} completed {m1} load cycles on specimen {loc} ({win}).",
        "Crucible pour {uid} degassed alloy charge inside chamber {loc} ({win}).",
        "Corrosion chamber {uid} inspected salt spray panel from rack {loc} ({win}).",
    ],
    "archives": [
        "Archival folio {uid} indexed {m1} parchment leaves in vault {loc} ({win}).",
        "Microfilm reel {uid} verified contrast density {m1} on cabinet {loc} ({win}).",
        "Catalog drawer {uid} reconciled {m1} call slips for wing {loc} ({win}).",
        "Conservation bench {uid} stabilized iron-gall ink on ledger {loc} ({win}).",
        "Climate logger {uid} maintained relative humidity {m1} pct in stack {loc} ({win}).",
        "Map room inventory {uid} flattened {m1} linen charts in drawer {loc} ({win}).",
        "Binding workshop {uid} repaired linen spine for codex {loc} during {win}.",
        "Digitization cradle {uid} scanned {m1} dpi plates from box {loc} ({win}).",
        "Provenance file {uid} cross-checked accession stamp on shelf {loc} ({win}).",
        "Periodicals desk {uid} collated {m1} quarterly issues in aisle {loc} ({win}).",
        "Deacidification log {uid} recorded paper pH {m1} for folder {loc} ({win}).",
        "Reading room register {uid} verified box retrieval from bay {loc} ({win}).",
    ],
}

DOMAIN_NAMES = sorted(DOMAIN_SENTENCE_BANKS.keys())
LOC_PREFIXES = ["North", "South", "East", "West", "Upper", "Lower", "Inner", "Outer", "Mid", "Far"]
LOC_NOUNS = ["Sector", "Station", "Ward", "Block", "Zone", "Deck", "Unit", "Grid", "Span", "Node"]


def _normalize_exact(text: str) -> str:
    s = (text or "").strip().strip("`\"'.,;:!?()[]{}")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _strict_exact_match(text: str, target: str) -> bool:
    return _normalize_exact(text) == _normalize_exact(target)


def _pr_auc_tie_aware(y: List[int], s: List[float]) -> float:
    """Standard average precision with tie-grouped thresholds (matches sklearn average_precision_score)."""
    n_pos = sum(y)
    if n_pos == 0:
        return 0.0
    pairs = sorted(zip(s, y), key=lambda x: -x[0])
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            if pairs[j][1] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return float(ap)


def _make_case_specific_pad(tok: Any, target_tokens: int, case_seed: int, case_uid_base: int) -> str:
    """Build case-specific non-shared multi-domain filler text (~1 single encode call per segment)."""
    rng = random.Random(case_seed)
    # Pick 2 domains specific to this case and shuffle their 24 sentence templates
    d_idx = case_seed % len(DOMAIN_NAMES)
    d2_idx = (case_seed // len(DOMAIN_NAMES) + 1 + (case_seed % (len(DOMAIN_NAMES) - 1))) % len(DOMAIN_NAMES)
    if d2_idx == d_idx:
        d2_idx = (d_idx + 1) % len(DOMAIN_NAMES)
    templates = list(DOMAIN_SENTENCE_BANKS[DOMAIN_NAMES[d_idx]]) + list(DOMAIN_SENTENCE_BANKS[DOMAIN_NAMES[d2_idx]])
    rng.shuffle(templates)

    # Each sentence is ~17-20 tokens; oversample by ~1.35x and trim cleanly to exact target_tokens in ONE encode!
    est_sents = max(12, int(target_tokens / 14) + 16)
    parts: List[str] = []
    for step in range(est_sents):
        tpl = templates[step % len(templates)]
        uid = f"C{case_uid_base:03d}-{step:04d}-{rng.randint(100, 999)}"
        m1 = str(rng.randint(11, 989))
        loc = f"{rng.choice(LOC_PREFIXES)}-{rng.choice(LOC_NOUNS)}-{rng.randint(10, 99)}"
        win = f"W{rng.randint(100, 999)}-{chr(65 + rng.randint(0, 25))}"
        parts.append(tpl.format(uid=uid, m1=m1, loc=loc, win=win))

    raw_text = " ".join(parts)
    ids = tok.encode(raw_text, add_special_tokens=False)
    while len(ids) < target_tokens:
        extra = []
        base_s = len(parts)
        for k in range(32):
            tpl = templates[(base_s + k) % len(templates)]
            uid = f"C{case_uid_base:03d}-{base_s + k:04d}-{rng.randint(100, 999)}"
            m1 = str(rng.randint(11, 989))
            loc = f"{rng.choice(LOC_PREFIXES)}-{rng.choice(LOC_NOUNS)}-{rng.randint(10, 99)}"
            win = f"W{rng.randint(100, 999)}-{chr(65 + rng.randint(0, 25))}"
            extra.append(tpl.format(uid=uid, m1=m1, loc=loc, win=win))
        parts.extend(extra)
        ids = tok.encode(" ".join(parts), add_special_tokens=False)
    return tok.decode(ids[:target_tokens], skip_special_tokens=True)


def _compute_corpus_diversity(prompts: List[str]) -> Dict[str, Any]:
    """Compute exact pairwise word 8-gram Jaccard overlap across generated prompts."""
    ngram_sets: List[set[Tuple[str, ...]]] = []
    for p in prompts:
        words = p.split()
        # Sample sliding 8-grams across the entire prompt (stride 4 keeps O(N^2) under 0.5s for 100x16K)
        gs = {tuple(words[i : i + 8]) for i in range(0, max(1, len(words) - 8), 4)}
        ngram_sets.append(gs)
    n = len(ngram_sets)
    if n < 2:
        return {"n_pairs": 0, "mean_pairwise_8gram_jaccard": 0.0, "max_pairwise_8gram_jaccard": 0.0}
    tot = 0.0
    mx = 0.0
    pairs = 0
    for i in range(n):
        si = ngram_sets[i]
        for j in range(i + 1, n):
            sj = ngram_sets[j]
            inter = len(si & sj)
            union = len(si) + len(sj) - inter
            jac = inter / max(1, union)
            tot += jac
            if jac > mx:
                mx = jac
            pairs += 1
    return {
        "n_pairs": pairs,
        "mean_pairwise_8gram_jaccard": round(tot / max(1, pairs), 6),
        "max_pairwise_8gram_jaccard": round(mx, 6),
    }


def build_unique_16k_serving_workload(
    tok: Any,
    n_linear: int,
    n_clean_fact: int = 40,
    n_varlen_json: int = 40,
    n_coupled: int = 20,
    target_ctx_tokens: int = 16384,
    seed: int = 20261128,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate 100% unique-entity, diverse-context, multi-template 16K serving cases."""
    total_needed = n_clean_fact + n_varlen_json + n_coupled
    assert total_needed <= len(UNIQUE_ENTITIES_120), f"Need {total_needed} unique entities, pool has {len(UNIQUE_ENTITIES_120)}"

    rng = random.Random(seed)
    pre_tgt = target_ctx_tokens // 5
    suf_tgt = (target_ctx_tokens * 4) // 5

    cases: List[Dict[str, Any]] = []
    seen_hashes: set[str] = set()
    used_codes: set[str] = set()
    used_entities: set[str] = set()
    global_idx = 0

    def _fresh_code_pair() -> Tuple[str, str]:
        while True:
            a = str(rng.randint(1000, 4999))
            b = str(rng.randint(5000, 9999))
            if a[0] != b[0] and a not in used_codes and b not in used_codes:
                used_codes.add(a)
                used_codes.add(b)
                return a, b

    def _next_entity() -> str:
        nonlocal global_idx
        nm = UNIQUE_ENTITIES_120[global_idx]
        global_idx += 1
        assert nm not in used_entities
        used_entities.add(nm)
        return nm

    # 1. benign_clean_fact (n=40, 4 distinct templates x 10 each, 40 unique entities, unique backgrounds)
    cf_templates = [
        ("cf_vault", "Audit log: {nm} holds vault key {val}.", "Question: What is the assigned vault key for {nm}? Reply with 4 digits."),
        ("cf_pin", "Access ledger: {nm} authenticates with locker PIN {val}.", "Question: What is the locker PIN assigned to {nm}? Reply with 4 digits."),
        ("cf_badge", "Security memo: {nm} carries gate badge code {val}.", "Question: What is the gate badge code for {nm}? Reply with 4 digits."),
        ("cf_serial", "Registry entry: {nm} uses terminal serial {val}.", "Question: What is the terminal serial recorded for {nm}? Reply with 4 digits."),
    ]
    for cnt in range(n_clean_fact):
        nm = _next_entity()
        v_o, v_n = _fresh_code_pair()
        tpl_id, stmt_tpl, q_tpl = cf_templates[cnt % len(cf_templates)]
        pre = _make_case_specific_pad(tok, max(64, pre_tgt), case_seed=seed + global_idx * 101 + 1, case_uid_base=global_idx * 2)
        suf = _make_case_specific_pad(tok, max(128, suf_tgt), case_seed=seed + global_idx * 101 + 2, case_uid_base=global_idx * 2 + 1)
        q_s = q_tpl.format(nm=nm)
        u_o = f"{pre} {stmt_tpl.format(nm=nm, val=v_o)} {suf}\n\n{q_s}"
        u_e = f"{pre} {stmt_tpl.format(nm=nm, val=v_n)} {suf}\n\n{q_s}"
        h = hashlib.sha256((u_o + "\x00" + u_e).encode("utf-8")).hexdigest()
        assert h not in seen_hashes
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
        assert al is not None, f"Failed alignment for clean_fact idx={cnt}"
        seen_hashes.add(h)
        cases.append({
            "suite": "optb_unique_16k",
            "regime": "benign_clean_fact",
            "template_id": tpl_id,
            "entity_name": nm,
            "idx": cnt,
            "healable_by_suffix": True,
            "local_edit_expected_to_fail": False,
            "old_v": v_o,
            "new_v": v_n,
            "prompt_old": u_o,
            "prompt_new": u_e,
            "query_text": q_s,
            "prompt_sha256": h,
            "al": al,
        })

    # 2. benign_varlen_json (n=40, 4 distinct JSON schema templates x 10 each, 40 unique entities, unique backgrounds)
    js_templates = [
        ("js_deploy", "deploy_mode", 'Config: {{"owner": "{nm}", "tier": "{tier}", "deploy_mode": "{mode}"}}', "Question: What is {nm}'s updated deploy_mode in the config? Reply with only the string."),
        ("js_route", "routing_mode", 'ServiceSpec: {{"service": "{nm}", "ring": "{tier}", "routing_mode": "{mode}"}}', "Question: What is {nm}'s updated routing_mode in the ServiceSpec? Reply with only the string."),
        ("js_cache", "eviction_policy", 'CacheManifest: {{"node": "{nm}", "class": "{tier}", "eviction_policy": "{mode}"}}', "Question: What is {nm}'s updated eviction_policy in the CacheManifest? Reply with only the string."),
        ("js_cluster", "sync_profile", 'ClusterRecord: {{"unit": "{nm}", "zone": "{tier}", "sync_profile": "{mode}"}}', "Question: What is {nm}'s updated sync_profile in the ClusterRecord? Reply with only the string."),
    ]
    for cnt in range(n_varlen_json):
        nm = _next_entity()
        tpl_id, _fld, body_tpl, q_tpl = js_templates[cnt % len(js_templates)]
        t_old = UNIQUE_JSON_TIERS[cnt % len(UNIQUE_JSON_TIERS)]
        t_new = UNIQUE_JSON_TIERS[(cnt + 3) % len(UNIQUE_JSON_TIERS)] + f"-r{cnt + 1}"
        m_old, m_new = UNIQUE_JSON_MODES[cnt % len(UNIQUE_JSON_MODES)]
        m_old_u = f"{m_old}-{cnt + 10}"
        m_new_u = f"{m_new}-v{cnt + 10}"
        pre = _make_case_specific_pad(tok, max(64, pre_tgt), case_seed=seed + global_idx * 101 + 3, case_uid_base=global_idx * 2)
        suf = _make_case_specific_pad(tok, max(128, suf_tgt), case_seed=seed + global_idx * 101 + 4, case_uid_base=global_idx * 2 + 1)
        q_s = q_tpl.format(nm=nm)
        u_o = f"{pre}\n{body_tpl.format(nm=nm, tier=t_old, mode=m_old_u)}\n{suf}\n\n{q_s}"
        u_e = f"{pre}\n{body_tpl.format(nm=nm, tier=t_new, mode=m_new_u)}\n{suf}\n\n{q_s}"
        h = hashlib.sha256((u_o + "\x00" + u_e).encode("utf-8")).hexdigest()
        assert h not in seen_hashes
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
        assert al is not None, f"Failed alignment for varlen_json idx={cnt}"
        seen_hashes.add(h)
        cases.append({
            "suite": "optb_unique_16k",
            "regime": "benign_varlen_json",
            "template_id": tpl_id,
            "entity_name": nm,
            "idx": cnt,
            "healable_by_suffix": True,
            "local_edit_expected_to_fail": False,
            "old_v": m_old_u,
            "new_v": m_new_u,
            "prompt_old": u_o,
            "prompt_new": u_e,
            "query_text": q_s,
            "prompt_sha256": h,
            "al": al,
        })

    # 3. coupled_twohop_fail (n=20, 4 distinct two-hop structural templates x 5 each, 20 unique entities, unique backgrounds)
    cp_templates = [
        (
            "cp_routing",
            "Policy: {nm} routes via {hop}.",
            "Routing table: if {nm} uses {hop_o}, key is {co}; if {nm} uses {hop_n}, key is {cn}.",
            "Question: What is the final routing key for {nm}? Reply with 4 digits.",
            "ALPHA", "OMEGA",
        ),
        (
            "cp_clearance",
            "Assignment: {nm} operates in sector {hop}.",
            "Clearance matrix: if {nm} is in {hop_o}, clearance code is {co}; if {nm} is in {hop_n}, clearance code is {cn}.",
            "Question: What is the final clearance code for {nm}? Reply with 4 digits.",
            "NORTH", "SOUTH",
        ),
        (
            "cp_dispatch",
            "Manifest: {nm} ships through depot {hop}.",
            "Dispatch chart: when {nm} uses {hop_o}, bay code is {co}; when {nm} uses {hop_n}, bay code is {cn}.",
            "Question: What is the final bay code for {nm}? Reply with 4 digits.",
            "DOCKA", "DOCKB",
        ),
        (
            "cp_orbital",
            "Schedule: {nm} transmits on band {hop}.",
            "Telemetry table: for {nm} on {hop_o}, cipher key is {co}; for {nm} on {hop_n}, cipher key is {cn}.",
            "Question: What is the final cipher key for {nm}? Reply with 4 digits.",
            "BANDX", "BANDY",
        ),
    ]
    for cnt in range(n_coupled):
        nm = _next_entity()
        tpl_id, hop_tpl, tbl_tpl, q_tpl, pfx_o, pfx_n = cp_templates[cnt % len(cp_templates)]
        old_s = f"{pfx_o}-{cnt + 10}"
        new_s = f"{pfx_n}-{cnt + 10}"
        c_o, c_n = _fresh_code_pair()
        pre = _make_case_specific_pad(tok, max(64, pre_tgt), case_seed=seed + global_idx * 101 + 5, case_uid_base=global_idx * 2)
        suf = _make_case_specific_pad(tok, max(128, suf_tgt), case_seed=seed + global_idx * 101 + 6, case_uid_base=global_idx * 2 + 1)
        q_s = q_tpl.format(nm=nm)
        tbl = tbl_tpl.format(nm=nm, hop_o=old_s, hop_n=new_s, co=c_o, cn=c_n)
        u_o = f"{pre} {hop_tpl.format(nm=nm, hop=old_s)} {suf} {tbl}\n\n{q_s}"
        u_e = f"{pre} {hop_tpl.format(nm=nm, hop=new_s)} {suf} {tbl}\n\n{q_s}"
        h = hashlib.sha256((u_o + "\x00" + u_e).encode("utf-8")).hexdigest()
        assert h not in seen_hashes
        al = _align_contiguous_edit_span(tok, u_o, u_e, q_s, n_linear)
        assert al is not None, f"Failed alignment for coupled idx={cnt}"
        seen_hashes.add(h)
        cases.append({
            "suite": "optb_unique_16k",
            "regime": "coupled_twohop_fail",
            "template_id": tpl_id,
            "entity_name": nm,
            "idx": cnt,
            "healable_by_suffix": True,
            "local_edit_expected_to_fail": True,
            "old_v": c_o,
            "new_v": c_n,
            "prompt_old": u_o,
            "prompt_new": u_e,
            "query_text": q_s,
            "prompt_sha256": h,
            "al": al,
        })

    assert len(cases) == total_needed, f"Built {len(cases)} cases, expected {total_needed}"
    assert len({c["prompt_sha256"] for c in cases}) == len(cases), "Duplicate prompt SHA-256 detected!"
    assert len({c["entity_name"] for c in cases}) == len(cases), "Duplicate entity_name detected!"
    assert len({(c["old_v"], c["new_v"]) for c in cases}) == len(cases), "Duplicate (old_v, new_v) pair detected!"

    div = _compute_corpus_diversity([c["prompt_new"] for c in cases])
    tpl_counts: Dict[str, Dict[str, int]] = {}
    for c in cases:
        tpl_counts.setdefault(c["regime"], {})
        tpl_counts[c["regime"]][c["template_id"]] = tpl_counts[c["regime"]].get(c["template_id"], 0) + 1
    diversity_audit = {
        "n_cases": len(cases),
        "unique_entities": len({c["entity_name"] for c in cases}),
        "unique_prompt_hashes": len({c["prompt_sha256"] for c in cases}),
        "unique_old_new_pairs": len({(c["old_v"], c["new_v"]) for c in cases}),
        "template_counts_by_regime": tpl_counts,
        **div,
    }
    return cases, diversity_audit


def evaluate_live_unique_case(
    model: Any,
    base_model: Any,
    tok: Any,
    eos_ids: List[int],
    n_linear: int,
    d_k: int,
    w_conv: int,
    device: str,
    use_l2k: bool,
    case: Dict[str, Any],
    tau_osc: float = FROZEN_OSC_TAU_DEV_2B,
) -> Dict[str, Any]:
    """Run live conditional OSC@tau routing on GPU and persist full prompt/entity/strings/timing."""
    al = case["al"]
    ids_o = torch.tensor([al["ids_o_list"]], device=device, dtype=torch.long)
    ids_e = torch.tensor([al["ids_e_list"]], device=device, dtype=torch.long)
    ids_q = torch.tensor([al["q_ids_list"]], device=device, dtype=torch.long)
    p0, b_old, b_new = al["p0"], al["b_old"], al["b_new"]
    T_old, T_new, delta_L = al["T_old"], al["T_new"], al["delta_L"]
    old_v, new_v = case["old_v"], case["new_v"]

    with torch.no_grad():
        _, c_p0 = run_chunked_prefill(model, base_model, ids_o[:, :p0], 0, None, 2048, use_l2k)
        c_p0_cpu = move_cache_device(c_p0, "cpu")
        _, c_old_b = run_chunked_prefill(model, base_model, ids_o[:, p0:b_old], p0, clone_cache(c_p0), 2048, use_l2k)
        s_ob = snap_states(c_old_b)
        del c_p0

        with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as col:
            _, c_ctx_old = run_chunked_prefill(model, base_model, ids_o[:, b_old:T_old], b_old, c_old_b, 2048, use_l2k)
        hs = col.finalize(expected_suffix_len=T_old - b_old, device=device)
        c_ctx_old_cpu = move_cache_device(c_ctx_old, "cpu")
        # Precomputed outside timed GPU edit path (Turn-0 cached query logits)
        ev_old = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_ctx_old, T_old, ids_q, use_l2k, max_new_tokens=8)
        lg_old = ev_old["first_logits"].detach().float()
        del c_ctx_old

        # Standalone always_suffix baseline timing (GPU path)
        c_s0 = move_cache_device(c_p0_cpu, device)
        t_sf0 = _sync_dev(device)
        _, c_sf = run_chunked_prefill(model, base_model, ids_e[:, p0:T_new], p0, c_s0, 2048, use_l2k)
        ev_sf = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_sf, T_new, ids_q, use_l2k, max_new_tokens=8)
        always_suffix_ms = (_sync_dev(device) - t_sf0) * 1000.0
        lg_sf = ev_sf["first_logits"].detach().float()
        del c_sf

        # Live routed GPU path: O_local -> detector -> conditional suffix fallback
        c_p0_g = move_cache_device(c_p0_cpu, device)
        c_jl = move_cache_device(c_ctx_old_cpu, device)
        t_j0 = _sync_dev(device)
        _, c_nb = run_chunked_prefill(model, base_model, ids_e[:, p0:b_new], p0, c_p0_g, 2048, use_l2k)
        delta = compute_state_delta_fp64(s_ob, snap_states(c_nb))
        new_kv = extract_kv_slice(c_nb, p0, b_new)
        apply_jlocal_scalar_update(c_jl, delta, hs["gamma_scalar"], b_old, T_old, w_conv, use_decay=True)
        apply_delta_rope_to_kv_cache(c_jl, new_kv, p0, b_old, b_new, T_old, delta_L, base_model, apply_delta_rope=True)
        ev_jl = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_jl, T_new, ids_q, use_l2k, max_new_tokens=8)
        olocal_ms = (_sync_dev(device) - t_j0) * 1000.0
        lg_jl = ev_jl["first_logits"].detach().float()
        del c_jl

        t_d0 = _sync_dev(device)
        feats = extract_features_on_gpu(s_ob, delta, hs, delta_L, False, lg_jl, lg_old)
        score, flagged, osc_raw = score_frozen_osc(feats, tau_osc)
        detector_ms = (_sync_dev(device) - t_d0) * 1000.0

        fallback_ms = 0.0
        if flagged:
            t_fb0 = _sync_dev(device)
            _, c_fb = run_chunked_prefill(model, base_model, ids_e[:, b_new:T_new], b_new, c_nb, 2048, use_l2k)
            ev_rt = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_fb, T_new, ids_q, use_l2k, max_new_tokens=8)
            fallback_ms = (_sync_dev(device) - t_fb0) * 1000.0
            lg_rt = ev_rt["first_logits"].detach().float()
            del c_fb
        else:
            ev_rt = ev_jl
            lg_rt = lg_jl
            del c_nb

        del c_p0_cpu, c_ctx_old_cpu, hs

    jl_txt = str(ev_jl["text"])
    rt_txt = str(ev_rt["text"])
    sf_txt = str(ev_sf["text"])

    jl_cn = bool(new_v in jl_txt)
    rt_cn = bool(new_v in rt_txt)
    sf_cn = bool(new_v in sf_txt)

    jl_em = _strict_exact_match(jl_txt, new_v)
    rt_em = _strict_exact_match(rt_txt, new_v)
    sf_em = _strict_exact_match(sf_txt, new_v)

    jl_leak = bool((old_v in jl_txt) and not jl_cn)
    rt_leak = bool((old_v in rt_txt) and not rt_cn)
    sf_leak = bool((old_v in sf_txt) and not sf_cn)

    is_fail = int((not jl_cn) or jl_leak)
    routed_e2e_ms = float(olocal_ms + detector_ms + fallback_ms)

    return {
        "suite": case["suite"],
        "regime": case["regime"],
        "template_id": case["template_id"],
        "entity_name": case["entity_name"],
        "idx": int(case["idx"]),
        "healable_by_suffix": bool(case["healable_by_suffix"]),
        "local_edit_expected_to_fail": bool(case["local_edit_expected_to_fail"]),
        "prompt_sha256": case["prompt_sha256"],
        "prompt_old": case["prompt_old"],
        "prompt_new": case["prompt_new"],
        "query_text": case["query_text"],
        "T_old": int(T_old),
        "T_new": int(T_new),
        "delta_L": int(delta_L),
        "old_v": old_v,
        "new_v": new_v,
        "features": feats,
        "osc_score": float(osc_raw),
        "detector_score": float(score),
        "flagged": int(flagged),
        "is_failure_jlocal": int(is_fail),
        "jlocal_text": jl_txt,
        "routed_text": rt_txt,
        "suffix_text": sf_txt,
        "jlocal_contains_new": jl_cn,
        "routed_contains_new": rt_cn,
        "suffix_contains_new": sf_cn,
        "jlocal_exact_match": jl_em,
        "routed_exact_match": rt_em,
        "suffix_exact_match": sf_em,
        "jlocal_old_leak": jl_leak,
        "routed_old_leak": rt_leak,
        "suffix_old_leak": sf_leak,
        "kl_jlocal_vs_suffix": float(_kl_logits(lg_sf, lg_jl)),
        "kl_routed_vs_suffix": float(_kl_logits(lg_sf, lg_rt)),
        "timing_ms": {
            "olocal_ms": round(olocal_ms, 3),
            "detector_ms": round(detector_ms, 3),
            "fallback_ms": round(fallback_ms, 3),
            "live_routed_e2e_ms": round(routed_e2e_ms, 3),
            "always_suffix_e2e_ms": round(always_suffix_ms, 3),
        },
    }


def summarize_unique_records(rows: List[Dict[str, Any]], tau_osc: float) -> Dict[str, Any]:
    n = len(rows)
    assert n > 0
    unique_hashes = len({r["prompt_sha256"] for r in rows})
    unique_entities = len({r["entity_name"] for r in rows})
    # Verify persisted prompts re-hash to prompt_sha256 on 100% of records
    verified_hashes = sum(
        1
        for r in rows
        if hashlib.sha256((r["prompt_old"] + "\x00" + r["prompt_new"]).encode("utf-8")).hexdigest() == r["prompt_sha256"]
    )
    assert verified_hashes == n, f"Prompt SHA-256 verification failed: {verified_hashes}/{n}"

    y = [int(r["is_failure_jlocal"]) for r in rows]
    s = [float(r["detector_score"]) for r in rows]
    fl = [int(r["flagged"]) for r in rows]
    tp = sum(1 for yi, fi in zip(y, fl) if yi == 1 and fi == 1)
    fp = sum(1 for yi, fi in zip(y, fl) if yi == 0 and fi == 1)
    fn = sum(1 for yi, fi in zip(y, fl) if yi == 1 and fi == 0)
    tn = sum(1 for yi, fi in zip(y, fl) if yi == 0 and fi == 0)
    n_pos = tp + fn
    n_neg = tn + fp
    tp_lb = float(clopper_pearson_lower(tp, n_pos, 0.05)) if n_pos else 0.0
    tn_lb = float(clopper_pearson_lower(tn, n_neg, 0.05)) if n_neg else 0.0
    auroc = float(_auroc_np(np.array(y), np.array(s))) if (n_pos and n_neg) else 0.5
    pr_rank = float(_pr_auc_np(np.array(y), np.array(s))) if n_pos else 0.0
    pr_tie = _pr_auc_tie_aware(y, s)

    rt_ms = [r["timing_ms"]["live_routed_e2e_ms"] for r in rows]
    sf_ms = [r["timing_ms"]["always_suffix_e2e_ms"] for r in rows]
    ol_ms = [r["timing_ms"]["olocal_ms"] for r in rows]
    rt_mean = sum(rt_ms) / n
    sf_mean = sum(sf_ms) / n

    tau_sweep: Dict[str, Any] = {}
    for tv in (0.2, 0.4, 0.7):
        ft = [1 if r["osc_score"] >= tv else 0 for r in rows]
        t_tp = sum(1 for yi, fi in zip(y, ft) if yi == 1 and fi == 1)
        t_fp = sum(1 for yi, fi in zip(y, ft) if yi == 0 and fi == 1)
        t_cn = sum(1 for r, fi in zip(rows, ft) if (r["suffix_contains_new"] if fi else r["jlocal_contains_new"]))
        tau_sweep[f"tau_{tv:.1f}"] = {
            "tp": t_tp, "fp": t_fp, "recompute_rate": round(sum(ft) / n, 4),
            "contains_new_rate": round(t_cn / n, 4),
        }

    by_regime: Dict[str, Any] = {}
    for reg in sorted({r["regime"] for r in rows}):
        sub = [r for r in rows if r["regime"] == reg]
        nr = len(sub)
        r_rt = [r["timing_ms"]["live_routed_e2e_ms"] for r in sub]
        r_sf = [r["timing_ms"]["always_suffix_e2e_ms"] for r in sub]
        by_regime[reg] = {
            "n": nr,
            "unique_prompts": len({r["prompt_sha256"] for r in sub}),
            "unique_entities": len({r["entity_name"] for r in sub}),
            "unique_templates": len({r["template_id"] for r in sub}),
            "flagged": sum(int(r["flagged"]) for r in sub),
            "jlocal_contains_new": round(sum(1 for r in sub if r["jlocal_contains_new"]) / nr, 4),
            "routed_contains_new": round(sum(1 for r in sub if r["routed_contains_new"]) / nr, 4),
            "suffix_contains_new": round(sum(1 for r in sub if r["suffix_contains_new"]) / nr, 4),
            "routed_exact_match": round(sum(1 for r in sub if r["routed_exact_match"]) / nr, 4),
            "suffix_exact_match": round(sum(1 for r in sub if r["suffix_exact_match"]) / nr, 4),
            "routed_mean_ms": round(sum(r_rt) / nr, 3),
            "suffix_mean_ms": round(sum(r_sf) / nr, 3),
            "speedup_mean": round((sum(r_sf) / nr) / max(1e-6, sum(r_rt) / nr), 3),
        }

    by_template: Dict[str, Any] = {}
    for tid in sorted({r["template_id"] for r in rows}):
        sub = [r for r in rows if r["template_id"] == tid]
        nt = len(sub)
        by_template[tid] = {
            "n": nt,
            "regime": sub[0]["regime"],
            "flagged": sum(int(r["flagged"]) for r in sub),
            "is_failure_jlocal": sum(int(r["is_failure_jlocal"]) for r in sub),
            "jlocal_contains_new": round(sum(1 for r in sub if r["jlocal_contains_new"]) / nt, 4),
            "routed_contains_new": round(sum(1 for r in sub if r["routed_contains_new"]) / nt, 4),
            "suffix_contains_new": round(sum(1 for r in sub if r["suffix_contains_new"]) / nt, 4),
            "routed_exact_match": round(sum(1 for r in sub if r["routed_exact_match"]) / nt, 4),
        }

    return {
        "n": n,
        "unique_prompt_hashes": unique_hashes,
        "unique_entities": unique_entities,
        "prompt_hashes_recomputed_verified": verified_hashes,
        "all_prompts_unique": bool(unique_hashes == n),
        "all_entities_unique": bool(unique_entities == n),
        "tau_osc": float(tau_osc),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_pos": n_pos, "n_neg": n_neg,
        "tpr": round(tp / max(1, n_pos), 4),
        "tnr": round(tn / max(1, n_neg), 4),
        "ci_convention": "two_sided_95pct_clopper_pearson_lower_bound (alpha/2=0.025)",
        "tpr_cp95_lb": round(tp_lb, 5),
        "tnr_cp95_lb": round(tn_lb, 5),
        "benign_fp_rate": round(fp / max(1, n_neg), 4),
        "benign_fp_wilson95": [round(x, 4) for x in wilson_ci(fp, n_neg)],
        "recompute_rate": round(sum(fl) / n, 4),
        "auroc": round(auroc, 6),
        "pr_auc_rank": round(pr_rank, 6),
        "pr_auc_tie_aware": round(pr_tie, 6),
        "joint_auc_cp95_lb": round(max(0.0, tp_lb + tn_lb - 1.0), 5),
        "always_jlocal_contains_new": round(sum(1 for r in rows if r["jlocal_contains_new"]) / n, 4),
        "live_routed_contains_new": round(sum(1 for r in rows if r["routed_contains_new"]) / n, 4),
        "always_suffix_contains_new": round(sum(1 for r in rows if r["suffix_contains_new"]) / n, 4),
        "always_jlocal_exact_match": round(sum(1 for r in rows if r["jlocal_exact_match"]) / n, 4),
        "live_routed_exact_match": round(sum(1 for r in rows if r["routed_exact_match"]) / n, 4),
        "always_suffix_exact_match": round(sum(1 for r in rows if r["suffix_exact_match"]) / n, 4),
        "always_jlocal_old_leak": round(sum(1 for r in rows if r["jlocal_old_leak"]) / n, 4),
        "live_routed_old_leak": round(sum(1 for r in rows if r["routed_old_leak"]) / n, 4),
        "always_suffix_old_leak": round(sum(1 for r in rows if r["suffix_old_leak"]) / n, 4),
        "mean_kl_routed_vs_suffix": round(sum(r["kl_routed_vs_suffix"] for r in rows) / n, 8),
        "percentile_convention": "upper_order_statistic (sorted[n//2], sorted[min(n-1, int(0.95*n))])",
        "timing_gpu_path_ms": {
            "olocal_mean_ms": round(sum(ol_ms) / n, 3),
            "live_routed_mean_ms": round(rt_mean, 3),
            "live_routed_p50_ms": round(_upper_percentile(rt_ms, 50.0), 3),
            "live_routed_p95_ms": round(_upper_percentile(rt_ms, 95.0), 3),
            "always_suffix_mean_ms": round(sf_mean, 3),
            "always_suffix_p50_ms": round(_upper_percentile(sf_ms, 50.0), 3),
            "always_suffix_p95_ms": round(_upper_percentile(sf_ms, 95.0), 3),
            "speedup_mean_vs_always_suffix": round(sf_mean / max(1e-6, rt_mean), 3),
        },
        "tau_sensitivity": tau_sweep,
        "by_regime": by_regime,
        "by_template": by_template,
    }


def run_synthetic_dry_run(out_dir: Path) -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
    if hasattr(qm.torch_chunk_gated_delta_rule, "__wrapped__"):
        qm.torch_chunk_gated_delta_rule = qm.torch_chunk_gated_delta_rule.__wrapped__
    if hasattr(qm.torch_recurrent_gated_delta_rule, "__wrapped__"):
        qm.torch_recurrent_gated_delta_rule = qm.torch_recurrent_gated_delta_rule.__wrapped__

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True, local_files_only=True)
    full_cases, div_audit = build_unique_16k_serving_workload(
        tok, n_linear=24, n_clean_fact=40, n_varlen_json=40, n_coupled=20, target_ctx_tokens=16384
    )
    assert len(full_cases) == 100
    assert div_audit["unique_entities"] == 100 and div_audit["unique_prompt_hashes"] == 100
    assert div_audit["mean_pairwise_8gram_jaccard"] < 0.05, f"High prompt overlap: {div_audit}"

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
    tiny_cases, _ = build_unique_16k_serving_workload(
        tok, n_linear=3, n_clean_fact=1, n_varlen_json=1, n_coupled=1, target_ctx_tokens=320
    )
    rows = [evaluate_live_unique_case(model, base_model, tok, eos_ids, 3, 32, 4, "cpu", False, c, tau_osc=0.40) for c in tiny_cases]
    sm = summarize_unique_records(rows, 0.40)
    assert sm["all_prompts_unique"] is True and sm["all_entities_unique"] is True and sm["prompt_hashes_recomputed_verified"] == 3
    print(f"[DRY-RUN PASS] w14_optb_live_16k_unique_osc.py 100/100 unique entities & diverse 16K prompts OK: {div_audit}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tag", type=str, default="9b_l4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default=str(SCRIPT_DIR))
    parser.add_argument("--ctx-len", type=int, default=16384)
    parser.add_argument("--n-clean-fact", type=int, default=40)
    parser.add_argument("--n-varlen-json", type=int, default=40)
    parser.add_argument("--n-coupled", type=int, default=20)
    parser.add_argument("--tau-osc", type=float, default=FROZEN_OSC_TAU_DEV_2B)
    parser.add_argument("--seed", type=int, default=20261128)
    parser.add_argument("--clean-out", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--synthetic-dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.synthetic_dry_run:
        run_synthetic_dry_run(out_dir)
        return

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    t_cfg = getattr(cfg, "text_config", cfg)
    n_linear = count_linear_layers(t_cfg)
    d_k = int(getattr(t_cfg, "linear_key_head_dim", 128))
    w_conv = int(getattr(t_cfg, "linear_conv_kernel_dim", 4))

    ncf, nvj, ncp = (2, 2, 4) if args.smoke else (args.n_clean_fact, args.n_varlen_json, args.n_coupled)
    cases, diversity_audit = build_unique_16k_serving_workload(
        tok, n_linear=n_linear, n_clean_fact=ncf, n_varlen_json=nvj, n_coupled=ncp,
        target_ctx_tokens=args.ctx_len, seed=args.seed,
    )
    print(f"[Unique16K] Built {len(cases)} unique-entity cases; diversity={json.dumps(diversity_audit)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True, local_files_only=args.local_files_only,
    ).eval()
    base_model = getattr(model.model, "language_model", model.model)
    eos_ids = get_eos_ids(model, tok)
    use_l2k = has_logits_to_keep_arg(model)

    jsonl_p = out_dir / f"w14_optb_live_16k_unique_{args.tag}_records.jsonl"
    if args.clean_out:
        jsonl_p.unlink(missing_ok=True)
    done = load_done_keys(jsonl_p)

    for ci, case in enumerate(cases):
        rk = f"{args.tag}:unique16k:{case['regime']}:{case['idx']}"
        if rk in done:
            continue
        rec = evaluate_live_unique_case(
            model, base_model, tok, eos_ids, n_linear, d_k, w_conv,
            args.device, use_l2k, case, tau_osc=float(args.tau_osc),
        )
        rec["record_key"] = rk
        append_record_jsonl(jsonl_p, rec)
        done.add(rk)
        if (ci + 1) % 10 == 0 or (ci + 1) == len(cases):
            print(
                f"[Unique16K] {ci + 1}/{len(cases)} {case['regime']}/{case['template_id']} "
                f"ent={case['entity_name']} flag={rec['flagged']} fail={rec['is_failure_jlocal']} "
                f"cn={rec['routed_contains_new']} em={rec['routed_exact_match']} rt={rec['timing_ms']['live_routed_e2e_ms']}ms",
                flush=True,
            )

    rows = [json.loads(l) for l in jsonl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    summ = {
        "runner": "pilot/w14_optb_live_16k_unique_osc.py",
        "model": args.model,
        "tag": args.tag,
        "device_name": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else "cpu",
        "os_name": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": _tf.__version__,
        "seed": int(args.seed),
        "ctx_len_target": int(args.ctx_len),
        "detector": "osc",
        "tau_osc": float(args.tau_osc),
        "decode_max_new_tokens": 8,
        "decode_eos_can_stop_early": True,
        "corpus_diversity_audit": diversity_audit,
        "timing_scope": {
            "included_in_gpu_path": ["olocal_block_forward_and_update", "detector_feature_and_score", "conditional_suffix_fallback"],
            "excluded_from_gpu_path": [
                "prefix_h2d",
                "ctx_cache_h2d_or_clone",
                "precomputed_suffix_decay_hooks",
                "precomputed_lg_old_query_logits",
            ],
            "serving_assumption_note": "Precomputed lg_old assumes Turn-0 cached query logits are available (re-querying after edit); unseen post-edit queries require an extra forward pass on c_ctx_old.",
        },
        "summary": summarize_unique_records(rows, float(args.tau_osc)),
    }
    summ_p = out_dir / f"w14_optb_live_16k_unique_{args.tag}_summary.json"
    summ_p.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Wrote {summ_p}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
