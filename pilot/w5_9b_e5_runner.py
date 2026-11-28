#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wave 5 — Qwen3.5-9B E5 Staged Long-Context Runner (4K n=200 / 16K n=50->200)
=============================================================================
Standalone, single-file, resumable E5 experiment runner designed for Colab L4
(23.66 GB VRAM, <=20 min per colab exec call) and local 0-CU smoke verification.

Key Capabilities & Guardrails (per Wave 5 Dispatch #50):
1. Modes (`--mode o1` vs `--mode hybrid`):
   - `--mode o1`:
     * Runs BOTH O(1) transport operators in the same run:
       - `jlocal`: scalar `transport-decay` (gamma = exp(G_{b->T})) in fp64 (Decision B)
       - `jlocal_iso`: isotropic decay `transport-iso-decay`
         (gamma_iso = exp(G_{b->T} - B_{b->T} / d_k)) in fp64 (Decision B)
         (also exposed under alias `iso_decay` in per-sample and summary outputs)
     * Runs mandatory oracles & isolations:
       - `oneshot_oracle`: full unsplit prefill on edited context `ids_e`
       - `split_oracle` / `recompute`: shared prefix [0, p0) + edited block [p0, b_deep)
         + recomputed suffix [b_deep, T_ctx)
       - `jfull`: Decision A direct `copy_` overwrite at `b_deep` + patched KV + suffix recompute
         (verifies G1a <= 1e-5 vs `split_oracle` and tracks G1b vs `oneshot_oracle`)
       - `stale`: gamma = 1 (unattenuated recurrent state delta) + patched KV
       - `state_only`: scalar `transport-decay` state update + unpatched `KV_old`
     * Strictly O(1) suffix memory footprint:
       Inside the GDN forward hook (`torch_chunk_gated_delta_rule` AND
       `torch_recurrent_gated_delta_rule`), `--mode o1` NEVER stores full
       `(T, H_k, d_k)` key tensors! It accumulates only 1D per-head fp64 scalars
       `(G_suf, B_suf, L_suf, tok_count)` (< 25 KiB total across all 24 layers).
     * Single-active-cache GPU schedule:
       Offloads intermediate prefix/block snapshots (`c_pre`, `c_old_b`) to CPU RAM
       during setup and evaluates `jlocal`, `jlocal_iso`, `stale`, `state_only`
       in-place on `c_ctx_old` with exact state/KV restoration (`restore_ctx_cache_in_place`,
       verified bitwise identical to `clone_cache` on Sample 0), keeping at most 1.0x
       full `T_ctx` `DynamicCache` in GPU VRAM at any instant.
   - `--mode hybrid`:
     * Captures `[b_deep, T_ctx)` normalized keys (`eps=1e-6`, offloaded to CPU `float32`
       per chunk) ONLY when `--mode hybrid` is active.
     * Implements exact Wave 4 `transport-hybrid-gated` (`J-local+`):
       Switches head-wise between scalar `transport-decay` and selective WY
       `transport-householder` iff `(c_hat >= 0.05) and (gamma_scalar >= 1e-4)`.
       STRICTLY NEVER mixes `iso_decay` inside `hybrid_gated`!
     * Logs `hybrid_edit_ms`, `hybrid_active_head_frac`, and
       `complexity_tag="O(active_frac*(T-b)*d_k^2)_NOT_O1"` strictly separated
       from O(1) latency/speedup metrics (`T*/G6`).

2. Deterministic 200-Sample E5 Bank & Depth Cycling:
   - Reuses the verified 3-template (`tpl_id` in {0, 1, 2}), `digit4` disjoint
     `(old, new)` + 3-distractor pool for `["Alice", "Bob", "Carol", "Dave"]`,
     `SYSTEM_PROMPT`, `len(pos_o) == len(pos_e) == 4`, `tok.encode(val)[0]`,
     `assert_oracle_first_token`, explicit `position_ids`, Decision A `copy_`,
     Decision B `fp64`, and `--block-horizon deep` (`b_deep = min(T_ctx, p1 + 3*(N_linear-1))`,
     +51 on 2B `n_linear=18`, +69 on 9B `n_linear=24`).
   - Deterministic depth cycling across `n=200`: `depth = depths[sample_index % len(depths)]`
     (`0.25, 0.50, 0.75` -> 67/67/66 across 200 samples; `sample_index=2` tests `d=0.75`
     immediately in any `n=50` probe).

3. Resumable Batching (`--start-idx`, `--end-idx`, `--resume`, `--out-jsonl`, `--out-summary`):
   - Every completed sample is immediately appended to `--out-jsonl` with `flush()` + `os.fsync()`.
   - `--out-summary` always aggregates over all deduplicated records in `--out-jsonl`
     matching the active `--mode`, so multi-batch runs produce a single complete summary JSON.
"""

import os
import sys

# Must be set before importing torch to prevent 16K CUDA allocator fragmentation on L4 (23.66 GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import copy
import gc
import inspect
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================================
# Constants & Templates (Cleaned 3-Template E2/E5 Design)
# ============================================================================

SYSTEM_PROMPT = (
    "You are a helpful assistant. Reply with only the exact requested value, "
    "no reasoning. Do not use markdown or code blocks."
)

NAMES_POOL = ["Alice", "Bob", "Carol", "Dave"]

FILLERS = [
    "Meanwhile, standard atmospheric pressure at sea level remains close to 1013 hectopascals under calm weather conditions.",
    "In classical mechanics, the angular momentum of an isolated rigid system is conserved in the absence of external torque.",
    "Oceanic thermohaline circulation is driven primarily by global gradients in water temperature and salinity.",
    "Photosynthetic organisms convert incident solar radiation into chemical energy stored within organic carbohydrate molecules.",
    "The crystalline structure of quartz consists of a continuous framework of silicon-oxygen tetrahedra.",
    "Seismic body waves propagate through the interior of the Earth at velocities determined by local rock density and elastic moduli.",
    "Continental glaciers advance and retreat in response to multi-decadal variations in regional snowfall and summer ablation.",
    "Estuarine ecosystems support high biological productivity due to nutrient influx from both riverine drainage and tidal mixing.",
    "Tropospheric weather patterns are strongly modulated by the meandering Rossby wave train in the upper-level jet stream.",
    "In thermodynamics, the entropy of a closed macroscopic system never decreases during an irreversible spontaneous process.",
    "Metamorphic foliation develops when platy mineral grains align perpendicular to the direction of maximum compressive stress.",
    "In vacuum, electromagnetic radiation propagates at a constant phase velocity of approximately 299792458 meters per second.",
]

FILLER_TEXT_JOINED = " ".join(FILLERS)


def format_fact(tpl_id: int, subject: str, val: str) -> str:
    """Cleaned 3-template fact sentences (consistent with w2/w3/w4)."""
    if tpl_id == 0:
        return f"Record state: the security code for {subject} is {val}."
    elif tpl_id == 1:
        return f"Database entry: {subject} is assigned access key {val}."
    elif tpl_id == 2:
        return f"System log: {subject} currently holds token {val}."
    else:
        raise ValueError(f"Unknown tpl_id={tpl_id}")


def format_query(tpl_id: int, subject: str) -> str:
    """Cleaned 3-template query prompts (consistent with w2/w3/w4)."""
    if tpl_id == 0:
        return f"Question: What is the exact security code for {subject}? Answer with only the value."
    elif tpl_id == 1:
        return f"Question: What is the exact access key assigned to {subject}? Answer with only the value."
    elif tpl_id == 2:
        return f"Question: What is the exact token held by {subject}? Answer with only the value."
    else:
        raise ValueError(f"Unknown tpl_id={tpl_id}")


# ============================================================================
# Deterministic 200-Sample Bank Generator (Zero Collisions with FILLERS)
# ============================================================================

def build_digit4_pool() -> Dict[str, List[str]]:
    """
    Build deterministic pool of 4-digit codes (1000..9999) grouped by leading digit '1'..'9',
    strictly excluding any 4-digit string that occurs as a substring in FILLERS
    (e.g. '1013', '2997', '9979', '9792', '7924', '9245', '2458').
    """
    by_first_digit: Dict[str, List[str]] = {str(d): [] for d in range(1, 10)}
    for code_int in range(1000, 10000):
        code_str = str(code_int)
        if code_str in FILLER_TEXT_JOINED:
            continue
        # Also ensure all 4 digits are not identical (avoid trivial '1111' repeats)
        if len(set(code_str)) < 3:
            continue
        by_first_digit[code_str[0]].append(code_str)
    return by_first_digit


def generate_e5_sample_bank(
    n_samples: int = 200,
    depths: Optional[List[float]] = None,
    seed: int = 202605,
) -> List[Dict[str, Any]]:
    """
    Generate `n_samples` (default 200) deterministic E5 cases:
    - Cycles `target` across `["Alice", "Bob", "Carol", "Dave"]`
    - Cycles `tpl_id` across `[0, 1, 2]` (shifted by `s_idx // 4` so all (target, tpl_id)
      pairs are balanced across the bank)
    - Cycles `depth` across `depths` (`[0.25, 0.50, 0.75]` -> 67 / 67 / 66 across 200 samples;
      `s_idx=0 -> 0.25`, `s_idx=1 -> 0.50`, `s_idx=2 -> 0.75` so any `n=50` slice tests `d=0.75`
      starting at `s_idx=2`!)
    - Sample 0 uses the canonical calibration tuple `("Alice", "2846", "6193", tpl_id=0)`
      for direct continuity with Wave 3 / 9-cell calibration.
    - Every sample picks 5 disjoint 4-digit codes (`old`, `new`, plus 3 distractor values)
      with strictly distinct leading digits in `'1'..'9'` and zero substring overlap with `FILLERS`.
    """
    if depths is None or len(depths) == 0:
        depths = [0.25, 0.50, 0.75]

    rng = random.Random(seed)
    pool_by_d = build_digit4_pool()
    first_digits = [str(d) for d in range(1, 10)]

    used_tuples = set()
    samples: List[Dict[str, Any]] = []

    for s_idx in range(n_samples):
        depth = float(depths[s_idx % len(depths)])
        if s_idx == 0:
            target = "Alice"
            tpl_id = 0
            old_val = "2846"
            new_val = "6193"
            dist_vals = ["5921", "7384", "8412"]
            used_tuples.add((target, tpl_id, old_val, new_val))
        else:
            target = NAMES_POOL[s_idx % len(NAMES_POOL)]
            tpl_id = (s_idx // len(NAMES_POOL) + s_idx) % 3
            while True:
                chosen_firsts = rng.sample(first_digits, 5)
                vals = [rng.choice(pool_by_d[fd]) for fd in chosen_firsts]
                old_val, new_val = vals[0], vals[1]
                dist_vals = vals[2:]
                key = (target, tpl_id, old_val, new_val)
                if key not in used_tuples and len(set(vals)) == 5:
                    used_tuples.add(key)
                    break

        other_names = [nm for nm in NAMES_POOL if nm != target]
        distractors = [
            {"subject": other_names[i], "val": dist_vals[i]}
            for i in range(3)
        ]

        samples.append({
            "sample_index": s_idx,
            "target": target,
            "tpl_id": int(tpl_id),
            "depth": depth,
            "old": old_val,
            "new": new_val,
            "distractors": distractors,
            "filler_shift": s_idx % len(FILLERS),
        })

    return samples


# ============================================================================
# Prompt Construction & Token Span Verification
# ============================================================================

def apply_chat_no_think(tok, messages: List[Dict[str, str]]) -> str:
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def find_value_token_span(
    text: str,
    val_str: str,
    occurrence_char_Hint: int,
    offsets: List[Tuple[int, int]],
) -> List[int]:
    """Locate exact token indices covering `val_str` at `occurrence_char_Hint`."""
    c0 = text.find(val_str, max(0, occurrence_char_Hint - 10))
    if c0 < 0:
        raise ValueError(f"Could not find {val_str!r} near char {occurrence_char_Hint}")
    c1 = c0 + len(val_str)
    pos = [i for i, (s, e) in enumerate(offsets) if e > c0 and s < c1 and e > s]
    if not pos:
        raise ValueError(f"Empty token span for {val_str!r} at [{c0}, {c1})")
    return pos


def build_e5_prompt(
    tok,
    sample: Dict[str, Any],
    target_length: int,
    n_linear: int,
    block_horizon: str = "deep",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Build exact token-aligned `(ids_o, ids_e, ids_q)` for an E5 sample at `target_length`
    and `sample['depth']`.
    Verifies:
      1. `len(ids_o) == len(ids_e)`
      2. `len(pos_o) == len(pos_e) == 4` (for 4-digit codes)
      3. `ids_o[:p0] == ids_e[:p0]` and `ids_o[p1:] == ids_e[p1:]`
      4. `tok.encode(old)[0]` and `tok.encode(new)[0]` match the first token of `pos_o`/`pos_e`
         and are strictly distinct (`f_o != f_e`).
    """
    tpl_id = int(sample["tpl_id"])
    target = sample["target"]
    old_val = sample["old"]
    new_val = sample["new"]
    depth = float(sample["depth"])
    filler_shift = int(sample.get("filler_shift", sample["sample_index"]))
    distractors = sample["distractors"]

    target_fact_o = format_fact(tpl_id, target, old_val)
    target_fact_e = format_fact(tpl_id, target, new_val)

    dist_sents = [
        format_fact(tpl_id, d["subject"], d["val"])
        for d in distractors
    ]
    query_str = format_query(tpl_id, target)

    # Calibrate filler token lengths to hit `target_length` with edit at `depth * target_length`
    filler_tok_lens = [
        len(tok.encode(s + " ", add_special_tokens=False)) for s in FILLERS
    ]
    avg_filler_toks = sum(filler_tok_lens) / len(filler_tok_lens)

    # Reserve ~55 tokens for system chat header + 3 distractors
    usable_budget = max(128, target_length - 65)
    pre_tok_budget = max(24, int(usable_budget * depth) - 20)
    suf_tok_budget = max(64, int(usable_budget * (1.0 - depth)) - 20)

    n_pre_sents = max(2, int(round(pre_tok_budget / avg_filler_toks)))
    n_suf_sents = max(4, int(round(suf_tok_budget / avg_filler_toks)))

    pre_parts = [dist_sents[0]]
    for i in range(n_pre_sents):
        pre_parts.append(FILLERS[(filler_shift + i) % len(FILLERS)])
    pre_parts.append(dist_sents[1])
    prefix_body = " ".join(pre_parts)

    suf_parts = [dist_sents[2]]
    for i in range(n_suf_sents):
        suf_parts.append(FILLERS[(filler_shift + n_pre_sents + i) % len(FILLERS)])
    suffix_body = " ".join(suf_parts)

    user_o = f"{prefix_body} {target_fact_o} {suffix_body}\n\n{query_str}"
    user_e = f"{prefix_body} {target_fact_e} {suffix_body}\n\n{query_str}"

    full_o = apply_chat_no_think(
        tok,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_o}],
    )
    full_e = apply_chat_no_think(
        tok,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_e}],
    )

    q_char_o = full_o.rfind(query_str)
    q_char_e = full_e.rfind(query_str)
    if q_char_o < 0 or q_char_e < 0:
        raise RuntimeError("Failed to locate query_str in formatted chat prompt")

    ctx_str_o = full_o[:q_char_o]
    ctx_str_e = full_e[:q_char_e]
    q_str = full_o[q_char_o:]

    enc_o = tok(ctx_str_o, return_offsets_mapping=True, add_special_tokens=False)
    enc_e = tok(ctx_str_e, return_offsets_mapping=True, add_special_tokens=False)
    enc_q = tok(q_str, add_special_tokens=False)

    ids_o_list = enc_o["input_ids"]
    ids_e_list = enc_e["input_ids"]
    ids_q_list = enc_q["input_ids"]

    if len(ids_o_list) != len(ids_e_list):
        raise RuntimeError(
            f"Sample {sample['sample_index']}: token length mismatch "
            f"len(ids_o)={len(ids_o_list)} != len(ids_e)={len(ids_e_list)}"
        )

    # Locate the exact occurrence of old_val / new_val inside target_fact_o / target_fact_e
    fact_char_o = ctx_str_o.find(target_fact_o)
    fact_char_e = ctx_str_e.find(target_fact_e)
    val_offset_in_fact_o = target_fact_o.rfind(old_val)
    val_offset_in_fact_e = target_fact_e.rfind(new_val)

    pos_o = find_value_token_span(
        ctx_str_o, old_val, fact_char_o + val_offset_in_fact_o, enc_o["offset_mapping"]
    )
    pos_e = find_value_token_span(
        ctx_str_e, new_val, fact_char_e + val_offset_in_fact_e, enc_e["offset_mapping"]
    )

    if pos_o != pos_e or len(pos_o) != 4:
        raise RuntimeError(
            f"Sample {sample['sample_index']}: span mismatch pos_o={pos_o}, pos_e={pos_e}"
        )

    p0, p1 = pos_o[0], pos_o[-1] + 1
    if ids_o_list[:p0] != ids_e_list[:p0] or ids_o_list[p1:] != ids_e_list[p1:]:
        raise RuntimeError(
            f"Sample {sample['sample_index']}: tokens outside [{p0}, {p1}) differ between old and new!"
        )

    f_o = tok.encode(old_val, add_special_tokens=False)[0]
    f_e = tok.encode(new_val, add_special_tokens=False)[0]
    if f_o == f_e or ids_o_list[p0] != f_o or ids_e_list[p0] != f_e:
        raise RuntimeError(
            f"Sample {sample['sample_index']}: first-token verification failed "
            f"(f_o={f_o}, f_e={f_e}, ids_o[p0]={ids_o_list[p0]}, ids_e[p0]={ids_e_list[p0]})"
        )

    T_ctx = len(ids_o_list)
    if block_horizon == "deep":
        b_end = min(T_ctx, p1 + 3 * (n_linear - 1))
    else:
        b_end = min(T_ctx, p1 + 3)

    ids_o_t = torch.tensor([ids_o_list], dtype=torch.long, device=device)
    ids_e_t = torch.tensor([ids_e_list], dtype=torch.long, device=device)
    ids_q_t = torch.tensor([ids_q_list], dtype=torch.long, device=device)

    return {
        "ids_o": ids_o_t,
        "ids_e": ids_e_t,
        "ids_q": ids_q_t,
        "p0": int(p0),
        "p1": int(p1),
        "b_end": int(b_end),
        "b_local": int(min(T_ctx, p1 + 3)),
        "b_deep": int(min(T_ctx, p1 + 3 * (n_linear - 1))),
        "T_ctx": int(T_ctx),
        "q_len": int(len(ids_q_list)),
        "actual_depth_ratio": float(p0 / max(1, T_ctx)),
        "suffix_len": int(T_ctx - b_end),
        "f_o": int(f_o),
        "f_e": int(f_e),
    }


# ============================================================================
# Model Helpers & Explicit Position IDs
# ============================================================================

def count_linear_layers(t_cfg) -> int:
    layer_types = getattr(t_cfg, "layer_types", None)
    if layer_types is not None:
        return sum(1 for lt in layer_types if "linear" in str(lt))
    n_layers = getattr(t_cfg, "num_hidden_layers", 24)
    interval = getattr(t_cfg, "full_attention_interval", 4)
    return n_layers - (n_layers // interval)


def get_eos_ids(model, tok) -> List[int]:
    eos: List[int] = []
    if getattr(tok, "eos_token_id", None) is not None:
        eos.append(int(tok.eos_token_id))
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(gen_eos, int):
        eos.append(gen_eos)
    elif isinstance(gen_eos, (list, tuple)):
        eos.extend(int(x) for x in gen_eos if x is not None)
    for s in ["<|im_end|>", "<|endoftext|>"]:
        tid = tok.convert_tokens_to_ids(s)
        if isinstance(tid, int) and tid >= 0:
            eos.append(tid)
    return list(dict.fromkeys(eos))


def make_position_ids(pos_start: int, seq_len: int, device, base_model) -> torch.Tensor:
    pos_1d = torch.arange(
        pos_start, pos_start + seq_len, device=device, dtype=torch.long
    ).unsqueeze(0)
    if hasattr(base_model, "rotary_emb") and "mrope" in type(base_model.rotary_emb).__name__.lower():
        return pos_1d.unsqueeze(0).expand(3, 1, seq_len).contiguous()
    return pos_1d


def has_logits_to_keep_arg(model) -> bool:
    try:
        sig = inspect.signature(model.forward)
        return "logits_to_keep" in sig.parameters
    except Exception:
        return False


# ============================================================================
# Cache Manipulation & Single-Active-Cache Memory Schedule
# ============================================================================

def _layer_get_state_tensor(layer, attr_name: str) -> Optional[torch.Tensor]:
    d = getattr(layer, attr_name, None)
    if isinstance(d, dict) and 0 in d and torch.is_tensor(d[0]):
        return d[0]
    if torch.is_tensor(d):
        return d
    return None


def clone_cache(cache):
    if cache is None:
        return None
    new_cache = copy.copy(cache)
    new_layers = []
    for layer in cache.layers:
        nl = copy.copy(layer)
        if getattr(layer, "keys", None) is not None and torch.is_tensor(layer.keys):
            nl.keys = layer.keys.detach().clone()
        if getattr(layer, "values", None) is not None and torch.is_tensor(layer.values):
            nl.values = layer.values.detach().clone()
        if hasattr(layer, "conv_states") and isinstance(layer.conv_states, dict):
            nl.conv_states = {
                k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in layer.conv_states.items()
            }
        if hasattr(layer, "recurrent_states") and isinstance(layer.recurrent_states, dict):
            nl.recurrent_states = {
                k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in layer.recurrent_states.items()
            }
        for attr in (
            "has_previous_state",
            "is_conv_states_initialized",
            "is_recurrent_states_initialized",
            "conv_kernel_size",
        ):
            if hasattr(layer, attr) and isinstance(getattr(layer, attr), dict):
                setattr(nl, attr, dict(getattr(layer, attr)))
        new_layers.append(nl)
    new_cache.layers = new_layers
    return new_cache


def move_cache_device(cache, target_device: str):
    """Move all tensors in `cache` to `target_device` (e.g., 'cpu' or 'cuda:0')."""
    if cache is None:
        return None
    new_cache = copy.copy(cache)
    new_layers = []
    for layer in cache.layers:
        nl = copy.copy(layer)
        if getattr(layer, "keys", None) is not None and torch.is_tensor(layer.keys):
            nl.keys = layer.keys.detach().to(target_device, copy=True)
        if getattr(layer, "values", None) is not None and torch.is_tensor(layer.values):
            nl.values = layer.values.detach().to(target_device, copy=True)
        if hasattr(layer, "conv_states") and isinstance(layer.conv_states, dict):
            nl.conv_states = {
                k: (v.detach().to(target_device, copy=True) if torch.is_tensor(v) else v)
                for k, v in layer.conv_states.items()
            }
        if hasattr(layer, "recurrent_states") and isinstance(layer.recurrent_states, dict):
            nl.recurrent_states = {
                k: (v.detach().to(target_device, copy=True) if torch.is_tensor(v) else v)
                for k, v in layer.recurrent_states.items()
            }
        for attr in (
            "has_previous_state",
            "is_conv_states_initialized",
            "is_recurrent_states_initialized",
            "conv_kernel_size",
        ):
            if hasattr(layer, attr) and isinstance(getattr(layer, attr), dict):
                setattr(nl, attr, dict(getattr(layer, attr)))
        new_layers.append(nl)
    new_cache.layers = new_layers
    return new_cache


def snap_states(cache) -> Dict[str, List[Optional[torch.Tensor]]]:
    """Snapshot only linear-attention conv_states & recurrent_states (~12 MB on 9B)."""
    conv_list: List[Optional[torch.Tensor]] = []
    rec_list: List[Optional[torch.Tensor]] = []
    for layer in cache.layers:
        c_t = _layer_get_state_tensor(layer, "conv_states")
        r_t = _layer_get_state_tensor(layer, "recurrent_states")
        conv_list.append(c_t.detach().clone() if c_t is not None else None)
        rec_list.append(r_t.detach().clone() if r_t is not None else None)
    return {"conv": conv_list, "rec": rec_list}


def overwrite_states(cache, snap: Dict[str, List[Optional[torch.Tensor]]]):
    """Decision A: direct `copy_` overwrite of linear-attention states in-place."""
    for i, t in enumerate(snap["conv"]):
        if t is not None:
            dst = _layer_get_state_tensor(cache.layers[i], "conv_states")
            if dst is not None:
                dst.copy_(t.to(dtype=dst.dtype, device=dst.device))
    for i, t in enumerate(snap["rec"]):
        if t is not None:
            dst = _layer_get_state_tensor(cache.layers[i], "recurrent_states")
            if dst is not None:
                dst.copy_(t.to(dtype=dst.dtype, device=dst.device))


def extract_kv_slice(cache, start: int, end: int) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Extract full-attention KV slice `[start, end)` (~2.3 MB for 69 tokens on 9B)."""
    out: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
    for layer in cache.layers:
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        if torch.is_tensor(k) and torch.is_tensor(v) and k.numel() > 0:
            out.append((k[:, :, start:end, :].detach().clone(), v[:, :, start:end, :].detach().clone()))
        else:
            out.append(None)
    return out


def patch_kv_slice(
    cache,
    kv_slice: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    start: int,
    end: int,
):
    """Patch full-attention KV slice `[start, end)` in-place via `.copy_()`."""
    for i, item in enumerate(kv_slice):
        if item is not None:
            layer = cache.layers[i]
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if torch.is_tensor(k) and torch.is_tensor(v) and k.numel() > 0:
                k_sub, v_sub = item
                k[:, :, start:end, :].copy_(k_sub.to(dtype=k.dtype, device=k.device))
                v[:, :, start:end, :].copy_(v_sub.to(dtype=v.dtype, device=v.device))


def truncate_cache_kv(cache, target_len: int):
    """Truncate full-attention `keys` / `values` back to `target_len` tokens."""
    for layer in cache.layers:
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        if torch.is_tensor(k) and k.numel() > 0 and k.shape[-2] > target_len:
            layer.keys = k[:, :, :target_len, :].contiguous()
            layer.values = v[:, :, :target_len, :].contiguous()
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = target_len


def restore_ctx_cache_in_place(
    cache,
    T_ctx: int,
    orig_T_states: Dict[str, List[Optional[torch.Tensor]]],
    orig_blk_kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    p0: int,
    b_end: int,
):
    """
    Restore `cache` back to its exact post-context `T_ctx` state in-place without
    cloning the full `T_ctx` KV cache!
    1. Truncates any appended query/greedy tokens above `T_ctx`.
    2. Restores `conv_states` and `recurrent_states` at `T_ctx` via `.copy_()`.
    3. Restores the `[p0, b_end)` slice of `keys` and `values` via `.copy_()`.
    """
    truncate_cache_kv(cache, T_ctx)
    overwrite_states(cache, orig_T_states)
    patch_kv_slice(cache, orig_blk_kv, p0, b_end)


def assert_cache_bitwise_equal(cache_a, cache_b):
    """Verify two `DynamicCache` instances are bitwise identical across all states and KV."""
    for i, (la, lb) in enumerate(zip(cache_a.layers, cache_b.layers)):
        ca = _layer_get_state_tensor(la, "conv_states")
        cb = _layer_get_state_tensor(lb, "conv_states")
        if ca is not None or cb is not None:
            assert ca is not None and cb is not None
            if not torch.equal(ca, cb):
                diff = (ca.float() - cb.float()).abs().max().item()
                raise AssertionError(f"conv_states[{i}] mismatch max_diff={diff}")
        ra = _layer_get_state_tensor(la, "recurrent_states")
        rb = _layer_get_state_tensor(lb, "recurrent_states")
        if ra is not None or rb is not None:
            assert ra is not None and rb is not None
            if not torch.equal(ra, rb):
                diff = (ra.float() - rb.float()).abs().max().item()
                raise AssertionError(f"recurrent_states[{i}] mismatch max_diff={diff}")
        ka, kb = getattr(la, "keys", None), getattr(lb, "keys", None)
        if (torch.is_tensor(ka) and ka.numel() > 0) or (torch.is_tensor(kb) and kb.numel() > 0):
            assert torch.is_tensor(ka) and torch.is_tensor(kb)
            if ka.shape != kb.shape or not torch.equal(ka, kb):
                diff = (ka.float() - kb.float()).abs().max().item() if ka.shape == kb.shape else float("inf")
                raise AssertionError(f"keys[{i}] mismatch shape={ka.shape} vs {kb.shape}, diff={diff}")
        va, vb = getattr(la, "values", None), getattr(lb, "values", None)
        if (torch.is_tensor(va) and va.numel() > 0) or (torch.is_tensor(vb) and vb.numel() > 0):
            assert torch.is_tensor(va) and torch.is_tensor(vb)
            if va.shape != vb.shape or not torch.equal(va, vb):
                diff = (va.float() - vb.float()).abs().max().item() if va.shape == vb.shape else float("inf")
                raise AssertionError(f"values[{i}] mismatch shape={va.shape} vs {vb.shape}, diff={diff}")


# ============================================================================
# Forward Pass & In-Place Evaluation Helpers
# ============================================================================

@torch.no_grad()
def run_chunked_prefill(
    model,
    base_model,
    input_ids: torch.Tensor,
    pos_start: int = 0,
    cache=None,
    max_chunk: int = 2048,
    use_logits_to_keep: bool = True,
):
    """
    Advance `cache` through `input_ids` in chunks of at most `max_chunk` tokens.
    Slices `logits[:, -1:, :]` immediately and frees full chunk logits to avoid
    allocating `(1, chunk_len, 248320)` (~1 GB) across chunks.
    """
    L = input_ids.shape[1]
    assert L > 0, "run_chunked_prefill requires L > 0"
    last_logits = None
    for s in range(0, L, max_chunk):
        e = min(L, s + max_chunk)
        p_ids = make_position_ids(pos_start + s, e - s, input_ids.device, base_model)
        kwargs = {
            "input_ids": input_ids[:, s:e],
            "position_ids": p_ids,
            "use_cache": True,
            "past_key_values": cache,
        }
        if use_logits_to_keep:
            kwargs["logits_to_keep"] = 1
        out = model(**kwargs)
        cache = out.past_key_values
        if e == L:
            last_logits = out.logits[:, -1:, :].detach().clone()
        del out
    return last_logits, cache


@torch.no_grad()
def eval_from_ctx_cache_in_place(
    model,
    base_model,
    tok,
    eos_ids: List[int],
    ctx_cache,
    T_ctx: int,
    ids_q: torch.Tensor,
    use_logits_to_keep: bool = True,
    max_new_tokens: int = 8,
) -> Dict[str, Any]:
    """
    Evaluate `ids_q` and greedy generation IN-PLACE on `ctx_cache` (which is at length `T_ctx`).
    Caller then restores `ctx_cache` back to `T_ctx` via `restore_ctx_cache_in_place`.
    Zero extra full-context `DynamicCache` clones are allocated on GPU!
    """
    q_len = ids_q.shape[1]
    q_pos = make_position_ids(T_ctx, q_len, ids_q.device, base_model)
    kwargs = {
        "input_ids": ids_q,
        "position_ids": q_pos,
        "use_cache": True,
        "past_key_values": ctx_cache,
    }
    if use_logits_to_keep:
        kwargs["logits_to_keep"] = 1
    out = model(**kwargs)
    first_logits = out.logits[:, -1, :].detach().float().clone()
    q_cache = out.past_key_values
    del out

    cur_pos = T_ctx + q_len
    nxt = first_logits.argmax(dim=-1, keepdim=True)
    gen_ids: List[int] = [int(nxt.item())]

    for _ in range(max_new_tokens - 1):
        if gen_ids[-1] in eos_ids:
            break
        step_pos = make_position_ids(cur_pos, 1, ids_q.device, base_model)
        step_kwargs = {
            "input_ids": nxt,
            "position_ids": step_pos,
            "use_cache": True,
            "past_key_values": q_cache,
        }
        if use_logits_to_keep:
            step_kwargs["logits_to_keep"] = 1
        out = model(**step_kwargs)
        step_logits = out.logits[:, -1, :]
        q_cache = out.past_key_values
        nxt = step_logits.argmax(dim=-1, keepdim=True)
        del out
        cur_pos += 1
        tid = int(nxt.item())
        if tid in eos_ids:
            break
        gen_ids.append(tid)

    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return {
        "first_logits": first_logits,
        "gen_ids": gen_ids,
        "text": text,
    }


def score_condition(
    res: Dict[str, Any],
    new_val: str,
    old_val: str,
    f_e: int,
    f_o: int,
    ref_logits: torch.Tensor,
    ans_ref_logits: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    lg = res["first_logits"].float()
    ref = ref_logits.float()

    max_abs_diff = float((lg - ref).abs().max().item())
    lp = F.log_softmax(lg, dim=-1)
    lq = F.log_softmax(ref, dim=-1)
    kl_div = float(F.kl_div(lp, lq.exp(), reduction="batchmean").item())

    top1_id = int(lg.argmax(dim=-1).item())
    ref_top1_id = int(ref.argmax(dim=-1).item())

    margin_new_vs_old = float((lg[0, f_e] - lg[0, f_o]).item())
    txt = res["text"]

    res_out: Dict[str, Any] = {
        "text": txt,
        "contains_new": bool(new_val in txt),
        "contains_old": bool(old_val in txt),
        "top1_eq_oracle": bool(top1_id == ref_top1_id),
        "first_tok_is_new": bool(top1_id == f_e),
        "first_tok_is_old": bool(top1_id == f_o),
        "margin_new_vs_old": margin_new_vs_old,
        "kl_vs_oneshot": kl_div,
        "max_logit_diff_vs_oneshot": max_abs_diff,
    }

    if ans_ref_logits is not None:
        ans_ref = ans_ref_logits.float()
        ans_lp = F.log_softmax(lg, dim=-1)
        ans_lq = F.log_softmax(ans_ref, dim=-1)
        res_out["answer_kl_vs_oneshot"] = float(
            F.kl_div(ans_lp, ans_lq.exp(), reduction="batchmean").item()
        )
        res_out["answer_logit_diff_vs_oneshot"] = float(
            (lg - ans_ref).abs().max().item()
        )
    return res_out


# ============================================================================
# Dual-Hook GDN Suffix Collector (`--mode o1` Scalar-Only vs `--mode hybrid`)
# ============================================================================

class GDNSuffixHookCollector:
    """
    Hooks BOTH `qm.torch_chunk_gated_delta_rule` and `qm.torch_recurrent_gated_delta_rule`
    across all `n_linear` GDN layers while running suffix `[b_deep, T_ctx)`.

    - When `scalar_only=True` (`--mode o1`):
      Accumulates ONLY 1D per-head fp64 scalars:
        * `G_suf[li]` = sum_{t=b}^{T-1} g_t               (shape `(H,)`)
        * `B_suf[li]` = sum_{t=b}^{T-1} beta_t            (shape `(H,)`)
        * `L_suf[li]` = sum_{t=b}^{T-1} (-log(1 - beta_t)) (shape `(H,)`)
        * `tok_count[li]` = number of suffix tokens captured
      Stores ZERO `(T, H, d_k)` key tensors! Total memory across 24 layers is < 25 KiB.

    - When `scalar_only=False` (`--mode hybrid`):
      Also normalizes `k` on GPU (`F.normalize(k.float(), p=2.0, dim=-1, eps=1e-6)`)
      and offloads `(k_norm, g, beta)` chunks in `float32` to CPU RAM for selective WY.
    """

    def __init__(self, n_linear: int, d_k: int, scalar_only: bool = True):
        self.n_linear = int(n_linear)
        self.d_k = int(d_k)
        self.scalar_only = bool(scalar_only)

        self.G_suf: List[Optional[torch.Tensor]] = [None for _ in range(self.n_linear)]
        self.B_suf: List[Optional[torch.Tensor]] = [None for _ in range(self.n_linear)]
        self.L_suf: List[Optional[torch.Tensor]] = [None for _ in range(self.n_linear)]
        self.tok_count: List[int] = [0 for _ in range(self.n_linear)]

        self.k_chunks: List[List[torch.Tensor]] = [[] for _ in range(self.n_linear)]
        self.g_chunks: List[List[torch.Tensor]] = [[] for _ in range(self.n_linear)]
        self.b_chunks: List[List[torch.Tensor]] = [[] for _ in range(self.n_linear)]

        self._call_idx = 0
        self._orig_chunk_fn = None
        self._orig_recurrent_fn = None

    def _record_step(
        self,
        key: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        use_qk_l2norm_in_kernel: bool,
    ):
        li = self._call_idx % self.n_linear
        self._call_idx += 1

        g_f64 = g.detach().to(torch.float64)
        b_f64 = beta.detach().to(torch.float64)

        g_sum = g_f64.sum(dim=(0, 1)).cpu()
        b_sum = b_f64.sum(dim=(0, 1)).cpu()
        l_sum = (-torch.log((1.0 - b_f64).clamp(min=1e-12))).sum(dim=(0, 1)).cpu()

        if self.G_suf[li] is None:
            self.G_suf[li] = g_sum
            self.B_suf[li] = b_sum
            self.L_suf[li] = l_sum
        else:
            self.G_suf[li] = self.G_suf[li] + g_sum
            self.B_suf[li] = self.B_suf[li] + b_sum
            self.L_suf[li] = self.L_suf[li] + l_sum

        self.tok_count[li] += int(g.shape[1])

        if not self.scalar_only:
            k_f = key.detach().float()
            if use_qk_l2norm_in_kernel:
                k_f = F.normalize(k_f, p=2.0, dim=-1, eps=1e-6)
            self.k_chunks[li].append(k_f.cpu())
            self.g_chunks[li].append(g.detach().float().cpu())
            self.b_chunks[li].append(beta.detach().float().cpu())

    def __enter__(self):
        self._orig_chunk_fn = qm.torch_chunk_gated_delta_rule
        self._orig_recurrent_fn = getattr(qm, "torch_recurrent_gated_delta_rule", None)
        collector = self

        def hooked_chunk(query, key, value, g, beta, *args, **kwargs):
            use_l2 = bool(kwargs.get("use_qk_l2norm_in_kernel", True))
            collector._record_step(key, g, beta, use_l2)
            return collector._orig_chunk_fn(query, key, value, g, beta, *args, **kwargs)

        qm.torch_chunk_gated_delta_rule = hooked_chunk

        if self._orig_recurrent_fn is not None:

            def hooked_recurrent(query, key, value, g, beta, *args, **kwargs):
                use_l2 = bool(kwargs.get("use_qk_l2norm_in_kernel", True))
                collector._record_step(key, g, beta, use_l2)
                return collector._orig_recurrent_fn(
                    query, key, value, g, beta, *args, **kwargs
                )

            qm.torch_recurrent_gated_delta_rule = hooked_recurrent

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._orig_chunk_fn is not None:
            qm.torch_chunk_gated_delta_rule = self._orig_chunk_fn
        if self._orig_recurrent_fn is not None:
            qm.torch_recurrent_gated_delta_rule = self._orig_recurrent_fn

    def finalize(self, expected_suffix_len: int, device: str) -> Dict[str, Any]:
        for li in range(self.n_linear):
            if self.tok_count[li] != expected_suffix_len:
                raise RuntimeError(
                    f"GDNSuffixHookCollector layer {li}: captured tok_count={self.tok_count[li]} "
                    f"!= expected_suffix_len={expected_suffix_len}"
                )

        G_suf_gpu: List[torch.Tensor] = []
        B_suf_gpu: List[torch.Tensor] = []
        L_suf_gpu: List[torch.Tensor] = []
        gamma_scalar_gpu: List[torch.Tensor] = []
        gamma_iso_gpu: List[torch.Tensor] = []
        c_hat_cpu: List[torch.Tensor] = []

        for li in range(self.n_linear):
            G_l = self.G_suf[li].to(device=device, dtype=torch.float64)
            B_l = self.B_suf[li].to(device=device, dtype=torch.float64)
            L_l = self.L_suf[li].to(device=device, dtype=torch.float64)

            gam_s = torch.exp(G_l).view(1, -1, 1, 1)
            gam_iso = torch.exp(G_l - (B_l / float(self.d_k))).view(1, -1, 1, 1)

            G_suf_gpu.append(G_l)
            B_suf_gpu.append(B_l)
            L_suf_gpu.append(L_l)
            gamma_scalar_gpu.append(gam_s)
            gamma_iso_gpu.append(gam_iso)

            # Directional collision proxy c_hat = L_suf / d_k on CPU
            c_hat_cpu.append((self.L_suf[li] / float(self.d_k)).clone())

        out: Dict[str, Any] = {
            "G_suf": G_suf_gpu,
            "B_suf": B_suf_gpu,
            "L_suf": L_suf_gpu,
            "gamma_scalar": gamma_scalar_gpu,
            "gamma_iso": gamma_iso_gpu,
            "c_hat_cpu": c_hat_cpu,
            "suffix_len": expected_suffix_len,
        }

        if not self.scalar_only:
            k_cat: List[torch.Tensor] = []
            g_cat: List[torch.Tensor] = []
            b_cat: List[torch.Tensor] = []
            for li in range(self.n_linear):
                k_l = torch.cat(self.k_chunks[li], dim=1)
                g_l = torch.cat(self.g_chunks[li], dim=1)
                b_l = torch.cat(self.b_chunks[li], dim=1)
                assert k_l.shape[1] == expected_suffix_len
                k_cat.append(k_l)
                g_cat.append(g_l)
                b_cat.append(b_l)
                self.k_chunks[li].clear()
                self.g_chunks[li].clear()
                self.b_chunks[li].clear()
            out["k_cpu"] = k_cat
            out["g_cpu"] = g_cat
            out["beta_cpu"] = b_cat

        return out


# ============================================================================
# State Delta & Transport Operators (`jlocal`, `jlocal_iso`, `hybrid_gated`)
# ============================================================================

def compute_state_delta_fp64(
    s_old: Dict[str, List[Optional[torch.Tensor]]],
    s_new: Dict[str, List[Optional[torch.Tensor]]],
) -> Dict[str, List[Optional[torch.Tensor]]]:
    """Decision B: compute `s_new - s_old` in `float64`."""
    d_conv: List[Optional[torch.Tensor]] = []
    d_rec: List[Optional[torch.Tensor]] = []
    for co, cn in zip(s_old["conv"], s_new["conv"]):
        if co is not None and cn is not None:
            d_conv.append(cn.to(torch.float64) - co.to(torch.float64))
        else:
            d_conv.append(None)
    for ro, rn in zip(s_old["rec"], s_new["rec"]):
        if ro is not None and rn is not None:
            d_rec.append(rn.to(torch.float64) - ro.to(torch.float64))
        else:
            d_rec.append(None)
    return {"conv": d_conv, "rec": d_rec}


def apply_conv_overlap_update(
    cache,
    delta_conv: List[Optional[torch.Tensor]],
    block_end: int,
    T_ctx: int,
    w_conv: int = 4,
):
    """Apply conv_state overlap update when `block_end >= T_ctx - w_conv + 1`."""
    overlap = block_end - (T_ctx - w_conv + 1)
    if overlap <= 0:
        return
    shift = T_ctx - block_end
    for i, dc in enumerate(delta_conv):
        if dc is not None:
            dst = _layer_get_state_tensor(cache.layers[i], "conv_states")
            if dst is not None:
                orig_dt = dst.dtype
                if shift == 0:
                    c64 = dst.to(torch.float64) + dc
                else:
                    c64 = dst.to(torch.float64).clone()
                    c64[..., : w_conv - shift] = (
                        c64[..., : w_conv - shift] + dc[..., shift:]
                    )
                dst.copy_(c64.to(orig_dt))


@torch.no_grad()
def apply_jlocal_scalar_update(
    cache,
    delta: Dict[str, List[Optional[torch.Tensor]]],
    gamma_scalar_list: List[torch.Tensor],
    block_end: int,
    T_ctx: int,
    w_conv: int = 4,
    use_decay: bool = True,
):
    """
    `jlocal` (`transport-decay`, `use_decay=True`) or `stale` (`use_decay=False`):
    Applies scalar decay `gamma_{b->T} = exp(G_{b->T})` in `float64` (Decision B).
    """
    li = 0
    for i, dr in enumerate(delta["rec"]):
        if dr is not None:
            dst = _layer_get_state_tensor(cache.layers[i], "recurrent_states")
            if dst is not None:
                orig_dt = dst.dtype
                s64 = dst.to(torch.float64)
                if use_decay:
                    gam = gamma_scalar_list[li]
                    s64 = s64 + gam * dr
                else:
                    s64 = s64 + dr
                dst.copy_(s64.to(orig_dt))
                li += 1
    apply_conv_overlap_update(cache, delta["conv"], block_end, T_ctx, w_conv=w_conv)


@torch.no_grad()
def apply_jlocal_iso_update(
    cache,
    delta: Dict[str, List[Optional[torch.Tensor]]],
    gamma_iso_list: List[torch.Tensor],
    block_end: int,
    T_ctx: int,
    w_conv: int = 4,
):
    """
    `jlocal_iso` (`transport-iso-decay`):
    Applies isotropic Householder-corrected scalar decay
    `gamma_iso = exp(G_{b->T} - B_{b->T} / d_k)` in `float64` (Decision B).
    """
    li = 0
    for i, dr in enumerate(delta["rec"]):
        if dr is not None:
            dst = _layer_get_state_tensor(cache.layers[i], "recurrent_states")
            if dst is not None:
                orig_dt = dst.dtype
                s64 = dst.to(torch.float64)
                gam_iso = gamma_iso_list[li]
                s64 = s64 + gam_iso * dr
                dst.copy_(s64.to(orig_dt))
                li += 1
    apply_conv_overlap_update(cache, delta["conv"], block_end, T_ctx, w_conv=w_conv)


@torch.no_grad()
def apply_householder_wy_fp64(
    dS_b: torch.Tensor,
    k_seq: torch.Tensor,
    beta_seq: torch.Tensor,
    g_seq: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Exact chunked WY representation of `prod_{t=b}^{T-1} alpha_t (I - beta_t k_t k_t^T) dS_b`
    in `float64` along key dimension `dim=-2` (`dS_b` has shape `(1, H_active, d_k, d_v)`).
    Uses `torch.linalg.solve_triangular` (`A T = diag(beta)` with `A = I + diag(beta)*tril(K K^T, -1)`),
    matching `w4_e6_multikey_randomkey.py` to ~1e-14 in fp64.
    """
    B, H_v, d_k, d_v = dS_b.shape
    assert B == 1, f"Expected B=1, got B={B}"
    dtype = torch.float64
    device = dS_b.device

    H_k = int(k_seq.shape[2])
    rep = H_v // H_k
    k64 = k_seq.squeeze(0).to(device=device, dtype=dtype)  # (L, H_k, d_k)
    if rep > 1:
        k64 = k64.repeat_interleave(rep, dim=1)            # (L, H_v, d_k)

    k_f = k64.transpose(0, 1).contiguous()                 # (H_v, L, d_k)
    b_f = beta_seq.squeeze(0).to(device=device, dtype=dtype).transpose(0, 1).contiguous()  # (H_v, L)
    g_f = g_seq.squeeze(0).to(device=device, dtype=dtype).transpose(0, 1).contiguous()     # (H_v, L)
    L = int(k_f.shape[1])

    dS = dS_b.squeeze(0).to(device=device, dtype=dtype).clone()  # (H_v, d_k, d_v)

    for c0 in range(0, L, chunk_size):
        c1 = min(L, c0 + chunk_size)
        m = c1 - c0
        Kc = k_f[:, c0:c1, :]                              # (H_v, m, d_k)
        bc = b_f[:, c0:c1]                                 # (H_v, m)
        KK = torch.bmm(Kc, Kc.transpose(-1, -2))           # (H_v, m, m)
        strict_lower = torch.tril(KK, diagonal=-1)
        A = torch.eye(m, device=device, dtype=dtype).unsqueeze(0) + bc.unsqueeze(-1) * strict_lower
        rhs = torch.diag_embed(bc)                         # (H_v, m, m)
        T_mat = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)
        KS = torch.bmm(Kc, dS)                             # (H_v, m, d_v)
        TKS = torch.bmm(T_mat, KS)                         # (H_v, m, d_v)
        dS = dS - torch.bmm(Kc.transpose(-1, -2), TKS)

    gamma = torch.exp(g_f.sum(dim=1)).view(1, H_v, 1, 1)
    return gamma * dS.unsqueeze(0)


@torch.no_grad()
def apply_hybrid_gated_update(
    cache,
    delta: Dict[str, List[Optional[torch.Tensor]]],
    hook_stats: Dict[str, Any],
    block_end: int,
    T_ctx: int,
    device: str,
    tau_c: float = 0.05,
    tau_gamma: float = 1e-4,
    w_conv: int = 4,
) -> Dict[str, Any]:
    """
    Exact Wave 4 `transport-hybrid-gated` (`J-local+`):
    - Head-wise gate: `active_mask = (c_hat >= tau_c) & (gamma_scalar >= tau_gamma)`
    - Inactive heads (`~active_mask`): scalar `transport-decay` (`gamma_scalar = exp(G_{b->T})`).
      STRICTLY NEVER mixes `iso_decay`!
    - Active heads (`active_mask`): selective WY `apply_householder_wy_fp64` along `dim=-2`.
    """
    total_heads = 0
    active_heads = 0
    all_c_hat: List[float] = []

    li = 0
    for i, dr in enumerate(delta["rec"]):
        if dr is None:
            continue
        dst = _layer_get_state_tensor(cache.layers[i], "recurrent_states")
        if dst is None:
            continue

        orig_dt = dst.dtype
        s64 = dst.to(torch.float64)
        gam_s = hook_stats["gamma_scalar"][li]           # (1, H, 1, 1) on GPU
        gam_s_1d = gam_s.view(-1).detach().cpu()         # (H,)
        c_hat_1d = hook_stats["c_hat_cpu"][li].view(-1)  # (H,)
        H_v = int(dr.shape[1])

        all_c_hat.extend(float(x) for x in c_hat_1d.tolist())
        active_mask = (c_hat_1d >= tau_c) & (gam_s_1d >= tau_gamma)
        n_act = int(active_mask.sum().item())
        total_heads += H_v
        active_heads += n_act

        # Start with scalar transport-decay across all heads, then overwrite active heads with WY
        dS_out = gam_s * dr
        if n_act > 0:
            act_idx_cpu = torch.nonzero(active_mask, as_tuple=False).view(-1)
            act_idx_gpu = act_idx_cpu.to(device)

            k_full = hook_stats["k_cpu"][li]       # (1, L, H_k, d_k) on CPU
            g_full = hook_stats["g_cpu"][li]       # (1, L, H_v) on CPU
            b_full = hook_stats["beta_cpu"][li]    # (1, L, H_v) on CPU

            H_k = int(k_full.shape[2])
            rep = H_v // H_k
            if rep > 1:
                k_exp = k_full.repeat_interleave(rep, dim=2)
            else:
                k_exp = k_full

            k_sub = k_exp[:, :, act_idx_cpu, :].to(device=device, dtype=torch.float64)
            g_sub = g_full[:, :, act_idx_cpu].to(device=device, dtype=torch.float64)
            b_sub = b_full[:, :, act_idx_cpu].to(device=device, dtype=torch.float64)
            dr_sub = dr[:, act_idx_gpu, :, :]

            dS_wy = apply_householder_wy_fp64(dr_sub, k_sub, b_sub, g_sub, chunk_size=64)
            dS_out[:, act_idx_gpu, :, :] = dS_wy
            del k_sub, g_sub, b_sub, dr_sub, dS_wy

        s64 = s64 + dS_out
        dst.copy_(s64.to(orig_dt))
        del dS_out
        li += 1

    apply_conv_overlap_update(cache, delta["conv"], block_end, T_ctx, w_conv=w_conv)

    active_frac = float(active_heads / max(1, total_heads))
    return {
        "hybrid_active_heads": int(active_heads),
        "hybrid_total_heads": int(total_heads),
        "hybrid_active_head_frac": active_frac,
        "c_hat_mean": float(np.mean(all_c_hat)) if all_c_hat else 0.0,
        "c_hat_max": float(np.max(all_c_hat)) if all_c_hat else 0.0,
        "tau_c": float(tau_c),
        "tau_gamma": float(tau_gamma),
        "complexity_tag": "O(active_frac*(T-b)*d_k^2)_NOT_O1",
    }


# ============================================================================
# Single-Sample E5 Execution (Single-Active-Cache VRAM Schedule)
# ============================================================================

@torch.no_grad()
def run_e5_sample(
    model,
    base_model,
    tok,
    eos_ids: List[int],
    sample: Dict[str, Any],
    target_length: int,
    mode: str,
    n_linear: int,
    d_k: int,
    w_conv: int,
    block_horizon: str,
    max_chunk: int,
    tau_c: float,
    tau_gamma: float,
    device: str,
    total_vram_gb: float,
    use_logits_to_keep: bool,
    verify_in_place_restore: bool = False,
) -> Dict[str, Any]:
    """
    Execute a single E5 sample at `(target_length, sample['depth'])` with at most
    1.0x full `T_ctx` `DynamicCache` resident in GPU VRAM at any moment.
    """
    assert mode in ("o1", "hybrid"), f"Unsupported mode={mode}"

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t_sample_start = time.perf_counter()

    prompt_info = build_e5_prompt(
        tok=tok,
        sample=sample,
        target_length=target_length,
        n_linear=n_linear,
        block_horizon=block_horizon,
        device=device,
    )
    ids_o = prompt_info["ids_o"]
    ids_e = prompt_info["ids_e"]
    ids_q = prompt_info["ids_q"]
    p0 = prompt_info["p0"]
    p1 = prompt_info["p1"]
    b_end = prompt_info["b_end"]
    T_ctx = prompt_info["T_ctx"]
    suffix_len = prompt_info["suffix_len"]
    f_o = prompt_info["f_o"]
    f_e = prompt_info["f_e"]
    old_val = sample["old"]
    new_val = sample["new"]

    # ------------------------------------------------------------------
    # Step 1: Oneshot Oracle (`ids_e` full unsplit prefill `[0, T_ctx)`)
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    _, c_one = run_chunked_prefill(
        model,
        base_model,
        ids_e,
        pos_start=0,
        cache=None,
        max_chunk=max_chunk,
        use_logits_to_keep=use_logits_to_keep,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    oneshot_prefill_ms = (time.perf_counter() - t0) * 1000.0

    eval_one = eval_from_ctx_cache_in_place(
        model, base_model, tok, eos_ids, c_one, T_ctx, ids_q, use_logits_to_keep=use_logits_to_keep
    )
    lg_one = eval_one["first_logits"].detach().clone()
    ans_lg_one = lg_one.clone()

    # Free `c_one` immediately so GPU VRAM returns to 0 active caches
    del c_one
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Verify oracle first token predicts edited value's first token
    one_top1 = int(lg_one.argmax(dim=-1).item())
    oracle_first_tok_ok = bool(one_top1 == f_e or (new_val in eval_one["text"]))

    # ------------------------------------------------------------------
    # Step 2: Shared Prefix `[0, p0)` + Old/New Block `[p0, b_end)`
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    _, c_pre = run_chunked_prefill(
        model,
        base_model,
        ids_o[:, :p0],
        pos_start=0,
        cache=None,
        max_chunk=max_chunk,
        use_logits_to_keep=use_logits_to_keep,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    prefill_prefix_ms = (time.perf_counter() - t0) * 1000.0

    # Offload `c_pre` copy to CPU RAM so GPU holds only 1 cache during block branching
    c_pre_cpu = move_cache_device(c_pre, "cpu")

    # Advance `c_pre` in-place on old block `ids_o[:, p0:b_end]` -> `c_old_b`
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    _, c_old_b = run_chunked_prefill(
        model,
        base_model,
        ids_o[:, p0:b_end],
        pos_start=p0,
        cache=c_pre,
        max_chunk=max_chunk,
        use_logits_to_keep=use_logits_to_keep,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    old_block_ms = (time.perf_counter() - t0) * 1000.0
    del c_pre

    s_old_b = snap_states(c_old_b)
    old_blk_kv = extract_kv_slice(c_old_b, p0, b_end)
    c_old_b_cpu = move_cache_device(c_old_b, "cpu")
    del c_old_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Restore `c_pre` to GPU and advance on new block `ids_e[:, p0:b_end]` -> `c_new_b`
    c_pre_gpu = move_cache_device(c_pre_cpu, device)
    del c_pre_cpu

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    _, c_new_b = run_chunked_prefill(
        model,
        base_model,
        ids_e[:, p0:b_end],
        pos_start=p0,
        cache=c_pre_gpu,
        max_chunk=max_chunk,
        use_logits_to_keep=use_logits_to_keep,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    new_block_ms = (time.perf_counter() - t0) * 1000.0
    del c_pre_gpu

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    s_new_b = snap_states(c_new_b)
    new_blk_kv = extract_kv_slice(c_new_b, p0, b_end)
    delta = compute_state_delta_fp64(s_old_b, s_new_b)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    delta_compute_ms = (time.perf_counter() - t0) * 1000.0

    block_forward_ms = old_block_ms + new_block_ms + delta_compute_ms
    new_block_only_ms = new_block_ms + delta_compute_ms

    # ------------------------------------------------------------------
    # Step 3: Split Oracle / Suffix Recompute (`c_new_b` + `ids_e[:, b_end:T_ctx]`)
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    _, c_split = run_chunked_prefill(
        model,
        base_model,
        ids_e[:, b_end:T_ctx],
        pos_start=b_end,
        cache=c_new_b,
        max_chunk=max_chunk,
        use_logits_to_keep=use_logits_to_keep,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    suffix_recompute_ms = (time.perf_counter() - t0) * 1000.0
    del c_new_b

    eval_split = eval_from_ctx_cache_in_place(
        model, base_model, tok, eos_ids, c_split, T_ctx, ids_q, use_logits_to_keep=use_logits_to_keep
    )
    lg_split = eval_split["first_logits"].detach().clone()
    del c_split
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 4: J-full (`c_old_b` overwritten at `b_end` via Decision A + suffix recompute)
    # ------------------------------------------------------------------
    c_jfull = move_cache_device(c_old_b_cpu, device)
    overwrite_states(c_jfull, s_new_b)
    patch_kv_slice(c_jfull, new_blk_kv, p0, b_end)
    _, c_jfull = run_chunked_prefill(
        model,
        base_model,
        ids_o[:, b_end:T_ctx],
        pos_start=b_end,
        cache=c_jfull,
        max_chunk=max_chunk,
        use_logits_to_keep=use_logits_to_keep,
    )
    eval_jfull = eval_from_ctx_cache_in_place(
        model, base_model, tok, eos_ids, c_jfull, T_ctx, ids_q, use_logits_to_keep=use_logits_to_keep
    )
    lg_jfull = eval_jfull["first_logits"].detach().clone()
    del c_jfull
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    g1a_diff_vs_split = float((lg_jfull.float() - lg_split.float()).abs().max().item())
    if g1a_diff_vs_split > 1e-5:
        raise AssertionError(
            f"Sample {sample['sample_index']} (T={target_length}): G1a failed! "
            f"g1a_diff_vs_split={g1a_diff_vs_split:.3e} > 1e-5"
        )

    # ------------------------------------------------------------------
    # Step 5: Old Suffix `[b_end, T_ctx)` with Dual GDN Hook -> `c_ctx_old`
    # ------------------------------------------------------------------
    c_ctx_old = move_cache_device(c_old_b_cpu, device)
    del c_old_b_cpu

    scalar_only_hook = bool(mode == "o1")
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with GDNSuffixHookCollector(
        n_linear=n_linear, d_k=d_k, scalar_only=scalar_only_hook
    ) as collector:
        _, c_ctx_old = run_chunked_prefill(
            model,
            base_model,
            ids_o[:, b_end:T_ctx],
            pos_start=b_end,
            cache=c_ctx_old,
            max_chunk=max_chunk,
            use_logits_to_keep=use_logits_to_keep,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    suffix_hook_prefill_ms = (time.perf_counter() - t0) * 1000.0
    prefill_ms = prefill_prefix_ms + old_block_ms + suffix_hook_prefill_ms

    hook_stats = collector.finalize(expected_suffix_len=suffix_len, device=device)

    # Snapshot only the linear states at `T_ctx` (~12 MB on 9B) for in-place restoration
    orig_T_states = snap_states(c_ctx_old)

    # Optional Sample-0 bitwise verification that `restore_ctx_cache_in_place` is exact
    if verify_in_place_restore:
        ref_copy = clone_cache(c_ctx_old)

    # ------------------------------------------------------------------
    # Step 6: Evaluate O(1) Operators & Isolations In-Place on `c_ctx_old`
    # ------------------------------------------------------------------
    # 6a. `jlocal` (scalar `transport-decay` + patched KV)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    apply_jlocal_scalar_update(
        c_ctx_old,
        delta=delta,
        gamma_scalar_list=hook_stats["gamma_scalar"],
        block_end=b_end,
        T_ctx=T_ctx,
        w_conv=w_conv,
        use_decay=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    jlocal_edit_ms = (time.perf_counter() - t0) * 1000.0

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    patch_kv_slice(c_ctx_old, new_blk_kv, p0, b_end)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    kv_patch_ms = (time.perf_counter() - t0) * 1000.0

    eval_jlocal = eval_from_ctx_cache_in_place(
        model, base_model, tok, eos_ids, c_ctx_old, T_ctx, ids_q, use_logits_to_keep=use_logits_to_keep
    )
    restore_ctx_cache_in_place(c_ctx_old, T_ctx, orig_T_states, old_blk_kv, p0, b_end)

    if verify_in_place_restore:
        assert_cache_bitwise_equal(c_ctx_old, ref_copy)
        del ref_copy

    # 6b. `jlocal_iso` (`transport-iso-decay` + patched KV)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    apply_jlocal_iso_update(
        c_ctx_old,
        delta=delta,
        gamma_iso_list=hook_stats["gamma_iso"],
        block_end=b_end,
        T_ctx=T_ctx,
        w_conv=w_conv,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    jlocal_iso_edit_ms = (time.perf_counter() - t0) * 1000.0

    patch_kv_slice(c_ctx_old, new_blk_kv, p0, b_end)
    eval_jlocal_iso = eval_from_ctx_cache_in_place(
        model, base_model, tok, eos_ids, c_ctx_old, T_ctx, ids_q, use_logits_to_keep=use_logits_to_keep
    )
    restore_ctx_cache_in_place(c_ctx_old, T_ctx, orig_T_states, old_blk_kv, p0, b_end)

    # 6c. `stale` (gamma = 1 + patched KV)
    apply_jlocal_scalar_update(
        c_ctx_old,
        delta=delta,
        gamma_scalar_list=hook_stats["gamma_scalar"],
        block_end=b_end,
        T_ctx=T_ctx,
        w_conv=w_conv,
        use_decay=False,
    )
    patch_kv_slice(c_ctx_old, new_blk_kv, p0, b_end)
    eval_stale = eval_from_ctx_cache_in_place(
        model, base_model, tok, eos_ids, c_ctx_old, T_ctx, ids_q, use_logits_to_keep=use_logits_to_keep
    )
    restore_ctx_cache_in_place(c_ctx_old, T_ctx, orig_T_states, old_blk_kv, p0, b_end)

    # 6d. `state_only` (scalar `transport-decay` + unpatched `KV_old`)
    apply_jlocal_scalar_update(
        c_ctx_old,
        delta=delta,
        gamma_scalar_list=hook_stats["gamma_scalar"],
        block_end=b_end,
        T_ctx=T_ctx,
        w_conv=w_conv,
        use_decay=True,
    )
    eval_state_only = eval_from_ctx_cache_in_place(
        model, base_model, tok, eos_ids, c_ctx_old, T_ctx, ids_q, use_logits_to_keep=use_logits_to_keep
    )
    restore_ctx_cache_in_place(c_ctx_old, T_ctx, orig_T_states, old_blk_kv, p0, b_end)

    # 6e. Optional `--mode hybrid`: `hybrid_gated` (`J-local+`)
    eval_hybrid = None
    hybrid_meta = None
    if mode == "hybrid":
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        hybrid_meta = apply_hybrid_gated_update(
            c_ctx_old,
            delta=delta,
            hook_stats=hook_stats,
            block_end=b_end,
            T_ctx=T_ctx,
            device=device,
            tau_c=tau_c,
            tau_gamma=tau_gamma,
            w_conv=w_conv,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        hybrid_edit_ms = (time.perf_counter() - t0) * 1000.0
        hybrid_meta["hybrid_edit_ms"] = float(hybrid_edit_ms)
        hybrid_meta["hybrid_total_ms"] = float(block_forward_ms + hybrid_edit_ms + kv_patch_ms)

        patch_kv_slice(c_ctx_old, new_blk_kv, p0, b_end)
        eval_hybrid = eval_from_ctx_cache_in_place(
            model, base_model, tok, eos_ids, c_ctx_old, T_ctx, ids_q, use_logits_to_keep=use_logits_to_keep
        )
        restore_ctx_cache_in_place(c_ctx_old, T_ctx, orig_T_states, old_blk_kv, p0, b_end)

    # Measure peak GPU memory before freeing `c_ctx_old`
    if torch.cuda.is_available():
        peak_alloc_gb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
        peak_res_gb = float(torch.cuda.max_memory_reserved(device) / (1024 ** 3))
    else:
        peak_alloc_gb = 0.0
        peak_res_gb = 0.0
    peak_headroom_gb = float(max(0.0, total_vram_gb - peak_res_gb))

    # Clean up all sample tensors
    del c_ctx_old, orig_T_states, s_old_b, s_new_b, delta, old_blk_kv, new_blk_kv, hook_stats
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Score all conditions against `oneshot_oracle`
    cond_scores: Dict[str, Dict[str, Any]] = {
        "oneshot_oracle": score_condition(eval_one, new_val, old_val, f_e, f_o, lg_one, ans_lg_one),
        "split_oracle": score_condition(eval_split, new_val, old_val, f_e, f_o, lg_one, ans_lg_one),
        "recompute": score_condition(eval_split, new_val, old_val, f_e, f_o, lg_one, ans_lg_one),
        "jfull": score_condition(eval_jfull, new_val, old_val, f_e, f_o, lg_one, ans_lg_one),
        "jlocal": score_condition(eval_jlocal, new_val, old_val, f_e, f_o, lg_one, ans_lg_one),
        "jlocal_iso": score_condition(eval_jlocal_iso, new_val, old_val, f_e, f_o, lg_one, ans_lg_one),
        "iso_decay": score_condition(eval_jlocal_iso, new_val, old_val, f_e, f_o, lg_one, ans_lg_one),
        "stale": score_condition(eval_stale, new_val, old_val, f_e, f_o, lg_one, ans_lg_one),
        "state_only": score_condition(eval_state_only, new_val, old_val, f_e, f_o, lg_one, ans_lg_one),
    }
    if eval_hybrid is not None:
        cond_scores["hybrid_gated"] = score_condition(
            eval_hybrid, new_val, old_val, f_e, f_o, lg_one, ans_lg_one
        )

    jfull_diff_one = cond_scores["jfull"]["max_logit_diff_vs_oneshot"]
    jfull_kl_one = cond_scores["jfull"]["kl_vs_oneshot"]
    g1b_pass_strict = bool(jfull_kl_one <= 1e-4)
    g1b_pass_relaxed = bool(g1b_pass_strict or (jfull_diff_one <= 0.28125 and jfull_kl_one <= 1.5e-3))

    jlocal_total_ms = float(block_forward_ms + jlocal_edit_ms + kv_patch_ms)
    jlocal_iso_total_ms = float(block_forward_ms + jlocal_iso_edit_ms + kv_patch_ms)
    jlocal_cached_old_total_ms = float(new_block_only_ms + jlocal_edit_ms + kv_patch_ms)
    recompute_dual_ms = float(block_forward_ms + suffix_recompute_ms)
    recompute_single_ms = float(new_block_only_ms + suffix_recompute_ms)

    o1_latencies = {
        "prefill_ms": float(prefill_ms),
        "oneshot_prefill_ms": float(oneshot_prefill_ms),
        "full_prefill_ms": float(oneshot_prefill_ms),
        "prefill_prefix_ms": float(prefill_prefix_ms),
        "old_block_ms": float(old_block_ms),
        "new_block_ms": float(new_block_ms),
        "delta_compute_ms": float(delta_compute_ms),
        "block_forward_ms": float(block_forward_ms),
        "new_block_only_ms": float(new_block_only_ms),
        "jlocal_edit_ms": float(jlocal_edit_ms),
        "jlocal_iso_edit_ms": float(jlocal_iso_edit_ms),
        "kv_patch_ms": float(kv_patch_ms),
        "jlocal_total_ms": jlocal_total_ms,
        "jlocal_iso_total_ms": jlocal_iso_total_ms,
        "jlocal_cached_old_total_ms": jlocal_cached_old_total_ms,
        "suffix_recompute_ms": float(suffix_recompute_ms),
        "recompute_dual_ms": recompute_dual_ms,
        "recompute_single_ms": recompute_single_ms,
        "tstar_speedup_vs_full_prefill": float(oneshot_prefill_ms / max(1e-6, jlocal_total_ms)),
        "tstar_speedup_vs_suffix_recompute": float(recompute_dual_ms / max(1e-6, jlocal_total_ms)),
        "tstar_cached_speedup_vs_suffix_recompute": float(
            recompute_single_ms / max(1e-6, jlocal_cached_old_total_ms)
        ),
        "complexity_tag": "O(1)_strict_suffix_independent",
    }

    sample_wall_s = time.perf_counter() - t_sample_start

    record: Dict[str, Any] = {
        "mode": mode,
        "target_length": int(target_length),
        "sample_index": int(sample["sample_index"]),
        "depth": float(sample["depth"]),
        "actual_depth_ratio": float(prompt_info["actual_depth_ratio"]),
        "tpl_id": int(sample["tpl_id"]),
        "target": sample["target"],
        "old": old_val,
        "new": new_val,
        "T_ctx": int(T_ctx),
        "q_len": int(prompt_info["q_len"]),
        "p0": int(p0),
        "p1": int(p1),
        "b_end": int(b_end),
        "b_local": int(prompt_info["b_local"]),
        "b_deep": int(prompt_info["b_deep"]),
        "suffix_len": int(suffix_len),
        "oracle_first_tok_ok": oracle_first_tok_ok,
        "g1a_diff_vs_split": g1a_diff_vs_split,
        "g1a_pass": bool(g1a_diff_vs_split <= 1e-5),
        "g1b_diff_vs_oneshot": float(jfull_diff_one),
        "g1b_kl_vs_oneshot": float(jfull_kl_one),
        "g1b_pass_strict": g1b_pass_strict,
        "g1b_pass_relaxed": g1b_pass_relaxed,
        "o1_latencies": o1_latencies,
        "vram": {
            "peak_allocated_gb": peak_alloc_gb,
            "peak_reserved_gb": peak_res_gb,
            "peak_headroom_gb": peak_headroom_gb,
            "device_total_vram_gb": float(total_vram_gb),
        },
        "conditions": cond_scores,
        "sample_wall_s": float(sample_wall_s),
    }

    if hybrid_meta is not None:
        record["non_o1_hybrid_metrics"] = hybrid_meta

    return record


# ============================================================================
# Optional Steady-State `T*` Crossover Benchmark (`--bench-tstar`)
# ============================================================================

@torch.no_grad()
def run_tstar_benchmark(
    model,
    base_model,
    tok,
    sample_0: Dict[str, Any],
    n_linear: int,
    d_k: int,
    w_conv: int,
    block_horizon: str,
    max_chunk: int,
    device: str,
    use_logits_to_keep: bool,
    smoke: bool = False,
) -> Dict[str, Any]:
    """
    Steady-state `T*` latency benchmark across context lengths to measure exact
    crossover token length `T*` where `J-local` beats `full_prefill` and `suffix_recompute`.
    Strictly O(1) (`jlocal` and `jlocal_iso`), completely separate from `hybrid_gated`.
    """
    lengths = [256, 512] if smoke else [256, 512, 1024, 2048, 4096]
    n_repeats = 2 if smoke else 3
    rows: List[Dict[str, Any]] = []

    for T_len in lengths:
        prompt_info = build_e5_prompt(
            tok=tok,
            sample=sample_0,
            target_length=T_len,
            n_linear=n_linear,
            block_horizon=block_horizon,
            device=device,
        )
        ids_o = prompt_info["ids_o"]
        ids_e = prompt_info["ids_e"]
        p0 = prompt_info["p0"]
        b_end = prompt_info["b_end"]
        T_ctx = prompt_info["T_ctx"]
        suffix_len = prompt_info["suffix_len"]

        # Warmup once
        _, c_w = run_chunked_prefill(
            model, base_model, ids_e, pos_start=0, cache=None, max_chunk=max_chunk, use_logits_to_keep=use_logits_to_keep
        )
        del c_w
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

        full_prefill_times: List[float] = []
        block_dual_times: List[float] = []
        block_single_times: List[float] = []
        suffix_recompute_times: List[float] = []
        jlocal_edit_times: List[float] = []
        jlocal_iso_edit_times: List[float] = []
        kv_patch_times: List[float] = []

        for _ in range(n_repeats):
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _, c_full = run_chunked_prefill(
                model, base_model, ids_e, pos_start=0, cache=None, max_chunk=max_chunk, use_logits_to_keep=use_logits_to_keep
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            full_prefill_times.append((time.perf_counter() - t0) * 1000.0)
            del c_full

            _, c_pre = run_chunked_prefill(
                model, base_model, ids_o[:, :p0], pos_start=0, cache=None, max_chunk=max_chunk, use_logits_to_keep=use_logits_to_keep
            )
            c_pre_cpu = move_cache_device(c_pre, "cpu")

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _, c_old_b = run_chunked_prefill(
                model, base_model, ids_o[:, p0:b_end], pos_start=p0, cache=c_pre, max_chunk=max_chunk, use_logits_to_keep=use_logits_to_keep
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t_old_blk = (time.perf_counter() - t0) * 1000.0
            del c_pre
            s_old_b = snap_states(c_old_b)
            old_blk_kv = extract_kv_slice(c_old_b, p0, b_end)

            with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as coll:
                _, c_ctx_old = run_chunked_prefill(
                    model, base_model, ids_o[:, b_end:T_ctx], pos_start=b_end, cache=c_old_b, max_chunk=max_chunk, use_logits_to_keep=use_logits_to_keep
                )
            del c_old_b
            h_stats = coll.finalize(expected_suffix_len=suffix_len, device=device)

            c_pre_2 = move_cache_device(c_pre_cpu, device)
            del c_pre_cpu
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _, c_new_b = run_chunked_prefill(
                model, base_model, ids_e[:, p0:b_end], pos_start=p0, cache=c_pre_2, max_chunk=max_chunk, use_logits_to_keep=use_logits_to_keep
            )
            s_new_b = snap_states(c_new_b)
            new_blk_kv = extract_kv_slice(c_new_b, p0, b_end)
            delta = compute_state_delta_fp64(s_old_b, s_new_b)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t_new_blk = (time.perf_counter() - t0) * 1000.0
            del c_pre_2

            block_dual_times.append(t_old_blk + t_new_blk)
            block_single_times.append(t_new_blk)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _, c_suf = run_chunked_prefill(
                model, base_model, ids_e[:, b_end:T_ctx], pos_start=b_end, cache=c_new_b, max_chunk=max_chunk, use_logits_to_keep=use_logits_to_keep
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            suffix_recompute_times.append((time.perf_counter() - t0) * 1000.0)
            del c_new_b, c_suf

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            apply_jlocal_scalar_update(
                c_ctx_old, delta, h_stats["gamma_scalar"], b_end, T_ctx, w_conv=w_conv, use_decay=True
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            jlocal_edit_times.append((time.perf_counter() - t0) * 1000.0)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            apply_jlocal_iso_update(
                c_ctx_old, delta, h_stats["gamma_iso"], b_end, T_ctx, w_conv=w_conv
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            jlocal_iso_edit_times.append((time.perf_counter() - t0) * 1000.0)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            patch_kv_slice(c_ctx_old, new_blk_kv, p0, b_end)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            kv_patch_times.append((time.perf_counter() - t0) * 1000.0)

            del c_ctx_old, s_old_b, s_new_b, old_blk_kv, new_blk_kv, delta, h_stats
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        med_full = float(np.median(full_prefill_times))
        med_blk_dual = float(np.median(block_dual_times))
        med_blk_single = float(np.median(block_single_times))
        med_suf = float(np.median(suffix_recompute_times))
        med_jl_edit = float(np.median(jlocal_edit_times))
        med_iso_edit = float(np.median(jlocal_iso_edit_times))
        med_kv = float(np.median(kv_patch_times))

        jl_dual_total = med_blk_dual + med_jl_edit + med_kv
        jl_single_total = med_blk_single + med_jl_edit + med_kv
        iso_dual_total = med_blk_dual + med_iso_edit + med_kv
        recomp_dual_total = med_blk_dual + med_suf
        recomp_single_total = med_blk_single + med_suf

        rows.append({
            "target_length": int(T_len),
            "T_ctx": int(T_ctx),
            "suffix_len": int(suffix_len),
            "full_prefill_ms": med_full,
            "block_forward_ms": med_blk_dual,
            "new_block_only_ms": med_blk_single,
            "jlocal_edit_ms": med_jl_edit,
            "jlocal_iso_edit_ms": med_iso_edit,
            "kv_patch_ms": med_kv,
            "suffix_recompute_ms": med_suf,
            "jlocal_total_ms": jl_dual_total,
            "jlocal_iso_total_ms": iso_dual_total,
            "jlocal_cached_old_total_ms": jl_single_total,
            "recompute_dual_ms": recomp_dual_total,
            "recompute_single_ms": recomp_single_total,
            "speedup_vs_full_prefill": float(med_full / max(1e-6, jl_dual_total)),
            "speedup_vs_suffix_recompute": float(recomp_dual_total / max(1e-6, jl_dual_total)),
            "speedup_cached_vs_suffix_recompute": float(recomp_single_total / max(1e-6, jl_single_total)),
        })

    # Interpolate crossover T* where speedup >= 1.0
    def find_crossover(rows_list: List[Dict[str, Any]], key_speedup: str) -> Optional[float]:
        for idx, r in enumerate(rows_list):
            if r[key_speedup] >= 1.0:
                if idx == 0:
                    return float(r["T_ctx"])
                r_prev = rows_list[idx - 1]
                s0, s1 = r_prev[key_speedup], r[key_speedup]
                t0_c, t1_c = float(r_prev["T_ctx"]), float(r["T_ctx"])
                if abs(s1 - s0) < 1e-9:
                    return t1_c
                frac = (1.0 - s0) / (s1 - s0)
                return float(t0_c + frac * (t1_c - t0_c))
        return None

    return {
        "rows": rows,
        "tstar_vs_full_prefill_tokens": find_crossover(rows, "speedup_vs_full_prefill"),
        "tstar_vs_suffix_recompute_tokens": find_crossover(rows, "speedup_vs_suffix_recompute"),
        "tstar_cached_vs_suffix_recompute_tokens": find_crossover(
            rows, "speedup_cached_vs_suffix_recompute"
        ),
        "g6_pass_4k_ge_2x": bool(
            any(
                r["target_length"] >= 4096 and r["speedup_vs_full_prefill"] >= 2.0
                for r in rows
            )
        ),
    }


# ============================================================================
# Statistical Aggregation & JSONL / Summary I/O
# ============================================================================

def bootstrap_mean_ci(
    vals: List[float], n_boot: int = 2000, seed: int = 42
) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "iqr": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    med_v = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25]) if n > 1 else (med_v, med_v)
    if n == 1:
        return {
            "n": 1,
            "mean": mean_v,
            "std": 0.0,
            "median": med_v,
            "iqr": 0.0,
            "ci95_low": mean_v,
            "ci95_high": mean_v,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n": n,
        "mean": mean_v,
        "std": std_v,
        "median": med_v,
        "iqr": float(q75 - q25),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def exact_sign_test_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial sign test p-value for `k` successes out of `n` under p=0.5."""
    if n <= 0:
        return 1.0
    m = min(k, n - k)
    if 2 * m == n:
        return 1.0
    # Use log-sum-exp with math.lgamma for numerical stability up to large n
    log_half_n = -n * math.log(2.0)
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n
        for i in range(m + 1)
    ]
    max_lt = max(log_terms)
    one_tail = math.exp(max_lt) * sum(math.exp(lt - max_lt) for lt in log_terms)
    return float(min(1.0, 2.0 * one_tail))


def wilcoxon_signed_rank_two_sided_p(diffs: np.ndarray) -> float:
    """
    Two-sided Wilcoxon signed-rank p-value (pure numpy/math, zero scipy dependency):
    Uses normal approximation with tie-variance adjustment and continuity correction.
    """
    nz = diffs[np.abs(diffs) > 1e-12]
    n = int(nz.size)
    if n < 2:
        return 1.0
    abs_d = np.abs(nz)
    order = np.argsort(abs_d)
    sorted_abs = abs_d[order]

    ranks = np.empty(n, dtype=np.float64)
    tie_counts: List[int] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(sorted_abs[j] - sorted_abs[i]) <= 1e-12:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        if j - i > 1:
            tie_counts.append(j - i)
        i = j

    w_pos = float(np.sum(ranks[nz > 0]))
    w_neg = float(np.sum(ranks[nz < 0]))
    w_stat = min(w_pos, w_neg)

    mean_w = n * (n + 1.0) / 4.0
    tie_corr = sum(t * (t * t - 1.0) for t in tie_counts) / 48.0
    var_w = (n * (n + 1.0) * (2.0 * n + 1.0) / 24.0) - tie_corr
    if var_w <= 1e-12:
        return 1.0
    z = (abs(w_stat - mean_w) - 0.5) / math.sqrt(var_w)
    if z <= 0.0:
        return 1.0
    return float(min(1.0, math.erfc(z / math.sqrt(2.0))))


def paired_comparison(
    records: List[Dict[str, Any]], cond_a: str, cond_b: str
) -> Dict[str, Any]:
    """
    Paired comparison between `cond_a` and `cond_b` across `records`:
    - KL difference `kl(cond_a) - kl(cond_b)` (negative means `cond_a` has lower KL!)
    - Exact two-sided binomial sign test p-value & Wilcoxon signed-rank p-value on KL
    - Exact accuracy difference `contains_new(cond_a) - contains_new(cond_b)`
    """
    kl_a: List[float] = []
    kl_b: List[float] = []
    acc_a: List[float] = []
    acc_b: List[float] = []
    top1_a: List[float] = []
    top1_b: List[float] = []

    for r in records:
        ca = r["conditions"].get(cond_a)
        cb = r["conditions"].get(cond_b)
        if ca is None or cb is None:
            continue
        kl_a.append(float(ca["kl_vs_oneshot"]))
        kl_b.append(float(cb["kl_vs_oneshot"]))
        acc_a.append(1.0 if ca["contains_new"] else 0.0)
        acc_b.append(1.0 if cb["contains_new"] else 0.0)
        top1_a.append(1.0 if ca["top1_eq_oracle"] else 0.0)
        top1_b.append(1.0 if cb["top1_eq_oracle"] else 0.0)

    n = len(kl_a)
    if n == 0:
        return {"n": 0}

    arr_kl_a = np.asarray(kl_a, dtype=np.float64)
    arr_kl_b = np.asarray(kl_b, dtype=np.float64)
    diff_kl = arr_kl_a - arr_kl_b  # < 0 means cond_a beats cond_b

    a_lower_kl = int(np.sum(diff_kl < -1e-9))
    b_lower_kl = int(np.sum(diff_kl > 1e-9))
    ties_kl = int(n - a_lower_kl - b_lower_kl)

    n_non_tie = a_lower_kl + b_lower_kl
    sign_p = exact_sign_test_two_sided_p(a_lower_kl, n_non_tie)
    wilcoxon_p = wilcoxon_signed_rank_two_sided_p(diff_kl)

    arr_acc_a = np.asarray(acc_a, dtype=np.float64)
    arr_acc_b = np.asarray(acc_b, dtype=np.float64)
    diff_acc = arr_acc_a - arr_acc_b

    a_wins_acc = int(np.sum(diff_acc > 0.5))
    b_wins_acc = int(np.sum(diff_acc < -0.5))

    return {
        "n": n,
        "cond_a": cond_a,
        "cond_b": cond_b,
        "kl_diff_a_minus_b": bootstrap_mean_ci(diff_kl.tolist()),
        "a_lower_kl_count": a_lower_kl,
        "b_lower_kl_count": b_lower_kl,
        "ties_kl_count": ties_kl,
        "kl_sign_test_p_two_sided": sign_p,
        "kl_wilcoxon_p_two_sided": wilcoxon_p,
        "contains_new_diff_a_minus_b": bootstrap_mean_ci(diff_acc.tolist()),
        "a_wins_contains_new": a_wins_acc,
        "b_wins_contains_new": b_wins_acc,
        "top1_eq_oracle_diff_a_minus_b": bootstrap_mean_ci(
            (np.asarray(top1_a) - np.asarray(top1_b)).tolist()
        ),
    }


def aggregate_records_slice(records: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    if not records:
        return {"n": 0}

    cond_names = [
        "oneshot_oracle",
        "split_oracle",
        "recompute",
        "jfull",
        "jlocal",
        "jlocal_iso",
        "iso_decay",
        "stale",
        "state_only",
    ]
    if mode == "hybrid" or any("hybrid_gated" in r.get("conditions", {}) for r in records):
        cond_names.append("hybrid_gated")

    by_condition: Dict[str, Any] = {}
    for cname in cond_names:
        sub = [r["conditions"][cname] for r in records if cname in r.get("conditions", {})]
        if not sub:
            continue
        by_condition[cname] = {
            "n": len(sub),
            "contains_new": bootstrap_mean_ci([1.0 if s["contains_new"] else 0.0 for s in sub]),
            "contains_old": bootstrap_mean_ci([1.0 if s["contains_old"] else 0.0 for s in sub]),
            "top1_eq_oracle": bootstrap_mean_ci([1.0 if s["top1_eq_oracle"] else 0.0 for s in sub]),
            "first_tok_is_new": bootstrap_mean_ci([1.0 if s["first_tok_is_new"] else 0.0 for s in sub]),
            "margin_new_vs_old": bootstrap_mean_ci([float(s["margin_new_vs_old"]) for s in sub]),
            "kl_vs_oneshot": bootstrap_mean_ci([float(s["kl_vs_oneshot"]) for s in sub]),
            "max_logit_diff_vs_oneshot": bootstrap_mean_ci(
                [float(s["max_logit_diff_vs_oneshot"]) for s in sub]
            ),
        }

    # Latency summary (strictly O(1) operators)
    lat_keys = [
        "prefill_ms",
        "oneshot_prefill_ms",
        "full_prefill_ms",
        "block_forward_ms",
        "new_block_only_ms",
        "jlocal_edit_ms",
        "jlocal_iso_edit_ms",
        "kv_patch_ms",
        "jlocal_total_ms",
        "jlocal_iso_total_ms",
        "jlocal_cached_old_total_ms",
        "suffix_recompute_ms",
        "recompute_dual_ms",
        "recompute_single_ms",
        "tstar_speedup_vs_full_prefill",
        "tstar_speedup_vs_suffix_recompute",
        "tstar_cached_speedup_vs_suffix_recompute",
    ]
    o1_latency_summary = {
        k: bootstrap_mean_ci([float(r["o1_latencies"][k]) for r in records if k in r.get("o1_latencies", {})])
        for k in lat_keys
    }
    o1_latency_summary["complexity_tag"] = "O(1)_strict_suffix_independent"

    # Paired comparisons
    paired = {
        "jlocal_iso_vs_jlocal": paired_comparison(records, "jlocal_iso", "jlocal"),
        "jlocal_vs_stale": paired_comparison(records, "jlocal", "stale"),
        "jlocal_iso_vs_stale": paired_comparison(records, "jlocal_iso", "stale"),
        "jlocal_vs_state_only": paired_comparison(records, "jlocal", "state_only"),
    }
    if "hybrid_gated" in by_condition:
        paired["hybrid_gated_vs_jlocal"] = paired_comparison(records, "hybrid_gated", "jlocal")
        paired["hybrid_gated_vs_jlocal_iso"] = paired_comparison(records, "hybrid_gated", "jlocal_iso")
        paired["hybrid_gated_vs_stale"] = paired_comparison(records, "hybrid_gated", "stale")

    # G1a / G1b / VRAM summary
    g1a_diffs = [float(r["g1a_diff_vs_split"]) for r in records]
    vram_alloc = [float(r["vram"]["peak_allocated_gb"]) for r in records]
    vram_res = [float(r["vram"]["peak_reserved_gb"]) for r in records]
    vram_head = [float(r["vram"]["peak_headroom_gb"]) for r in records]

    out_slice: Dict[str, Any] = {
        "n": len(records),
        "g1a_pass_count": sum(1 for r in records if r["g1a_pass"]),
        "g1a_max_diff_vs_split": float(max(g1a_diffs)) if g1a_diffs else 0.0,
        "g1b_pass_strict_count": sum(1 for r in records if float(r["g1b_kl_vs_oneshot"]) <= 1e-4),
        "g1b_pass_relaxed_count": sum(1 for r in records if r["g1b_pass_relaxed"]),
        "oracle_first_tok_ok_count": sum(1 for r in records if r["oracle_first_tok_ok"]),
        "vram_summary": {
            "max_peak_allocated_gb": float(max(vram_alloc)) if vram_alloc else 0.0,
            "max_peak_reserved_gb": float(max(vram_res)) if vram_res else 0.0,
            "min_peak_headroom_gb": float(min(vram_head)) if vram_head else 0.0,
        },
        "o1_latency_summary": o1_latency_summary,
        "conditions": by_condition,
        "paired_comparisons": paired,
    }

    # Separate non-O(1) hybrid summary if present
    hyb_recs = [r["non_o1_hybrid_metrics"] for r in records if "non_o1_hybrid_metrics" in r]
    if hyb_recs:
        out_slice["non_o1_hybrid_summary"] = {
            "n": len(hyb_recs),
            "complexity_tag": "O(active_frac*(T-b)*d_k^2)_NOT_O1",
            "hybrid_edit_ms": bootstrap_mean_ci([float(h["hybrid_edit_ms"]) for h in hyb_recs]),
            "hybrid_total_ms": bootstrap_mean_ci([float(h["hybrid_total_ms"]) for h in hyb_recs]),
            "hybrid_active_head_frac": bootstrap_mean_ci(
                [float(h["hybrid_active_head_frac"]) for h in hyb_recs]
            ),
            "c_hat_mean": bootstrap_mean_ci([float(h["c_hat_mean"]) for h in hyb_recs]),
            "c_hat_max": bootstrap_mean_ci([float(h["c_hat_max"]) for h in hyb_recs]),
        }

    return out_slice


def load_deduplicated_jsonl(jsonl_path: Path, mode_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    by_key: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            m = str(rec.get("mode", "o1"))
            if mode_filter is not None and m != mode_filter:
                continue
            if "g1b_kl_vs_oneshot" in rec:
                rec["g1b_pass_strict"] = bool(float(rec["g1b_kl_vs_oneshot"]) <= 1e-4)
                rec["g1b_pass_relaxed"] = bool(
                    rec["g1b_pass_strict"]
                    or (
                        float(rec.get("g1b_diff_vs_oneshot", 0.0)) <= 0.28125
                        and float(rec["g1b_kl_vs_oneshot"]) <= 1.5e-3
                    )
                )
            key = (m, int(rec["target_length"]), int(rec["sample_index"]))
            by_key[key] = rec
    sorted_keys = sorted(by_key.keys(), key=lambda x: (x[0], x[1], x[2]))
    return [by_key[k] for k in sorted_keys]


def append_record_jsonl(jsonl_path: Path, record: Dict[str, Any]):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def build_full_summary(
    records: List[Dict[str, Any]],
    model_name: str,
    mode: str,
    n_linear: int,
    d_k: int,
    block_horizon: str,
    tstar_bench: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    lengths = sorted({int(r["target_length"]) for r in records})
    depths = sorted({float(r["depth"]) for r in records})
    tpls = sorted({int(r["tpl_id"]) for r in records})

    by_length = {
        str(L): aggregate_records_slice([r for r in records if int(r["target_length"]) == L], mode)
        for L in lengths
    }
    by_depth = {
        f"{d:.2f}": aggregate_records_slice([r for r in records if abs(float(r["depth"]) - d) < 1e-6], mode)
        for d in depths
    }
    by_length_and_depth: Dict[str, Any] = {}
    for L in lengths:
        for d in depths:
            sub = [
                r
                for r in records
                if int(r["target_length"]) == L and abs(float(r["depth"]) - d) < 1e-6
            ]
            if sub:
                by_length_and_depth[f"T{L}_d{d:.2f}"] = aggregate_records_slice(sub, mode)

    by_template = {
        f"tpl_{t}": aggregate_records_slice([r for r in records if int(r["tpl_id"]) == t], mode)
        for t in tpls
    }

    summary: Dict[str, Any] = {
        "experiment": "Wave5_9B_E5_LongContext",
        "model": model_name,
        "mode": mode,
        "n_linear": int(n_linear),
        "d_k": int(d_k),
        "block_horizon": block_horizon,
        "total_records": len(records),
        "context_lengths_present": lengths,
        "depths_present": depths,
        "overall": aggregate_records_slice(records, mode),
        "by_length": by_length,
        "by_depth": by_depth,
        "by_length_and_depth": by_length_and_depth,
        "by_template": by_template,
    }
    if tstar_bench is not None:
        summary["tstar_benchmark"] = tstar_bench
    return summary


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Wave 5 Qwen3.5-9B E5 Staged Runner (4K/16K, O(1) vs Hybrid-Gated)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("RUNNER_MODEL", "Qwen/Qwen3.5-9B"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("RUNNER_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["o1", "hybrid"],
        default=os.environ.get("RUNNER_MODE", "o1"),
    )
    parser.add_argument(
        "--context-lengths",
        type=str,
        default=os.environ.get("RUNNER_CONTEXT_LENGTHS", "4096"),
        help="Comma-separated target lengths, e.g. '4096' or '16384' or '4096,16384'",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default=os.environ.get("RUNNER_DEPTHS", "0.25,0.50,0.75"),
        help="Comma-separated edit depths cycled deterministically across sample_index",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=int(os.environ.get("RUNNER_N_SAMPLES", "200")),
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=int(os.environ.get("RUNNER_START_IDX", "0")),
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=(
            int(os.environ["RUNNER_END_IDX"])
            if "RUNNER_END_IDX" in os.environ and os.environ["RUNNER_END_IDX"].strip() != ""
            else None
        ),
    )
    parser.add_argument(
        "--block-horizon",
        type=str,
        choices=["local", "deep"],
        default=os.environ.get("RUNNER_BLOCK_HORIZON", "deep"),
    )
    parser.add_argument(
        "--max-chunk",
        type=int,
        default=int(os.environ.get("RUNNER_MAX_CHUNK", "2048")),
    )
    parser.add_argument(
        "--tau-c",
        type=float,
        default=float(os.environ.get("RUNNER_TAU_C", "0.05")),
    )
    parser.add_argument(
        "--tau-gamma",
        type=float,
        default=float(os.environ.get("RUNNER_TAU_GAMMA", "1e-4")),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("RUNNER_SEED", "202605")),
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default=os.environ.get("RUNNER_OUT_JSONL", "pilot/w5_9b_e5_records.jsonl"),
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default=os.environ.get("RUNNER_OUT_SUMMARY", "pilot/w5_9b_e5_summary.json"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=os.environ.get("RUNNER_RESUME", "0") == "1",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=os.environ.get("RUNNER_SMOKE", "0") == "1",
    )
    parser.add_argument(
        "--bench-tstar",
        action="store_true",
        default=os.environ.get("RUNNER_BENCH_TSTAR", "0") == "1",
    )
    parser.add_argument(
        "--dry-run-bank",
        action="store_true",
        help="Verify all n=200 sample prompts/spans on CPU without loading model weights",
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    t_run_start = time.perf_counter()

    depths = [float(x.strip()) for x in args.depths.split(",") if x.strip()]
    if args.smoke and "RUNNER_CONTEXT_LENGTHS" not in os.environ and args.context_lengths == "4096":
        context_lengths = [512]
    else:
        context_lengths = [int(x.strip()) for x in args.context_lengths.split(",") if x.strip()]

    sample_bank = generate_e5_sample_bank(
        n_samples=args.n_samples, depths=depths, seed=args.seed
    )

    if args.end_idx is not None:
        end_idx = min(args.n_samples, int(args.end_idx))
    elif args.smoke:
        end_idx = min(args.n_samples, max(args.start_idx + 3, 3))
    else:
        end_idx = args.n_samples
    start_idx = max(0, int(args.start_idx))

    print(
        f"[W5 E5 Runner] model={args.model} | mode={args.mode} | "
        f"lengths={context_lengths} | depths={depths} | "
        f"slice=[{start_idx}:{end_idx}]/{args.n_samples} | "
        f"max_chunk={args.max_chunk} | resume={args.resume} | smoke={args.smoke}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    t_cfg = getattr(cfg, "text_config", cfg)
    n_linear = count_linear_layers(t_cfg)
    d_k = int(getattr(t_cfg, "linear_key_head_dim", 128))
    w_conv = int(getattr(t_cfg, "linear_conv_kernel_dim", 4))

    # Optional CPU-only dry-run of the full 200-sample bank
    if args.dry_run_bank:
        t0 = time.perf_counter()
        depth_counts: Dict[str, int] = {}
        tpl_counts: Dict[int, int] = {}
        for L in context_lengths:
            for s in sample_bank[start_idx:end_idx]:
                p_info = build_e5_prompt(
                    tok=tok,
                    sample=s,
                    target_length=L,
                    n_linear=n_linear,
                    block_horizon=args.block_horizon,
                    device="cpu",
                )
                assert p_info["p1"] - p_info["p0"] == 4
                assert p_info["f_o"] != p_info["f_e"]
                if L == context_lengths[0]:
                    d_key = f"{s['depth']:.2f}"
                    depth_counts[d_key] = depth_counts.get(d_key, 0) + 1
                    tpl_counts[s["tpl_id"]] = tpl_counts.get(s["tpl_id"], 0) + 1
        dt = time.perf_counter() - t0
        print(
            f"[DRY-RUN BANK PASS] Verified {end_idx - start_idx} samples x {len(context_lengths)} "
            f"lengths ({(end_idx - start_idx) * len(context_lengths)} prompts) in {dt:.2f}s | "
            f"n_linear={n_linear}, d_k={d_k}, w_conv={w_conv} | "
            f"depth_counts={depth_counts} | tpl_counts={tpl_counts}",
            flush=True,
        )
        return

    # Load model on GPU in bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    base_model = getattr(model.model, "language_model", model.model)
    eos_ids = get_eos_ids(model, tok)
    use_logits_to_keep = has_logits_to_keep_arg(model)

    if torch.cuda.is_available() and "cuda" in str(args.device):
        total_vram_gb = float(
            torch.cuda.get_device_properties(args.device).total_memory / (1024 ** 3)
        )
    else:
        total_vram_gb = 0.0

    print(
        f"[Model Loaded] n_linear={n_linear}, d_k={d_k}, w_conv={w_conv}, "
        f"use_logits_to_keep={use_logits_to_keep}, total_vram_gb={total_vram_gb:.2f}",
        flush=True,
    )

    out_jsonl_path = Path(args.out_jsonl)
    out_summary_path = Path(args.out_summary)

    existing_records = load_deduplicated_jsonl(out_jsonl_path, mode_filter=None)
    done_keys = {
        (str(r.get("mode", "o1")), int(r["target_length"]), int(r["sample_index"]))
        for r in existing_records
    }
    if args.resume and done_keys:
        print(f"[Resume] Found {len(done_keys)} completed records in {out_jsonl_path}", flush=True)

    verified_restore_once = False
    newly_ran = 0
    skipped = 0

    for target_length in context_lengths:
        for s_idx in range(start_idx, end_idx):
            key = (args.mode, int(target_length), int(s_idx))
            if args.resume and key in done_keys:
                skipped += 1
                continue

            sample = sample_bank[s_idx]
            rec = run_e5_sample(
                model=model,
                base_model=base_model,
                tok=tok,
                eos_ids=eos_ids,
                sample=sample,
                target_length=target_length,
                mode=args.mode,
                n_linear=n_linear,
                d_k=d_k,
                w_conv=w_conv,
                block_horizon=args.block_horizon,
                max_chunk=args.max_chunk,
                tau_c=args.tau_c,
                tau_gamma=args.tau_gamma,
                device=args.device,
                total_vram_gb=total_vram_gb,
                use_logits_to_keep=use_logits_to_keep,
                verify_in_place_restore=(not verified_restore_once),
            )
            verified_restore_once = True
            append_record_jsonl(out_jsonl_path, rec)
            done_keys.add(key)
            newly_ran += 1

            jl = rec["conditions"]["jlocal"]
            iso = rec["conditions"]["jlocal_iso"]
            st = rec["conditions"]["stale"]
            hyb_str = ""
            if "hybrid_gated" in rec["conditions"]:
                hyb = rec["conditions"]["hybrid_gated"]
                hm = rec["non_o1_hybrid_metrics"]
                hyb_str = (
                    f" | hyb_KL={hyb['kl_vs_oneshot']:.4f} "
                    f"(act={hm['hybrid_active_head_frac']*100:.1f}%, "
                    f"edit={hm['hybrid_edit_ms']:.1f}ms)"
                )

            print(
                f"  [T={target_length} s={s_idx:03d} d={sample['depth']:.2f} tpl={sample['tpl_id']}] "
                f"G1a={rec['g1a_diff_vs_split']:.2e} | "
                f"jlocal_KL={jl['kl_vs_oneshot']:.4f} (new={int(jl['contains_new'])}) | "
                f"iso_KL={iso['kl_vs_oneshot']:.4f} (new={int(iso['contains_new'])}) | "
                f"stale_KL={st['kl_vs_oneshot']:.4f}{hyb_str} | "
                f"edit={rec['o1_latencies']['jlocal_edit_ms']:.2f}ms "
                f"suf_recomp={rec['o1_latencies']['suffix_recompute_ms']:.1f}ms | "
                f"VRAM_res={rec['vram']['peak_reserved_gb']:.2f}GB "
                f"(headroom={rec['vram']['peak_headroom_gb']:.2f}GB) | "
                f"{rec['sample_wall_s']:.2f}s",
                flush=True,
            )

    tstar_bench = None
    if args.bench_tstar or (args.smoke and args.mode == "o1"):
        print("[T* Benchmark] Running steady-state O(1) T* crossover benchmark...", flush=True)
        tstar_bench = run_tstar_benchmark(
            model=model,
            base_model=base_model,
            tok=tok,
            sample_0=sample_bank[0],
            n_linear=n_linear,
            d_k=d_k,
            w_conv=w_conv,
            block_horizon=args.block_horizon,
            max_chunk=args.max_chunk,
            device=args.device,
            use_logits_to_keep=use_logits_to_keep,
            smoke=args.smoke,
        )

    # Aggregate over all deduplicated records in `out_jsonl_path` matching `args.mode`
    all_mode_records = load_deduplicated_jsonl(out_jsonl_path, mode_filter=args.mode)
    summary = build_full_summary(
        records=all_mode_records,
        model_name=args.model,
        mode=args.mode,
        n_linear=n_linear,
        d_k=d_k,
        block_horizon=args.block_horizon,
        tstar_bench=tstar_bench,
    )
    summary["run_metadata"] = {
        "newly_ran_records": newly_ran,
        "skipped_resume_records": skipped,
        "total_mode_records": len(all_mode_records),
        "wall_clock_s": float(time.perf_counter() - t_run_start),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "max_chunk": int(args.max_chunk),
        "smoke": bool(args.smoke),
    }

    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    # Explicitly release GPU memory before process exit
    del model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ov = summary["overall"]
    jl_kl = ov.get("conditions", {}).get("jlocal", {}).get("kl_vs_oneshot", {}).get("mean", 0.0)
    iso_kl = ov.get("conditions", {}).get("jlocal_iso", {}).get("kl_vs_oneshot", {}).get("mean", 0.0)
    st_kl = ov.get("conditions", {}).get("stale", {}).get("kl_vs_oneshot", {}).get("mean", 0.0)
    print(
        f"[DONE] Wrote {len(all_mode_records)} records -> {out_summary_path} "
        f"(newly_ran={newly_ran}, skipped={skipped}) | "
        f"G1a_pass={ov.get('g1a_pass_count', 0)}/{ov.get('n', 0)} | "
        f"mean_KL: jlocal={jl_kl:.4f}, jlocal_iso={iso_kl:.4f}, stale={st_kl:.4f} | "
        f"wall={summary['run_metadata']['wall_clock_s']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
