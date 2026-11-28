"""
Wave 10 / W10-4 (P1-3 + CounterFact):
  1. E4 (`Delta-RoPE`): Variable-length edits (`delta_L > 0` 1->3/4 tok and `delta_L < 0` 4->1 tok),
     isolating `jlocal_delta_rope` vs `jlocal_no_delta_rope` vs `kv_only_delta_rope` vs `suffix_recompute`.
  2. E7 (`causal_append`): 3 sequential overwrites (`v0 -> v1 -> v2 -> v3`) of the same entity,
     comparing `O_local` in-place multi-turn edits vs naive `causal_append` (EM, stale leakage across
     v0/v1/v2, KL vs fresh oracle, and latency/token bloat).
  3. E11 (`CounterFact` naturalistic evaluation): Real cases from `counterfact.json` with
     natural multi-token objects (handling `delta_L != 0` via `Delta-RoPE`), plus held-out
     `paraphrase_prompts` (generalization) and `neighborhood_prompts` (collateral specificity).
"""

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import copy
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from w5_9b_e5_runner import (
    FILLERS,
    SYSTEM_PROMPT,
    GDNSuffixHookCollector,
    append_record_jsonl,
    apply_chat_no_think,
    apply_jlocal_scalar_update,
    bootstrap_mean_ci,
    build_digit4_pool,
    clone_cache,
    compute_state_delta_fp64,
    count_linear_layers,
    eval_from_ctx_cache_in_place,
    extract_kv_slice,
    find_value_token_span,
    format_fact,
    format_query,
    get_eos_ids,
    has_logits_to_keep_arg,
    make_position_ids,
    move_cache_device,
    paired_comparison,
    patch_kv_slice,
    restore_ctx_cache_in_place,
    run_chunked_prefill,
    score_condition,
    snap_states,
)
from w10_p1_1_read_evict import eval_from_ctx_cache_with_mask


def get_rotary_emb(base_model):
    tm = getattr(base_model, "language_model", base_model)
    return tm.rotary_emb


@torch.no_grad()
def apply_delta_rope_to_kv_cache(
    cache,
    new_blk_kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    p0: int,
    b_old: int,
    b_new: int,
    T_ctx_old: int,
    delta_L: int,
    base_model,
    apply_delta_rope: bool = True,
):
    """
    Splice variable-length KV cache:
      [0, p0) from old cache + [p0, b_new) from new_blk_kv + [b_old, T_ctx_old) shifted by `delta_L`
      using exact `Qwen3_5TextRotaryEmbedding` on `k[..., :rotary_dim]` (`rotary_dim=64`) only!
    Mutates `cache` layer keys/values cleanly in-place to length `T_ctx_new = T_ctx_old + delta_L`.
    """
    rotary_emb = get_rotary_emb(base_model)
    first_k = None
    for layer in cache.layers:
        k = getattr(layer, "keys", None)
        if torch.is_tensor(k) and k.numel() > 0:
            first_k = k
            break
    if first_k is None:
        return

    cos_d, sin_d = None, None
    if apply_delta_rope and delta_L != 0 and b_old < T_ctx_old:
        pos_delta = make_position_ids(delta_L, 1, first_k.device, base_model)
        cos_d, sin_d = rotary_emb(first_k.to(dtype=torch.float32), pos_delta)
        cos_d = cos_d.float()
        sin_d = sin_d.float()
        if int(cos_d.shape[-1]) != 64:
            raise AssertionError(f"Expected rotary_dim=64, got cos_d.shape[-1]={cos_d.shape[-1]}")

    T_ctx_new = T_ctx_old + delta_L
    for i, item in enumerate(new_blk_kv):
        layer = cache.layers[i]
        k_old = getattr(layer, "keys", None)
        v_old = getattr(layer, "values", None)
        if not (torch.is_tensor(k_old) and torch.is_tensor(v_old) and k_old.numel() > 0):
            continue
        assert item is not None
        k_blk, v_blk = item
        k_pre = k_old[:, :, :p0, :]
        v_pre = v_old[:, :, :p0, :]

        if b_old < T_ctx_old:
            k_suf = k_old[:, :, b_old:T_ctx_old, :]
            v_suf = v_old[:, :, b_old:T_ctx_old, :]
            if apply_delta_rope and delta_L != 0:
                k_suf_f = k_suf.float()
                _, k_suf_rot = apply_rotary_pos_emb(
                    k_suf_f, k_suf_f, cos_d, sin_d, unsqueeze_dim=1
                )
                k_suf = k_suf_rot.to(dtype=k_old.dtype)
            layer.keys = torch.cat([k_pre, k_blk.to(device=k_old.device, dtype=k_old.dtype), k_suf], dim=-2).contiguous()
            layer.values = torch.cat([v_pre, v_blk.to(device=v_old.device, dtype=v_old.dtype), v_suf], dim=-2).contiguous()
        else:
            layer.keys = torch.cat([k_pre, k_blk.to(device=k_old.device, dtype=k_old.dtype)], dim=-2).contiguous()
            layer.values = torch.cat([v_pre, v_blk.to(device=v_old.device, dtype=v_old.dtype)], dim=-2).contiguous()

    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = T_ctx_new


# ============================================================================
# Part 1: E4 Variable-Length + Delta-RoPE
# ============================================================================

VARLEN_POS_PHRASES = [
    "silver falcon crest", "amber canyon gate", "cobalt harbor key", "crimson summit pass",
    "emerald lantern seal", "ivory compass vault", "obsidian ridge code", "sapphire beacon node",
]
VARLEN_NEG_TARGETS = ["oak", "elm", "ash", "fir", "sky", "gem", "bay", "fox", "orb", "sun"]


def build_e4_sample(tok, s_idx: int, direction: str, n_linear: int, device: str, rng: random.Random) -> Optional[Dict[str, Any]]:
    """
    Build variable-length edit case:
      - `pos`: `1-digit -> 3+ word phrase` (`delta_L > 0`, empirically `+2` tokens)
      - `neg`: `4-digit -> 1 word` (`delta_L < 0`, empirically `-4` tokens)
    Ensures exact post-edit suffix alignment `ids_o[b_old:] == ids_e[b_new:]`.
    """
    target = ["Alice", "Bob", "Carol", "Dave"][s_idx % 4]
    if direction == "pos":
        old_val = str(rng.randint(1, 9))
        new_val = VARLEN_POS_PHRASES[s_idx % len(VARLEN_POS_PHRASES)]
    else:
        old_val = f"{rng.randint(1,9)}{rng.randint(0,9)}{rng.randint(0,9)}{rng.randint(1,9)}"
        new_val = VARLEN_NEG_TARGETS[s_idx % len(VARLEN_NEG_TARGETS)]

    fact_o = f"The assigned access code for {target} is {old_val}."
    fact_e = f"The assigned access code for {target} is {new_val}."
    query_str = f"Question: What is the assigned access code for {target}? Output only the exact access code."

    pre_sents = [
        f"The assigned access code for {['Bob','Carol','Dave','Alice'][s_idx % 4]} is {rng.randint(1000,9999)}.",
        FILLERS[s_idx % len(FILLERS)],
        FILLERS[(s_idx + 1) % len(FILLERS)],
    ]
    suf_sents = [FILLERS[(s_idx + 2 + j) % len(FILLERS)] for j in range(14)]
    user_o = " ".join(pre_sents) + f" {fact_o} " + " ".join(suf_sents) + f"\n\n{query_str}"
    user_e = " ".join(pre_sents) + f" {fact_e} " + " ".join(suf_sents) + f"\n\n{query_str}"

    full_o = apply_chat_no_think(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_o}])
    full_e = apply_chat_no_think(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_e}])
    qo, qe = full_o.rfind(query_str), full_e.rfind(query_str)
    if qo < 0 or qe < 0:
        return None
    ctx_o, ctx_e, q_str = full_o[:qo], full_e[:qe], full_o[qo:]
    enc_o = tok(ctx_o, return_offsets_mapping=True, add_special_tokens=False)
    enc_e = tok(ctx_e, return_offsets_mapping=True, add_special_tokens=False)
    enc_q = tok(q_str, add_special_tokens=False)
    ids_o, ids_e, ids_q = enc_o["input_ids"], enc_e["input_ids"], enc_q["input_ids"]

    fo, fe = ctx_o.find(fact_o), ctx_e.find(fact_e)
    if fo < 0 or fe < 0:
        return None
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
    if p1_o <= p0 or p1_e <= p0 or delta_L == 0 or ids_o[:p0] != ids_e[:p0] or ids_o[p1_o:] != ids_e[p1_e:]:
        return None

    T_ctx_old = len(ids_o)
    T_ctx_new = len(ids_e)
    b_old = min(T_ctx_old, p1_o + 3 * (n_linear - 1))
    b_new = min(T_ctx_new, p1_e + 3 * (n_linear - 1))
    if ids_o[b_old:] != ids_e[b_new:] or (b_new - b_old) != delta_L:
        return None

    f_o = tok.encode(old_val, add_special_tokens=False)[0]
    f_e = tok.encode(new_val, add_special_tokens=False)[0]
    if f_o == f_e:
        return None

    return {
        "direction": direction,
        "sample_index": s_idx,
        "old_val": old_val,
        "new_val": new_val,
        "f_o": int(f_o),
        "f_e": int(f_e),
        "p0": p0,
        "p1_o": p1_o,
        "p1_e": p1_e,
        "b_old": int(b_old),
        "b_new": int(b_new),
        "delta_L": int(delta_L),
        "T_ctx_old": int(T_ctx_old),
        "T_ctx_new": int(T_ctx_new),
        "ids_o": torch.tensor([ids_o], dtype=torch.long, device=device),
        "ids_e": torch.tensor([ids_e], dtype=torch.long, device=device),
        "ids_q": torch.tensor([ids_q], dtype=torch.long, device=device),
    }


def _extract_full_attn_suffix_k(cache, start: int, end: int) -> List[torch.Tensor]:
    out = []
    for layer in cache.layers:
        k = getattr(layer, "keys", None)
        if torch.is_tensor(k) and k.numel() > 0:
            out.append(k[:, :, start:end, :].detach().float().cpu())
    return out


@torch.no_grad()
def run_e4_sample(model, base_model, tok, eos_ids, s: Dict[str, Any], n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool, audit_kspace: bool = False) -> Dict[str, Any]:
    ids_o, ids_e, ids_q = s["ids_o"], s["ids_e"], s["ids_q"]
    p0, b_old, b_new = s["p0"], s["b_old"], s["b_new"]
    T_old, T_new, delta_L = s["T_ctx_old"], s["T_ctx_new"], s["delta_L"]
    suf_len = T_old - b_old
    old_val, new_val, f_o, f_e = s["old_val"], s["new_val"], s["f_o"], s["f_e"]

    _, c_one = run_chunked_prefill(model, base_model, ids_e, 0, None, 2048, use_l2k)
    oracle_suf_k = _extract_full_attn_suffix_k(c_one, b_new, T_new) if (audit_kspace and suf_len > 0) else None
    ev_one = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_one, T_new, ids_q, use_l2k)
    lg_one = ev_one["first_logits"].detach().clone()
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
    _, c_new_b = run_chunked_prefill(model, base_model, ids_e[:, p0:b_new], p0, c_pre_gpu, 2048, use_l2k)
    del c_pre_gpu
    s_new_b = snap_states(c_new_b)
    new_blk_kv = extract_kv_slice(c_new_b, p0, b_new)
    delta = compute_state_delta_fp64(s_old_b, s_new_b)

    # Split recompute upper bound from `c_new_b` + `ids_e[:, b_new:T_new]`
    if suf_len > 0:
        _, c_split = run_chunked_prefill(model, base_model, ids_e[:, b_new:T_new], b_new, c_new_b, 2048, use_l2k)
    else:
        c_split = c_new_b
    ev_recomp = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_split, T_new, ids_q, use_l2k)
    del c_split

    # Old context + suffix hooks
    c_ctx_old = move_cache_device(c_old_b_cpu, device)
    del c_old_b_cpu
    with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as collector:
        if suf_len > 0:
            _, c_ctx_old = run_chunked_prefill(model, base_model, ids_o[:, b_old:T_old], b_old, c_ctx_old, 2048, use_l2k)
    hook_stats = collector.finalize(expected_suffix_len=suf_len, device=device)

    # Cond A: `jlocal_delta_rope` (O_local delta + Delta-RoPE rotated suffix KV)
    c_dr = clone_cache(c_ctx_old)
    apply_jlocal_scalar_update(c_dr, delta, hook_stats["gamma_scalar"], b_old, T_old, w_conv, use_decay=True)
    apply_delta_rope_to_kv_cache(c_dr, new_blk_kv, p0, b_old, b_new, T_old, delta_L, base_model, apply_delta_rope=True)
    dr_suf_k = _extract_full_attn_suffix_k(c_dr, b_new, T_new) if oracle_suf_k is not None else None
    ev_dr = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_dr, T_new, ids_q, use_l2k)
    del c_dr

    # Cond B: `jlocal_no_delta_rope` (O_local delta + unrotated suffix KV splice)
    c_ndr = clone_cache(c_ctx_old)
    apply_jlocal_scalar_update(c_ndr, delta, hook_stats["gamma_scalar"], b_old, T_old, w_conv, use_decay=True)
    apply_delta_rope_to_kv_cache(c_ndr, new_blk_kv, p0, b_old, b_new, T_old, delta_L, base_model, apply_delta_rope=False)
    ndr_suf_k = _extract_full_attn_suffix_k(c_ndr, b_new, T_new) if oracle_suf_k is not None else None
    ev_ndr = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_ndr, T_new, ids_q, use_l2k)
    del c_ndr

    kspace_metrics = None
    if oracle_suf_k is not None and dr_suf_k is not None and ndr_suf_k is not None:
        rot_dr_max, rot_no_max, pass_dr_max, pass_no_max, full_dr_max, full_no_max = [], [], [], [], [], []
        for ko, kd, kn in zip(oracle_suf_k, dr_suf_k, ndr_suf_k):
            d_dr = (kd - ko).abs()
            d_no = (kn - ko).abs()
            rot_dr_max.append(float(d_dr[..., :64].max().item()))
            rot_no_max.append(float(d_no[..., :64].max().item()))
            full_dr_max.append(float(d_dr.max().item()))
            full_no_max.append(float(d_no.max().item()))
            if ko.shape[-1] > 64:
                pass_dr_max.append(float(d_dr[..., 64:].max().item()))
                pass_no_max.append(float(d_no[..., 64:].max().item()))
        mean_dr = sum(rot_dr_max) / max(1, len(rot_dr_max))
        mean_no = sum(rot_no_max) / max(1, len(rot_no_max))
        kspace_metrics = {
            "n_full_layers": len(oracle_suf_k),
            "rotary_dim": 64,
            "rotary_0_64_max_dr_mean_across_layers": mean_dr,
            "rotary_0_64_max_no_mean_across_layers": mean_no,
            "rotary_error_reduction_ratio": float(mean_no / max(1e-12, mean_dr)),
            "pass_64_end_max_dr_mean": float(sum(pass_dr_max) / max(1, len(pass_dr_max))) if pass_dr_max else 0.0,
            "pass_64_end_max_no_mean": float(sum(pass_no_max) / max(1, len(pass_no_max))) if pass_no_max else 0.0,
            "full_head_max_dr_mean": float(sum(full_dr_max) / max(1, len(full_dr_max))),
            "full_head_max_no_mean": float(sum(full_no_max) / max(1, len(full_no_max))),
        }

    # Cond C: `kv_only_delta_rope` (stale linear states + Delta-RoPE KV splice)
    c_kv_dr = clone_cache(c_ctx_old)
    apply_delta_rope_to_kv_cache(c_kv_dr, new_blk_kv, p0, b_old, b_new, T_old, delta_L, base_model, apply_delta_rope=True)
    ev_kv_dr = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_kv_dr, T_new, ids_q, use_l2k)
    del c_kv_dr

    del c_ctx_old, s_old_b, s_new_b, delta, new_blk_kv, hook_stats
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    conds = {
        "oneshot_oracle": score_condition(ev_one, new_val, old_val, f_e, f_o, lg_one),
        "suffix_recompute": score_condition(ev_recomp, new_val, old_val, f_e, f_o, lg_one),
        "jlocal_delta_rope": score_condition(ev_dr, new_val, old_val, f_e, f_o, lg_one),
        "jlocal_no_delta_rope": score_condition(ev_ndr, new_val, old_val, f_e, f_o, lg_one),
        "kv_only_delta_rope": score_condition(ev_kv_dr, new_val, old_val, f_e, f_o, lg_one),
    }
    return {
        "task_group": "e4_delta_rope",
        "direction": s["direction"],
        "delta_L": delta_L,
        "sample_index": s["sample_index"],
        "old_val": old_val,
        "new_val": new_val,
        "p0": p0,
        "p1_o": s["p1_o"],
        "p1_e": s["p1_e"],
        "b_old": b_old,
        "b_new": b_new,
        "T_ctx_old": T_old,
        "T_ctx_new": T_new,
        "suffix_len": suf_len,
        "kspace": kspace_metrics,
        "oracle_ok": bool(conds["oneshot_oracle"]["contains_new"]),
        "conditions": conds,
    }


# ============================================================================
# Part 2: E7 Multi-Turn Sequential Overwrites (`O_local` vs `causal_append`)
# ============================================================================

@torch.no_grad()
def run_e7_sample(model, base_model, tok, eos_ids, s_idx: int, n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool, pool_by_d: Dict[str, List[str]]) -> Dict[str, Any]:
    rng = random.Random(20260700 + s_idx)
    target = ["Alice", "Bob", "Carol", "Dave"][s_idx % 4]
    tpl_id = s_idx % 3
    fds = rng.sample([str(d) for d in range(1, 10)], 7)
    vals = [rng.choice(pool_by_d[d]) for d in fds]
    v0, v1, v2, v3 = vals[0], vals[1], vals[2], vals[3]
    dist_vals = vals[4:]
    other_names = [n for n in ["Alice", "Bob", "Carol", "Dave"] if n != target]

    pre_sents = [format_fact(tpl_id, other_names[0], dist_vals[0]), FILLERS[s_idx % len(FILLERS)], format_fact(tpl_id, other_names[1], dist_vals[1])]
    suf_sents = [format_fact(tpl_id, other_names[2], dist_vals[2])] + [FILLERS[(s_idx + 1 + j) % len(FILLERS)] for j in range(12)]
    query_str = format_query(tpl_id, target)

    def make_ctx(val: str):
        user = " ".join(pre_sents) + f" {format_fact(tpl_id, target, val)} " + " ".join(suf_sents) + f"\n\n{query_str}"
        full = apply_chat_no_think(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
        qpos = full.rfind(query_str)
        ctx_s, q_s = full[:qpos], full[qpos:]
        enc_c = tok(ctx_s, return_offsets_mapping=True, add_special_tokens=False)
        enc_q = tok(q_s, add_special_tokens=False)
        fact_s = format_fact(tpl_id, target, val)
        fc = ctx_s.find(fact_s)
        vo = fact_s.rfind(val)
        pos = find_value_token_span(ctx_s, val, fc + vo, enc_c["offset_mapping"])
        return enc_c["input_ids"], enc_q["input_ids"], int(pos[0]), int(pos[-1] + 1)

    ids0, q_ids, p0, p1 = make_ctx(v0)
    ids1, _, p0_1, p1_1 = make_ctx(v1)
    ids2, _, p0_2, p1_2 = make_ctx(v2)
    ids3, _, p0_3, p1_3 = make_ctx(v3)
    assert len(ids0) == len(ids1) == len(ids2) == len(ids3)
    assert p0 == p0_1 == p0_2 == p0_3 and p1 == p1_1 == p1_2 == p1_3
    T_ctx = len(ids0)
    b_end = min(T_ctx, p1 + 3 * (n_linear - 1))
    suf_len = T_ctx - b_end

    t0_ids = torch.tensor([ids0], dtype=torch.long, device=device)
    t3_ids = torch.tensor([ids3], dtype=torch.long, device=device)
    ids_q = torch.tensor([q_ids], dtype=torch.long, device=device)
    turn_tensors = [torch.tensor([x], dtype=torch.long, device=device) for x in (ids0, ids1, ids2, ids3)]

    f_v3 = int(tok.encode(v3, add_special_tokens=False)[0])
    f_v0 = int(tok.encode(v0, add_special_tokens=False)[0])

    # Fresh single-fact oracle for final value `v3`
    _, c_one3 = run_chunked_prefill(model, base_model, t3_ids, 0, None, 2048, use_l2k)
    ev_one3 = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_one3, T_ctx, ids_q, use_l2k)
    lg_one3 = ev_one3["first_logits"].detach().clone()
    del c_one3

    # Shared prefix + initial v0 prefill with suffix decay rates
    _, c_pre = run_chunked_prefill(model, base_model, t0_ids[:, :p0], 0, None, 2048, use_l2k)
    c_pre_cpu = move_cache_device(c_pre, "cpu")
    _, c_b_curr = run_chunked_prefill(model, base_model, t0_ids[:, p0:b_end], p0, c_pre, 2048, use_l2k)
    del c_pre
    s_b_curr = snap_states(c_b_curr)
    del c_b_curr

    c_curr, _ = None, None
    c_curr_b_init = move_cache_device(c_pre_cpu, device)
    _, c_curr = run_chunked_prefill(model, base_model, t0_ids[:, p0:b_end], p0, c_curr_b_init, 2048, use_l2k)
    with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as collector:
        if suf_len > 0:
            _, c_curr = run_chunked_prefill(model, base_model, t0_ids[:, b_end:T_ctx], b_end, c_curr, 2048, use_l2k)
    hook_stats = collector.finalize(expected_suffix_len=suf_len, device=device)

    # Also clone for causal_append, kv_only_blk_multiturn, state_only_multiturn baselines from identical `v0` cache
    c_app = clone_cache(c_curr)
    c_kvblk = clone_cache(c_curr)
    c_stonly = clone_cache(c_curr)
    c_neither = clone_cache(c_curr)

    # Apply 3 sequential in-place updates: v0 -> v1 -> v2 -> v3
    t_olocal_start = time.perf_counter()
    for step_t in (1, 2, 3):
        c_p = move_cache_device(c_pre_cpu, device)
        _, c_new_b = run_chunked_prefill(model, base_model, turn_tensors[step_t][:, p0:b_end], p0, c_p, 2048, use_l2k)
        del c_p
        s_b_new = snap_states(c_new_b)
        blk_kv = extract_kv_slice(c_new_b, p0, b_end)
        del c_new_b
        delta_t = compute_state_delta_fp64(s_b_curr, s_b_new)
        apply_jlocal_scalar_update(c_curr, delta_t, hook_stats["gamma_scalar"], b_end, T_ctx, w_conv, use_decay=True)
        patch_kv_slice(c_curr, blk_kv, p0, b_end)
        # 2x2 controls isolating block KV vs GDN state over 3 sequential overwrites
        patch_kv_slice(c_kvblk, blk_kv, p0, b_end)
        apply_jlocal_scalar_update(c_stonly, delta_t, hook_stats["gamma_scalar"], b_end, T_ctx, w_conv, use_decay=True)
        s_b_curr = s_b_new
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    olocal_3turns_ms = (time.perf_counter() - t_olocal_start) * 1000.0
    ev_olocal = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_curr, T_ctx, ids_q, use_l2k)
    ev_kvblk = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_kvblk, T_ctx, ids_q, use_l2k)
    ev_stonly = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_stonly, T_ctx, ids_q, use_l2k)
    ev_neither = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_neither, T_ctx, ids_q, use_l2k)
    del c_curr, c_kvblk, c_stonly, c_neither, c_pre_cpu, s_b_curr, hook_stats

    # Causal append baseline: append 3 update sentences onto `c_app`
    t_app_start = time.perf_counter()
    cur_pos = T_ctx
    for step_t, val_new in ((1, v1), (2, v2), (3, v3)):
        upd_str = f" Correction update {step_t}: the updated value for {target} is now {val_new}."
        upd_ids = tok(upd_str, add_special_tokens=False)["input_ids"]
        upd_t = torch.tensor([upd_ids], dtype=torch.long, device=device)
        _, c_app = run_chunked_prefill(model, base_model, upd_t, cur_pos, c_app, 2048, use_l2k)
        cur_pos += len(upd_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    causal_append_ms = (time.perf_counter() - t_app_start) * 1000.0
    ev_app = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_app, cur_pos, ids_q, use_l2k)
    del c_app

    sc_one3 = score_condition(ev_one3, v3, v0, f_v3, f_v0, lg_one3)
    sc_olocal = score_condition(ev_olocal, v3, v0, f_v3, f_v0, lg_one3)
    sc_kvblk = score_condition(ev_kvblk, v3, v0, f_v3, f_v0, lg_one3)
    sc_stonly = score_condition(ev_stonly, v3, v0, f_v3, f_v0, lg_one3)
    sc_neither = score_condition(ev_neither, v3, v0, f_v3, f_v0, lg_one3)
    sc_app = score_condition(ev_app, v3, v0, f_v3, f_v0, lg_one3)

    sc_olocal["stale_leak_any_prior"] = any(pv in ev_olocal["text"] for pv in (v0, v1, v2))
    sc_kvblk["stale_leak_any_prior"] = any(pv in ev_kvblk["text"] for pv in (v0, v1, v2))
    sc_stonly["stale_leak_any_prior"] = any(pv in ev_stonly["text"] for pv in (v0, v1, v2))
    sc_neither["stale_leak_any_prior"] = any(pv in ev_neither["text"] for pv in (v0, v1, v2))
    sc_app["stale_leak_any_prior"] = any(pv in ev_app["text"] for pv in (v0, v1, v2))

    return {
        "task_group": "e7_multiturn_append",
        "sample_index": s_idx,
        "T_ctx_olocal": T_ctx,
        "T_ctx_causal_append": cur_pos,
        "olocal_3turns_ms": olocal_3turns_ms,
        "causal_append_3turns_ms": causal_append_ms,
        "oracle_ok": bool(sc_one3["contains_new"]),
        "conditions": {
            "oneshot_oracle_v3": sc_one3,
            "olocal_multiturn": sc_olocal,
            "kv_only_blk_multiturn": sc_kvblk,
            "state_only_multiturn": sc_stonly,
            "stale_v0_neither": sc_neither,
            "causal_append": sc_app,
        },
    }


# ============================================================================
# Part 3: Naturalistic CounterFact (`counterfact.json`)
# ============================================================================

def load_and_prepare_counterfact(tok, cf_path: Path, max_scan: int, n_target: int, target_ctx_len: int, n_linear: int, device: str, rng: random.Random) -> List[Dict[str, Any]]:
    raw = json.loads(cf_path.read_text(encoding="utf-8"))
    prepared: List[Dict[str, Any]] = []
    for idx, row in enumerate(raw[:max_scan]):
        if len(prepared) >= n_target:
            break
        rw = row["requested_rewrite"]
        subj = rw["subject"]
        old_val = rw["target_true"]["str"].strip()
        new_val = rw["target_new"]["str"].strip()
        if not old_val or not new_val or old_val == new_val:
            continue
        prompt_prefix = rw["prompt"].format(subj).strip()
        fact_o = f"{prompt_prefix} {old_val}."
        fact_e = f"{prompt_prefix} {new_val}."

        paras = row.get("paraphrase_prompts", [])
        neighs = row.get("neighborhood_prompts", [])
        if not paras or not neighs:
            continue

        # Construct realistic Wikipedia/article distractor wrapper around the CounterFact statement
        filler_tokens_approx = 25
        n_suf = max(6, int((target_ctx_len - 120) / filler_tokens_approx))
        pre_txt = " ".join([FILLERS[idx % len(FILLERS)], FILLERS[(idx + 1) % len(FILLERS)]])
        suf_txt = " ".join([FILLERS[(idx + 2 + j) % len(FILLERS)] for j in range(n_suf)])

        q_direct = f"Complete this fact with only the answer entity (no explanation): {prompt_prefix}"
        q_para = f"Complete this fact with only the answer entity (no explanation): {paras[0].strip()}"
        q_neigh = f"Complete this fact with only the answer entity (no explanation): {neighs[0].strip()}"

        user_o = f"{pre_txt} {fact_o} {suf_txt}\n\n{q_direct}"
        user_e = f"{pre_txt} {fact_e} {suf_txt}\n\n{q_direct}"

        full_o = apply_chat_no_think(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_o}])
        full_e = apply_chat_no_think(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_e}])
        qo, qe = full_o.rfind(q_direct), full_e.rfind(q_direct)
        if qo < 0 or qe < 0:
            continue

        ctx_o, ctx_e, q_s = full_o[:qo], full_e[:qe], full_o[qo:]
        enc_o = tok(ctx_o, return_offsets_mapping=True, add_special_tokens=False)
        enc_e = tok(ctx_e, return_offsets_mapping=True, add_special_tokens=False)
        ids_o, ids_e = enc_o["input_ids"], enc_e["input_ids"]

        fo, fe = ctx_o.find(fact_o), ctx_e.find(fact_e)
        if fo < 0 or fe < 0:
            continue
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
            continue
        T_old, T_new = len(ids_o), len(ids_e)
        b_old = min(T_old, p1_o + 3 * (n_linear - 1))
        b_new = min(T_new, p1_e + 3 * (n_linear - 1))
        if ids_o[b_old:] != ids_e[b_new:] or (b_new - b_old) != delta_L:
            continue

        f_o = tok.encode(old_val, add_special_tokens=False)[0]
        f_e = tok.encode(new_val, add_special_tokens=False)[0]
        if f_o == f_e:
            continue

        q_dir_ids = tok(q_s, add_special_tokens=False)["input_ids"]
        q_para_ids = tok(full_o[qo:].replace(q_direct, q_para), add_special_tokens=False)["input_ids"]
        q_neigh_ids = tok(full_o[qo:].replace(q_direct, q_neigh), add_special_tokens=False)["input_ids"]

        prepared.append({
            "case_id": int(row["case_id"]),
            "relation_id": rw["relation_id"],
            "subject": subj,
            "old_val": old_val,
            "new_val": new_val,
            "f_o": int(f_o),
            "f_e": int(f_e),
            "p0": p0,
            "b_old": int(b_old),
            "b_new": int(b_new),
            "delta_L": int(delta_L),
            "T_old": int(T_old),
            "T_new": int(T_new),
            "ids_o": torch.tensor([ids_o], dtype=torch.long, device=device),
            "ids_e": torch.tensor([ids_e], dtype=torch.long, device=device),
            "ids_q": torch.tensor([q_dir_ids], dtype=torch.long, device=device),
            "ids_q_para": torch.tensor([q_para_ids], dtype=torch.long, device=device),
            "ids_q_neigh": torch.tensor([q_neigh_ids], dtype=torch.long, device=device),
        })
    return prepared


@torch.no_grad()
def run_counterfact_sample(model, base_model, tok, eos_ids, s: Dict[str, Any], n_linear: int, d_k: int, w_conv: int, device: str, use_l2k: bool, eval_recomp: bool = False) -> Dict[str, Any]:
    ids_o, ids_e = s["ids_o"], s["ids_e"]
    ids_q, ids_qp, ids_qn = s["ids_q"], s["ids_q_para"], s["ids_q_neigh"]
    p0, b_old, b_new = s["p0"], s["b_old"], s["b_new"]
    T_old, T_new, delta_L = s["T_old"], s["T_new"], s["delta_L"]
    suf_len = T_old - b_old
    old_val, new_val, f_o, f_e = s["old_val"], s["new_val"], s["f_o"], s["f_e"]

    _, c_one = run_chunked_prefill(model, base_model, ids_e, 0, None, 2048, use_l2k)
    ev_one = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_one), T_new, ids_q, use_l2k)
    ev_one_para = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_one), T_new, ids_qp, use_l2k)
    ev_one_neigh = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_one, T_new, ids_qn, use_l2k)
    lg_one = ev_one["first_logits"].detach().clone()
    lg_one_para = ev_one_para["first_logits"].detach().clone()
    lg_one_neigh = ev_one_neigh["first_logits"].detach().clone()
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
    _, c_new_b = run_chunked_prefill(model, base_model, ids_e[:, p0:b_new], p0, c_pre_gpu, 2048, use_l2k)
    del c_pre_gpu
    s_new_b = snap_states(c_new_b)
    new_blk_kv = extract_kv_slice(c_new_b, p0, b_new)
    delta = compute_state_delta_fp64(s_old_b, s_new_b)
    recomp_kl = None
    if eval_recomp:
        if suf_len > 0:
            _, c_sp = run_chunked_prefill(model, base_model, ids_e[:, b_new:T_new], b_new, c_new_b, 2048, use_l2k)
        else:
            c_sp = c_new_b
        ev_sp = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_sp, T_new, ids_q, use_l2k)
        recomp_kl = score_condition(ev_sp, new_val, old_val, f_e, f_o, lg_one)["kl_vs_oneshot"]
        del c_sp
    else:
        del c_new_b

    c_ctx_old = move_cache_device(c_old_b_cpu, device)
    del c_old_b_cpu
    with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as collector:
        if suf_len > 0:
            _, c_ctx_old = run_chunked_prefill(model, base_model, ids_o[:, b_old:T_old], b_old, c_ctx_old, 2048, use_l2k)
    hook_stats = collector.finalize(expected_suffix_len=suf_len, device=device)

    # old baseline (direct, paraphrase, neighbor)
    ev_old = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_ctx_old), T_old, ids_q, use_l2k)
    ev_old_neigh = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_ctx_old), T_old, ids_qn, use_l2k)

    # jlocal (with Delta-RoPE when delta_L != 0)
    c_jl = clone_cache(c_ctx_old)
    apply_jlocal_scalar_update(c_jl, delta, hook_stats["gamma_scalar"], b_old, T_old, w_conv, use_decay=True)
    apply_delta_rope_to_kv_cache(c_jl, new_blk_kv, p0, b_old, b_new, T_old, delta_L, base_model, apply_delta_rope=True)
    ev_jl = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_jl), T_new, ids_q, use_l2k)
    ev_jl_para = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_jl), T_new, ids_qp, use_l2k)
    ev_jl_neigh = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_jl, T_new, ids_qn, use_l2k)
    del c_jl

    # kv_only_patched (with Delta-RoPE if delta_L != 0)
    c_kv = clone_cache(c_ctx_old)
    apply_delta_rope_to_kv_cache(c_kv, new_blk_kv, p0, b_old, b_new, T_old, delta_L, base_model, apply_delta_rope=True)
    ev_kv = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_kv), T_new, ids_q, use_l2k)
    ev_kv_para = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, clone_cache(c_kv), T_new, ids_qp, use_l2k)
    ev_kv_neigh = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_kv, T_new, ids_qn, use_l2k)
    del c_kv

    # state_only (when delta_L == 0 clean exact length; if delta_L != 0 use c_ctx_old with updated states)
    c_so = clone_cache(c_ctx_old)
    apply_jlocal_scalar_update(c_so, delta, hook_stats["gamma_scalar"], b_old, T_old, w_conv, use_decay=True)
    ev_so = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_so, T_old, ids_q, use_l2k)
    del c_so

    del c_ctx_old, s_old_b, s_new_b, delta, new_blk_kv, hook_stats
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    neigh_oracle_txt = ev_one_neigh["text"]
    neigh_old_txt = ev_old_neigh["text"]

    conds = {
        "oneshot_oracle": score_condition(ev_one, new_val, old_val, f_e, f_o, lg_one),
        "jlocal": score_condition(ev_jl, new_val, old_val, f_e, f_o, lg_one),
        "kv_only": score_condition(ev_kv, new_val, old_val, f_e, f_o, lg_one),
        "state_only": score_condition(ev_so, new_val, old_val, f_e, f_o, lg_one),
        "old": score_condition(ev_old, new_val, old_val, f_e, f_o, lg_one),
    }
    conds["jlocal"]["para_contains_new"] = bool(new_val.lower() in ev_jl_para["text"].lower())
    conds["jlocal"]["para_kl_vs_oracle"] = score_condition(ev_jl_para, new_val, old_val, f_e, f_o, lg_one_para)["kl_vs_oneshot"]
    conds["jlocal"]["neigh_preserved_vs_old"] = bool(ev_jl_neigh["text"] == neigh_old_txt)
    conds["jlocal"]["neigh_preserved_vs_oracle"] = bool(ev_jl_neigh["text"] == neigh_oracle_txt)
    conds["jlocal"]["neigh_kl_vs_oracle"] = score_condition(ev_jl_neigh, new_val, old_val, f_e, f_o, lg_one_neigh)["kl_vs_oneshot"]

    conds["kv_only"]["para_contains_new"] = bool(new_val.lower() in ev_kv_para["text"].lower())
    conds["kv_only"]["para_kl_vs_oracle"] = score_condition(ev_kv_para, new_val, old_val, f_e, f_o, lg_one_para)["kl_vs_oneshot"]
    conds["kv_only"]["neigh_preserved_vs_old"] = bool(ev_kv_neigh["text"] == neigh_old_txt)
    conds["kv_only"]["neigh_preserved_vs_oracle"] = bool(ev_kv_neigh["text"] == neigh_oracle_txt)
    conds["kv_only"]["neigh_kl_vs_oracle"] = score_condition(ev_kv_neigh, new_val, old_val, f_e, f_o, lg_one_neigh)["kl_vs_oneshot"]

    conds["oneshot_oracle"]["para_contains_new"] = bool(new_val.lower() in ev_one_para["text"].lower())
    conds["oneshot_oracle"]["neigh_preserved_vs_old"] = bool(neigh_oracle_txt == neigh_old_txt)

    return {
        "task_group": "e11_counterfact",
        "case_id": s["case_id"],
        "relation_id": s["relation_id"],
        "delta_L": delta_L,
        "T_old": T_old,
        "T_new": T_new,
        "recomp_floor_kl": recomp_kl,
        "oracle_flip_ok": bool(conds["oneshot_oracle"]["contains_new"] and not conds["old"]["contains_new"]),
        "conditions": conds,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    p.add_argument("--tag", type=str, default="2b")
    p.add_argument("--n-e4", type=int, default=60)
    p.add_argument("--n-e7", type=int, default=60)
    p.add_argument("--n-cf", type=int, default=150)
    p.add_argument("--cf-path", type=str, default=r"D:\applicationData\HuggingFace\counterfact.json")
    p.add_argument("--out-dir", type=str, default="pilot")
    p.add_argument("--tasks", type=str, default="e4,e7,cf")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()

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

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(202612)
    pool_by_d = build_digit4_pool()
    summary_path = out_dir / f"w10_p1_3_cf_{args.tag}_summary.json"
    if summary_path.exists() and tasks != {"e4", "e7", "cf"}:
        summary: Dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["model"] = args.model
    else:
        summary = {"model": args.model}

    # --- E4 ---
    if "e4" in tasks:
        e4_recs = []
        for idx in range(args.n_e4):
            d = "pos" if idx % 2 == 0 else "neg"
            samp = build_e4_sample(tok, idx, d, n_linear, device, rng)
            if samp is None:
                continue
            rec = run_e4_sample(model, base_model, tok, eos_ids, samp, n_linear, d_k, w_conv, device, use_l2k, audit_kspace=(idx < 20))
            e4_recs.append(rec)
        (out_dir / f"w10_e4_{args.tag}_records.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in e4_recs) + "\n", encoding="utf-8"
        )
        e4_sum = {"n_total": len(e4_recs), "n_pos": sum(1 for r in e4_recs if r["delta_L"] > 0), "n_neg": sum(1 for r in e4_recs if r["delta_L"] < 0)}
        ks_recs = [r["kspace"] for r in e4_recs if r.get("kspace") is not None]
        if ks_recs:
            agg_dr = sum(k["rotary_0_64_max_dr_mean_across_layers"] for k in ks_recs) / len(ks_recs)
            agg_no = sum(k["rotary_0_64_max_no_mean_across_layers"] for k in ks_recs) / len(ks_recs)
            fh_dr = sum(k["full_head_max_dr_mean"] for k in ks_recs) / len(ks_recs)
            fh_no = sum(k["full_head_max_no_mean"] for k in ks_recs) / len(ks_recs)
            e4_sum["kspace_diagnostic"] = {
                "n_audited": len(ks_recs),
                "rotary_dim": ks_recs[0]["rotary_dim"],
                "rotary_0_64_max_dr_mean": agg_dr,
                "rotary_0_64_max_no_mean": agg_no,
                "aggregate_rotary_ratio": float(agg_no / max(1e-12, agg_dr)),
                "mean_of_per_sample_rotary_ratios": sum(k["rotary_error_reduction_ratio"] for k in ks_recs) / len(ks_recs),
                "pass_64_end_max_dr_mean": sum(k["pass_64_end_max_dr_mean"] for k in ks_recs) / len(ks_recs),
                "pass_64_end_max_no_mean": sum(k["pass_64_end_max_no_mean"] for k in ks_recs) / len(ks_recs),
                "full_head_max_dr_mean": fh_dr,
                "full_head_max_no_mean": fh_no,
                "aggregate_full_head_ratio": float(fh_no / max(1e-12, fh_dr)),
            }
        for cn in ("oneshot_oracle", "suffix_recompute", "jlocal_delta_rope", "jlocal_no_delta_rope", "kv_only_delta_rope"):
            e4_sum[cn] = {
                "contains_new_rate": float(sum(1 for r in e4_recs if r["conditions"][cn]["contains_new"]) / max(1, len(e4_recs))),
                "top1_eq_oracle_rate": float(sum(1 for r in e4_recs if r["conditions"][cn]["top1_eq_oracle"]) / max(1, len(e4_recs))),
                "kl_mean": float(sum(r["conditions"][cn]["kl_vs_oneshot"] for r in e4_recs) / max(1, len(e4_recs))),
            }
        e4_sum["paired_delta_rope_vs_no"] = paired_comparison(e4_recs, "jlocal_delta_rope", "jlocal_no_delta_rope")
        summary["e4"] = e4_sum
        print("E4 SUMMARY:", json.dumps(e4_sum, indent=2))

    # --- E7 ---
    if "e7" in tasks:
        e7_recs = [
            run_e7_sample(model, base_model, tok, eos_ids, idx, n_linear, d_k, w_conv, device, use_l2k, pool_by_d)
            for idx in range(args.n_e7)
        ]
        (out_dir / f"w10_e7_{args.tag}_records.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in e7_recs) + "\n", encoding="utf-8"
        )
        e7_sum = {
            "n_total": len(e7_recs),
            "olocal_multiturn": {
                "em_rate": float(sum(1 for r in e7_recs if r["conditions"]["olocal_multiturn"]["contains_new"]) / max(1, len(e7_recs))),
                "stale_leak_rate": float(sum(1 for r in e7_recs if r["conditions"]["olocal_multiturn"]["stale_leak_any_prior"]) / max(1, len(e7_recs))),
                "kl_mean": float(sum(r["conditions"]["olocal_multiturn"]["kl_vs_oneshot"] for r in e7_recs) / max(1, len(e7_recs))),
                "mean_ms": float(sum(r["olocal_3turns_ms"] for r in e7_recs) / max(1, len(e7_recs))),
                "mean_T_ctx": float(sum(r["T_ctx_olocal"] for r in e7_recs) / max(1, len(e7_recs))),
            },
            "kv_only_blk_multiturn": {
                "em_rate": float(sum(1 for r in e7_recs if r["conditions"]["kv_only_blk_multiturn"]["contains_new"]) / max(1, len(e7_recs))),
                "stale_leak_rate": float(sum(1 for r in e7_recs if r["conditions"]["kv_only_blk_multiturn"]["stale_leak_any_prior"]) / max(1, len(e7_recs))),
                "kl_mean": float(sum(r["conditions"]["kv_only_blk_multiturn"]["kl_vs_oneshot"] for r in e7_recs) / max(1, len(e7_recs))),
            },
            "state_only_multiturn": {
                "em_rate": float(sum(1 for r in e7_recs if r["conditions"]["state_only_multiturn"]["contains_new"]) / max(1, len(e7_recs))),
                "stale_leak_rate": float(sum(1 for r in e7_recs if r["conditions"]["state_only_multiturn"]["stale_leak_any_prior"]) / max(1, len(e7_recs))),
                "kl_mean": float(sum(r["conditions"]["state_only_multiturn"]["kl_vs_oneshot"] for r in e7_recs) / max(1, len(e7_recs))),
            },
            "stale_v0_neither": {
                "em_rate": float(sum(1 for r in e7_recs if r["conditions"]["stale_v0_neither"]["contains_new"]) / max(1, len(e7_recs))),
                "stale_leak_rate": float(sum(1 for r in e7_recs if r["conditions"]["stale_v0_neither"]["stale_leak_any_prior"]) / max(1, len(e7_recs))),
                "kl_mean": float(sum(r["conditions"]["stale_v0_neither"]["kl_vs_oneshot"] for r in e7_recs) / max(1, len(e7_recs))),
            },
            "causal_append": {
                "em_rate": float(sum(1 for r in e7_recs if r["conditions"]["causal_append"]["contains_new"]) / max(1, len(e7_recs))),
                "stale_leak_rate": float(sum(1 for r in e7_recs if r["conditions"]["causal_append"]["stale_leak_any_prior"]) / max(1, len(e7_recs))),
                "kl_mean": float(sum(r["conditions"]["causal_append"]["kl_vs_oneshot"] for r in e7_recs) / max(1, len(e7_recs))),
                "mean_ms": float(sum(r["causal_append_3turns_ms"] for r in e7_recs) / max(1, len(e7_recs))),
                "mean_T_ctx": float(sum(r["T_ctx_causal_append"] for r in e7_recs) / max(1, len(e7_recs))),
            },
            "paired_olocal_vs_append": paired_comparison(e7_recs, "olocal_multiturn", "causal_append"),
            "paired_olocal_vs_kv_blk": paired_comparison(e7_recs, "olocal_multiturn", "kv_only_blk_multiturn"),
        }
        summary["e7"] = e7_sum
        print("E7 SUMMARY:", json.dumps(e7_sum, indent=2))

    # --- CounterFact ---
    if "cf" in tasks:
        cf_cases = load_and_prepare_counterfact(tok, Path(args.cf_path), max_scan=1500, n_target=args.n_cf, target_ctx_len=512, n_linear=n_linear, device=device, rng=rng)
        cf_recs = [
            run_counterfact_sample(model, base_model, tok, eos_ids, s, n_linear, d_k, w_conv, device, use_l2k, eval_recomp=(i < 20))
            for i, s in enumerate(cf_cases)
        ]
        (out_dir / f"w10_cf_{args.tag}_records.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in cf_recs) + "\n", encoding="utf-8"
        )
        valid_cf = [r for r in cf_recs if r["oracle_flip_ok"]]
        recomp_vals = [float(r["recomp_floor_kl"]) for r in valid_cf if r.get("recomp_floor_kl") is not None]
        cf_sum = {
            "n_prepared": len(cf_recs),
            "n_oracle_flip_ok": len(valid_cf),
            "n_nonzero_delta_L": sum(1 for r in cf_recs if r["delta_L"] != 0),
            "suffix_recompute_floor_kl_mean": (sum(recomp_vals) / len(recomp_vals)) if recomp_vals else None,
            "suffix_recompute_floor_n": len(recomp_vals),
        }
        for cn in ("oneshot_oracle", "jlocal", "kv_only", "state_only", "old"):
            sub = valid_cf if valid_cf else cf_recs
            entry = {
                "contains_new_rate": float(sum(1 for r in sub if r["conditions"][cn]["contains_new"]) / max(1, len(sub))),
                "top1_eq_oracle_rate": float(sum(1 for r in sub if r["conditions"][cn]["top1_eq_oracle"]) / max(1, len(sub))),
                "kl_mean": float(sum(r["conditions"][cn]["kl_vs_oneshot"] for r in sub) / max(1, len(sub))),
            }
            if cn in ("jlocal", "kv_only"):
                entry["para_contains_new_rate"] = float(sum(1 for r in sub if r["conditions"][cn]["para_contains_new"]) / max(1, len(sub)))
                entry["para_kl_mean"] = float(sum(r["conditions"][cn]["para_kl_vs_oracle"] for r in sub) / max(1, len(sub)))
                entry["neigh_preserved_rate"] = float(sum(1 for r in sub if r["conditions"][cn]["neigh_preserved_vs_old"]) / max(1, len(sub)))
                entry["neigh_preserved_oracle_rate"] = float(sum(1 for r in sub if r["conditions"][cn]["neigh_preserved_vs_oracle"]) / max(1, len(sub)))
                entry["neigh_kl_mean"] = float(sum(r["conditions"][cn]["neigh_kl_vs_oracle"] for r in sub) / max(1, len(sub)))
            cf_sum[cn] = entry
        if valid_cf:
            cf_sum["paired_jlocal_vs_kv_only"] = paired_comparison(valid_cf, "jlocal", "kv_only")
        summary["counterfact"] = cf_sum
        print("CF SUMMARY:", json.dumps(cf_sum, indent=2))

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
