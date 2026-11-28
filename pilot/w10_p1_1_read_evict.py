"""
Wave 10 / P1-1: Read-Side Hard Attention-Mask KV Eviction vs. Soft-Zero Ablation (`read_kv_evict`)
====================================================================================================
Resolves the Pilot 3/4 `soft-zero (KV=0)` vs `What Attention Recalls` (2609.04434) discrepancy on
unmodified prompts across both `Qwen/Qwen3.5-2B` (RTX 4060, 0 CU) and `Qwen/Qwen3.5-9B` (Colab L4).

Tests 8 paired read-side conditions on the identical shared unmodified context cache `c_ctx` [0, T_ctx):
  1. `full_read`           : unmodified cache + unmasked attention (baseline EM & p_old).
  2. `soft_zero_val`       : zero keys/values in value digits [v0, v1), attention_mask=1 (Pilot 3/4 legacy).
  3. `soft_zero_clause`    : zero keys/values in entire fact clause [c0, c1), attention_mask=1.
  4. `read_kv_evict_val`   : hard attention_mask[:, v0:v1] = 0 (keys/vals untouched, seq_len preserved).
  5. `read_kv_evict_clause`: hard attention_mask[:, c0:c1] = 0 (symmetric counterpart to edit evict).
  6. `read_kv_evict_block` : hard attention_mask[:, c0:b_end] = 0 (fact clause + linear deep horizon).
  7. `read_kv_evict_ctx`   : hard attention_mask[:, 0:T_ctx] = 0 (pure recurrent state read over query).
  8. `state_zero_read`     : zero recurrent_states & conv_states at T_ctx, full KV intact (KV-only read).

Reuses prompt bank, chunked prefill, cache helpers, and stats from `w5_9b_e5_runner.py`.
"""

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from w5_9b_e5_runner import (
    FILLERS,
    SYSTEM_PROMPT,
    append_record_jsonl,
    apply_chat_no_think,
    bootstrap_mean_ci,
    clone_cache,
    count_linear_layers,
    exact_sign_test_two_sided_p,
    extract_kv_slice,
    find_value_token_span,
    format_fact,
    format_query,
    generate_e5_sample_bank,
    get_eos_ids,
    has_logits_to_keep_arg,
    load_deduplicated_jsonl,
    make_position_ids,
    overwrite_states,
    patch_kv_slice,
    run_chunked_prefill,
    snap_states,
    truncate_cache_kv,
    wilcoxon_signed_rank_two_sided_p,
    assert_cache_bitwise_equal,
)


def find_subspan_tokens(
    text: str,
    substr: str,
    start_hint: int,
    offsets: List[Tuple[int, int]],
) -> Tuple[int, int]:
    """Return inclusive-exclusive `[tok_start, tok_end)` covering `substr` in `text`."""
    c0 = text.find(substr, max(0, start_hint - 4))
    if c0 < 0:
        raise ValueError(f"Could not find substring {substr!r} near char {start_hint}")
    c1 = c0 + len(substr)
    pos = [i for i, (s, e) in enumerate(offsets) if e > c0 and s < c1 and e > s]
    if not pos:
        raise ValueError(f"Empty token span for {substr!r} at [{c0}, {c1})")
    return int(pos[0]), int(pos[-1] + 1)


def build_unmodified_read_prompt(
    tok,
    sample: Dict[str, Any],
    target_length: int,
    n_linear: int,
    device: str,
) -> Dict[str, Any]:
    """
    Construct unmodified S-NIAH prompt matching `w5_9b_e5_runner.build_e5_prompt`, plus exact token spans
    for value digits `[v0, v1)`, full fact sentence `[c0, c1)`, and linear block horizon `b_end`.
    """
    tpl_id = int(sample["tpl_id"])
    target = sample["target"]
    target_val = sample["old"]
    depth = float(sample["depth"])
    filler_shift = int(sample.get("filler_shift", sample["sample_index"]))
    distractors = sample["distractors"]

    target_fact = format_fact(tpl_id, target, target_val)
    dist_sents = [format_fact(tpl_id, d["subject"], d["val"]) for d in distractors]
    query_str = format_query(tpl_id, target)

    filler_tok_lens = [len(tok.encode(s + " ", add_special_tokens=False)) for s in FILLERS]
    avg_filler_toks = sum(filler_tok_lens) / len(filler_tok_lens)

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

    user_str = f"{prefix_body} {target_fact} {suffix_body}\n\n{query_str}"
    full_str = apply_chat_no_think(
        tok,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_str}],
    )
    q_char = full_str.rfind(query_str)
    if q_char < 0:
        raise RuntimeError("Failed to locate query_str in prompt")

    ctx_str = full_str[:q_char]
    q_str = full_str[q_char:]

    enc_ctx = tok(ctx_str, return_offsets_mapping=True, add_special_tokens=False)
    enc_q = tok(q_str, add_special_tokens=False)
    ids_ctx_list = enc_ctx["input_ids"]
    ids_q_list = enc_q["input_ids"]
    T_ctx = len(ids_ctx_list)

    fact_char = ctx_str.find(target_fact)
    if fact_char < 0:
        raise RuntimeError(f"Could not find target_fact {target_fact!r} in ctx_str")
    c0, c1 = find_subspan_tokens(ctx_str, target_fact, fact_char, enc_ctx["offset_mapping"])

    val_offset = target_fact.rfind(target_val)
    pos_val = find_value_token_span(ctx_str, target_val, fact_char + val_offset, enc_ctx["offset_mapping"])
    v0, v1 = int(pos_val[0]), int(pos_val[-1] + 1)

    val_toks = tok.encode(target_val, add_special_tokens=False)
    if len(pos_val) != len(val_toks) or ids_ctx_list[v0] != val_toks[0]:
        raise RuntimeError(
            f"Sample {sample['sample_index']}: value span alignment mismatch pos_val={pos_val}, val_toks={val_toks}"
        )

    b_end = min(T_ctx, v1 + 3 * (n_linear - 1))
    return {
        "ids_ctx": torch.tensor([ids_ctx_list], dtype=torch.long, device=device),
        "ids_q": torch.tensor([ids_q_list], dtype=torch.long, device=device),
        "v0": v0,
        "v1": v1,
        "c0": c0,
        "c1": c1,
        "b_end": int(b_end),
        "T_ctx": int(T_ctx),
        "q_len": int(len(ids_q_list)),
        "actual_depth_ratio": float(v0 / max(1, T_ctx)),
        "target_val": target_val,
        "f_target": int(val_toks[0]),
        "target_toks": [int(t) for t in val_toks],
        "dist_vals": [d["val"] for d in distractors],
        "dist_first_toks": [int(tok.encode(d["val"], add_special_tokens=False)[0]) for d in distractors],
    }


def soft_zero_kv_slice(cache, start: int, end: int):
    """Legacy Pilot 3/4 `soft-zero`: zero K and V slots in `[start, end)` while keeping mask=1."""
    for layer in cache.layers:
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        if torch.is_tensor(k) and torch.is_tensor(v) and k.numel() > 0:
            k[:, :, start:end, :].zero_()
            v[:, :, start:end, :].zero_()


def zero_all_linear_states(cache):
    """Zero out all GDN recurrent_states and conv_states while leaving full-attention KV untouched."""
    for layer in cache.layers:
        for attr in ("recurrent_states", "conv_states"):
            d = getattr(layer, attr, None)
            if isinstance(d, dict):
                for t in d.values():
                    if torch.is_tensor(t):
                        t.zero_()
            elif torch.is_tensor(d):
                d.zero_()


@torch.no_grad()
def eval_from_ctx_cache_with_mask(
    model,
    base_model,
    tok,
    eos_ids: List[int],
    ctx_cache,
    T_ctx: int,
    ids_q: torch.Tensor,
    attn_mask_ctx: Optional[torch.Tensor] = None,
    use_logits_to_keep: bool = True,
    max_new_tokens: int = 8,
) -> Dict[str, Any]:
    """
    Evaluate `ids_q` and greedy generation IN-PLACE on `ctx_cache` (`T_ctx` seen tokens),
    optionally applying a 2D `attn_mask_ctx` of shape `(1, T_ctx)` where `0`s perform true
    hard attention-mask eviction without touching `position_ids` or cache sequence lengths.
    """
    q_len = int(ids_q.shape[1])
    q_pos = make_position_ids(T_ctx, q_len, ids_q.device, base_model)
    cur_mask = None
    if attn_mask_ctx is not None:
        if attn_mask_ctx.shape != (1, T_ctx):
            raise ValueError(f"Expected attn_mask_ctx shape (1, {T_ctx}), got {attn_mask_ctx.shape}")
        q_ones = torch.ones((1, q_len), dtype=attn_mask_ctx.dtype, device=attn_mask_ctx.device)
        cur_mask = torch.cat([attn_mask_ctx, q_ones], dim=-1)

    kwargs: Dict[str, Any] = {
        "input_ids": ids_q,
        "position_ids": q_pos,
        "use_cache": True,
        "past_key_values": ctx_cache,
    }
    if cur_mask is not None:
        kwargs["attention_mask"] = cur_mask
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
        if cur_mask is not None:
            one = torch.ones((1, 1), dtype=cur_mask.dtype, device=cur_mask.device)
            cur_mask = torch.cat([cur_mask, one], dim=-1)
        step_kwargs: Dict[str, Any] = {
            "input_ids": nxt,
            "position_ids": step_pos,
            "use_cache": True,
            "past_key_values": q_cache,
        }
        if cur_mask is not None:
            step_kwargs["attention_mask"] = cur_mask
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

    gen_no_eos = [t for t in gen_ids if t not in eos_ids]
    text = tok.decode(gen_no_eos, skip_special_tokens=True).strip()
    return {
        "first_logits": first_logits,
        "gen_ids": gen_ids,
        "text": text,
    }


def score_read_condition(
    res: Dict[str, Any],
    target_val: str,
    f_target: int,
    dist_first_toks: List[int],
    ref_full_logits: torch.Tensor,
) -> Dict[str, Any]:
    lg = res["first_logits"].float()
    ref = ref_full_logits.float()

    probs = F.softmax(lg, dim=-1)
    p_target_1st = float(probs[0, f_target].item())
    target_logit = float(lg[0, f_target].item())

    # Margin vs best competing distractor first digit and vs best non-target vocab token
    best_dist_logit = max(float(lg[0, dt].item()) for dt in dist_first_toks)
    masked_lg = lg.clone()
    masked_lg[0, f_target] = float("-inf")
    best_other_logit = float(masked_lg.max().item())

    lp = F.log_softmax(lg, dim=-1)
    lq = F.log_softmax(ref, dim=-1)
    kl_vs_full = float(F.kl_div(lp, lq.exp(), reduction="batchmean").item())
    max_diff_vs_full = float((lg - ref).abs().max().item())

    top1_id = int(lg.argmax(dim=-1).item())
    ref_top1_id = int(ref.argmax(dim=-1).item())
    txt = res["text"]

    return {
        "text": txt,
        "exact_match": bool(txt == target_val),
        "contains_target": bool(target_val in txt),
        "first_tok_hit": bool(top1_id == f_target),
        "top1_eq_full": bool(top1_id == ref_top1_id),
        "p_target_1st": p_target_1st,
        "margin_vs_other": float(target_logit - best_other_logit),
        "margin_vs_distractor": float(target_logit - best_dist_logit),
        "kl_vs_full": kl_vs_full,
        "max_diff_vs_full": max_diff_vs_full,
    }


def run_p1_1_sample(
    model,
    base_model,
    tok,
    eos_ids: List[int],
    sample: Dict[str, Any],
    target_length: int,
    n_linear: int,
    max_chunk: int,
    device: str,
    use_logits_to_keep: bool,
    verify_restore: bool = False,
) -> Dict[str, Any]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_start = time.perf_counter()
    pinfo = build_unmodified_read_prompt(tok, sample, target_length, n_linear=n_linear, device=device)
    ids_ctx = pinfo["ids_ctx"]
    ids_q = pinfo["ids_q"]
    T_ctx = pinfo["T_ctx"]
    v0, v1 = pinfo["v0"], pinfo["v1"]
    c0, c1 = pinfo["c0"], pinfo["c1"]
    b_end = pinfo["b_end"]

    # Single shared prefill on unmodified prompt
    _, c_ctx = run_chunked_prefill(
        model,
        base_model,
        ids_ctx,
        pos_start=0,
        cache=None,
        max_chunk=max_chunk,
        use_logits_to_keep=use_logits_to_keep,
    )

    # Snap linear states and KV on `[c0, b_end)` for exact in-place restoration
    orig_states = snap_states(c_ctx)
    kv_start = min(c0, v0)
    kv_end = max(c1, b_end)
    orig_kv = extract_kv_slice(c_ctx, kv_start, kv_end)

    ref_copy = clone_cache(c_ctx) if verify_restore else None

    def restore():
        truncate_cache_kv(c_ctx, T_ctx)
        overwrite_states(c_ctx, orig_states)
        patch_kv_slice(c_ctx, orig_kv, kv_start, kv_end)

    # 1. full_read
    res_full = eval_from_ctx_cache_with_mask(
        model, base_model, tok, eos_ids, c_ctx, T_ctx, ids_q,
        attn_mask_ctx=None, use_logits_to_keep=use_logits_to_keep,
    )
    restore()
    if verify_restore and ref_copy is not None:
        assert_cache_bitwise_equal(c_ctx, ref_copy)

    lg_full = res_full["first_logits"].detach().clone()

    # 2. soft_zero_val (`[v0, v1)` K=0,V=0, mask=1)
    soft_zero_kv_slice(c_ctx, v0, v1)
    res_sz_val = eval_from_ctx_cache_with_mask(
        model, base_model, tok, eos_ids, c_ctx, T_ctx, ids_q,
        attn_mask_ctx=None, use_logits_to_keep=use_logits_to_keep,
    )
    restore()

    # 3. soft_zero_clause (`[c0, c1)` K=0,V=0, mask=1)
    soft_zero_kv_slice(c_ctx, c0, c1)
    res_sz_clause = eval_from_ctx_cache_with_mask(
        model, base_model, tok, eos_ids, c_ctx, T_ctx, ids_q,
        attn_mask_ctx=None, use_logits_to_keep=use_logits_to_keep,
    )
    restore()

    # 4-7. Hard attention-mask eviction conditions (KV untouched, seq_len preserved)
    masks = {}
    for name, s_idx, e_idx in (
        ("read_kv_evict_val", v0, v1),
        ("read_kv_evict_clause", c0, c1),
        ("read_kv_evict_block", c0, b_end),
        ("read_kv_evict_ctx", 0, T_ctx),
    ):
        m = torch.ones((1, T_ctx), dtype=torch.long, device=device)
        m[:, s_idx:e_idx] = 0
        res = eval_from_ctx_cache_with_mask(
            model, base_model, tok, eos_ids, c_ctx, T_ctx, ids_q,
            attn_mask_ctx=m, use_logits_to_keep=use_logits_to_keep,
        )
        restore()
        masks[name] = res

    # 8. state_zero_read (zero GDN states, full KV intact)
    zero_all_linear_states(c_ctx)
    res_state_zero = eval_from_ctx_cache_with_mask(
        model, base_model, tok, eos_ids, c_ctx, T_ctx, ids_q,
        attn_mask_ctx=None, use_logits_to_keep=use_logits_to_keep,
    )
    restore()
    if verify_restore and ref_copy is not None:
        assert_cache_bitwise_equal(c_ctx, ref_copy)
        del ref_copy

    del c_ctx, orig_states, orig_kv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_map = {
        "full_read": res_full,
        "soft_zero_val": res_sz_val,
        "soft_zero_clause": res_sz_clause,
        "read_kv_evict_val": masks["read_kv_evict_val"],
        "read_kv_evict_clause": masks["read_kv_evict_clause"],
        "read_kv_evict_block": masks["read_kv_evict_block"],
        "read_kv_evict_ctx": masks["read_kv_evict_ctx"],
        "state_zero_read": res_state_zero,
    }
    cond_scores = {
        cname: score_read_condition(
            cres, pinfo["target_val"], pinfo["f_target"], pinfo["dist_first_toks"], lg_full
        )
        for cname, cres in raw_map.items()
    }

    return {
        "sample_index": int(sample["sample_index"]),
        "target_length": int(target_length),
        "depth": float(sample["depth"]),
        "actual_depth_ratio": pinfo["actual_depth_ratio"],
        "tpl_id": int(sample["tpl_id"]),
        "target": sample["target"],
        "target_val": pinfo["target_val"],
        "T_ctx": int(T_ctx),
        "q_len": int(pinfo["q_len"]),
        "v0": int(v0),
        "v1": int(v1),
        "c0": int(c0),
        "c1": int(c1),
        "b_end": int(b_end),
        "verify_restore_ok": bool(verify_restore),
        "wall_s": float(time.perf_counter() - t_start),
        "conditions": cond_scores,
    }


def paired_metric_diff(recs: List[Dict[str, Any]], cond_a: str, cond_b: str, metric: str) -> Dict[str, Any]:
    diffs = np.array([
        r["conditions"][cond_a][metric] - r["conditions"][cond_b][metric]
        for r in recs
    ], dtype=np.float64)
    n = len(diffs)
    stats_d = bootstrap_mean_ci(diffs.tolist(), seed=42)
    wins = int(np.sum(diffs > 0))
    losses = int(np.sum(diffs < 0))
    ties = int(np.sum(diffs == 0))
    return {
        "cond_a": cond_a,
        "cond_b": cond_b,
        "metric": metric,
        "n": n,
        "mean_diff_a_minus_b": float(stats_d["mean"]),
        "ci95": [float(stats_d["ci95_low"]), float(stats_d["ci95_high"])],
        "a_gt_b": wins,
        "a_lt_b": losses,
        "ties": ties,
        "sign_p": float(exact_sign_test_two_sided_p(wins, wins + losses)) if (wins + losses) > 0 else 1.0,
        "wilcoxon_p": float(wilcoxon_signed_rank_two_sided_p(diffs)),
    }


def summarize_records(recs: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    by_len: Dict[int, List[Dict[str, Any]]] = {}
    for r in recs:
        by_len.setdefault(int(r["target_length"]), []).append(r)

    summary_by_length = {}
    for tlen, sub in sorted(by_len.items()):
        n = len(sub)
        cond_names = list(sub[0]["conditions"].keys())
        cond_stats = {}
        for cn in cond_names:
            em_cnt = sum(1 for r in sub if r["conditions"][cn]["exact_match"])
            ct_cnt = sum(1 for r in sub if r["conditions"][cn]["contains_target"])
            ft_cnt = sum(1 for r in sub if r["conditions"][cn]["first_tok_hit"])
            p_arr = np.array([r["conditions"][cn]["p_target_1st"] for r in sub], dtype=np.float64)
            kl_arr = np.array([r["conditions"][cn]["kl_vs_full"] for r in sub], dtype=np.float64)
            m_arr = np.array([r["conditions"][cn]["margin_vs_other"] for r in sub], dtype=np.float64)
            cond_stats[cn] = {
                "n": n,
                "exact_match_count": em_cnt,
                "exact_match_rate": float(em_cnt / max(1, n)),
                "contains_target_count": ct_cnt,
                "contains_target_rate": float(ct_cnt / max(1, n)),
                "first_tok_hit_count": ft_cnt,
                "first_tok_hit_rate": float(ft_cnt / max(1, n)),
                "p_target_mean": float(np.mean(p_arr)),
                "p_target_median": float(np.median(p_arr)),
                "margin_other_mean": float(np.mean(m_arr)),
                "kl_vs_full_mean": float(np.mean(kl_arr)),
                "kl_vs_full_median": float(np.median(kl_arr)),
            }

        comparisons = [
            paired_metric_diff(sub, "soft_zero_val", "read_kv_evict_val", "p_target_1st"),
            paired_metric_diff(sub, "soft_zero_clause", "read_kv_evict_clause", "p_target_1st"),
            paired_metric_diff(sub, "state_zero_read", "read_kv_evict_ctx", "p_target_1st"),
            paired_metric_diff(sub, "full_read", "read_kv_evict_val", "p_target_1st"),
            paired_metric_diff(sub, "full_read", "state_zero_read", "p_target_1st"),
        ]
        summary_by_length[str(tlen)] = {
            "n": n,
            "mean_wall_s": float(np.mean([r["wall_s"] for r in sub])),
            "conditions": cond_stats,
            "paired_comparisons": comparisons,
        }

    return {
        "model": model_name,
        "total_records": len(recs),
        "by_length": summary_by_length,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Wave 10 P1-1: read_kv_evict vs soft-zero benchmark")
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    p.add_argument("--lengths", type=str, default="1024,4096")
    p.add_argument("--n-samples", type=int, default=120)
    p.add_argument("--max-chunk", type=int, default=2048)
    p.add_argument("--seed", type=int, default=202605)
    p.add_argument("--tag", type=str, default="2b")
    p.add_argument("--out-dir", type=str, default="pilot")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--timeout-s", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    t_run_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"w10_p1_1_read_evict_{args.tag}_records.jsonl"
    summary_path = out_dir / f"w10_p1_1_read_evict_{args.tag}_summary.json"

    print(f"Loading model {args.model} on {device}...")
    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True, local_files_only=args.local_files_only,
    ).eval()

    base_model = getattr(model, "model", model)
    t_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_linear = count_linear_layers(t_cfg)
    eos_ids = get_eos_ids(model, tok)
    use_l2k = has_logits_to_keep_arg(model)
    print(f"Loaded {args.model}: n_linear={n_linear}, attn_impl={getattr(t_cfg, '_attn_implementation', 'unknown')}")

    samples = generate_e5_sample_bank(n_samples=args.n_samples, depths=[0.25, 0.50, 0.75], seed=args.seed)

    done_keys = set()
    existing_recs: List[Dict[str, Any]] = []
    if args.resume and jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (int(rec["target_length"]), int(rec["sample_index"]))
                if key not in done_keys:
                    done_keys.add(key)
                    existing_recs.append(rec)
        print(f"Resuming: {len(existing_recs)} completed records loaded.")

    recs = list(existing_recs)
    stopped_by_timeout = False

    for tlen in lengths:
        for s_idx, sample in enumerate(samples):
            if (tlen, s_idx) in done_keys:
                continue
            if args.timeout_s > 0 and (time.perf_counter() - t_run_start) >= args.timeout_s:
                stopped_by_timeout = True
                break

            verify_restore = (s_idx == 0 and tlen == lengths[0])
            rec = run_p1_1_sample(
                model=model,
                base_model=base_model,
                tok=tok,
                eos_ids=eos_ids,
                sample=sample,
                target_length=tlen,
                n_linear=n_linear,
                max_chunk=args.max_chunk,
                device=device,
                use_logits_to_keep=use_l2k,
                verify_restore=verify_restore,
            )
            append_record_jsonl(jsonl_path, rec)
            recs.append(rec)
            done_keys.add((tlen, s_idx))

            if (s_idx + 1) % 20 == 0 or s_idx == 0 or (s_idx + 1) == len(samples):
                c = rec["conditions"]
                print(
                    f"[T={tlen} s={s_idx+1}/{len(samples)}] "
                    f"full={c['full_read']['p_target_1st']:.3f} | "
                    f"sz_val={c['soft_zero_val']['p_target_1st']:.3f} | "
                    f"evict_val={c['read_kv_evict_val']['p_target_1st']:.3f} | "
                    f"evict_cls={c['read_kv_evict_clause']['p_target_1st']:.3f} | "
                    f"evict_ctx={c['read_kv_evict_ctx']['p_target_1st']:.3f} | "
                    f"state0={c['state_zero_read']['p_target_1st']:.3f}",
                    flush=True,
                )
        if stopped_by_timeout:
            break

    summary = summarize_records(recs, args.model)
    summary["stopped_by_timeout"] = stopped_by_timeout
    summary["wall_time_sec"] = float(time.perf_counter() - t_run_start)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary -> {summary_path}")
    for tl, blk in summary["by_length"].items():
        print(f"=== SUMMARY T={tl} (n={blk['n']}) ===")
        for cn, st in blk["conditions"].items():
            print(
                f"  {cn:20s}: EM={st['exact_match_count']}/{st['n']} ({st['exact_match_rate']*100:.1f}%) "
                f"| hit1={st['first_tok_hit_rate']*100:.1f}% | p_mean={st['p_target_mean']:.4f} "
                f"| kl_med={st['kl_vs_full_median']:.4g}"
            )


if __name__ == "__main__":
    main()
