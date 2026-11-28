"""Cross-family E2, E3, and E6 experiments for Mamba-2 hybrids (Falcon-H1-1.5B & Zamba2-1.2B).

Implements unified O_local Mamba-2 additive block editing:
  - Extracts per-SSM-head step decay g_t = dt_eff * A in fp32 (with Falcon-H1 mup_vector),
    accumulates G_t = sum(g_t) over [b, T_ctx) in fp64, and injects at edit time in fp64:
        S_T_new = S_T_old + exp(G_T - G_b) * (S_b_new - S_b_old)
  - Patches hybrid attention KV on [p0, b) (all 24 layers in Falcon-H1; hybrid_layer_ids in Zamba2)
    and tail conv_states if T_ctx - b < W_conv.
  - Verifies epsilon_H === 0 on isolated Mamba-2 dynamics so (J-local-recompute vs J-local) cleanly
    isolates pure cross-layer coupling epsilon_cross.

Suites:
  E2: Single-fact editing (11 conditions: old, oracle, J-full, J-local, J-local-stale,
      J-local-evict, J-control, J-local-recompute, state_only, kv_only, placebo_unrelated_span)
  E3: Multi-fact editing (M=4 k=2 all 2 perms; M=8 k=4 P=6 perms) + collateral preservation
  E6: Long-range / multi-key transport ablation (stale vs transport-decay vs recomputed/J-local-recompute)

Supports --synthetic-dry-run on CPU (`CUDA_VISIBLE_DEVICES=""`) for protocol/schema audit before GPU execution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", r"D:\applicationData\HuggingFace")

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    FalconH1Config,
    FalconH1ForCausalLM,
    Zamba2Config,
    Zamba2ForCausalLM,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w2_crossfamily_e0c import (  # noqa: E402
    DIGIT4_PAIRS,
    MODEL_REGISTRY,
    NAMES,
    POST_FILLER_TEMPLATES,
    PRE_FILLER_TEMPLATES,
    SINGLE_TOKEN_PAIRS,
    assert_cache_unchanged,
    build_fact_prompt,
    check_oracle_clean,
    clone_cache,
    extract_mamba_mixers,
    format_fact_sentence,
    format_query_suffix,
    get_eos_ids,
    get_format_spec,
    kl_div_from_logits,
    make_position_ids,
    manual_greedy,
    overwrite_ssm_states,
    patch_kv_range,
    render_chat_prompt,
    resolve_block_offset,
    run_segment,
    snap_all_cache_tensors,
)


def sync_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_completed_keys(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "record_key" in rec:
                    done.add(str(rec["record_key"]))
            except json.JSONDecodeError:
                continue
    return done


def run_segment_with_mamba_decay(
    model: Any,
    cache: DynamicCache,
    ids_seg: torch.LongTensor,
    pos_start: int,
    mixers: list[tuple[int, Any]],
    max_chunk: int = 1024,
    record_per_token: bool = False,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
    """Run ids_seg while accumulating fp32 dt_eff * A decay steps in fp64 per SSM layer/head."""
    seq_len = int(ids_seg.shape[1])
    if seq_len == 0:
        empty_sums = {
            li: torch.zeros((1, mix.num_heads), device=ids_seg.device, dtype=torch.float64)
            for li, mix in mixers
        }
        empty_seq = (
            {li: torch.zeros((1, 0, mix.num_heads), device=ids_seg.device, dtype=torch.float32) for li, mix in mixers}
            if record_per_token
            else None
        )
        return torch.empty(0, device=ids_seg.device), empty_sums, empty_seq

    g_sums: dict[int, torch.Tensor] = {}
    g_chunks_tok: dict[int, list[torch.Tensor]] | None = {li: [] for li, _ in mixers} if record_per_token else None
    hooks = []

    def make_hook(layer_idx: int, mixer: Any):
        def _hook(_mod, _inp, out):
            proj = out * mixer.mup_vector if hasattr(mixer, "mup_vector") else out
            dt_raw = proj[..., mixer.intermediate_size + mixer.conv_dim :]
            dt_eff = torch.clamp(
                F.softplus(dt_raw.float() + mixer.dt_bias.float()),
                min=mixer.time_step_limit[0],
                max=mixer.time_step_limit[1],
            )
            A = -torch.exp(mixer.A_log.float())
            g_step = dt_eff * A  # (1, L, num_heads) in fp32
            chunk_sum = g_step.sum(dim=1, dtype=torch.float64)
            g_sums[layer_idx] = g_sums[layer_idx] + chunk_sum if layer_idx in g_sums else chunk_sum
            if g_chunks_tok is not None:
                g_chunks_tok[layer_idx].append(g_step.detach())

        return _hook

    for li, mix in mixers:
        hooks.append(mix.in_proj.register_forward_hook(make_hook(li, mix)))

    last_logits = None
    try:
        for off in range(0, seq_len, max_chunk):
            sub = ids_seg[:, off : off + max_chunk]
            pos = make_position_ids(pos_start + off, int(sub.shape[1]), ids_seg.device)
            out = model(input_ids=sub, position_ids=pos, past_key_values=cache, use_cache=True)
            last_logits = out.logits
    finally:
        for hk in hooks:
            hk.remove()

    per_tok = (
        {li: torch.cat(chunks, dim=1) for li, chunks in g_chunks_tok.items()}
        if g_chunks_tok is not None
        else None
    )
    return last_logits, g_sums, per_tok


def compute_state_delta_fp64(cache_b_old: DynamicCache, cache_b_new: DynamicCache) -> dict[int, torch.Tensor]:
    deltas: dict[int, torch.Tensor] = {}
    for li, (l_old, l_new) in enumerate(zip(cache_b_old.layers, cache_b_new.layers)):
        if hasattr(l_old, "recurrent_states") and l_old.recurrent_states.get(0) is not None:
            deltas[li] = l_new.recurrent_states[0].double() - l_old.recurrent_states[0].double()
    return deltas


def patch_conv_tail(
    c_target: DynamicCache,
    c_new_blk: DynamicCache,
    block_end: int,
    T_ctx: int,
    w_conv: int = 4,
) -> None:
    overlap = w_conv - (T_ctx - block_end)
    if overlap <= 0:
        return
    k = min(w_conv, overlap)
    for l_dst, l_src in zip(c_target.layers, c_new_blk.layers):
        if hasattr(l_src, "conv_states") and l_src.conv_states.get(0) is not None:
            l_dst.conv_states[0][:, :, :k].copy_(l_src.conv_states[0][:, :, -k:])


def apply_mamba2_jlocal_update(
    cache_old_ctx: DynamicCache,
    c_new_blk: DynamicCache,
    delta_b_fp64: dict[int, torch.Tensor],
    g_sums: dict[int, torch.Tensor],
    start_pos: int,
    block_end: int,
    T_ctx: int,
    w_conv: int = 4,
    use_decay: bool = True,
    patch_state: bool = True,
    patch_kv: bool = True,
) -> DynamicCache:
    out = clone_cache(cache_old_ctx)
    if patch_state:
        for li, delta in delta_b_fp64.items():
            s_old = out.layers[li].recurrent_states[0]
            if use_decay and T_ctx > block_end:
                gamma = torch.exp(g_sums[li])[:, :, None, None]  # fp64, broadcast to (1, H, d_head, d_state)
                s_new = (s_old.double() + gamma * delta).to(s_old.dtype)
            else:
                s_new = (s_old.double() + delta).to(s_old.dtype)
            out.layers[li].recurrent_states[0].copy_(s_new)
        patch_conv_tail(out, c_new_blk, block_end, T_ctx, w_conv=w_conv)
    if patch_kv:
        patch_kv_range(out, c_new_blk, start_pos, block_end)
    return out


def prepare_bckpt_cache(
    model: Any,
    cfg: Any,
    ids_prefix: torch.LongTensor,
    p0: int,
    chunk_size: int = 256,
    max_chunk: int = 1024,
) -> tuple[DynamicCache, DynamicCache | None]:
    """Build B-ckpt at c <= p0 and advance [c, p0) to return c_pre (and optional c_ckpt)."""
    c_ckpt_pos = max(0, p0 - (p0 % chunk_size)) if chunk_size > 0 else 0
    c_ckpt = DynamicCache(config=cfg)
    if c_ckpt_pos > 0:
        run_segment(model, c_ckpt, ids_prefix[:, :c_ckpt_pos], 0, max_chunk=max_chunk)
        c_pre = clone_cache(c_ckpt)
        if p0 > c_ckpt_pos:
            run_segment(model, c_pre, ids_prefix[:, c_ckpt_pos:p0], c_ckpt_pos, max_chunk=max_chunk)
        return c_pre, c_ckpt
    c_pre = DynamicCache(config=cfg)
    if p0 > 0:
        run_segment(model, c_pre, ids_prefix[:, :p0], 0, max_chunk=max_chunk)
    return c_pre, None


def run_segment_with_evict(
    model: Any,
    cache: DynamicCache,
    ids_seg: torch.LongTensor,
    pos_start: int,
    evict_span: tuple[int, int] | None = None,
) -> torch.Tensor:
    if evict_span is None:
        return run_segment(model, cache, ids_seg, pos_start)
    p0, p1 = evict_span
    seq_len = int(ids_seg.shape[1])
    total_len = pos_start + seq_len
    mask = torch.ones((1, total_len), device=ids_seg.device, dtype=torch.long)
    mask[:, p0:p1] = 0
    pos = make_position_ids(pos_start, seq_len, ids_seg.device)
    out = model(input_ids=ids_seg, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
    return out.logits


def manual_greedy_with_evict(
    model: Any,
    cache: DynamicCache,
    first_logits: torch.Tensor,
    pos_next: int,
    max_new: int,
    eos_ids: list[int],
    evict_span: tuple[int, int] | None = None,
) -> list[int]:
    if evict_span is None:
        return manual_greedy(model, cache, first_logits, pos_next, max_new, eos_ids)
    p0, p1 = evict_span
    gen: list[int] = []
    cur_tok = int(first_logits[:, -1, :].argmax(dim=-1).item())
    gen.append(cur_tok)
    if cur_tok in eos_ids:
        return gen
    cur_pos = pos_next
    for _ in range(max_new - 1):
        inp = torch.tensor([[cur_tok]], device=first_logits.device, dtype=torch.long)
        mask = torch.ones((1, cur_pos + 1), device=first_logits.device, dtype=torch.long)
        mask[:, p0:p1] = 0
        pos = make_position_ids(cur_pos, 1, first_logits.device)
        out = model(input_ids=inp, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
        cur_tok = int(out.logits[:, -1, :].argmax(dim=-1).item())
        gen.append(cur_tok)
        cur_pos += 1
        if cur_tok in eos_ids:
            break
    return gen


_CHAT_ANCHOR_CACHE: dict[int, tuple[str, list[int]]] = {}
_ANS_TOK_CACHE: dict[tuple[int, str], tuple[list[int], list[int]]] = {}


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def resolve_answer_tokens(tok: Any, val: str) -> tuple[list[int], list[int]]:
    """Return canonical chat-continuation ans_ids and valid 1st-token candidate IDs across BPE/SP."""
    if tok is None:
        return [], []
    key = (id(tok), val)
    if key in _ANS_TOK_CACHE:
        return _ANS_TOK_CACHE[key]

    tid_key = id(tok)
    if tid_key not in _CHAT_ANCHOR_CACHE:
        chat_str = render_chat_prompt(tok, "Q:")
        chat_ids = tok.encode(chat_str, add_special_tokens=False)
        _CHAT_ANCHOR_CACHE[tid_key] = (chat_str, chat_ids)
    chat_str, chat_ids = _CHAT_ANCHOR_CACHE[tid_key]

    def _encode_cont(s: str) -> list[int]:
        full = tok.encode(chat_str + s, add_special_tokens=False)
        if len(full) > len(chat_ids) and full[: len(chat_ids)] == chat_ids:
            return list(full[len(chat_ids) :])
        return list(tok.encode(s, add_special_tokens=False))

    ans_ids = _encode_cont(val)
    cand_1st: list[int] = []
    seen: set[int] = set()
    for variant in (val, val.capitalize(), " " + val, " " + val.capitalize()):
        cont = _encode_cont(variant)
        for t in cont:
            if tok.decode([t]).strip():
                if t not in seen:
                    seen.add(t)
                    cand_1st.append(t)
                break
    _ANS_TOK_CACHE[key] = (ans_ids, cand_1st)
    return ans_ids, cand_1st


def mean_ans_kl(logits_ref: torch.Tensor, logits_cand: torch.Tensor) -> float:
    n_tok = min(int(logits_ref.shape[1]), int(logits_cand.shape[1]))
    if n_tok <= 0:
        return 0.0
    total = 0.0
    for k in range(n_tok):
        total += kl_div_from_logits(logits_ref[:, k, :], logits_cand[:, k, :])
    return total / n_tok


def score_generation(gen_text: str, val_new: str, val_old: str, fmt: str) -> dict[str, Any]:
    cleaned = gen_text.strip()
    if fmt == "digit4":
        m = re.search(r"\b(\d{4})\b", cleaned)
        extracted = m.group(1) if m else ""
        pred_for_edit = extracted if extracted else cleaned.rstrip(".,!?:;\"'")
        strict = cleaned.rstrip(".,!?:;\"'") == val_new or extracted == val_new
        old_hit = cleaned.rstrip(".,!?:;\"'") == val_old or extracted == val_old
        contains_new = bool(re.search(rf"(?<!\d){re.escape(val_new)}(?!\d)", cleaned))
        contains_old = bool(re.search(rf"(?<!\d){re.escape(val_old)}(?!\d)", cleaned))
        norm_edit = _levenshtein(pred_for_edit, val_new) / max(len(pred_for_edit), len(val_new), 1)
    else:
        first_word = cleaned.split()[0].rstrip(".,!?:;\"'").lower() if cleaned.split() else ""
        strict = first_word == val_new.lower()
        old_hit = first_word == val_old.lower()
        contains_new = bool(re.search(rf"\b{re.escape(val_new.lower())}\b", cleaned.lower()))
        contains_old = bool(re.search(rf"\b{re.escape(val_old.lower())}\b", cleaned.lower()))
        norm_edit = _levenshtein(first_word, val_new.lower()) / max(len(first_word), len(val_new), 1)
    return {
        "acc_new": bool(strict or (contains_new and not contains_old)),
        "strict_new": bool(strict),
        "acc_old": bool(old_hit or (contains_old and not contains_new)),
        "strict_old": bool(old_hit),
        "norm_edit_dist": float(norm_edit),
    }


def eval_suffix_condition(
    model: Any,
    tok: Any,
    ctx_cache: DynamicCache,
    ids_suffix: torch.LongTensor,
    T_ctx: int,
    val_new: str,
    val_old: str,
    fmt: str,
    max_new: int,
    eos_ids: list[int],
    evict_span: tuple[int, int] | None = None,
) -> dict[str, Any]:
    ans_ids_new, cand_1st_new = resolve_answer_tokens(tok, val_new)
    ans_ids_old, cand_1st_old = resolve_answer_tokens(tok, val_old)

    with torch.no_grad():
        work = clone_cache(ctx_cache)
        lg_full = run_segment_with_evict(model, work, ids_suffix, T_ctx, evict_span=evict_span)
        last_logits = lg_full[:, -1, :].detach()
        T_tot = T_ctx + int(ids_suffix.shape[1])

        def _teacher_force(ans_ids: list[int]) -> tuple[torch.Tensor, float]:
            if not ans_ids:
                return last_logits.unsqueeze(1), 0.0
            if len(ans_ids) > 1:
                tf_cache = clone_cache(work)
                tf_in = torch.tensor([ans_ids[:-1]], device=last_logits.device, dtype=torch.long)
                tf_out = run_segment_with_evict(model, tf_cache, tf_in, T_tot, evict_span=evict_span)
                all_lg = torch.cat([last_logits.unsqueeze(1), tf_out.detach()], dim=1)
                del tf_cache
            else:
                all_lg = last_logits.unsqueeze(1)
            lprobs = F.log_softmax(all_lg[0].double(), dim=-1)
            seq_lp = sum(float(lprobs[k, ans_ids[k]].item()) for k in range(len(ans_ids)))
            return all_lg, float(seq_lp)

        if tok is not None:
            probs0 = F.softmax(last_logits[0].double(), dim=-1)
            p_new_1st = float(probs0[cand_1st_new].sum().item()) if cand_1st_new else 0.0
            p_old_1st = float(probs0[cand_1st_old].sum().item()) if cand_1st_old else 0.0
            ans_logits_new, seq_lp_new = _teacher_force(ans_ids_new)
            _, seq_lp_old = _teacher_force(ans_ids_old)
        else:
            p_new_1st = 0.0
            p_old_1st = 0.0
            ans_logits_new = last_logits.unsqueeze(1)
            seq_lp_new = 0.0
            seq_lp_old = 0.0

        gen_toks = manual_greedy_with_evict(
            model, work, lg_full, T_tot, max_new=max_new, eos_ids=eos_ids, evict_span=evict_span
        )
    gen_text = tok.decode(gen_toks, skip_special_tokens=True) if tok is not None else " ".join(map(str, gen_toks))
    scores = score_generation(gen_text, val_new, val_old, fmt)
    top1_id = int(last_logits.argmax(dim=-1).item())
    del work
    return {
        **scores,
        "p_new_1st": p_new_1st,
        "p_old_1st": p_old_1st,
        "seq_logprob_new": seq_lp_new,
        "seq_logprob_old": seq_lp_old,
        "gen_text": gen_text[:80],
        "gen_toks": gen_toks,
        "top1_id": top1_id,
        "last_logits": last_logits,
        "ans_logits_new": ans_logits_new,
    }


def verify_epsilon_h_zero_single_layer(
    model: Any,
    cfg: Any,
    mixers: list[tuple[int, Any]],
    device: torch.device,
    dtype: torch.dtype,
    layer_idx: int = 0,
) -> float:
    """Measure isolated single-layer Mamba-2 transport error epsilon_H empirically."""
    _, mix = mixers[layer_idx]
    with torch.no_grad():
        h = torch.randn(1, 64, cfg.hidden_size, device=device, dtype=dtype) * 0.1
        h2 = h.clone()
        h2[:, 10:14, :] = torch.randn(1, 4, cfg.hidden_size, device=device, dtype=dtype) * 0.1
        g_chunks = []

        def _hk(_m, _inp, out):
            proj = out * mix.mup_vector if hasattr(mixer := mix, "mup_vector") else out
            _, _, dt = proj.split([mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], dim=-1)
            dt_eff = torch.clamp(
                F.softplus(dt.float() + mixer.dt_bias.float()),
                min=mixer.time_step_limit[0],
                max=mixer.time_step_limit[1],
            )
            A = -torch.exp(mixer.A_log.float())
            g_chunks.append((dt_eff * A).sum(dim=1, dtype=torch.float64))

        hk = mix.in_proj.register_forward_hook(_hk)
        try:
            mc_o = DynamicCache(config=cfg)
            _ = mix(h[:, :24], cache_params=mc_o)
            S_b_o = mc_o.get_recurrent_state(mix.layer_idx).clone()
            g_chunks.clear()
            _ = mix(h[:, 24:], cache_params=mc_o)
            S_T_o = mc_o.get_recurrent_state(mix.layer_idx).clone()
            G_diff = g_chunks[-1]

            mc_n = DynamicCache(config=cfg)
            _ = mix(h2[:, :24], cache_params=mc_n)
            S_b_n = mc_n.get_recurrent_state(mix.layer_idx).clone()
            _ = mix(h2[:, 24:], cache_params=mc_n)
            S_T_ex = mc_n.get_recurrent_state(mix.layer_idx).clone()
        finally:
            hk.remove()

        gamma = torch.exp(G_diff)[:, :, None, None]
        S_T_tr = (S_T_o.double() + gamma * (S_b_n.double() - S_b_o.double())).to(S_T_o.dtype)
        return float((S_T_tr - S_T_ex).abs().max().item())


def build_placebo_ids(
    tok: Any,
    case: dict[str, Any],
) -> tuple[list[int], int, int] | None:
    """Create a grammatical, token-length-matched placebo edit strictly before p0."""
    ids_o = case["ids_o"]
    p0 = case["p0"]
    span_len = case["expected_len"]
    if tok is None or "full_old" not in case:
        if p0 < span_len + 24:
            return None
        q0 = max(25, min(p0 - span_len - 16, 48))
        q1 = q0 + span_len
        if q1 + 6 >= p0:
            return None
        ids_p = list(ids_o)
        vocab_n = 128
        for idx in range(q0, q1):
            ids_p[idx] = (ids_p[idx] + 5) % vocab_n
        return ids_p, q0, q1

    full_old = case["full_old"]
    if span_len == 4:
        src, dst = "Routine maintenance record #2 (unit 1024):", "Routine maintenance record #2 (unit 7391):"
    else:
        src, dst = "System telemetry archive note #1: calibration sector nominal and humidity stable.", "System telemetry archive note #1: calibration sector nominal and humidity normal."
    if src not in full_old:
        return None
    full_p = full_old.replace(src, dst, 1)
    ids_p = tok(full_p, add_special_tokens=False)["input_ids"]
    if len(ids_p) != len(ids_o):
        return None
    diffs = [i for i in range(len(ids_o)) if ids_o[i] != ids_p[i]]
    if not diffs or diffs != list(range(diffs[0], diffs[-1] + 1)) or len(diffs) != span_len or diffs[-1] + 6 >= p0:
        return None
    return ids_p, diffs[0], diffs[-1] + 1


def summarize_gamma(g_sums: dict[int, torch.Tensor]) -> dict[str, float]:
    vals = torch.cat([torch.exp(v).reshape(-1) for v in g_sums.values()]).float()
    return {
        "min": float(vals.min().item()),
        "median": float(vals.median().item()),
        "mean": float(vals.mean().item()),
        "max": float(vals.max().item()),
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def mean_ci(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
    n = len(vals)
    mu = sum(vals) / n
    if n == 1:
        return {"mean": mu, "ci95_low": mu, "ci95_high": mu, "n": 1}
    var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    half = 1.96 * math.sqrt(var / n)
    return {"mean": mu, "ci95_low": mu - half, "ci95_high": mu + half, "n": n}


def run_e2_suite(
    model_key: str,
    model: Any,
    tok: Any,
    cfg: Any,
    mixers: list[tuple[int, Any]],
    device: torch.device,
    eos_ids: list[int],
    n_samples: int,
    block_horizon: str,
    custom_offset: int,
    bckpt_chunk: int,
    jsonl_path: Path,
    synthetic_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    n_linear = len(mixers)
    w_conv = int(getattr(cfg, "mamba_d_conv", getattr(cfg, "conv_kernel_size", 4)))
    b_offset = resolve_block_offset(block_horizon, n_linear, w_conv, custom_offset)
    done_keys = load_completed_keys(jsonl_path)

    # Assemble clean balanced E2 sample pool (half digit4, half single_token across 3 templates)
    samples: list[dict[str, Any]] = []
    if synthetic_cases is not None:
        samples = synthetic_cases
    else:
        pool_d = []
        pool_s = []
        for idx, (old, new) in enumerate(DIGIT4_PAIRS):
            tpl_id = idx % 3
            name = NAMES[idx % len(NAMES)]
            case = build_fact_prompt(tok, tpl_id, name, old, new, pre_fillers=30, post_fillers=16)
            if case["ok"]:
                pool_d.append(case)
        for idx, (old, new) in enumerate(SINGLE_TOKEN_PAIRS):
            tpl_id = idx % 3
            name = NAMES[(idx + 7) % len(NAMES)]
            case = build_fact_prompt(tok, tpl_id, name, old, new, pre_fillers=30, post_fillers=16)
            if case["ok"]:
                pool_s.append(case)
        half = n_samples // 2
        for i in range(max(half, n_samples - half)):
            if i < len(pool_d) and len(samples) < n_samples:
                samples.append(pool_d[i])
            if i < len(pool_s) and len(samples) < n_samples:
                samples.append(pool_s[i])
        samples = samples[:n_samples]

    cond_names = [
        "old", "oracle", "J-full", "J-local", "J-local-stale",
        "J-local-evict", "J-control", "J-local-recompute",
        "state_only", "kv_only", "placebo_unrelated_span",
    ]
    records: list[dict[str, Any]] = []

    for s_idx, case in enumerate(samples):
        rec_key = f"{model_key}:e2:{s_idx}"
        if rec_key in done_keys:
            continue

        ids_o = torch.tensor([case["ids_o"]], device=device, dtype=torch.long)
        ids_e = torch.tensor([case["ids_n"]], device=device, dtype=torch.long)
        p0, p1, T_ctx = case["p0"], case["p1"], case["T_ctx"]
        block_end = min(T_ctx, p1 + b_offset)
        ids_suffix = ids_e[:, T_ctx:]
        fmt = case["format"]

        with torch.no_grad():
            # 1. B-ckpt + pre-edit cache c_pre at p0
            c_pre, _ = prepare_bckpt_cache(model, cfg, ids_o, p0, chunk_size=bckpt_chunk)
            snaps_pre = snap_all_cache_tensors(c_pre) if s_idx == 0 else None

            # 2. Old path [p0, block_end) -> S_b_old, and [block_end, T_ctx) -> c_old_ctx + G_T - G_b
            c_old_b = clone_cache(c_pre)
            _ = run_segment(model, c_old_b, ids_o[:, p0:block_end], p0)
            c_old_ctx = clone_cache(c_old_b)
            _, g_sums, _ = run_segment_with_mamba_decay(
                model, c_old_ctx, ids_o[:, block_end:T_ctx], block_end, mixers
            )

            # 3. New local block [p0, block_end) -> c_new_blk, S_b_new
            t_blk_start = sync_time(device)
            c_new_blk = clone_cache(c_pre)
            _ = run_segment(model, c_new_blk, ids_e[:, p0:block_end], p0)
            t_blk_ms = (sync_time(device) - t_blk_start) * 1000.0
            delta_b_fp64 = compute_state_delta_fp64(c_old_b, c_new_blk)
            c_jlocal = apply_mamba2_jlocal_update(
                c_old_ctx, c_new_blk, delta_b_fp64, g_sums,
                start_pos=p0, block_end=block_end, T_ctx=T_ctx, w_conv=w_conv,
                use_decay=True, patch_state=True, patch_kv=True,
            )
            t_jlocal_ms = (sync_time(device) - t_blk_start) * 1000.0

            # 4. Oracle (fresh full context on ids_e)
            c_oracle = DynamicCache(config=cfg)
            _ = run_segment(model, c_oracle, ids_e[:, :T_ctx], 0)

            # 5. J-full (Decision A O_block splice onto c_old_b at block_end + [block_end, T_ctx) recompute)
            t0 = sync_time(device)
            c_jfull = clone_cache(c_old_b)
            patch_kv_range(c_jfull, c_new_blk, p0, block_end)
            overwrite_ssm_states(c_jfull, c_new_blk)
            if T_ctx > block_end:
                assert torch.equal(ids_o[:, block_end:T_ctx], ids_e[:, block_end:T_ctx])
                _ = run_segment(model, c_jfull, ids_e[:, block_end:T_ctx], block_end)
            t_jfull_ms = t_blk_ms + (sync_time(device) - t0) * 1000.0

            # 6. J-local-stale, state_only, kv_only
            c_stale = apply_mamba2_jlocal_update(
                c_old_ctx, c_new_blk, delta_b_fp64, g_sums,
                start_pos=p0, block_end=block_end, T_ctx=T_ctx, w_conv=w_conv,
                use_decay=False, patch_state=True, patch_kv=True,
            )
            c_state_only = apply_mamba2_jlocal_update(
                c_old_ctx, c_new_blk, delta_b_fp64, g_sums,
                start_pos=p0, block_end=block_end, T_ctx=T_ctx, w_conv=w_conv,
                use_decay=True, patch_state=True, patch_kv=False,
            )
            c_kv_only = apply_mamba2_jlocal_update(
                c_old_ctx, c_new_blk, delta_b_fp64, g_sums,
                start_pos=p0, block_end=block_end, T_ctx=T_ctx, w_conv=w_conv,
                use_decay=True, patch_state=False, patch_kv=True,
            )

            # 7. J-control: old KV at b + new SSM/conv at b, recompute [b, T_ctx)
            c_jctrl = clone_cache(c_old_b)
            overwrite_ssm_states(c_jctrl, c_new_blk)
            if T_ctx > block_end:
                _ = run_segment(model, c_jctrl, ids_o[:, block_end:T_ctx], block_end)

            # 8. J-local-recompute: patched KV[p0:b) + new SSM/conv at b, recompute [b, T_ctx)
            t0 = sync_time(device)
            c_jrecomp = clone_cache(c_new_blk)
            if T_ctx > block_end:
                _ = run_segment(model, c_jrecomp, ids_o[:, block_end:T_ctx], block_end)
            t_recomp_ms = t_blk_ms + (sync_time(device) - t0) * 1000.0

            # 9. Placebo unrelated span before p0
            p_res = build_placebo_ids(tok, case)
            if p_res is not None:
                ids_p_list, q0, q1 = p_res
                ids_placebo = torch.tensor([ids_p_list], device=device, dtype=torch.long)
                q_b = min(p0 - 1, q1 + 6)
                c_pre_q, _ = prepare_bckpt_cache(model, cfg, ids_o, q0, chunk_size=bckpt_chunk)
                c_q_old = clone_cache(c_pre_q)
                _ = run_segment(model, c_q_old, ids_o[:, q0:q_b], q0)
                c_q_tmp = clone_cache(c_q_old)
                _, g_q_sums, _ = run_segment_with_mamba_decay(
                    model, c_q_tmp, ids_o[:, q_b:T_ctx], q_b, mixers
                )
                c_q_new = clone_cache(c_pre_q)
                _ = run_segment(model, c_q_new, ids_placebo[:, q0:q_b], q0)
                delta_q = compute_state_delta_fp64(c_q_old, c_q_new)
                c_placebo = apply_mamba2_jlocal_update(
                    c_old_ctx, c_q_new, delta_q, g_q_sums,
                    start_pos=q0, block_end=q_b, T_ctx=T_ctx, w_conv=w_conv,
                    use_decay=True, patch_state=True, patch_kv=True,
                )
                del c_pre_q, c_q_old, c_q_tmp, c_q_new
            else:
                c_placebo = clone_cache(c_old_ctx)

            if snaps_pre is not None:
                assert_cache_unchanged(c_pre, snaps_pre)

        cond_caches = {
            "old": (c_old_ctx, None),
            "oracle": (c_oracle, None),
            "J-full": (c_jfull, None),
            "J-local": (c_jlocal, None),
            "J-local-stale": (c_stale, None),
            "J-local-evict": (c_state_only, (p0, p1)),
            "J-control": (c_jctrl, None),
            "J-local-recompute": (c_jrecomp, None),
            "state_only": (c_state_only, None),
            "kv_only": (c_kv_only, None),
            "placebo_unrelated_span": (c_placebo, None),
        }

        evals = {}
        for cname in cond_names:
            cc, ev_span = cond_caches[cname]
            evals[cname] = eval_suffix_condition(
                model, tok, cc, ids_suffix, T_ctx,
                case["new"], case["old"], fmt, max_new=8, eos_ids=eos_ids, evict_span=ev_span,
            )

        lg_ora = evals["oracle"]["last_logits"]
        ans_ora = evals["oracle"]["ans_logits_new"]
        lg_jf = evals["J-full"]["last_logits"]
        lg_recomp = evals["J-local-recompute"]["last_logits"]
        lg_old = evals["old"]["last_logits"]

        eps_cross_kl = kl_div_from_logits(lg_recomp, evals["J-local"]["last_logits"])
        eps_cross_ld = float((lg_recomp - evals["J-local"]["last_logits"]).abs().max().item())
        jctrl_vs_recomp_kl = kl_div_from_logits(lg_recomp, evals["J-control"]["last_logits"])
        ora_chk = (
            check_oracle_clean(evals["oracle"]["gen_text"], case["new"], fmt)
            if tok is not None
            else {"oracle_clean": True, "first_token_clean": True}
        )

        per_cond_out = {}
        for cname in cond_names:
            ev = evals[cname]
            per_cond_out[cname] = {
                "acc_new": ev["acc_new"],
                "strict_new": ev["strict_new"],
                "acc_old": ev["acc_old"],
                "p_new_1st": ev["p_new_1st"],
                "p_old_1st": ev["p_old_1st"],
                "norm_edit_dist": ev["norm_edit_dist"],
                "seq_logprob_new": ev["seq_logprob_new"],
                "seq_logprob_old": ev["seq_logprob_old"],
                "gen_text": ev["gen_text"],
                "kl_vs_oracle": kl_div_from_logits(lg_ora, ev["last_logits"]),
                "ans_mean_kl": mean_ans_kl(ans_ora, ev["ans_logits_new"]),
                "kl_vs_jfull": kl_div_from_logits(lg_jf, ev["last_logits"]),
                "kl_vs_old": kl_div_from_logits(lg_old, ev["last_logits"]),
                "top1_eq_oracle": ev["top1_id"] == evals["oracle"]["top1_id"],
                "top1_eq_jfull": ev["top1_id"] == evals["J-full"]["top1_id"],
            }

        rec = {
            "record_key": rec_key,
            "suite": "E2",
            "model": model_key,
            "sample_idx": s_idx,
            "tpl_id": case["tpl_id"],
            "format": fmt,
            "p0": p0,
            "p1": p1,
            "block_end": block_end,
            "b_offset": b_offset,
            "suffix_len": T_ctx - block_end,
            "T_ctx": T_ctx,
            "oracle_clean": ora_chk["oracle_clean"],
            "oracle_first_token_clean": ora_chk["first_token_clean"],
            "epsilon_H_theoretical": 0.0,
            "epsilon_cross_kl": eps_cross_kl,
            "epsilon_cross_logit_diff": eps_cross_ld,
            "jcontrol_vs_recompute_kl": jctrl_vs_recomp_kl,
            "gamma_summary": summarize_gamma(g_sums),
            "timing_ms": {
                "jlocal": t_jlocal_ms,
                "jlocal_recompute": t_recomp_ms,
                "jfull": t_jfull_ms,
            },
            "conditions": per_cond_out,
        }
        append_jsonl(jsonl_path, rec)
        records.append(rec)

        del c_pre, c_old_b, c_old_ctx, c_new_blk, c_jlocal, c_oracle
        del c_jfull, c_stale, c_state_only, c_kv_only, c_jctrl, c_jrecomp, c_placebo
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Load all E2 records from jsonl (supports resume)
    all_e2 = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("suite") == "E2" and r.get("model") == model_key:
                        all_e2.append(r)

    n = len(all_e2)
    agg_conds = {}
    for cname in cond_names:
        k_new = sum(1 for r in all_e2 if r["conditions"][cname]["acc_new"])
        k_old = sum(1 for r in all_e2 if r["conditions"][cname]["acc_old"])
        k_strict = sum(1 for r in all_e2 if r["conditions"][cname]["strict_new"])
        k_top1_ora = sum(1 for r in all_e2 if r["conditions"][cname]["top1_eq_oracle"])
        agg_conds[cname] = {
            "n": n,
            "acc_new": k_new / max(1, n),
            "acc_new_ci95": wilson_ci(k_new, n),
            "strict_new": k_strict / max(1, n),
            "acc_old": k_old / max(1, n),
            "top1_eq_oracle": k_top1_ora / max(1, n),
            "p_new_1st": mean_ci([r["conditions"][cname]["p_new_1st"] for r in all_e2]),
            "p_old_1st": mean_ci([r["conditions"][cname]["p_old_1st"] for r in all_e2]),
            "norm_edit_dist": mean_ci([r["conditions"][cname]["norm_edit_dist"] for r in all_e2]),
            "seq_logprob_new": mean_ci([r["conditions"][cname]["seq_logprob_new"] for r in all_e2]),
            "seq_logprob_old": mean_ci([r["conditions"][cname]["seq_logprob_old"] for r in all_e2]),
            "kl_vs_oracle": mean_ci([r["conditions"][cname]["kl_vs_oracle"] for r in all_e2]),
            "ans_mean_kl": mean_ci([r["conditions"][cname]["ans_mean_kl"] for r in all_e2]),
            "kl_vs_jfull": mean_ci([r["conditions"][cname]["kl_vs_jfull"] for r in all_e2]),
        }

    return {
        "n": n,
        "b_offset": b_offset,
        "block_horizon": block_horizon,
        "oracle_clean_rate": sum(1 for r in all_e2 if r.get("oracle_clean", False)) / max(1, n),
        "oracle_first_token_clean_rate": sum(1 for r in all_e2 if r.get("oracle_first_token_clean", False)) / max(1, n),
        "epsilon_H_theoretical": 0.0,
        "epsilon_cross_kl": mean_ci([r["epsilon_cross_kl"] for r in all_e2]),
        "epsilon_cross_logit_diff": mean_ci([r["epsilon_cross_logit_diff"] for r in all_e2]),
        "jcontrol_vs_recompute_kl": mean_ci([r["jcontrol_vs_recompute_kl"] for r in all_e2]),
        "timing_ms": {
            "jlocal": mean_ci([r["timing_ms"]["jlocal"] for r in all_e2]),
            "jlocal_recompute": mean_ci([r["timing_ms"]["jlocal_recompute"] for r in all_e2]),
            "jfull": mean_ci([r["timing_ms"]["jfull"] for r in all_e2]),
        },
        "conditions": agg_conds,
    }


def build_multifact_prompt(
    tok: Any,
    tpl_id: int,
    m_facts: int,
    fact_pairs: list[tuple[str, str, str]],
    filler_between: int = 12,
    post_fillers: int = 16,
) -> dict[str, Any]:
    """Build an M-fact prompt where each fact span [p0_i, p1_i) is well separated and verified."""
    blocks = []
    # Pre-filler so fact 0 sits past C=256 when possible
    pre = [PRE_FILLER_TEMPLATES[i % len(PRE_FILLER_TEMPLATES)].format(idx=i + 1) for i in range(24)]
    blocks.extend(pre)
    for fi in range(m_facts):
        name, old_val, _ = fact_pairs[fi]
        blocks.append(format_fact_sentence(tpl_id, name, old_val))
        n_post = post_fillers if fi == m_facts - 1 else filler_between
        for j in range(n_post):
            blocks.append(
                POST_FILLER_TEMPLATES[(fi * 7 + j) % len(POST_FILLER_TEMPLATES)].format(idx=fi * 100 + j + 1)
            )
    dummy_q = format_query_suffix(tpl_id, fact_pairs[0][0], "4 digits")
    user_old = "\n".join(blocks + [dummy_q])
    full_old = render_chat_prompt(tok, user_old)
    q_marker = "\n" + dummy_q
    ctx_str_old = full_old[: full_old.rfind(q_marker) + 1]

    enc_ctx_o = tok(ctx_str_old, return_offsets_mapping=True, add_special_tokens=False)
    ids_ctx_o = list(enc_ctx_o["input_ids"])
    T_ctx = len(ids_ctx_o)

    # Individually verify each fact slot i in {0..m_facts-1}
    fact_metas = []
    for fi in range(m_facts):
        name, old_val, new_val = fact_pairs[fi]
        fmt_name, fmt_desc, expected_len = get_format_spec(old_val, new_val)
        old_sent = format_fact_sentence(tpl_id, name, old_val)
        new_sent = format_fact_sentence(tpl_id, name, new_val)
        assert ctx_str_old.count(old_sent) == 1
        ctx_str_single = ctx_str_old.replace(old_sent, new_sent, 1)
        ids_single = tok(ctx_str_single, add_special_tokens=False)["input_ids"]
        diffs = [k for k in range(min(len(ids_ctx_o), len(ids_single))) if ids_ctx_o[k] != ids_single[k]]
        if len(ids_single) != T_ctx or not diffs or diffs != list(range(diffs[0], diffs[-1] + 1)) or len(diffs) != expected_len:
            return {"ok": False, "reason": f"fact_{fi}_span_mismatch"}
        p0_i, p1_i = diffs[0], diffs[-1] + 1
        q_suf_str = format_query_suffix(tpl_id, name, fmt_desc)
        full_probe = render_chat_prompt(tok, "\n".join(blocks + [q_suf_str]))
        suf_str = full_probe[full_probe.rfind("\n" + q_suf_str) + 1 :]
        ids_suf = tok(suf_str, add_special_tokens=False)["input_ids"]
        fact_metas.append({
            "fact_idx": fi,
            "name": name,
            "old": old_val,
            "new": new_val,
            "format": fmt_name,
            "p0": p0_i,
            "p1": p1_i,
            "new_span_ids": ids_single[p0_i:p1_i],
            "old_span_ids": ids_ctx_o[p0_i:p1_i],
            "query_suffix_ids": ids_suf,
        })

    return {
        "ok": True,
        "tpl_id": tpl_id,
        "m_facts": m_facts,
        "T_ctx": T_ctx,
        "ids_ctx_o": ids_ctx_o,
        "facts": fact_metas,
    }


def run_e3_suite(
    model_key: str,
    model: Any,
    tok: Any,
    cfg: Any,
    mixers: list[tuple[int, Any]],
    device: torch.device,
    eos_ids: list[int],
    k2_n: int,
    k4_n: int,
    block_horizon: str,
    custom_offset: int,
    bckpt_chunk: int,
    jsonl_path: Path,
    synthetic_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    n_linear = len(mixers)
    w_conv = int(getattr(cfg, "mamba_d_conv", getattr(cfg, "conv_kernel_size", 4)))
    b_offset = resolve_block_offset(block_horizon, n_linear, w_conv, custom_offset)
    done_keys = load_completed_keys(jsonl_path)

    e3_configs = []
    if synthetic_cases is not None:
        e3_configs = synthetic_cases
    else:
        # M=4, k=2 (all 2 perms) x k2_n sets
        for s_idx in range(k2_n):
            pairs = [(NAMES[s_idx * 4 + j], DIGIT4_PAIRS[s_idx * 4 + j][0], DIGIT4_PAIRS[s_idx * 4 + j][1]) for j in range(4)]
            mf = build_multifact_prompt(tok, s_idx % 3, 4, pairs, filler_between=12, post_fillers=16)
            if mf["ok"]:
                e3_configs.append({"set_id": f"M4_k2_{s_idx}", "edit_indices": [1, 2], "perms": [[1, 2], [2, 1]], "mf": mf})
        # M=8, k=4 (P=6 perms) x k4_n sets
        perms_k4 = [
            [1, 3, 5, 6],
            [6, 5, 3, 1],
            [1, 5, 3, 6],
            [6, 3, 5, 1],
            [3, 6, 1, 5],
            [5, 1, 6, 3],
        ]
        for s_idx in range(k4_n):
            pairs = [(NAMES[(s_idx * 8 + j) % len(NAMES)], DIGIT4_PAIRS[(s_idx * 8 + j) % len(DIGIT4_PAIRS)][0], DIGIT4_PAIRS[(s_idx * 8 + j) % len(DIGIT4_PAIRS)][1]) for j in range(8)]
            mf = build_multifact_prompt(tok, s_idx % 3, 8, pairs, filler_between=12, post_fillers=16)
            if mf["ok"]:
                e3_configs.append({"set_id": f"M8_k4_{s_idx}", "edit_indices": [1, 3, 5, 6], "perms": perms_k4, "mf": mf})

    records = []
    for item in e3_configs:
        mf = item["mf"]
        T_ctx = mf["T_ctx"]
        ids_o_ctx = torch.tensor([mf["ids_ctx_o"]], device=device, dtype=torch.long)
        facts = mf["facts"]
        edit_set = set(item["edit_indices"])
        collat_indices = [i for i in range(len(facts)) if i not in edit_set]

        # Build gold edited full sequence for oracle/J-full comparison
        ids_gold = ids_o_ctx.clone()
        for fi in item["edit_indices"]:
            fm = facts[fi]
            ids_gold[:, fm["p0"] : fm["p1"]] = torch.tensor([fm["new_span_ids"]], device=device, dtype=torch.long)

        # Precompute old context cache + per-token g_step across [0, T_ctx) ONCE per sample set
        with torch.no_grad():
            c_old_ctx = DynamicCache(config=cfg)
            _, _, g_tok_old = run_segment_with_mamba_decay(
                model, c_old_ctx, ids_o_ctx, 0, mixers, record_per_token=True
            )
            c_oracle = DynamicCache(config=cfg)
            _ = run_segment(model, c_oracle, ids_gold, 0)

        for perm_idx, perm in enumerate(item["perms"]):
            rec_key = f"{model_key}:e3:{item['set_id']}:p{perm_idx}"
            if rec_key in done_keys:
                continue

            with torch.no_grad():
                c_cur = clone_cache(c_old_ctx)
                cur_ids = ids_o_ctx.clone()
                step_block_ranges = []
                for fj in perm:
                    fm = facts[fj]
                    p0_j, p1_j = fm["p0"], fm["p1"]
                    # Keep sequential edit blocks disjoint with neighboring facts when possible
                    next_p0 = facts[fj + 1]["p0"] if fj + 1 < len(facts) else T_ctx
                    b_j = min(T_ctx, min(next_p0 - 1, p1_j + b_offset))
                    b_j = max(b_j, p1_j)
                    step_block_ranges.append((fj, p0_j, p1_j, b_j))

                    c_pre_j, _ = prepare_bckpt_cache(model, cfg, cur_ids, p0_j, chunk_size=bckpt_chunk)
                    c_blk_old = clone_cache(c_pre_j)
                    _ = run_segment(model, c_blk_old, cur_ids[:, p0_j:b_j], p0_j)
                    new_ids = cur_ids.clone()
                    new_ids[:, p0_j:p1_j] = torch.tensor([fm["new_span_ids"]], device=device, dtype=torch.long)
                    c_blk_new = clone_cache(c_pre_j)
                    _ = run_segment(model, c_blk_new, new_ids[:, p0_j:b_j], p0_j)

                    delta_j = compute_state_delta_fp64(c_blk_old, c_blk_new)
                    g_sums_j = {
                        li: g_tok_old[li][:, b_j:T_ctx, :].sum(dim=1, dtype=torch.float64)
                        for li, _ in mixers
                    }
                    c_cur = apply_mamba2_jlocal_update(
                        c_cur, c_blk_new, delta_j, g_sums_j,
                        start_pos=p0_j, block_end=b_j, T_ctx=T_ctx, w_conv=w_conv,
                        use_decay=True, patch_state=True, patch_kv=True,
                    )
                    cur_ids = new_ids
                    del c_pre_j, c_blk_old, c_blk_new

                # Evaluate J-control / J-local-recompute / J-full on canonical perm 0
                c_jfull_e3 = None
                c_jrecomp_e3 = None
                c_jctrl_e3 = None
                if perm_idx == 0:
                    first_p0 = min(facts[i]["p0"] for i in item["edit_indices"])
                    last_fj = max(item["edit_indices"])
                    last_b = min(T_ctx, facts[last_fj]["p1"] + b_offset)
                    c_pre_first, _ = prepare_bckpt_cache(model, cfg, ids_o_ctx, first_p0, chunk_size=bckpt_chunk)

                    c_old_at_lastb = DynamicCache(config=cfg)
                    _ = run_segment(model, c_old_at_lastb, ids_o_ctx[:, :last_b], 0)
                    c_new_at_lastb = clone_cache(c_pre_first)
                    _ = run_segment(model, c_new_at_lastb, ids_gold[:, first_p0:last_b], first_p0)

                    # Decision A J-full: splice [first_p0, last_b) KV + SSM/conv onto c_old_at_lastb, recompute [last_b, T_ctx)
                    c_jfull_e3 = clone_cache(c_old_at_lastb)
                    patch_kv_range(c_jfull_e3, c_new_at_lastb, first_p0, last_b)
                    overwrite_ssm_states(c_jfull_e3, c_new_at_lastb)
                    if T_ctx > last_b:
                        _ = run_segment(model, c_jfull_e3, ids_gold[:, last_b:T_ctx], last_b)

                    c_jctrl_e3 = clone_cache(c_old_at_lastb)
                    overwrite_ssm_states(c_jctrl_e3, c_new_at_lastb)
                    if T_ctx > last_b:
                        _ = run_segment(model, c_jctrl_e3, ids_gold[:, last_b:T_ctx], last_b)

                    c_jrecomp_e3 = clone_cache(c_new_at_lastb)
                    if T_ctx > last_b:
                        _ = run_segment(model, c_jrecomp_e3, ids_gold[:, last_b:T_ctx], last_b)
                    del c_pre_first, c_old_at_lastb, c_new_at_lastb

            # Evaluate all edited target facts and untouched collateral facts
            target_results = {}
            for fi in item["edit_indices"]:
                fm = facts[fi]
                q_suf = torch.tensor([fm["query_suffix_ids"]], device=device, dtype=torch.long)
                ev_jl = eval_suffix_condition(model, tok, c_cur, q_suf, T_ctx, fm["new"], fm["old"], fm["format"], 8, eos_ids)
                ev_ora = eval_suffix_condition(model, tok, c_oracle, q_suf, T_ctx, fm["new"], fm["old"], fm["format"], 8, eos_ids)
                ev_old = eval_suffix_condition(model, tok, c_old_ctx, q_suf, T_ctx, fm["new"], fm["old"], fm["format"], 8, eos_ids)
                res_fi = {
                    "jlocal_acc_new": ev_jl["acc_new"],
                    "jlocal_strict_new": ev_jl["strict_new"],
                    "jlocal_p_new_1st": ev_jl["p_new_1st"],
                    "jlocal_p_old_1st": ev_jl["p_old_1st"],
                    "jlocal_norm_edit_dist": ev_jl["norm_edit_dist"],
                    "oracle_acc_new": ev_ora["acc_new"],
                    "old_acc_new": ev_old["acc_new"],
                    "kl_vs_oracle": kl_div_from_logits(ev_ora["last_logits"], ev_jl["last_logits"]),
                    "ans_mean_kl": mean_ans_kl(ev_ora["ans_logits_new"], ev_jl["ans_logits_new"]),
                }
                if c_jfull_e3 is not None:
                    ev_jf = eval_suffix_condition(model, tok, c_jfull_e3, q_suf, T_ctx, fm["new"], fm["old"], fm["format"], 8, eos_ids)
                    ev_jr = eval_suffix_condition(model, tok, c_jrecomp_e3, q_suf, T_ctx, fm["new"], fm["old"], fm["format"], 8, eos_ids)
                    ev_jc = eval_suffix_condition(model, tok, c_jctrl_e3, q_suf, T_ctx, fm["new"], fm["old"], fm["format"], 8, eos_ids)
                    res_fi["jfull_acc_new"] = ev_jf["acc_new"]
                    res_fi["jrecomp_acc_new"] = ev_jr["acc_new"]
                    res_fi["jcontrol_acc_new"] = ev_jc["acc_new"]
                    res_fi["epsilon_cross_kl"] = kl_div_from_logits(ev_jr["last_logits"], ev_jl["last_logits"])
                target_results[str(fi)] = res_fi

            collateral_results = {}
            for fi in collat_indices:
                fm = facts[fi]
                q_suf = torch.tensor([fm["query_suffix_ids"]], device=device, dtype=torch.long)
                ev_jl = eval_suffix_condition(model, tok, c_cur, q_suf, T_ctx, fm["new"], fm["old"], fm["format"], 8, eos_ids)
                ev_old = eval_suffix_condition(model, tok, c_old_ctx, q_suf, T_ctx, fm["new"], fm["old"], fm["format"], 8, eos_ids)
                collateral_results[str(fi)] = {
                    "jlocal_preserved_old": ev_jl["acc_old"],
                    "old_baseline_acc": ev_old["acc_old"],
                    "kl_vs_old": kl_div_from_logits(ev_old["last_logits"], ev_jl["last_logits"]),
                }

            rec = {
                "record_key": rec_key,
                "suite": "E3",
                "model": model_key,
                "set_id": item["set_id"],
                "M": len(facts),
                "k": len(perm),
                "perm": perm,
                "T_ctx": T_ctx,
                "step_blocks": step_block_ranges,
                "target_mean_acc_new": sum(1 for v in target_results.values() if v["jlocal_acc_new"]) / len(target_results),
                "oracle_mean_acc_new": sum(1 for v in target_results.values() if v["oracle_acc_new"]) / len(target_results),
                "collateral_preservation": sum(1 for v in collateral_results.values() if v["jlocal_preserved_old"]) / max(1, len(collateral_results)),
                "targets": target_results,
                "collateral": collateral_results,
            }
            append_jsonl(jsonl_path, rec)
            records.append(rec)

            del c_cur, c_jfull_e3, c_jrecomp_e3, c_jctrl_e3
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del c_old_ctx, c_oracle, g_tok_old
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_e3 = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("suite") == "E3" and r.get("model") == model_key:
                        all_e3.append(r)

    by_k = {}
    for k_val in (2, 4):
        sub = [r for r in all_e3 if r["k"] == k_val]
        if not sub:
            continue
        eps_cross_vals = [
            v["epsilon_cross_kl"]
            for r in sub
            for v in r["targets"].values()
            if "epsilon_cross_kl" in v
        ]
        jctrl_accs = [
            1.0 if v["jcontrol_acc_new"] else 0.0
            for r in sub
            for v in r["targets"].values()
            if "jcontrol_acc_new" in v
        ]
        by_k[f"k={k_val}"] = {
            "n_runs": len(sub),
            "jlocal_target_acc": mean_ci([r["target_mean_acc_new"] for r in sub]),
            "oracle_target_acc": mean_ci([r["oracle_mean_acc_new"] for r in sub]),
            "collateral_preservation": mean_ci([r["collateral_preservation"] for r in sub]),
            "epsilon_cross_kl_perm0": mean_ci(eps_cross_vals) if eps_cross_vals else None,
            "jcontrol_acc_perm0": mean_ci(jctrl_accs) if jctrl_accs else None,
        }
    return {"n_total_runs": len(all_e3), "by_k": by_k}


def build_long_range_prompt(
    tok: Any,
    tpl_id: int,
    target_name: str,
    old: str,
    new: str,
    target_ctx_tokens: int,
    m_keys: int = 1,
    max_pos_emb: int = 131072,
) -> dict[str, Any]:
    """Construct context calibrated by binary search to land cleanly near target_ctx_tokens."""
    if target_ctx_tokens <= 512:
        n_pre = 6 if m_keys > 1 else 12
    else:
        n_pre = 18 if m_keys > 1 else 22

    pre_lines = [PRE_FILLER_TEMPLATES[i % len(PRE_FILLER_TEMPLATES)].format(idx=i + 1) for i in range(n_pre)]
    if m_keys > 1:
        other_names = [nm for nm in NAMES if nm != target_name]
        for ki in range(1, m_keys):
            nm = other_names[(ki - 1) % len(other_names)]
            cd = DIGIT4_PAIRS[(ki + 8) % len(DIGIT4_PAIRS)][0]
            pre_lines.append(format_fact_sentence(tpl_id, nm, cd))

    fact_old = format_fact_sentence(tpl_id, target_name, old)
    fact_new = format_fact_sentence(tpl_id, target_name, new)
    fmt_name, fmt_desc, expected_len = get_format_spec(old, new)
    q_line = format_query_suffix(tpl_id, target_name, fmt_desc)
    q_marker = "\n" + q_line
    total_cap = min(target_ctx_tokens + 64, max_pos_emb - 16)

    def _build_candidate(n_post: int) -> tuple[int, int, dict[str, Any] | None]:
        post_lines = [
            POST_FILLER_TEMPLATES[i % len(POST_FILLER_TEMPLATES)].format(idx=i + 1)
            for i in range(max(0, n_post))
        ]
        user_old = "\n".join(pre_lines + [fact_old] + post_lines + [q_line])
        user_new = "\n".join(pre_lines + [fact_new] + post_lines + [q_line])
        full_old = render_chat_prompt(tok, user_old)
        full_new = render_chat_prompt(tok, user_new)
        q_pos_o = full_old.rfind(q_marker)
        q_pos_n = full_new.rfind(q_marker)
        fact_ctx_o = full_old[: q_pos_o + 1]
        fact_ctx_n = full_new[: q_pos_n + 1]
        ids_o = tok(full_old, add_special_tokens=False)["input_ids"]
        ids_n = tok(full_new, add_special_tokens=False)["input_ids"]
        ctx_o = tok(fact_ctx_o, add_special_tokens=False)["input_ids"]
        ctx_n = tok(fact_ctx_n, add_special_tokens=False)["input_ids"]
        T_ctx = len(ctx_o)
        T_tot = len(ids_o)
        diffs = [i for i in range(min(len(ids_o), len(ids_n))) if ids_o[i] != ids_n[i]]
        clean = (
            len(ctx_o) == len(ctx_n)
            and len(ids_o) == len(ids_n)
            and ids_o[:T_ctx] == ctx_o
            and ids_n[:T_ctx] == ctx_n
            and ids_o[T_ctx:] == ids_n[T_ctx:]
            and bool(diffs)
            and diffs == list(range(diffs[0], diffs[-1] + 1))
            and len(diffs) == expected_len
            and diffs[-1] + 1 <= T_ctx
        )
        if not clean:
            return T_ctx, T_tot, None
        case = {
            "ok": True,
            "tpl_id": tpl_id,
            "format": fmt_name,
            "target_name": target_name,
            "old": old,
            "new": new,
            "expected_len": expected_len,
            "p0": diffs[0],
            "p1": diffs[-1] + 1,
            "T_ctx": T_ctx,
            "T_total": T_tot,
            "ids_o": ids_o,
            "ids_n": ids_n,
            "m_keys": m_keys,
        }
        return T_ctx, T_tot, case

    t_ctx0, _, _ = _build_candidate(0)
    t_ctx16, _, _ = _build_candidate(16)
    tok_per_line = max(8.0, (t_ctx16 - t_ctx0) / 16.0)
    lo = 1
    hi = max(16, int((target_ctx_tokens - t_ctx0) / tok_per_line) + 32)

    best_below: tuple[int, dict[str, Any]] | None = None
    best_near: tuple[int, dict[str, Any]] | None = None
    best_n = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        t_mid, tot_mid, case_mid = _build_candidate(mid)
        if case_mid is not None and tot_mid <= total_cap:
            dist = abs(t_mid - target_ctx_tokens)
            if best_near is None or dist < best_near[0]:
                best_near = (dist, case_mid)
        if t_mid <= target_ctx_tokens and tot_mid <= total_cap:
            if case_mid is not None:
                if best_below is None or t_mid > best_below[0]:
                    best_below = (t_mid, case_mid)
            best_n = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Local +/-2 refinement around best_n to land as close to target_ctx_tokens as possible
    for cand_n in range(max(1, best_n - 2), best_n + 3):
        t_c, tot_c, case_c = _build_candidate(cand_n)
        if case_c is not None and tot_c <= total_cap:
            dist = abs(t_c - target_ctx_tokens)
            if best_near is None or dist < best_near[0]:
                best_near = (dist, case_c)
            if t_c <= target_ctx_tokens and (best_below is None or t_c > best_below[0]):
                best_below = (t_c, case_c)

    chosen = best_below[1] if best_below is not None else (best_near[1] if best_near is not None else None)
    if chosen is not None:
        return chosen
    return {"ok": False, "reason": f"length_build_failed_L{target_ctx_tokens}_M{m_keys}_base{t_ctx0}"}


def run_e6_suite(
    model_key: str,
    model: Any,
    tok: Any,
    cfg: Any,
    mixers: list[tuple[int, Any]],
    device: torch.device,
    eos_ids: list[int],
    lengths: list[int],
    e6_n: int,
    bckpt_chunk: int,
    jsonl_path: Path,
    synthetic_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Long-range & multi-key transport ablation: stale vs transport-decay vs recomputed."""
    n_linear = len(mixers)
    w_conv = int(getattr(cfg, "mamba_d_conv", getattr(cfg, "conv_kernel_size", 4)))
    max_pos_emb = int(getattr(cfg, "max_position_embeddings", 4096))
    done_keys = load_completed_keys(jsonl_path)

    grid: list[tuple[int, int, str, int]] = []
    if synthetic_cases is not None:
        for idx, item in enumerate(synthetic_cases):
            grid.append((idx, item["L"], item["horizon"], item["M"]))
    else:
        for L in lengths:
            if L + 64 > max_pos_emb:
                continue
            for m_keys in (1, 8):
                grid.append((len(grid), L, "deep", m_keys))
        # Horizon sweep at L=2048 (if 2048 <= max_pos_emb)
        if 2048 + 64 <= max_pos_emb:
            for h in ("single", "stage"):
                grid.append((len(grid), 2048, h, 1))

    for cell_id, L_target, horizon, m_keys in grid:
        b_offset = resolve_block_offset(horizon, n_linear, w_conv)
        for s_idx in range(e6_n):
            rec_key = f"{model_key}:e6:L{L_target}_{horizon}_M{m_keys}_s{s_idx}"
            if rec_key in done_keys:
                continue

            if synthetic_cases is not None:
                case = synthetic_cases[cell_id]["case"]
            else:
                old, new = DIGIT4_PAIRS[(cell_id * 3 + s_idx) % len(DIGIT4_PAIRS)]
                name = NAMES[(cell_id * 3 + s_idx) % len(NAMES)]
                case = build_long_range_prompt(
                    tok,
                    s_idx % 3,
                    name,
                    old,
                    new,
                    target_ctx_tokens=L_target,
                    m_keys=m_keys,
                    max_pos_emb=max_pos_emb,
                )
                if not case["ok"]:
                    raise RuntimeError(
                        f"E6 prompt construction failed on {model_key} cell=L{L_target}_{horizon}_M{m_keys}_s{s_idx}: {case['reason']}"
                    )

            ids_o = torch.tensor([case["ids_o"]], device=device, dtype=torch.long)
            ids_e = torch.tensor([case["ids_n"]], device=device, dtype=torch.long)
            p0, p1, T_ctx = case["p0"], case["p1"], case["T_ctx"]
            block_end = min(T_ctx, p1 + b_offset)
            ids_suffix = ids_e[:, T_ctx:]
            fmt = case["format"]

            with torch.no_grad():
                c_pre, _ = prepare_bckpt_cache(model, cfg, ids_o, p0, chunk_size=bckpt_chunk)
                c_old_b = clone_cache(c_pre)
                _ = run_segment(model, c_old_b, ids_o[:, p0:block_end], p0)
                c_old_ctx = clone_cache(c_old_b)
                _, g_sums, _ = run_segment_with_mamba_decay(
                    model, c_old_ctx, ids_o[:, block_end:T_ctx], block_end, mixers, max_chunk=1024
                )

                t_blk_start = sync_time(device)
                c_new_blk = clone_cache(c_pre)
                _ = run_segment(model, c_new_blk, ids_e[:, p0:block_end], p0)
                t_blk_ms = (sync_time(device) - t_blk_start) * 1000.0
                delta_b_fp64 = compute_state_delta_fp64(c_old_b, c_new_blk)
                c_jlocal = apply_mamba2_jlocal_update(
                    c_old_ctx, c_new_blk, delta_b_fp64, g_sums,
                    start_pos=p0, block_end=block_end, T_ctx=T_ctx, w_conv=w_conv,
                    use_decay=True, patch_state=True, patch_kv=True,
                )
                t_jlocal_ms = (sync_time(device) - t_blk_start) * 1000.0

                c_stale = apply_mamba2_jlocal_update(
                    c_old_ctx, c_new_blk, delta_b_fp64, g_sums,
                    start_pos=p0, block_end=block_end, T_ctx=T_ctx, w_conv=w_conv,
                    use_decay=False, patch_state=True, patch_kv=True,
                )
                c_state_only = apply_mamba2_jlocal_update(
                    c_old_ctx, c_new_blk, delta_b_fp64, g_sums,
                    start_pos=p0, block_end=block_end, T_ctx=T_ctx, w_conv=w_conv,
                    use_decay=True, patch_state=True, patch_kv=False,
                )

                t0 = sync_time(device)
                c_jrecomp = clone_cache(c_new_blk)
                if T_ctx > block_end:
                    _ = run_segment(model, c_jrecomp, ids_o[:, block_end:T_ctx], block_end, max_chunk=1024)
                t_recomp_ms = t_blk_ms + (sync_time(device) - t0) * 1000.0

                c_jctrl = clone_cache(c_old_b)
                overwrite_ssm_states(c_jctrl, c_new_blk)
                if T_ctx > block_end:
                    _ = run_segment(model, c_jctrl, ids_o[:, block_end:T_ctx], block_end, max_chunk=1024)

                c_oracle = DynamicCache(config=cfg)
                _ = run_segment(model, c_oracle, ids_e[:, :T_ctx], 0, max_chunk=1024)

            evals = {}
            for cname, cc, ev_span in [
                ("old", c_old_ctx, None),
                ("oracle", c_oracle, None),
                ("J-local-stale", c_stale, None),
                ("J-local", c_jlocal, None),
                ("J-local-evict", c_state_only, (p0, p1)),
                ("J-local-recompute", c_jrecomp, None),
                ("J-control", c_jctrl, None),
            ]:
                evals[cname] = eval_suffix_condition(
                    model, tok, cc, ids_suffix, T_ctx,
                    case["new"], case["old"], fmt, max_new=8, eos_ids=eos_ids, evict_span=ev_span,
                )

            lg_ora = evals["oracle"]["last_logits"]
            ans_ora = evals["oracle"]["ans_logits_new"]
            lg_recomp = evals["J-local-recompute"]["last_logits"]
            ans_recomp = evals["J-local-recompute"]["ans_logits_new"]
            eps_cross_kl = kl_div_from_logits(lg_recomp, evals["J-local"]["last_logits"])
            stale_kl = kl_div_from_logits(lg_recomp, evals["J-local-stale"]["last_logits"])
            eps_cross_ld = float((lg_recomp - evals["J-local"]["last_logits"]).abs().max().item())

            rec = {
                "record_key": rec_key,
                "suite": "E6",
                "model": model_key,
                "L_target": L_target,
                "T_ctx": T_ctx,
                "M_keys": m_keys,
                "horizon": horizon,
                "b_offset": b_offset,
                "block_end": block_end,
                "suffix_horizon_len": T_ctx - block_end,
                "sample_idx": s_idx,
                "epsilon_H_theoretical": 0.0,
                "epsilon_cross_kl": eps_cross_kl,
                "epsilon_cross_logit_diff": eps_cross_ld,
                "stale_vs_recompute_kl": stale_kl,
                "decay_vs_stale_kl_ratio": eps_cross_kl / max(stale_kl, 1e-12),
                "gamma_summary": summarize_gamma(g_sums),
                "timing_ms": {
                    "jlocal": t_jlocal_ms,
                    "jlocal_recompute": t_recomp_ms,
                },
                "conditions": {
                    k: {
                        "acc_new": v["acc_new"],
                        "acc_old": v["acc_old"],
                        "p_new_1st": v["p_new_1st"],
                        "p_old_1st": v["p_old_1st"],
                        "norm_edit_dist": v["norm_edit_dist"],
                        "seq_logprob_new": v["seq_logprob_new"],
                        "seq_logprob_old": v["seq_logprob_old"],
                        "kl_vs_oracle": kl_div_from_logits(lg_ora, v["last_logits"]),
                        "kl_vs_recompute": kl_div_from_logits(lg_recomp, v["last_logits"]),
                        "ans_mean_kl_vs_oracle": mean_ans_kl(ans_ora, v["ans_logits_new"]),
                        "ans_mean_kl_vs_recompute": mean_ans_kl(ans_recomp, v["ans_logits_new"]),
                        "top1_eq_recompute": v["top1_id"] == evals["J-local-recompute"]["top1_id"],
                    }
                    for k, v in evals.items()
                },
            }
            append_jsonl(jsonl_path, rec)
            del c_pre, c_old_b, c_old_ctx, c_new_blk, c_jlocal, c_stale, c_state_only, c_jrecomp, c_jctrl, c_oracle
            if device.type == "cuda":
                torch.cuda.empty_cache()

    all_e6 = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("suite") == "E6" and r.get("model") == model_key:
                        all_e6.append(r)

    cells: dict[str, Any] = {}
    for r in all_e6:
        ck = f"L{r['L_target']}_{r['horizon']}_M{r['M_keys']}"
        cells.setdefault(ck, []).append(r)

    assert len(cells) == len(grid), f"Expected {len(grid)} E6 cells for {model_key}, got {len(cells)}"

    agg_cells = {}
    for ck, grp in sorted(cells.items()):
        cond_summary = {}
        for cname in ("old", "oracle", "J-local-stale", "J-local", "J-local-evict", "J-local-recompute", "J-control"):
            cond_summary[cname] = {
                "acc_new": sum(1 for r in grp if r["conditions"][cname]["acc_new"]) / len(grp),
                "p_new_1st": mean_ci([r["conditions"][cname]["p_new_1st"] for r in grp]),
                "p_old_1st": mean_ci([r["conditions"][cname]["p_old_1st"] for r in grp]),
                "norm_edit_dist": mean_ci([r["conditions"][cname]["norm_edit_dist"] for r in grp]),
                "seq_logprob_new": mean_ci([r["conditions"][cname]["seq_logprob_new"] for r in grp]),
                "kl_vs_recompute": mean_ci([r["conditions"][cname]["kl_vs_recompute"] for r in grp]),
                "kl_vs_oracle": mean_ci([r["conditions"][cname]["kl_vs_oracle"] for r in grp]),
                "ans_mean_kl_vs_oracle": mean_ci([r["conditions"][cname]["ans_mean_kl_vs_oracle"] for r in grp]),
                "ans_mean_kl_vs_recompute": mean_ci([r["conditions"][cname]["ans_mean_kl_vs_recompute"] for r in grp]),
            }
        agg_cells[ck] = {
            "n": len(grp),
            "mean_T_ctx": sum(r["T_ctx"] for r in grp) / len(grp),
            "mean_suffix_horizon": sum(r["suffix_horizon_len"] for r in grp) / len(grp),
            "epsilon_cross_kl": mean_ci([r["epsilon_cross_kl"] for r in grp]),
            "stale_vs_recompute_kl": mean_ci([r["stale_vs_recompute_kl"] for r in grp]),
            "gamma_mean": sum(r["gamma_summary"]["mean"] for r in grp) / len(grp),
            "timing_ms": {
                "jlocal": mean_ci([r["timing_ms"]["jlocal"] for r in grp]),
                "jlocal_recompute": mean_ci([r["timing_ms"]["jlocal_recompute"] for r in grp]),
            },
            "conditions": cond_summary,
        }
    return {"n_total_records": len(all_e6), "n_expected_cells": len(grid), "cells": agg_cells}


def run_synthetic_dry_run(output_path: str) -> None:
    """CPU-only pure PyTorch synthetic smoke test exercising E2, E3, and E6 pipelines end-to-end."""
    torch.manual_seed(42)
    f_cfg = FalconH1Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        mamba_d_ssm=128, mamba_n_heads=4, mamba_d_head=32, mamba_n_groups=1,
        mamba_d_state=16, mamba_d_conv=4, mamba_chunk_size=16,
        ssm_multipliers=[0.35355, 0.25, 0.17677, 0.5, 0.35355],
    )
    z_cfg = Zamba2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, ffn_hidden_size=128,
        num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=4,
        attention_hidden_size=128, attention_head_dim=32, kv_channels=16,
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, n_mamba_heads=4, mamba_headdim=32,
        mamba_ngroups=1, chunk_size=16, use_mem_rope=True, use_shared_attention_adapter=True,
        use_shared_mlp_adapter=True, num_mem_blocks=1,
        layers_block_type=["mamba", "mamba", "mamba", "hybrid", "mamba"],
        hybrid_layer_ids=[3], adapter_rank=8,
    )

    tmp_jsonl = Path("pilot/w2_crossfamily_synthetic_dryrun_records.jsonl")
    if tmp_jsonl.exists():
        tmp_jsonl.unlink()

    summary = {"mode": "synthetic_cpu_dry_run", "models": {}}
    dev = torch.device("cpu")

    for mkey, cls, cfg in [("synthetic_falcon_h1", FalconH1ForCausalLM, f_cfg), ("synthetic_zamba2", Zamba2ForCausalLM, z_cfg)]:
        mod = cls(cfg).eval()
        mixers = extract_mamba_mixers(mod)
        eps_h = verify_epsilon_h_zero_single_layer(mod, cfg, mixers, dev, torch.float32)

        ids_o = torch.randint(10, 120, (1, 140)).tolist()[0]
        ids_e = list(ids_o)
        ids_e[60:64] = [3, 4, 5, 6]
        syn_e2 = [{
            "tpl_id": 0, "format": "digit4", "old": "1111", "new": "2222", "expected_len": 4,
            "p0": 60, "p1": 64, "T_ctx": 130, "T_total": 140,
            "ids_o": ids_o, "ids_n": ids_e,
        }]
        e2_res = run_e2_suite(
            mkey, mod, None, cfg, mixers, dev, eos_ids=[127],
            n_samples=1, block_horizon="deep", custom_offset=9, bckpt_chunk=32,
            jsonl_path=tmp_jsonl, synthetic_cases=syn_e2,
        )

        # Synthetic E3 M=4 k=2
        syn_e3 = [{
            "set_id": "M4_k2_syn",
            "edit_indices": [1, 2],
            "perms": [[1, 2], [2, 1]],
            "mf": {
                "T_ctx": 130,
                "ids_ctx_o": ids_o[:130],
                "facts": [
                    {"fact_idx": 0, "old": "10", "new": "20", "format": "digit4", "p0": 20, "p1": 24, "new_span_ids": [1,2,3,4], "old_span_ids": ids_o[20:24], "query_suffix_ids": [90, 91, 92, 93, 94]},
                    {"fact_idx": 1, "old": "11", "new": "21", "format": "digit4", "p0": 50, "p1": 54, "new_span_ids": [2,3,4,5], "old_span_ids": ids_o[50:54], "query_suffix_ids": [95, 96, 97, 98, 99]},
                    {"fact_idx": 2, "old": "12", "new": "22", "format": "digit4", "p0": 80, "p1": 84, "new_span_ids": [3,4,5,6], "old_span_ids": ids_o[80:84], "query_suffix_ids": [100, 101, 102, 103, 104]},
                    {"fact_idx": 3, "old": "13", "new": "23", "format": "digit4", "p0": 105, "p1": 109, "new_span_ids": [4,5,6,7], "old_span_ids": ids_o[105:109], "query_suffix_ids": [105, 106, 107, 108, 109]},
                ],
            },
        }]
        e3_res = run_e3_suite(
            mkey, mod, None, cfg, mixers, dev, eos_ids=[127],
            k2_n=1, k4_n=0, block_horizon="stage", custom_offset=9, bckpt_chunk=32,
            jsonl_path=tmp_jsonl, synthetic_cases=syn_e3,
        )

        # Synthetic E6
        syn_e6 = [{"L": 130, "horizon": "stage", "M": 1, "case": syn_e2[0]}]
        e6_res = run_e6_suite(
            mkey, mod, None, cfg, mixers, dev, eos_ids=[127],
            lengths=[130], e6_n=1, bckpt_chunk=32, jsonl_path=tmp_jsonl, synthetic_cases=syn_e6,
        )
        summary["models"][mkey] = {
            "isolated_epsilon_H": eps_h,
            "epsilon_H_zero_pass": eps_h <= 1e-10,
            "e2": e2_res,
            "e3": e3_res,
            "e6": e6_res,
        }

    out = Path(output_path)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if tmp_jsonl.exists():
        tmp_jsonl.unlink()
    print(f"Synthetic E2/E3/E6 CPU dry run PASSED -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-family E2/E3/E6 unified O_local Mamba-2 evaluation")
    parser.add_argument("--model", choices=["falcon-h1", "zamba2", "all"], default="all")
    parser.add_argument("--falcon-id", default=MODEL_REGISTRY["falcon-h1"]["repo_id"])
    parser.add_argument("--zamba2-id", default=MODEL_REGISTRY["zamba2"]["repo_id"])
    parser.add_argument("--suites", nargs="+", choices=["e2", "e3", "e6"], default=["e2", "e3", "e6"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--block-horizon", choices=["deep", "stage", "single", "custom"], default="deep")
    parser.add_argument("--custom-offset", type=int, default=9)
    parser.add_argument("--bckpt-chunk", type=int, default=256)
    parser.add_argument("--e2-n", type=int, default=30)
    parser.add_argument("--e3-k2-n", type=int, default=3)
    parser.add_argument("--e3-k4-n", type=int, default=2)
    parser.add_argument("--e6-n", type=int, default=3)
    parser.add_argument("--e6-lengths", type=int, nargs="+", default=[512, 1024, 2048, 3584, 6144])
    parser.add_argument("--output", default="pilot/w2_crossfamily_e2_e3_e6_results.json")
    parser.add_argument("--jsonl", default="pilot/w2_crossfamily_e2_e3_e6_records.jsonl")
    parser.add_argument("--synthetic-dry-run", action="store_true")
    args = parser.parse_args()

    if args.synthetic_dry_run:
        run_synthetic_dry_run("pilot/w2_crossfamily_synthetic_dryrun_results.json")
        return

    targets = (
        [("falcon-h1", args.falcon_id), ("zamba2", args.zamba2_id)]
        if args.model == "all"
        else [("falcon-h1", args.falcon_id) if args.model == "falcon-h1" else ("zamba2", args.zamba2_id)]
    )

    dev = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    jsonl_path = Path(args.jsonl)
    out_path = Path(args.output)

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": args.device,
        "dtype": args.dtype,
        "block_horizon": args.block_horizon,
        "bckpt_chunk": args.bckpt_chunk,
        "models": {},
    }

    for mkey, repo_id in targets:
        print(f"=== Running {mkey} ({repo_id}) suites={args.suites} ===")
        tok = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=dtype).to(dev).eval()
        cfg = model.config
        mixers = extract_mamba_mixers(model)
        eos_ids = get_eos_ids(model, tok)

        eps_h = verify_epsilon_h_zero_single_layer(model, cfg, mixers, dev, dtype)
        m_res: dict[str, Any] = {
            "repo_id": repo_id,
            "n_linear": len(mixers),
            "isolated_layer0_epsilon_H": eps_h,
            "epsilon_H_theoretical": 0.0,
        }

        if "e2" in args.suites:
            m_res["e2"] = run_e2_suite(
                mkey, model, tok, cfg, mixers, dev, eos_ids,
                n_samples=args.e2_n,
                block_horizon=args.block_horizon,
                custom_offset=args.custom_offset,
                bckpt_chunk=args.bckpt_chunk,
                jsonl_path=jsonl_path,
            )
            results["models"][mkey] = m_res
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        if "e3" in args.suites:
            m_res["e3"] = run_e3_suite(
                mkey, model, tok, cfg, mixers, dev, eos_ids,
                k2_n=args.e3_k2_n,
                k4_n=args.e3_k4_n,
                block_horizon=args.block_horizon,
                custom_offset=args.custom_offset,
                bckpt_chunk=args.bckpt_chunk,
                jsonl_path=jsonl_path,
            )
            results["models"][mkey] = m_res
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        if "e6" in args.suites:
            m_res["e6"] = run_e6_suite(
                mkey, model, tok, cfg, mixers, dev, eos_ids,
                lengths=args.e6_lengths,
                e6_n=args.e6_n,
                bckpt_chunk=args.bckpt_chunk,
                jsonl_path=jsonl_path,
            )
            results["models"][mkey] = m_res
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        del model, tok, mixers
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved cross-family results -> {out_path}")


if __name__ == "__main__":
    main()
