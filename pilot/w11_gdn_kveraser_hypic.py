"""Stage-B Wave 11-2: Faithful SB1 Baselines (KVEraser & HYPIC) on GDN Hybrids (Qwen3.5-2B / 9B).

Reuses verified helpers from `w5_9b_e5_runner`, `w10_p1_3_e4_e7_counterfact`, and `w11_crossfamily_scaleup`.
Compares O_local (`jlocal`) & selective WY (`jlocal_plus`) against:
  1. `kveraser_lora_steer` (arXiv:2606.17034): rank-16 LoRA (r=16, alpha=32) on frozen full-attn
     projections (q/k/v/o), trained via Eq.(6) CE+KL loss on disjoint train split (S-NIAH + Collision + CF),
     leaving suffix KV and GDN recurrent states untouched.
  2. `kveraser_delete_shift`: excise [p0, p1) from full-attn KV and shift suffix keys by -(p1 - p0)
     using exact Qwen3.5 partial MRoPE (rotary_dim=64), leaving GDN state stale.
  3. `local_suffix_repair_15pct`: prefix-conditioned KV repair over [p0, p1 + 0.15*suf_len) without GDN transport.
  4. `hypic_segmented_seam32` (arXiv:2607.01299): chunk-aligned [s0, s1) + W_seam=32 recompute + exact chunk transition,
     plus exact memory accounting (Phi_m d_k x d_k operator vs scalar G_t).
  5. `hypic_unaligned_seam32`: unaligned mid-segment [p0, p1+32) KV splice without GDN suffix transport.
  6. `hypic_chunk_zero_init`: independent segment [s0, seam_end) prefills from zero GDN state + suffix transport.

Across 3 held-out GDN test regimes:
  - `sniah` (long-context S-NIAH, 1536 tokens)
  - `collision` (dense suffix multi-key interference in GDN recurrent subspace)
  - `counterfact` (real naturalistic rows from counterfact.json)

Supports `--synthetic-dry-run` on CPU using a tiny 4-layer Qwen3_5ForCausalLM before touching GPUs.
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
    apply_hybrid_gated_update,
    apply_jlocal_scalar_update,
    build_digit4_pool,
    build_e5_prompt,
    clone_cache,
    compute_state_delta_fp64,
    count_linear_layers,
    eval_from_ctx_cache_in_place,
    extract_kv_slice,
    find_value_token_span,
    format_fact,
    format_query,
    generate_e5_sample_bank,
    get_eos_ids,
    has_logits_to_keep_arg,
    make_position_ids,
    move_cache_device,
    score_condition,
    snap_states,
)
from w10_p1_1_read_evict import eval_from_ctx_cache_with_mask  # noqa: E402
from w10_p1_3_e4_e7_counterfact import (  # noqa: E402
    apply_delta_rope_to_kv_cache,
    load_and_prepare_counterfact,
)
from w11_crossfamily_scaleup import (  # noqa: E402
    LoRALinear,
    attach_kveraser_lora,
    detach_kveraser_lora,
    set_lora_enabled,
)


def wilson_ci(k: int, n: int, z: float = 1.96) -> List[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def load_done_keys(jsonl_path: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if "record_key" in r:
                    done.add(str(r["record_key"]))
            except json.JSONDecodeError:
                continue
    return done


# ---------------------------------------------------------------------------
# Disjoint Benchmark Case Builders (S-NIAH, Entity Collision, CounterFact)
# ---------------------------------------------------------------------------

EXTRA_ENTITIES = [
    "Victor", "Wendy", "Xavier", "Yara", "Zack", "Aaron", "Bella", "Caleb",
    "Diana", "Ethan", "Fiona", "Gavin", "Hannah", "Isaac", "Julia", "Kevin",
    "Laura", "Mason", "Nora", "Oscar", "Paula", "Quentin", "Rachel", "Simon",
    "Tessa", "Uriel", "Valerie", "Walter", "Zoe", "Amber",
]


def build_collision_case(
    tok: Any,
    idx: int,
    n_linear: int,
    rng: random.Random,
    pool_by_d: Dict[str, List[str]],
    n_suffix_entities: int = 20,
    device: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """Build single-fact case whose suffix packs dense same-template entities colliding on key subspace."""
    tpl_id = idx % 3
    target = ["Alice", "Bob", "Carol", "Dave"][idx % 4]
    firsts = rng.sample([str(d) for d in range(1, 10)], 2)
    old_val = rng.choice(pool_by_d[firsts[0]])
    new_val = rng.choice(pool_by_d[firsts[1]])
    f_o = tok.encode(old_val, add_special_tokens=False)[0]
    f_e = tok.encode(new_val, add_special_tokens=False)[0]
    if f_o == f_e:
        return None

    pre_lines = [FILLERS[(idx + j) % len(FILLERS)] for j in range(10)]
    target_o = format_fact(tpl_id, target, old_val)
    target_e = format_fact(tpl_id, target, new_val)

    # Pack same-template entity statements right into the suffix after a short spacer
    suf_lines = [FILLERS[(idx + 12 + j) % len(FILLERS)] for j in range(6)]
    for k in range(n_suffix_entities):
        ent = EXTRA_ENTITIES[(idx * 3 + k) % len(EXTRA_ENTITIES)]
        if ent == target:
            ent = "Operator" + str(k)
        c_k = rng.choice(pool_by_d[str((k % 9) + 1)])
        suf_lines.append(format_fact(tpl_id, ent, c_k))
    suf_lines.extend([FILLERS[(idx + 20 + j) % len(FILLERS)] for j in range(6)])

    q_str = format_query(tpl_id, target)
    user_o = " ".join(pre_lines + [target_o] + suf_lines) + "\n\n" + q_str
    user_e = " ".join(pre_lines + [target_e] + suf_lines) + "\n\n" + q_str

    full_o = apply_chat_no_think(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_o}])
    full_e = apply_chat_no_think(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_e}])
    qo, qe = full_o.rfind(q_str), full_e.rfind(q_str)
    if qo < 0 or qe < 0:
        return None

    ctx_o, ctx_e, q_s = full_o[:qo], full_e[:qe], full_o[qo:]
    enc_o = tok(ctx_o, return_offsets_mapping=True, add_special_tokens=False)
    enc_e = tok(ctx_e, return_offsets_mapping=True, add_special_tokens=False)
    ids_o, ids_e = enc_o["input_ids"], enc_e["input_ids"]
    if len(ids_o) != len(ids_e):
        return None

    fo, fe = ctx_o.find(target_o), ctx_e.find(target_e)
    if fo < 0 or fe < 0:
        return None
    pos_o = find_value_token_span(ctx_o, old_val, fo + target_o.rfind(old_val), enc_o["offset_mapping"])
    pos_e = find_value_token_span(ctx_e, new_val, fe + target_e.rfind(new_val), enc_e["offset_mapping"])
    if pos_o != pos_e or len(pos_o) != 4:
        return None
    p0, p1 = pos_o[0], pos_o[-1] + 1
    if ids_o[:p0] != ids_e[:p0] or ids_o[p1:] != ids_e[p1:]:
        return None
    T_ctx = len(ids_o)
    b_end = min(T_ctx, p1 + 3 * (n_linear - 1))
    return {
        "regime": "collision",
        "case_id": f"col_{idx}",
        "ids_o": torch.tensor([ids_o], dtype=torch.long, device=device),
        "ids_e": torch.tensor([ids_e], dtype=torch.long, device=device),
        "ids_q": torch.tensor([tok(q_s, add_special_tokens=False)["input_ids"]], dtype=torch.long, device=device),
        "p0": p0,
        "p1": p1,
        "b_end": b_end,
        "T_ctx": T_ctx,
        "old_val": old_val,
        "new_val": new_val,
        "f_o": int(f_o),
        "f_e": int(f_e),
    }


def build_gdn_splits(
    tok: Any,
    cf_path: Optional[Path],
    n_linear: int,
    n_train_per_regime: int,
    n_test_per_regime: int,
    sniah_len: int,
    device: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Assemble strictly disjoint train and test splits across S-NIAH, Collision, and CounterFact."""
    rng_tr = random.Random(202611)
    rng_te = random.Random(202622)
    pool_by_d = build_digit4_pool()

    train_cases: List[Dict[str, Any]] = []
    test_cases: List[Dict[str, Any]] = []

    # 1. S-NIAH (disjoint seeds & index ranges)
    tr_bank = generate_e5_sample_bank(n_samples=n_train_per_regime + 5, depths=[0.25, 0.50], seed=88101)
    te_bank = generate_e5_sample_bank(n_samples=n_test_per_regime + 5, depths=[0.25, 0.50], seed=88202)
    for i in range(min(n_train_per_regime, len(tr_bank))):
        p = build_e5_prompt(tok, tr_bank[i], target_length=sniah_len, n_linear=n_linear, block_horizon="deep", device=device)
        train_cases.append({
            "regime": "sniah", "case_id": f"sniah_tr_{i}",
            "ids_o": p["ids_o"], "ids_e": p["ids_e"], "ids_q": p["ids_q"],
            "p0": p["p0"], "p1": p["p1"], "b_end": p["b_end"], "T_ctx": p["T_ctx"],
            "old_val": tr_bank[i]["old"], "new_val": tr_bank[i]["new"],
            "f_o": p["f_o"], "f_e": p["f_e"],
        })
    for i in range(min(n_test_per_regime, len(te_bank))):
        p = build_e5_prompt(tok, te_bank[i], target_length=sniah_len, n_linear=n_linear, block_horizon="deep", device=device)
        test_cases.append({
            "regime": "sniah", "case_id": f"sniah_te_{i}",
            "ids_o": p["ids_o"], "ids_e": p["ids_e"], "ids_q": p["ids_q"],
            "p0": p["p0"], "p1": p["p1"], "b_end": p["b_end"], "T_ctx": p["T_ctx"],
            "old_val": te_bank[i]["old"], "new_val": te_bank[i]["new"],
            "f_o": p["f_o"], "f_e": p["f_e"],
        })

    # 2. Entity Collision (disjoint index offsets + RNGs)
    cnt = 0
    idx = 0
    while cnt < n_train_per_regime and idx < n_train_per_regime * 4:
        c = build_collision_case(tok, idx, n_linear, rng_tr, pool_by_d, n_suffix_entities=18, device=device)
        idx += 1
        if c is not None:
            c["case_id"] = f"col_tr_{cnt}"
            train_cases.append(c)
            cnt += 1
    cnt = 0
    idx = 1000
    while cnt < n_test_per_regime and idx < 1000 + n_test_per_regime * 4:
        c = build_collision_case(tok, idx, n_linear, rng_te, pool_by_d, n_suffix_entities=18, device=device)
        idx += 1
        if c is not None:
            c["case_id"] = f"col_te_{cnt}"
            test_cases.append(c)
            cnt += 1

    # 3. CounterFact (split by index; use high n_target so Colab subset with delta1-first still reaches all eq rows)
    if cf_path is not None and cf_path.exists():
        cf_all = load_and_prepare_counterfact(
            tok, cf_path, max_scan=2500, n_target=900,
            target_ctx_len=600, n_linear=n_linear, device=device, rng=rng_te,
        )
        # Keep delta_L == 0 cleanly for uniform surrogate splice
        cf_eq = [r for r in cf_all if r["delta_L"] == 0]
        for j, r in enumerate(cf_eq[:n_train_per_regime]):
            p1 = r["b_old"] - min(r["b_old"] - r["p0"], 3 * (n_linear - 1))
            train_cases.append({
                "regime": "counterfact", "case_id": f"cf_tr_{r['case_id']}",
                "ids_o": r["ids_o"], "ids_e": r["ids_e"], "ids_q": r["ids_q"],
                "p0": r["p0"], "p1": max(r["p0"] + 1, p1), "b_end": r["b_old"], "T_ctx": r["T_old"],
                "old_val": r["old_val"], "new_val": r["new_val"],
                "f_o": r["f_o"], "f_e": r["f_e"],
            })
        for j, r in enumerate(cf_eq[n_train_per_regime : n_train_per_regime + n_test_per_regime]):
            p1 = r["b_old"] - min(r["b_old"] - r["p0"], 3 * (n_linear - 1))
            test_cases.append({
                "regime": "counterfact", "case_id": f"cf_te_{r['case_id']}",
                "ids_o": r["ids_o"], "ids_e": r["ids_e"], "ids_q": r["ids_q"],
                "p0": r["p0"], "p1": max(r["p0"] + 1, p1), "b_end": r["b_old"], "T_ctx": r["T_old"],
                "old_val": r["old_val"], "new_val": r["new_val"],
                "f_o": r["f_o"], "f_e": r["f_e"],
            })

    # 4. S-NIAH under hard KV eviction of [p0, b_end) (simulating evicted-KV read boundary from P1-1)
    sniah_te = [dict(x) for x in test_cases if x["regime"] == "sniah"]
    for tc in sniah_te:
        tc["regime"] = "sniah_evict"
        tc["case_id"] = tc["case_id"].replace("sniah_te_", "sniah_evict_te_")
        tc["eval_mask_span"] = True
        test_cases.append(tc)

    return train_cases, test_cases


# ---------------------------------------------------------------------------
# GDN Cache Surrogate / Delete&Shift / HYPIC Helpers
# ---------------------------------------------------------------------------

def _detached_cache_clone_for_grad(cache: Any) -> Any:
    """Clone cache keeping linear states detached and copying full-attention KVs."""
    c_out = clone_cache(cache)
    return c_out


def build_gdn_kveraser_surrogate(c_old_ctx: Any, c_lora_span: Any, p0: int, p1: int, T_ctx: int) -> Any:
    """Eq.(4) on Qwen3.5 full-attn layers via torch.cat (keeps grad on KV_steer), leaving GDN state stale."""
    c_surr = clone_cache(c_old_ctx)
    for l_dst, l_old, l_sp in zip(c_surr.layers, c_old_ctx.layers, c_lora_span.layers):
        k_old = getattr(l_old, "keys", None)
        v_old = getattr(l_old, "values", None)
        if torch.is_tensor(k_old) and torch.is_tensor(v_old) and k_old.numel() > 0:
            k_sp = l_sp.keys[:, :, p0:p1, :]
            v_sp = l_sp.values[:, :, p0:p1, :]
            l_dst.keys = torch.cat([k_old[:, :, :p0, :].detach(), k_sp, k_old[:, :, p1:T_ctx, :].detach()], dim=-2)
            l_dst.values = torch.cat([v_old[:, :, :p0, :].detach(), v_sp, v_old[:, :, p1:T_ctx, :].detach()], dim=-2)
    return c_surr


def build_gdn_delete_shift_cache(c_old_ctx: Any, p0: int, p1: int, T_ctx: int, base_model: Any) -> Tuple[Any, int]:
    """KVEraser Delete-and-Shift on GDN: drop [p0, p1) from full-attn KV and apply Delta-RoPE (delta_L = -(p1-p0)); GDN stale."""
    del_len = p1 - p0
    c_del = clone_cache(c_old_ctx)
    empty_blk: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
    for layer in c_del.layers:
        k = getattr(layer, "keys", None)
        if torch.is_tensor(k) and k.numel() > 0:
            empty_blk.append((k.new_empty((k.shape[0], k.shape[1], 0, k.shape[3])),
                              k.new_empty((k.shape[0], k.shape[1], 0, k.shape[3]))))
        else:
            empty_blk.append(None)
    apply_delta_rope_to_kv_cache(
        c_del, empty_blk, p0=p0, b_old=p1, b_new=p0, T_ctx_old=T_ctx,
        delta_L=-del_len, base_model=base_model, apply_delta_rope=True,
    )
    return c_del, T_ctx - del_len


def train_gdn_kveraser_lora(
    model: Any,
    base_model: Any,
    tok: Any,
    adapters: List[LoRALinear],
    train_cases: List[Dict[str, Any]],
    device: str,
    use_l2k: bool,
    epochs: int = 2,
    lr: float = 2e-4,
    grad_clip: float = 1.0,
) -> Dict[str, Any]:
    """Train rank-16 LoRA on Qwen3.5 attention layers with Eq.(6) teacher-forced CE+KL loss."""
    trainable = [p for ad in adapters for p in (ad.lora_A, ad.lora_B)]
    if not trainable or not train_cases:
        return {"train_time_s": 0.0, "steps": 0, "grad_norm_nonzero": False}

    from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
    orig_rec = qm.torch_recurrent_gated_delta_rule
    qm.torch_recurrent_gated_delta_rule = getattr(orig_rec, "__wrapped__", orig_rec)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    t0 = time.perf_counter()
    losses: List[float] = []
    grad_nonzero = False

    try:
      for ep in range(epochs):
        # Shuffle deterministically per epoch
        order = list(range(len(train_cases)))
        random.Random(900 + ep).shuffle(order)
        for idx in order:
            c = train_cases[idx]
            p0, p1, T_ctx = c["p0"], c["p1"], c["T_ctx"]
            ids_o, ids_e, ids_q = c["ids_o"], c["ids_e"], c["ids_q"]
            ans_ids = tok.encode(c["new_val"], add_special_tokens=False) or [c["f_e"]]
            ans_t = torch.tensor([ans_ids], dtype=torch.long, device=device)
            ids_tf = torch.cat([ids_q, ans_t[:, :-1]], dim=1) if ans_t.shape[1] > 1 else ids_q

            with torch.no_grad():
                set_lora_enabled(adapters, False)
                _, c_ora = _run_prefill(model, base_model, ids_e[:, :T_ctx], 0, None, use_l2k)
                pos_tf = make_position_ids(T_ctx, int(ids_tf.shape[1]), device, base_model)
                ora_out = model(input_ids=ids_tf, position_ids=pos_tf, past_key_values=c_ora, use_cache=True)
                ora_ans_logits = ora_out.logits[:, -ans_t.shape[1]:, :].detach()
                del ora_out, c_ora

                _, c_pre = _run_prefill(model, base_model, ids_o[:, :p0], 0, None, use_l2k)
                c_pre_cpu = move_cache_device(c_pre, "cpu")
                _, c_old_ctx = _run_prefill(model, base_model, ids_o[:, p0:T_ctx], p0, c_pre, use_l2k)
                del c_pre
                c_old_ctx_cpu = move_cache_device(c_old_ctx, "cpu")
                del c_old_ctx

            opt.zero_grad(set_to_none=True)
            with torch.autograd.graph.saved_tensors_hooks(
                lambda t: t.clone() if (torch.is_tensor(t) and t.ndim >= 3) else t,
                lambda t: t,
            ):
                set_lora_enabled(adapters, True)
                c_lora = move_cache_device(c_pre_cpu, device)
                del c_pre_cpu
                pos_sp = make_position_ids(p0, p1 - p0, device, base_model)
                _ = model(input_ids=ids_e[:, p0:p1], position_ids=pos_sp, past_key_values=c_lora, use_cache=True)

                c_old_ctx_gpu = move_cache_device(c_old_ctx_cpu, device)
                del c_old_ctx_cpu
                c_surr = build_gdn_kveraser_surrogate(c_old_ctx_gpu, c_lora, p0, p1, T_ctx)
                del c_lora, c_old_ctx_gpu
                set_lora_enabled(adapters, False)
                pos_tf = make_position_ids(T_ctx, int(ids_tf.shape[1]), device, base_model)
                cand_out = model(input_ids=ids_tf, position_ids=pos_tf, past_key_values=c_surr, use_cache=True)
                cand_ans_logits = cand_out.logits[:, -ans_t.shape[1]:, :]

                ce = F.cross_entropy(cand_ans_logits.reshape(-1, cand_ans_logits.shape[-1]).float(), ans_t.reshape(-1))
                lq = F.log_softmax(cand_ans_logits.float(), dim=-1)
                lp = F.log_softmax(ora_ans_logits.float(), dim=-1)
                kl = (lp.exp() * (lp - lq)).sum(dim=-1).mean()
                loss = ce + 0.5 * kl
            loss.backward()

            gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, grad_clip).item())
            if gnorm > 0.0:
                grad_nonzero = True
            opt.step()
            losses.append(float(loss.item()))

            del c_surr, cand_out, ora_ans_logits
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    finally:
        qm.torch_recurrent_gated_delta_rule = orig_rec
        set_lora_enabled(adapters, False)
    return {
        "train_time_s": float(time.perf_counter() - t0),
        "steps": len(losses),
        "initial_loss": float(losses[0]) if losses else 0.0,
        "final_loss": float(losses[-1]) if losses else 0.0,
        "mean_last10_loss": float(sum(losses[-10:]) / max(1, len(losses[-10:]))),
        "grad_norm_nonzero": grad_nonzero,
    }


def _run_prefill(model: Any, base_model: Any, input_ids: torch.Tensor, pos_start: int, cache: Any, use_l2k: bool):
    L = int(input_ids.shape[1])
    if L == 0:
        return None, cache
    last_lg = None
    for s in range(0, L, 1024):
        e = min(L, s + 1024)
        p_ids = make_position_ids(pos_start + s, e - s, input_ids.device, base_model)
        kw = {"input_ids": input_ids[:, s:e], "position_ids": p_ids, "use_cache": True, "past_key_values": cache}
        if use_l2k:
            kw["logits_to_keep"] = 1
        out = model(**kw)
        cache = out.past_key_values
        if e == L:
            last_lg = out.logits[:, -1:, :].detach().clone()
        del out
    return last_lg, cache


def eval_gdn_cache_copy(
    model: Any,
    base_model: Any,
    tok: Any,
    eos_ids: List[int],
    cache: Any,
    T_ctx: int,
    ids_q: torch.Tensor,
    use_l2k: bool,
    new_val: str,
    old_val: str,
    f_e: int,
    f_o: int,
    lg_ora: torch.Tensor,
) -> Dict[str, Any]:
    work = clone_cache(cache)
    ev = eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, work, T_ctx, ids_q, use_l2k, max_new_tokens=8)
    sc = score_condition(ev, new_val, old_val, f_e, f_o, lg_ora)
    del work
    return sc


def compute_gdn_hypic_memory(t_cfg: Any, n_linear: int, d_k: int, T_ctx: int, C_seg: int = 128) -> Dict[str, Any]:
    n_heads = int(getattr(t_cfg, "linear_num_value_heads", 32))
    n_segs = max(1, (T_ctx + C_seg - 1) // C_seg)
    hypic_phi_bytes_per_seg = n_linear * n_heads * d_k * d_k * 2
    hypic_total_bytes = n_segs * hypic_phi_bytes_per_seg
    olocal_scalar_per_step = n_linear * n_heads * 4
    olocal_dense_total_bytes = T_ctx * olocal_scalar_per_step
    olocal_strided_total_bytes = n_segs * olocal_scalar_per_step
    return {
        "T_ctx": int(T_ctx), "C_seg": int(C_seg), "n_segs": int(n_segs),
        "n_linear": int(n_linear), "n_heads": int(n_heads), "d_k": int(d_k),
        "hypic_phi_bytes_per_seg": int(hypic_phi_bytes_per_seg),
        "hypic_total_mib_for_T": float(hypic_total_bytes / (1024 ** 2)),
        "hypic_kib_per_token": float((hypic_phi_bytes_per_seg / max(1, C_seg)) / 1024.0),
        "olocal_scalar_bytes_per_step": int(olocal_scalar_per_step),
        "olocal_dense_total_mib_for_T": float(olocal_dense_total_bytes / (1024 ** 2)),
        "olocal_dense_kib_per_token": float(olocal_scalar_per_step / 1024.0),
        "olocal_strided_total_mib_for_T": float(olocal_strided_total_bytes / (1024 ** 2)),
        "normalized_dense_per_tok_ratio": float(hypic_total_bytes / max(1, olocal_dense_total_bytes)),
        "normalized_matched_stride_ratio": float(hypic_phi_bytes_per_seg / max(1, olocal_scalar_per_step)),
    }


def evaluate_gdn_sb1_suite(
    model: Any,
    base_model: Any,
    tok: Any,
    t_cfg: Any,
    adapters: List[LoRALinear],
    test_cases: List[Dict[str, Any]],
    device: str,
    use_l2k: bool,
    model_tag: str,
    jsonl_path: Path,
    c_seg: int = 128,
    w_seam: int = 32,
) -> List[Dict[str, Any]]:
    n_linear = count_linear_layers(t_cfg)
    d_k = int(getattr(t_cfg, "linear_key_head_dim", 128))
    w_conv = int(getattr(t_cfg, "linear_conv_kernel_dim", 4))
    eos_ids = get_eos_ids(model, tok)
    done = load_done_keys(jsonl_path)

    records: List[Dict[str, Any]] = []
    cond_names = [
        "oracle", "old", "J-local", "J-local+", "kv_only_patched_blk",
        "kveraser_lora_steer", "kveraser_delete_shift", "local_suffix_repair_15pct",
        "hypic_segmented_seam32", "hypic_unaligned_seam32", "hypic_chunk_zero_init",
    ]

    for idx, c in enumerate(test_cases):
        rkey = f"{model_tag}:{c['case_id']}"
        if rkey in done:
            continue

        ids_o, ids_e, ids_q = c["ids_o"], c["ids_e"], c["ids_q"]
        p0, p1, b_end, T_ctx = c["p0"], c["p1"], c["b_end"], c["T_ctx"]
        suf_len = T_ctx - b_end
        s0 = (p0 // c_seg) * c_seg
        s1 = min(T_ctx, s0 + c_seg)
        seam_end = min(T_ctx, s1 + w_seam)
        rep15_end = min(T_ctx, max(b_end, p1 + max(9, int(0.15 * max(1, suf_len)))))
        unalign_end = min(T_ctx, p1 + w_seam)

        with torch.no_grad():
            set_lora_enabled(adapters, False)
            cond_scores: Dict[str, Any] = {}

            def _eval_maybe_masked(cache_obj: Any, t_len: int) -> Dict[str, Any]:
                if not c.get("eval_mask_span"):
                    return eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, cache_obj, t_len, ids_q, use_l2k, max_new_tokens=8)
                mask_ctx = torch.ones((1, t_len), dtype=torch.long, device=device)
                m_end = min(t_len, b_end if t_len == T_ctx else max(p0, b_end - (p1 - p0)))
                mask_ctx[:, p0:m_end] = 0
                return eval_from_ctx_cache_with_mask(model, base_model, tok, eos_ids, cache_obj, t_len, ids_q, attn_mask_ctx=mask_ctx, use_logits_to_keep=use_l2k, max_new_tokens=8)

            def _score_and_free(cname: str, cache_obj: Any, t_len: int) -> None:
                ev = _eval_maybe_masked(cache_obj, t_len)
                cond_scores[cname] = score_condition(ev, c["new_val"], c["old_val"], c["f_e"], c["f_o"], lg_ora)
                del cache_obj

            # 1. Oracle
            _, c_ora = _run_prefill(model, base_model, ids_e, 0, None, use_l2k)
            ev_ora = _eval_maybe_masked(c_ora, T_ctx)
            lg_ora = ev_ora["first_logits"].detach().clone()
            cond_scores["oracle"] = score_condition(ev_ora, c["new_val"], c["old_val"], c["f_e"], c["f_o"], lg_ora)
            del c_ora

            # 2. Old + suffix GDN collectors (single active GPU cache, CPU host backup)
            _, c_pre = _run_prefill(model, base_model, ids_o[:, :p0], 0, None, use_l2k)
            c_pre_cpu = move_cache_device(c_pre, "cpu")
            _, c_old_b = _run_prefill(model, base_model, ids_o[:, p0:b_end], p0, c_pre, use_l2k)
            del c_pre
            s_old_b = snap_states(c_old_b)
            c_ctx_old = c_old_b
            with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=False) as col:
                if suf_len > 0:
                    _, c_ctx_old = _run_prefill(model, base_model, ids_o[:, b_end:T_ctx], b_end, c_ctx_old, use_l2k)
            hstats = col.finalize(expected_suffix_len=suf_len, device=device)
            c_ctx_old_cpu = move_cache_device(c_ctx_old, "cpu")
            _score_and_free("old", c_ctx_old, T_ctx)

            # 3. Local block new -> J-local, J-local+, kv_only (evaluate sequentially & free immediately!)
            t0 = time.perf_counter()
            _, c_new_b = _run_prefill(model, base_model, ids_e[:, p0:b_end], p0, move_cache_device(c_pre_cpu, device), use_l2k)
            s_new_b = snap_states(c_new_b)
            new_blk_kv = extract_kv_slice(c_new_b, p0, b_end)
            del c_new_b
            delta = compute_state_delta_fp64(s_old_b, s_new_b)

            c_jl = move_cache_device(c_ctx_old_cpu, device)
            apply_jlocal_scalar_update(c_jl, delta, hstats["gamma_scalar"], b_end, T_ctx, w_conv, use_decay=True)
            for layer_i, sl in enumerate(new_blk_kv):
                if sl is not None:
                    c_jl.layers[layer_i].keys[:, :, p0:b_end, :].copy_(sl[0])
                    c_jl.layers[layer_i].values[:, :, p0:b_end, :].copy_(sl[1])
            t_jl_ms = (time.perf_counter() - t0) * 1000.0
            _score_and_free("J-local", c_jl, T_ctx)

            c_jlp = move_cache_device(c_ctx_old_cpu, device)
            apply_hybrid_gated_update(c_jlp, delta, hstats, b_end, T_ctx, device=device, tau_c=0.05, tau_gamma=1e-4, w_conv=w_conv)
            for layer_i, sl in enumerate(new_blk_kv):
                if sl is not None:
                    c_jlp.layers[layer_i].keys[:, :, p0:b_end, :].copy_(sl[0])
                    c_jlp.layers[layer_i].values[:, :, p0:b_end, :].copy_(sl[1])
            _score_and_free("J-local+", c_jlp, T_ctx)

            c_kvonly = move_cache_device(c_ctx_old_cpu, device)
            for layer_i, sl in enumerate(new_blk_kv):
                if sl is not None:
                    c_kvonly.layers[layer_i].keys[:, :, p0:b_end, :].copy_(sl[0])
                    c_kvonly.layers[layer_i].values[:, :, p0:b_end, :].copy_(sl[1])
            _score_and_free("kv_only_patched_blk", c_kvonly, T_ctx)
            del new_blk_kv, hstats

            # 4. KVEraser LoRA Steer
            t0 = time.perf_counter()
            set_lora_enabled(adapters, True)
            _, c_lora = _run_prefill(model, base_model, ids_e[:, p0:p1], p0, move_cache_device(c_pre_cpu, device), use_l2k)
            set_lora_enabled(adapters, False)
            c_old_tmp = move_cache_device(c_ctx_old_cpu, device)
            c_kve = build_gdn_kveraser_surrogate(c_old_tmp, c_lora, p0, p1, T_ctx)
            del c_lora, c_old_tmp
            t_kve_ms = (time.perf_counter() - t0) * 1000.0
            _score_and_free("kveraser_lora_steer", c_kve, T_ctx)

            # 5. KVEraser Delete-and-Shift
            c_old_tmp = move_cache_device(c_ctx_old_cpu, device)
            c_del, T_del = build_gdn_delete_shift_cache(c_old_tmp, p0, p1, T_ctx, base_model)
            del c_old_tmp
            _score_and_free("kveraser_delete_shift", c_del, T_del)

            # 6. Local suffix repair 15%
            _, c_rep15_blk = _run_prefill(model, base_model, ids_e[:, p0:rep15_end], p0, move_cache_device(c_pre_cpu, device), use_l2k)
            rep15_kv = extract_kv_slice(c_rep15_blk, p0, rep15_end)
            del c_rep15_blk
            c_rep15 = move_cache_device(c_ctx_old_cpu, device)
            for layer_i, sl in enumerate(rep15_kv):
                if sl is not None:
                    c_rep15.layers[layer_i].keys[:, :, p0:rep15_end, :].copy_(sl[0])
                    c_rep15.layers[layer_i].values[:, :, p0:rep15_end, :].copy_(sl[1])
            del rep15_kv
            _score_and_free("local_suffix_repair_15pct", c_rep15, T_ctx)

            # 7. HYPIC segmented + W_seam=32
            t0 = time.perf_counter()
            _, c_s0 = _run_prefill(model, base_model, ids_o[:, :s0], 0, None, use_l2k)
            c_s0_cpu = move_cache_device(c_s0, "cpu")
            _, c_hyp_o = _run_prefill(model, base_model, ids_o[:, s0:seam_end], s0, c_s0, use_l2k)
            s_hyp_o = snap_states(c_hyp_o)
            with GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as col_h:
                if T_ctx > seam_end:
                    _, _ = _run_prefill(model, base_model, ids_o[:, seam_end:T_ctx], seam_end, c_hyp_o, use_l2k)
            del c_hyp_o
            hstats_h = col_h.finalize(expected_suffix_len=max(0, T_ctx - seam_end), device=device)
            _, c_hyp_n = _run_prefill(model, base_model, ids_e[:, s0:seam_end], s0, move_cache_device(c_s0_cpu, device), use_l2k)
            del c_s0_cpu
            delta_hyp = compute_state_delta_fp64(s_hyp_o, snap_states(c_hyp_n))
            hyp_kv = extract_kv_slice(c_hyp_n, s0, seam_end)
            del c_hyp_n
            c_hypic_seg = move_cache_device(c_ctx_old_cpu, device)
            apply_jlocal_scalar_update(c_hypic_seg, delta_hyp, hstats_h["gamma_scalar"], seam_end, T_ctx, w_conv, use_decay=True)
            for layer_i, sl in enumerate(hyp_kv):
                if sl is not None:
                    c_hypic_seg.layers[layer_i].keys[:, :, s0:seam_end, :].copy_(sl[0])
                    c_hypic_seg.layers[layer_i].values[:, :, s0:seam_end, :].copy_(sl[1])
            del hyp_kv
            t_hypic_ms = (time.perf_counter() - t0) * 1000.0
            _score_and_free("hypic_segmented_seam32", c_hypic_seg, T_ctx)

            # 8. HYPIC unaligned seam32 without transport
            _, c_unal_blk = _run_prefill(model, base_model, ids_e[:, p0:unalign_end], p0, move_cache_device(c_pre_cpu, device), use_l2k)
            unal_kv = extract_kv_slice(c_unal_blk, p0, unalign_end)
            del c_unal_blk, c_pre_cpu
            c_hypic_unal = move_cache_device(c_ctx_old_cpu, device)
            for layer_i, sl in enumerate(unal_kv):
                if sl is not None:
                    c_hypic_unal.layers[layer_i].keys[:, :, p0:unalign_end, :].copy_(sl[0])
                    c_hypic_unal.layers[layer_i].values[:, :, p0:unalign_end, :].copy_(sl[1])
            del unal_kv
            _score_and_free("hypic_unaligned_seam32", c_hypic_unal, T_ctx)

            # 9. HYPIC zero-init chunk
            _, c_hz_o = _run_prefill(model, base_model, ids_o[:, s0:seam_end], s0, None, use_l2k)
            s_hz_o = snap_states(c_hz_o)
            del c_hz_o
            _, c_hz_n = _run_prefill(model, base_model, ids_e[:, s0:seam_end], s0, None, use_l2k)
            delta_hz = compute_state_delta_fp64(s_hz_o, snap_states(c_hz_n))
            hz_kv = extract_kv_slice(c_hz_n, 0, seam_end - s0)
            del c_hz_n
            c_hypic_zero = move_cache_device(c_ctx_old_cpu, device)
            del c_ctx_old_cpu
            apply_jlocal_scalar_update(c_hypic_zero, delta_hz, hstats_h["gamma_scalar"], seam_end, T_ctx, w_conv, use_decay=True)
            for layer_i, sl in enumerate(hz_kv):
                if sl is not None:
                    c_hypic_zero.layers[layer_i].keys[:, :, s0:seam_end, :].copy_(sl[0])
                    c_hypic_zero.layers[layer_i].values[:, :, s0:seam_end, :].copy_(sl[1])
            del hz_kv
            _score_and_free("hypic_chunk_zero_init", c_hypic_zero, T_ctx)

        rec = {
            "record_key": rkey,
            "model": model_tag,
            "regime": c["regime"],
            "case_id": c["case_id"],
            "p0": p0, "p1": p1, "b_end": b_end, "T_ctx": T_ctx,
            "tokens_recomputed": {
                "J-local": int(b_end - p0),
                "kveraser_lora_steer": int(p1 - p0),
                "local_suffix_repair_15pct": int(rep15_end - p0),
                "hypic_segmented_seam32": int(seam_end - s0),
                "hypic_unaligned_seam32": int(unalign_end - p0),
            },
            "timing_ms": {"J-local": t_jl_ms, "kveraser_lora_steer": t_kve_ms, "hypic_segmented_seam32": t_hypic_ms},
            "conditions": cond_scores,
        }
        append_record_jsonl(jsonl_path, rec)
        records.append(rec)

        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    return records


def aggregate_sb1_gdn(jsonl_path: Path, model_tag: str) -> Dict[str, Any]:
    all_recs = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("model") == model_tag:
                        all_recs.append(r)

    regimes = sorted({r["regime"] for r in all_recs})
    out: Dict[str, Any] = {"n_total": len(all_recs), "overall": {}, "by_regime": {}}
    cond_names = list(all_recs[0]["conditions"].keys()) if all_recs else []

    def _agg(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(sub)
        res = {"n": n, "conditions": {}}
        for cn in cond_names:
            k_em = sum(1 for r in sub if r["conditions"][cn]["contains_new"])
            k_old = sum(1 for r in sub if r["conditions"][cn]["contains_old"])
            k_top1 = sum(1 for r in sub if r["conditions"][cn]["top1_eq_oracle"])
            kls = [r["conditions"][cn]["kl_vs_oneshot"] for r in sub]
            res["conditions"][cn] = {
                "em_rate": k_em / max(1, n),
                "em_ci95": wilson_ci(k_em, n),
                "stale_rate": k_old / max(1, n),
                "top1_eq_oracle": k_top1 / max(1, n),
                "mean_kl": sum(kls) / max(1, n),
            }
        res["mean_tokens_recomputed"] = {
            k: sum(r["tokens_recomputed"][k] for r in sub) / max(1, n)
            for k in ("J-local", "kveraser_lora_steer", "local_suffix_repair_15pct", "hypic_segmented_seam32", "hypic_unaligned_seam32")
        }
        return res

    out["overall"] = _agg(all_recs)
    for reg in regimes:
        out["by_regime"][reg] = _agg([r for r in all_recs if r["regime"] == reg])
    return out


def run_synthetic_dry_run(out_dir: Path) -> Dict[str, Any]:
    torch.manual_seed(0)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
    if hasattr(qm.torch_chunk_gated_delta_rule, "__wrapped__"):
        qm.torch_chunk_gated_delta_rule = qm.torch_chunk_gated_delta_rule.__wrapped__
    if hasattr(qm.torch_recurrent_gated_delta_rule, "__wrapped__"):
        qm.torch_recurrent_gated_delta_rule = qm.torch_recurrent_gated_delta_rule.__wrapped__
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B", local_files_only=True)
    tc = getattr(cfg, "text_config", cfg)
    tc.num_hidden_layers = 4
    tc.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    tc.vocab_size = 128
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

    class _MockTok:
        eos_token_id = 0
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(x) for x in ids)
        def encode(self, s, add_special_tokens=False):
            return [int(x) % 120 + 1 if x.isdigit() else (sum(ord(c) for c in x) % 120 + 1) for x in s.split()] or [5]
        def convert_tokens_to_ids(self, s):
            return 0

    tok = _MockTok()
    model = Qwen3_5ForCausalLM(tc).eval()
    base_model = model.model
    adapters, lora_p, tot_p = attach_kveraser_lora(model, rank=16, alpha=32.0)

    cases = []
    for i, span_w in enumerate((4, 1, 4)):
        ids_o = torch.randint(10, 120, (1, 96), dtype=torch.long)
        ids_e = ids_o.clone()
        ids_e[0, 32:32 + span_w] = (ids_o[0, 32:32 + span_w] + 11) % 120 + 1
        cases.append({
            "regime": "sniah" if span_w == 4 else "counterfact", "case_id": f"dry_{i}",
            "ids_o": ids_o, "ids_e": ids_e, "ids_q": torch.tensor([[5, 6, 7, 8]]),
            "p0": 32, "p1": 32 + span_w, "b_end": 45, "T_ctx": 96,
            "old_val": "10", "new_val": "21", "f_o": 10, "f_e": 21,
        })

    tr_stats = train_gdn_kveraser_lora(model, base_model, tok, adapters, cases[:2], "cpu", False, epochs=1)
    assert tr_stats["grad_norm_nonzero"], "Zero LoRA gradient on Qwen3.5 synthetic dry-run!"
    tmp_p = out_dir / "_tmp_gdn_dry.jsonl"
    tmp_p.unlink(missing_ok=True)
    recs = evaluate_gdn_sb1_suite(model, base_model, tok, tc, adapters, cases[2:], "cpu", False, "syn_qwen", tmp_p, c_seg=32, w_seam=8)
    tmp_p.unlink(missing_ok=True)
    detach_kveraser_lora(model)

    rep = {"grad_norm_nonzero": tr_stats["grad_norm_nonzero"], "lora_params": lora_p, "total_params": tot_p, "n_eval": len(recs)}
    (out_dir / "w11_gdn_sb1_dryrun.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    p.add_argument("--tag", type=str, default="2b")
    p.add_argument("--cf-path", type=str, default=r"D:\applicationData\HuggingFace\counterfact.json")
    p.add_argument("--sniah-len", type=int, default=1536)
    p.add_argument("--n-train-per-regime", type=int, default=15)
    p.add_argument("--n-test-per-regime", type=int, default=20)
    p.add_argument("--lora-epochs", type=int, default=2)
    p.add_argument("--out-dir", type=str, default="pilot")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--synthetic-dry-run", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.synthetic_dry_run:
        rep = run_synthetic_dry_run(out_dir)
        print("GDN_SB1_DRYRUN_OK", json.dumps(rep))
        return

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
    use_l2k = has_logits_to_keep_arg(model)

    cf_p = Path(args.cf_path) if args.cf_path else None
    train_cases, test_cases = build_gdn_splits(
        tok, cf_p, n_linear,
        n_train_per_regime=args.n_train_per_regime,
        n_test_per_regime=args.n_test_per_regime,
        sniah_len=args.sniah_len,
        device=device,
    )
    print(f"[GDN SB1 {args.tag}] train={len(train_cases)}, test={len(test_cases)}", flush=True)

    jsonl_path = out_dir / f"w11_gdn_sb1_{args.tag}_records.jsonl"
    done = load_done_keys(jsonl_path)
    pending = [c for c in test_cases if f"{args.tag}:{c['case_id']}" not in done]

    adapters, lora_p, tot_p = attach_kveraser_lora(model, rank=16, alpha=32.0)
    tr_stats = {"skipped": True}
    if pending:
        tr_stats = train_gdn_kveraser_lora(
            model, base_model, tok, adapters, train_cases, device, use_l2k,
            epochs=args.lora_epochs, lr=2e-4,
        )
        print(f"[GDN SB1 {args.tag}] LoRA train done: {tr_stats}", flush=True)

    evaluate_gdn_sb1_suite(
        model, base_model, tok, t_cfg, adapters, test_cases, device, use_l2k,
        args.tag, jsonl_path, c_seg=128, w_seam=32,
    )
    detach_kveraser_lora(model)

    agg = aggregate_sb1_gdn(jsonl_path, args.tag)
    agg["lora_config"] = {"rank": 16, "alpha": 32.0, "lora_params": lora_p, "total_params": tot_p}
    agg["lora_train_stats"] = tr_stats
    agg["memory_comparison_C128"] = compute_gdn_hypic_memory(t_cfg, n_linear, d_k, args.sniah_len, C_seg=128)
    (out_dir / f"w11_gdn_sb1_{args.tag}_summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[GDN SB1 {args.tag}] SUMMARY READY", flush=True)


if __name__ == "__main__":
    main()
