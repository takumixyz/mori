"""Cross-family E0c inventory, tokenizer screening, and G1a/G1b pipeline validation.

Covers both Mamba-2 hybrid families:
  - Falcon-H1-1.5B (tiiuae/Falcon-H1-1.5B-Instruct): 24 parallel hybrid layers
  - Zamba2-1.2B (Zyphra/Zamba2-1.2B-Instruct-v2): 32 Mamba-2 + 6 hybrid shared-attn layers

Supports:
  --cpu-only   Run full architecture/config inspection, real tokenizer screening
               across all 3 clean templates, and pure-PyTorch CPU synthetic G1a/G1b
               + epsilon_H===0 verification without allocating any CUDA tensors.
  --run-g1     Load real downloaded weights onto --device (e.g. cuda:0) and run full
               G1a/G1b, Decision A J-full, clone invariance, manual==generate, and
               single-layer Mamba-2 epsilon_H===0 parity check sequentially per model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    FalconH1Config,
    FalconH1ForCausalLM,
    Zamba2Config,
    Zamba2ForCausalLM,
)

MODEL_REGISTRY = {
    "falcon-h1": {
        "repo_id": "tiiuae/Falcon-H1-1.5B-Instruct",
        "family": "falcon_h1",
    },
    "zamba2": {
        "repo_id": "Zyphra/Zamba2-1.2B-Instruct-v2",
        "family": "zamba2",
    },
}

NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve",
    "Frank", "Grace", "Hank", "Ivy", "Jack",
    "Karen", "Leo", "Mona", "Nate", "Olive",
    "Paul", "Quinn", "Rita", "Sam", "Tina",
]

DIGIT4_PAIRS = [
    ("4821", "7394"), ("1956", "6208"), ("8532", "3047"), ("9165", "2749"),
    ("5610", "4093"), ("3284", "6719"), ("7045", "1832"), ("6491", "2508"),
    ("2378", "8941"), ("5129", "9460"), ("3817", "6254"), ("7903", "1486"),
    ("4562", "8179"), ("6038", "2914"), ("9724", "5381"), ("1647", "7820"),
    ("8359", "4106"), ("2904", "6573"), ("5748", "3192"), ("4186", "9625"),
]

# Verified single-token (including leading space/SP-prefix) across both Falcon-H1 and Zamba2
SINGLE_TOKEN_PAIRS = [
    ("silver", "copper"), ("gold", "stone"), ("oak", "pine"), ("river", "stone"),
    ("cloud", "spark"), ("flame", "wolf"), ("bear", "rose"), ("pulse", "orbit"),
    ("solar", "vault"), ("blue", "green"), ("red", "black"), ("white", "gray"),
    ("pink", "gold"), ("alpha", "beta"), ("delta", "blue"), ("one", "two"),
    ("three", "four"), ("five", "six"), ("seven", "eight"), ("silver", "stone"),
]

PRE_FILLER_TEMPLATES = [
    "System telemetry archive note #{idx}: calibration sector nominal and humidity stable.",
    "Routine maintenance record #{idx} (unit 1024): backup generator standby voltage checked.",
    "Security audit digest #{idx}: perimeter gate sensor heartbeat verified.",
    "Facility dispatch bulletin #{idx}: cooling manifold pressure steady.",
]

POST_FILLER_TEMPLATES = [
    "Shift handoff note #{idx}: routine inspection completed without anomalies.",
    "Archive entry #{idx}: daily checkpoint counters synchronized cleanly.",
    "Operations bulletin #{idx}: ambient corridor monitors remain nominal.",
    "Telemetry summary #{idx}: standby power bus voltage remained steady.",
]


def get_eos_ids(model: Any, tok: Any) -> list[int]:
    ids: list[int] = []
    if getattr(tok, "eos_token_id", None) is not None:
        ids.append(int(tok.eos_token_id))
    for special in ("<|im_end|>", "<|end_of_text|>", "<|eot_id|>"):
        tid = tok.convert_tokens_to_ids(special)
        if isinstance(tid, int) and tid >= 0 and tid not in ids:
            ids.append(tid)
    cfg_eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(cfg_eos, int) and cfg_eos not in ids:
        ids.append(cfg_eos)
    elif isinstance(cfg_eos, (list, tuple)):
        for x in cfg_eos:
            if int(x) not in ids:
                ids.append(int(x))
    return ids


def clone_cache(cache: DynamicCache) -> DynamicCache:
    """Deep clone DynamicCache including conv_states, recurrent_states, keys, values."""
    new_cache = DynamicCache(config=None)
    new_cache.layers = []
    for layer in cache.layers:
        nl = object.__new__(type(layer))
        nl.__dict__.update(layer.__dict__)
        if hasattr(layer, "keys") and layer.keys is not None:
            nl.keys = layer.keys.clone()
        if hasattr(layer, "values") and layer.values is not None:
            nl.values = layer.values.clone()
        if hasattr(layer, "conv_states"):
            nl.conv_states = {
                k: (v.clone() if v is not None else None) for k, v in layer.conv_states.items()
            }
            nl.recurrent_states = {
                k: (v.clone() if v is not None else None) for k, v in layer.recurrent_states.items()
            }
            nl.is_conv_states_initialized = dict(layer.is_conv_states_initialized)
            nl.is_recurrent_states_initialized = dict(layer.is_recurrent_states_initialized)
            nl.has_previous_state = dict(layer.has_previous_state)
            nl.conv_kernel_size = dict(layer.conv_kernel_size)
        new_cache.layers.append(nl)
    return new_cache


def snap_all_cache_tensors(cache: DynamicCache) -> list[tuple[int, str, torch.Tensor]]:
    snaps: list[tuple[int, str, torch.Tensor]] = []
    for idx, layer in enumerate(cache.layers):
        if hasattr(layer, "keys") and layer.keys is not None:
            snaps.append((idx, "keys", layer.keys.detach().clone()))
        if hasattr(layer, "values") and layer.values is not None:
            snaps.append((idx, "values", layer.values.detach().clone()))
        if hasattr(layer, "conv_states"):
            for k, v in layer.conv_states.items():
                if v is not None:
                    snaps.append((idx, f"conv_{k}", v.detach().clone()))
        if hasattr(layer, "recurrent_states"):
            for k, v in layer.recurrent_states.items():
                if v is not None:
                    snaps.append((idx, f"ssm_{k}", v.detach().clone()))
    return snaps


def assert_cache_unchanged(cache: DynamicCache, snaps: list[tuple[int, str, torch.Tensor]]) -> None:
    cur = snap_all_cache_tensors(cache)
    assert len(cur) == len(snaps), f"Cache tensor count changed: {len(snaps)} -> {len(cur)}"
    for (i1, k1, t1), (i2, k2, t2) in zip(snaps, cur):
        assert i1 == i2 and k1 == k2, f"Cache slot mismatch: {(i1, k1)} vs {(i2, k2)}"
        assert t1.shape == t2.shape, f"Shape changed at layer {i1} {k1}: {t1.shape} -> {t2.shape}"
        assert torch.equal(t1, t2), f"Read-only cache mutated at layer {i1} {k1}!"


def make_position_ids(pos_start: int, seq_len: int, device: torch.device) -> torch.LongTensor:
    return torch.arange(pos_start, pos_start + seq_len, device=device, dtype=torch.long).unsqueeze(0)


def kl_div_from_logits(logits_ref: torch.Tensor, logits_cand: torch.Tensor) -> float:
    log_p = F.log_softmax(logits_ref.double(), dim=-1)
    log_q = F.log_softmax(logits_cand.double(), dim=-1)
    p = log_p.exp()
    val = float(torch.sum(p * (log_p - log_q), dim=-1).mean().item())
    return max(0.0, val)


def run_segment(
    model: Any,
    cache: DynamicCache,
    ids_seg: torch.LongTensor,
    pos_start: int,
    max_chunk: int = 2048,
) -> torch.Tensor:
    seq_len = int(ids_seg.shape[1])
    assert seq_len > 0
    last_logits = None
    for off in range(0, seq_len, max_chunk):
        sub = ids_seg[:, off : off + max_chunk]
        pos = make_position_ids(pos_start + off, int(sub.shape[1]), ids_seg.device)
        out = model(input_ids=sub, position_ids=pos, past_key_values=cache, use_cache=True)
        last_logits = out.logits
    return last_logits


def manual_greedy(
    model: Any,
    cache: DynamicCache,
    last_logits: torch.Tensor,
    pos_next: int,
    max_new: int,
    eos_ids: list[int],
) -> list[int]:
    gen: list[int] = []
    cur_tok = int(last_logits[:, -1, :].argmax(dim=-1).item())
    gen.append(cur_tok)
    if cur_tok in eos_ids:
        return gen
    cur_pos = pos_next
    for _ in range(max_new - 1):
        inp = torch.tensor([[cur_tok]], device=last_logits.device, dtype=torch.long)
        pos = make_position_ids(cur_pos, 1, last_logits.device)
        out = model(input_ids=inp, position_ids=pos, past_key_values=cache, use_cache=True)
        cur_tok = int(out.logits[:, -1, :].argmax(dim=-1).item())
        gen.append(cur_tok)
        cur_pos += 1
        if cur_tok in eos_ids:
            break
    return gen


def resolve_block_offset(horizon: str, n_linear: int, w_conv: int = 4, custom_offset: int = 9) -> int:
    if horizon == "single":
        return 3
    if horizon == "stage":
        return 9
    if horizon == "deep":
        return 3 * (n_linear - 1)
    if horizon == "custom":
        return int(custom_offset)
    raise ValueError(f"Unknown block horizon: {horizon}")


def patch_kv_range(c_target: DynamicCache, c_source: DynamicCache, start_pos: int, end_pos: int) -> None:
    for l_dst, l_src in zip(c_target.layers, c_source.layers):
        if getattr(l_src, "keys", None) is not None and getattr(l_dst, "keys", None) is not None:
            l_dst.keys[:, :, start_pos:end_pos, :].copy_(l_src.keys[:, :, start_pos:end_pos, :])
            l_dst.values[:, :, start_pos:end_pos, :].copy_(l_src.values[:, :, start_pos:end_pos, :])


def overwrite_ssm_states(c_target: DynamicCache, c_source: DynamicCache) -> None:
    for l_dst, l_src in zip(c_target.layers, c_source.layers):
        if hasattr(l_src, "recurrent_states"):
            for k, v in l_src.recurrent_states.items():
                if v is not None:
                    l_dst.recurrent_states[k].copy_(v)
            for k, v in l_src.conv_states.items():
                if v is not None:
                    l_dst.conv_states[k].copy_(v)


def check_oracle_clean(gen_text: str, expected_val: str, fmt: str) -> dict[str, Any]:
    """Verify oracle generation begins cleanly with the expected answer without markdown or chat filler."""
    stripped = gen_text.strip()
    lower = stripped.lower()
    has_markdown = "```" in stripped
    has_filler_prefix = bool(
        re.match(r"^(?:here|the|sure|certainly|of course|answer|based|according|output|result)\b", lower)
    )
    no_filler = (not has_markdown) and (not has_filler_prefix)
    if fmt == "digit4":
        starts_with_digit = bool(re.match(r"^\d", stripped))
        m = re.match(r"^(\d{4})(?:$|\b|[.,!?:;\"'])", stripped)
        exact = bool(m and m.group(1) == expected_val)
        first_token_clean = bool(starts_with_digit and no_filler)
    else:
        words = stripped.split()
        first_word = words[0].rstrip(".,!?:;\"'").lower() if words else ""
        exact = first_word == expected_val.lower()
        first_token_clean = bool(first_word and no_filler and exact)
    return {
        "oracle_clean": bool(first_token_clean and exact),
        "first_token_clean": bool(first_token_clean),
        "oracle_exact": bool(exact),
        "has_markdown": has_markdown,
        "has_filler_prefix": has_filler_prefix,
        "gen_preview": stripped[:60],
    }


def get_format_spec(old: str, new: str) -> tuple[str, str, int]:
    if old.isdigit() and new.isdigit() and len(old) == 4 and len(new) == 4:
        return "digit4", "4 digits", 4
    return "single_token", "single word", 1


def format_fact_sentence(tpl_id: int, name: str, code: str) -> str:
    if tpl_id == 0:
        return f"Record: {name} has vault code {code}."
    if tpl_id == 1:
        return f"Access log: {name} is assigned badge {code}."
    if tpl_id == 2:
        return f"System entry: {name} uses access token {code}."
    raise ValueError(f"Unknown template id: {tpl_id}")


def format_query_suffix(tpl_id: int, target_name: str, fmt_desc: str) -> str:
    if tpl_id == 0:
        return f"What is {target_name}'s vault code? Answer with only the {fmt_desc}."
    if tpl_id == 1:
        return f"What is {target_name}'s badge? Answer with only the {fmt_desc}."
    if tpl_id == 2:
        return f"What is {target_name}'s access token? Answer with only the {fmt_desc}."
    raise ValueError(f"Unknown template id: {tpl_id}")


def render_chat_prompt(tok: Any, user_content: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_fact_prompt(
    tok: Any,
    tpl_id: int,
    target_name: str,
    old: str,
    new: str,
    pre_fillers: int = 0,
    post_fillers: int = 8,
) -> dict[str, Any]:
    fmt_name, fmt_desc, expected_len = get_format_spec(old, new)
    pre_lines = [
        PRE_FILLER_TEMPLATES[i % len(PRE_FILLER_TEMPLATES)].format(idx=i + 1)
        for i in range(pre_fillers)
    ]
    post_lines = [
        POST_FILLER_TEMPLATES[i % len(POST_FILLER_TEMPLATES)].format(idx=i + 1)
        for i in range(post_fillers)
    ]
    fact_old = format_fact_sentence(tpl_id, target_name, old)
    fact_new = format_fact_sentence(tpl_id, target_name, new)
    q_line = format_query_suffix(tpl_id, target_name, fmt_desc)

    user_old = "\n".join(pre_lines + [fact_old] + post_lines + [q_line])
    user_new = "\n".join(pre_lines + [fact_new] + post_lines + [q_line])

    full_old = render_chat_prompt(tok, user_old)
    full_new = render_chat_prompt(tok, user_new)

    # Strict fact_context vs query_suffix split at exact newline before query question
    q_marker = "\n" + q_line
    q_pos_o = full_old.rfind(q_marker)
    q_pos_n = full_new.rfind(q_marker)
    assert q_pos_o > 0 and q_pos_n > 0, "Query marker not found in rendered prompt"
    fact_ctx_o = full_old[: q_pos_o + 1]  # includes trailing '\n'
    fact_ctx_n = full_new[: q_pos_n + 1]
    query_suffix = full_old[q_pos_o + 1 :]
    assert query_suffix == full_new[q_pos_n + 1 :], "Query suffix mismatch between old and new"

    enc_o = tok(full_old, return_offsets_mapping=True, add_special_tokens=False)
    enc_n = tok(full_new, return_offsets_mapping=True, add_special_tokens=False)
    ids_o = enc_o["input_ids"]
    ids_n = enc_n["input_ids"]
    off_o = enc_o["offset_mapping"]
    off_n = enc_n["offset_mapping"]

    # Boundary for T_ctx = token count of fact_ctx
    ctx_enc_o = tok(fact_ctx_o, add_special_tokens=False)["input_ids"]
    ctx_enc_n = tok(fact_ctx_n, add_special_tokens=False)["input_ids"]
    T_ctx = len(ctx_enc_o)
    clean_ctx_split = (
        len(ctx_enc_o) == len(ctx_enc_n)
        and ids_o[:T_ctx] == ctx_enc_o
        and ids_n[:T_ctx] == ctx_enc_n
        and ids_o[T_ctx:] == ids_n[T_ctx:]
    )

    # Locate exact span [p0, p1) where old vs new value resides
    # Anchor via exact fact sentence occurrence inside fact_ctx
    sent_start_o = fact_ctx_o.index(fact_old)
    val_char_start_o = sent_start_o + fact_old.rindex(old)
    val_char_end_o = val_char_start_o + len(old)

    sent_start_n = fact_ctx_n.index(fact_new)
    val_char_start_n = sent_start_n + fact_new.rindex(new)
    val_char_end_n = val_char_start_n + len(new)

    diff_idxs = [i for i in range(min(len(ids_o), len(ids_n))) if ids_o[i] != ids_n[i]]
    ok = True
    reason = "ok"
    p0 = -1
    p1 = -1

    if not clean_ctx_split:
        ok = False
        reason = "unclean_context_suffix_split"
    elif len(ids_o) != len(ids_n):
        ok = False
        reason = f"token_length_mismatch:{len(ids_o)}!={len(ids_n)}"
    elif not diff_idxs:
        ok = False
        reason = "no_token_diff"
    else:
        p0 = diff_idxs[0]
        p1 = diff_idxs[-1] + 1
        if diff_idxs != list(range(p0, p1)):
            ok = False
            reason = "non_contiguous_diff"
        elif (p1 - p0) != expected_len:
            ok = False
            reason = f"span_len_mismatch:{p1-p0}!={expected_len}"
        elif p1 > T_ctx:
            ok = False
            reason = "diff_crosses_into_query_suffix"
        else:
            # Verify offset alignment (allow merged preceding single space char_start - 1)
            so0, so1 = off_o[p0][0], off_o[p1 - 1][1]
            sn0, sn1 = off_n[p0][0], off_n[p1 - 1][1]
            dec_o = tok.decode(ids_o[p0:p1]).strip()
            dec_n = tok.decode(ids_n[p0:p1]).strip()
            if so0 not in (val_char_start_o, val_char_start_o - 1) or so1 != val_char_end_o:
                ok = False
                reason = f"old_offset_mismatch:({so0},{so1}) vs ({val_char_start_o},{val_char_end_o})"
            elif sn0 not in (val_char_start_n, val_char_start_n - 1) or sn1 != val_char_end_n:
                ok = False
                reason = f"new_offset_mismatch:({sn0},{sn1}) vs ({val_char_start_n},{val_char_end_n})"
            elif dec_o != old or dec_n != new:
                ok = False
                reason = f"decode_mismatch:{dec_o!r}!={old!r} or {dec_n!r}!={new!r}"

    return {
        "ok": ok,
        "reason": reason,
        "tpl_id": tpl_id,
        "format": fmt_name,
        "target_name": target_name,
        "old": old,
        "new": new,
        "expected_len": expected_len,
        "p0": p0,
        "p1": p1,
        "T_ctx": T_ctx,
        "T_total": len(ids_o),
        "ids_o": ids_o,
        "ids_n": ids_n,
        "full_old": full_old,
        "full_old_preview": full_old[:220],
        "query_suffix": query_suffix,
    }


def extract_mamba_mixers(model: Any) -> list[tuple[int, Any]]:
    mixers = []
    layers = model.model.layers
    for idx, layer in enumerate(layers):
        if hasattr(layer, "mamba"):
            mixers.append((idx, layer.mamba))
        elif hasattr(layer, "mamba_decoder") and hasattr(layer.mamba_decoder, "mamba"):
            mixers.append((idx, layer.mamba_decoder.mamba))
    return mixers


def build_inventory(repo_id: str, family_key: str) -> dict[str, Any]:
    cfg = AutoConfig.from_pretrained(repo_id)
    ltypes = list(getattr(cfg, "layer_types", None) or getattr(cfg, "layers_block_type", []))
    is_falcon = cfg.model_type == "falcon_h1"

    mamba_ssm_installed = importlib.util.find_spec("mamba_ssm") is not None
    causal_conv1d_installed = importlib.util.find_spec("causal_conv1d") is not None

    if is_falcon:
        n_mamba = len(ltypes)
        n_attn = sum(1 for t in ltypes if t in ("hybrid", "full_attention"))
        attn_kv_layers = [i for i, t in enumerate(ltypes) if t in ("hybrid", "full_attention")]
        shared_attn = False
        use_mem_rope = True
        rope_cfg = getattr(cfg, "rope_parameters", None) or {"rope_theta": getattr(cfg, "rope_theta", None)}
        mamba_params = {
            "num_ssm_heads": int(cfg.mamba_n_heads),
            "head_dim": int(cfg.mamba_d_head),
            "ssm_state_size": int(cfg.mamba_d_state),
            "n_groups": int(cfg.mamba_n_groups),
            "conv_kernel_size": int(cfg.mamba_d_conv),
            "chunk_size": int(cfg.mamba_chunk_size),
            "time_step_limit": [
                float(cfg.time_step_limit[0]),
                "inf" if cfg.time_step_limit[1] == float("inf") else float(cfg.time_step_limit[1]),
            ],
            "ssm_in_multiplier": float(cfg.ssm_in_multiplier),
            "ssm_multipliers": [float(x) for x in cfg.ssm_multipliers],
            "has_mup_vector": True,
        }
    else:
        n_mamba = len(ltypes)
        attn_kv_layers = list(getattr(cfg, "hybrid_layer_ids", [i for i, t in enumerate(ltypes) if t == "hybrid"]))
        n_attn = len(attn_kv_layers)
        shared_attn = True
        use_mem_rope = bool(getattr(cfg, "use_mem_rope", False))
        rope_cfg = getattr(cfg, "rope_parameters", None) or {"rope_theta": getattr(cfg, "rope_theta", None)}
        mamba_params = {
            "num_ssm_heads": int(cfg.n_mamba_heads),
            "head_dim": int(cfg.mamba_headdim),
            "ssm_state_size": int(cfg.mamba_d_state),
            "n_groups": int(cfg.mamba_ngroups),
            "conv_kernel_size": int(cfg.mamba_d_conv),
            "chunk_size": int(cfg.chunk_size),
            "time_step_limit": [float(getattr(cfg, "time_step_min", 0.001)), "inf"],
            "use_shared_attention_adapter": bool(getattr(cfg, "use_shared_attention_adapter", False)),
            "has_mup_vector": False,
        }

    # Inspect DynamicCache layer types initialized from real config (zero GPU allocation)
    dummy_cache = DynamicCache(config=cfg)
    cache_type_counts: dict[str, int] = {}
    for l in dummy_cache.layers:
        name = type(l).__name__
        cache_type_counts[name] = cache_type_counts.get(name, 0) + 1

    w_conv = mamba_params["conv_kernel_size"]
    return {
        "family": family_key,
        "repo_id": repo_id,
        "model_type": cfg.model_type,
        "architectures": list(getattr(cfg, "architectures", [])),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "hidden_size": int(cfg.hidden_size),
        "vocab_size": int(cfg.vocab_size),
        "max_position_embeddings": int(cfg.max_position_embeddings),
        "layers_block_type": ltypes,
        "n_linear_layers": n_mamba,
        "n_attn_kv_layers": n_attn,
        "attention_kv_layer_indices": attn_kv_layers,
        "shared_attention_weights": shared_attn,
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(cfg.num_key_value_heads),
        "attn_head_dim": int(getattr(cfg, "head_dim", getattr(cfg, "attention_head_dim", 128))),
        "use_mem_rope": use_mem_rope,
        "rope_config": rope_cfg,
        "mamba2": mamba_params,
        "deep_conv_reach": int((n_mamba - 1) * (w_conv - 1)),
        "cache_class": type(dummy_cache).__name__,
        "cache_layer_type_counts": cache_type_counts,
        "mamba_ssm_installed": mamba_ssm_installed,
        "causal_conv1d_installed": causal_conv1d_installed,
        "using_pytorch_fallback": (not mamba_ssm_installed) or (not causal_conv1d_installed),
        "epsilon_H_theoretical": 0.0,
    }


def run_tokenizer_screening(repo_id: str, pre_fillers: int = 30, post_fillers: int = 16) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    summary: dict[str, Any] = {
        "repo_id": repo_id,
        "tokenizer_class": type(tok).__name__,
        "is_fast": bool(tok.is_fast),
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
        "pre_fillers": pre_fillers,
        "post_fillers": post_fillers,
        "formats": {},
    }

    for fmt_name, pairs in [("digit4", DIGIT4_PAIRS), ("single_token", SINGLE_TOKEN_PAIRS)]:
        fmt_stats = {"n_tested": 0, "n_accepted": 0, "rejections": [], "templates": {}}
        for tpl_id in (0, 1, 2):
            tpl_ok = 0
            tpl_tot = 0
            sample_lengths = []
            p0_list = []
            for idx, (old, new) in enumerate(pairs):
                name = NAMES[idx % len(NAMES)]
                res = build_fact_prompt(
                    tok, tpl_id, name, old, new,
                    pre_fillers=pre_fillers, post_fillers=post_fillers,
                )
                tpl_tot += 1
                fmt_stats["n_tested"] += 1
                if res["ok"]:
                    tpl_ok += 1
                    fmt_stats["n_accepted"] += 1
                    sample_lengths.append((res["p0"], res["p1"], res["T_ctx"], res["T_total"]))
                    p0_list.append(res["p0"])
                else:
                    fmt_stats["rejections"].append({
                        "tpl_id": tpl_id,
                        "old": old,
                        "new": new,
                        "reason": res["reason"],
                    })
            fmt_stats["templates"][f"T{tpl_id}"] = {
                "tested": tpl_tot,
                "accepted": tpl_ok,
                "rejection_rate": (tpl_tot - tpl_ok) / max(1, tpl_tot),
                "sample_dims_first": sample_lengths[0] if sample_lengths else None,
            }
        fmt_stats["rejection_rate"] = (
            (fmt_stats["n_tested"] - fmt_stats["n_accepted"]) / max(1, fmt_stats["n_tested"])
        )
        summary["formats"][fmt_name] = fmt_stats
    return summary


def run_synthetic_cpu_selftest() -> dict[str, Any]:
    """Run CPU-only tiny synthetic Falcon-H1 and Zamba2 pipeline & epsilon_H===0 verification."""
    torch.manual_seed(0)
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

    report = {}
    for key, mod_cls, cfg in [
        ("synthetic_falcon_h1", FalconH1ForCausalLM, f_cfg),
        ("synthetic_zamba2", Zamba2ForCausalLM, z_cfg),
    ]:
        mod = mod_cls(cfg).eval()
        mixers_syn = extract_mamba_mixers(mod)
        b_offset_syn = resolve_block_offset("deep", len(mixers_syn), w_conv=int(getattr(cfg, "mamba_d_conv", 4)))
        ids_o = torch.randint(0, 128, (1, 40))
        ids_e = ids_o.clone()
        p0, p1, T_ctx = 10, 14, 32
        b = min(T_ctx, p1 + b_offset_syn)
        ids_e[:, p0:p1] = (ids_o[:, p0:p1] + 7) % 128
        pos_full = make_position_ids(0, 40, torch.device("cpu"))

        with torch.no_grad():
            # 1. Oneshot oracle on ids_e
            c_oneshot = DynamicCache(config=cfg)
            lg_oneshot = mod(input_ids=ids_e, position_ids=pos_full, past_key_values=c_oneshot, use_cache=True).logits[:, -1, :]

            # 2. Split oracle [0, p0) -> [p0, b) -> [b, T_ctx) -> [T_ctx, T)
            c_pre = DynamicCache(config=cfg)
            _ = run_segment(mod, c_pre, ids_e[:, :p0], 0)
            snaps_pre = snap_all_cache_tensors(c_pre)

            c_split = clone_cache(c_pre)
            _ = run_segment(mod, c_split, ids_e[:, p0:b], p0)
            if T_ctx > b:
                _ = run_segment(mod, c_split, ids_e[:, b:T_ctx], b)
            lg_split_full = run_segment(mod, c_split, ids_e[:, T_ctx:], T_ctx)
            lg_split = lg_split_full[:, -1, :]
            assert_cache_unchanged(c_pre, snaps_pre)

            # 3. Decision A J-full: O_block splice at b onto c_old_b + recompute [b, T_ctx)
            assert torch.equal(ids_o[:, :p0], ids_e[:, :p0])
            if T_ctx > b:
                assert torch.equal(ids_o[:, b:T_ctx], ids_e[:, b:T_ctx])
            c_old_b = clone_cache(c_pre)
            _ = run_segment(mod, c_old_b, ids_o[:, p0:b], p0)
            c_new_blk = clone_cache(c_pre)
            _ = run_segment(mod, c_new_blk, ids_e[:, p0:b], p0)

            c_jfull = clone_cache(c_old_b)
            patch_kv_range(c_jfull, c_new_blk, p0, b)
            overwrite_ssm_states(c_jfull, c_new_blk)
            if T_ctx > b:
                _ = run_segment(mod, c_jfull, ids_e[:, b:T_ctx], b)

            lg_jfull_suffix = run_segment(mod, clone_cache(c_jfull), ids_e[:, T_ctx:], T_ctx)
            lg_jfull = lg_jfull_suffix[:, -1, :]

            # 4. manual_greedy vs model.generate parity
            c_gen = clone_cache(c_jfull)
            first_suffix = run_segment(mod, c_gen, ids_e[:, T_ctx:], T_ctx)
            gen_manual = manual_greedy(mod, c_gen, first_suffix, 40, max_new=6, eos_ids=[127])
            gen_hf = mod.generate(
                input_ids=ids_e, max_new_tokens=6, do_sample=False,
                pad_token_id=0, eos_token_id=127,
            )[0, 40:].tolist()

            # 5. Isolated single-layer Mamba-2 G_t hook parity & epsilon_H===0 in fp64
            _, mix = extract_mamba_mixers(mod)[0]
            h = torch.randn(1, 40, cfg.hidden_size)
            h2 = h.clone()
            h2[:, p0:p1, :] = torch.randn(1, p1 - p0, cfg.hidden_size)
            g_chunks = []

            def hook_fn(_m, _inp, out):
                proj = out * mix.mup_vector if hasattr(mix, "mup_vector") else out
                _, _, dt = proj.split([mix.intermediate_size, mix.conv_dim, mix.num_heads], dim=-1)
                dt_eff = torch.clamp(
                    F.softplus(dt.float() + mix.dt_bias.float()),
                    min=mix.time_step_limit[0],
                    max=mix.time_step_limit[1],
                )
                A = -torch.exp(mix.A_log.float())
                g_chunks.append((dt_eff * A).sum(dim=1, dtype=torch.float64))

            hk = mix.in_proj.register_forward_hook(hook_fn)
            mc_o = DynamicCache(config=cfg)
            _ = mix(h[:, :b], cache_params=mc_o)
            S_b_old = mc_o.get_recurrent_state(mix.layer_idx).clone()
            g_chunks.clear()
            _ = mix(h[:, b:T_ctx], cache_params=mc_o)
            S_T_old = mc_o.get_recurrent_state(mix.layer_idx).clone()
            G_diff = g_chunks[-1]

            mc_n = DynamicCache(config=cfg)
            _ = mix(h2[:, :b], cache_params=mc_n)
            S_b_new = mc_n.get_recurrent_state(mix.layer_idx).clone()
            _ = mix(h2[:, b:T_ctx], cache_params=mc_n)
            S_T_exact = mc_n.get_recurrent_state(mix.layer_idx).clone()
            hk.remove()

            gamma = torch.exp(G_diff)[:, :, None, None]
            S_T_transport = (S_T_old.double() + gamma * (S_b_new.double() - S_b_old.double())).to(S_T_old.dtype)
            eps_H_err = float((S_T_transport - S_T_exact).abs().max().item())

        split_diff = float((lg_split - lg_oneshot).abs().max().item())
        jfull_diff = float((lg_jfull - lg_oneshot).abs().max().item())
        jfull_vs_split = float((lg_jfull - lg_split).abs().max().item())
        split_kl = kl_div_from_logits(lg_oneshot, lg_split)
        jfull_kl = kl_div_from_logits(lg_oneshot, lg_jfull)

        g1a_ok = jfull_vs_split <= 1e-5
        g1b_ok = split_kl <= 1e-4 and jfull_kl <= 1e-4
        report[key] = {
            "b": b,
            "b_offset": b_offset_syn,
            "split_vs_oneshot_max_diff": split_diff,
            "split_vs_oneshot_kl": split_kl,
            "jfull_vs_oneshot_max_diff": jfull_diff,
            "jfull_vs_oneshot_kl": jfull_kl,
            "jfull_vs_split_max_diff": jfull_vs_split,
            "g1a_pass_1e_5": g1a_ok,
            "g1b_pass_1e_4": g1b_ok,
            "all_g1_pass": g1a_ok and g1b_ok,
            "manual_eq_generate": gen_manual == gen_hf,
            "isolated_mamba2_epsilon_H": eps_H_err,
            "epsilon_H_zero_pass": eps_H_err <= 1e-10,
        }
    return report


def apply_g1_for_model(
    model_key: str,
    repo_id: str,
    device: str,
    dtype: torch.dtype,
    n_prompts: int = 3,
    allow_rounding_exception: bool = False,
    g1b_thresh: float = 1e-4,
    block_horizon: str = "deep",
    pre_fillers: int = 30,
    post_fillers: int = 16,
) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=dtype).to(device).eval()
    eos_ids = get_eos_ids(model, tok)
    cfg = model.config
    w_conv = int(getattr(cfg, "mamba_d_conv", 4))
    mixers = extract_mamba_mixers(model)
    n_linear = len(mixers)
    b_offset = resolve_block_offset(block_horizon, n_linear, w_conv)

    checks = []
    dev = torch.device(device)

    for tpl_id in range(min(3, n_prompts)):
        old, new = DIGIT4_PAIRS[tpl_id]
        name = NAMES[tpl_id]
        case = build_fact_prompt(
            tok, tpl_id, name, old, new,
            pre_fillers=pre_fillers, post_fillers=post_fillers,
        )
        if not case["ok"]:
            raise RuntimeError(f"G1 screening failed on {model_key} tpl={tpl_id}: {case['reason']}")

        ids_o = torch.tensor([case["ids_o"]], device=dev, dtype=torch.long)
        ids_e = torch.tensor([case["ids_n"]], device=dev, dtype=torch.long)
        p0, p1, T_ctx, T_tot = case["p0"], case["p1"], case["T_ctx"], case["T_total"]
        b = min(T_ctx, p1 + b_offset)

        with torch.no_grad():
            assert torch.equal(ids_o[:, :p0], ids_e[:, :p0]), "Prefix mismatch before p0"
            if T_ctx > b:
                assert torch.equal(ids_o[:, b:T_ctx], ids_e[:, b:T_ctx]), "Context mismatch after b"

            c_one = DynamicCache(config=cfg)
            lg_one_full = run_segment(model, c_one, ids_e, 0)
            lg_one = lg_one_full[:, -1, :]

            c_one_gen = clone_cache(c_one)
            ora_toks = manual_greedy(model, c_one_gen, lg_one_full, T_tot, max_new=8, eos_ids=eos_ids)
            ora_text = tok.decode(ora_toks, skip_special_tokens=True)
            ora_chk = check_oracle_clean(ora_text, new, case["format"])

            c_pre = DynamicCache(config=cfg)
            _ = run_segment(model, c_pre, ids_e[:, :p0], 0)
            snaps_pre = snap_all_cache_tensors(c_pre)

            c_split = clone_cache(c_pre)
            _ = run_segment(model, c_split, ids_e[:, p0:b], p0)
            if T_ctx > b:
                _ = run_segment(model, c_split, ids_e[:, b:T_ctx], b)
            lg_split = run_segment(model, c_split, ids_e[:, T_ctx:], T_ctx)[:, -1, :]
            assert_cache_unchanged(c_pre, snaps_pre)

            # Decision A J-full: O_block splice at b onto c_old_b + recompute [b, T_ctx)
            c_old_b = clone_cache(c_pre)
            _ = run_segment(model, c_old_b, ids_o[:, p0:b], p0)
            c_new_blk = clone_cache(c_pre)
            _ = run_segment(model, c_new_blk, ids_e[:, p0:b], p0)

            c_jfull = clone_cache(c_old_b)
            patch_kv_range(c_jfull, c_new_blk, p0, b)
            overwrite_ssm_states(c_jfull, c_new_blk)
            if T_ctx > b:
                _ = run_segment(model, c_jfull, ids_e[:, b:T_ctx], b)

            c_jf_run = clone_cache(c_jfull)
            lg_jf_full = run_segment(model, c_jf_run, ids_e[:, T_ctx:], T_ctx)
            lg_jfull = lg_jf_full[:, -1, :]

            # manual_greedy == model.generate check on tpl 0
            if tpl_id == 0:
                man_toks = manual_greedy(model, c_jf_run, lg_jf_full, T_tot, max_new=6, eos_ids=eos_ids)
                hf_toks = model.generate(
                    input_ids=ids_e,
                    max_new_tokens=6,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id or eos_ids[0],
                    eos_token_id=eos_ids,
                )[0, T_tot:].tolist()
                man_eq_gen = (man_toks == hf_toks)
            else:
                man_toks, hf_toks, man_eq_gen = None, None, None

        diff_split = float((lg_split - lg_one).abs().max().item())
        diff_jfull = float((lg_jfull - lg_one).abs().max().item())
        diff_jf_sp = float((lg_jfull - lg_split).abs().max().item())
        kl_split = kl_div_from_logits(lg_one, lg_split)
        kl_jfull = kl_div_from_logits(lg_one, lg_jfull)
        top1_split = int(lg_split.argmax(dim=-1).item()) == int(lg_one.argmax(dim=-1).item())
        top1_jfull = int(lg_jfull.argmax(dim=-1).item()) == int(lg_one.argmax(dim=-1).item())

        # G1a: J-full (Decision A O_block at b + recompute [b, T_ctx)) vs split_oracle with identical segmentation
        g1a = diff_jf_sp <= 1e-5
        g1b_strict = (kl_split <= g1b_thresh) and (kl_jfull <= g1b_thresh)
        g1b_exc = (
            allow_rounding_exception
            and max(diff_split, diff_jfull) <= 0.28125
            and max(kl_split, kl_jfull) <= 1.5e-3
        )
        g1b = g1b_strict or g1b_exc

        checks.append({
            "tpl_id": tpl_id,
            "T_ctx": T_ctx,
            "T_total": T_tot,
            "p0": p0,
            "p1": p1,
            "b": b,
            "b_offset": b_offset,
            "recompute_suffix_len": max(0, T_ctx - b),
            "split_vs_oneshot_max_diff": diff_split,
            "split_vs_oneshot_kl": kl_split,
            "jfull_vs_oneshot_max_diff": diff_jfull,
            "jfull_vs_oneshot_kl": kl_jfull,
            "jfull_vs_split_max_diff": diff_jf_sp,
            "top1_split_match": top1_split,
            "top1_jfull_match": top1_jfull,
            "g1a_exact": g1a,
            "g1b_strict": g1b_strict,
            "g1b_pass": g1b,
            "g1b_via_rounding_exception": (not g1b_strict) and g1b_exc,
            "manual_eq_generate": man_eq_gen,
            "manual_tokens": man_toks,
            "generate_tokens": hf_toks,
            "oracle_clean": ora_chk["oracle_clean"],
            "first_token_clean": ora_chk["first_token_clean"],
            "oracle_gen_preview": ora_chk["gen_preview"],
        })

    # Isolated single-layer Mamba-2 transport parity on actual real weights
    _, mix = mixers[0]
    with torch.no_grad():
        h = torch.randn(1, 64, cfg.hidden_size, device=dev, dtype=dtype) * 0.1
        h2 = h.clone()
        h2[:, 12:16, :] = torch.randn(1, 4, cfg.hidden_size, device=dev, dtype=dtype) * 0.1
        g_chunks = []

        def hook_fn(_m, _inp, out):
            proj = out * mix.mup_vector if hasattr(mix, "mup_vector") else out
            _, _, dt = proj.split([mix.intermediate_size, mix.conv_dim, mix.num_heads], dim=-1)
            dt_eff = torch.clamp(
                F.softplus(dt.float() + mix.dt_bias.float()),
                min=mix.time_step_limit[0],
                max=mix.time_step_limit[1],
            )
            A = -torch.exp(mix.A_log.float())
            g_chunks.append((dt_eff * A).sum(dim=1, dtype=torch.float64))

        hk = mix.in_proj.register_forward_hook(hook_fn)
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
        hk.remove()

        gamma = torch.exp(G_diff)[:, :, None, None]
        S_T_tr = (S_T_o.double() + gamma * (S_b_n.double() - S_b_o.double())).to(S_T_o.dtype)
        eps_H_real = float((S_T_tr - S_T_ex).abs().max().item())

    # Capture real runtime cache layer shapes
    sample_cache_shapes = {}
    for li, layer in enumerate(c_one.layers):
        ltype = cfg.layers_block_type[li]
        if ltype not in sample_cache_shapes:
            entry = {
                "example_layer_idx": li,
                "layer_class": type(layer).__name__,
                "conv_state_shape": list(layer.conv_states[0].shape) if hasattr(layer, "conv_states") else None,
                "conv_state_dtype": str(layer.conv_states[0].dtype) if hasattr(layer, "conv_states") else None,
                "recurrent_state_shape": list(layer.recurrent_states[0].shape) if hasattr(layer, "recurrent_states") else None,
                "recurrent_state_dtype": str(layer.recurrent_states[0].dtype) if hasattr(layer, "recurrent_states") else None,
                "keys_shape": list(layer.keys.shape) if getattr(layer, "keys", None) is not None else None,
                "values_shape": list(layer.values.shape) if getattr(layer, "values", None) is not None else None,
            }
            sample_cache_shapes[ltype] = entry

    del model, c_one, c_one_gen, c_pre, c_split, c_old_b, c_new_blk, c_jfull, c_jf_run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "device": device,
        "dtype": str(dtype),
        "w_conv": w_conv,
        "n_linear": n_linear,
        "block_horizon": block_horizon,
        "b_offset": b_offset,
        "g1b_thresh": g1b_thresh,
        "allow_rounding_exception": allow_rounding_exception,
        "sample_cache_shapes": sample_cache_shapes,
        "g1_checks": checks,
        "all_g1a_pass": all(c["g1a_exact"] for c in checks),
        "all_g1b_pass": all(c["g1b_pass"] for c in checks),
        "all_top1_match": all(c["top1_jfull_match"] for c in checks),
        "all_g1_pass": all(c["g1a_exact"] and c["g1b_pass"] and c["top1_jfull_match"] for c in checks),
        "oracle_clean_rate": sum(1 for c in checks if c["oracle_clean"]) / max(1, len(checks)),
        "oracle_first_token_clean_rate": sum(1 for c in checks if c["first_token_clean"]) / max(1, len(checks)),
        "isolated_layer0_epsilon_H": eps_H_real,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-family E0c inventory + screening + G1a/G1b")
    parser.add_argument("--model", choices=["falcon-h1", "zamba2", "all"], default="all")
    parser.add_argument("--falcon-id", default=MODEL_REGISTRY["falcon-h1"]["repo_id"])
    parser.add_argument("--zamba2-id", default=MODEL_REGISTRY["zamba2"]["repo_id"])
    parser.add_argument("--cpu-only", action="store_true", help="Run inventory + tokenizer screening + CPU synthetic self-test only")
    parser.add_argument("--run-g1", action="store_true", help="Run real-weight G1a/G1b verification on --device")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--block-horizon", choices=["deep", "stage", "single"], default="deep")
    parser.add_argument("--g1b-thresh", type=float, default=1e-4)
    parser.add_argument("--n-prompts", type=int, default=3)
    parser.add_argument("--pre-fillers", type=int, default=30)
    parser.add_argument("--post-fillers", type=int, default=16)
    parser.add_argument("--allow-rounding-exception", action="store_true", default=False)
    parser.add_argument("--output", default="pilot/w2_crossfamily_e0c_results.json")
    args = parser.parse_args()

    if args.cpu_only and args.run_g1:
        raise ValueError("Cannot pass both --cpu-only and --run-g1")

    targets = (
        [("falcon-h1", args.falcon_id), ("zamba2", args.zamba2_id)]
        if args.model == "all"
        else [("falcon-h1", args.falcon_id) if args.model == "falcon-h1" else ("zamba2", args.zamba2_id)]
    )

    out: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "gpu_g1" if args.run_g1 else "cpu_only",
        "models": {},
    }

    out["synthetic_cpu_selftest"] = run_synthetic_cpu_selftest()
    print("[CPU synthetic self-test]", json.dumps(out["synthetic_cpu_selftest"], indent=2))

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    for mkey, repo_id in targets:
        print(f"=== Inspecting {mkey} ({repo_id}) ===")
        inv = build_inventory(repo_id, mkey)
        tok_scr = run_tokenizer_screening(repo_id, pre_fillers=args.pre_fillers, post_fillers=args.post_fillers)
        entry: dict[str, Any] = {
            "inventory": inv,
            "tokenizer_screening": tok_scr,
        }
        if args.run_g1:
            print(f"Running real-model G1a/G1b check on {args.device} ({args.dtype}, horizon={args.block_horizon})...")
            entry["gpu_g1"] = apply_g1_for_model(
                mkey, repo_id, device=args.device, dtype=dtype,
                n_prompts=args.n_prompts,
                allow_rounding_exception=args.allow_rounding_exception,
                g1b_thresh=args.g1b_thresh,
                block_horizon=args.block_horizon,
                pre_fillers=args.pre_fillers,
                post_fillers=args.post_fillers,
            )
        out["models"][mkey] = entry

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved E0c report to {out_path}")


if __name__ == "__main__":
    main()
