"""Stage-B Wave 11-1: Mamba-2 Cross-Family Scale-Up (SB5) + SB1 Baselines (KVEraser & HYPIC).

Reuses w2_crossfamily_e0c / w2_crossfamily_e2_e3_e6 helpers via import (zero duplicate E2/E3 runners).
Covers:
  - SB5 (Falcon-H1-1.5B & Zamba2-1.2B):
      * Scaled E2 to n=100 unique balanced samples (50 digit4 + 50 shared 1-token words)
      * Scaled E3 to M4_k2 (n=10 sets = 20 perm runs) & M8_k4 (n=5 sets = 30 perm runs)
  - SB1 (Mamba-2 hybrids):
      * Faithful KVEraser (arXiv:2606.17034): rank-16 LoRA (r=16, alpha=32) on frozen attention
        projections (q/k/v/o) trained on disjoint n=40 split via Eq.(6) teacher-forced CE+KL loss,
        leaving suffix KV and Mamba-2 recurrent states untouched; plus Delete-and-Shift (RoPE-shifted KV).
      * Faithful HYPIC (arXiv:2607.01299): segment-aligned chunk operator Phi_m + W_seam=32
        recompute vs unaligned mid-segment splice without suffix transport vs zero-init chunk.
      * Direct comparison against O_local (J-local), oracle, stale/kv_only on n=60 disjoint test set.

Supports --synthetic-dry-run on CPU (`CUDA_VISIBLE_DEVICES=""`) to verify autograd, cache shapes,
KVEraser LoRA training, HYPIC accounting, and scaled E2/E3 protocols before running GPU jobs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", r"D:\applicationData\HuggingFace")

import torch
import torch.nn as nn
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
    MODEL_REGISTRY,
    build_fact_prompt,
    clone_cache,
    extract_mamba_mixers,
    get_eos_ids,
    kl_div_from_logits,
    make_position_ids,
    patch_kv_range,
    resolve_block_offset,
    run_segment,
)
from w2_crossfamily_e2_e3_e6 import (  # noqa: E402
    append_jsonl,
    apply_mamba2_jlocal_update,
    build_multifact_prompt,
    compute_state_delta_fp64,
    eval_suffix_condition,
    load_completed_keys,
    mean_ans_kl,
    prepare_bckpt_cache,
    resolve_answer_tokens,
    run_e2_suite,
    run_e3_suite,
    run_segment_with_mamba_decay,
    sync_time,
    wilson_ci,
)


CANDIDATE_WORDS = [
    "ash", "bay", "birch", "black", "blue", "bone", "box", "brass", "brick", "brook",
    "cane", "cape", "cedar", "chalk", "clay", "cliff", "cloud", "coal", "coast", "cobalt",
    "copper", "coral", "cove", "creek", "crest", "crown", "dawn", "delta", "dew", "dome",
    "dove", "drift", "dune", "dust", "elm", "ember", "fern", "field", "fig", "finch",
    "fir", "five", "flame", "flint", "foam", "fog", "ford", "forge", "four", "frost",
    "gale", "gate", "gem", "glen", "gold", "grain", "grape", "gray", "green", "grove",
    "gull", "hare", "hawk", "hazel", "heath", "hill", "hive", "horn", "ice", "iron",
    "jade", "jay", "jet", "kelp", "key", "knot", "lake", "lark", "lava", "leaf",
    "lime", "linden", "loam", "lock", "lodge", "lynx", "maple", "marsh", "mast", "mead",
    "mesa", "mill", "mint", "mist", "moon", "moor", "moss", "moth", "oak", "oat",
    "olive", "opal", "orb", "ore", "owl", "palm", "path", "peak", "pearl", "peat",
    "pine", "pink", "plum", "pond", "port", "pulse", "quartz", "rain", "ram", "red",
    "reed", "reef", "ridge", "ring", "river", "rock", "root", "rose", "ruby", "rush",
    "rust", "rye", "sage", "salt", "sand", "seed", "seven", "shade", "shale", "shell",
    "shore", "silk", "silver", "six", "sky", "slate", "smoke", "snow", "solar", "spark",
    "spire", "spruce", "star", "steam", "steel", "stem", "stone", "storm", "straw", "stream",
    "sun", "swan", "tar", "teal", "thorn", "three", "tide", "timber", "tin", "torch",
    "trail", "tree", "two", "vale", "vault", "veil", "vine", "wave", "well", "wheat",
    "white", "willow", "wind", "wing", "wolf", "wood", "wren", "yew", "zinc", "zone",
    "amber", "alpha", "beta", "gamma", "orbit", "bear", "zero", "one", "eight", "nine",
]

SCALED_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
    "Ivy", "Jack", "Karen", "Leo", "Mona", "Nate", "Olive", "Paul",
    "Quinn", "Rita", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zack", "Aaron", "Bella", "Caleb", "Diana", "Ethan", "Fiona",
    "Gavin", "Hannah", "Isaac", "Julia", "Kevin", "Laura", "Mason", "Nora",
    "Oscar", "Paula", "Quentin", "Rachel", "Simon", "Tessa", "Uriel", "Valerie",
    "Walter", "Zoe",
]


def generate_disjoint_digit4_pairs(count: int, seed: int = 2026) -> list[tuple[str, str]]:
    """Deterministically generate `count` distinct valid 4-digit (old, new) pairs."""
    pairs: list[tuple[str, str]] = []
    seen_vals: set[str] = set()
    state = seed
    while len(pairs) < count:
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        a = 1000 + (state % 9000)
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        b = 1000 + (state % 9000)
        sa, sb = str(a), str(b)
        if sa != sb and sa not in seen_vals and sb not in seen_vals:
            seen_vals.add(sa)
            seen_vals.add(sb)
            pairs.append((sa, sb))
    return pairs


def collect_shared_single_tokens(toks: list[Any]) -> list[str]:
    """Return words that are single-token (bare + leading space) across every tokenizer."""
    shared = []
    for w in CANDIDATE_WORDS:
        ok = True
        for tok in toks:
            if len(tok.encode(w, add_special_tokens=False)) != 1:
                ok = False
                break
            if len(tok.encode(" " + w, add_special_tokens=False)) != 1:
                ok = False
                break
        if ok:
            shared.append(w)
    return shared


def build_single_token_pairs(words: list[str], count: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    n = len(words)
    assert n >= 10, f"Need >= 10 shared words, got {n}"
    step = 1
    while len(pairs) < count:
        for i in range(n):
            j = (i + step) % n
            if i != j:
                pairs.append((words[i], words[j]))
                if len(pairs) >= count:
                    break
        step += 1
        if step >= n:
            step = 1
    return pairs[:count]


def build_case_pools(
    tok: Any,
    shared_words: list[str],
    n_e2: int = 100,
    n_sb1_train: int = 40,
    n_sb1_test: int = 60,
    pre_fillers: int = 30,
    post_fillers: int = 16,
) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic clean pools for E2 scale-up and disjoint SB1 train/test."""
    # Generate enough candidate pairs so every split has strictly distinct (old, new) pairs
    d_pairs = generate_disjoint_digit4_pairs(n_e2 + n_sb1_train + n_sb1_test + 40, seed=7121)
    s_pairs = build_single_token_pairs(shared_words, n_e2 + n_sb1_train + n_sb1_test + 40)

    def _collect(pairs_list: list[tuple[str, str]], start_idx: int, need: int, name_off: int) -> tuple[list[dict[str, Any]], int]:
        out: list[dict[str, Any]] = []
        idx = start_idx
        while len(out) < need and idx < len(pairs_list):
            old_v, new_v = pairs_list[idx]
            tpl_id = idx % 3
            name = SCALED_NAMES[(idx + name_off) % len(SCALED_NAMES)]
            case = build_fact_prompt(
                tok, tpl_id, name, old_v, new_v,
                pre_fillers=pre_fillers, post_fillers=post_fillers,
            )
            idx += 1
            if case["ok"]:
                out.append(case)
        return out, idx

    d_sb1_tr, d_ptr = _collect(d_pairs, 0, n_sb1_train // 2, 0)
    s_sb1_tr, s_ptr = _collect(s_pairs, 0, n_sb1_train - len(d_sb1_tr), 3)
    sb1_train = []
    for i in range(max(len(d_sb1_tr), len(s_sb1_tr))):
        if i < len(d_sb1_tr):
            sb1_train.append(d_sb1_tr[i])
        if i < len(s_sb1_tr):
            sb1_train.append(s_sb1_tr[i])

    d_sb1_te, d_ptr = _collect(d_pairs, d_ptr, n_sb1_test // 2, 11)
    s_sb1_te, s_ptr = _collect(s_pairs, s_ptr, n_sb1_test - len(d_sb1_te), 17)
    sb1_test = []
    for i in range(max(len(d_sb1_te), len(s_sb1_te))):
        if i < len(d_sb1_te):
            sb1_test.append(d_sb1_te[i])
        if i < len(s_sb1_te):
            sb1_test.append(s_sb1_te[i])

    # E2 n=100: balanced 50 digit4 + 50 single_token
    d_e2, _ = _collect(d_pairs, d_ptr, n_e2 // 2, 5)
    s_e2, _ = _collect(s_pairs, s_ptr, n_e2 - len(d_e2), 23)
    e2_cases = []
    for i in range(max(len(d_e2), len(s_e2))):
        if i < len(d_e2):
            e2_cases.append(d_e2[i])
        if i < len(s_e2):
            e2_cases.append(s_e2[i])

    return {
        "e2_cases": e2_cases[:n_e2],
        "sb1_train": sb1_train[:n_sb1_train],
        "sb1_test": sb1_test[:n_sb1_test],
    }


def build_scaled_e3_configs(tok: Any, k2_n: int = 10, k4_n: int = 5) -> list[dict[str, Any]]:
    """Build E3 M4_k2 and M8_k4 configs with unique 4-digit pools (first-digit distinct)."""
    all_d_raw = generate_disjoint_digit4_pairs(k2_n * 24 + k4_n * 48 + 128, seed=9931)
    all_d = [(a, b) for (a, b) in all_d_raw if a[0] != b[0]]
    d_ptr = 0
    configs: list[dict[str, Any]] = []

    m4_cnt = 0
    cand_idx = 0
    while m4_cnt < k2_n and d_ptr + 4 <= len(all_d):
        pairs = []
        for j in range(4):
            old_v, new_v = all_d[d_ptr]
            d_ptr += 1
            pairs.append((SCALED_NAMES[(cand_idx * 4 + j) % len(SCALED_NAMES)], old_v, new_v))
        mf = build_multifact_prompt(tok, cand_idx % 3, 4, pairs, filler_between=12, post_fillers=16)
        cand_idx += 1
        if mf["ok"]:
            configs.append({"set_id": f"M4_k2_{m4_cnt}", "edit_indices": [1, 2], "perms": [[1, 2], [2, 1]], "mf": mf})
            m4_cnt += 1

    perms_k4 = [
        [1, 3, 5, 6], [6, 5, 3, 1], [1, 5, 3, 6],
        [6, 3, 5, 1], [3, 6, 1, 5], [5, 1, 6, 3],
    ]
    m8_cnt = 0
    cand_idx = 0
    while m8_cnt < k4_n and d_ptr + 8 <= len(all_d):
        pairs = []
        for j in range(8):
            old_v, new_v = all_d[d_ptr]
            d_ptr += 1
            pairs.append((SCALED_NAMES[(cand_idx * 8 + j) % len(SCALED_NAMES)], old_v, new_v))
        mf = build_multifact_prompt(tok, cand_idx % 3, 8, pairs, filler_between=12, post_fillers=16)
        cand_idx += 1
        if mf["ok"]:
            configs.append({"set_id": f"M8_k4_{m8_cnt}", "edit_indices": [1, 3, 5, 6], "perms": perms_k4, "mf": mf})
            m8_cnt += 1

    return configs


# ---------------------------------------------------------------------------
# Faithful KVEraser (arXiv:2606.17034) LoRA KV Steerer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Rank-r LoRA adapter wrapping a frozen nn.Linear layer (toggleable)."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        dev = base.weight.device
        dtype = base.weight.dtype
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.enabled = False
        self.lora_A = nn.Parameter(torch.empty((rank, base.in_features), device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, rank), device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out + delta


def attach_kveraser_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
) -> tuple[list[LoRALinear], int, int]:
    """Freeze all base weights and wrap attention projections with rank-r LoRA."""
    for p in model.parameters():
        p.requires_grad_(False)

    adapters: list[LoRALinear] = []
    seen_mods: dict[int, LoRALinear] = {}
    targets = ("q_proj", "k_proj", "v_proj", "o_proj")

    def _recurse(mod: nn.Module, in_attn: bool) -> None:
        for child_name, child in list(mod.named_children()):
            now_attn = in_attn or ("self_attn" in child_name) or ("attn" in child_name and "layernorm" not in child_name)
            if now_attn and child_name in targets and isinstance(child, nn.Linear):
                mid = id(child)
                if mid not in seen_mods:
                    wrap = LoRALinear(child, rank=rank, alpha=alpha)
                    seen_mods[mid] = wrap
                    adapters.append(wrap)
                setattr(mod, child_name, seen_mods[mid])
            else:
                _recurse(child, now_attn)

    _recurse(model, False)
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return adapters, lora_params, total_params


def detach_kveraser_lora(model: nn.Module) -> None:
    """Restore unwrapped nn.Linear layers in-place."""
    def _recurse(mod: nn.Module) -> None:
        for child_name, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                setattr(mod, child_name, child.base)
            else:
                _recurse(child)
    _recurse(model)


def set_lora_enabled(adapters: list[LoRALinear], flag: bool) -> None:
    for ad in adapters:
        ad.enabled = flag


def build_kveraser_surrogate_cache(
    c_old_ctx: DynamicCache,
    c_lora_span: DynamicCache,
    p0: int,
    p1: int,
    T_ctx: int,
    retain_grad: bool = True,
) -> DynamicCache:
    """Eq.(4) KV_surrogate = KV_{1:p0-1} (+) KV_steer_{p0:p1} (+) KV_{p1:T_ctx}, keeping SSM states old."""
    surr = DynamicCache(config=None)
    surr.layers = []
    for l_old, l_sp in zip(c_old_ctx.layers, c_lora_span.layers):
        nl = object.__new__(type(l_old))
        nl.__dict__.update(l_old.__dict__)
        if hasattr(l_old, "keys") and l_old.keys is not None:
            k_old = l_old.keys if not retain_grad else l_old.keys.detach()
            v_old = l_old.values if not retain_grad else l_old.values.detach()
            k_sp = l_sp.keys[:, :, p0:p1, :]
            v_sp = l_sp.values[:, :, p0:p1, :]
            nl.keys = torch.cat([k_old[:, :, :p0, :], k_sp, k_old[:, :, p1:T_ctx, :]], dim=-2)
            nl.values = torch.cat([v_old[:, :, :p0, :], v_sp, v_old[:, :, p1:T_ctx, :]], dim=-2)
        if hasattr(l_old, "conv_states"):
            nl.conv_states = {
                k: (v.detach().clone() if v is not None else None) for k, v in l_old.conv_states.items()
            }
            nl.recurrent_states = {
                k: (v.detach().clone() if v is not None else None) for k, v in l_old.recurrent_states.items()
            }
            nl.is_conv_states_initialized = dict(l_old.is_conv_states_initialized)
            nl.is_recurrent_states_initialized = dict(l_old.is_recurrent_states_initialized)
            nl.has_previous_state = dict(l_old.has_previous_state)
            nl.conv_kernel_size = dict(l_old.conv_kernel_size)
        surr.layers.append(nl)
    return surr


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_shift_keys(keys: torch.Tensor, rope_mod: Any, delta_pos: int, pos_start: int) -> torch.Tensor:
    """Shift RoPE angles on keys[:, :, pos_start:, :] by `delta_pos` via rotary_emb."""
    if keys is None or keys.shape[-2] <= pos_start or delta_pos == 0 or rope_mod is None:
        return keys
    tail = keys[:, :, pos_start:, :]
    n_tail = tail.shape[-2]
    old_pos = torch.arange(pos_start, pos_start + n_tail, device=tail.device, dtype=torch.long).unsqueeze(0)
    new_pos = torch.clamp(old_pos + delta_pos, min=0)
    cos_o, sin_o = rope_mod(tail, old_pos)
    cos_n, sin_n = rope_mod(tail, new_pos)
    # Apply inverse(old) then forward(new) rotary over head_dim or rotary_dim
    rot_dim = cos_o.shape[-1]
    t_rot = tail[..., :rot_dim]
    t_pass = tail[..., rot_dim:]
    co, so = cos_o.unsqueeze(1).to(tail.dtype), sin_o.unsqueeze(1).to(tail.dtype)
    cn, sn = cos_n.unsqueeze(1).to(tail.dtype), sin_n.unsqueeze(1).to(tail.dtype)
    t_unrot = t_rot * co + rotate_half(t_rot) * (-so)
    t_rerot = t_unrot * cn + rotate_half(t_unrot) * sn
    shifted = torch.cat([t_rerot, t_pass], dim=-1) if t_pass.shape[-1] > 0 else t_rerot
    return torch.cat([keys[:, :, :pos_start, :], shifted], dim=-2)


def build_delete_and_shift_cache(
    model: Any,
    c_old_ctx: DynamicCache,
    p0: int,
    p1: int,
    T_ctx: int,
) -> tuple[DynamicCache, int]:
    """KVEraser Delete-and-Shift baseline: excise [p0, p1) from KV and RoPE-shift suffix left; SSM untouched."""
    del_len = p1 - p0
    out = DynamicCache(config=None)
    out.layers = []
    rope_mod = getattr(model.model, "rotary_emb", None)
    for l_old in c_old_ctx.layers:
        nl = object.__new__(type(l_old))
        nl.__dict__.update(l_old.__dict__)
        if hasattr(l_old, "keys") and l_old.keys is not None:
            k_cat = torch.cat([l_old.keys[:, :, :p0, :], l_old.keys[:, :, p1:T_ctx, :]], dim=-2)
            v_cat = torch.cat([l_old.values[:, :, :p0, :], l_old.values[:, :, p1:T_ctx, :]], dim=-2)
            nl.keys = apply_rope_shift_keys(k_cat, rope_mod, delta_pos=del_len, pos_start=p0) if False else None
            # Wait: old positions were p1..T_ctx-1, new shifted positions are p0..T_ctx-1-del_len (shift = -del_len)
            tail = l_old.keys[:, :, p1:T_ctx, :]
            if rope_mod is not None and tail.shape[-2] > 0:
                pos_o = torch.arange(p1, T_ctx, device=tail.device, dtype=torch.long).unsqueeze(0)
                pos_n = torch.arange(p0, T_ctx - del_len, device=tail.device, dtype=torch.long).unsqueeze(0)
                co, so = rope_mod(tail, pos_o)
                cn, sn = rope_mod(tail, pos_n)
                rot_dim = co.shape[-1]
                t_rot, t_pass = tail[..., :rot_dim], tail[..., rot_dim:]
                co, so = co.unsqueeze(1).to(tail.dtype), so.unsqueeze(1).to(tail.dtype)
                cn, sn = cn.unsqueeze(1).to(tail.dtype), sn.unsqueeze(1).to(tail.dtype)
                t_unrot = t_rot * co + rotate_half(t_rot) * (-so)
                t_rerot = t_unrot * cn + rotate_half(t_unrot) * sn
                tail_shifted = torch.cat([t_rerot, t_pass], dim=-1) if t_pass.shape[-1] > 0 else t_rerot
            else:
                tail_shifted = tail.clone()
            nl.keys = torch.cat([l_old.keys[:, :, :p0, :].clone(), tail_shifted], dim=-2)
            nl.values = v_cat.clone()
        if hasattr(l_old, "conv_states"):
            nl.conv_states = {
                k: (v.clone() if v is not None else None) for k, v in l_old.conv_states.items()
            }
            nl.recurrent_states = {
                k: (v.clone() if v is not None else None) for k, v in l_old.recurrent_states.items()
            }
            nl.is_conv_states_initialized = dict(l_old.is_conv_states_initialized)
            nl.is_recurrent_states_initialized = dict(l_old.is_recurrent_states_initialized)
            nl.has_previous_state = dict(l_old.has_previous_state)
            nl.conv_kernel_size = dict(l_old.conv_kernel_size)
        out.layers.append(nl)
    return out, T_ctx - del_len


def train_kveraser_lora(
    model: Any,
    tok: Any,
    cfg: Any,
    adapters: list[LoRALinear],
    train_cases: list[dict[str, Any]],
    device: torch.device,
    epochs: int = 3,
    lr: float = 2e-4,
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    """Train KVEraser rank-16 LoRA via Eq.(6) teacher-forced CE+KL against oracle suffix logits."""
    trainable = [p for ad in adapters for p in (ad.lora_A, ad.lora_B)]
    if not trainable:
        return {"train_time_s": 0.0, "steps": 0, "final_loss": 0.0, "grad_norm_nonzero": False}

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    t0 = sync_time(device)
    losses: list[float] = []
    grad_nonzero = False
    max_steps = epochs * len(train_cases)

    # Pre-cache oracle suffix target distributions & pre-edit caches on train_cases one-by-one inside loop
    # to avoid excessive VRAM duplication on 8GB GPU
    for ep in range(epochs):
        for case in train_cases:
            p0, p1, T_ctx = case["p0"], case["p1"], case["T_ctx"]
            ids_o = torch.tensor([case["ids_o"]], device=device, dtype=torch.long)
            ids_e = torch.tensor([case["ids_n"]], device=device, dtype=torch.long)
            ids_suffix = ids_e[:, T_ctx:]
            ans_ids, _ = resolve_answer_tokens(tok, case["new"])
            if not ans_ids:
                ans_ids = tok.encode(case["new"], add_special_tokens=False) or [0]
            ans_tensor = torch.tensor([ans_ids], device=device, dtype=torch.long)
            ids_suf_tf = torch.cat([ids_suffix, ans_tensor[:, :-1]], dim=1) if ans_tensor.shape[1] > 1 else ids_suffix

            with torch.no_grad():
                set_lora_enabled(adapters, False)
                c_pre = DynamicCache(config=cfg)
                if p0 > 0:
                    _ = run_segment(model, c_pre, ids_o[:, :p0], 0, max_chunk=1024)
                c_old_ctx = clone_cache(c_pre)
                _ = run_segment(model, c_old_ctx, ids_o[:, p0:T_ctx], p0, max_chunk=1024)

                c_ora = DynamicCache(config=cfg)
                _ = run_segment(model, c_ora, ids_e[:, :T_ctx], 0, max_chunk=1024)
                ora_suf_logits = run_segment(model, c_ora, ids_suf_tf, T_ctx, max_chunk=1024)
                ora_ans_logits = ora_suf_logits[:, -ans_tensor.shape[1]:, :].detach()

            opt.zero_grad(set_to_none=True)
            with torch.autograd.graph.saved_tensors_hooks(
                lambda t: t.clone() if (torch.is_tensor(t) and t.ndim >= 3) else t,
                lambda t: t,
            ):
                set_lora_enabled(adapters, True)
                c_lora = clone_cache(c_pre)
                pos_span = make_position_ids(p0, p1 - p0, device)
                _ = model(input_ids=ids_e[:, p0:p1], position_ids=pos_span, past_key_values=c_lora, use_cache=True)

                # Splice steered KV into frozen c_old_ctx; disable LoRA during suffix readout (Eq. 4+6)
                c_surr = build_kveraser_surrogate_cache(c_old_ctx, c_lora, p0, p1, T_ctx, retain_grad=True)
                set_lora_enabled(adapters, False)
                pos_suf = make_position_ids(T_ctx, int(ids_suf_tf.shape[1]), device)
                cand_suf_logits = model(
                    input_ids=ids_suf_tf, position_ids=pos_suf, past_key_values=c_surr, use_cache=True
                ).logits
                cand_ans_logits = cand_suf_logits[:, -ans_tensor.shape[1]:, :]

                # Eq.(6) KL alignment to clean oracle + teacher-forced CE toward edited answer tokens
                ce_loss = F.cross_entropy(cand_ans_logits.reshape(-1, cand_ans_logits.shape[-1]).float(), ans_tensor.reshape(-1))
                log_q = F.log_softmax(cand_ans_logits.float(), dim=-1)
                log_p = F.log_softmax(ora_ans_logits.float(), dim=-1)
                kl_loss = (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()
                loss = ce_loss + 0.5 * kl_loss
            loss.backward()

            total_gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, grad_clip).item())
            if total_gnorm > 0.0:
                grad_nonzero = True
            opt.step()
            losses.append(float(loss.item()))

            del c_pre, c_old_ctx, c_ora, c_lora, c_surr
        if device.type == "cuda":
            torch.cuda.empty_cache()

    set_lora_enabled(adapters, False)
    train_time_s = sync_time(device) - t0
    return {
        "train_time_s": float(train_time_s),
        "steps": int(max_steps),
        "initial_loss": float(losses[0]) if losses else 0.0,
        "final_loss": float(losses[-1]) if losses else 0.0,
        "mean_last10_loss": float(sum(losses[-10:]) / max(1, len(losses[-10:]))),
        "grad_norm_nonzero": grad_nonzero,
    }


def compute_hypic_vs_olocal_memory(cfg: Any, T_ctx: int, C_seg: int = 128) -> dict[str, Any]:
    """Exact theoretical metadata memory comparison: HYPIC Phi_m (d_k x d_k) vs O_local scalar G_t."""
    n_layers = int(getattr(cfg, "num_hidden_layers", 24))
    n_heads = int(getattr(cfg, "mamba_n_heads", getattr(cfg, "n_mamba_heads", 16)))
    d_head = int(getattr(cfg, "mamba_d_head", getattr(cfg, "mamba_headdim", 64)))
    d_state = int(getattr(cfg, "mamba_d_state", 64))
    n_segs = max(1, (T_ctx + C_seg - 1) // C_seg)

    # Full dense transition operator Phi_m in bf16 (2B): d_state x d_state per head per segment (or d_k x d_k)
    hypic_phi_bytes_per_seg = n_layers * n_heads * d_state * d_state * 2
    hypic_total_bytes = n_segs * hypic_phi_bytes_per_seg

    # O_local cumulative scalar log-decay G_t: fp32 per step (4B) or checkpoint every C=512 in fp32
    olocal_scalar_per_step_bytes = n_layers * n_heads * 4
    olocal_per_seg_ckpt_bytes = olocal_scalar_per_step_bytes
    return {
        "T_ctx": int(T_ctx),
        "C_seg": int(C_seg),
        "n_segs": int(n_segs),
        "d_state": int(d_state),
        "hypic_phi_bytes_per_seg": int(hypic_phi_bytes_per_seg),
        "hypic_total_bytes_for_T": int(hypic_total_bytes),
        "olocal_scalar_bytes_per_step": int(olocal_scalar_per_step_bytes),
        "olocal_bytes_per_seg_ckpt": int(olocal_per_seg_ckpt_bytes),
        "matrix_to_scalar_ratio_per_seg": float(hypic_phi_bytes_per_seg / max(1, olocal_per_seg_ckpt_bytes)),
    }


def run_sb1_suite(
    model_key: str,
    model: Any,
    tok: Any,
    cfg: Any,
    mixers: list[tuple[int, Any]],
    device: torch.device,
    eos_ids: list[int],
    train_cases: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    block_horizon: str,
    bckpt_chunk: int,
    jsonl_path: Path,
    c_seg: int = 128,
    w_seam: int = 32,
    lora_epochs: int = 3,
) -> dict[str, Any]:
    """Run SB1 baselines (KVEraser LoRA, Delete&Shift, HYPIC variants) vs O_local."""
    n_linear = len(mixers)
    w_conv = int(getattr(cfg, "mamba_d_conv", getattr(cfg, "conv_kernel_size", 4)))
    b_offset = resolve_block_offset(block_horizon, n_linear, w_conv, 9)
    done_keys = load_completed_keys(jsonl_path)

    # Check if already all completed before paying LoRA train time again
    pending = [idx for idx in range(len(test_cases)) if f"{model_key}:sb1:{idx}" not in done_keys]

    adapters, lora_params, total_params = attach_kveraser_lora(model, rank=16, alpha=32.0)
    lora_train_stats = {"skipped": True}
    if pending:
        lora_train_stats = train_kveraser_lora(
            model, tok, cfg, adapters, train_cases, device, epochs=lora_epochs, lr=2e-4
        )
    else:
        set_lora_enabled(adapters, False)

    cond_names = [
        "old", "oracle", "J-local", "kv_only",
        "kveraser_lora_steer", "kveraser_delete_shift",
        "hypic_segmented_seam32", "hypic_unaligned_seam32", "hypic_chunk_zero_init",
    ]

    for s_idx, case in enumerate(test_cases):
        rec_key = f"{model_key}:sb1:{s_idx}"
        if rec_key in done_keys:
            continue

        ids_o = torch.tensor([case["ids_o"]], device=device, dtype=torch.long)
        ids_e = torch.tensor([case["ids_n"]], device=device, dtype=torch.long)
        p0, p1, T_ctx = case["p0"], case["p1"], case["T_ctx"]
        block_end = min(T_ctx, p1 + b_offset)
        ids_suffix = ids_e[:, T_ctx:]
        fmt = case["format"]

        # Aligned HYPIC chunk boundaries [s0, s1) containing p0
        s0 = (p0 // c_seg) * c_seg
        s1 = min(T_ctx, s0 + c_seg)
        seam_end = min(T_ctx, s1 + w_seam)
        unalign_end = min(T_ctx, p1 + w_seam)

        with torch.no_grad():
            set_lora_enabled(adapters, False)
            # Base caches
            c_pre, _ = prepare_bckpt_cache(model, cfg, ids_o, p0, chunk_size=bckpt_chunk)
            c_old_b = clone_cache(c_pre)
            _ = run_segment(model, c_old_b, ids_o[:, p0:block_end], p0)
            c_old_ctx = clone_cache(c_old_b)
            _, g_sums, _ = run_segment_with_mamba_decay(model, c_old_ctx, ids_o[:, block_end:T_ctx], block_end, mixers)

            # 1. O_local (J-local)
            t0 = sync_time(device)
            c_new_blk = clone_cache(c_pre)
            _ = run_segment(model, c_new_blk, ids_e[:, p0:block_end], p0)
            delta_b = compute_state_delta_fp64(c_old_b, c_new_blk)
            c_jlocal = apply_mamba2_jlocal_update(
                c_old_ctx, c_new_blk, delta_b, g_sums,
                start_pos=p0, block_end=block_end, T_ctx=T_ctx, w_conv=w_conv,
                use_decay=True, patch_state=True, patch_kv=True,
            )
            t_jlocal_ms = (sync_time(device) - t0) * 1000.0

            # 2. Oracle & kv_only
            c_oracle = DynamicCache(config=cfg)
            _ = run_segment(model, c_oracle, ids_e[:, :T_ctx], 0)
            c_kv_only = apply_mamba2_jlocal_update(
                c_old_ctx, c_new_blk, delta_b, g_sums,
                start_pos=p0, block_end=block_end, T_ctx=T_ctx, w_conv=w_conv,
                use_decay=True, patch_state=False, patch_kv=True,
            )

            # 3. KVEraser LoRA Steer (Eq. 4)
            t0 = sync_time(device)
            set_lora_enabled(adapters, True)
            c_lora = clone_cache(c_pre)
            _ = run_segment(model, c_lora, ids_e[:, p0:p1], p0)
            set_lora_enabled(adapters, False)
            c_kveraser = build_kveraser_surrogate_cache(c_old_ctx, c_lora, p0, p1, T_ctx, retain_grad=False)
            t_kveraser_ms = (sync_time(device) - t0) * 1000.0

            # 4. KVEraser Delete-and-Shift (2606.17034 Baseline)
            t0 = sync_time(device)
            c_delshift, T_ctx_del = build_delete_and_shift_cache(model, c_old_ctx, p0, p1, T_ctx)
            t_delshift_ms = (sync_time(device) - t0) * 1000.0

            # 5. HYPIC segmented + W_seam=32 + exact chunk operator transport across [seam_end, T_ctx)
            t0 = sync_time(device)
            c_s0_old, _ = prepare_bckpt_cache(model, cfg, ids_o, s0, chunk_size=bckpt_chunk)
            c_hyp_old_end = clone_cache(c_s0_old)
            _ = run_segment(model, c_hyp_old_end, ids_o[:, s0:seam_end], s0)
            c_hyp_tmp = clone_cache(c_hyp_old_end)
            _, g_hyp_suf, _ = run_segment_with_mamba_decay(
                model, c_hyp_tmp, ids_o[:, seam_end:T_ctx], seam_end, mixers
            )
            c_hyp_new_end = clone_cache(c_s0_old)
            _ = run_segment(model, c_hyp_new_end, ids_e[:, s0:seam_end], s0)
            delta_hyp = compute_state_delta_fp64(c_hyp_old_end, c_hyp_new_end)
            c_hypic_seg = apply_mamba2_jlocal_update(
                c_old_ctx, c_hyp_new_end, delta_hyp, g_hyp_suf,
                start_pos=s0, block_end=seam_end, T_ctx=T_ctx, w_conv=w_conv,
                use_decay=True, patch_state=True, patch_kv=True,
            )
            t_hypic_seg_ms = (sync_time(device) - t0) * 1000.0

            # 6. HYPIC unaligned mid-segment seam32 splice without suffix transport (stale SSM S_T)
            t0 = sync_time(device)
            c_unalign = clone_cache(c_pre)
            _ = run_segment(model, c_unalign, ids_e[:, p0:unalign_end], p0)
            c_hypic_unalign = clone_cache(c_old_ctx)
            patch_kv_range(c_hypic_unalign, c_unalign, p0, unalign_end)
            t_hypic_unalign_ms = (sync_time(device) - t0) * 1000.0

            # 7. HYPIC chunk from S=0 (independent chunk encoding, zero pre-chunk state)
            c_hyp_zero_old = DynamicCache(config=cfg)
            _ = run_segment(model, c_hyp_zero_old, ids_o[:, s0:seam_end], s0)
            c_hyp_zero_new = DynamicCache(config=cfg)
            _ = run_segment(model, c_hyp_zero_new, ids_e[:, s0:seam_end], s0)
            delta_hyp_z = compute_state_delta_fp64(c_hyp_zero_old, c_hyp_zero_new)
            c_hypic_zero = apply_mamba2_jlocal_update(
                c_old_ctx, c_hyp_zero_new, delta_hyp_z, g_hyp_suf,
                start_pos=s0, block_end=seam_end, T_ctx=T_ctx, w_conv=w_conv,
                use_decay=True, patch_state=True, patch_kv=False,
            )
            seg_len_z = seam_end - s0
            for l_dst, l_src in zip(c_hypic_zero.layers, c_hyp_zero_new.layers):
                if getattr(l_src, "keys", None) is not None and getattr(l_dst, "keys", None) is not None:
                    l_dst.keys[:, :, s0:seam_end, :].copy_(l_src.keys[:, :, :seg_len_z, :])
                    l_dst.values[:, :, s0:seam_end, :].copy_(l_src.values[:, :, :seg_len_z, :])

        cond_map = {
            "old": (c_old_ctx, T_ctx),
            "oracle": (c_oracle, T_ctx),
            "J-local": (c_jlocal, T_ctx),
            "kv_only": (c_kv_only, T_ctx),
            "kveraser_lora_steer": (c_kveraser, T_ctx),
            "kveraser_delete_shift": (c_delshift, T_ctx_del),
            "hypic_segmented_seam32": (c_hypic_seg, T_ctx),
            "hypic_unaligned_seam32": (c_hypic_unalign, T_ctx),
            "hypic_chunk_zero_init": (c_hypic_zero, T_ctx),
        }

        evals = {}
        for cname, (ccache, t_pos) in cond_map.items():
            evals[cname] = eval_suffix_condition(
                model, tok, ccache, ids_suffix, t_pos,
                case["new"], case["old"], fmt, max_new=8, eos_ids=eos_ids,
            )

        lg_ora = evals["oracle"]["last_logits"]
        ans_ora = evals["oracle"]["ans_logits_new"]
        out_conds = {}
        for cname in cond_names:
            ev = evals[cname]
            out_conds[cname] = {
                "acc_new": ev["acc_new"],
                "strict_new": ev["strict_new"],
                "acc_old": ev["acc_old"],
                "p_new_1st": ev["p_new_1st"],
                "p_old_1st": ev["p_old_1st"],
                "kl_vs_oracle": kl_div_from_logits(lg_ora, ev["last_logits"]),
                "ans_mean_kl": mean_ans_kl(ans_ora, ev["ans_logits_new"]),
                "top1_eq_oracle": ev["top1_id"] == evals["oracle"]["top1_id"],
                "gen_text": ev["gen_text"],
            }

        rec = {
            "record_key": rec_key,
            "suite": "SB1",
            "model": model_key,
            "sample_idx": s_idx,
            "format": fmt,
            "p0": p0,
            "p1": p1,
            "s0": s0,
            "seam_end": seam_end,
            "T_ctx": T_ctx,
            "tokens_recomputed": {
                "J-local": int(block_end - p0),
                "kveraser_lora_steer": int(p1 - p0),
                "hypic_segmented_seam32": int(seam_end - s0),
                "hypic_unaligned_seam32": int(unalign_end - p0),
            },
            "timing_ms": {
                "J-local": t_jlocal_ms,
                "kveraser_lora_steer": t_kveraser_ms,
                "kveraser_delete_shift": t_delshift_ms,
                "hypic_segmented_seam32": t_hypic_seg_ms,
                "hypic_unaligned_seam32": t_hypic_unalign_ms,
            },
            "conditions": out_conds,
        }
        append_jsonl(jsonl_path, rec)

        del c_pre, c_old_b, c_old_ctx, c_jlocal, c_oracle, c_kv_only
        del c_lora, c_kveraser, c_delshift, c_s0_old, c_hyp_old_end, c_hyp_tmp
        del c_hyp_new_end, c_hypic_seg, c_unalign, c_hypic_unalign, c_hyp_zero_old, c_hyp_zero_new, c_hypic_zero
        if device.type == "cuda":
            torch.cuda.empty_cache()

    detach_kveraser_lora(model)

    # Aggregate SB1
    all_recs = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("suite") == "SB1" and r.get("model") == model_key:
                        all_recs.append(r)

    n = len(all_recs)
    agg = {}
    for cname in cond_names:
        k_new = sum(1 for r in all_recs if r["conditions"][cname]["acc_new"])
        k_old = sum(1 for r in all_recs if r["conditions"][cname]["acc_old"])
        k_top1 = sum(1 for r in all_recs if r["conditions"][cname]["top1_eq_oracle"])
        kls = [r["conditions"][cname]["kl_vs_oracle"] for r in all_recs]
        agg[cname] = {
            "n": n,
            "acc_new": k_new / max(1, n),
            "acc_new_ci95": wilson_ci(k_new, n),
            "acc_old": k_old / max(1, n),
            "top1_eq_oracle": k_top1 / max(1, n),
            "mean_kl_vs_oracle": sum(kls) / max(1, n),
        }

    mean_toks = {
        k: sum(r["tokens_recomputed"][k] for r in all_recs) / max(1, n)
        for k in ("J-local", "kveraser_lora_steer", "hypic_segmented_seam32", "hypic_unaligned_seam32")
    }
    mean_ms = {
        k: sum(r["timing_ms"][k] for r in all_recs) / max(1, n)
        for k in ("J-local", "kveraser_lora_steer", "kveraser_delete_shift", "hypic_segmented_seam32", "hypic_unaligned_seam32")
    }

    return {
        "suite": "SB1",
        "model": model_key,
        "n_test": n,
        "n_train": len(train_cases),
        "lora_config": {"rank": 16, "alpha": 32.0, "lora_params": lora_params, "total_params": total_params},
        "lora_train_stats": lora_train_stats,
        "memory_comparison_C128": compute_hypic_vs_olocal_memory(cfg, all_recs[0]["T_ctx"] if all_recs else 512, C_seg=c_seg),
        "mean_tokens_recomputed": mean_toks,
        "mean_timing_ms": mean_ms,
        "conditions": agg,
    }


def run_synthetic_dry_run(out_dir: Path) -> dict[str, Any]:
    """Verify LoRA autograd, Delete-and-Shift, HYPIC, and scaled E2/E3 on tiny CPU synthetic models."""
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

    class _MockTok:
        eos_token_id = 0
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(x) for x in ids)
        def encode(self, s, add_special_tokens=False):
            return [int(x) % 120 + 1 if x.isdigit() else (sum(ord(c) for c in x) % 120 + 1) for x in s.split()] or [5]
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return msgs[0]["content"] + " "
        def convert_tokens_to_ids(self, s):
            return 0

    tok = _MockTok()
    rep = {}
    for mkey, cls, cfg in [("syn_falcon", FalconH1ForCausalLM, f_cfg), ("syn_zamba2", Zamba2ForCausalLM, z_cfg)]:
        mod = cls(cfg).eval()
        mixers = extract_mamba_mixers(mod)
        cases = []
        for idx in range(2):
            ids_o = torch.randint(10, 120, (1, 96)).tolist()[0]
            ids_n = list(ids_o)
            ids_n[34:38] = [(ids_n[j] + 13) % 128 for j in range(34, 38)]
            cases.append({
                "ok": True, "tpl_id": 0, "format": "digit4", "target_name": "Alice",
                "old": "10", "new": "23", "expected_len": 4,
                "p0": 34, "p1": 38, "T_ctx": 84, "T_total": 96,
                "ids_o": ids_o, "ids_n": ids_n, "full_old": "", "query_suffix": "Q:",
            })
        sb1_p = out_dir / f"_tmp_dry_{mkey}_sb1.jsonl"
        sb1_p.unlink(missing_ok=True)
        res_sb1 = run_sb1_suite(
            mkey, mod, tok, cfg, mixers, torch.device("cpu"), [0],
            train_cases=cases[:1], test_cases=cases[1:],
            block_horizon="stage", bckpt_chunk=32, jsonl_path=sb1_p,
            c_seg=32, w_seam=8, lora_epochs=1,
        )
        sb1_p.unlink(missing_ok=True)
        assert res_sb1["lora_train_stats"]["grad_norm_nonzero"], f"Zero LoRA grad on {mkey}"
        rep[mkey] = {
            "grad_norm_nonzero": res_sb1["lora_train_stats"]["grad_norm_nonzero"],
            "lora_params": res_sb1["lora_config"]["lora_params"],
            "n_test": res_sb1["n_test"],
        }

    out_path = out_dir / "w11_crossfamily_dryrun.json"
    out_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="falcon-h1,zamba2")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--e2-n", type=int, default=100)
    parser.add_argument("--e3-k2-n", type=int, default=10)
    parser.add_argument("--e3-k4-n", type=int, default=5)
    parser.add_argument("--sb1-train-n", type=int, default=40)
    parser.add_argument("--sb1-test-n", type=int, default=60)
    parser.add_argument("--lora-epochs", type=int, default=3)
    parser.add_argument("--block-horizon", type=str, default="deep")
    parser.add_argument("--bckpt-chunk", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default="pilot")
    parser.add_argument("--synthetic-dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic_dry_run:
        rep = run_synthetic_dry_run(out_dir)
        print("DRYRUN_OK", json.dumps(rep))
        return

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    toks = [AutoTokenizer.from_pretrained(MODEL_REGISTRY[k]["repo_id"], use_fast=True) for k in model_keys]
    shared_words = collect_shared_single_tokens(toks)
    print(f"[SB5+SB1] Shared single-token words across {model_keys}: {len(shared_words)}", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    e2_jsonl = out_dir / "w11_crossfamily_e2_records.jsonl"
    e3_jsonl = out_dir / "w11_crossfamily_e3_records.jsonl"
    sb1_jsonl = out_dir / "w11_crossfamily_sb1_records.jsonl"
    summary_all: dict[str, Any] = {
        "shared_words_count": len(shared_words),
        "device": str(device),
        "models": {},
    }

    for mkey, tok in zip(model_keys, toks):
        repo_id = MODEL_REGISTRY[mkey]["repo_id"]
        print(f"=== Loading {mkey} ({repo_id}) on {device} ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map={"": str(device)})
        model.eval()
        cfg = model.config
        mixers = extract_mamba_mixers(model)
        eos_ids = get_eos_ids(model, tok)

        pools = build_case_pools(
            tok, shared_words,
            n_e2=args.e2_n, n_sb1_train=args.sb1_train_n, n_sb1_test=args.sb1_test_n,
        )
        print(f"[{mkey}] Pools: E2={len(pools['e2_cases'])}, SB1_train={len(pools['sb1_train'])}, SB1_test={len(pools['sb1_test'])}", flush=True)

        # 1. SB5 Scaled E2
        t0 = time.perf_counter()
        e2_sum = run_e2_suite(
            mkey, model, tok, cfg, mixers, device, eos_ids,
            n_samples=len(pools["e2_cases"]), block_horizon=args.block_horizon,
            custom_offset=9, bckpt_chunk=args.bckpt_chunk,
            jsonl_path=e2_jsonl, synthetic_cases=pools["e2_cases"],
        )
        print(f"[{mkey}] E2 n={e2_sum['n']} done in {time.perf_counter()-t0:.1f}s: J-local={e2_sum['conditions']['J-local']['acc_new']:.3f}, oracle={e2_sum['conditions']['oracle']['acc_new']:.3f}", flush=True)

        # 2. SB5 Scaled E3
        t0 = time.perf_counter()
        e3_cfgs = build_scaled_e3_configs(tok, k2_n=args.e3_k2_n, k4_n=args.e3_k4_n)
        e3_sum = run_e3_suite(
            mkey, model, tok, cfg, mixers, device, eos_ids,
            k2_n=args.e3_k2_n, k4_n=args.e3_k4_n, block_horizon=args.block_horizon,
            custom_offset=9, bckpt_chunk=args.bckpt_chunk,
            jsonl_path=e3_jsonl, synthetic_cases=e3_cfgs,
        )
        print(f"[{mkey}] E3 done in {time.perf_counter()-t0:.1f}s", flush=True)

        # 3. SB1 KVEraser + HYPIC vs O_local
        t0 = time.perf_counter()
        sb1_sum = run_sb1_suite(
            mkey, model, tok, cfg, mixers, device, eos_ids,
            train_cases=pools["sb1_train"], test_cases=pools["sb1_test"],
            block_horizon=args.block_horizon, bckpt_chunk=args.bckpt_chunk,
            jsonl_path=sb1_jsonl, c_seg=128, w_seam=32, lora_epochs=args.lora_epochs,
        )
        print(
            f"[{mkey}] SB1 done in {time.perf_counter()-t0:.1f}s: "
            f"J-local={sb1_sum['conditions']['J-local']['acc_new']:.3f}, "
            f"kveraser={sb1_sum['conditions']['kveraser_lora_steer']['acc_new']:.3f}, "
            f"delshift={sb1_sum['conditions']['kveraser_delete_shift']['acc_new']:.3f}, "
            f"hypic_seg={sb1_sum['conditions']['hypic_segmented_seam32']['acc_new']:.3f}, "
            f"hypic_unalign={sb1_sum['conditions']['hypic_unaligned_seam32']['acc_new']:.3f}",
            flush=True,
        )

        summary_all["models"][mkey] = {"e2": e2_sum, "e3": e3_sum, "sb1": sb1_sum}

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        sum_path = out_dir / "w11_crossfamily_scaleup_summary.json"
        sum_path.write_text(json.dumps(summary_all, indent=2, ensure_ascii=False), encoding="utf-8")

    print("ALL_W11_CROSSFAMILY_DONE", flush=True)


if __name__ == "__main__":
    main()
