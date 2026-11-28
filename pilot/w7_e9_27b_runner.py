#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W7 E9 27B 4-bit (NF4) Validation Runner for Qwen/Qwen3.5-27B (Colab A100-40GB Target).

Combines:
  - Proven E2/E3 evaluation harness from pilot/w2_2b_full_e2_e3.py
  - Proven A100 4-bit NF4 model setup from pilot/w5_e9_27b_smoke.py
  - Wave 4+5 locked protocol:
    * Universal O(1) default: scalar transport-decay ("scalar_decay") in BOTH E2 & E3
    * Non-default ablation preserved as "J_local_iso" ("iso_decay")
    * Expanded POST_FILLER_BLOCKS (~245-265 toks) so horizon +141 leaves >=25 suffix tokens
    * E9 stratified split: E2=120 (round-robin across 3 templates) + E3=80 (24/18/18/20)
    * --probe-10 / --e2-limit 6 --e3-per-cell-limit 1 strict prefix compatible with --resume
    * Per-sample wall_s, peak_allocated_gb, peak_reserved_gb, headroom_gb
    * --timeout-s graceful exit saving JSON summary before Colab's 15-min exec ceiling
    * --dataset-only (0-CU local E0c audit) & --self-test (tiny CPU synthetic Qwen3_5Config)
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a helpful assistant. Reply with only the exact requested value, "
    "no reasoning. Do not use markdown or code blocks."
)

ENTITIES_8 = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]
LETTERS_26 = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
DIGITS_10 = [str(d) for d in range(10)]

PRE_FILLER_TEMPLATES = [
    "Archive batch {idx} records routine system parameters before the session update.",
    "Internal ledger section {idx} stores verified access credentials for active personnel.",
    "Compliance note {idx} summarizes reference entries logged earlier this morning.",
]

# Expanded post-fillers (~245-265 tokens each) provide >=25 non-trivial suffix tokens
# even at 27B deep horizon (+141) after both target facts and the "stone" placebo span.
# Zero digits, zero single-letter codes, zero ENTITIES_8 names, and unique "stone" span.
POST_FILLER_BLOCKS = [
    (
        "Operational memo alpha notes that maintenance teams checked backup generators on Tuesday. "
        "The courtyard walkway uses weathered stone tiles along the eastern perimeter wall. "
        "Telemetry archives confirm baseline humidity remained steady inside server room four throughout the afternoon shift. "
        "Regional dispatchers scheduled standard inspection rounds for all perimeter gates before sunset. "
        "Facility logs also document lighting calibration across the main lobby and loading dock. "
        "Custodial supervisors verified supply closets along the lower mezzanine corridor before closing time. "
        "HVAC technicians replaced air intake filters across roof units north and south without interruption. "
        "Evening reception desks filed standard visitor badges and locked exterior service doors on schedule. "
        "Plumbers flushed reserve cooling loops along the east mechanical gallery before noon. "
        "Mailroom clerks sorted incoming parcel bins according to standard routing tables. "
        "Grounds crews cleared fallen leaves from storm drains surrounding the southern terrace. "
        "Archivists labeled storage cartons containing retired telemetry binders for offsite retention. "
        "Electricians tested backup transfer switches inside the lower utility vault without incident. "
        "Safety coordinators restocked first aid cabinets across every workshop floor after inspection. "
        "Janitorial teams polished terrazzo hallways leading toward the central freight elevator. "
        "Fleet mechanics checked tire pressure across campus utility carts prior to morning rounds. "
        "Warehouse attendants stacked wooden pallets along designated storage aisles inside annex bay. "
        "Night watchmen logged routine perimeter patrols across the outer courtyard gates without exception."
    ),
    (
        "Engineering digest beta reports routine firmware audits completed without alerts on Wednesday. "
        "A smooth gray stone marker stands beside the north service entrance path. "
        "Environmental sensors recorded stable airflow across cabinet rows three and seven during overnight diagnostics. "
        "Logistics coordinators updated delivery manifests for incoming hardware shipments next quarter. "
        "Security briefings confirmed badge readers operated normally across every stairwell checkpoint. "
        "Network operators inspected fiber patch panels inside basement rack enclosures early Thursday morning. "
        "Groundskeepers trimmed hedges bordering the visitor parking apron before midday showers arrived. "
        "Quality assurance engineers archived weekly thermal bench logs into central storage repositories. "
        "Technicians calibrated barometer gauges mounted inside the upper ventilation penthouse. "
        "Carpenters tightened brass hinges along heavy timber doors in the west atrium. "
        "Supply clerks balanced monthly toner inventories across shared copy rooms on Friday. "
        "Drain specialists inspected rainwater sumps beneath the service driveway during dry weather. "
        "Acoustic engineers measured fan vibration levels beside the primary chiller plant. "
        "Reception assistants updated conference room booking schedules for upcoming vendor visits. "
        "Painters touched up scuffed baseboards along the north corridor before weekend hours. "
        "Landscapers mulched ornamental planters flanking the main pedestrian plaza near dusk. "
        "Instrument specialists verified thermometer probes inside cold storage lockers without drift. "
        "Night operators reviewed automated pump runtimes across municipal water boost stations."
    ),
    (
        "Operations bulletin gamma summarizes scheduled cooling checks across auxiliary transformer bays. "
        "Workers repaired one cracked stone step near the west courtyard loading ramp. "
        "Power distribution panels maintained nominal voltage across all primary and secondary circuits overnight. "
        "Inventory clerks verified spare cabling crates stored inside warehouse annex two before noon. "
        "Administrative staff circulated updated emergency contact rosters to every floor supervisor. "
        "Safety inspectors tested stairwell emergency lighting circuits throughout both wing corridors yesterday. "
        "Shuttle drivers followed adjusted perimeter routes while paving crews resurfaced the outer service lane. "
        "Procurement officers finalized seasonal stationery requisitions for administrative support offices. "
        "Machinists greased overhead crane rails inside the central maintenance bay on Monday. "
        "Roofing crews sealed flashing seams around skylights above the archive reading room. "
        "Glaziers inspected tempered window seals along the upper mezzanine walkway during morning light. "
        "Transport planners posted revised loading dock hours for regional freight carriers. "
        "Calibration staff checked pressure relief valves attached to auxiliary steam manifolds. "
        "Custodians mopped tiled entryways near the eastern bicycle shelter prior to arrival. "
        "Welders fabricated replacement steel brackets for outdoor handrail repairs along the ramp. "
        "Dispatch assistants filed completed work orders inside chronological binder shelves after lunch. "
        "Grounds workers pruned low branches overhanging the perimeter fence along maple lane. "
        "Facility auditors compiled quarterly energy meter readings into standard summary ledgers."
    ),
]

PLACEBO_OLD = "stone"
PLACEBO_NEW = "brick"

SUFFIX_INSTR = {
    "digit4": "Do not use markdown or code blocks. Output exactly the 4-digit code.",
    "single_digit": "Reply with only the single digit. Do not use markdown or code blocks.",
    "single_letter": "Reply with only the single letter. Do not use markdown or code blocks.",
}

E3_PERMS_K2 = [(0, 1), (1, 0)]
# P=6 representative permutations for k=4: canonical + full reverse + head-tail swap + middle swap + pairwise swap + interleaved
E3_PERMS_K4 = [
    (0, 1, 2, 3),
    (3, 2, 1, 0),
    (3, 1, 2, 0),
    (0, 2, 1, 3),
    (1, 0, 3, 2),
    (2, 0, 3, 1),
]

E9_E3_CELL_SPECS = [
    ("e3_m4_k2", 4, 2, 24),
    ("e3_m4_k4", 4, 4, 18),
    ("e3_m8_k2", 8, 2, 18),
    ("e3_m8_k4", 8, 4, 20),
]


def get_pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def configure_hf_environment(offline: bool):
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if os.name == "nt":
            os.environ.setdefault("HF_HOME", r"D:\applicationData\HuggingFace")
    else:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def sync_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def get_vram_snapshot(device: torch.device, total_vram_gb: float) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "peak_allocated_gb": 0.0,
            "peak_reserved_gb": 0.0,
            "headroom_gb": 0.0,
            "total_vram_gb": float(total_vram_gb),
        }
    alloc_gb = float(torch.cuda.memory_allocated(device) / (1024 ** 3))
    res_gb = float(torch.cuda.memory_reserved(device) / (1024 ** 3))
    peak_alloc_gb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
    peak_res_gb = float(torch.cuda.max_memory_reserved(device) / (1024 ** 3))
    headroom_gb = float(max(0.0, total_vram_gb - peak_res_gb))
    return {
        "allocated_gb": alloc_gb,
        "reserved_gb": res_gb,
        "peak_allocated_gb": peak_alloc_gb,
        "peak_reserved_gb": peak_res_gb,
        "headroom_gb": headroom_gb,
        "total_vram_gb": float(total_vram_gb),
    }


def format_fact_parts(tpl_id: int, name: str, code: str) -> tuple[str, int, int]:
    """Return (sentence, val_start_char, val_end_char) inside the sentence."""
    if tpl_id == 0:
        prefix = f"The secret identifier for {name} is "
    elif tpl_id == 1:
        prefix = f"For {name}, the assigned passkey is "
    elif tpl_id == 2:
        prefix = f"{name} has access tag "
    else:
        raise ValueError(f"Unsupported tpl_id={tpl_id}")
    return f"{prefix}{code}.", len(prefix), len(prefix) + len(code)


def format_query_suffix(tpl_id: int, target_fact: str, pair_format: str) -> str:
    if tpl_id == 0:
        q = f" Question: What is the secret identifier for {target_fact}?"
    elif tpl_id == 1:
        q = f" Question: What is the assigned passkey for {target_fact}?"
    elif tpl_id == 2:
        q = f" Question: What is {target_fact}'s access tag?"
    else:
        raise ValueError(f"Unsupported tpl_id={tpl_id}")
    return q + " " + SUFFIX_INSTR[pair_format]


def render_chat(tok, user_content: str) -> tuple[str, dict[str, Any]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        rendered = tok.apply_chat_template(messages, **kwargs)
        thinking_flag_supported = True
    except TypeError:
        kwargs.pop("enable_thinking", None)
        rendered = tok.apply_chat_template(messages, **kwargs)
        thinking_flag_supported = False
    meta = {
        "thinking_flag_supported": thinking_flag_supported,
        "has_empty_think_block": ("<think>\n\n</think>" in rendered or "<think></think>" in rendered),
    }
    return rendered, meta


def get_text_cfg(cfg):
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", cfg)


def count_linear_layers(t_cfg) -> int:
    lt = getattr(t_cfg, "layer_types", None)
    if lt:
        return sum(1 for x in lt if "linear" in str(x).lower())
    return int(getattr(t_cfg, "num_hidden_layers", 64))


def count_full_layers(t_cfg) -> int:
    lt = getattr(t_cfg, "layer_types", None)
    if lt:
        return sum(1 for x in lt if "full" in str(x).lower())
    return 0


def get_conv_kernel_dim(t_cfg) -> int:
    return int(getattr(t_cfg, "linear_conv_kernel_dim", getattr(t_cfg, "conv_kernel_dim", 4)))


def get_base_model(model):
    return getattr(model, "model", model)


def get_text_model(model):
    base = get_base_model(model)
    return getattr(base, "language_model", base)


def resolve_exec_device(model, fallback: str = "cuda:0") -> torch.device:
    text_m = get_text_model(model)
    emb = getattr(text_m, "embed_tokens", None)
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback if torch.cuda.is_available() else "cpu")


def make_position_ids(start: int, length: int, device, base_model, batch_size: int = 1) -> torch.Tensor:
    pos_1d = torch.arange(start, start + length, device=device, dtype=torch.long)
    if hasattr(base_model, "language_model"):
        return pos_1d.view(1, 1, -1).expand(3, batch_size, -1)
    return pos_1d.view(1, 1, -1).expand(4, batch_size, -1)


def clone_cache(cache):
    if cache is None:
        return None
    return copy.deepcopy(cache)


def truncate_cache_kv(cache, target_len: int):
    """Truncate KV sequence length to target_len on attention layers without mutating linear states."""
    for layer in getattr(cache, "layers", []):
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        if k is not None and k.numel() > 0:
            layer.keys = k[..., :target_len, :].detach().clone()
        if v is not None and v.numel() > 0:
            layer.values = v[..., :target_len, :].detach().clone()
        if hasattr(layer, "cumulative_length") and isinstance(layer.cumulative_length, int):
            layer.cumulative_length = target_len
        if hasattr(layer, "cumulative_length_int"):
            layer.cumulative_length_int = target_len
    return cache


def cache_seq_len(cache) -> int:
    if hasattr(cache, "get_seq_length"):
        try:
            return int(cache.get_seq_length())
        except Exception:
            pass
    for layer in getattr(cache, "layers", []):
        k = getattr(layer, "keys", None)
        if k is not None and k.numel() > 0:
            return int(k.shape[-2])
    return 0


def _iter_layer_state_items(layer, attr: str):
    val = getattr(layer, attr, None)
    if val is None:
        return
    if isinstance(val, dict):
        for k_idx, tensor in val.items():
            if tensor is not None:
                yield k_idx, tensor
    else:
        yield None, val


def snap_states(cache) -> dict[tuple[int, str, Any], torch.Tensor]:
    out = {}
    for li, layer in enumerate(cache.layers):
        for attr in ("recurrent_states", "conv_states"):
            for k_idx, val in _iter_layer_state_items(layer, attr):
                out[(li, attr, k_idx)] = val.detach().clone()
    return out


def overwrite_states(cache, states: dict[tuple[int, str, Any], torch.Tensor]):
    """Decision A direct copy_ of s_new (recurrent_states + conv_states)."""
    for (li, attr, k_idx), val in states.items():
        container = getattr(cache.layers[li], attr, None)
        if container is None:
            continue
        if k_idx is not None:
            target = container.get(k_idx)
            if target is None:
                container[k_idx] = val.detach().clone()
            else:
                target.copy_(val.to(dtype=target.dtype, device=target.device))
        else:
            container.copy_(val.to(dtype=container.dtype, device=container.device))


def compute_state_delta_fp64(
    old_states: dict[tuple[int, str, Any], torch.Tensor],
    new_states: dict[tuple[int, str, Any], torch.Tensor],
) -> dict[tuple[int, str, Any], torch.Tensor]:
    """Decision B: compute state delta in fp64 (double)."""
    return {
        key: new_states[key].detach().to(torch.float64) - old_states[key].detach().to(torch.float64)
        for key in new_states
    }


def patch_kv_range(cache, src_cache, start_pos: int, end_pos: int):
    for dst_l, src_l in zip(cache.layers, src_cache.layers):
        dst_k = getattr(dst_l, "keys", None)
        src_k = getattr(src_l, "keys", None)
        dst_v = getattr(dst_l, "values", None)
        src_v = getattr(src_l, "values", None)
        if dst_k is None or src_k is None or dst_v is None or src_v is None:
            continue
        if dst_k.numel() == 0 or src_k.numel() == 0:
            continue
        dst_k[..., start_pos:end_pos, :].copy_(src_k[..., start_pos:end_pos, :].to(device=dst_k.device, dtype=dst_k.dtype))
        dst_v[..., start_pos:end_pos, :].copy_(src_v[..., start_pos:end_pos, :].to(device=dst_v.device, dtype=dst_v.dtype))


def snapshot_kv_subset(cache, indices: list[int]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    out = {}
    for li, layer in enumerate(cache.layers):
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        if k is not None and v is not None and k.numel() > 0:
            out[li] = (k[..., indices, :].detach().clone(), v[..., indices, :].detach().clone())
    return out


def assert_cache_unchanged(cache, kv_snap, state_snap, indices: list[int], label: str):
    for (li, attr, k_idx), ref in state_snap.items():
        container = getattr(cache.layers[li], attr)
        cur = container[k_idx] if k_idx is not None else container
        if not torch.equal(cur, ref):
            maxd = float((cur.float() - ref.float()).abs().max().item())
            raise AssertionError(f"{label}: state ({li},{attr},{k_idx}) mutated in-place (max_diff={maxd})")
    for li, (ref_k, ref_v) in kv_snap.items():
        cur_k = getattr(cache.layers[li], "keys")[..., indices, :]
        cur_v = getattr(cache.layers[li], "values")[..., indices, :]
        if not (torch.equal(cur_k, ref_k) and torch.equal(cur_v, ref_v)):
            raise AssertionError(f"{label}: KV layer {li} mutated in-place")


def cache_fingerprint(cache) -> str:
    h = hashlib.sha256()
    for layer in cache.layers:
        for attr in ("recurrent_states", "conv_states"):
            for _, val in _iter_layer_state_items(layer, attr):
                arr = val.detach().float().cpu().numpy().tobytes()
                h.update(arr[:4096])
                if len(arr) > 4096:
                    h.update(arr[-4096:])
        for attr in ("keys", "values"):
            val = getattr(layer, attr, None)
            if val is not None and val.numel() > 0:
                arr = val.detach().float().cpu().numpy().tobytes()
                h.update(arr[:4096])
                if len(arr) > 4096:
                    h.update(arr[-4096:])
    return h.hexdigest()[:16]


def resolve_horizon_offset(horizon: str, n_linear: int, w_conv: int) -> int:
    if horizon == "deep":
        return max(1, n_linear - 1) * max(1, w_conv - 1)  # 3 * 47 = 141 for 27B
    if horizon == "stage":
        return (w_conv - 1) * 3  # +9
    if horizon == "single":
        return w_conv - 1  # +3
    raise ValueError(f"Unknown block horizon: {horizon}")


def run_model_segment(
    model,
    cache,
    input_ids: torch.Tensor,
    start_pos: int,
    attention_mask: torch.Tensor | None = None,
    logits_to_keep: int | None = None,
):
    base = get_base_model(model)
    bs, seg_len = input_ids.shape
    pos_ids = make_position_ids(start_pos, seg_len, input_ids.device, base, batch_size=bs)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "past_key_values": cache,
        "position_ids": pos_ids,
        "use_cache": True,
        "return_dict": True,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = logits_to_keep
    return model(**kwargs)


class GDNRateCollector:
    """Forward hooks on Qwen3_5GatedDeltaNet in_proj_a and in_proj_b to record fp32 g_t and beta_t."""

    def __init__(self, model):
        self.text_model = get_text_model(model)
        self.handles = []
        self.g_by_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.beta_by_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.active = False

        for li, layer in enumerate(self.text_model.layers):
            attn = getattr(layer, "linear_attn", None)
            if attn is None:
                continue
            self.handles.append(attn.in_proj_a.register_forward_hook(self._make_a_hook(li, attn)))
            self.handles.append(attn.in_proj_b.register_forward_hook(self._make_b_hook(li)))

    def _make_a_hook(self, layer_idx: int, mod):
        def _hook(_mod, _inp, out):
            if not self.active:
                return
            with torch.no_grad():
                a = out.detach().to(torch.float64)
                dt = mod.dt_bias.detach().to(torch.float64)
                a_log = mod.A_log.detach().to(torch.float64)
                g = -torch.exp(a_log) * F.softplus(a + dt)
                self.g_by_layer[layer_idx].append(g.to(torch.float32))
        return _hook

    def _make_b_hook(self, layer_idx: int):
        def _hook(_mod, _inp, out):
            if not self.active:
                return
            with torch.no_grad():
                beta = torch.sigmoid(out.detach().to(torch.float64))
                self.beta_by_layer[layer_idx].append(beta.to(torch.float32))
        return _hook

    def clear(self):
        self.g_by_layer.clear()
        self.beta_by_layer.clear()

    def collect_concat(self) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        g_cat = {li: torch.cat(chunks, dim=1) for li, chunks in self.g_by_layer.items() if chunks}
        b_cat = {li: torch.cat(chunks, dim=1) for li, chunks in self.beta_by_layer.items() if chunks}
        return g_cat, b_cat

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.no_grad()
def run_context_with_rates(
    model,
    collector: GDNRateCollector,
    ctx_ids: torch.Tensor,
    chunk_size: int = 256,
) -> tuple[Any, dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, Any]]:
    """Chunked prefill over ctx_ids (0..T_ctx) storing B-ckpt checkpoints and fp32 g_t, beta_t."""
    T_ctx = int(ctx_ids.shape[1])
    ckpts: dict[int, Any] = {}
    cache = None
    collector.clear()
    collector.active = True
    try:
        for c0 in range(0, T_ctx, chunk_size):
            if c0 > 0:
                ckpts[c0] = clone_cache(cache)
            c1 = min(T_ctx, c0 + chunk_size)
            out = run_model_segment(model, cache, ctx_ids[:, c0:c1], c0)
            cache = out.past_key_values
        g_all, beta_all = collector.collect_concat()
    finally:
        collector.active = False
        collector.clear()
    return cache, g_all, beta_all, ckpts


@torch.no_grad()
def prepare_bckpt_cache(
    model,
    ckpts: dict[int, Any],
    ctx_ids: torch.Tensor,
    p0: int,
    chunk_size: int = 256,
):
    """Restore nearest chunk checkpoint c <= p0 and advance [c, p0) to obtain exact S_{p0-1}."""
    c = (p0 // chunk_size) * chunk_size
    cache = clone_cache(ckpts[c]) if c > 0 and c in ckpts else None
    if c > 0 and c not in ckpts:
        c = 0
        cache = None
    if p0 > c:
        out = run_model_segment(model, cache, ctx_ids[:, c:p0], c)
        cache = out.past_key_values
    return cache, c


@torch.no_grad()
def build_split_oracle_context_cache(
    model,
    ids_e_ctx: torch.Tensor,
    p0: int,
    block_end: int,
    T_ctx: int,
    chunk_size: int = 256,
):
    """Split oracle matching B-ckpt + J-full segment boundaries."""
    ckpt_c = (p0 // chunk_size) * chunk_size
    c_or = None
    for c0 in range(0, ckpt_c, chunk_size):
        c1 = min(ckpt_c, c0 + chunk_size)
        c_or = run_model_segment(model, c_or, ids_e_ctx[:, c0:c1], c0).past_key_values
    if p0 > ckpt_c:
        c_or = run_model_segment(model, c_or, ids_e_ctx[:, ckpt_c:p0], ckpt_c).past_key_values
    c_or = run_model_segment(model, c_or, ids_e_ctx[:, p0:block_end], p0).past_key_values
    if block_end < T_ctx:
        c_or = run_model_segment(model, c_or, ids_e_ctx[:, block_end:T_ctx], block_end).past_key_values
    return c_or


@torch.no_grad()
def apply_jlocal_update(
    cache,
    delta: dict[tuple[int, str, Any], torch.Tensor],
    g_rates: dict[int, torch.Tensor],
    beta_rates: dict[int, torch.Tensor],
    block_end: int,
    T_ctx: int,
    w_conv: int,
    d_k: int,
    transport_mode: str = "scalar_decay",  # "scalar_decay" (default) | "iso_decay" | "stale"
):
    """
    Decision B: compute transport scalar & injection in fp64 (double), then cast back to target cache dtype.
    Also enforces tail conv_states overlap rule across shift = T_ctx - block_end.
    """
    shift = T_ctx - block_end
    if shift < 0:
        raise ValueError(f"block_end={block_end} > T_ctx={T_ctx}")

    b_suf_means = []
    c_hat_means = []

    for (li, attr, k_idx), d_val in delta.items():
        container = getattr(cache.layers[li], attr, None)
        if container is None:
            continue
        target = container.get(k_idx) if k_idx is not None else container
        if target is None:
            continue
        d_dev = d_val.detach().to(device=target.device, dtype=torch.float64)
        if attr == "conv_states":
            if shift == 0:
                target.copy_((target.double() + d_dev).to(dtype=target.dtype))
            elif shift < w_conv:
                keep = w_conv - shift
                if keep > 0:
                    updated = target[..., :keep].double() + d_dev[..., :keep]
                    target[..., :keep].copy_(updated.to(dtype=target.dtype))
            continue

        assert target.shape[2] == d_k, f"Unexpected d_k {target.shape[2]} != {d_k}"
        if shift == 0 or transport_mode == "stale":
            target.copy_((target.double() + d_dev).to(dtype=target.dtype))
            continue

        g_suf = g_rates[li][:, block_end:T_ctx, :].to(device=target.device, dtype=torch.float64)
        b_suf = beta_rates[li][:, block_end:T_ctx, :].to(device=target.device, dtype=torch.float64)
        g_diff = g_suf.sum(dim=1)
        b_sum = b_suf.sum(dim=1)
        b_suf_means.append(float(b_sum.mean().item()))
        ratio = torch.clamp(b_suf / float(d_k), max=0.999999)
        l_diff = torch.log1p(-ratio).sum(dim=1)
        c_hat_means.append(float((1.0 - torch.exp(l_diff)).mean().item()))

        if transport_mode == "scalar_decay":
            gamma = torch.exp(g_diff)
        elif transport_mode == "iso_decay":
            gamma = torch.exp(g_diff - b_sum / float(d_k))
        else:
            raise ValueError(f"Unknown transport_mode={transport_mode}")

        H_v = target.shape[1]
        assert gamma.shape == (1, H_v), f"Expected gamma shape (1, {H_v}), got {gamma.shape}"
        scale = gamma.view(1, H_v, 1, 1)
        target.copy_((target.double() + d_dev * scale).to(dtype=target.dtype))

    return {
        "b_suf_mean": float(statistics.mean(b_suf_means)) if b_suf_means else 0.0,
        "c_hat_mean": float(statistics.mean(c_hat_means)) if c_hat_means else 0.0,
    }


def kl_div_from_logits(logits_ref: torch.Tensor, logits_cand: torch.Tensor) -> float:
    p = F.softmax(logits_ref.float(), dim=-1)
    log_p = F.log_softmax(logits_ref.float(), dim=-1)
    log_q = F.log_softmax(logits_cand.float().to(logits_ref.device), dim=-1)
    kl = torch.sum(p * (log_p - log_q), dim=-1).mean().item()
    return max(0.0, float(kl))


def max_abs_logit_diff(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    return float((logits_a.float() - logits_b.float().to(logits_a.device)).abs().max().item())


def normalized_edit_distance(pred: str, target: str) -> float:
    if pred == target:
        return 0.0
    m, n = len(pred), len(target)
    if max(m, n) == 0:
        return 0.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if pred[i - 1] == target[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return float(dp[n]) / float(max(m, n))


@torch.no_grad()
def manual_greedy_from_cache(
    model,
    cache,
    last_tok: torch.Tensor,
    start_pos: int,
    max_new: int,
    eos_ids: set[int],
    attn_mask_base: torch.Tensor | None = None,
    return_trace: bool = False,
):
    """Greedy decode from cloned cache using explicit position_ids."""
    c = clone_cache(cache)
    gen_ids: list[int] = []
    trace_logits: list[torch.Tensor] = []
    cur_tok = last_tok
    cur_pos = start_pos
    first_logits = None
    cur_mask = attn_mask_base

    for step in range(max_new):
        if cur_mask is not None:
            ones = torch.ones((1, 1), dtype=cur_mask.dtype, device=cur_mask.device)
            cur_mask = torch.cat([cur_mask, ones], dim=-1)
        out = run_model_segment(model, c, cur_tok, cur_pos, attention_mask=cur_mask)
        logits = out.logits[:, -1, :]
        lg_det = logits.detach().clone()
        trace_logits.append(lg_det)
        if step == 0:
            first_logits = lg_det
        nxt = int(torch.argmax(logits, dim=-1).item())
        gen_ids.append(nxt)
        if nxt in eos_ids:
            break
        cur_tok = torch.tensor([[nxt]], dtype=torch.long, device=last_tok.device)
        cur_pos += 1

    del c
    if return_trace:
        return first_logits, gen_ids, trace_logits
    return first_logits, gen_ids


@torch.no_grad()
def compute_answer_seq_metrics(
    model,
    cache_at_tpref_minus_1,
    last_tok: torch.Tensor,
    T_pref_minus_1: int,
    ans_ids: list[int],
    oracle_ans_logits: list[torch.Tensor] | None = None,
    attn_mask_base: torch.Tensor | None = None,
    precomputed_logits: list[torch.Tensor] | None = None,
) -> tuple[float, list[torch.Tensor], float, float, float]:
    """Teacher-forced sequence log-prob of ans_ids and per-token KL vs oracle_ans_logits."""
    if precomputed_logits is not None and len(precomputed_logits) >= len(ans_ids):
        step_logits = [lg.detach() for lg in precomputed_logits[: len(ans_ids)]]
        total_lp = sum(float(F.log_softmax(lg.float(), dim=-1)[0, tok_id].item()) for lg, tok_id in zip(step_logits, ans_ids))
        if oracle_ans_logits is None:
            return total_lp, step_logits, 0.0, 0.0, 0.0
        kls = [kl_div_from_logits(ref_lg, lg) for ref_lg, lg in zip(oracle_ans_logits, step_logits)]
        return total_lp, step_logits, float(statistics.mean(kls)), float(kls[0]), float(max(kls))

    c = clone_cache(cache_at_tpref_minus_1)
    cur_tok = last_tok
    cur_pos = T_pref_minus_1
    cur_mask = attn_mask_base
    step_logits = []
    total_lp = 0.0

    for tok_id in ans_ids:
        if cur_mask is not None:
            ones = torch.ones((1, 1), dtype=cur_mask.dtype, device=cur_mask.device)
            cur_mask = torch.cat([cur_mask, ones], dim=-1)
        out = run_model_segment(model, c, cur_tok, cur_pos, attention_mask=cur_mask)
        lg = out.logits[:, -1, :]
        step_logits.append(lg.detach())
        lp = F.log_softmax(lg.float(), dim=-1)[0, tok_id].item()
        total_lp += float(lp)
        cur_tok = torch.tensor([[tok_id]], dtype=torch.long, device=last_tok.device)
        cur_pos += 1

    del c
    if oracle_ans_logits is None:
        return total_lp, step_logits, 0.0, 0.0, 0.0
    kls = [kl_div_from_logits(ref_lg, lg) for ref_lg, lg in zip(oracle_ans_logits, step_logits)]
    return total_lp, step_logits, float(statistics.mean(kls)), float(kls[0]), float(max(kls))


@torch.no_grad()
def eval_suffix_from_ctx_cache(
    model,
    tok,
    cache_ctx,
    ids_q: torch.Tensor,
    T_ctx: int,
    T_pref_minus_1: int,
    last_tok: torch.Tensor,
    max_new: int,
    eos_ids: set[int],
    old_val: str,
    new_val: str,
    old_ids: list[int],
    new_ids: list[int],
    regime: str,
    attn_mask_ctx: torch.Tensor | None = None,
    oracle_first_logits: torch.Tensor | None = None,
    oracle_new_ans_logits: list[torch.Tensor] | None = None,
    first_logits_only: bool = False,
) -> dict[str, Any]:
    """Clone cache_ctx, forward query suffix [T_ctx, T_pref_minus_1), then decode & score."""
    c = clone_cache(cache_ctx)
    suf_len = T_pref_minus_1 - T_ctx
    suf_mask = None
    if attn_mask_ctx is not None:
        suf_ones = torch.ones((1, suf_len), dtype=attn_mask_ctx.dtype, device=attn_mask_ctx.device)
        suf_mask = torch.cat([attn_mask_ctx, suf_ones], dim=-1)

    if suf_len > 0:
        out_suf = run_model_segment(model, c, ids_q[:, T_ctx:T_pref_minus_1], T_ctx, attention_mask=suf_mask)
        c = out_suf.past_key_values

    if first_logits_only:
        cur_mask = suf_mask
        if cur_mask is not None:
            ones = torch.ones((1, 1), dtype=cur_mask.dtype, device=cur_mask.device)
            cur_mask = torch.cat([cur_mask, ones], dim=-1)
        out_1st = run_model_segment(model, c, last_tok, T_pref_minus_1, attention_mask=cur_mask)
        first_logits = out_1st.logits[:, -1, :].detach().clone()
        del c
        return {"first_logits": first_logits}

    first_logits, gen_ids, trace_logits = manual_greedy_from_cache(
        model, c, last_tok, T_pref_minus_1, max_new=max_new, eos_ids=eos_ids, attn_mask_base=suf_mask, return_trace=True
    )

    gen_no_eos = [t for t in gen_ids if t not in eos_ids]
    gen_text = tok.decode(gen_no_eos, skip_special_tokens=True).strip()
    top1_id = gen_ids[0] if gen_ids else -1

    probs = F.softmax(first_logits.float(), dim=-1)
    p_old_1st = float(probs[0, old_ids[0]].item())
    p_new_1st = float(probs[0, new_ids[0]].item())

    pre_new = trace_logits[: len(new_ids)] if (len(new_ids) == 1 or (len(gen_ids) >= len(new_ids) and gen_ids[: len(new_ids)] == new_ids)) else None
    pre_old = trace_logits[: len(old_ids)] if (len(old_ids) == 1 or (len(gen_ids) >= len(old_ids) and gen_ids[: len(old_ids)] == old_ids)) else None
    lp_new, new_step_logits, mean_ans_kl, _, max_ans_kl = compute_answer_seq_metrics(
        model, c, last_tok, T_pref_minus_1, new_ids, oracle_ans_logits=oracle_new_ans_logits, attn_mask_base=suf_mask, precomputed_logits=pre_new
    )
    lp_old, _, _, _, _ = compute_answer_seq_metrics(
        model, c, last_tok, T_pref_minus_1, old_ids, oracle_ans_logits=None, attn_mask_base=suf_mask, precomputed_logits=pre_old
    )

    del c

    exact_new = bool(gen_text == new_val)
    old_leak = bool(gen_text == old_val)
    kl_vs_oracle_first = kl_div_from_logits(oracle_first_logits, first_logits) if oracle_first_logits is not None else 0.0
    max_diff_vs_oracle = max_abs_logit_diff(oracle_first_logits, first_logits) if oracle_first_logits is not None else 0.0

    return {
        "first_logits": first_logits,
        "new_step_logits": new_step_logits,
        "gen_ids": gen_ids,
        "gen_text": gen_text,
        "top1_id": top1_id,
        "first_tok_new_hit": bool(top1_id == new_ids[0]),
        "exact_match": exact_new,
        "old_leak": old_leak,
        "p_new_1st": p_new_1st,
        "p_old_1st": p_old_1st,
        "seq_logprob_new": lp_new,
        "seq_logprob_old": lp_old,
        "p_new_seq": float(math.exp(max(-50.0, lp_new))),
        "p_old_seq": float(math.exp(max(-50.0, lp_old))),
        "norm_edit_dist": normalized_edit_distance(gen_text, new_val),
        "kl_first_vs_oracle": kl_vs_oracle_first,
        "drift_mean_ans_kl": mean_ans_kl if oracle_new_ans_logits is not None else kl_vs_oracle_first,
        "drift_max_ans_kl": max_ans_kl if oracle_new_ans_logits is not None else kl_vs_oracle_first,
        "max_abs_logit_diff_vs_oracle": max_diff_vs_oracle,
        "regime": regime,
    }


def strip_tensors(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not isinstance(v, torch.Tensor) and k != "new_step_logits"}


def assert_oracle_first_token(
    tok,
    lg_split: torch.Tensor,
    split_gen_ids: list[int],
    lg_one: torch.Tensor,
    one_gen_ids: list[int],
    old_val: str,
    new_val: str,
    label: str = "",
):
    ans_old_id = tok.encode(old_val, add_special_tokens=False)[0]
    ans_new_id = tok.encode(new_val, add_special_tokens=False)[0]
    assert ans_old_id != ans_new_id, f"[{label}] First-token collision {old_val} vs {new_val}"

    for tag, lg, gids in [("split_oracle", lg_split, split_gen_ids), ("oneshot_oracle", lg_one, one_gen_ids)]:
        top1_id = int(torch.argmax(lg, dim=-1).item())
        top1_txt = tok.decode([top1_id])
        gen_txt = tok.decode(gids, skip_special_tokens=True)
        probs = F.softmax(lg.float(), dim=-1)
        p_new = float(probs[0, ans_new_id].item())
        p_old = float(probs[0, ans_old_id].item())

        if top1_txt.lstrip().startswith("`") or "```" in gen_txt:
            raise AssertionError(f"[{label}] {tag} emitted markdown/code block: {gen_txt!r}")
        if new_val.isdigit() and not top1_txt.strip().isdigit():
            raise AssertionError(f"[{label}] {tag} non-digit first token {top1_txt!r} for {new_val!r}")
        if top1_id != ans_new_id:
            raise AssertionError(
                f"[{label}] {tag} first-token mismatch: got {top1_id}({top1_txt!r}) expected {ans_new_id}({new_val!r}), gen={gen_txt!r}"
            )
        if not (p_new > p_old):
            raise AssertionError(f"[{label}] {tag} p_new ({p_new:.4f}) <= p_old ({p_old:.4f})")


def verify_pure_text_mrope_equivalence(model, device) -> dict[str, Any]:
    base = get_base_model(model)
    text_m = get_text_model(model)
    rotary = getattr(text_m, "rotary_emb", None)
    if rotary is None:
        return {"supported": False, "cos_max_diff": 0.0, "sin_max_diff": 0.0}
    L = 32
    pos_full = make_position_ids(0, L, device, base, batch_size=1)
    pos_mrope = pos_full[1:] if pos_full.shape[0] == 4 else pos_full
    pos_1 = torch.arange(0, L, device=device, dtype=torch.long).unsqueeze(0)
    t_cfg = get_text_cfg(model.config)
    hidden_sz = int(getattr(t_cfg, "hidden_size", 2048))
    emb = getattr(text_m, "embed_tokens", None)
    if emb is not None and hasattr(emb, "weight") and emb.weight.is_floating_point():
        dummy_dtype = emb.weight.dtype
    else:
        dummy_dtype = torch.bfloat16
    dummy = torch.zeros((1, L, hidden_sz), device=device, dtype=dummy_dtype)
    with torch.no_grad():
        c3, s3 = rotary(dummy, pos_mrope)
        c1, s1 = rotary(dummy, pos_1.view(1, 1, -1))
    cos_diff = float((c3.float() - c1.float()).abs().max().item())
    sin_diff = float((s3.float() - s1.float()).abs().max().item())
    assert cos_diff == 0.0 and sin_diff == 0.0, f"M-RoPE mismatch: cos={cos_diff}, sin={sin_diff}"
    return {"supported": True, "cos_max_diff": cos_diff, "sin_max_diff": sin_diff, "pos_rows": int(pos_full.shape[0])}


def locate_exact_char_span_tokens(tok, full_text: str, char_start: int, char_end: int) -> tuple[list[int], list[int]]:
    enc = tok(full_text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"][0].tolist()
    offsets = enc["offset_mapping"][0].tolist()
    pos = [i for i, (a, b) in enumerate(offsets) if a < char_end and b > char_start and b > a]
    if not pos:
        raise ValueError(f"Empty token span for [{char_start}:{char_end}] = {full_text[char_start:char_end]!r}")
    return ids, pos


def check_e0c_alignment(
    tok,
    text_o: str,
    cs_o: int,
    ce_o: int,
    text_e: str,
    cs_e: int,
    ce_e: int,
    old_val: str,
    new_val: str,
    regime: str,
) -> tuple[list[int], list[int], list[int], list[int]]:
    ids_o, pos_o = locate_exact_char_span_tokens(tok, text_o, cs_o, ce_o)
    ids_e, pos_e = locate_exact_char_span_tokens(tok, text_e, cs_e, ce_e)
    assert len(pos_o) == len(pos_e), f"len(pos_o)={len(pos_o)} != len(pos_e)={len(pos_e)} for {old_val}->{new_val}"
    assert pos_o == pos_e, f"pos_o={pos_o} != pos_e={pos_e}"
    assert len(ids_o) == len(ids_e), f"len(ids_o)={len(ids_o)} != len(ids_e)={len(ids_e)}"
    assert ids_o[: pos_o[0]] == ids_e[: pos_e[0]], "Prefix token mismatch!"
    assert ids_o[pos_o[-1] + 1 :] == ids_e[pos_e[-1] + 1 :], "Suffix token mismatch!"

    enc_o = tok.encode(old_val, add_special_tokens=False)
    enc_e = tok.encode(new_val, add_special_tokens=False)
    assert enc_o[0] != enc_e[0], f"First-token collision: {old_val} ({enc_o[0]}) == {new_val} ({enc_e[0]})"
    if regime in ("single_letter", "single_digit"):
        assert len(pos_o) == 1 and len(enc_o) == 1 and len(enc_e) == 1, (
            f"Single-token violation: {old_val}->{new_val}, pos={pos_o}, enc_o={enc_o}, enc_e={enc_e}"
        )
    elif regime == "digit4":
        assert len(pos_o) == 4 and len(enc_o) == 4 and len(enc_e) == 4, (
            f"digit4 violation: {old_val}->{new_val}, pos={pos_o}, enc_o={enc_o}, enc_e={enc_e}"
        )
    return ids_o, pos_o, enc_o, enc_e


def build_prompt_with_spans(
    tok,
    tpl_id: int,
    facts: list[tuple[str, str]],
    target_entity: str,
    pair_format: str,
    pre_idx: int,
    post_idx: int,
    fact_overrides: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Build user prompt and rendered chat text tracking exact char/token spans for facts, T_ctx, and stone."""
    parts: list[str] = []
    cur_len = 0
    pre_str = PRE_FILLER_TEMPLATES[pre_idx % len(PRE_FILLER_TEMPLATES)].format(idx=(pre_idx % 97) + 1)
    parts.append(pre_str + " ")
    cur_len += len(parts[-1])

    fact_char_spans: list[tuple[int, int]] = []
    for fi, (name, code) in enumerate(facts):
        val = fact_overrides[fi] if (fact_overrides and fi in fact_overrides) else code
        sent, vs, ve = format_fact_parts(tpl_id, name, val)
        fact_char_spans.append((cur_len + vs, cur_len + ve))
        parts.append(sent + " ")
        cur_len += len(parts[-1])

    post_str = POST_FILLER_BLOCKS[post_idx % len(POST_FILLER_BLOCKS)]
    stone_rel = post_str.index(PLACEBO_OLD)
    stone_char_span = (cur_len + stone_rel, cur_len + stone_rel + len(PLACEBO_OLD))
    parts.append(post_str)
    cur_len += len(parts[-1])

    fact_context = "".join(parts)
    query_str = format_query_suffix(tpl_id, target_entity, pair_format)
    full_user = fact_context + query_str
    rendered, chat_meta = render_chat(tok, full_user)
    ctx_offset = rendered.index(fact_context)
    q_char_start = rendered.index(query_str, ctx_offset + len(fact_context) - 1)

    enc = tok(rendered, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()
    q_tokens = [idx for idx, (a, b) in enumerate(offsets) if a >= q_char_start and b > a]
    T_ctx = min(q_tokens)

    fact_tok_spans = []
    for cs, ce in fact_char_spans:
        acs, ace = ctx_offset + cs, ctx_offset + ce
        toks = [idx for idx, (a, b) in enumerate(offsets) if a < ace and b > acs and b > a]
        fact_tok_spans.append((min(toks), max(toks) + 1))

    stone_toks = [
        idx
        for idx, (a, b) in enumerate(offsets)
        if a < ctx_offset + stone_char_span[1] and b > ctx_offset + stone_char_span[0] and b > a
    ]
    stone_tok_span = (min(stone_toks), max(stone_toks) + 1)

    return {
        "rendered": rendered,
        "fact_context": fact_context,
        "ctx_offset": ctx_offset,
        "fact_char_spans": [(ctx_offset + a, ctx_offset + b) for (a, b) in fact_char_spans],
        "fact_tok_spans": fact_tok_spans,
        "stone_char_span": (ctx_offset + stone_char_span[0], ctx_offset + stone_char_span[1]),
        "stone_tok_span": stone_tok_span,
        "input_ids": enc["input_ids"],
        "T_ctx": T_ctx,
        "T_pref_minus_1": int(enc["input_ids"].shape[1]) - 1,
        "chat_meta": chat_meta,
    }


def build_e2_dataset(tok, per_tpl: int, seed: int, horizon_offset: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Generate 3 x per_tpl distinct E2 samples (default per_tpl=40 -> 120 total), pre-screened via E0c,
    then ordered round-robin across templates [tpl0_0, tpl1_0, tpl2_0, tpl0_1, ...] with sample_id=0..N-1.
    """
    rng = random.Random(seed)
    all_digit4_codes = [f"{n:04d}" for n in range(1000, 10000)]
    by_first_d = defaultdict(list)
    for c in all_digit4_codes:
        by_first_d[c[0]].append(c)
    for k in by_first_d:
        rng.shuffle(by_first_d[k])

    letter_pairs = [(a, b) for a in LETTERS_26 for b in LETTERS_26 if a != b]
    digit_pairs = [(a, b) for a in DIGITS_10 for b in DIGITS_10 if a != b]
    rng.shuffle(letter_pairs)
    rng.shuffle(digit_pairs)

    by_tpl: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    rejections = 0
    checked = 0
    seen_signatures = set()
    d_cursor = {d: 0 for d in "123456789"}
    lp_idx = 0
    dp_idx = 0
    suffix_lens: list[int] = []
    placebo_suffix_lens: list[int] = []

    for tpl_id in range(3):
        n_digit4 = per_tpl // 2
        n_single = per_tpl - n_digit4
        n_letter = n_single // 2
        n_digit1 = n_single - n_letter
        sub_schedule = (["digit4"] * n_digit4) + (["single_letter"] * n_letter) + (["single_digit"] * n_digit1)
        interleaved = []
        buckets = {
            "digit4": [x for x in sub_schedule if x == "digit4"],
            "single_letter": [x for x in sub_schedule if x == "single_letter"],
            "single_digit": [x for x in sub_schedule if x == "single_digit"],
        }
        while any(buckets.values()):
            for bname in ("digit4", "single_letter", "single_digit"):
                if buckets[bname]:
                    interleaved.append(buckets[bname].pop(0))

        for local_i, pair_format in enumerate(interleaved):
            raw_idx = tpl_id * per_tpl + local_i
            ent_start = raw_idx % len(ENTITIES_8)
            entities = [ENTITIES_8[(ent_start + j) % len(ENTITIES_8)] for j in range(4)]
            target_idx = local_i % 4
            pre_idx = raw_idx % len(PRE_FILLER_TEMPLATES)
            post_idx = (raw_idx + tpl_id) % len(POST_FILLER_BLOCKS)

            while True:
                if pair_format == "digit4":
                    first_digits = rng.sample(list("123456789"), 5)
                    vals = []
                    for fd in first_digits:
                        pool = by_first_d[fd]
                        code = pool[d_cursor[fd] % len(pool)]
                        d_cursor[fd] += 1
                        vals.append(code)
                    old_val, new_val = vals[0], vals[1]
                    dist_vals = vals[2:]
                elif pair_format == "single_letter":
                    old_val, new_val = letter_pairs[lp_idx % len(letter_pairs)]
                    lp_idx += 1
                    rem = [x for x in LETTERS_26 if x not in (old_val, new_val)]
                    dist_vals = rng.sample(rem, 3)
                else:
                    old_val, new_val = digit_pairs[dp_idx % len(digit_pairs)]
                    dp_idx += 1
                    rem = [x for x in DIGITS_10 if x not in (old_val, new_val)]
                    dist_vals = rng.sample(rem, 3)

                sig = (tpl_id, pair_format, entities[target_idx], old_val, new_val, tuple(dist_vals), target_idx, post_idx)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    break

            facts = []
            d_pos = 0
            for fi, name in enumerate(entities):
                if fi == target_idx:
                    facts.append((name, old_val))
                else:
                    facts.append((name, dist_vals[d_pos]))
                    d_pos += 1

            info_o = build_prompt_with_spans(tok, tpl_id, facts, entities[target_idx], pair_format, pre_idx, post_idx)
            info_e = build_prompt_with_spans(
                tok, tpl_id, facts, entities[target_idx], pair_format, pre_idx, post_idx, fact_overrides={target_idx: new_val}
            )
            checked += 1
            try:
                cs_o, ce_o = info_o["fact_char_spans"][target_idx]
                cs_e, ce_e = info_e["fact_char_spans"][target_idx]
                _, pos_o, enc_o, enc_e = check_e0c_alignment(
                    tok, info_o["rendered"], cs_o, ce_o, info_e["rendered"], cs_e, ce_e, old_val, new_val, pair_format
                )
                u_text_e = (
                    info_o["rendered"][: info_o["stone_char_span"][0]]
                    + PLACEBO_NEW
                    + info_o["rendered"][info_o["stone_char_span"][1] :]
                )
                u_cs, u_ce = info_o["stone_char_span"][0], info_o["stone_char_span"][0] + len(PLACEBO_NEW)
                _, pos_u, _, _ = check_e0c_alignment(
                    tok,
                    info_o["rendered"],
                    info_o["stone_char_span"][0],
                    info_o["stone_char_span"][1],
                    u_text_e,
                    u_cs,
                    u_ce,
                    PLACEBO_OLD,
                    PLACEBO_NEW,
                    "placebo",
                )
                assert info_o["T_ctx"] == info_e["T_ctx"]
                T_ctx = info_o["T_ctx"]
                p0, p1 = pos_o[0], pos_o[-1] + 1
                u0, u1 = pos_u[0], pos_u[-1] + 1
                assert u0 > p1, f"Placebo span {u0}..{u1} must follow target {p0}..{p1}"
                b_end = min(T_ctx, p1 + horizon_offset)
                b_unrel = min(T_ctx, u1 + horizon_offset)
                assert b_end < T_ctx - 20, f"Suffix too short: b_end={b_end}, T_ctx={T_ctx}"
                assert b_unrel < T_ctx - 20, f"Placebo suffix too short: b_unrel={b_unrel}, T_ctx={T_ctx}"
            except AssertionError as exc:
                rejections += 1
                raise RuntimeError(f"E0c screening failed on tpl={tpl_id} local_i={local_i}: {exc}") from exc

            suffix_lens.append(T_ctx - b_end)
            placebo_suffix_lens.append(T_ctx - b_unrel)
            by_tpl[tpl_id].append(
                {
                    "task_group": "e2",
                    "sample_id": -1,
                    "tpl_id": tpl_id,
                    "tpl_local_idx": local_i,
                    "regime": "digit4" if pair_format == "digit4" else "single_token",
                    "pair_format": pair_format,
                    "entities": entities,
                    "target_idx": target_idx,
                    "target_entity": entities[target_idx],
                    "facts": facts,
                    "old_val": old_val,
                    "new_val": new_val,
                    "pre_idx": pre_idx,
                    "post_idx": post_idx,
                    "T_ctx": T_ctx,
                    "T_pref_minus_1": info_o["T_pref_minus_1"],
                    "p0": p0,
                    "p1": p1,
                    "b_end": b_end,
                    "u0": u0,
                    "u1": u1,
                    "b_unrel": b_unrel,
                    "rendered_o": info_o["rendered"],
                    "rendered_e": info_e["rendered"],
                    "rendered_placebo": u_text_e,
                    "ids_o": info_o["input_ids"],
                    "ids_e": info_e["input_ids"],
                    "ids_placebo": tok(u_text_e, return_tensors="pt", add_special_tokens=False)["input_ids"],
                    "old_ids": enc_o,
                    "new_ids": enc_e,
                    "chat_meta": info_o["chat_meta"],
                }
            )

    # Round-robin across templates [tpl0_0, tpl1_0, tpl2_0, tpl0_1, ...] with sequential sample_id = 0..N-1
    samples: list[dict[str, Any]] = []
    for li in range(per_tpl):
        for tid in range(3):
            item = by_tpl[tid][li]
            item["sample_id"] = len(samples)
            samples.append(item)

    inv_summary = {
        "n_e2": len(samples),
        "checked": checked,
        "rejections": rejections,
        "rejection_rate": rejections / max(1, checked),
        "min_suffix_len": min(suffix_lens) if suffix_lens else 0,
        "min_placebo_suffix_len": min(placebo_suffix_lens) if placebo_suffix_lens else 0,
    }
    return samples, inv_summary


def build_e3_dataset(
    tok,
    seed: int,
    horizon_offset: int,
    per_cell_limit: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build E9 stratified E3 multi-fact cells (total n=80):
      - e3_m4_k2: 24 samples
      - e3_m4_k4: 18 samples
      - e3_m8_k2: 18 samples
      - e3_m8_k4: 20 samples
    Builds all 80 deterministically first, then slices per-cell [:per_cell_limit] if > 0
    so probe prefixes are bitwise identical to full-run prefixes.
    """
    rng = random.Random(seed + 777)
    all_e3: list[dict[str, Any]] = []
    rejections = 0
    suffix_lens: list[int] = []
    cell_counts_full: dict[str, int] = {}

    for task_group, M, k, n_samples in E9_E3_CELL_SPECS:
        perms = E3_PERMS_K2 if k == 2 else E3_PERMS_K4
        cell_samples: list[dict[str, Any]] = []
        for s_idx in range(n_samples):
            tpl_id = s_idx % 3
            pair_format = "digit4" if (s_idx % 2 == 0) else "single_letter"
            regime = "digit4" if pair_format == "digit4" else "single_token"

            entities = ENTITIES_8[:M]
            all_pos = list(range(M))
            rng.shuffle(all_pos)
            edit_indices = sorted(all_pos[:k])
            collat_indices = [idx for idx in range(M) if idx not in edit_indices]

            if pair_format == "single_letter":
                chosen = rng.sample(LETTERS_26, M + k)
                old_vals = chosen[:M]
                new_vals = chosen[M:]
            else:
                init_first = rng.sample(list("123456789"), M)
                used_codes = set()
                old_vals = []
                for fd in init_first:
                    while True:
                        c = fd + f"{rng.randint(0, 999):03d}"
                        if c not in used_codes:
                            used_codes.add(c)
                            old_vals.append(c)
                            break
                new_first_pool = rng.sample(list("123456789"), k)
                for _ in range(50):
                    if all(new_first_pool[j] != old_vals[edit_indices[j]][0] for j in range(k)):
                        break
                    rng.shuffle(new_first_pool)
                else:
                    for j in range(k):
                        if new_first_pool[j] == old_vals[edit_indices[j]][0]:
                            cand = [d for d in "123456789" if d != old_vals[edit_indices[j]][0] and d not in new_first_pool]
                            new_first_pool[j] = cand[0] if cand else [d for d in "123456789" if d != old_vals[edit_indices[j]][0]][0]
                new_vals = []
                for fd in new_first_pool:
                    while True:
                        c = fd + f"{rng.randint(0, 999):03d}"
                        if c not in used_codes:
                            used_codes.add(c)
                            new_vals.append(c)
                            break

            facts = [(entities[i], old_vals[i]) for i in range(M)]
            new_map = {edit_indices[j]: new_vals[j] for j in range(k)}
            pre_idx = s_idx % len(PRE_FILLER_TEMPLATES)
            post_idx = (s_idx + tpl_id) % len(POST_FILLER_BLOCKS)

            queries = {}
            base_info = build_prompt_with_spans(tok, tpl_id, facts, entities[0], pair_format, pre_idx, post_idx)
            T_ctx = base_info["T_ctx"]
            T_pref_minus_1 = base_info["T_pref_minus_1"]

            edit_slots = []
            try:
                for j, ei in enumerate(edit_indices):
                    info_old_slot = build_prompt_with_spans(
                        tok, tpl_id, facts, entities[ei], pair_format, pre_idx, post_idx
                    )
                    info_single = build_prompt_with_spans(
                        tok, tpl_id, facts, entities[ei], pair_format, pre_idx, post_idx, fact_overrides={ei: new_vals[j]}
                    )
                    cs_o, ce_o = info_old_slot["fact_char_spans"][ei]
                    cs_e, ce_e = info_single["fact_char_spans"][ei]
                    _, pos_o, enc_o, enc_e = check_e0c_alignment(
                        tok, info_old_slot["rendered"], cs_o, ce_o, info_single["rendered"], cs_e, ce_e, old_vals[ei], new_vals[j], pair_format
                    )
                    p0, p1 = pos_o[0], pos_o[-1] + 1
                    b_end = min(T_ctx, p1 + horizon_offset)
                    assert b_end < T_ctx - 20, f"E3 suffix too short for slot {ei}: b_end={b_end}, T_ctx={T_ctx}"
                    suffix_lens.append(T_ctx - b_end)
                    edit_slots.append(
                        {
                            "slot_idx": j,
                            "fact_idx": ei,
                            "entity": entities[ei],
                            "old_val": old_vals[ei],
                            "new_val": new_vals[j],
                            "old_ids": enc_o,
                            "new_ids": enc_e,
                            "p0": p0,
                            "p1": p1,
                            "b_end": b_end,
                        }
                    )

                for fi in range(M):
                    q_old = build_prompt_with_spans(tok, tpl_id, facts, entities[fi], pair_format, pre_idx, post_idx)
                    q_all_e = build_prompt_with_spans(
                        tok, tpl_id, facts, entities[fi], pair_format, pre_idx, post_idx, fact_overrides=new_map
                    )
                    assert q_old["T_ctx"] == T_ctx and q_all_e["T_ctx"] == T_ctx
                    queries[fi] = {
                        "entity": entities[fi],
                        "is_edited": fi in new_map,
                        "old_val": old_vals[fi],
                        "target_val": new_map[fi] if fi in new_map else old_vals[fi],
                        "old_ids": tok.encode(old_vals[fi], add_special_tokens=False),
                        "target_ids": tok.encode(new_map[fi] if fi in new_map else old_vals[fi], add_special_tokens=False),
                        "ids_old": q_old["input_ids"],
                        "ids_oracle": q_all_e["input_ids"],
                        "T_pref_minus_1": q_old["T_pref_minus_1"],
                    }
            except AssertionError as exc:
                rejections += 1
                raise RuntimeError(f"E3 E0c screening failed on {task_group} #{s_idx}: {exc}") from exc

            cell_samples.append(
                {
                    "task_group": task_group,
                    "sample_id": s_idx,
                    "M": M,
                    "k": k,
                    "tpl_id": tpl_id,
                    "regime": regime,
                    "pair_format": pair_format,
                    "pre_idx": pre_idx,
                    "post_idx": post_idx,
                    "facts": facts,
                    "edit_indices": edit_indices,
                    "collat_indices": collat_indices,
                    "new_map": new_map,
                    "edit_slots": edit_slots,
                    "perms": perms,
                    "T_ctx": T_ctx,
                    "T_pref_minus_1": T_pref_minus_1,
                    "queries": queries,
                    "ids_ctx_old": base_info["input_ids"][:, :T_ctx],
                }
            )

        cell_counts_full[task_group] = len(cell_samples)
        if per_cell_limit > 0:
            all_e3.extend(cell_samples[:per_cell_limit])
        else:
            all_e3.extend(cell_samples)

    inv = {
        "n_e3_full": sum(cell_counts_full.values()),
        "n_e3_selected": len(all_e3),
        "cell_counts_full": cell_counts_full,
        "rejections": rejections,
        "min_suffix_len": min(suffix_lens) if suffix_lens else 0,
    }
    return all_e3, inv


@torch.no_grad()
def evaluate_e2_sample(
    model,
    tok,
    collector: GDNRateCollector,
    device: torch.device,
    total_vram_gb: float,
    s: dict[str, Any],
    w_conv: int,
    d_k: int,
    max_new: int,
    eos_ids: set[int],
    chunk_size: int,
    g1b_thresh: float,
    allow_rounding_exception: bool,
    run_startup_checks: bool = False,
    skip_oracle_precheck: bool = False,
) -> dict[str, Any]:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0_wall = sync_time(device)

    T_ctx = s["T_ctx"]
    T_pref_m1 = s["T_pref_minus_1"]
    p0, p1, b_end = s["p0"], s["p1"], s["b_end"]
    u0, b_unrel = s["u0"], s["b_unrel"]
    old_val, new_val = s["old_val"], s["new_val"]
    old_ids, new_ids = s["old_ids"], s["new_ids"]
    regime = s["regime"]

    ids_o = s["ids_o"].to(device)
    ids_e = s["ids_e"].to(device)
    ids_placebo = s["ids_placebo"].to(device)
    last_tok = ids_o[:, T_pref_m1 : T_pref_m1 + 1]

    # 1. Shared OLD fact_context prefill with B-ckpt + g/beta capture
    c_ctx_old, g_old, beta_old, ckpts_old = run_context_with_rates(model, collector, ids_o[:, :T_ctx], chunk_size=chunk_size)
    old_states_snap = snap_states(c_ctx_old)
    check_kv_idx = sorted(set([0, min(p0, T_ctx - 1), max(0, b_end - 1), T_ctx - 1]))
    old_kv_snap = snapshot_kv_subset(c_ctx_old, check_kv_idx)

    # 2. One-shot oracle (independent forward over full ids_e[:, :T_pref_m1])
    out_one = run_model_segment(model, None, ids_e[:, :T_pref_m1], 0)
    c_one_pref = out_one.past_key_values
    lg_one, gen_one, trace_one = manual_greedy_from_cache(
        model, c_one_pref, last_tok, T_pref_m1, max_new=max_new, eos_ids=eos_ids, return_trace=True
    )
    pre_one = trace_one[: len(new_ids)] if (len(new_ids) == 1 or (len(gen_one) >= len(new_ids) and gen_one[: len(new_ids)] == new_ids)) else None
    lp_one_new, oracle_new_ans_logits, _, _, _ = compute_answer_seq_metrics(
        model, c_one_pref, last_tok, T_pref_m1, new_ids, precomputed_logits=pre_one
    )

    startup_meta = {}
    if run_startup_checks:
        c_clone_chk = clone_cache(c_ctx_old)
        assert c_clone_chk is not c_ctx_old
        for li, layer in enumerate(c_ctx_old.layers):
            if getattr(layer, "keys", None) is not None and layer.keys.numel() > 0:
                assert c_clone_chk.layers[li].keys.data_ptr() != layer.keys.data_ptr(), "clone_cache shared KV ptr"
            for attr in ("recurrent_states", "conv_states"):
                for k_idx, ref_t in _iter_layer_state_items(layer, attr):
                    chk_container = getattr(c_clone_chk.layers[li], attr)
                    chk_t = chk_container[k_idx] if k_idx is not None else chk_container
                    assert chk_t.data_ptr() != ref_t.data_ptr(), f"clone_cache shared {attr} ptr"
        del c_clone_chk

        gen_inp = ids_e[:, : T_pref_m1 + 1]
        gen_out = model.generate(
            input_ids=gen_inp,
            attention_mask=torch.ones_like(gen_inp),
            max_new_tokens=max_new,
            do_sample=False,
            use_cache=True,
            eos_token_id=sorted(eos_ids),
            pad_token_id=tok.eos_token_id,
        )
        hf_gen_ids = gen_out[0, T_pref_m1 + 1 :].tolist()
        assert gen_one == hf_gen_ids, f"manual greedy {gen_one} != model.generate {hf_gen_ids}"

        # Identity zero-drift check via default scalar_decay path
        c_ident = clone_cache(c_ctx_old)
        zero_delta = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in old_states_snap.items()}
        apply_jlocal_update(c_ident, zero_delta, g_old, beta_old, b_end, T_ctx, w_conv, d_k, transport_mode="scalar_decay")
        ident_eval = eval_suffix_from_ctx_cache(
            model, tok, c_ident, ids_o, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
            old_val, new_val, old_ids, new_ids, regime, oracle_first_logits=None, first_logits_only=True
        )
        del c_ident

        # Chunked vs step numerical precision floor check
        c_step = None
        step_cs = max(1, min(64, T_ctx // 2))
        for x0 in range(0, T_ctx, step_cs):
            x1 = min(T_ctx, x0 + step_cs)
            c_step = run_model_segment(model, c_step, ids_o[:, x0:x1], x0).past_key_values
        step_eval = eval_suffix_from_ctx_cache(
            model, tok, c_step, ids_o, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
            old_val, new_val, old_ids, new_ids, regime, first_logits_only=True
        )
        del c_step
        startup_meta = {
            "clone_cache_ptr_independent": True,
            "manual_eq_generate": True,
            "step_eval_logits": step_eval["first_logits"],
            "ident_eval_logits": ident_eval["first_logits"],
        }

    del c_one_pref, out_one

    # 3. Split oracle
    c_split_ctx = build_split_oracle_context_cache(model, ids_e[:, :T_ctx], p0, b_end, T_ctx, chunk_size=chunk_size)
    split_eval = eval_suffix_from_ctx_cache(
        model, tok, c_split_ctx, ids_e, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
        old_val, new_val, old_ids, new_ids, regime, oracle_first_logits=lg_one, oracle_new_ans_logits=oracle_new_ans_logits
    )
    del c_split_ctx

    if not skip_oracle_precheck:
        assert_oracle_first_token(
            tok, split_eval["first_logits"], split_eval["gen_ids"], lg_one, gen_one, old_val, new_val, label=f"E2-{s['sample_id']}"
        )

    oneshot_eval_gen = tok.decode([t for t in gen_one if t not in eos_ids], skip_special_tokens=True).strip()
    one_probs = F.softmax(lg_one.float(), dim=-1)
    oneshot_summary = {
        "gen_text": oneshot_eval_gen,
        "exact_match": bool(oneshot_eval_gen == new_val),
        "first_tok_new_hit": bool(gen_one[0] == new_ids[0]) if gen_one else False,
        "old_leak": bool(oneshot_eval_gen == old_val),
        "p_new_1st": float(one_probs[0, new_ids[0]].item()),
        "p_old_1st": float(one_probs[0, old_ids[0]].item()),
        "seq_logprob_new": lp_one_new,
    }

    # 4. Old unedited baseline
    old_eval = eval_suffix_from_ctx_cache(
        model, tok, c_ctx_old, ids_o, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
        old_val, new_val, old_ids, new_ids, regime, oracle_first_logits=lg_one, oracle_new_ans_logits=oracle_new_ans_logits
    )
    if run_startup_checks:
        startup_meta["identity_kl"] = kl_div_from_logits(old_eval["first_logits"], startup_meta.pop("ident_eval_logits"))
        startup_meta["chunked_vs_step_kl"] = kl_div_from_logits(old_eval["first_logits"], startup_meta.pop("step_eval_logits"))
        assert startup_meta["identity_kl"] == 0.0, f"Identity KL != 0: {startup_meta['identity_kl']}"

    # 5. B-ckpt local window [c, p0) -> [p0, b_end)
    t0_local = sync_time(device)
    c_p, ckpt_c = prepare_bckpt_cache(model, ckpts_old, ids_o[:, :T_ctx], p0, chunk_size=chunk_size)
    c_old_b = run_model_segment(model, clone_cache(c_p), ids_o[:, p0:b_end], p0).past_key_values
    c_new_b = run_model_segment(model, c_p, ids_e[:, p0:b_end], p0).past_key_values
    s_old_b = snap_states(c_old_b)
    s_new_b = snap_states(c_new_b)
    delta_blk = compute_state_delta_fp64(s_old_b, s_new_b)

    # 6. J-local (UNIVERSAL DEFAULT: scalar_decay + patched KV)
    c_jlocal = clone_cache(c_ctx_old)
    scalar_stats = apply_jlocal_update(
        c_jlocal, delta_blk, g_old, beta_old, b_end, T_ctx, w_conv, d_k, transport_mode="scalar_decay"
    )
    patch_kv_range(c_jlocal, c_new_b, p0, b_end)
    local_edit_ms = (sync_time(device) - t0_local) * 1000.0

    jlocal_eval = eval_suffix_from_ctx_cache(
        model, tok, c_jlocal, ids_o, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
        old_val, new_val, old_ids, new_ids, regime, oracle_first_logits=lg_one, oracle_new_ans_logits=oracle_new_ans_logits
    )
    del c_jlocal

    # 7. J-local ablations: J_local_iso (non-default iso_decay), J_local_stale, J_local_evict (scalar_decay+mask), state_only (scalar_decay+old KV), kv_only
    c_iso = clone_cache(c_ctx_old)
    iso_proxy = apply_jlocal_update(
        c_iso, delta_blk, g_old, beta_old, b_end, T_ctx, w_conv, d_k, transport_mode="iso_decay"
    )
    patch_kv_range(c_iso, c_new_b, p0, b_end)
    jlocal_iso_eval = eval_suffix_from_ctx_cache(
        model, tok, c_iso, ids_o, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
        old_val, new_val, old_ids, new_ids, regime, oracle_first_logits=lg_one, oracle_new_ans_logits=oracle_new_ans_logits
    )
    del c_iso

    c_stale = clone_cache(c_ctx_old)
    apply_jlocal_update(c_stale, delta_blk, g_old, beta_old, b_end, T_ctx, w_conv, d_k, transport_mode="stale")
    patch_kv_range(c_stale, c_new_b, p0, b_end)
    jlocal_stale_eval = eval_suffix_from_ctx_cache(
        model, tok, c_stale, ids_o, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
        old_val, new_val, old_ids, new_ids, regime, oracle_first_logits=lg_one, oracle_new_ans_logits=oracle_new_ans_logits
    )
    del c_stale

    c_evict = clone_cache(c_ctx_old)
    apply_jlocal_update(c_evict, delta_blk, g_old, beta_old, b_end, T_ctx, w_conv, d_k, transport_mode="scalar_decay")
    evict_mask = torch.ones((1, T_ctx), dtype=torch.long, device=device)
    evict_mask[:, p0:p1] = 0  # evict_at_fact via attention_mask only
    jlocal_evict_eval = eval_suffix_from_ctx_cache(
        model, tok, c_evict, ids_o, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
        old_val, new_val, old_ids, new_ids, regime, attn_mask_ctx=evict_mask,
        oracle_first_logits=lg_one, oracle_new_ans_logits=oracle_new_ans_logits
    )
    del c_evict

    c_state_only = clone_cache(c_ctx_old)
    apply_jlocal_update(c_state_only, delta_blk, g_old, beta_old, b_end, T_ctx, w_conv, d_k, transport_mode="scalar_decay")
    state_only_eval = eval_suffix_from_ctx_cache(
        model, tok, c_state_only, ids_o, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
        old_val, new_val, old_ids, new_ids, regime, oracle_first_logits=lg_one, oracle_new_ans_logits=oracle_new_ans_logits
    )
    del c_state_only

    c_kv_only = clone_cache(c_ctx_old)
    patch_kv_range(c_kv_only, c_new_b, p0, b_end)
    kv_only_eval = eval_suffix_from_ctx_cache(
        model, tok, c_kv_only, ids_o, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
        old_val, new_val, old_ids, new_ids, regime, oracle_first_logits=lg_one, oracle_new_ans_logits=oracle_new_ans_logits
    )
    del c_kv_only

    # 8. J-full (Decision A direct copy_ of s_new + [p0, b_end) KV patch + recompute [b_end, T_ctx))
    c_jfull = clone_cache(c_old_b)
    overwrite_states(c_jfull, s_new_b)
    patch_kv_range(c_jfull, c_new_b, p0, b_end)
    if b_end < T_ctx:
        c_jfull = run_model_segment(model, c_jfull, ids_o[:, b_end:T_ctx], b_end).past_key_values
    assert cache_seq_len(c_jfull) == T_ctx, f"J-full cache len {cache_seq_len(c_jfull)} != {T_ctx}"
    jfull_eval = eval_suffix_from_ctx_cache(
        model, tok, c_jfull, ids_o, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
        old_val, new_val, old_ids, new_ids, regime, oracle_first_logits=lg_one, oracle_new_ans_logits=oracle_new_ans_logits
    )
    del c_jfull

    # 9. Mandatory Isolation: J-control (O_local s_new at b, OLD KV, recomputed suffix [b, T_ctx))
    c_jctrl = truncate_cache_kv(clone_cache(c_ctx_old), b_end)
    overwrite_states(c_jctrl, s_new_b)
    if b_end < T_ctx:
        c_jctrl = run_model_segment(model, c_jctrl, ids_o[:, b_end:T_ctx], b_end).past_key_values
    assert cache_seq_len(c_jctrl) == T_ctx, f"J-control cache len {cache_seq_len(c_jctrl)} != {T_ctx}"
    jcontrol_eval = eval_suffix_from_ctx_cache(
        model, tok, c_jctrl, ids_o, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
        old_val, new_val, old_ids, new_ids, regime, oracle_first_logits=lg_one, oracle_new_ans_logits=oracle_new_ans_logits
    )
    del c_jctrl

    # 10. Mandatory Isolation: J-local-recompute (fp64 state reconstruction at b + patched KV + recomputed suffix)
    c_jlrec = truncate_cache_kv(clone_cache(c_ctx_old), b_end)
    reconstructed_b = {
        key: (s_old_b[key].double() + delta_blk[key]).to(dtype=s_new_b[key].dtype) for key in delta_blk
    }
    overwrite_states(c_jlrec, reconstructed_b)
    patch_kv_range(c_jlrec, c_new_b, p0, b_end)
    if b_end < T_ctx:
        c_jlrec = run_model_segment(model, c_jlrec, ids_o[:, b_end:T_ctx], b_end).past_key_values
    assert cache_seq_len(c_jlrec) == T_ctx, f"J-local-recompute cache len {cache_seq_len(c_jlrec)} != {T_ctx}"
    jlrecomp_eval = eval_suffix_from_ctx_cache(
        model, tok, c_jlrec, ids_o, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
        old_val, new_val, old_ids, new_ids, regime, oracle_first_logits=lg_one, oracle_new_ans_logits=oracle_new_ans_logits
    )
    del c_jlrec, c_old_b, c_new_b

    # 11. Placebo unrelated span edit ("stone" -> "brick") using scalar_decay
    c_pu, _ = prepare_bckpt_cache(model, ckpts_old, ids_o[:, :T_ctx], u0, chunk_size=chunk_size)
    c_old_u = run_model_segment(model, clone_cache(c_pu), ids_o[:, u0:b_unrel], u0).past_key_values
    c_new_u = run_model_segment(model, c_pu, ids_placebo[:, u0:b_unrel], u0).past_key_values
    delta_u = compute_state_delta_fp64(snap_states(c_old_u), snap_states(c_new_u))
    c_placebo = clone_cache(c_ctx_old)
    apply_jlocal_update(c_placebo, delta_u, g_old, beta_old, b_unrel, T_ctx, w_conv, d_k, transport_mode="scalar_decay")
    patch_kv_range(c_placebo, c_new_u, u0, b_unrel)
    placebo_eval = eval_suffix_from_ctx_cache(
        model, tok, c_placebo, ids_o, T_ctx, T_pref_m1, last_tok, max_new, eos_ids,
        old_val, new_val, old_ids, new_ids, regime, first_logits_only=True
    )
    placebo_kl_vs_orig = kl_div_from_logits(old_eval["first_logits"], placebo_eval["first_logits"])
    del c_old_u, c_new_u, c_placebo

    assert_cache_unchanged(c_ctx_old, old_kv_snap, old_states_snap, check_kv_idx, label=f"E2-{s['sample_id']}")

    vram = get_vram_snapshot(device, total_vram_gb)
    wall_s = sync_time(device) - t0_wall

    del c_ctx_old, ckpts_old, g_old, beta_old, old_states_snap, old_kv_snap, s_old_b, s_new_b, delta_blk, delta_u

    # G1 dual-track checks on J-full
    g1a_diff_vs_split = max_abs_logit_diff(jfull_eval["first_logits"], split_eval["first_logits"])
    g1a_pass = bool(g1a_diff_vs_split <= 1e-5)

    g1b_kl_vs_oneshot = kl_div_from_logits(lg_one, jfull_eval["first_logits"])
    g1b_diff_vs_oneshot = max_abs_logit_diff(lg_one, jfull_eval["first_logits"])
    g1b_strict = bool(g1b_kl_vs_oneshot <= g1b_thresh)
    g1b_rounding_eligible = bool(g1b_diff_vs_oneshot <= 0.28125 and g1b_kl_vs_oneshot <= 1.5e-3)
    g1b_rounding_exception = bool((not g1b_strict) and g1b_rounding_eligible)
    g1b_pass = bool(g1b_strict or (allow_rounding_exception and g1b_rounding_eligible))

    return {
        "task_group": "e2",
        "sample_id": s["sample_id"],
        "tpl_id": s["tpl_id"],
        "regime": regime,
        "pair_format": s["pair_format"],
        "target_entity": s["target_entity"],
        "old_val": old_val,
        "new_val": new_val,
        "T_ctx": T_ctx,
        "p0": p0,
        "p1": p1,
        "b_end": b_end,
        "u0": u0,
        "b_unrel": b_unrel,
        "ckpt_c": ckpt_c,
        "suffix_len": T_ctx - b_end,
        "placebo_suffix_len": T_ctx - b_unrel,
        "tail_conv_overlap": bool(b_end >= T_ctx - w_conv + 1),
        "transport_mode": "scalar_decay",
        "oracle_precheck_pass": not skip_oracle_precheck,
        "local_edit_ms": local_edit_ms,
        "wall_s": wall_s,
        "peak_allocated_gb": vram["peak_allocated_gb"],
        "peak_reserved_gb": vram["peak_reserved_gb"],
        "headroom_gb": vram["headroom_gb"],
        "scalar_stats": scalar_stats,
        "iso_proxy": iso_proxy,
        "placebo_kl_vs_orig": placebo_kl_vs_orig,
        "g1a_diff_vs_split": g1a_diff_vs_split,
        "g1a_pass": g1a_pass,
        "g1b_kl_vs_oneshot": g1b_kl_vs_oneshot,
        "g1b_diff_vs_oneshot": g1b_diff_vs_oneshot,
        "g1b_strict": g1b_strict,
        "g1b_rounding_eligible": g1b_rounding_eligible,
        "g1b_rounding_exception": g1b_rounding_exception,
        "g1b_pass": g1b_pass,
        "startup_meta": startup_meta if startup_meta else None,
        "conditions": {
            "old": strip_tensors(old_eval),
            "oneshot_oracle": oneshot_summary,
            "split_oracle": strip_tensors(split_eval),
            "J_full": strip_tensors(jfull_eval),
            "J_local": strip_tensors(jlocal_eval),
            "J_local_iso": strip_tensors(jlocal_iso_eval),
            "J_local_stale": strip_tensors(jlocal_stale_eval),
            "J_local_evict": strip_tensors(jlocal_evict_eval),
            "state_only": strip_tensors(state_only_eval),
            "kv_only": strip_tensors(kv_only_eval),
            "J_control": strip_tensors(jcontrol_eval),
            "J_local_recompute": strip_tensors(jlrecomp_eval),
        },
    }


def _replace_span_tokens(
    tok, facts, tpl_id, pre_idx, post_idx, target_entity, pair_format, overrides, device
) -> torch.Tensor:
    info = build_prompt_with_spans(tok, tpl_id, facts, target_entity, pair_format, pre_idx, post_idx, fact_overrides=overrides)
    return info["input_ids"][:, : info["T_ctx"]].to(device)


@torch.no_grad()
def evaluate_e3_sample(
    model,
    tok,
    collector: GDNRateCollector,
    device: torch.device,
    total_vram_gb: float,
    s: dict[str, Any],
    w_conv: int,
    d_k: int,
    max_new: int,
    eos_ids: set[int],
    chunk_size: int,
    skip_oracle_precheck: bool = False,
) -> dict[str, Any]:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0_wall = sync_time(device)

    T_ctx = s["T_ctx"]
    ids_ctx_old = s["ids_ctx_old"].to(device)
    M, k = s["M"], s["k"]
    edit_indices = s["edit_indices"]
    collat_indices = s["collat_indices"]
    slots_by_fi = {slot["fact_idx"]: slot for slot in s["edit_slots"]}

    # Shared OLD prefill + rates
    c_ctx_old, g_old, beta_old, ckpts_old = run_context_with_rates(model, collector, ids_ctx_old, chunk_size=chunk_size)
    old_states_snap = snap_states(c_ctx_old)
    check_kv_idx = sorted(set([0, T_ctx // 2, T_ctx - 1]))
    old_kv_snap = snapshot_kv_subset(c_ctx_old, check_kv_idx)

    # 1. One-shot oracle & old baseline per fact query
    oracle_q_refs: dict[int, dict[str, Any]] = {}
    old_baseline_q: dict[int, dict[str, Any]] = {}
    oracle_baseline_q: dict[int, dict[str, Any]] = {}

    for fi in range(M):
        q = s["queries"][fi]
        ids_q_old = q["ids_old"].to(device)
        ids_q_ora = q["ids_oracle"].to(device)
        Tp_m1 = q["T_pref_minus_1"]
        last_tok_o = ids_q_old[:, Tp_m1 : Tp_m1 + 1]
        last_tok_e = ids_q_ora[:, Tp_m1 : Tp_m1 + 1]

        c_one_pref = run_model_segment(model, None, ids_q_ora[:, :Tp_m1], 0).past_key_values
        lg_one, gen_one, trace_one = manual_greedy_from_cache(
            model, c_one_pref, last_tok_e, Tp_m1, max_new=max_new, eos_ids=eos_ids, return_trace=True
        )
        t_ids = q["target_ids"]
        pre_one = trace_one[: len(t_ids)] if (len(t_ids) == 1 or (len(gen_one) >= len(t_ids) and gen_one[: len(t_ids)] == t_ids)) else None
        _, ora_ans_logits, _, _, _ = compute_answer_seq_metrics(
            model, c_one_pref, last_tok_e, Tp_m1, t_ids, precomputed_logits=pre_one
        )
        gen_one_txt = tok.decode([t for t in gen_one if t not in eos_ids], skip_special_tokens=True).strip()
        del c_one_pref

        if q["is_edited"] and not skip_oracle_precheck:
            assert_oracle_first_token(
                tok, lg_one, gen_one, lg_one, gen_one, q["old_val"], q["target_val"],
                label=f"{s['task_group']}-{s['sample_id']}-f{fi}"
            )
        oracle_q_refs[fi] = {"first_logits": lg_one, "ans_logits": ora_ans_logits}
        oracle_baseline_q[fi] = {
            "gen_text": gen_one_txt,
            "exact_match": bool(gen_one_txt == q["target_val"]),
            "is_edited": q["is_edited"],
        }

        ev_old = eval_suffix_from_ctx_cache(
            model, tok, c_ctx_old, ids_q_old, T_ctx, Tp_m1, last_tok_o, max_new, eos_ids,
            q["old_val"], q["target_val"], q["old_ids"], q["target_ids"], s["regime"],
            oracle_first_logits=lg_one, oracle_new_ans_logits=ora_ans_logits
        )
        old_baseline_q[fi] = {
            "exact_target": ev_old["exact_match"],
            "exact_old": bool(ev_old["gen_text"] == q["old_val"]),
            "gen_text": ev_old["gen_text"],
            "p_target": ev_old["p_new_1st"],
        }

    # Sequential O_local edit step using locked default transport_mode="scalar_decay"
    def _step_local_edit(c_cur, g_cur, b_cur, cur_ids, overrides_next, slot):
        p0, b_end = slot["p0"], slot["b_end"]
        next_ids = _replace_span_tokens(
            tok, s["facts"], s["tpl_id"], s["pre_idx"], s["post_idx"], slot["entity"], s["pair_format"], overrides_next, device
        )
        if overrides_next and len(overrides_next) == 1:
            c_p, _ = prepare_bckpt_cache(model, ckpts_old, cur_ids, p0, chunk_size=chunk_size)
        else:
            c_p = run_model_segment(model, None, cur_ids[:, :p0], 0).past_key_values if p0 > 0 else None

        c_old_b = run_model_segment(model, clone_cache(c_p), cur_ids[:, p0:b_end], p0).past_key_values
        collector.clear()
        collector.active = True
        try:
            c_new_b = run_model_segment(model, c_p, next_ids[:, p0:b_end], p0).past_key_values
            g_blk, b_blk = collector.collect_concat()
        finally:
            collector.active = False
            collector.clear()

        delta = compute_state_delta_fp64(snap_states(c_old_b), snap_states(c_new_b))
        c_next = clone_cache(c_cur)
        apply_jlocal_update(c_next, delta, g_cur, b_cur, b_end, T_ctx, w_conv, d_k, transport_mode="scalar_decay")
        patch_kv_range(c_next, c_new_b, p0, b_end)

        g_next = {li: t.clone() for li, t in g_cur.items()}
        b_next = {li: t.clone() for li, t in b_cur.items()}
        for li in g_blk:
            g_next[li][:, p0:b_end, :].copy_(g_blk[li].to(g_next[li].device))
            b_next[li][:, p0:b_end, :].copy_(b_blk[li].to(b_next[li].device))

        del c_old_b, c_new_b
        return c_next, g_next, b_next, next_ids, delta

    # 2. Run all permutations for J-local (scalar_decay)
    perm_results = []
    for perm_idx, perm in enumerate(s["perms"]):
        c_cur = clone_cache(c_ctx_old)
        g_cur = g_old
        b_cur = beta_old
        cur_ids = ids_ctx_old
        overrides: dict[int, str] = {}
        step_hashes: list[str] = []
        t0_p = sync_time(device)

        for slot_pos in perm:
            ei = edit_indices[slot_pos]
            slot = slots_by_fi[ei]
            overrides[ei] = slot["new_val"]
            c_next, g_next, b_next, next_ids, _ = _step_local_edit(c_cur, g_cur, b_cur, cur_ids, overrides, slot)
            if c_cur is not c_ctx_old:
                del c_cur
            c_cur, g_cur, b_cur, cur_ids = c_next, g_next, b_next, next_ids
            step_hashes.append(cache_fingerprint(c_cur))

        perm_edit_ms = (sync_time(device) - t0_p) * 1000.0

        edited_q_res = {}
        collat_q_res = {}
        for fi in range(M):
            q = s["queries"][fi]
            ids_q_old = q["ids_old"].to(device)
            Tp_m1 = q["T_pref_minus_1"]
            last_tok = ids_q_old[:, Tp_m1 : Tp_m1 + 1]
            ev = eval_suffix_from_ctx_cache(
                model, tok, c_cur, ids_q_old, T_ctx, Tp_m1, last_tok, max_new, eos_ids,
                q["old_val"], q["target_val"], q["old_ids"], q["target_ids"], s["regime"],
                oracle_first_logits=oracle_q_refs[fi]["first_logits"],
                oracle_new_ans_logits=oracle_q_refs[fi]["ans_logits"],
            )
            rec = strip_tensors(ev)
            rec["fact_idx"] = fi
            rec["entity"] = q["entity"]
            if q["is_edited"]:
                edited_q_res[str(fi)] = rec
            else:
                collat_q_res[str(fi)] = rec

        del c_cur
        all_edited_ok = all(r["exact_match"] for r in edited_q_res.values())
        all_collat_ok = all(r["exact_match"] for r in collat_q_res.values()) if collat_q_res else True
        perm_results.append(
            {
                "perm_idx": perm_idx,
                "perm": list(perm),
                "is_canonical": (list(perm) == list(range(k))),
                "transport_mode": "scalar_decay",
                "edit_ms": perm_edit_ms,
                "step_hashes": step_hashes,
                "all_edited_exact": all_edited_ok,
                "edited_exact_count": sum(1 for r in edited_q_res.values() if r["exact_match"]),
                "all_collat_exact": all_collat_ok,
                "collat_exact_count": sum(1 for r in collat_q_res.values() if r["exact_match"]),
                "edited_queries": edited_q_res,
                "collateral_queries": collat_q_res,
            }
        )

    # 3. Mandatory isolations on canonical order: J-control (old KV + recompute) & J-local-recompute (patched KV + recompute)
    def _run_canonical_recompute(patch_kv: bool) -> dict[str, Any]:
        c_rec = clone_cache(c_ctx_old)
        cur_ids_rec = ids_ctx_old
        overrides_rec: dict[int, str] = {}
        for ei in edit_indices:
            slot = slots_by_fi[ei]
            p0, b_end = slot["p0"], slot["b_end"]
            overrides_rec[ei] = slot["new_val"]
            next_ids_rec = _replace_span_tokens(
                tok, s["facts"], s["tpl_id"], s["pre_idx"], s["post_idx"], slot["entity"], s["pair_format"], overrides_rec, device
            )
            c_p = run_model_segment(model, None, cur_ids_rec[:, :p0], 0).past_key_values if p0 > 0 else None
            c_new_b = run_model_segment(model, c_p, next_ids_rec[:, p0:b_end], p0).past_key_values
            c_rec = truncate_cache_kv(c_rec, b_end)
            overwrite_states(c_rec, snap_states(c_new_b))
            if patch_kv:
                patch_kv_range(c_rec, c_new_b, p0, b_end)
            cur_ids_rec = next_ids_rec
            suf_ids = cur_ids_rec[:, b_end:T_ctx] if patch_kv else ids_ctx_old[:, b_end:T_ctx]
            if b_end < T_ctx:
                c_rec = run_model_segment(model, c_rec, suf_ids, b_end).past_key_values
            assert cache_seq_len(c_rec) == T_ctx, f"E3 canonical recompute len {cache_seq_len(c_rec)} != {T_ctx}"
            del c_new_b

        ed_out, col_out = {}, {}
        for fi in range(M):
            q = s["queries"][fi]
            ids_q_old = q["ids_old"].to(device)
            Tp_m1 = q["T_pref_minus_1"]
            last_tok = ids_q_old[:, Tp_m1 : Tp_m1 + 1]
            ev = eval_suffix_from_ctx_cache(
                model, tok, c_rec, ids_q_old, T_ctx, Tp_m1, last_tok, max_new, eos_ids,
                q["old_val"], q["target_val"], q["old_ids"], q["target_ids"], s["regime"],
                oracle_first_logits=oracle_q_refs[fi]["first_logits"],
                oracle_new_ans_logits=oracle_q_refs[fi]["ans_logits"],
            )
            rec = strip_tensors(ev)
            rec["fact_idx"] = fi
            if q["is_edited"]:
                ed_out[str(fi)] = rec
            else:
                col_out[str(fi)] = rec
        del c_rec
        return {
            "all_edited_exact": all(r["exact_match"] for r in ed_out.values()),
            "edited_exact_count": sum(1 for r in ed_out.values() if r["exact_match"]),
            "collat_exact_count": sum(1 for r in col_out.values() if r["exact_match"]),
            "edited_queries": ed_out,
            "collateral_queries": col_out,
        }

    jcontrol_e3 = _run_canonical_recompute(patch_kv=False)
    jlrecomp_e3 = _run_canonical_recompute(patch_kv=True)

    assert_cache_unchanged(c_ctx_old, old_kv_snap, old_states_snap, check_kv_idx, label=f"{s['task_group']}-{s['sample_id']}")

    vram = get_vram_snapshot(device, total_vram_gb)
    wall_s = sync_time(device) - t0_wall

    del c_ctx_old, ckpts_old, g_old, beta_old, oracle_q_refs, old_states_snap, old_kv_snap

    worst_all_edited_ok = all(pr["all_edited_exact"] for pr in perm_results)
    worst_edited_frac = min(pr["edited_exact_count"] / k for pr in perm_results)

    return {
        "task_group": s["task_group"],
        "sample_id": s["sample_id"],
        "M": M,
        "k": k,
        "tpl_id": s["tpl_id"],
        "regime": s["regime"],
        "pair_format": s["pair_format"],
        "T_ctx": T_ctx,
        "min_slot_suffix_len": min(T_ctx - sl["b_end"] for sl in s["edit_slots"]),
        "edit_indices": edit_indices,
        "collat_indices": collat_indices,
        "n_perms": len(perm_results),
        "n_collat_facts": len(collat_indices),
        "transport_mode": "scalar_decay",
        "oracle_precheck_pass": not skip_oracle_precheck,
        "wall_s": wall_s,
        "peak_allocated_gb": vram["peak_allocated_gb"],
        "peak_reserved_gb": vram["peak_reserved_gb"],
        "headroom_gb": vram["headroom_gb"],
        "worst_order_all_edited_exact": worst_all_edited_ok,
        "worst_order_edited_frac": worst_edited_frac,
        "old_baseline": {str(fi): v for fi, v in old_baseline_q.items()},
        "oracle_baseline": {str(fi): v for fi, v in oracle_baseline_q.items()},
        "perm_results": perm_results,
        "J_control": jcontrol_e3,
        "J_local_recompute": jlrecomp_e3,
    }


# ---------- Statistics, Paired Tests & Summary Helpers ----------

def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total)))
    return max(0.0, center - half), min(1.0, center + half)


def one_sided_wilson_lower(successes: int, total: int, z: float = 1.6448536269514722) -> float:
    lo, _ = wilson_ci(successes, total, z=z)
    return lo


def one_sided_wilson_upper(successes: int, total: int, z: float = 1.6448536269514722) -> float:
    _, hi = wilson_ci(successes, total, z=z)
    return hi


def paired_diff_one_sided_upper(diffs: list[float], z: float = 1.6448536269514722) -> dict[str, float]:
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "one_sided_95_upper": 0.0}
    mean_d = sum(diffs) / n
    if n == 1:
        return {"n": 1, "mean": mean_d, "sd": 0.0, "one_sided_95_upper": mean_d}
    var = sum((x - mean_d) ** 2 for x in diffs) / (n - 1)
    sd = math.sqrt(max(0.0, var))
    se = sd / math.sqrt(n)
    return {"n": n, "mean": mean_d, "sd": sd, "one_sided_95_upper": mean_d + z * se}


def bootstrap_mean_ci(vals: list[float], n_boot: int = 2000, seed: int = 42) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "iqr": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    med_v = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25]) if n > 1 else (med_v, med_v)
    if n == 1:
        return {"n": 1, "mean": mean_v, "std": 0.0, "median": med_v, "iqr": 0.0, "ci95_low": mean_v, "ci95_high": mean_v}
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
    """Exact two-sided binomial sign test p-value for k successes out of n under p=0.5."""
    if n <= 0:
        return 1.0
    m = min(k, n - k)
    if 2 * m == n:
        return 1.0
    log_half_n = -n * math.log(2.0)
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n
        for i in range(m + 1)
    ]
    max_lt = max(log_terms)
    one_tail = math.exp(max_lt) * sum(math.exp(lt - max_lt) for lt in log_terms)
    return float(min(1.0, 2.0 * one_tail))


def wilcoxon_signed_rank_two_sided_p(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value (normal approx + tie-variance adjustment, zero scipy dep)."""
    nz = diffs[np.abs(diffs) > 1e-12]
    n = int(nz.size)
    if n < 2:
        return 1.0
    abs_d = np.abs(nz)
    order = np.argsort(abs_d)
    sorted_abs = abs_d[order]

    ranks = np.empty(n, dtype=np.float64)
    tie_counts: list[int] = []
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


def paired_e2_comparison(records: list[dict[str, Any]], cond_a: str, cond_b: str) -> dict[str, Any]:
    """Paired comparison between cond_a and cond_b on drift_mean_ans_kl, kl_first_vs_oracle, and exact_match."""
    ans_a, ans_b = [], []
    f_a, f_b = [], []
    em_a, em_b = [], []

    for r in records:
        ca = r["conditions"].get(cond_a)
        cb = r["conditions"].get(cond_b)
        if ca is None or cb is None:
            continue
        ans_a.append(float(ca["drift_mean_ans_kl"]))
        ans_b.append(float(cb["drift_mean_ans_kl"]))
        f_a.append(float(ca["kl_first_vs_oracle"]))
        f_b.append(float(cb["kl_first_vs_oracle"]))
        em_a.append(1.0 if ca["exact_match"] else 0.0)
        em_b.append(1.0 if cb["exact_match"] else 0.0)

    n = len(ans_a)
    if n == 0:
        return {"n": 0}

    def _kl_block(va: list[float], vb: list[float]) -> dict[str, Any]:
        diff = np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64)
        a_lower = int(np.sum(diff < -1e-9))
        b_lower = int(np.sum(diff > 1e-9))
        ties = int(n - a_lower - b_lower)
        return {
            "diff_a_minus_b": bootstrap_mean_ci(diff.tolist()),
            "a_lower_count": a_lower,
            "b_lower_count": b_lower,
            "ties_count": ties,
            "sign_test_p_two_sided": exact_sign_test_two_sided_p(a_lower, a_lower + b_lower),
            "wilcoxon_p_two_sided": wilcoxon_signed_rank_two_sided_p(diff),
        }

    diff_em = np.asarray(em_a, dtype=np.float64) - np.asarray(em_b, dtype=np.float64)
    return {
        "n": n,
        "cond_a": cond_a,
        "cond_b": cond_b,
        "drift_mean_ans_kl": _kl_block(ans_a, ans_b),
        "kl_first_vs_oracle": _kl_block(f_a, f_b),
        "exact_match_diff_a_minus_b": bootstrap_mean_ci(diff_em.tolist()),
        "a_wins_exact_match": int(np.sum(diff_em > 0.5)),
        "b_wins_exact_match": int(np.sum(diff_em < -0.5)),
    }


def sanitize_for_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def summarize_e2_records(
    records: list[dict[str, Any]],
    g1b_thresh: float,
    allow_rounding_exception: bool,
) -> dict[str, Any]:
    if not records:
        return {}
    n = len(records)

    placebo_kls = [r["placebo_kl_vs_orig"] for r in records if r.get("placebo_kl_vs_orig") is not None]
    p_mean = statistics.mean(placebo_kls) if placebo_kls else 0.0
    p_sd = statistics.stdev(placebo_kls) if len(placebo_kls) > 1 else 0.0
    drift_delta_floor = max(0.01, p_mean + 3.0 * p_sd)

    startup_entries = [r["startup_meta"] for r in records if r.get("startup_meta")]

    g1a_cnt = sum(1 for r in records if r["g1a_pass"])
    g1b_strict_cnt = sum(1 for r in records if r["g1b_strict"])
    g1b_exc_cnt = sum(1 for r in records if r["g1b_rounding_exception"])
    g1b_pass_cnt = sum(1 for r in records if r["g1b_pass"])
    g1b_tail_fail_cnt = sum(1 for r in records if not r["g1b_pass"])

    g1b_tail_inventory = []
    for r in records:
        if not r["g1b_strict"]:
            g1b_tail_inventory.append(
                {
                    "sample_id": r["sample_id"],
                    "tpl_id": r["tpl_id"],
                    "regime": r["regime"],
                    "pair_format": r["pair_format"],
                    "old_val": r["old_val"],
                    "new_val": r["new_val"],
                    "g1a_diff_vs_split": r["g1a_diff_vs_split"],
                    "g1b_diff_vs_oneshot": r["g1b_diff_vs_oneshot"],
                    "g1b_kl_vs_oneshot": r["g1b_kl_vs_oneshot"],
                    "g1b_rounding_eligible": r["g1b_rounding_eligible"],
                    "g1b_pass": r["g1b_pass"],
                    "jfull_gen_text": r["conditions"]["J_full"]["gen_text"],
                    "oneshot_gen_text": r["conditions"]["oneshot_oracle"]["gen_text"],
                }
            )

    cond_names = list(records[0]["conditions"].keys())

    def _agg_subset(sub: list[dict[str, Any]]) -> dict[str, Any]:
        m = len(sub)
        if m == 0:
            return {"n": 0}
        out = {}
        for cname in cond_names:
            c_rows = [r["conditions"][cname] for r in sub]
            succ_cnt = sum(1 for x in c_rows if x["exact_match"])
            first_cnt = sum(1 for x in c_rows if x["first_tok_new_hit"])
            leak_cnt = sum(1 for x in c_rows if x["old_leak"])
            w_lo, w_hi = wilson_ci(succ_cnt, m)
            l_lo, l_hi = wilson_ci(leak_cnt, m)
            entry: dict[str, Any] = {
                "n": m,
                "exact_match_count": succ_cnt,
                "exact_match_rate": succ_cnt / m,
                "exact_match_95ci": [w_lo, w_hi],
                "exact_match_1sided95_lower": one_sided_wilson_lower(succ_cnt, m),
                "first_tok_hit_rate": first_cnt / m,
                "old_leak_count": leak_cnt,
                "old_leak_rate": leak_cnt / m,
                "old_leak_95ci": [l_lo, l_hi],
                "old_leak_1sided95_upper": one_sided_wilson_upper(leak_cnt, m),
                "mean_p_new_1st": statistics.mean(x["p_new_1st"] for x in c_rows),
                "min_p_new_1st": min(x["p_new_1st"] for x in c_rows),
                "mean_p_old_1st": statistics.mean(x["p_old_1st"] for x in c_rows),
                "mean_seq_logprob_new": statistics.mean(x["seq_logprob_new"] for x in c_rows),
            }
            if "kl_first_vs_oracle" in c_rows[0]:
                raw_kls = [x["drift_mean_ans_kl"] for x in c_rows]
                first_kls = [x["kl_first_vs_oracle"] for x in c_rows]
                max_kls = [x["drift_max_ans_kl"] for x in c_rows]
                diffs = [x["max_abs_logit_diff_vs_oracle"] for x in c_rows]
                net_kls = [max(0.0, k - drift_delta_floor) for k in raw_kls]
                s_raw = sorted(raw_kls)
                entry.update(
                    {
                        "mean_drift_ans_kl": statistics.mean(raw_kls),
                        "median_drift_ans_kl": statistics.median(raw_kls),
                        "p95_drift_ans_kl": s_raw[min(m - 1, int(0.95 * m))],
                        "max_drift_ans_kl": max(raw_kls),
                        "mean_first_tok_kl": statistics.mean(first_kls),
                        "mean_max_ans_kl": statistics.mean(max_kls),
                        "mean_net_drift_kl": statistics.mean(net_kls),
                        "max_abs_logit_diff_mean": statistics.mean(diffs),
                        "max_abs_logit_diff_max": max(diffs),
                        "within_drift_floor_count": sum(1 for k in raw_kls if k <= drift_delta_floor),
                    }
                )
            out[cname] = entry
        return out

    overall_conds = _agg_subset(records)
    by_template = {str(tid): _agg_subset([r for r in records if r["tpl_id"] == tid]) for tid in (0, 1, 2)}
    by_regime = {reg: _agg_subset([r for r in records if r["regime"] == reg]) for reg in ("digit4", "single_token")}
    by_pair_format = {
        pf: _agg_subset([r for r in records if r["pair_format"] == pf]) for pf in ("digit4", "single_letter", "single_digit")
    }

    # G2 gates & non-inferiority computed strictly on J_local (scalar_decay)
    diffs_vs_oneshot = [
        (1.0 if r["conditions"]["oneshot_oracle"]["exact_match"] else 0.0)
        - (1.0 if r["conditions"]["J_local"]["exact_match"] else 0.0)
        for r in records
    ]
    noninf_vs_oneshot = paired_diff_one_sided_upper(diffs_vs_oneshot)

    jl_stats = overall_conds["J_local"]
    latencies = [r["local_edit_ms"] for r in records]
    walls = [r["wall_s"] for r in records if "wall_s" in r]
    g2_checks = {
        "jlocal_transport_mode": "scalar_decay",
        "sufficient_power_n_ge_20": bool(n >= 20),
        "exact_match_ge_90pct": bool(jl_stats["exact_match_rate"] >= 0.90),
        "exact_match_1sided95_lower_ge_85pct": bool(jl_stats["exact_match_1sided95_lower"] >= 0.85),
        "non_inferior_vs_oneshot_5pp": bool(noninf_vs_oneshot["one_sided_95_upper"] <= 0.05),
        "old_leak_1sided95_upper_le_5pct": bool(jl_stats["old_leak_1sided95_upper"] <= 0.05),
        "mean_drift_kl_le_delta": bool(jl_stats["mean_drift_ans_kl"] <= drift_delta_floor),
        "all_g2_pass": bool(
            jl_stats["exact_match_rate"] >= 0.90
            and jl_stats["exact_match_1sided95_lower"] >= 0.85
            and noninf_vs_oneshot["one_sided_95_upper"] <= 0.05
            and jl_stats["old_leak_1sided95_upper"] <= 0.05
            and jl_stats["mean_drift_ans_kl"] <= drift_delta_floor
        ),
    }

    paired_comparisons = {
        "J_local_vs_J_local_iso": paired_e2_comparison(records, "J_local", "J_local_iso"),
        "J_local_vs_J_local_stale": paired_e2_comparison(records, "J_local", "J_local_stale"),
        "J_local_vs_state_only": paired_e2_comparison(records, "J_local", "state_only"),
    }

    return {
        "n_samples": n,
        "oracle_precheck_pass_count": sum(1 for r in records if r["oracle_precheck_pass"]),
        "startup_checks": startup_entries,
        "g1a_vs_split": {
            "thresh": 1e-5,
            "pass_count": g1a_cnt,
            "pass_rate": g1a_cnt / n,
            "max_diff": max(r["g1a_diff_vs_split"] for r in records),
            "mean_diff": statistics.mean(r["g1a_diff_vs_split"] for r in records),
        },
        "g1b_vs_oneshot": {
            "strict_thresh": g1b_thresh,
            "allow_rounding_exception": allow_rounding_exception,
            "strict_pass_count": g1b_strict_cnt,
            "strict_pass_rate": g1b_strict_cnt / n,
            "rounding_exception_count": g1b_exc_cnt,
            "rounding_exception_rate": g1b_exc_cnt / n,
            "effective_pass_count": g1b_pass_cnt,
            "effective_pass_rate": g1b_pass_cnt / n,
            "tail_beyond_exception_count": g1b_tail_fail_cnt,
            "max_kl": max(r["g1b_kl_vs_oneshot"] for r in records),
            "mean_kl": statistics.mean(r["g1b_kl_vs_oneshot"] for r in records),
            "max_diff": max(r["g1b_diff_vs_oneshot"] for r in records),
            "tail_inventory_count": len(g1b_tail_inventory),
            "tail_inventory": g1b_tail_inventory,
        },
        "placebo_calibration": {
            "transport_mode": "scalar_decay",
            "n_placebo": len(placebo_kls),
            "placebo_mean_kl": p_mean,
            "placebo_sd_kl": p_sd,
            "drift_floor_delta": drift_delta_floor,
        },
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "sd": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
        },
        "wall_s": {
            "mean": statistics.mean(walls) if walls else 0.0,
            "max": max(walls) if walls else 0.0,
            "min": min(walls) if walls else 0.0,
        },
        "non_inferiority_jlocal_vs_oneshot": noninf_vs_oneshot,
        "g2_gate": g2_checks,
        "paired_comparisons": paired_comparisons,
        "overall_conditions": overall_conds,
        "by_template": by_template,
        "by_regime": by_regime,
        "by_pair_format": by_pair_format,
    }


def summarize_e3_records(e3_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not e3_records:
        return {}
    by_group = defaultdict(list)
    for r in e3_records:
        by_group[r["task_group"]].append(r)

    cells = {}
    all_cells_pass = True
    for group_name in sorted(by_group.keys()):
        rows = by_group[group_name]
        m = len(rows)
        M, k = rows[0]["M"], rows[0]["k"]
        n_perms = rows[0]["n_perms"]

        worst_sample_cnt = sum(1 for r in rows if r["worst_order_all_edited_exact"])
        wo_lo, wo_hi = wilson_ci(worst_sample_cnt, m)

        canonical_edited_cnt = 0
        canonical_edited_tot = 0
        allperm_edited_cnt = 0
        allperm_edited_tot = 0
        for r in rows:
            for pr in r["perm_results"]:
                c_ok = pr["edited_exact_count"]
                allperm_edited_cnt += c_ok
                allperm_edited_tot += k
                if pr["is_canonical"]:
                    canonical_edited_cnt += c_ok
                    canonical_edited_tot += k

        n_unique_collat = sum(r["n_collat_facts"] for r in rows)
        collat_old_ok = 0
        collat_ora_ok = 0
        collat_jl_canon_ok = 0
        collat_jl_worst_ok = 0
        collat_jctrl_ok = 0
        collat_jlrec_ok = 0
        paired_diff_vs_old = []
        paired_diff_vs_ora = []
        paired_diff_worst_vs_old = []

        for r in rows:
            canon_pr = next(pr for pr in r["perm_results"] if pr["is_canonical"])
            for fi in r["collat_indices"]:
                fis = str(fi)
                old_ok = 1.0 if r["old_baseline"][fis]["exact_target"] else 0.0
                ora_ok = 1.0 if r["oracle_baseline"][fis]["exact_match"] else 0.0
                jl_canon_ok = 1.0 if canon_pr["collateral_queries"][fis]["exact_match"] else 0.0
                jl_worst_ok = 1.0 if all(pr["collateral_queries"][fis]["exact_match"] for pr in r["perm_results"]) else 0.0
                jc_ok = 1.0 if r["J_control"]["collateral_queries"][fis]["exact_match"] else 0.0
                jlr_ok = 1.0 if r["J_local_recompute"]["collateral_queries"][fis]["exact_match"] else 0.0

                collat_old_ok += int(old_ok)
                collat_ora_ok += int(ora_ok)
                collat_jl_canon_ok += int(jl_canon_ok)
                collat_jl_worst_ok += int(jl_worst_ok)
                collat_jctrl_ok += int(jc_ok)
                collat_jlrec_ok += int(jlr_ok)
                paired_diff_vs_old.append(old_ok - jl_canon_ok)
                paired_diff_vs_ora.append(ora_ok - jl_canon_ok)
                paired_diff_worst_vs_old.append(old_ok - jl_worst_ok)

        jctrl_ed_ok = sum(r["J_control"]["edited_exact_count"] for r in rows)
        jlrec_ed_ok = sum(r["J_local_recompute"]["edited_exact_count"] for r in rows)
        jctrl_all_ok = sum(1 for r in rows if r["J_control"]["all_edited_exact"])
        jlrec_all_ok = sum(1 for r in rows if r["J_local_recompute"]["all_edited_exact"])

        diff_old_stats = paired_diff_one_sided_upper(paired_diff_vs_old)
        diff_ora_stats = paired_diff_one_sided_upper(paired_diff_vs_ora)
        diff_worst_stats = paired_diff_one_sided_upper(paired_diff_worst_vs_old)

        worst_sample_rate = worst_sample_cnt / m
        collat_pass = True if n_unique_collat == 0 else (diff_old_stats["one_sided_95_upper"] <= 0.02 and diff_worst_stats["one_sided_95_upper"] <= 0.02)
        cell_pass = bool(worst_sample_rate >= 0.90 and collat_pass)
        all_cells_pass = all_cells_pass and cell_pass

        by_tpl = {}
        for tid in (0, 1, 2):
            sub = [r for r in rows if r["tpl_id"] == tid]
            if sub:
                cnt = sum(1 for r in sub if r["worst_order_all_edited_exact"])
                by_tpl[str(tid)] = {"n": len(sub), "worst_order_all_edited_exact_rate": cnt / len(sub)}

        walls = [r["wall_s"] for r in rows if "wall_s" in r]
        cells[group_name] = {
            "M": M,
            "k": k,
            "n_samples": m,
            "n_perms_per_sample": n_perms,
            "transport_mode": "scalar_decay",
            "oracle_precheck_pass_count": sum(1 for r in rows if r["oracle_precheck_pass"]),
            "worst_order_all_edited_exact_count": worst_sample_cnt,
            "worst_order_all_edited_exact_rate": worst_sample_rate,
            "worst_order_95ci": [wo_lo, wo_hi],
            "canonical_edited_query_acc": canonical_edited_cnt / max(1, canonical_edited_tot),
            "all_perms_edited_query_acc": allperm_edited_cnt / max(1, allperm_edited_tot),
            "wall_s": {
                "mean": statistics.mean(walls) if walls else 0.0,
                "max": max(walls) if walls else 0.0,
            },
            "J_control": {
                "all_edited_exact_rate": jctrl_all_ok / m,
                "edited_query_acc": jctrl_ed_ok / (m * k),
                "collat_acc": (collat_jctrl_ok / n_unique_collat) if n_unique_collat > 0 else None,
            },
            "J_local_recompute": {
                "all_edited_exact_rate": jlrec_all_ok / m,
                "edited_query_acc": jlrec_ed_ok / (m * k),
                "collat_acc": (collat_jlrec_ok / n_unique_collat) if n_unique_collat > 0 else None,
            },
            "collateral": {
                "n_unique_probes": n_unique_collat,
                "note": "M==k==4 has 0 unedited schema facts" if n_unique_collat == 0 else None,
                "old_acc": (collat_old_ok / n_unique_collat) if n_unique_collat > 0 else None,
                "oracle_acc": (collat_ora_ok / n_unique_collat) if n_unique_collat > 0 else None,
                "jlocal_canonical_acc": (collat_jl_canon_ok / n_unique_collat) if n_unique_collat > 0 else None,
                "jlocal_worst_order_acc": (collat_jl_worst_ok / n_unique_collat) if n_unique_collat > 0 else None,
                "paired_delta_old_minus_jlocal_canonical": diff_old_stats,
                "paired_delta_oracle_minus_jlocal_canonical": diff_ora_stats,
                "paired_delta_old_minus_jlocal_worst": diff_worst_stats,
                "ci_upper_le_2pp_pass": collat_pass,
            },
            "by_template": by_tpl,
            "g3_cell_pass": cell_pass,
        }

    return {"cells": cells, "g3_all_cells_pass": all_cells_pass}


def load_completed_jsonl(path: str) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (str(rec["task_group"]), int(rec["sample_id"]))
                records[key] = rec
            except json.JSONDecodeError as exc:
                print(f"[warn] Skipping incomplete/corrupt JSONL line {line_no}: {exc}", flush=True)
    return records


def append_jsonl(path: str, record: dict[str, Any]):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    clean = sanitize_for_json(record)
    line = json.dumps(clean, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def collect_eos_ids(tok) -> set[int]:
    eos_ids: set[int] = set()
    if tok.eos_token_id is not None:
        eos_ids.add(int(tok.eos_token_id))
    for tok_str in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.add(tid)
    return eos_ids


def build_tiny_synthetic_qwen35(tok_model_id: str, offline: bool):
    """Build a tiny synthetic Qwen3_5Config model on CPU matching Qwen3.5 hybrid GDN + full attention."""
    from transformers import Qwen3_5Config, Qwen3_5TextConfig

    ref_cfg = AutoConfig.from_pretrained(tok_model_id, trust_remote_code=True, local_files_only=offline)
    ref_t = get_text_cfg(ref_cfg)
    tiny_rope = {
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
    }
    tiny_text_cfg = Qwen3_5TextConfig(
        vocab_size=int(getattr(ref_t, "vocab_size", 248320)),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        rope_parameters=tiny_rope,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        dtype=torch.float32,
    )
    tiny_cfg = Qwen3_5Config(text_config=tiny_text_cfg, tie_word_embeddings=True)
    model = AutoModelForCausalLM.from_config(tiny_cfg).to(dtype=torch.float32, device="cpu").eval()
    return model, tiny_text_cfg


def run_dataset_only_check(tok, args) -> dict[str, Any]:
    horizon_offset = args.horizon_override if args.horizon_override > 0 else 141
    e2_all, e2_inv = build_e2_dataset(tok, per_tpl=args.e2_per_tpl, seed=args.seed, horizon_offset=horizon_offset)
    e3_all, e3_inv = build_e3_dataset(tok, seed=args.seed, horizon_offset=horizon_offset, per_cell_limit=0)

    e2_probe = e2_all[:6]
    e3_probe, _ = build_e3_dataset(tok, seed=args.seed, horizon_offset=horizon_offset, per_cell_limit=1)

    e2_tpl_counts = {str(t): sum(1 for s in e2_all if s["tpl_id"] == t) for t in range(3)}
    e2_reg_counts = {r: sum(1 for s in e2_all if s["regime"] == r) for r in ("digit4", "single_token")}
    e2_pf_counts = {pf: sum(1 for s in e2_all if s["pair_format"] == pf) for pf in ("digit4", "single_letter", "single_digit")}
    probe_tpl_counts = {str(t): sum(1 for s in e2_probe if s["tpl_id"] == t) for t in range(3)}
    probe_pf_counts = {pf: sum(1 for s in e2_probe if s["pair_format"] == pf) for pf in ("digit4", "single_letter", "single_digit")}

    post_tok_lens = [len(tok.encode(blk, add_special_tokens=False)) for blk in POST_FILLER_BLOCKS]
    report = {
        "horizon_offset": horizon_offset,
        "post_filler_tok_lens": post_tok_lens,
        "e2_inventory": e2_inv,
        "e2_tpl_counts": e2_tpl_counts,
        "e2_regime_counts": e2_reg_counts,
        "e2_pair_format_counts": e2_pf_counts,
        "e2_sample_ids_contiguous": ([s["sample_id"] for s in e2_all] == list(range(len(e2_all)))),
        "e3_inventory": e3_inv,
        "probe10_e2_count": len(e2_probe),
        "probe10_e2_tpl_counts": probe_tpl_counts,
        "probe10_e2_pair_formats": probe_pf_counts,
        "probe10_e2_sample_ids": [s["sample_id"] for s in e2_probe],
        "probe10_e3_count": len(e3_probe),
        "probe10_e3_keys": [(s["task_group"], s["sample_id"]) for s in e3_probe],
    }
    print("DATASET_ONLY_REPORT=" + json.dumps(report, ensure_ascii=False), flush=True)
    assert e2_inv["rejections"] == 0 and e3_inv["rejections"] == 0
    assert e2_inv["min_suffix_len"] >= 25 and e2_inv["min_placebo_suffix_len"] >= 25 and e3_inv["min_suffix_len"] >= 25
    assert len(e2_all) == 3 * args.e2_per_tpl and len(e3_all) == 80
    assert len(e2_probe) == 6 and len(e3_probe) == 4
    return report


class _CpuTorchQwen35Fallback:
    """Temporarily unwrap FLA/Triton-decorated Qwen3.5 ops to pure-PyTorch for CPU self-test only."""

    _NAMES = (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    )

    def __init__(self) -> None:
        self._saved: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> "_CpuTorchQwen35Fallback":
        import transformers.models.qwen3_5.modeling_qwen3_5 as qm

        for name in self._NAMES:
            fn = getattr(qm, name, None)
            wrapped = getattr(fn, "__wrapped__", None) if fn is not None else None
            if wrapped is not None:
                self._saved.append((qm, name, fn))
                setattr(qm, name, wrapped)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for qm, name, orig in self._saved:
            setattr(qm, name, orig)
        self._saved.clear()


def run_self_test(args) -> None:
    """
    CPU end-to-end synthetic Qwen3_5Config self-test:
      1) Runs E2 (n=2) + E3 (2 cells) with startup checks + horizon=+141 + --skip-oracle-precheck
      2) Resumes with --resume to run +1 more E2 sample and regenerates JSON summary
      3) Cleans up its temporary scratch directory completely.
    """
    with _CpuTorchQwen35Fallback():
        _run_self_test_inner(args)


def _run_self_test_inner(args) -> None:
    print("[self-test] Building tiny synthetic Qwen3_5Config model on CPU...", flush=True)
    tok_id = "Qwen/Qwen3.5-2B" if args.model == "Qwen/Qwen3.5-27B" else args.model
    configure_hf_environment(offline=True)
    tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True, local_files_only=True)
    model, t_cfg = build_tiny_synthetic_qwen35(tok_id, offline=True)
    exec_device = resolve_exec_device(model, "cpu")
    eos_ids = collect_eos_ids(tok)
    w_conv = get_conv_kernel_dim(t_cfg)
    d_k = int(getattr(t_cfg, "linear_key_head_dim", 16))
    horizon_offset = args.horizon_override if args.horizon_override > 0 else 141

    mrope_info = verify_pure_text_mrope_equivalence(model, exec_device)
    print(f"[self-test] M-RoPE check passed: {mrope_info}", flush=True)

    e2_pool, e2_inv = build_e2_dataset(tok, per_tpl=2, seed=args.seed, horizon_offset=horizon_offset)
    e3_pool, e3_inv = build_e3_dataset(tok, seed=args.seed, horizon_offset=horizon_offset, per_cell_limit=1)
    assert e2_inv["min_suffix_len"] >= 25 and e2_inv["min_placebo_suffix_len"] >= 25
    assert e3_inv["min_suffix_len"] >= 25

    tmp_dir = tempfile.mkdtemp(prefix="w7_e9_selftest_")
    try:
        jsonl_path = os.path.join(tmp_dir, "selftest_records.jsonl")
        json_path = os.path.join(tmp_dir, "selftest_summary.json")

        collector = GDNRateCollector(model)
        completed: dict[tuple[str, int], dict[str, Any]] = {}
        try:
            # Phase A: E2[0:2] + E3[m4_k2, m8_k4]
            with open(jsonl_path, "w", encoding="utf-8"):
                pass
            for idx, s in enumerate(e2_pool[:2]):
                rec = evaluate_e2_sample(
                    model, tok, collector, exec_device, 0.0, s,
                    w_conv=w_conv, d_k=d_k, max_new=4, eos_ids=eos_ids,
                    chunk_size=128, g1b_thresh=args.g1b_thresh,
                    allow_rounding_exception=True,
                    run_startup_checks=(idx == 0),
                    skip_oracle_precheck=True,
                )
                assert rec["transport_mode"] == "scalar_decay"
                assert rec["g1a_pass"], f"Synthetic G1a failed: {rec['g1a_diff_vs_split']}"
                append_jsonl(jsonl_path, rec)
                completed[(rec["task_group"], rec["sample_id"])] = rec

            for s in (e3_pool[0], e3_pool[2]):  # m4_k2 #0 and m8_k2 #0
                rec3 = evaluate_e3_sample(
                    model, tok, collector, exec_device, 0.0, s,
                    w_conv=w_conv, d_k=d_k, max_new=4, eos_ids=eos_ids,
                    chunk_size=128, skip_oracle_precheck=True,
                )
                assert rec3["transport_mode"] == "scalar_decay"
                assert rec3["min_slot_suffix_len"] >= 25
                append_jsonl(jsonl_path, rec3)
                completed[(rec3["task_group"], rec3["sample_id"])] = rec3

            # Phase B: --resume test adding E2[2] and verifying skipped counts
            resumed_map = load_completed_jsonl(jsonl_path)
            assert len(resumed_map) == 4, f"Expected 4 records before resume, got {len(resumed_map)}"
            skipped = 0
            added = 0
            for s in e2_pool[:3]:
                key = (s["task_group"], s["sample_id"])
                if key in resumed_map:
                    skipped += 1
                    continue
                rec = evaluate_e2_sample(
                    model, tok, collector, exec_device, 0.0, s,
                    w_conv=w_conv, d_k=d_k, max_new=4, eos_ids=eos_ids,
                    chunk_size=128, g1b_thresh=args.g1b_thresh,
                    allow_rounding_exception=True,
                    run_startup_checks=False,
                    skip_oracle_precheck=True,
                )
                append_jsonl(jsonl_path, rec)
                resumed_map[key] = rec
                added += 1
            assert skipped == 2 and added == 1, f"Resume mismatch: skipped={skipped}, added={added}"
        finally:
            collector.close()

        all_recs = [resumed_map[k] for k in sorted(resumed_map.keys())]
        e2_recs = [r for r in all_recs if r["task_group"] == "e2"]
        e3_recs = [r for r in all_recs if r["task_group"].startswith("e3_")]
        summary = sanitize_for_json(
            {
                "meta": {
                    "mode": "self_test_cpu",
                    "horizon_offset": horizon_offset,
                    "n_e2": len(e2_recs),
                    "n_e3": len(e3_recs),
                    "mrope_equivalence": mrope_info,
                },
                "e2": summarize_e2_records(e2_recs, args.g1b_thresh, True),
                "e3": summarize_e3_records(e3_recs),
            }
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        loaded_sum = json.load(open(json_path, encoding="utf-8"))
        assert loaded_sum["e2"]["n_samples"] == 3
        assert "J_local_vs_J_local_iso" in loaded_sum["e2"]["paired_comparisons"]
        assert "J_local_vs_J_local_stale" in loaded_sum["e2"]["paired_comparisons"]
        assert set(loaded_sum["e3"]["cells"].keys()) == {"e3_m4_k2", "e3_m8_k2"}
        print(
            f"[self-test] PASS: E2 n={loaded_sum['e2']['n_samples']} (G1a={loaded_sum['e2']['g1a_vs_split']['pass_count']}/3, "
            f"startup_identity_kl={loaded_sum['e2']['startup_checks'][0]['identity_kl']:.1e}, "
            f"manual_eq_generate={loaded_sum['e2']['startup_checks'][0]['manual_eq_generate']}), "
            f"E3 cells={sorted(loaded_sum['e3']['cells'].keys())}, resume skipped=2 added=1",
            flush=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Wave 7 E9 27B 4-bit (NF4) Validation Runner")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--quant", type=str, default="nf4", choices=["nf4", "bf16"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--block-horizon", type=str, default="deep", choices=["deep", "stage", "single"])
    parser.add_argument("--horizon-override", type=int, default=0)
    parser.add_argument("--ckpt-size", type=int, default=256)
    parser.add_argument("--g1b-thresh", type=float, default=1e-4)
    parser.add_argument("--allow-rounding-exception", "--allow-1ulp-exception", action="store_true", dest="allow_rounding_exception")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--e2-per-tpl", type=int, default=40)
    parser.add_argument("--e2-limit", type=int, default=0)
    parser.add_argument("--e3-per-cell-limit", type=int, default=0)
    parser.add_argument("--probe-10", action="store_true", help="Select strict balanced 6 E2 + 4 E3 prefix")
    parser.add_argument("--only", type=str, default="all", choices=["all", "e2", "e3"])
    parser.add_argument("--max-new", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=0.0, help="Graceful wall-clock stop before Colab 15m limit")
    parser.add_argument("--jsonl-out", type=str, default="pilot/w7_e9_27b_results.jsonl")
    parser.add_argument("--json-out", type=str, default="pilot/w7_e9_27b_results.json")
    parser.add_argument("--dataset-only", "--verify-dataset-only", action="store_true", dest="dataset_only")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--skip-oracle-precheck", action="store_true")
    args = parser.parse_args()

    if args.skip_oracle_precheck and not args.self_test:
        raise ValueError("--skip-oracle-precheck is strictly restricted to --self-test mode")

    if args.self_test:
        run_self_test(args)
        return

    configure_hf_environment(offline=args.offline)

    if args.probe_10:
        if args.e2_limit <= 0:
            args.e2_limit = 6
        if args.e3_per_cell_limit <= 0:
            args.e3_per_cell_limit = 1

    print(
        f"Versions: torch={torch.__version__}, transformers={get_pkg_version('transformers')}, "
        f"bitsandbytes={get_pkg_version('bitsandbytes')}",
        flush=True,
    )

    if args.dataset_only:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.offline)
        run_dataset_only_check(tok, args)
        return

    t_start = time.perf_counter()
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.offline)
    t_cfg = get_text_cfg(cfg)
    n_linear = count_linear_layers(t_cfg)
    n_full = count_full_layers(t_cfg)
    w_conv = get_conv_kernel_dim(t_cfg)
    d_k = int(getattr(t_cfg, "linear_key_head_dim", 128))
    horizon_offset = args.horizon_override if args.horizon_override > 0 else resolve_horizon_offset(args.block_horizon, n_linear, w_conv)
    print(
        f"Model geometry: layers={n_linear + n_full} (linear={n_linear}, full={n_full}), "
        f"w_conv={w_conv}, d_k={d_k}, horizon={args.block_horizon} (+{horizon_offset})",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.offline)

    # Load model (NF4 4-bit matches pilot/w5_e9_27b_smoke.py)
    t0_load = time.perf_counter()
    if args.quant == "nf4":
        try:
            import bitsandbytes  # noqa: F401
            import accelerate  # noqa: F401
        except ImportError:
            if args.offline:
                raise
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "accelerate"])
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=args.offline,
        ).eval()
    else:
        torch_dtype = getattr(torch, args.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=args.offline,
        ).eval()
    weights_load_s = time.perf_counter() - t0_load

    exec_device = resolve_exec_device(model)
    total_vram_gb = (
        float(torch.cuda.get_device_properties(exec_device).total_memory / (1024 ** 3))
        if exec_device.type == "cuda" and torch.cuda.is_available()
        else 0.0
    )
    w_vram = get_vram_snapshot(exec_device, total_vram_gb)
    weights_allocated_gb = w_vram["allocated_gb"]
    weights_reserved_gb = w_vram["reserved_gb"]
    weights_headroom_gb = float(max(0.0, total_vram_gb - weights_reserved_gb)) if total_vram_gb > 0 else 0.0
    print(
        f"Loaded {args.model} ({args.quant}) in {weights_load_s:.1f}s on {exec_device}: "
        f"weights_allocated_gb={weights_allocated_gb:.4f}, weights_reserved_gb={weights_reserved_gb:.4f}, "
        f"weights_headroom_gb={weights_headroom_gb:.4f} / {total_vram_gb:.4f} GB",
        flush=True,
    )

    mrope_info = verify_pure_text_mrope_equivalence(model, exec_device)
    print(f"E0a M-RoPE check: {mrope_info}", flush=True)
    eos_ids = collect_eos_ids(tok)

    # Build full deterministic dataset first, then apply prefix slices
    e2_all, e2_inv = build_e2_dataset(tok, per_tpl=args.e2_per_tpl, seed=args.seed, horizon_offset=horizon_offset)
    e2_samples = e2_all[: args.e2_limit] if args.e2_limit > 0 else e2_all
    e3_samples, e3_inv = build_e3_dataset(
        tok, seed=args.seed, horizon_offset=horizon_offset, per_cell_limit=args.e3_per_cell_limit
    )
    print(
        f"Dataset pre-screened (E0c): E2={len(e2_samples)}/{e2_inv['n_e2']} (min_suf={e2_inv['min_suffix_len']}, "
        f"min_placebo_suf={e2_inv['min_placebo_suffix_len']}, rej={e2_inv['rejections']}), "
        f"E3={len(e3_samples)}/{e3_inv['n_e3_full']} (min_suf={e3_inv['min_suffix_len']}, rej={e3_inv['rejections']})",
        flush=True,
    )

    if args.resume:
        completed_map = load_completed_jsonl(args.jsonl_out)
        print(f"Resume enabled: loaded {len(completed_map)} existing records from {args.jsonl_out}", flush=True)
    else:
        completed_map = {}
        parent = os.path.dirname(os.path.abspath(args.jsonl_out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.jsonl_out, "w", encoding="utf-8"):
            pass

    stopped_by_timeout = False
    collector = GDNRateCollector(model)
    try:
        if args.only in ("all", "e2"):
            done_e2 = 0
            for idx, s in enumerate(e2_samples):
                if args.timeout_s > 0 and (time.perf_counter() - t_start) >= args.timeout_s:
                    print(
                        f"[timeout] Elapsed {time.perf_counter() - t_start:.1f}s >= {args.timeout_s:.1f}s before E2 id={s['sample_id']}; stopping gracefully.",
                        flush=True,
                    )
                    stopped_by_timeout = True
                    break
                key = (s["task_group"], s["sample_id"])
                if key in completed_map:
                    continue
                rec = evaluate_e2_sample(
                    model, tok, collector, exec_device, total_vram_gb, s,
                    w_conv=w_conv, d_k=d_k, max_new=args.max_new, eos_ids=eos_ids,
                    chunk_size=args.ckpt_size, g1b_thresh=args.g1b_thresh,
                    allow_rounding_exception=args.allow_rounding_exception,
                    run_startup_checks=(done_e2 == 0 and len(completed_map) == 0),
                    skip_oracle_precheck=False,
                )
                append_jsonl(args.jsonl_out, rec)
                completed_map[key] = rec
                done_e2 += 1
                jl = rec["conditions"]["J_local"]
                print(
                    f"[E2 {done_e2}/{len(e2_samples)}] id={s['sample_id']} tpl={s['tpl_id']} {s['pair_format']} "
                    f"{s['old_val']}->{s['new_val']} | G1a={rec['g1a_diff_vs_split']:.1e} "
                    f"G1b_KL={rec['g1b_kl_vs_oneshot']:.2e} (exc={rec['g1b_rounding_exception']}) | "
                    f"Jlocal={jl['gen_text']!r}(ok={jl['exact_match']}, kl={jl['drift_mean_ans_kl']:.2e}) | "
                    f"wall={rec['wall_s']:.2f}s res={rec['peak_reserved_gb']:.2f}GB head={rec['headroom_gb']:.2f}GB",
                    flush=True,
                )
                if done_e2 % 10 == 0:
                    gc.collect()
                    if exec_device.type == "cuda":
                        torch.cuda.empty_cache()

        if args.only in ("all", "e3") and not stopped_by_timeout:
            done_e3 = 0
            for s in e3_samples:
                if args.timeout_s > 0 and (time.perf_counter() - t_start) >= args.timeout_s:
                    print(
                        f"[timeout] Elapsed {time.perf_counter() - t_start:.1f}s >= {args.timeout_s:.1f}s before {s['task_group']} id={s['sample_id']}; stopping gracefully.",
                        flush=True,
                    )
                    stopped_by_timeout = True
                    break
                key = (s["task_group"], s["sample_id"])
                if key in completed_map:
                    continue
                rec = evaluate_e3_sample(
                    model, tok, collector, exec_device, total_vram_gb, s,
                    w_conv=w_conv, d_k=d_k, max_new=args.max_new, eos_ids=eos_ids,
                    chunk_size=args.ckpt_size, skip_oracle_precheck=False,
                )
                append_jsonl(args.jsonl_out, rec)
                completed_map[key] = rec
                done_e3 += 1
                print(
                    f"[E3 {done_e3}/{len(e3_samples)}] {s['task_group']} id={s['sample_id']} tpl={s['tpl_id']} "
                    f"{s['pair_format']} | worst_ok={rec['worst_order_all_edited_exact']} "
                    f"ctrl_ok={rec['J_control']['all_edited_exact']} rec_ok={rec['J_local_recompute']['all_edited_exact']} | "
                    f"wall={rec['wall_s']:.2f}s res={rec['peak_reserved_gb']:.2f}GB head={rec['headroom_gb']:.2f}GB",
                    flush=True,
                )
                if done_e3 % 5 == 0:
                    gc.collect()
                    if exec_device.type == "cuda":
                        torch.cuda.empty_cache()
    finally:
        collector.close()

    all_recs = [completed_map[k] for k in sorted(completed_map.keys())]
    e2_recs = [r for r in all_recs if r["task_group"] == "e2"]
    e3_recs = [r for r in all_recs if r["task_group"].startswith("e3_")]

    max_peak_alloc_gb = max((r.get("peak_allocated_gb", 0.0) for r in all_recs), default=weights_allocated_gb)
    max_peak_res_gb = max((r.get("peak_reserved_gb", 0.0) for r in all_recs), default=weights_reserved_gb)
    min_headroom_gb = (
        min((r["headroom_gb"] for r in all_recs if "headroom_gb" in r), default=weights_headroom_gb)
        if total_vram_gb > 0
        else 0.0
    )
    wall_s = time.perf_counter() - t_start

    vram_summary = {
        "total_vram_gb": total_vram_gb,
        "weights_load_s": weights_load_s,
        "weights_allocated_gb": weights_allocated_gb,
        "weights_reserved_gb": weights_reserved_gb,
        "weights_headroom_gb": weights_headroom_gb,
        "max_peak_allocated_gb": max_peak_alloc_gb,
        "max_peak_reserved_gb": max_peak_res_gb,
        "min_headroom_gb": min_headroom_gb,
    }

    summary = {
        "meta": {
            "experiment": "Wave7_E9_27B_NF4",
            "model": args.model,
            "quant": args.quant,
            "dtype": args.dtype,
            "torch_version": torch.__version__,
            "transformers_version": get_pkg_version("transformers"),
            "bitsandbytes_version": get_pkg_version("bitsandbytes"),
            "block_horizon": args.block_horizon,
            "horizon_offset": horizon_offset,
            "n_linear": n_linear,
            "n_full": n_full,
            "w_conv": w_conv,
            "d_k": d_k,
            "ckpt_size": args.ckpt_size,
            "g1b_thresh": args.g1b_thresh,
            "allow_rounding_exception": args.allow_rounding_exception,
            "transport_mode_default": "scalar_decay",
            "e2_limit": args.e2_limit,
            "e3_per_cell_limit": args.e3_per_cell_limit,
            "probe_10": args.probe_10,
            "seed": args.seed,
            "stopped_by_timeout": stopped_by_timeout,
            "completed_e2": len(e2_recs),
            "completed_e3": len(e3_recs),
            "mrope_equivalence": mrope_info,
            "e2_inventory_audit": e2_inv,
            "e3_inventory_audit": e3_inv,
            **vram_summary,
            "wall_time_sec": wall_s,
            "jsonl_out": args.jsonl_out,
        },
        "vram_summary": vram_summary,
        "e2": summarize_e2_records(e2_recs, args.g1b_thresh, args.allow_rounding_exception),
        "e3": summarize_e3_records(e3_recs),
    }

    parent = os.path.dirname(os.path.abspath(args.json_out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary), f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    print("=" * 72, flush=True)
    print(
        f"Completed E2={len(e2_recs)}, E3={len(e3_recs)} in {wall_s:.2f}s (timeout={stopped_by_timeout}) | "
        f"max_peak_alloc={max_peak_alloc_gb:.3f}GB, max_peak_res={max_peak_res_gb:.3f}GB, "
        f"min_headroom={min_headroom_gb:.3f}/{total_vram_gb:.3f}GB",
        flush=True,
    )
    if e2_recs:
        e2s = summary["e2"]
        print(
            f"E2 (n={e2s['n_samples']}): G1a={e2s['g1a_vs_split']['pass_count']}/{e2s['n_samples']}, "
            f"G1b_strict={e2s['g1b_vs_oneshot']['strict_pass_count']}/{e2s['n_samples']} "
            f"(+rounding_exc={e2s['g1b_vs_oneshot']['rounding_exception_count']}, eff={e2s['g1b_vs_oneshot']['effective_pass_count']}/{e2s['n_samples']}), "
            f"J-local(scalar) exact={e2s['overall_conditions']['J_local']['exact_match_rate']*100:.2f}%, "
            f"old_leak={e2s['overall_conditions']['J_local']['old_leak_rate']*100:.2f}%, "
            f"mean_KL={e2s['overall_conditions']['J_local']['mean_drift_ans_kl']:.3e}, "
            f"G2_pass={e2s['g2_gate']['all_g2_pass']}",
            flush=True,
        )
    if e3_recs:
        e3s = summary["e3"]
        for cname, cdata in e3s["cells"].items():
            print(
                f"E3 {cname} (n={cdata['n_samples']}): worst_order={cdata['worst_order_all_edited_exact_rate']*100:.1f}%, "
                f"collat_probes={cdata['collateral']['n_unique_probes']}, "
                f"Jctrl={cdata['J_control']['all_edited_exact_rate']*100:.1f}%, "
                f"Jlrec={cdata['J_local_recompute']['all_edited_exact_rate']*100:.1f}%, pass={cdata['g3_cell_pass']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
