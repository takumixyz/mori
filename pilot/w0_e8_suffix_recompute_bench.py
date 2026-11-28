# -*- coding: utf-8 -*-
"""E8 Complete Steady-State Timing & True Suffix Recompute Benchmark for Qwen/Qwen3.5-9B on Colab L4.

Reference:
  - Board Message #23 from coordinator
  - refine-logs/STATEEDIT_J_EXPERIMENT_PLAN.md (§1.6, §2.3, §4 E8)
  - refine-logs/STATEEDIT_J_PROPOSAL.md (§3.2, §3.5, §5.4 G6)

Fixes & Enhancements:
  1. Complete steady-state J-local edit timing:
     - Captures g_by_layer during prefill via run_context_with_decay_rates.
     - Per-component breakdown (median of 5 runs after warmup, cuda.synchronize):
       * block_forward_ms: clone_cache(c_ckpt) + run_block_forward (len=73)
       * apply_jlocal_update_ms: fp64 state delta transport & injection
       * patch_kv_range_ms: [p0, b_deep) KV patching
       * clone_ctx_ms: clone_cache(c_ctx) at each T_end
       * half_chunk_128_replay_ms: CPU->GPU restore of c_ckpt + 128-token [c, p0) advance
     - Aggregated metrics:
       * jlocal_steady_inplace_ms: block_forward + apply_jlocal_update + patch_kv_range
       * jlocal_steady_cloned_ms: + clone_ctx
       * jlocal_steady_avg_offboundary_ms: + 128-tok replay
  2. Finely-bracketed suffix sweep:
     - L_suf in [128, 256, 320, 384, 512, 1024, 1500, 1980, 2500, 3500, 4000]
     - Locates exact T* crossover points for in-place, cloned, and off-boundary regimes.
  3. Clean output JSON E8_SUFFIX_RECOMPUTE_JSON and local file persistence.
"""
import argparse
import copy
import json
import os
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def sync_time(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def get_mem_stats(device):
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
        }
    return {
        "allocated_gb": round(torch.cuda.memory_allocated(device) / (1024**3), 4),
        "reserved_gb": round(torch.cuda.memory_reserved(device) / (1024**3), 4),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 4),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved(device) / (1024**3), 4),
    }


def get_base_model(model):
    return getattr(model, "model", model)


def make_position_ids(start, length, device, base_model, batch_size=1):
    pos_1d = torch.arange(start, start + length, device=device, dtype=torch.long)
    if hasattr(base_model, "language_model"):
        return pos_1d.view(1, 1, -1).expand(3, batch_size, -1)
    return pos_1d.view(1, 1, -1).expand(4, batch_size, -1)


def clone_cache(cache):
    return None if cache is None else copy.deepcopy(cache)


def cache_to_cpu(cache):
    if cache is None:
        return None
    c_cpu = copy.deepcopy(cache)
    for layer in c_cpu.layers:
        if hasattr(layer, "keys") and torch.is_tensor(layer.keys):
            layer.keys = layer.keys.cpu()
        if hasattr(layer, "values") and torch.is_tensor(layer.values):
            layer.values = layer.values.cpu()
        for name in ("recurrent_states", "conv_states"):
            d = getattr(layer, name, None)
            if isinstance(d, dict):
                for k in d:
                    if torch.is_tensor(d[k]):
                        d[k] = d[k].cpu()
    return c_cpu


def restore_cache_to_gpu(c_cpu, device):
    if c_cpu is None:
        return None
    c_gpu = copy.deepcopy(c_cpu)
    for layer in c_gpu.layers:
        if hasattr(layer, "keys") and torch.is_tensor(layer.keys):
            layer.keys = layer.keys.to(device)
        if hasattr(layer, "values") and torch.is_tensor(layer.values):
            layer.values = layer.values.to(device)
        for name in ("recurrent_states", "conv_states"):
            d = getattr(layer, name, None)
            if isinstance(d, dict):
                for k in d:
                    if torch.is_tensor(d[k]):
                        d[k] = d[k].to(device)
    return c_gpu


@torch.no_grad()
def run_model_segment(model, cache, ids_seg, pos_start, batch_size=1):
    seq_len = int(ids_seg.shape[1])
    if seq_len == 0:
        return cache
    base = get_base_model(model)
    device = ids_seg.device
    pos_ids = make_position_ids(pos_start, seq_len, device, base, batch_size=batch_size)
    attn_mask = torch.ones(batch_size, pos_start + seq_len, device=device, dtype=torch.long)
    cpos = torch.arange(pos_start, pos_start + seq_len, device=device, dtype=torch.long)
    out = base(
        input_ids=ids_seg,
        past_key_values=cache,
        attention_mask=attn_mask,
        position_ids=pos_ids,
        cache_position=cpos,
        use_cache=True,
    )
    return out.past_key_values


def snap_states(cache):
    s = {}
    for i, layer in enumerate(cache.layers):
        for name in ("recurrent_states", "conv_states"):
            d = getattr(layer, name, None)
            if isinstance(d, dict):
                for k, v in d.items():
                    if torch.is_tensor(v):
                        s[(i, name, k)] = v.detach().clone()
    return s


def patch_kv_range(cache, src, start_pos, end_pos):
    for lo, le in zip(cache.layers, src.layers):
        ko, vo = getattr(lo, "keys", None), getattr(lo, "values", None)
        ke, ve = getattr(le, "keys", None), getattr(le, "values", None)
        if torch.is_tensor(ko) and torch.is_tensor(ke) and ko.numel() > 0:
            ko[..., start_pos:end_pos, :] = ke[..., start_pos:end_pos, :]
            vo[..., start_pos:end_pos, :] = ve[..., start_pos:end_pos, :]


def expand_cache_batch(cache, batch_size=2):
    if cache is None:
        return None
    c_batched = copy.deepcopy(cache)
    for layer in c_batched.layers:
        ko, vo = getattr(layer, "keys", None), getattr(layer, "values", None)
        if torch.is_tensor(ko) and ko.numel() > 0:
            layer.keys = ko.repeat(batch_size, 1, 1, 1)
            layer.values = vo.repeat(batch_size, 1, 1, 1)
        for name in ("recurrent_states", "conv_states"):
            d = getattr(layer, name, None)
            if isinstance(d, dict):
                for k in d:
                    if torch.is_tensor(d[k]):
                        d[k] = d[k].repeat(batch_size, *([1] * (d[k].ndim - 1)))
    return c_batched


def compute_state_delta_fp64(s_old, s_new):
    assert set(s_old.keys()) == set(s_new.keys())
    return {k: s_new[k].double() - s_old[k].double() for k in s_old}


@torch.no_grad()
def run_block_forward(model, c_pre, ids_o, ids_e, p0, b_end):
    base = get_base_model(model)
    device = ids_o.device
    blk_len = b_end - p0
    try:
        ids_b2 = torch.cat([ids_o[:, p0:b_end], ids_e[:, p0:b_end]], dim=0)
        pos_ids = make_position_ids(p0, blk_len, device, base, batch_size=2)
        attn_mask = torch.ones(2, p0 + blk_len, device=device, dtype=torch.long)
        cpos = torch.arange(p0, p0 + blk_len, device=device, dtype=torch.long)

        c_batched = expand_cache_batch(c_pre, batch_size=2)
        out_b2 = base(
            input_ids=ids_b2,
            past_key_values=c_batched,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            cache_position=cpos,
            use_cache=True,
        )
        cache_out = out_b2.past_key_values
        s_old, s_new = {}, {}
        for i, layer in enumerate(cache_out.layers):
            for name in ("recurrent_states", "conv_states"):
                d = getattr(layer, name, None)
                if isinstance(d, dict):
                    for k, v in d.items():
                        if torch.is_tensor(v):
                            s_old[(i, name, k)] = v[0:1].detach().clone()
                            s_new[(i, name, k)] = v[1:2].detach().clone()

        c_new_blk = clone_cache(c_pre)
        for i, layer in enumerate(cache_out.layers):
            ko, vo = getattr(layer, "keys", None), getattr(layer, "values", None)
            if torch.is_tensor(ko) and ko.numel() > 0:
                c_new_blk.layers[i].keys = ko[1:2].detach().clone()
                c_new_blk.layers[i].values = vo[1:2].detach().clone()

        delta = compute_state_delta_fp64(s_old, s_new)
        return c_new_blk, delta
    except Exception:
        c_old_blk = run_model_segment(model, clone_cache(c_pre), ids_o[:, p0:b_end], p0)
        c_new_blk = run_model_segment(model, clone_cache(c_pre), ids_e[:, p0:b_end], p0)
        delta = compute_state_delta_fp64(snap_states(c_old_blk), snap_states(c_new_blk))
        return c_new_blk, delta


def apply_jlocal_update(cache_ctx, delta, g_by_layer, block_end, T_ctx, w_conv):
    shift = T_ctx - block_end
    for (i, name, k), d in delta.items():
        target = getattr(cache_ctx.layers[i], name)[k]
        d_d = d.double()
        if name == "recurrent_states":
            if i in g_by_layer:
                g_diff = g_by_layer[i][:, block_end:T_ctx, :].double().sum(dim=1)
                gamma = torch.exp(g_diff)
                eff_d = d_d * gamma[:, :, None, None]
            else:
                eff_d = d_d
            target.copy_((target.double() + eff_d).to(target.dtype))
        elif name == "conv_states":
            if block_end >= T_ctx - w_conv + 1 and 0 <= shift < w_conv:
                overlap = min(target.shape[-1], d_d.shape[-1]) - shift
                if overlap > 0:
                    target[..., :overlap].copy_(
                        (target[..., :overlap].double() + d_d[..., shift : shift + overlap]).to(target.dtype)
                    )


@torch.no_grad()
def run_context_with_decay_rates(model, ids_ctx):
    g_by_layer = {}
    hooks = []
    text_model = get_base_model(model)
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model

    def _make_hook(layer_mod):
        def _hook(_mod, _inp, a_out):
            g = -layer_mod.A_log.double().exp() * F.softplus(
                a_out.double() + layer_mod.dt_bias.double()
            )
            g_by_layer[int(layer_mod.layer_idx)] = g.detach()
        return _hook

    for mod in text_model.modules():
        if (
            hasattr(mod, "in_proj_a")
            and hasattr(mod, "A_log")
            and hasattr(mod, "dt_bias")
            and hasattr(mod, "layer_idx")
        ):
            hooks.append(mod.in_proj_a.register_forward_hook(_make_hook(mod)))

    try:
        c_ctx = run_model_segment(model, None, ids_ctx, 0)
    finally:
        for h in hooks:
            h.remove()
    return c_ctx, g_by_layer


def interpolate_crossover(sweep, metric_key):
    for i in range(len(sweep) - 1):
        r0 = sweep[i]
        r1 = sweep[i + 1]
        t_target_0 = r0[metric_key]
        t_target_1 = r1[metric_key]
        recomp_0 = r0["suffix_recompute_ms"]
        recomp_1 = r1["suffix_recompute_ms"]

        diff_0 = recomp_0 - t_target_0
        diff_1 = recomp_1 - t_target_1

        if diff_0 <= 0 and diff_1 >= 0:
            denom = (diff_1 - diff_0)
            frac = (-diff_0) / max(1e-6, denom)
            return round(r0["suffix_length"] + frac * (r1["suffix_length"] - r0["suffix_length"]), 1)
        elif diff_0 >= 0 and diff_1 <= 0:
            denom = (diff_0 - diff_1)
            frac = diff_0 / max(1e-6, denom)
            return round(r0["suffix_length"] + frac * (r1["suffix_length"] - r0["suffix_length"]), 1)

    if sweep:
        if sweep[0]["suffix_recompute_ms"] > sweep[0][metric_key]:
            return sweep[0]["suffix_length"]
        else:
            return f"> {sweep[-1]['suffix_length']}"
    return None


def main():
    ap = argparse.ArgumentParser(description="E8 Complete Steady-State Timing Benchmark.")
    ap.add_argument("model", nargs="?", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default="pilot/w0_e8_suffix_recompute_results.json")

    clean_argv = []
    for a in sys.argv[1:]:
        if a.endswith(".json") and ("kernel" in a or "runtime" in a):
            continue
        if a in ("-f", "--f"):
            continue
        clean_argv.append(a)

    args, _ = ap.parse_known_args(clean_argv)
    if os.environ.get("E8_MODEL"):
        args.model = os.environ.get("E8_MODEL")
    elif not args.model or "kernel" in str(args.model) or str(args.model).endswith(".json"):
        args.model = "Qwen/Qwen3.5-9B"

    if os.environ.get("E8_OUTPUT_JSON"):
        args.output_json = os.environ.get("E8_OUTPUT_JSON")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print("==================================================================", flush=True)
    print("=== E8 Complete Steady-State Timing & Suffix Recompute Benchmark ===", flush=True)
    print(f"Model: {args.model} | Device: {torch.cuda.get_device_name(device)}", flush=True)
    print("==================================================================", flush=True)

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    t_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", cfg)
    w_conv = int(getattr(t_cfg, "linear_conv_kernel_dim", getattr(t_cfg, "conv_kernel_dim", 4)))
    n_linear = sum(1 for x in getattr(t_cfg, "layer_types", []) if "linear" in str(x).lower()) or 24
    b_deep_offset = max(1, n_linear - 1) * (w_conv - 1)
    print(f"Model Config: n_linear={n_linear}, w_conv={w_conv} -> b_deep_offset=+{b_deep_offset}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="cuda", trust_remote_code=True
    ).eval()

    mem_load = get_mem_stats(device)
    print(f"Model loaded: allocated={mem_load['allocated_gb']:.2f} GB, reserved={mem_load['reserved_gb']:.2f} GB", flush=True)

    # Geometry:
    # Fact placed at p0 = 2048, b_deep = 2048 + 73 = 2121
    # Checkpoint at c = 1920 (chunk_size C = 256, half-chunk delta = 128 tokens)
    p0 = 2048
    b_deep = p0 + b_deep_offset
    c_ckpt_pos = p0 - 128  # 1920
    max_T = b_deep + 4000 + 64  # ~6185 tokens

    torch.manual_seed(42)
    ids_full = torch.randint(100, 2000, (1, max_T), device=device, dtype=torch.long)
    ids_new_fact = ids_full.clone()
    # Fact modification at [p0, p0+4)
    ids_new_fact[:, p0 : p0 + 4] = torch.tensor([[101, 102, 103, 104]], device=device, dtype=torch.long)

    # Step 1: Pre-build B-ckpt checkpoint at c = 1920 on CPU (host RAM) and GPU at p0 = 2048
    print(f"Pre-building B-ckpt at c={c_ckpt_pos} (CPU host storage) and p0={p0}...", flush=True)
    c_ckpt_1920_gpu = run_model_segment(model, None, ids_full[:, :c_ckpt_pos], 0)
    c_ckpt_1920_cpu = cache_to_cpu(c_ckpt_1920_gpu)

    c_ckpt_2048_gpu = run_model_segment(model, clone_cache(c_ckpt_1920_gpu), ids_full[:, c_ckpt_pos:p0], c_ckpt_pos)
    del c_ckpt_1920_gpu
    torch.cuda.empty_cache()

    # Step 2: Build base prefix cache [0, b_deep) for suffix_recompute
    print(f"Building prefix cache [0, b_deep={b_deep}) for suffix recompute...", flush=True)
    c_prefix_base = run_model_segment(model, clone_cache(c_ckpt_2048_gpu), ids_full[:, p0:b_deep], p0)

    # Step 3: Run full prefill with hooks to capture steady-state g_by_layer up to max_T
    print(f"Capturing steady-state decay rates g_by_layer up to T={max_T}...", flush=True)
    c_full_ctx, g_by_layer = run_context_with_decay_rates(model, ids_full[:, :max_T])
    print(f"Captured decay rates across {len(g_by_layer)} linear layers. Shape per layer: {list(g_by_layer.values())[0].shape}", flush=True)

    # Step 4: Component Benchmark (Median of 5 runs after warmup)
    print("\n--- Component Benchmark (Independent Component Latencies) ---", flush=True)

    # Component A: block_forward_ms (clone_cache(c_ckpt) + run_block_forward)
    print("Measuring block_forward_ms (clone c_ckpt + forward len=73)...", flush=True)
    # Warmup
    c_p = clone_cache(c_ckpt_2048_gpu)
    c_nb, delta = run_block_forward(model, c_p, ids_full, ids_new_fact, p0, b_deep)

    bf_times = []
    for _ in range(5):
        t0 = sync_time(device)
        c_p = clone_cache(c_ckpt_2048_gpu)
        c_nb, delta = run_block_forward(model, c_p, ids_full, ids_new_fact, p0, b_deep)
        bf_times.append((sync_time(device) - t0) * 1000.0)
    block_forward_ms = round(statistics.median(bf_times), 2)
    print(f"-> block_forward_ms: {block_forward_ms:.2f} ms", flush=True)

    # Component B: half_chunk_128_replay_ms (restore CPU c_ckpt_1920 + advance 128 tok [1920, 2048))
    print("Measuring half_chunk_128_replay_ms (CPU->GPU restore + 128-tok advance)...", flush=True)
    # Warmup
    c_gpu_warm = restore_cache_to_gpu(c_ckpt_1920_cpu, device)
    _ = run_model_segment(model, c_gpu_warm, ids_full[:, c_ckpt_pos:p0], c_ckpt_pos)

    replay_times = []
    for _ in range(5):
        t0 = sync_time(device)
        c_gpu = restore_cache_to_gpu(c_ckpt_1920_cpu, device)
        _ = run_model_segment(model, c_gpu, ids_full[:, c_ckpt_pos:p0], c_ckpt_pos)
        replay_times.append((sync_time(device) - t0) * 1000.0)
    half_chunk_128_replay_ms = round(statistics.median(replay_times), 2)
    print(f"-> half_chunk_128_replay_ms: {half_chunk_128_replay_ms:.2f} ms", flush=True)

    # Step 5: Finely Bracketed Suffix Sweep
    # L_suf in [128, 256, 320, 384, 512, 1024, 1500, 1980, 2500, 3500, 4000]
    suffix_lengths = [128, 256, 320, 384, 512, 1024, 1500, 1980, 2500, 3500, 4000]
    sweep_results = []

    print("\n--- Sweeping Suffix Lengths: Component Breakdown & Recompute Baselines ---", flush=True)
    for L_suf in suffix_lengths:
        T_end = b_deep + L_suf

        # Suffix cache at T_end
        # Build context cache at T_end by forwarding prefix + suffix
        # Or slice directly:
        c_ctx_T = clone_cache(c_prefix_base)
        c_ctx_T = run_model_segment(model, c_ctx_T, ids_full[:, b_deep:T_end], b_deep)

        # 1. clone_ctx_ms: clone_cache(c_ctx_T)
        c_clone_warm = clone_cache(c_ctx_T)
        clone_times = []
        for _ in range(5):
            t0 = sync_time(device)
            _ = clone_cache(c_ctx_T)
            clone_times.append((sync_time(device) - t0) * 1000.0)
        clone_ctx_ms = round(statistics.median(clone_times), 2)

        # 2. apply_jlocal_update_ms: fp64 state delta transport & injection
        # Warmup
        c_tmp = clone_cache(c_ctx_T)
        apply_jlocal_update(c_tmp, delta, g_by_layer, b_deep, T_end, w_conv)

        upd_times = []
        for _ in range(5):
            c_tmp = clone_cache(c_ctx_T)
            t0 = sync_time(device)
            apply_jlocal_update(c_tmp, delta, g_by_layer, b_deep, T_end, w_conv)
            upd_times.append((sync_time(device) - t0) * 1000.0)
        apply_jlocal_update_ms = round(statistics.median(upd_times), 2)

        # 3. patch_kv_range_ms: KV patching in [p0, b_deep)
        c_tmp = clone_cache(c_ctx_T)
        patch_kv_range(c_tmp, c_nb, p0, b_deep)

        kv_times = []
        for _ in range(5):
            c_tmp = clone_cache(c_ctx_T)
            t0 = sync_time(device)
            patch_kv_range(c_tmp, c_nb, p0, b_deep)
            kv_times.append((sync_time(device) - t0) * 1000.0)
        patch_kv_range_ms = round(statistics.median(kv_times), 2)

        # Combined steady-state J-local metrics:
        jlocal_steady_inplace_ms = round(block_forward_ms + apply_jlocal_update_ms + patch_kv_range_ms, 2)
        jlocal_steady_cloned_ms = round(jlocal_steady_inplace_ms + clone_ctx_ms, 2)
        jlocal_steady_avg_offboundary_ms = round(jlocal_steady_inplace_ms + half_chunk_128_replay_ms, 2)
        jlocal_steady_cloned_offboundary_ms = round(jlocal_steady_cloned_ms + half_chunk_128_replay_ms, 2)

        # 4. Suffix Recompute Baseline: freezing [0, b_deep), recomputing [b_deep, T_end)
        # Warmup
        c_sub = clone_cache(c_prefix_base)
        _ = run_model_segment(model, c_sub, ids_full[:, b_deep:T_end], b_deep)

        recompute_times = []
        for _ in range(5):
            t0 = sync_time(device)
            c_sub = clone_cache(c_prefix_base)
            _ = run_model_segment(model, c_sub, ids_full[:, b_deep:T_end], b_deep)
            recompute_times.append((sync_time(device) - t0) * 1000.0)
        suffix_recompute_ms = round(statistics.median(recompute_times), 2)

        # 5. Full Prefill Baseline: [0, T_end)
        # Warmup
        _ = run_model_segment(model, None, ids_full[:, :T_end], 0)

        prefill_times = []
        for _ in range(5):
            t0 = sync_time(device)
            _ = run_model_segment(model, None, ids_full[:, :T_end], 0)
            prefill_times.append((sync_time(device) - t0) * 1000.0)
        full_prefill_ms = round(statistics.median(prefill_times), 2)

        speedup_inplace = round(suffix_recompute_ms / max(1e-6, jlocal_steady_inplace_ms), 3)
        speedup_cloned = round(suffix_recompute_ms / max(1e-6, jlocal_steady_cloned_ms), 3)
        speedup_offboundary = round(suffix_recompute_ms / max(1e-6, jlocal_steady_avg_offboundary_ms), 3)
        speedup_full_prefill = round(full_prefill_ms / max(1e-6, jlocal_steady_inplace_ms), 3)

        print(
            f"L={L_suf:4d} (T={T_end:4d}): "
            f"Recompute={suffix_recompute_ms:7.2f}ms | "
            f"Inplace={jlocal_steady_inplace_ms:6.2f}ms ({speedup_inplace:5.2f}x) | "
            f"Cloned={jlocal_steady_cloned_ms:6.2f}ms ({speedup_cloned:5.2f}x) | "
            f"OffB={jlocal_steady_avg_offboundary_ms:6.2f}ms ({speedup_offboundary:5.2f}x) | "
            f"FullPrefill={full_prefill_ms:7.2f}ms ({speedup_full_prefill:5.2f}x) | "
            f"[comp: bf={block_forward_ms:.1f}, upd={apply_jlocal_update_ms:.2f}, kv={patch_kv_range_ms:.2f}, cl={clone_ctx_ms:.1f}, r128={half_chunk_128_replay_ms:.1f}]",
            flush=True,
        )

        sweep_results.append({
            "suffix_length": L_suf,
            "total_tokens": T_end,
            "suffix_recompute_ms": suffix_recompute_ms,
            "full_prefill_ms": full_prefill_ms,
            "components_ms": {
                "block_forward_ms": block_forward_ms,
                "apply_jlocal_update_ms": apply_jlocal_update_ms,
                "patch_kv_range_ms": patch_kv_range_ms,
                "clone_ctx_ms": clone_ctx_ms,
                "half_chunk_128_replay_ms": half_chunk_128_replay_ms,
            },
            "jlocal_steady_inplace_ms": jlocal_steady_inplace_ms,
            "jlocal_steady_cloned_ms": jlocal_steady_cloned_ms,
            "jlocal_steady_avg_offboundary_ms": jlocal_steady_avg_offboundary_ms,
            "jlocal_steady_cloned_offboundary_ms": jlocal_steady_cloned_offboundary_ms,
            "speedup_vs_suffix_recompute_inplace": speedup_inplace,
            "speedup_vs_suffix_recompute_cloned": speedup_cloned,
            "speedup_vs_suffix_recompute_offboundary": speedup_offboundary,
            "speedup_vs_full_prefill": speedup_full_prefill,
            "is_jlocal_inplace_faster": bool(jlocal_steady_inplace_ms < suffix_recompute_ms),
            "is_jlocal_cloned_faster": bool(jlocal_steady_cloned_ms < suffix_recompute_ms),
            "is_jlocal_offboundary_faster": bool(jlocal_steady_avg_offboundary_ms < suffix_recompute_ms),
        })

    # Crossover interpolations
    t_star_inplace = interpolate_crossover(sweep_results, "jlocal_steady_inplace_ms")
    t_star_cloned = interpolate_crossover(sweep_results, "jlocal_steady_cloned_ms")
    t_star_offboundary = interpolate_crossover(sweep_results, "jlocal_steady_avg_offboundary_ms")

    mem_final = get_mem_stats(device)
    summary = {
        "model": args.model,
        "device_name": torch.cuda.get_device_name(device),
        "total_vram_gb": round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 2),
        "b_deep": b_deep,
        "p0": p0,
        "c_ckpt_pos": c_ckpt_pos,
        "fixed_components_ms": {
            "block_forward_ms": block_forward_ms,
            "half_chunk_128_replay_ms": half_chunk_128_replay_ms,
        },
        "crossover_T_star_suffix_tokens": {
            "inplace": t_star_inplace,
            "cloned": t_star_cloned,
            "avg_offboundary": t_star_offboundary,
        },
        "memory_stats": mem_final,
        "sweep": sweep_results,
    }

    print("\n==================================================================", flush=True)
    print("=== E8 SUMMARY: COMPLETE STEADY-STATE TIMING & CROSSOVERS ===", flush=True)
    print(f"Fixed Components: block_forward = {block_forward_ms:.2f} ms | half_chunk_128_replay = {half_chunk_128_replay_ms:.2f} ms", flush=True)
    print(f"Crossover T* (Suffix Length): Inplace={t_star_inplace} tok | Cloned={t_star_cloned} tok | Avg-OffBoundary={t_star_offboundary} tok", flush=True)
    print(f"Peak VRAM Reserved: {mem_final['max_reserved_gb']:.2f} GB", flush=True)
    print("==================================================================\n", flush=True)

    json_str = json.dumps(summary, ensure_ascii=False)
    print("E8_SUFFIX_RECOMPUTE_JSON=" + json_str, flush=True)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(json_str + "\n")
        print(f"Saved results locally to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
