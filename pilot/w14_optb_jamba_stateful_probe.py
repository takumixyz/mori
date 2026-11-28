"""Option B Exp-4: Stateful Jamba-3B (Mamba-1 + Sparse NoPE Attention) Diagnostic & E2 Suite.

Tests whether resolving prior failures in Stage 4 allows clean third-family evaluation:
  1. Stateful Mamba-1 chunk forward wrapper (`install_stateful_jamba_patch`): feeds cached
     `conv_state` (left causal context) and `recurrent_state` (initial `ssm_state`) into
     `JambaMambaMixer.forward` when `seq_len > 1`, keeping `ssm_state` in `float32` across
     chunk seams (matching upstream `mamba_selective_scan` without `bf16` truncation).
  2. Captures true UNPATCHED native `JambaForCausalLM` single-pass logits BEFORE installing
     the monkey-patch, verifying both single-pass and chunked patched equivalence under strict
     `kl <= 1e-4` gating (`stateful_g1b_strict >= 0.80`).
  3. Uses tokenization-aligned prompt spacing (`"Answer: "` for `digit4`, `"Answer:"` for
     `single_token`) so `last_logits` scores the actual first answer token on 100% of cases,
     keeps `gate_cases` strictly disjoint from `e2_cases`, uses unified `--max-chunk` (default
     `256`) across oracle and `J-local`, and reports exact two-sided Wilcoxon p-values on both
     `ans_mean_kl` (primary) and `kl_vs_oracle`.

Supports `--synthetic-dry-run` on CPU (`CUDA_VISIBLE_DEVICES=""`).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HOME", r"D:\applicationData\HuggingFace")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
import transformers as _tf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache, JambaConfig, JambaForCausalLM
from transformers.models.jamba.modeling_jamba import (
    JambaMambaMixer,
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    causal_conv1d_update,
    mamba_selective_state_update,
)
from transformers.activations import ACT2FN

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from w2_crossfamily_e0c import (  # noqa: E402
    clone_cache,
    get_eos_ids,
    kl_div_from_logits,
    make_position_ids,
    overwrite_ssm_states,
    patch_kv_range,
)
from w2_crossfamily_e2_e3_e6 import (  # noqa: E402
    append_jsonl,
    eval_suffix_condition,
    load_completed_keys,
    mean_ans_kl,
    sync_time,
    wilson_ci,
)
from w11_crossfamily_scaleup import (  # noqa: E402
    CANDIDATE_WORDS,
    SCALED_NAMES,
    build_single_token_pairs,
    collect_shared_single_tokens,
    generate_disjoint_digit4_pairs,
)
from w12_c_third_family import (  # noqa: E402
    apply_third_family_jlocal_update,
    compute_state_delta_fp64,
    exact_mcnemar_p,
    extract_mixers,
    prepare_bckpt_cache,
    resolve_third_family_offset,
    run_segment,
    run_segment_with_jamba_decay,
    wilcoxon_p,
)


def stateful_mamba_selective_scan(
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    initial_state: torch.Tensor | None = None,
    return_last_state: bool = False,
):
    """Exact PyTorch Mamba-1 scan honoring initial `ssm_state` for multi-token chunks."""
    batch_size, intermediate_size, seq_len = hidden_states.shape
    input_dtype = hidden_states.dtype

    if delta_bias is not None:
        dt = dt + delta_bias.to(dt.dtype)[..., None]
    if delta_softplus:
        dt = F.softplus(dt)

    B = B.transpose(1, 2)
    C = C.transpose(1, 2)
    discrete_A = torch.exp(A[None, :, None, :] * dt[:, :, :, None])
    discrete_B = dt[:, :, :, None] * B[:, None, :, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    if initial_state is not None:
        ssm_state = initial_state.float().clone()
    else:
        ssm_state = torch.zeros(
            batch_size, intermediate_size, A.shape[-1],
            dtype=torch.float32, device=hidden_states.device,
        )

    scan_outputs = []
    for idx in range(seq_len):
        ssm_state = discrete_A[:, :, idx] * ssm_state + deltaB_u[:, :, idx]
        y_t = torch.matmul(ssm_state.to(input_dtype), C[:, idx, :].unsqueeze(-1))
        scan_outputs.append(y_t[:, :, 0])
    scan_output = torch.stack(scan_outputs, dim=-1)

    if D is not None:
        scan_output = scan_output + hidden_states * D[None, :, None]
    if z is not None:
        scan_output = scan_output * F.silu(z)
    if return_last_state:
        # Match upstream HF mamba_selective_scan: return float32 state so chunk seams avoid bf16 truncation.
        return scan_output, ssm_state
    return scan_output


def install_stateful_jamba_patch() -> None:
    """Monkey-patch `JambaMambaMixer.forward` to consume cached conv & recurrent states for any seq_len >= 1."""

    def _stateful_forward(
        self: JambaMambaMixer,
        hidden_states: torch.Tensor,
        cache_params: Any = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        seq_len = hidden_states.shape[1]
        dtype = hidden_states.dtype

        if cache_params is not None:
            conv_state = cache_params.get_conv_state(self.layer_idx, state_idx=0)
            recurrent_state = cache_params.get_recurrent_state(self.layer_idx, state_idx=0)
            has_rec_init = bool(
                cache_params.layers[self.layer_idx].is_recurrent_states_initialized.get(0, False)
                if hasattr(cache_params.layers[self.layer_idx], "is_recurrent_states_initialized")
                else (recurrent_state is not None)
            )
        else:
            conv_state, recurrent_state, has_rec_init = None, None, False

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        hidden_states_B_C, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation (matching upstream JambaMambaMixer.forward)
        if conv_state is not None and seq_len == 1 and not cache_params.is_recording_past(self.layer_idx):
            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
            )
        else:
            if cache_params is not None:
                hidden_states_B_C = cache_params.update_conv_state(
                    hidden_states_B_C,
                    self.layer_idx,
                    conv_kernel_size=self.conv_kernel_size,
                )
            hidden_states_B_C = causal_conv1d_fn(
                hidden_states_B_C,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )
            if cache_params is not None:
                hidden_states_B_C = hidden_states_B_C[:, :, -seq_len:]

        # 3. SSM projection + stateful scan
        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C.transpose(1, 2), attention_mask)
        time_step, B, C = torch.split(
            self.x_proj(hidden_states_B_C),
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)
        if hasattr(self.config, "_is_quantized"):
            time_step = (self.dt_proj(time_step) - self.dt_proj.bias).transpose(1, 2)
        else:
            time_step = self.dt_proj.weight @ time_step.transpose(1, 2)
        time_proj_bias = self.dt_proj.bias.float() if self.dt_proj.bias is not None else None

        if recurrent_state is not None and seq_len == 1:
            scan_output = mamba_selective_state_update(
                recurrent_state,
                hidden_states_B_C.transpose(1, 2)[..., 0],
                time_step[..., 0],
                A, B[:, 0], C[:, 0], self.D, z=gate[..., 0],
                dt_bias=time_proj_bias, dt_softplus=True,
            ).unsqueeze(-1)
        else:
            init_s = recurrent_state if (recurrent_state is not None and has_rec_init) else None
            scan_result = stateful_mamba_selective_scan(
                hidden_states_B_C.transpose(1, 2),
                time_step, A, B.transpose(1, 2), C.transpose(1, 2),
                D=self.D.float(), z=gate, delta_bias=time_proj_bias, delta_softplus=True,
                initial_state=init_s, return_last_state=(cache_params is not None),
            )
            if cache_params is not None:
                scan_output, final_state = scan_result
                cache_params.update_recurrent_state(final_state, self.layer_idx)
            else:
                scan_output = scan_result

        return self.out_proj(scan_output.transpose(1, 2).to(dtype))

    JambaMambaMixer.forward = _stateful_forward


def build_jamba_completion_cases(tok: Any, n_cases: int, pre_fillers: int = 20, post_fillers: int = 20) -> List[Dict[str, Any]]:
    """Build direct few-shot plain-completion cases avoiding `<think>` tag truncation."""
    d_pairs = generate_disjoint_digit4_pairs(n_cases + 24, seed=202688)
    s_words = collect_shared_single_tokens([tok])
    s_pairs = build_single_token_pairs(s_words, n_cases + 24) if len(s_words) >= 10 else []

    from w2_crossfamily_e0c import POST_FILLER_TEMPLATES, PRE_FILLER_TEMPLATES

    fewshot_prefix = (
        "Record: Mia has vault code 9142.\n"
        "Shift handoff note: nominal.\n"
        "Question: What is Mia's vault code? Answer with only 4 digits.\n"
        "Answer: 9142\n\n"
        "Record: Noah has security word amber.\n"
        "Shift handoff note: nominal.\n"
        "Question: What is Noah's security word? Answer with only 1 word.\n"
        "Answer: amber\n\n"
    )
    cases: List[Dict[str, Any]] = []
    pair_mix: List[Tuple[str, str, str]] = []
    for i in range(n_cases + 20):
        if i % 2 == 0 or not s_pairs:
            pair_mix.append((*d_pairs[i // 2], "digit4"))
        else:
            pair_mix.append((*s_pairs[i // 2], "single_token"))

    for idx, (old_v, new_v, fmt) in enumerate(pair_mix):
        if len(cases) >= n_cases:
            break
        name = SCALED_NAMES[idx % len(SCALED_NAMES)]
        pre_lines = [PRE_FILLER_TEMPLATES[k % len(PRE_FILLER_TEMPLATES)].format(idx=k + 1) for k in range(pre_fillers)]
        post_lines = [POST_FILLER_TEMPLATES[k % len(POST_FILLER_TEMPLATES)].format(idx=k + 1) for k in range(post_fillers)]
        kind = "vault code" if fmt == "digit4" else "security word"
        fact_o = f"Record: {name} has {kind} {old_v}."
        fact_e = f"Record: {name} has {kind} {new_v}."
        desc = "4 digits" if fmt == "digit4" else "1 word"
        # Jamba tokenizer splits 'Answer: 9142' as ['Answer', ':', '▁', '9', '1', '4', '2']
        # vs 'Answer: amber' as ['Answer', ':', '▁amber']; align prompt ending so logit[-1] predicts token 0.
        ans_anchor = "Answer: " if fmt == "digit4" else "Answer:"
        q_suf = f"Question: What is {name}'s {kind}? Answer with only {desc}.\n{ans_anchor}"

        ctx_o = fewshot_prefix + "\n".join(pre_lines + [fact_o] + post_lines) + "\n"
        ctx_e = fewshot_prefix + "\n".join(pre_lines + [fact_e] + post_lines) + "\n"
        full_o = ctx_o + q_suf
        full_e = ctx_e + q_suf

        enc_o = tok(full_o, add_special_tokens=False)
        enc_e = tok(full_e, add_special_tokens=False)
        enc_co = tok(ctx_o, add_special_tokens=False)
        enc_ce = tok(ctx_e, add_special_tokens=False)
        ids_o = enc_o["input_ids"]
        ids_e = enc_e["input_ids"]
        T_ctx = len(enc_co["input_ids"])
        if len(ids_o) != len(ids_e) or T_ctx != len(enc_ce["input_ids"]):
            continue
        if ids_o[:T_ctx] != enc_co["input_ids"] or ids_e[:T_ctx] != enc_ce["input_ids"] or ids_o[T_ctx:] != ids_e[T_ctx:]:
            continue
        diffs = [i for i in range(T_ctx) if ids_o[i] != ids_e[i]]
        if not diffs:
            continue
        p0, p1 = diffs[0], diffs[-1] + 1
        if ids_o[:p0] != ids_e[:p0] or ids_o[p1:T_ctx] != ids_e[p1:T_ctx]:
            continue
        cases.append({
            "ids_o": ids_o,
            "ids_n": ids_e,
            "p0": p0,
            "p1": p1,
            "T_ctx": T_ctx,
            "old": old_v,
            "new": new_v,
            "format": fmt,
        })
    return cases


def _prepare_bckpt_cache_chunked(
    model: Any,
    cfg: Any,
    ids_o: torch.LongTensor,
    p0: int,
    chunk_size: int = 256,
) -> Tuple[DynamicCache, int]:
    """Prepare B-ckpt cache honoring caller-specified `chunk_size` across all segments."""
    c_pre = DynamicCache(config=cfg)
    c_pos = (p0 // chunk_size) * chunk_size
    if c_pos > 0:
        _ = run_segment(model, c_pre, ids_o[:, :c_pos], 0, max_chunk=chunk_size)
    if p0 > c_pos:
        _ = run_segment(model, c_pre, ids_o[:, c_pos:p0], c_pos, max_chunk=chunk_size)
    return c_pre, c_pos


def exact_wilcoxon_p(a: List[float], b: List[float]) -> float:
    """Two-sided exact Wilcoxon signed-rank p-value (SciPy if installed, else exact integer DP)."""
    diffs = [float(x - y) for x, y in zip(a, b) if abs(float(x - y)) > 1e-18]
    n = len(diffs)
    if n == 0:
        return 1.0
    try:
        from scipy.stats import wilcoxon  # type: ignore

        res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="exact")
        return float(res.pvalue)
    except Exception:
        pass

    # Assign 1..n ranks with tie-averaging on doubled rank grid (all ranks * 2 are integers)
    indexed = sorted(enumerate(diffs), key=lambda kv: abs(kv[1]))
    doubled_ranks = [0] * n
    i = 0
    while i < n:
        j = i + 1
        val = abs(indexed[i][1])
        while j < n and abs(indexed[j][1]) == val:
            j += 1
        avg_r2 = (i + 1) + j  # 2 * ((i+1 + j) / 2)
        for k in range(i, j):
            doubled_ranks[indexed[k][0]] = avg_r2
        i = j

    w_plus2 = sum(r2 for d, r2 in zip(diffs, doubled_ranks) if d > 0.0)
    w_tot2 = sum(doubled_ranks)
    w_obs2 = min(w_plus2, w_tot2 - w_plus2)
    dp: Dict[int, int] = {0: 1}
    for r2 in doubled_ranks:
        nxt: Dict[int, int] = {}
        for s, cnt in dp.items():
            nxt[s] = nxt.get(s, 0) + cnt
            nxt[s + r2] = nxt.get(s + r2, 0) + cnt
        dp = nxt
    tail = sum(cnt for s, cnt in dp.items() if s <= w_obs2)
    return float(min(1.0, (2.0 * tail) / (2 ** n)))


def verify_stateful_chunk_equivalence(
    model: Any,
    cfg: Any,
    device: torch.device,
    cases: List[Dict[str, Any]],
    unpatched_refs: Optional[List[torch.Tensor]] = None,
    max_chunk: int = 256,
) -> Dict[str, Any]:
    """Compare patched single-pass & chunked forward against pre-patch native unpatched logits."""
    chk_vs_one_diffs, chk_vs_one_kls, chk_vs_one_t1 = [], [], []
    one_vs_nat_diffs, one_vs_nat_kls, one_vs_nat_t1 = [], [], []
    chk_vs_nat_diffs, chk_vs_nat_kls, chk_vs_nat_t1 = [], [], []
    effective_kls: List[float] = []
    effective_t1: List[bool] = []
    per_case: List[Dict[str, Any]] = []
    has_unpatched = unpatched_refs is not None and len(unpatched_refs) == len(cases)

    with torch.no_grad():
        for idx, c in enumerate(cases):
            ids = torch.tensor([c["ids_n"][: c["T_ctx"] + 4]], device=device, dtype=torch.long)
            split_pos = max(16, c["p0"])

            # Patched single pass (chunk=T_len >= len -> initial states zero at t=0 only)
            c_one = DynamicCache(config=cfg)
            lg_one = run_segment(model, c_one, ids, 0, max_chunk=max(2048, ids.shape[1] + 8))[:, -1:, :]

            # Patched two-chunk split across [0:split_pos] -> [split_pos:T_len] using unified max_chunk
            c_chk = DynamicCache(config=cfg)
            _ = run_segment(model, c_chk, ids[:, :split_pos], 0, max_chunk=max_chunk)
            lg_chk = run_segment(model, c_chk, ids[:, split_pos:], split_pos, max_chunk=max_chunk)[:, -1:, :]

            d_chk_one = float((lg_one.float() - lg_chk.float()).abs().max().item())
            k_chk_one = kl_div_from_logits(lg_one, lg_chk)
            t_chk_one = bool(int(lg_one.argmax(-1).item()) == int(lg_chk.argmax(-1).item()))
            chk_vs_one_diffs.append(d_chk_one)
            chk_vs_one_kls.append(k_chk_one)
            chk_vs_one_t1.append(t_chk_one)

            top5_one = torch.topk(lg_one[0, 0].float(), k=min(5, int(lg_one.shape[-1])))
            top5_chk = torch.topk(lg_chk[0, 0].float(), k=min(5, int(lg_chk.shape[-1])))
            rec_c: Dict[str, Any] = {
                "case_idx": int(idx),
                "seq_len": int(ids.shape[1]),
                "split_pos": int(split_pos),
                "chk_vs_one_kl": float(k_chk_one),
                "chk_vs_one_max_logit_diff": float(d_chk_one),
                "top1_one_id": int(lg_one.argmax(-1).item()),
                "top1_chk_id": int(lg_chk.argmax(-1).item()),
                "top5_one_ids": [int(x) for x in top5_one.indices.tolist()],
                "top5_one_logits": [round(float(x), 6) for x in top5_one.values.tolist()],
                "top5_chk_ids": [int(x) for x in top5_chk.indices.tolist()],
                "top5_chk_logits": [round(float(x), 6) for x in top5_chk.values.tolist()],
            }

            if has_unpatched:
                lg_nat = unpatched_refs[idx].to(device=device)
                d_one_nat = float((lg_nat.float() - lg_one.float()).abs().max().item())
                k_one_nat = kl_div_from_logits(lg_nat, lg_one)
                t_one_nat = bool(int(lg_nat.argmax(-1).item()) == int(lg_one.argmax(-1).item()))
                one_vs_nat_diffs.append(d_one_nat)
                one_vs_nat_kls.append(k_one_nat)
                one_vs_nat_t1.append(t_one_nat)

                d_chk_nat = float((lg_nat.float() - lg_chk.float()).abs().max().item())
                k_chk_nat = kl_div_from_logits(lg_nat, lg_chk)
                t_chk_nat = bool(int(lg_nat.argmax(-1).item()) == int(lg_chk.argmax(-1).item()))
                chk_vs_nat_diffs.append(d_chk_nat)
                chk_vs_nat_kls.append(k_chk_nat)
                chk_vs_nat_t1.append(t_chk_nat)

                effective_kls.append(max(k_chk_one, k_chk_nat))
                effective_t1.append(t_chk_one and t_chk_nat and t_one_nat)
                top5_nat = torch.topk(lg_nat[0, 0].float(), k=min(5, int(lg_nat.shape[-1])))
                rec_c.update({
                    "pat_one_vs_unpatched_kl": float(k_one_nat),
                    "pat_one_vs_unpatched_max_logit_diff": float(d_one_nat),
                    "pat_chk_vs_unpatched_kl": float(k_chk_nat),
                    "pat_chk_vs_unpatched_max_logit_diff": float(d_chk_nat),
                    "top1_nat_id": int(lg_nat.argmax(-1).item()),
                    "top5_nat_ids": [int(x) for x in top5_nat.indices.tolist()],
                    "top5_nat_logits": [round(float(x), 6) for x in top5_nat.values.tolist()],
                })
            else:
                effective_kls.append(k_chk_one)
                effective_t1.append(t_chk_one)
            per_case.append(rec_c)
            del c_one, c_chk

    out: Dict[str, Any] = {
        "n": len(cases),
        "max_chunk": int(max_chunk),
        "vs_unpatched_available": bool(has_unpatched),
        "all_top1_eq": bool(all(effective_t1)),
        "strict_kl_le_1e4": sum(1 for x in effective_kls if x <= 1e-4),
        "mean_kl": sum(effective_kls) / max(1, len(effective_kls)),
        "max_kl": max(effective_kls, default=0.0),
        "max_logit_diff": max(chk_vs_nat_diffs if has_unpatched else chk_vs_one_diffs, default=0.0),
        "chk_vs_one_all_top1_eq": bool(all(chk_vs_one_t1)),
        "chk_vs_one_strict_kl_le_1e4": sum(1 for x in chk_vs_one_kls if x <= 1e-4),
        "chk_vs_one_mean_kl": sum(chk_vs_one_kls) / max(1, len(chk_vs_one_kls)),
        "chk_vs_one_max_kl": max(chk_vs_one_kls, default=0.0),
        "chk_vs_one_max_logit_diff": max(chk_vs_one_diffs, default=0.0),
        "per_case": per_case,
    }
    if has_unpatched:
        out.update({
            "pat_one_vs_unpatched_all_top1_eq": bool(all(one_vs_nat_t1)),
            "pat_one_vs_unpatched_strict_kl_le_1e4": sum(1 for x in one_vs_nat_kls if x <= 1e-4),
            "pat_one_vs_unpatched_mean_kl": sum(one_vs_nat_kls) / max(1, len(one_vs_nat_kls)),
            "pat_one_vs_unpatched_max_kl": max(one_vs_nat_kls, default=0.0),
            "pat_one_vs_unpatched_max_logit_diff": max(one_vs_nat_diffs, default=0.0),
            "pat_chk_vs_unpatched_all_top1_eq": bool(all(chk_vs_nat_t1)),
            "pat_chk_vs_unpatched_strict_kl_le_1e4": sum(1 for x in chk_vs_nat_kls if x <= 1e-4),
            "pat_chk_vs_unpatched_mean_kl": sum(chk_vs_nat_kls) / max(1, len(chk_vs_nat_kls)),
            "pat_chk_vs_unpatched_max_kl": max(chk_vs_nat_kls, default=0.0),
            "pat_chk_vs_unpatched_max_logit_diff": max(chk_vs_nat_diffs, default=0.0),
        })
    return out


def run_synthetic_28l_fp32_control(out_dir: Path, device_str: str = "cuda:0") -> Path:
    """Run executable 4L & 28L scaled-width JambaConfig precision control with identical initial weights."""
    import hashlib as _hashlib

    orig_forward = JambaMambaMixer.forward
    dev = torch.device(device_str if (device_str.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    try:
        cases = [
            {"ids_n": [(i * 17 + k) % 300 + 10 for k in range(380)], "p0": 340, "T_ctx": 376}
            for i in range(3)
        ]
        results_by_depth_and_dtype: Dict[str, Any] = {}
        weight_fingerprints: Dict[str, str] = {}
        # Hold seed=202614 constant across all configs; include matched-layout (period=4, offset=2) 4L vs 28L AND Jamba-3B layout 28L (period=14, offset=6)
        configs = [
            ("4L_p4_o2", 4, 4, 2, 202614),
            ("28L_p4_o2_matched", 28, 4, 2, 202614),
            ("28L_p14_o6_jamba3b_layout", 28, 14, 6, 202614),
        ]
        for depth_key, n_layers, period, offset, seed_val in configs:
            cfg = JambaConfig(
                vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=n_layers,
                num_attention_heads=4, num_key_value_heads=2, mamba_d_conv=4, mamba_d_state=16,
                mamba_expand=2, mamba_dt_rank=8, attn_layer_period=period, attn_layer_offset=offset, num_experts=1,
            )
            torch.manual_seed(seed_val)
            base_model_cpu = JambaForCausalLM(cfg).to(dtype=torch.float32).eval()
            base_sd = {k: v.detach().cpu().clone() for k, v in base_model_cpu.state_dict().items()}
            h = _hashlib.sha256()
            for k in sorted(base_sd.keys()):
                h.update(k.encode("utf-8"))
                h.update(base_sd[k].numpy().tobytes())
            weight_fingerprints[depth_key] = h.hexdigest()[:16]
            del base_model_cpu

            dt_res: Dict[str, Any] = {}
            for dt_label, dt_val in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
                JambaMambaMixer.forward = orig_forward
                model = JambaForCausalLM(cfg)
                model.load_state_dict(base_sd, strict=True)
                model = model.to(device=dev, dtype=dt_val).eval()
                unpatched_refs: List[torch.Tensor] = []
                with torch.no_grad():
                    for c in cases:
                        ids = torch.tensor([c["ids_n"]], device=dev, dtype=torch.long)
                        c_nat = DynamicCache(config=cfg)
                        lg_nat = run_segment(model, c_nat, ids, 0, max_chunk=2048)[:, -1:, :].detach()
                        unpatched_refs.append(lg_nat)
                        del c_nat
                install_stateful_jamba_patch()
                eq = verify_stateful_chunk_equivalence(model, cfg, dev, cases, unpatched_refs=unpatched_refs, max_chunk=256)
                dt_res[dt_label] = eq
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results_by_depth_and_dtype[depth_key] = {
                "num_hidden_layers": n_layers,
                "attn_layer_period": period,
                "attn_layer_offset": offset,
                "seed": seed_val,
                "same_initial_weights_across_dtypes": True,
                "base_fp32_weight_sha256_prefix": weight_fingerprints[depth_key],
                "results_by_dtype": dt_res,
            }

        out_p = out_dir / "w14_optb_jamba_28l_fp32_control_summary.json"
        payload = {
            "runner": "pilot/w14_optb_jamba_stateful_probe.py --synthetic-28l-fp32-control",
            "scope_note": (
                "Scaled-width synthetic JambaConfig chunk-seam precision control (hidden_size=64, split=340, max_chunk=256) "
                "holding seed=202614 constant across matched-layout 4L vs 28L (period=4, offset=2) and Jamba-3B-layout 28L (period=14, offset=6), "
                "loading identical float32 weights before casting to bfloat16 within each architecture. Provides qualitative precision-scaling control only (not a quantitative match to 3B real-weight magnitude)."
            ),
            "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "os_name": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": _tf.__version__,
            "num_hidden_layers": 28,
            "attn_layer_period": 14,
            "attn_layer_offset": 6,
            "num_experts": 1,
            "split_pos": 340,
            "max_chunk": 256,
            "same_initial_weights_across_dtypes": True,
            "base_fp32_weight_sha256_prefix_28l": weight_fingerprints["28L_p14_o6_jamba3b_layout"],
            "results_by_dtype": results_by_depth_and_dtype["28L_p14_o6_jamba3b_layout"]["results_by_dtype"],
            "results_by_depth_and_dtype": results_by_depth_and_dtype,
        }
        out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[28L CONTROL PASS] Wrote {out_p}", flush=True)
        return out_p
    finally:
        JambaMambaMixer.forward = orig_forward


def run_synthetic_dry_run(out_dir: Path) -> None:
    orig_forward = JambaMambaMixer.forward
    try:
        cfg = JambaConfig(
            vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, mamba_d_conv=4, mamba_d_state=8,
            mamba_expand=2, mamba_dt_rank=8, attn_layer_period=4, attn_layer_offset=2, num_experts=1,
        )
        model = JambaForCausalLM(cfg).eval()
        cases = [{
            "ids_o": list(range(10, 90)),
            "ids_n": list(range(10, 30)) + [201, 202] + list(range(32, 90)),
            "p0": 20, "p1": 22, "T_ctx": 70, "old": "a", "new": "b", "format": "single_token",
        }]
        dev = torch.device("cpu")
        with torch.no_grad():
            ids_nat = torch.tensor([cases[0]["ids_n"][: cases[0]["T_ctx"] + 4]], device=dev, dtype=torch.long)
            c_nat = DynamicCache(config=cfg)
            lg_nat = run_segment(model, c_nat, ids_nat, 0, max_chunk=2048)[:, -1:, :].detach()
            del c_nat

        install_stateful_jamba_patch()
        eq = verify_stateful_chunk_equivalence(model, cfg, dev, cases, unpatched_refs=[lg_nat], max_chunk=16)
        assert eq["vs_unpatched_available"] is True
        assert eq["strict_kl_le_1e4"] == 1 and eq["max_logit_diff"] <= 1e-4, f"Stateful patch mismatch: {eq}"
        assert eq["pat_one_vs_unpatched_max_logit_diff"] <= 1e-4 and eq["pat_chk_vs_unpatched_max_logit_diff"] <= 1e-4

        # Verify fp32 state preservation under bf16 input dtype
        h_bf16 = torch.randn(1, 16, 4, dtype=torch.bfloat16)
        dt_bf16 = torch.randn(1, 16, 4, dtype=torch.bfloat16)
        A_fp32 = -torch.exp(torch.zeros(16, 8, dtype=torch.float32))
        B_bf16 = torch.randn(1, 8, 4, dtype=torch.bfloat16)
        C_bf16 = torch.randn(1, 8, 4, dtype=torch.bfloat16)
        scan_out, s_last = stateful_mamba_selective_scan(
            h_bf16, dt_bf16, A_fp32, B_bf16, C_bf16, delta_softplus=True, return_last_state=True
        )
        assert s_last.dtype == torch.float32, f"Expected float32 ssm_state, got {s_last.dtype}"

        # Verify exact_wilcoxon_p DP sanity (n=5 -> 1/16)
        p_test = exact_wilcoxon_p([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0])
        assert abs(p_test - 0.0625) < 1e-12, f"Unexpected exact_wilcoxon_p: {p_test}"
        print(f"[DRY-RUN PASS] w14_optb_jamba_stateful_probe.py unpatched->patched + fp32 seam + exact wilcoxon verified: {eq}", flush=True)
    finally:
        JambaMambaMixer.forward = orig_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default=str(SCRIPT_DIR))
    parser.add_argument("--n-gate", type=int, default=5)
    parser.add_argument("--n-e2", type=int, default=30)
    parser.add_argument("--max-chunk", type=int, default=256)
    parser.add_argument("--clean-out", action="store_true", help="Remove prior E2 JSONL output before running")
    parser.add_argument("--real-fp32-equiv-check", action="store_true", help="After bf16 check, verify real Jamba-3B weights in fp32 on CUDA (~12 GB)")
    parser.add_argument("--synthetic-28l-fp32-control", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--synthetic-dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.synthetic_dry_run:
        run_synthetic_dry_run(out_dir)
        return
    if args.synthetic_28l_fp32_control:
        run_synthetic_28l_fp32_control(out_dir, args.device)
        return

    repo_id = "ai21labs/AI21-Jamba-Reasoning-3B"
    dev = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    ).eval()
    eos_ids = get_eos_ids(model, tok)
    mixers = extract_mixers(model, "jamba")
    w_conv = int(getattr(cfg, "mamba_d_conv", 4))

    total_n = 4 if args.smoke else args.n_e2
    # Build disjoint gate and evaluation cohorts upfront
    cases = build_jamba_completion_cases(tok, n_cases=args.n_gate + total_n + 12, pre_fillers=18, post_fillers=18)
    assert len(cases) >= args.n_gate + total_n, f"Insufficient Jamba cases: {len(cases)} < {args.n_gate + total_n}"
    gate_cases = cases[: args.n_gate]
    run_cases = cases[args.n_gate : args.n_gate + total_n]
    gate_sigs = {tuple(c["ids_n"]) for c in gate_cases}
    run_sigs = {tuple(c["ids_n"]) for c in run_cases}
    assert gate_sigs.isdisjoint(run_sigs), "Gate and E2 cohorts must be strictly disjoint"
    eq_cases = gate_cases[: min(3, len(gate_cases))]

    # 1a. Capture native UNPATCHED single-pass reference logits before installing monkey-patch
    orig_jamba_fwd = JambaMambaMixer.forward
    unpatched_refs: List[torch.Tensor] = []
    with torch.no_grad():
        for c in eq_cases:
            ids = torch.tensor([c["ids_n"][: c["T_ctx"] + 4]], device=dev, dtype=torch.long)
            c_nat = DynamicCache(config=cfg)
            lg_nat = run_segment(model, c_nat, ids, 0, max_chunk=max(2048, ids.shape[1] + 8))[:, -1:, :].detach()
            unpatched_refs.append(lg_nat)
            del c_nat

    # 1b. Install stateful patch & verify against BOTH native unpatched single-pass AND patched single-pass
    install_stateful_jamba_patch()
    eq_res = verify_stateful_chunk_equivalence(
        model, cfg, dev, eq_cases, unpatched_refs=unpatched_refs, max_chunk=args.max_chunk
    )
    print("[Jamba Stateful Chunk Check]", json.dumps(eq_res), flush=True)

    # 2. Oracle prescreen on clean completion prompt (strictly disjoint gate cohort)
    ora_hits = 0
    ora_previews = []
    with torch.no_grad():
        for c in gate_cases:
            c_o = DynamicCache(config=cfg)
            ids_e = torch.tensor([c["ids_n"]], device=dev, dtype=torch.long)
            _ = run_segment(model, c_o, ids_e[:, : c["T_ctx"]], 0, max_chunk=args.max_chunk)
            ev = eval_suffix_condition(model, tok, c_o, ids_e[:, c["T_ctx"]:], c["T_ctx"], c["new"], c["old"], c["format"], 8, eos_ids)
            ora_hits += int(ev["acc_new"])
            ora_previews.append({"acc_new": bool(ev["acc_new"]), "new": c["new"], "format": c["format"], "gen_text": ev.get("gen_text", "")})
            del c_o
    ora_rate = ora_hits / max(1, len(gate_cases))
    print(f"[Jamba Gate Oracle] {ora_hits}/{len(gate_cases)} ({ora_rate:.3f})", flush=True)

    eq_strict_rate = eq_res["strict_kl_le_1e4"] / max(1, eq_res["n"])
    gate_ok = bool(eq_res["all_top1_eq"] and eq_strict_rate >= 0.80 and ora_rate >= 0.80)
    e2_summary: Dict[str, Any] = {}
    records_path = out_dir / "w14_optb_jamba3b_e2_records.jsonl"
    if args.clean_out:
        records_path.unlink(missing_ok=True)

    if gate_ok:
        done = load_completed_keys(records_path)
        b_offset = resolve_third_family_offset("jamba", len(mixers), w_conv, "deep")
        conds = ["old", "oracle", "J-local", "kv_only_deep", "kv_only_span_p1", "J-local-evict", "J-control", "J-local-stale", "J-local-recompute", "state_only"]

        for idx, case in enumerate(run_cases):
            rk = f"jamba3b_stateful:e2:{idx}"
            if rk in done:
                continue
            ids_o = torch.tensor([case["ids_o"]], device=dev, dtype=torch.long)
            ids_e = torch.tensor([case["ids_n"]], device=dev, dtype=torch.long)
            p0, p1, T_ctx = case["p0"], case["p1"], case["T_ctx"]
            b_end = min(T_ctx, p1 + b_offset)
            ids_suf = ids_e[:, T_ctx:]

            with torch.no_grad():
                c_pre, _ = _prepare_bckpt_cache_chunked(model, cfg, ids_o, p0, chunk_size=args.max_chunk)
                c_old_b = clone_cache(c_pre)
                _ = run_segment(model, c_old_b, ids_o[:, p0:b_end], p0, max_chunk=args.max_chunk)
                c_old_ctx = clone_cache(c_old_b)
                _, g_sums = run_segment_with_jamba_decay(model, c_old_ctx, ids_o[:, b_end:T_ctx], b_end, mixers, max_chunk=args.max_chunk)

                t0 = sync_time(dev)
                c_new_b = clone_cache(c_pre)
                _ = run_segment(model, c_new_b, ids_e[:, p0:b_end], p0, max_chunk=args.max_chunk)
                delta_b = compute_state_delta_fp64(c_old_b, c_new_b)
                c_jlocal = apply_third_family_jlocal_update(
                    c_old_ctx, c_new_b, delta_b, g_sums, p0, b_end, T_ctx, w_conv, True, True, True, True
                )
                t_jl_ms = (sync_time(dev) - t0) * 1000.0

                c_oracle = DynamicCache(config=cfg)
                _ = run_segment(model, c_oracle, ids_e[:, :T_ctx], 0, max_chunk=args.max_chunk)
                t0s = sync_time(dev)
                c_suf = clone_cache(c_pre)
                _ = run_segment(model, c_suf, ids_e[:, p0:T_ctx], p0, max_chunk=args.max_chunk)
                t_suf_ms = (sync_time(dev) - t0s) * 1000.0
                del c_suf

                c_kv_deep = apply_third_family_jlocal_update(c_old_ctx, c_new_b, delta_b, g_sums, p0, b_end, T_ctx, w_conv, True, True, False, True)
                c_np1 = clone_cache(c_pre)
                _ = run_segment(model, c_np1, ids_e[:, p0:p1], p0, max_chunk=args.max_chunk)
                c_kv_p1 = apply_third_family_jlocal_update(c_old_ctx, c_np1, {}, {}, p0, p1, T_ctx, w_conv, False, False, False, True)
                del c_np1
                c_ctrl = clone_cache(c_old_b)
                overwrite_ssm_states(c_ctrl, c_new_b)
                if T_ctx > b_end:
                    _ = run_segment(model, c_ctrl, ids_o[:, b_end:T_ctx], b_end, max_chunk=args.max_chunk)
                c_evict_b = apply_third_family_jlocal_update(c_old_ctx, c_new_b, delta_b, g_sums, p0, b_end, T_ctx, w_conv, True, True, True, False)
                c_stale = apply_third_family_jlocal_update(c_old_ctx, c_new_b, delta_b, g_sums, p0, b_end, T_ctx, w_conv, True, False, True, True)
                c_so = apply_third_family_jlocal_update(c_old_ctx, c_new_b, delta_b, g_sums, p0, b_end, T_ctx, w_conv, True, True, True, False)
                c_recomp = clone_cache(c_new_b)
                if T_ctx > b_end:
                    _ = run_segment(model, c_recomp, ids_o[:, b_end:T_ctx], b_end, max_chunk=args.max_chunk)

                cond_map = {
                    "old": (c_old_ctx, None), "oracle": (c_oracle, None), "J-local": (c_jlocal, None),
                    "kv_only_deep": (c_kv_deep, None), "kv_only_span_p1": (c_kv_p1, None),
                    "J-local-evict": (c_evict_b, (p0, p1)), "J-control": (c_ctrl, None),
                    "J-local-stale": (c_stale, None), "J-local-recompute": (c_recomp, None), "state_only": (c_so, None),
                }
                evals = {
                    cn: eval_suffix_condition(model, tok, cc, ids_suf, T_ctx, case["new"], case["old"], case["format"], 8, eos_ids, ev_sp)
                    for cn, (cc, ev_sp) in cond_map.items()
                }
            lg_ora = evals["oracle"]["last_logits"]
            ans_ora = evals["oracle"]["ans_logits_new"]
            out_c = {}
            for cn in conds:
                ev = evals[cn]
                out_c[cn] = {
                    "acc_new": bool(ev["acc_new"]),
                    "acc_old": bool(ev["acc_old"]),
                    "top1_eq_oracle": bool(ev["top1_id"] == evals["oracle"]["top1_id"]),
                    "kl_vs_oracle": float(kl_div_from_logits(lg_ora, ev["last_logits"])),
                    "ans_mean_kl": float(mean_ans_kl(ans_ora, ev["ans_logits_new"])),
                    "gen_text": ev.get("gen_text", ""),
                }
            append_jsonl(records_path, {
                "record_key": rk, "sample_idx": idx, "format": case["format"],
                "T_ctx": T_ctx, "p0": p0, "b_end": b_end,
                "jlocal_ms": t_jl_ms, "suffix_recompute_ms": t_suf_ms, "conditions": out_c,
            })
            done.add(rk)
            del c_pre, c_old_b, c_old_ctx, c_new_b, c_jlocal, c_oracle, c_kv_deep, c_kv_p1, c_ctrl, c_evict_b, c_stale, c_so, c_recomp

        all_rows = [json.loads(l) for l in records_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        n_r = len(all_rows)
        cond_agg: Dict[str, Any] = {}
        for cn in conds:
            k_new = sum(1 for r in all_rows if r["conditions"][cn]["acc_new"])
            k_old = sum(1 for r in all_rows if r["conditions"][cn]["acc_old"])
            k_t1 = sum(1 for r in all_rows if r["conditions"][cn]["top1_eq_oracle"])
            cond_agg[cn] = {
                "n": n_r,
                "contains_new_rate": k_new / n_r,
                "contains_new_ci95": wilson_ci(k_new, n_r),
                "acc_old_rate": k_old / n_r,
                "top1_eq_oracle_rate": k_t1 / n_r,
                "mean_kl_vs_oracle": sum(r["conditions"][cn]["kl_vs_oracle"] for r in all_rows) / n_r,
                "mean_ans_kl": sum(r["conditions"][cn]["ans_mean_kl"] for r in all_rows) / n_r,
            }
        # Exact McNemar J-local vs kv_only_deep & exact two-sided Wilcoxon on ans_mean_kl (primary) & kl_vs_oracle
        b_fl = sum(1 for r in all_rows if r["conditions"]["J-local"]["acc_new"] and not r["conditions"]["kv_only_deep"]["acc_new"])
        c_fl = sum(1 for r in all_rows if not r["conditions"]["J-local"]["acc_new"] and r["conditions"]["kv_only_deep"]["acc_new"])
        by_fmt: Dict[str, Any] = {}
        for fmt_key in ("digit4", "single_token"):
            sub = [r for r in all_rows if r.get("format") == fmt_key]
            if sub:
                by_fmt[fmt_key] = {
                    "n": len(sub),
                    "jlocal_mean_ans_kl": sum(r["conditions"]["J-local"]["ans_mean_kl"] for r in sub) / len(sub),
                    "stale_mean_ans_kl": sum(r["conditions"]["J-local-stale"]["ans_mean_kl"] for r in sub) / len(sub),
                    "ans_kl_wilcoxon_exact_p": exact_wilcoxon_p(
                        [r["conditions"]["J-local"]["ans_mean_kl"] for r in sub],
                        [r["conditions"]["J-local-stale"]["ans_mean_kl"] for r in sub],
                    ),
                    "kl_wilcoxon_exact_p": exact_wilcoxon_p(
                        [r["conditions"]["J-local"]["kl_vs_oracle"] for r in sub],
                        [r["conditions"]["J-local-stale"]["kl_vs_oracle"] for r in sub],
                    ),
                }
        kl_exact_p = exact_wilcoxon_p(
            [r["conditions"]["J-local"]["kl_vs_oracle"] for r in all_rows],
            [r["conditions"]["J-local-stale"]["kl_vs_oracle"] for r in all_rows],
        )
        ans_exact_p = exact_wilcoxon_p(
            [r["conditions"]["J-local"]["ans_mean_kl"] for r in all_rows],
            [r["conditions"]["J-local-stale"]["ans_mean_kl"] for r in all_rows],
        )
        e2_summary = {
            "n": n_r,
            "max_chunk": int(args.max_chunk),
            "mean_jlocal_ms": sum(r["jlocal_ms"] for r in all_rows) / n_r,
            "mean_suffix_recompute_ms": sum(r["suffix_recompute_ms"] for r in all_rows) / n_r,
            "jlocal_vs_kv_only_deep_mcnemar": {"wins_jlocal": b_fl, "wins_kv_only": c_fl, "p": exact_mcnemar_p(b_fl, c_fl)},
            "jlocal_vs_stale_kl_wilcoxon_p": kl_exact_p,
            "jlocal_vs_stale_kl_wilcoxon_exact_p": kl_exact_p,
            "jlocal_vs_stale_ans_kl_wilcoxon_exact_p": ans_exact_p,
            "by_format": by_fmt,
            "conditions": cond_agg,
        }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    real_fp32_check: Optional[Dict[str, Any]] = None
    if args.real_fp32_equiv_check:
        # Restore exact pre-patch JambaMambaMixer.forward (NO importlib.reload, which would create a second class!)
        try:
            JambaMambaMixer.forward = orig_jamba_fwd
            model_fp32 = AutoModelForCausalLM.from_pretrained(
                repo_id, torch_dtype=torch.float32, device_map=args.device, trust_remote_code=True
            ).eval()
            unpatched_fp32: List[torch.Tensor] = []
            with torch.no_grad():
                for c in eq_cases:
                    ids = torch.tensor([c["ids_n"][: c["T_ctx"] + 4]], device=dev, dtype=torch.long)
                    c_nat = DynamicCache(config=cfg)
                    lg_nat = run_segment(model_fp32, c_nat, ids, 0, max_chunk=max(2048, ids.shape[1] + 8))[:, -1:, :].detach()
                    unpatched_fp32.append(lg_nat)
                    del c_nat
            install_stateful_jamba_patch()
            real_fp32_check = verify_stateful_chunk_equivalence(
                model_fp32, cfg, dev, eq_cases, unpatched_refs=unpatched_fp32, max_chunk=args.max_chunk
            )
            print("[Jamba Real-Weights FP32 Check]", json.dumps(real_fp32_check), flush=True)
            del model_fp32
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            JambaMambaMixer.forward = orig_jamba_fwd

    summ = {
        "runner": "pilot/w14_optb_jamba_stateful_probe.py",
        "repo_id": repo_id,
        "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "os_name": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": _tf.__version__,
        "max_chunk": int(args.max_chunk),
        "clean_rerun": bool(args.clean_out),
        "stateful_chunk_equivalence": eq_res,
        "real_weights_fp32_equiv_check": real_fp32_check,
        "oracle_gate": {"n": len(gate_cases), "hits": ora_hits, "rate": ora_rate, "previews": ora_previews},
        "gate_passed": gate_ok,
        "e2_summary": e2_summary,
    }
    summ_p = out_dir / "w14_optb_jamba3b_summary.json"
    summ_p.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Wrote {summ_p} (gate_passed={gate_ok})", flush=True)


if __name__ == "__main__":
    main()
