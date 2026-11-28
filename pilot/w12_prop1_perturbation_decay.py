"""Wave 12: Proposition-1 Perturbation-Decay Curves, Forcing Isolation & Operational Sufficiency Criterion.

Directly validates the conditional forced Input-to-State Stability (ISS) decomposition (Prop. 1):
    Delta S_T = (Prod_{tau=b+1}^T H_tau) Delta S_b
              + Sum_{tau=b+1}^T (Prod_{j=tau+1}^T H_j)(B_tau Delta u_tau + Delta H_tau S_{tau-1}^{old})
              + eps_cross

Measures across horizon grid T - b in {0, 64, 128, 256, 512, 1024, 2048, 3072[, 3584]}:
  A. Exact Telescoping Decomposition (`prop1_curves`) via 3 counterfactual suffix rollouts:
       1. `old`:           (S_b^{old}, KV_b^{old}) --[b, T)--> S_T^{old}
       2. `recomp`:        (S_b^{new}, KV_b^{new}) --[b, T)--> S_T^{new}
       3. `kv_patched_fwd`:(S_b^{old}, KV_b^{new}) --[b, T)--> S_T^{kv_recomp}
     Yielding exact pointwise identity (verified in float64):
       Delta S_T^{total} === (S_T^{new} - S_T^{kv_recomp})  +  (S_T^{kv_recomp} - S_T^{old})
                             [pure initial-state propagation]    [realized KV-readout forcing]
     Comparing paired `uncoupled_sniah` vs `coupled_bridge` (bridge trigger inserted at offset +128)
     plus suffix full-attention mass onto [p0, b) (`suf_attn_blk_max/mean`), slow vs fast heads, and
     pre-attention layers (Qwen3.5 layers 0..2 / Zamba2 layers 0..4 where forcing === 0 exactly).
  B. Controlled Impulse Perturbation Injection (`impulse_inj`):
     Zero token/KV diff, injecting delta S_b in {edit_subspace, random_gaussian} at eps in {0.1, 0.5, 2.0}*||S_b||_F
     to trace pure nonlinear contraction ||delta S_T|| / ||delta S_b|| and output KL decay vs T - b.
  C. Long-Prefix Equilibrium Test (`warm_prefix_scale`, p0 ~ 2048):
     Verifies whether ||S_T^{old}||_F saturates rather than diverging linearly with T.
  D. Parameter-Free Operational Sufficiency Criterion (`op_criterion`):
     Tests zero-fitted-weight Prop-1 certificate across W12 curves + W11 clean & T3 organic OOD splits (2B & 9B).

Supports `--synthetic-dry-run` on CPU (`0 CU`) prior to local RTX 4060 / Colab L4 GPU runs.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HOME", r"D:\applicationData\HuggingFace")

import numpy as np
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
import transformers.models.qwen3_5.modeling_qwen3_5 as qm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w5_9b_e5_runner as qwen_r  # noqa: E402
import w2_crossfamily_e0c as mf0  # noqa: E402
import w2_crossfamily_e2_e3_e6 as mf2  # noqa: E402


PILOT_DIR = Path(__file__).resolve().parent

QWEN_L_GRID = [0, 64, 128, 256, 512, 1024, 2048, 3584]
MAMBA_L_GRID = [0, 64, 128, 256, 512, 1024, 2048, 3072]
WARM_L_GRID = [0, 64, 256, 1024, 1536]

FILLER_SENTENCES = [
    "Archive node logged standard heartbeat diagnostics across rack telemetry sensors without anomaly.",
    "Batch scheduler completed periodic ledger compaction across secondary storage partitions on cycle four.",
    "Environmental control maintained constant coolant flow and barometric equilibrium in corridor B.",
    "Routing daemon verified checksum integrity over static lookup manifests during scheduled window.",
    "Power distribution bus balanced phase loads uniformly across auxiliary inverter cabinets today.",
    "Telemetry aggregator normalized floating timestamp offsets across synchronized optical transceivers.",
    "Background audit rotated cryptographic nonces and archived dormant session headers without alerts.",
    "Calibration daemon confirmed thermal drift stayed well within nominal factory specifications.",
]

SUBJECTS = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
    "Ivy", "Jack", "Karen", "Leo", "Mona", "Nate", "Olive", "Paul",
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync_dev(device: str) -> float:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            if "record_key" in rec:
                done.add(str(rec["record_key"]))
    return done


def clone_cache_safe(cache: Any) -> Any:
    """Deep-clone cache handling both dict-based (DynamicCache) and tensor-based states."""
    if cache is None:
        return None
    nc = copy.copy(cache)
    new_layers = []
    for layer in cache.layers:
        nl = copy.copy(layer)
        if getattr(layer, "keys", None) is not None and torch.is_tensor(layer.keys):
            nl.keys = layer.keys.detach().clone()
        if getattr(layer, "values", None) is not None and torch.is_tensor(layer.values):
            nl.values = layer.values.detach().clone()
        for attr in ("conv_states", "recurrent_states"):
            val = getattr(layer, attr, None)
            if isinstance(val, dict):
                setattr(
                    nl,
                    attr,
                    {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in val.items()},
                )
            elif torch.is_tensor(val):
                setattr(nl, attr, val.detach().clone())
        for attr in (
            "has_previous_state",
            "is_conv_states_initialized",
            "is_recurrent_states_initialized",
            "conv_kernel_size",
        ):
            if hasattr(layer, attr) and isinstance(getattr(layer, attr), dict):
                setattr(nl, attr, dict(getattr(layer, attr)))
        new_layers.append(nl)
    nc.layers = new_layers
    return nc


def extract_rec_states(cache: Any) -> List[Optional[torch.Tensor]]:
    """Return per-linear-layer recurrent state tensors (cloned on CPU float64 for exact metrics)."""
    out: List[Optional[torch.Tensor]] = []
    for layer in cache.layers:
        t = qwen_r._layer_get_state_tensor(layer, "recurrent_states")
        if t is not None:
            out.append(t.detach().to(device="cpu", dtype=torch.float64).clone())
    return out


def compute_prop1_state_metrics(
    s_old_list: List[torch.Tensor],
    s_new_list: List[torch.Tensor],
    s_kvr_list: List[torch.Tensor],
    delta_b_list: List[torch.Tensor],
    gamma_b2t_list: List[torch.Tensor],
    pre_attn_linear_idx: set[int],
    slow_head_masks: List[torch.Tensor],
) -> Dict[str, Any]:
    """
    Exact float64 Proposition-1 telescoping state decomposition metrics.

    Verifies vector identity per linear layer:
      (S_new - S_old) === (S_new - S_kvr) + (S_kvr - S_old)
    and aggregates Frobenius norms overall, on pre-attention linear layers, post-attention
    linear layers, persistent/slow heads (gamma_end >= 0.1), and fast-decay heads (< 0.1).
    """
    sum_old_sq = 0.0
    sum_db_sq = 0.0
    sum_tot_sq = 0.0
    sum_hom_sq = 0.0
    sum_frc_sq = 0.0
    sum_gpred_sq = 0.0
    sum_res_jl_sq = 0.0
    sum_res_st_sq = 0.0

    pre_db_sq = pre_hom_sq = pre_frc_sq = 0.0
    post_db_sq = post_hom_sq = post_frc_sq = 0.0
    slow_db_sq = slow_hom_sq = slow_frc_sq = slow_gpred_sq = slow_old_sq = 0.0
    fast_db_sq = fast_hom_sq = fast_frc_sq = fast_gpred_sq = 0.0
    l0_old_sq = l0_db_sq = l0_hom_sq = l0_frc_sq = l0_gp_sq = 0.0

    max_id_err = 0.0
    for li, (so, sn, sk, db, gam) in enumerate(
        zip(s_old_list, s_new_list, s_kvr_list, delta_b_list, gamma_b2t_list)
    ):
        # Broadcast gamma cleanly over trailing dims: so shape is (1, H, d1, d2) or (H, d1, d2)
        g = gam.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if so.ndim == 4:
            g_bc = g.view(1, -1, 1, 1)
            head_axis = 1
        elif so.ndim == 3:
            g_bc = g.view(-1, 1, 1)
            head_axis = 0
        else:
            raise ValueError(f"Unexpected state ndim={so.ndim}, shape={tuple(so.shape)}")

        tot_t = sn - so
        hom_t = sn - sk
        frc_t = sk - so
        id_err = float((tot_t - (hom_t + frc_t)).abs().max().item())
        if id_err > max_id_err:
            max_id_err = id_err

        g_db_t = g_bc * db
        res_jl_t = sn - (so + g_db_t)
        res_st_t = sn - (so + db)

        old_sq = float((so * so).sum().item())
        db_sq = float((db * db).sum().item())
        tot_sq = float((tot_t * tot_t).sum().item())
        hom_sq = float((hom_t * hom_t).sum().item())
        frc_sq = float((frc_t * frc_t).sum().item())
        gp_sq = float((g_db_t * g_db_t).sum().item())
        rjl_sq = float((res_jl_t * res_jl_t).sum().item())
        rst_sq = float((res_st_t * res_st_t).sum().item())

        sum_old_sq += old_sq
        sum_db_sq += db_sq
        sum_tot_sq += tot_sq
        sum_hom_sq += hom_sq
        sum_frc_sq += frc_sq
        sum_gpred_sq += gp_sq
        sum_res_jl_sq += rjl_sq
        sum_res_st_sq += rst_sq

        if li == 0:
            l0_old_sq = old_sq
            l0_db_sq = db_sq
            l0_hom_sq = hom_sq
            l0_frc_sq = frc_sq
            l0_gp_sq = gp_sq

        if li in pre_attn_linear_idx:
            pre_db_sq += db_sq
            pre_hom_sq += hom_sq
            pre_frc_sq += frc_sq
        else:
            post_db_sq += db_sq
            post_hom_sq += hom_sq
            post_frc_sq += frc_sq

        smask = slow_head_masks[li].to(device="cpu", dtype=torch.bool)
        reduce_dims = tuple(ax for ax in range(so.ndim) if ax != head_axis)
        h_old_sq = (so * so).sum(dim=reduce_dims)
        h_db_sq = (db * db).sum(dim=reduce_dims)
        h_hom_sq = (hom_t * hom_t).sum(dim=reduce_dims)
        h_frc_sq = (frc_t * frc_t).sum(dim=reduce_dims)
        h_gp_sq = (g_db_t * g_db_t).sum(dim=reduce_dims)

        if bool(smask.any()):
            slow_old_sq += float(h_old_sq[smask].sum().item())
            slow_db_sq += float(h_db_sq[smask].sum().item())
            slow_hom_sq += float(h_hom_sq[smask].sum().item())
            slow_frc_sq += float(h_frc_sq[smask].sum().item())
            slow_gpred_sq += float(h_gp_sq[smask].sum().item())
        fmask = ~smask
        if bool(fmask.any()):
            fast_db_sq += float(h_db_sq[fmask].sum().item())
            fast_hom_sq += float(h_hom_sq[fmask].sum().item())
            fast_frc_sq += float(h_frc_sq[fmask].sum().item())
            fast_gpred_sq += float(h_gp_sq[fmask].sum().item())

    eps = 1e-12
    s_old = math.sqrt(sum_old_sq)
    d_b = math.sqrt(sum_db_sq)
    e_tot = math.sqrt(sum_tot_sq)
    e_hom = math.sqrt(sum_hom_sq)
    e_frc = math.sqrt(sum_frc_sq)
    e_gp = math.sqrt(sum_gpred_sq)
    e_rjl = math.sqrt(sum_res_jl_sq)
    e_rst = math.sqrt(sum_res_st_sq)

    return {
        "identity_max_abs_err": max_id_err,
        "S_old_frob": s_old,
        "delta_S_b_frob": d_b,
        "E_total_frob": e_tot,
        "E_hom_frob": e_hom,
        "E_forcing_frob": e_frc,
        "gamma_pred_frob": e_gp,
        "E_res_jlocal_frob": e_rjl,
        "E_res_stale_frob": e_rst,
        "E_total_rel_b": e_tot / max(eps, d_b),
        "E_hom_rel_b": e_hom / max(eps, d_b),
        "gamma_pred_rel_b": e_gp / max(eps, d_b),
        "E_forcing_rel_b": e_frc / max(eps, d_b),
        "E_forcing_rel_S": e_frc / max(eps, s_old),
        "E_res_jlocal_rel_b": e_rjl / max(eps, d_b),
        "E_res_jlocal_rel_S": e_rjl / max(eps, s_old),
        "E_res_stale_rel_b": e_rst / max(eps, d_b),
        "E_res_stale_rel_S": e_rst / max(eps, s_old),
        "pre_attn_E_hom_rel_b": math.sqrt(pre_hom_sq) / max(eps, math.sqrt(pre_db_sq)) if pre_db_sq > 0 else None,
        "pre_attn_E_forcing_frob": math.sqrt(pre_frc_sq) if pre_attn_linear_idx else None,
        "layer0_S_old_frob": math.sqrt(l0_old_sq) if l0_old_sq > 0 else 0.0,
        "layer0_E_hom_rel_b": math.sqrt(l0_hom_sq) / max(eps, math.sqrt(l0_db_sq)) if l0_db_sq > 0 else None,
        "layer0_gamma_pred_rel_b": math.sqrt(l0_gp_sq) / max(eps, math.sqrt(l0_db_sq)) if l0_db_sq > 0 else None,
        "layer0_E_forcing_frob": math.sqrt(l0_frc_sq),
        "post_attn_E_hom_rel_b": math.sqrt(post_hom_sq) / max(eps, math.sqrt(post_db_sq)) if post_db_sq > 0 else None,
        "post_attn_E_forcing_rel_b": math.sqrt(post_frc_sq) / max(eps, math.sqrt(post_db_sq)) if post_db_sq > 0 else None,
        "slow_heads_S_old_frob": math.sqrt(slow_old_sq) if slow_old_sq > 0 else 0.0,
        "slow_heads_E_hom_rel_b": math.sqrt(slow_hom_sq) / max(eps, math.sqrt(slow_db_sq)) if slow_db_sq > 0 else None,
        "slow_heads_gamma_pred_rel_b": math.sqrt(slow_gpred_sq) / max(eps, math.sqrt(slow_db_sq)) if slow_db_sq > 0 else None,
        "slow_heads_E_forcing_rel_b": math.sqrt(slow_frc_sq) / max(eps, math.sqrt(slow_db_sq)) if slow_db_sq > 0 else None,
        "fast_heads_E_hom_rel_b": math.sqrt(fast_hom_sq) / max(eps, math.sqrt(fast_db_sq)) if fast_db_sq > 0 else None,
        "fast_heads_gamma_pred_rel_b": math.sqrt(fast_gpred_sq) / max(eps, math.sqrt(fast_db_sq)) if fast_db_sq > 0 else None,
    }


def build_aligned_paired_sample(
    tok: Any,
    model_family: str,
    s_idx: int,
    rng: random.Random,
    pool_d: Dict[str, List[str]],
    n_linear: int,
    w_conv: int,
    l_grid: List[int],
    p0_target_fillers: int = 2,
    device: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """
    Build a paired (uncoupled_sniah, coupled_bridge) sample where `[0, b_end + 128)`
    is 100% token-for-token identical across regimes, `[b_end + 128, b_end + max_L)`
    inserts `Vault dispatch rule` early inside `[b+128, b+256)` ONLY in `coupled_bridge`
    (with identical suffix tokens between old & edit within each regime!).
    """
    fd = rng.sample([str(d) for d in range(1, 10)], 6)
    old_val = rng.choice(pool_d[fd[0]])
    new_val = rng.choice(pool_d[fd[1]])
    ans_old = rng.choice(pool_d[fd[2]])
    ans_new = rng.choice(pool_d[fd[3]])
    extra_v1 = rng.choice(pool_d[fd[4]])
    extra_v2 = rng.choice(pool_d[fd[5]])

    def _first_nonspace(s: str) -> int:
        toks = tok.encode(s, add_special_tokens=False)
        for tid in toks:
            if tok.decode([tid]).strip():
                return int(tid)
        return int(toks[0])

    f_o = _first_nonspace(old_val)
    f_e = _first_nonspace(new_val)
    b_o = _first_nonspace(ans_old)
    b_e = _first_nonspace(ans_new)
    if f_o == f_e or b_o == b_e:
        return None

    target = SUBJECTS[s_idx % len(SUBJECTS)]
    other = SUBJECTS[(s_idx + 1) % len(SUBJECTS)]
    b_offset = (w_conv - 1) * (n_linear - 1)

    pre_sents = [f"The assigned codename ID for {other} is {extra_v1}."]
    for j in range(p0_target_fillers):
        pre_sents.append(FILLER_SENTENCES[(s_idx + j) % len(FILLER_SENTENCES)])
    pre_str = " ".join(pre_sents)

    fact_o = f"The assigned codename ID for {target} is {old_val}."
    fact_e = f"The assigned codename ID for {target} is {new_val}."

    # Pre-128 block + filler pad well beyond b_offset + 128
    mid_sents = [FILLER_SENTENCES[(s_idx + p0_target_fillers + j) % len(FILLER_SENTENCES)] for j in range(32)]
    mid_str = " ".join(mid_sents)

    q_direct_str = f"What is the assigned codename ID for {target}? Answer only with the 4-digit codename."
    q_bridge_str = f"Which beacon is unlocked by {target}? Output only the 4-digit beacon number."

    if model_family == "qwen":
        base_o = qwen_r.apply_chat_no_think(
            tok, [{"role": "system", "content": qwen_r.SYSTEM_PROMPT}, {"role": "user", "content": f"{pre_str} {fact_o} {mid_str}"}]
        )
        base_e = qwen_r.apply_chat_no_think(
            tok, [{"role": "system", "content": qwen_r.SYSTEM_PROMPT}, {"role": "user", "content": f"{pre_str} {fact_e} {mid_str}"}]
        )
        # Strip trailing assistant header generated by apply_chat_no_think so we can cleanly concat suffix tokens + query
        cut_marker = mid_sents[-1]
        pos_cut_o = base_o.rfind(cut_marker) + len(cut_marker)
        pos_cut_e = base_e.rfind(cut_marker) + len(cut_marker)
        tail_template = base_o[pos_cut_o:]  # "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ctx_head_o = base_o[:pos_cut_o]
        ctx_head_e = base_e[:pos_cut_e]
        q_direct_toks = tok.encode("\n\n" + q_direct_str + tail_template, add_special_tokens=False)
        q_bridge_toks = tok.encode("\n\n" + q_bridge_str + tail_template, add_special_tokens=False)
    else:
        full_o = mf0.render_chat_prompt(tok, f"{pre_str} {fact_o} {mid_str}\n{q_direct_str}")
        full_e = mf0.render_chat_prompt(tok, f"{pre_str} {fact_e} {mid_str}\n{q_direct_str}")
        q_mark = "\n" + q_direct_str
        qo = full_o.rfind(q_mark)
        qe = full_e.rfind(q_mark)
        if qo < 0 or qe < 0:
            return None
        ctx_head_o = full_o[:qo]
        ctx_head_e = full_e[:qe]
        q_direct_toks = tok.encode(full_o[qo:], add_special_tokens=False)
        full_br = mf0.render_chat_prompt(tok, f"{pre_str} {fact_e} {mid_str}\n{q_bridge_str}")
        q_br_mark = "\n" + q_bridge_str
        q_bridge_toks = tok.encode(full_br[full_br.rfind(q_br_mark):], add_special_tokens=False)

    enc_o = tok(ctx_head_o, return_offsets_mapping=True, add_special_tokens=False)
    enc_e = tok(ctx_head_e, return_offsets_mapping=True, add_special_tokens=False)
    ids_head_o = list(enc_o["input_ids"])
    ids_head_e = list(enc_e["input_ids"])
    if len(ids_head_o) != len(ids_head_e):
        return None
    diffs = [i for i, (a, b) in enumerate(zip(ids_head_o, ids_head_e)) if a != b]
    if not diffs:
        return None
    p0, p1 = diffs[0], diffs[-1] + 1
    if (p1 - p0) > 6 or ids_head_o[:p0] != ids_head_e[:p0] or ids_head_o[p1:] != ids_head_e[p1:]:
        return None
    if old_val not in tok.decode(ids_head_o[max(0, p0 - 1) : p1 + 1]) or new_val not in tok.decode(ids_head_e[max(0, p0 - 1) : p1 + 1]):
        return None

    b_end = p1 + b_offset
    if b_end + 128 > len(ids_head_o):
        return None

    # Slice exact shared prefix [0, b_end + 128) across BOTH regimes!
    shared_pre128_o = ids_head_o[: b_end + 128]
    shared_pre128_e = ids_head_e[: b_end + 128]

    # Build bridge sentence tokens (<= 110 tokens so entire bridge finishes inside [b+128, b+256)!)
    if s_idx % 2 == 0:
        br_txt = (
            f" Vault dispatch rule: codename {old_val} unlocks beacon {ans_old},"
            f" while codename {new_val} unlocks beacon {ans_new}."
            f" Backup rule: codename {extra_v1} unlocks beacon {extra_v2}."
        )
    else:
        br_txt = (
            f" Vault dispatch rule: codename {new_val} unlocks beacon {ans_new},"
            f" while codename {old_val} unlocks beacon {ans_old}."
            f" Backup rule: codename {extra_v1} unlocks beacon {extra_v2}."
        )
    br_toks = tok.encode(br_txt, add_special_tokens=False)
    if len(br_toks) > 110:
        return None

    max_L = max(l_grid)
    need_after_128 = max_L - 128
    long_filler_text = " ".join(
        FILLER_SENTENCES[(s_idx + 20 + k) % len(FILLER_SENTENCES)] for k in range(320)
    )
    tail_toks = tok.encode(" " + long_filler_text, add_special_tokens=False)
    if len(tail_toks) < need_after_128:
        return None

    suf_unc_toks = tail_toks[:need_after_128]
    suf_cup_toks = br_toks + tail_toks[len(br_toks) : need_after_128]
    assert len(suf_unc_toks) == need_after_128 and len(suf_cup_toks) == need_after_128

    return {
        "sample_index": int(s_idx),
        "target": target,
        "old_val": old_val,
        "new_val": new_val,
        "ans_old": ans_old,
        "ans_new": ans_new,
        "f_o": int(f_o),
        "f_e": int(f_e),
        "b_o": int(b_o),
        "b_e": int(b_e),
        "p0": int(p0),
        "p1": int(p1),
        "b_end": int(b_end),
        "bridge_span_rel": [128, 128 + len(br_toks)],
        "ids_o_unc": torch.tensor([shared_pre128_o + suf_unc_toks], device=device, dtype=torch.long),
        "ids_e_unc": torch.tensor([shared_pre128_e + suf_unc_toks], device=device, dtype=torch.long),
        "ids_o_cup": torch.tensor([shared_pre128_o + suf_cup_toks], device=device, dtype=torch.long),
        "ids_e_cup": torch.tensor([shared_pre128_e + suf_cup_toks], device=device, dtype=torch.long),
        "ids_q_direct": torch.tensor([q_direct_toks], device=device, dtype=torch.long),
        "ids_q_bridge": torch.tensor([q_bridge_toks], device=device, dtype=torch.long),
    }


class QwenAttnMassProbe:
    """Temporarily wraps `qm.eager_attention_forward` to log softmax mass onto `[p0, b_end)` and `[p0, p1)`."""

    def __init__(self, base_model: Any, p0: int, p1: int, b_end: int):
        self.base_model = base_model
        self.p0 = int(p0)
        self.p1 = int(p1)
        self.b_end = int(b_end)
        self._orig_impl = getattr(base_model.config, "_attn_implementation", "sdpa")
        self._orig_eager = qm.eager_attention_forward
        self.max_mass = 0.0
        self.mean_sums = 0.0
        self.val_max_mass = 0.0
        self.val_mean_sums = 0.0
        self.call_count = 0

    def reset_bin(self) -> None:
        self.max_mass = 0.0
        self.mean_sums = 0.0
        self.val_max_mass = 0.0
        self.val_mean_sums = 0.0
        self.call_count = 0

    def read_bin(self) -> Dict[str, float]:
        return {
            "attn_blk_max": float(self.max_mass),
            "attn_blk_mean": float(self.mean_sums / max(1, self.call_count)),
            "val_attn_max": float(self.val_max_mass),
            "val_attn_mean": float(self.val_mean_sums / max(1, self.call_count)),
        }

    def __enter__(self):
        probe = self
        self.base_model.config._attn_implementation = "eager"

        def wrapped_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            attn_output, attn_weights = probe._orig_eager(
                module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
            )
            if attn_weights is not None and attn_weights.shape[-1] >= probe.b_end:
                blk_mass = attn_weights[..., probe.p0 : probe.b_end].float().sum(dim=-1)
                val_mass = attn_weights[..., probe.p0 : probe.p1].float().sum(dim=-1)
                probe.max_mass = max(probe.max_mass, float(blk_mass.max().item()))
                probe.mean_sums += float(blk_mass.mean().item())
                probe.val_max_mass = max(probe.val_max_mass, float(val_mass.max().item()))
                probe.val_mean_sums += float(val_mass.mean().item())
                probe.call_count += 1
            return attn_output, attn_weights

        qm.eager_attention_forward = wrapped_eager
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        qm.eager_attention_forward = self._orig_eager
        self.base_model.config._attn_implementation = self._orig_impl


def run_qwen_suffix_attn_bins_exact(
    model: Any,
    base_model: Any,
    c_old_b_cpu: Any,
    ids_ctx: torch.Tensor,
    ids_q_direct: torch.Tensor,
    ids_q_bridge: Optional[torch.Tensor],
    p0: int,
    p1: int,
    b_end: int,
    l_grid: List[int],
    device: str,
    use_l2k: bool,
) -> Dict[int, Dict[str, float]]:
    """Cleanly snap/restore states across query probes so suffix attention trace stays exact."""
    bins: Dict[int, Dict[str, float]] = {
        0: {"attn_blk_max": 0.0, "attn_blk_mean": 0.0, "val_attn_max": 0.0, "val_attn_mean": 0.0, "q_direct_attn_blk_max": 0.0, "q_direct_val_attn_max": 0.0}
    }
    c_attn = qwen_r.move_cache_device(c_old_b_cpu, device)
    with QwenAttnMassProbe(base_model, p0, p1, b_end) as probe:
        prev_d = 0
        for cur_d in l_grid[1:]:
            probe.reset_bin()
            seg = ids_ctx[:, b_end + prev_d : b_end + cur_d]
            _, c_attn = qwen_r.run_chunked_prefill(model, base_model, seg, b_end + prev_d, c_attn, 256, use_l2k)
            st = probe.read_bin()
            T_cur = b_end + cur_d
            snap_t = qwen_r.snap_states(c_attn)

            probe.reset_bin()
            q_pos = qwen_r.make_position_ids(T_cur, int(ids_q_direct.shape[1]), device, base_model)
            kw = {"input_ids": ids_q_direct, "position_ids": q_pos, "use_cache": True, "past_key_values": c_attn}
            if use_l2k:
                kw["logits_to_keep"] = 1
            _ = model(**kw)
            q_dir_st = probe.read_bin()
            qwen_r.truncate_cache_kv(c_attn, T_cur)
            qwen_r.overwrite_states(c_attn, snap_t)

            entry = {
                "attn_blk_max": st["attn_blk_max"],
                "attn_blk_mean": st["attn_blk_mean"],
                "val_attn_max": st["val_attn_max"],
                "val_attn_mean": st["val_attn_mean"],
                "q_direct_attn_blk_max": q_dir_st["attn_blk_max"],
                "q_direct_attn_blk_mean": q_dir_st["attn_blk_mean"],
                "q_direct_val_attn_max": q_dir_st["val_attn_max"],
                "q_direct_val_attn_mean": q_dir_st["val_attn_mean"],
            }
            if ids_q_bridge is not None and cur_d >= 256:
                probe.reset_bin()
                q_pos_b = qwen_r.make_position_ids(T_cur, int(ids_q_bridge.shape[1]), device, base_model)
                kw_b = {"input_ids": ids_q_bridge, "position_ids": q_pos_b, "use_cache": True, "past_key_values": c_attn}
                if use_l2k:
                    kw_b["logits_to_keep"] = 1
                _ = model(**kw_b)
                q_br_st = probe.read_bin()
                qwen_r.truncate_cache_kv(c_attn, T_cur)
                qwen_r.overwrite_states(c_attn, snap_t)
                entry["q_bridge_attn_blk_max"] = q_br_st["attn_blk_max"]
                entry["q_bridge_attn_blk_mean"] = q_br_st["attn_blk_mean"]
                entry["q_bridge_val_attn_max"] = q_br_st["val_attn_max"]
                entry["q_bridge_val_attn_mean"] = q_br_st["val_attn_mean"]
            del snap_t
            bins[int(cur_d)] = entry
            prev_d = cur_d
    del c_attn
    return bins


def _eval_qwen_checkpoint(
    model: Any,
    base_model: Any,
    tok: Any,
    eos_ids: List[int],
    c_recomp: Any,
    c_old: Any,
    T_cur: int,
    ids_q: torch.Tensor,
    delta_b: Dict[str, Any],
    gamma_b2t: List[torch.Tensor],
    old_blk_kv: Any,
    new_blk_kv: Any,
    p0: int,
    b_end: int,
    w_conv: int,
    new_ans: str,
    old_ans: str,
    f_e: int,
    f_o: int,
    use_l2k: bool,
) -> Dict[str, Any]:
    """Evaluate downstream query alignment at a checkpoint T_cur in-place and restore caches bit-cleanly."""
    snap_new = qwen_r.snap_states(c_recomp)
    ev_rec = qwen_r.eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_recomp, T_cur, ids_q, use_l2k)
    lg_rec = ev_rec["first_logits"].detach().clone()
    qwen_r.truncate_cache_kv(c_recomp, T_cur)
    qwen_r.overwrite_states(c_recomp, snap_new)
    del snap_new

    snap_old = qwen_r.snap_states(c_old)
    # old
    ev_old = qwen_r.eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_old, T_cur, ids_q, use_l2k)
    qwen_r.restore_ctx_cache_in_place(c_old, T_cur, snap_old, old_blk_kv, p0, b_end)
    # kv_only_blk
    qwen_r.patch_kv_slice(c_old, new_blk_kv, p0, b_end)
    ev_kv = qwen_r.eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_old, T_cur, ids_q, use_l2k)
    qwen_r.restore_ctx_cache_in_place(c_old, T_cur, snap_old, old_blk_kv, p0, b_end)
    # jlocal (O_local + decay)
    if T_cur > b_end:
        qwen_r.apply_jlocal_scalar_update(c_old, delta_b, gamma_b2t, b_end, T_cur, w_conv, use_decay=True)
    else:
        qwen_r.apply_jlocal_scalar_update(c_old, delta_b, gamma_b2t, b_end, T_cur, w_conv, use_decay=False)
    qwen_r.patch_kv_slice(c_old, new_blk_kv, p0, b_end)
    ev_jl = qwen_r.eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_old, T_cur, ids_q, use_l2k)
    qwen_r.restore_ctx_cache_in_place(c_old, T_cur, snap_old, old_blk_kv, p0, b_end)
    # stale (no decay)
    qwen_r.apply_jlocal_scalar_update(c_old, delta_b, gamma_b2t, b_end, T_cur, w_conv, use_decay=False)
    qwen_r.patch_kv_slice(c_old, new_blk_kv, p0, b_end)
    ev_st = qwen_r.eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_old, T_cur, ids_q, use_l2k)
    qwen_r.restore_ctx_cache_in_place(c_old, T_cur, snap_old, old_blk_kv, p0, b_end)
    del snap_old

    # Non-oracle O(1) features computed purely from old vs jlocal logits
    lg_jl = ev_jl["first_logits"].float()
    lg_old_q = ev_old["first_logits"].float()
    p_jl = F.softmax(lg_jl, dim=-1)
    p_old = F.softmax(lg_old_q, dim=-1)
    top2 = torch.topk(lg_jl[0], 2).values
    old_top1 = int(lg_old_q.argmax(dim=-1).item())
    jl_top1 = int(lg_jl.argmax(dim=-1).item())
    o1_feats = {
        "margin_jl": float((top2[0] - top2[1]).item()),
        "shift_kl_old_vs_jl": float(F.kl_div((p_old + 1e-12).log(), p_jl, reduction="batchmean").item()),
        "top1_flipped_from_old": float(old_top1 != jl_top1),
        "stale_top1_prob_under_jl": float(p_jl[0, old_top1].item()),
    }

    return {
        "recomp": qwen_r.score_condition(ev_rec, new_ans, old_ans, f_e, f_o, lg_rec),
        "jlocal": qwen_r.score_condition(ev_jl, new_ans, old_ans, f_e, f_o, lg_rec),
        "kv_only_blk": qwen_r.score_condition(ev_kv, new_ans, old_ans, f_e, f_o, lg_rec),
        "stale": qwen_r.score_condition(ev_st, new_ans, old_ans, f_e, f_o, lg_rec),
        "old": qwen_r.score_condition(ev_old, new_ans, old_ans, f_e, f_o, lg_rec),
        "o1_feats": o1_feats,
    }


@torch.no_grad()
def eval_qwen_prop1_sample(
    model: Any,
    base_model: Any,
    tok: Any,
    eos_ids: List[int],
    sample: Dict[str, Any],
    n_linear: int,
    d_k: int,
    w_conv: int,
    pre_attn_idx: set[int],
    l_grid: List[int],
    device: str,
    use_l2k: bool,
    verify_restore: bool = False,
) -> Dict[str, Any]:
    """Run paired `uncoupled_sniah` & `coupled_bridge` 3-rollout Prop-1 telescoping curves on Qwen3.5."""
    p0, p1, b_end = sample["p0"], sample["p1"], sample["b_end"]
    out_regimes: Dict[str, Any] = {}

    for reg_name in ("uncoupled_sniah", "coupled_bridge"):
        ids_o = sample["ids_o_unc"] if reg_name == "uncoupled_sniah" else sample["ids_o_cup"]
        ids_e = sample["ids_e_unc"] if reg_name == "uncoupled_sniah" else sample["ids_e_cup"]
        ids_q_dir = sample["ids_q_direct"]
        ids_q_br = sample["ids_q_bridge"] if reg_name == "coupled_bridge" else None

        # Shared prefix [0, p0) + old/new local block [p0, b_end)
        _, c_pre = qwen_r.run_chunked_prefill(model, base_model, ids_o[:, :p0], 0, None, 2048, use_l2k)
        c_pre_cpu = qwen_r.move_cache_device(c_pre, "cpu")
        _, c_old_b = qwen_r.run_chunked_prefill(model, base_model, ids_o[:, p0:b_end], p0, c_pre, 2048, use_l2k)
        del c_pre
        s_old_b_snap = qwen_r.snap_states(c_old_b)
        old_blk_kv = qwen_r.extract_kv_slice(c_old_b, p0, b_end)
        c_old_b_cpu = qwen_r.move_cache_device(c_old_b, "cpu")
        del c_old_b

        c_pre_gpu = qwen_r.move_cache_device(c_pre_cpu, device)
        del c_pre_cpu
        _, c_new_b = qwen_r.run_chunked_prefill(model, base_model, ids_e[:, p0:b_end], p0, c_pre_gpu, 2048, use_l2k)
        del c_pre_gpu
        s_new_b_snap = qwen_r.snap_states(c_new_b)
        new_blk_kv = qwen_r.extract_kv_slice(c_new_b, p0, b_end)
        delta_b = qwen_r.compute_state_delta_fp64(s_old_b_snap, s_new_b_snap)
        delta_b_rec_list = [d.detach().cpu().clone() for d in delta_b["rec"] if d is not None]

        # Collect suffix full-attn mass onto [p0, b_end) and [p0, p1) in one dedicated pass from c_old_b_cpu
        attn_bins = run_qwen_suffix_attn_bins_exact(
            model, base_model, c_old_b_cpu, ids_o, ids_q_dir, ids_q_br, p0, p1, b_end, l_grid, device, use_l2k
        )

        # Initialize 3 trajectories at b_end:
        c_old = qwen_r.move_cache_device(c_old_b_cpu, device)
        c_kvr = qwen_r.move_cache_device(c_old_b_cpu, device)
        qwen_r.patch_kv_slice(c_kvr, new_blk_kv, p0, b_end)
        c_new = c_new_b
        del c_old_b_cpu

        g_cum = [None for _ in range(n_linear)]
        chk_data: Dict[int, Dict[str, Any]] = {}
        eval_dir_map: Dict[int, Dict[str, Any]] = {}
        eval_br_map: Dict[int, Dict[str, Any]] = {}

        # Step through horizon grid cleanly
        prev_d = 0
        for idx_g, cur_d in enumerate(l_grid):
            T_cur = b_end + cur_d
            if cur_d > prev_d:
                seg_o = ids_o[:, b_end + prev_d : b_end + cur_d]
                seg_e = ids_e[:, b_end + prev_d : b_end + cur_d]
                with qwen_r.GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as col:
                    _, c_old = qwen_r.run_chunked_prefill(model, base_model, seg_o, b_end + prev_d, c_old, 2048, use_l2k)
                hs_inc = col.finalize(expected_suffix_len=cur_d - prev_d, device=device)
                for li in range(n_linear):
                    g_cum[li] = hs_inc["G_suf"][li] if g_cum[li] is None else (g_cum[li] + hs_inc["G_suf"][li])
                _, c_new = qwen_r.run_chunked_prefill(model, base_model, seg_e, b_end + prev_d, c_new, 2048, use_l2k)
                _, c_kvr = qwen_r.run_chunked_prefill(model, base_model, seg_o, b_end + prev_d, c_kvr, 2048, use_l2k)

            if cur_d == 0:
                H_v = delta_b_rec_list[0].shape[1] if delta_b_rec_list[0].ndim == 4 else delta_b_rec_list[0].shape[0]
                gam_list = [torch.ones((1, H_v, 1, 1), device=device, dtype=torch.float64) for _ in range(n_linear)]
            else:
                gam_list = [torch.exp(g_cum[li]).view(1, -1, 1, 1) for li in range(n_linear)]

            s_old_cur = extract_rec_states(c_old)
            s_new_cur = extract_rec_states(c_new)
            s_kvr_cur = extract_rec_states(c_kvr)
            chk_data[int(cur_d)] = {
                "s_old": s_old_cur,
                "s_new": s_new_cur,
                "s_kvr": s_kvr_cur,
                "gam": [g.detach().cpu().clone() for g in gam_list],
            }

            if verify_restore and idx_g == 0:
                ref_c_old = clone_cache_safe(c_old)
                ref_c_new = clone_cache_safe(c_new)

            eval_dir_map[int(cur_d)] = _eval_qwen_checkpoint(
                model, base_model, tok, eos_ids, c_new, c_old, T_cur, ids_q_dir,
                delta_b, gam_list, old_blk_kv, new_blk_kv, p0, b_end, w_conv,
                sample["new_val"], sample["old_val"], sample["f_e"], sample["f_o"], use_l2k,
            )
            if ids_q_br is not None and cur_d >= 256:
                eval_br_map[int(cur_d)] = _eval_qwen_checkpoint(
                    model, base_model, tok, eos_ids, c_new, c_old, T_cur, ids_q_br,
                    delta_b, gam_list, old_blk_kv, new_blk_kv, p0, b_end, w_conv,
                    sample["ans_new"], sample["ans_old"], sample["b_e"], sample["b_o"], use_l2k,
                )

            if verify_restore and idx_g == 0:
                qwen_r.assert_cache_bitwise_equal(c_old, ref_c_old)
                qwen_r.assert_cache_bitwise_equal(c_new, ref_c_new)
                del ref_c_old, ref_c_new

            prev_d = cur_d

        # Determine slow vs fast heads once at max_L (`gamma_final >= 0.1`)
        max_d = max(l_grid)
        slow_masks = [g.reshape(-1) >= 0.1 for g in chk_data[int(max_d)]["gam"]]

        curve_points: List[Dict[str, Any]] = []
        for cur_d in l_grid:
            cd = chk_data[int(cur_d)]
            st_m = compute_prop1_state_metrics(
                cd["s_old"], cd["s_new"], cd["s_kvr"], delta_b_rec_list, cd["gam"], pre_attn_idx, slow_masks
            )
            # Add normalized scale wrt initial state ||S_b^{old}||_F
            s_b_frob = chk_data[0]["s_old_frob"] if 0 in chk_data and "s_old_frob" in chk_data[0] else None
            if cur_d == 0:
                s_b0_frob = st_m["S_old_frob"]
            st_m["S_old_vs_b"] = st_m["S_old_frob"] / max(1e-12, s_b0_frob)

            pt: Dict[str, Any] = {
                "horizon": int(cur_d),
                "state_metrics": st_m,
                "attn_metrics": attn_bins.get(int(cur_d), {}),
                "q_direct": eval_dir_map[int(cur_d)],
            }
            if int(cur_d) in eval_br_map:
                pt["q_bridge"] = eval_br_map[int(cur_d)]
            curve_points.append(pt)

        out_regimes[reg_name] = {
            "n_slow_heads": int(sum(int(m.sum().item()) for m in slow_masks)),
            "n_total_heads": int(sum(int(m.numel()) for m in slow_masks)),
            "curve": curve_points,
        }
        del c_old, c_new, c_kvr, chk_data, delta_b, old_blk_kv, new_blk_kv, s_old_b_snap, s_new_b_snap
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "sample_index": int(sample["sample_index"]),
        "p0": int(p0),
        "p1": int(p1),
        "b_end": int(b_end),
        "bridge_span_rel": sample["bridge_span_rel"],
        "regimes": out_regimes,
    }


@torch.no_grad()
def eval_qwen_impulse_sample(
    model: Any,
    base_model: Any,
    tok: Any,
    eos_ids: List[int],
    sample: Dict[str, Any],
    n_linear: int,
    d_k: int,
    pre_attn_idx: set[int],
    l_grid: List[int],
    eps_list: List[float],
    device: str,
    use_l2k: bool,
) -> Dict[str, Any]:
    """Controlled initial-state shock ||delta S_T|| / ||delta S_b|| with ZERO token/KV edit."""
    p0, b_end = sample["p0"], sample["b_end"]
    ids_o = sample["ids_o_unc"]
    ids_e = sample["ids_e_unc"]
    ids_q = sample["ids_q_direct"]

    _, c_pre = qwen_r.run_chunked_prefill(model, base_model, ids_o[:, :p0], 0, None, 2048, use_l2k)
    c_pre_cpu = qwen_r.move_cache_device(c_pre, "cpu")
    _, c_old_b = qwen_r.run_chunked_prefill(model, base_model, ids_o[:, p0:b_end], p0, c_pre, 2048, use_l2k)
    del c_pre
    s_old_b = qwen_r.snap_states(c_old_b)
    c_old_b_cpu = qwen_r.move_cache_device(c_old_b, "cpu")
    del c_old_b

    c_pre_gpu = qwen_r.move_cache_device(c_pre_cpu, device)
    del c_pre_cpu
    _, c_new_b = qwen_r.run_chunked_prefill(model, base_model, ids_e[:, p0:b_end], p0, c_pre_gpu, 2048, use_l2k)
    del c_pre_gpu
    s_new_b = qwen_r.snap_states(c_new_b)
    del c_new_b
    delta_real = qwen_r.compute_state_delta_fp64(s_old_b, s_new_b)
    real_rec = [d.detach().cpu().clone() for d in delta_real["rec"] if d is not None]
    old_rec_b = [r.detach().to(device="cpu", dtype=torch.float64).clone() for r in s_old_b["rec"] if r is not None]

    # Build clean unperturbed baseline trajectory across checkpoints
    c_clean = qwen_r.move_cache_device(c_old_b_cpu, device)
    clean_states: Dict[int, List[torch.Tensor]] = {0: extract_rec_states(c_clean)}
    clean_logits: Dict[int, torch.Tensor] = {}
    g_cum = [None for _ in range(n_linear)]
    gam_at: Dict[int, List[torch.Tensor]] = {
        0: [torch.ones((1, r.shape[1], 1, 1), dtype=torch.float64) for r in old_rec_b]
    }
    snap0 = qwen_r.snap_states(c_clean)
    ev0 = qwen_r.eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_clean, b_end, ids_q, use_l2k, max_new_tokens=1)
    clean_logits[0] = ev0["first_logits"].detach().cpu().clone()
    qwen_r.truncate_cache_kv(c_clean, b_end)
    qwen_r.overwrite_states(c_clean, snap0)
    del snap0

    prev_d = 0
    for cur_d in l_grid[1:]:
        seg = ids_o[:, b_end + prev_d : b_end + cur_d]
        with qwen_r.GDNSuffixHookCollector(n_linear=n_linear, d_k=d_k, scalar_only=True) as col:
            _, c_clean = qwen_r.run_chunked_prefill(model, base_model, seg, b_end + prev_d, c_clean, 2048, use_l2k)
        hs_inc = col.finalize(expected_suffix_len=cur_d - prev_d, device=device)
        for li in range(n_linear):
            g_cum[li] = hs_inc["G_suf"][li] if g_cum[li] is None else (g_cum[li] + hs_inc["G_suf"][li])
        gam_at[int(cur_d)] = [torch.exp(g_cum[li]).detach().cpu().view(1, -1, 1, 1) for li in range(n_linear)]
        clean_states[int(cur_d)] = extract_rec_states(c_clean)

        T_cur = b_end + cur_d
        snap_t = qwen_r.snap_states(c_clean)
        ev_t = qwen_r.eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_clean, T_cur, ids_q, use_l2k, max_new_tokens=1)
        clean_logits[int(cur_d)] = ev_t["first_logits"].detach().cpu().clone()
        qwen_r.truncate_cache_kv(c_clean, T_cur)
        qwen_r.overwrite_states(c_clean, snap_t)
        del snap_t
        prev_d = cur_d
    del c_clean

    max_d = max(l_grid)
    slow_masks = [g.reshape(-1) >= 0.1 for g in gam_at[int(max_d)]]

    # Build unit-norm direction vectors matching ||S_b^{old}||_F
    norm_old_b = math.sqrt(sum(float((r * r).sum().item()) for r in old_rec_b))
    norm_real = math.sqrt(sum(float((r * r).sum().item()) for r in real_rec))
    dir_edit = [r * (norm_old_b / max(1e-12, norm_real)) for r in real_rec]

    gen = torch.Generator(device="cpu").manual_seed(9000 + int(sample["sample_index"]))
    raw_rand = [torch.randn(r.shape, generator=gen, dtype=torch.float64) for r in old_rec_b]
    norm_rand = math.sqrt(sum(float((r * r).sum().item()) for r in raw_rand))
    dir_rand = [r * (norm_old_b / max(1e-12, norm_rand)) for r in raw_rand]

    results: Dict[str, Any] = {}
    for dname, dvecs in (("edit_subspace", dir_edit), ("isotropic_random", dir_rand)):
        for eps in eps_list:
            key = f"{dname}:eps={eps:.2f}"
            c_pert = qwen_r.move_cache_device(c_old_b_cpu, device)
            # Inject delta S_b directly into c_pert recurrent_states (zero KV/conv diff!)
            inj_list = [eps * dv for dv in dvecs]
            li = 0
            for layer in c_pert.layers:
                dst = qwen_r._layer_get_state_tensor(layer, "recurrent_states")
                if dst is not None:
                    dst.copy_((dst.to(torch.float64) + inj_list[li].to(device=dst.device)).to(dst.dtype))
                    li += 1
            norm_inj0 = math.sqrt(sum(float((x * x).sum().item()) for x in inj_list))

            pts: List[Dict[str, Any]] = []
            prev_d = 0
            for cur_d in l_grid:
                if cur_d > prev_d:
                    seg = ids_o[:, b_end + prev_d : b_end + cur_d]
                    _, c_pert = qwen_r.run_chunked_prefill(model, base_model, seg, b_end + prev_d, c_pert, 2048, use_l2k)
                sp = extract_rec_states(c_pert)
                T_cur = b_end + cur_d
                snap_p = qwen_r.snap_states(c_pert)
                ev_p = qwen_r.eval_from_ctx_cache_in_place(model, base_model, tok, eos_ids, c_pert, T_cur, ids_q, use_l2k, max_new_tokens=1)
                qwen_r.truncate_cache_kv(c_pert, T_cur)
                qwen_r.overwrite_states(c_pert, snap_p)
                del snap_p

                kl_p = mf0.kl_div_from_logits(clean_logits[int(cur_d)], ev_p["first_logits"].detach().cpu())
                sc_list = clean_states[int(cur_d)]
                tot_sq = pre_sq = post_sq = slow_sq = fast_sq = gpred_sq = l0_sq = 0.0
                pre_0_sq = post_0_sq = slow_0_sq = fast_0_sq = l0_0_sq = 0.0
                for idx_l, (sc, spp, iv, gm) in enumerate(zip(sc_list, sp, inj_list, gam_at[int(cur_d)])):
                    diff = spp - sc
                    dsq = float((diff * diff).sum().item())
                    ivsq = float((iv * iv).sum().item())
                    gp = gm.reshape(1, -1, 1, 1) * iv
                    gpsq = float((gp * gp).sum().item())
                    tot_sq += dsq
                    gpred_sq += gpsq
                    if idx_l == 0:
                        l0_sq = dsq
                        l0_0_sq = ivsq
                    if idx_l in pre_attn_idx:
                        pre_sq += dsq
                        pre_0_sq += ivsq
                    else:
                        post_sq += dsq
                        post_0_sq += ivsq
                    sm = slow_masks[idx_l]
                    fm = ~sm
                    h_dsq = (diff * diff).sum(dim=(0, 2, 3))
                    h_isq = (iv * iv).sum(dim=(0, 2, 3))
                    if bool(sm.any()):
                        slow_sq += float(h_dsq[sm].sum().item())
                        slow_0_sq += float(h_isq[sm].sum().item())
                    if bool(fm.any()):
                        fast_sq += float(h_dsq[fm].sum().item())
                        fast_0_sq += float(h_isq[fm].sum().item())

                pts.append({
                    "horizon": int(cur_d),
                    "rel_pert_norm": math.sqrt(tot_sq) / max(1e-12, norm_inj0),
                    "gamma_pred_rel_norm": math.sqrt(gpred_sq) / max(1e-12, norm_inj0),
                    "layer0_rel_pert": math.sqrt(l0_sq) / max(1e-12, math.sqrt(l0_0_sq)) if l0_0_sq > 0 else None,
                    "pre_attn_rel_pert": math.sqrt(pre_sq) / max(1e-12, math.sqrt(pre_0_sq)) if pre_0_sq > 0 else None,
                    "post_attn_rel_pert": math.sqrt(post_sq) / max(1e-12, math.sqrt(post_0_sq)) if post_0_sq > 0 else None,
                    "slow_heads_rel_pert": math.sqrt(slow_sq) / max(1e-12, math.sqrt(slow_0_sq)) if slow_0_sq > 0 else None,
                    "fast_heads_rel_pert": math.sqrt(fast_sq) / max(1e-12, math.sqrt(fast_0_sq)) if fast_0_sq > 0 else None,
                    "kl_vs_clean": float(kl_p),
                })
                prev_d = cur_d
            results[key] = pts
            del c_pert

    del c_old_b_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sample_index": int(sample["sample_index"]), "norm_old_b": norm_old_b, "norm_real_delta": norm_real, "trajectories": results}


@torch.no_grad()
def eval_mamba_prop1_sample(
    model: Any,
    cfg: Any,
    tok: Any,
    mixers: List[Tuple[int, Any]],
    eos_ids: List[int],
    sample: Dict[str, Any],
    pre_attn_idx: set[int],
    l_grid: List[int],
    device: str,
) -> Dict[str, Any]:
    """Run paired `uncoupled_sniah` & `coupled_bridge` 3-rollout Prop-1 telescoping curves on Falcon-H1 / Zamba2."""
    p0, p1, b_end = sample["p0"], sample["p1"], sample["b_end"]
    n_linear = len(mixers)
    w_conv = int(getattr(cfg, "mamba_d_conv", getattr(cfg, "conv_kernel_size", 4)))
    out_regimes: Dict[str, Any] = {}

    def _eval_mamba_ckpt(c_rec_in, c_old_in, T_cur, ids_q, delta_dict, g_sums_dict, c_new_blk_ref, new_ans, old_ans, f_e, f_o):
        # Safe clone evaluation for Mamba-2 DynamicCache (~45 MB at 3K tok)
        res_rec = mf2.eval_suffix_condition(model, tok, mf0.clone_cache(c_rec_in), ids_q, T_cur, new_ans, old_ans, "digit4", 6, eos_ids)
        res_old = mf2.eval_suffix_condition(model, tok, mf0.clone_cache(c_old_in), ids_q, T_cur, new_ans, old_ans, "digit4", 6, eos_ids)
        c_kv = mf2.apply_mamba2_jlocal_update(c_old_in, c_new_blk_ref, delta_dict, g_sums_dict, p0, b_end, T_cur, w_conv, use_decay=True, patch_state=False, patch_kv=True)
        res_kv = mf2.eval_suffix_condition(model, tok, c_kv, ids_q, T_cur, new_ans, old_ans, "digit4", 6, eos_ids)
        c_jl = mf2.apply_mamba2_jlocal_update(c_old_in, c_new_blk_ref, delta_dict, g_sums_dict, p0, b_end, T_cur, w_conv, use_decay=True, patch_state=True, patch_kv=True)
        res_jl = mf2.eval_suffix_condition(model, tok, c_jl, ids_q, T_cur, new_ans, old_ans, "digit4", 6, eos_ids)
        c_st = mf2.apply_mamba2_jlocal_update(c_old_in, c_new_blk_ref, delta_dict, g_sums_dict, p0, b_end, T_cur, w_conv, use_decay=False, patch_state=True, patch_kv=True)
        res_st = mf2.eval_suffix_condition(model, tok, c_st, ids_q, T_cur, new_ans, old_ans, "digit4", 6, eos_ids)

        lg_rec = res_rec["last_logits"].float()
        lg_jl = res_jl["last_logits"].float()
        lg_old_q = res_old["last_logits"].float()
        p_jl = F.softmax(lg_jl, dim=-1)
        p_old = F.softmax(lg_old_q, dim=-1)
        old_t1 = int(lg_old_q.argmax(dim=-1).item())
        jl_t1 = int(lg_jl.argmax(dim=-1).item())

        def _fmt(r):
            return {
                "contains_new": bool(r["acc_new"]),
                "contains_old": bool(r["acc_old"]),
                "kl_vs_oneshot": float(mf0.kl_div_from_logits(lg_rec, r["last_logits"].float())),
            }

        return {
            "recomp": _fmt(res_rec),
            "jlocal": _fmt(res_jl),
            "kv_only_blk": _fmt(res_kv),
            "stale": _fmt(res_st),
            "old": _fmt(res_old),
            "o1_feats": {
                "shift_kl_old_vs_jl": float(F.kl_div((p_old + 1e-12).log(), p_jl, reduction="batchmean").item()),
                "top1_flipped_from_old": float(old_t1 != jl_t1),
                "stale_top1_prob_under_jl": float(p_jl[0, old_t1].item()),
            },
        }

    for reg_name in ("uncoupled_sniah", "coupled_bridge"):
        ids_o = sample["ids_o_unc"] if reg_name == "uncoupled_sniah" else sample["ids_o_cup"]
        ids_e = sample["ids_e_unc"] if reg_name == "uncoupled_sniah" else sample["ids_e_cup"]
        ids_q_dir = sample["ids_q_direct"]
        ids_q_br = sample["ids_q_bridge"] if reg_name == "coupled_bridge" else None

        c_pre = DynamicCache(config=cfg)
        if p0 > 0:
            mf0.run_segment(model, c_pre, ids_o[:, :p0], 0, max_chunk=1024)
        c_old_b = mf0.clone_cache(c_pre)
        mf0.run_segment(model, c_old_b, ids_o[:, p0:b_end], p0, max_chunk=1024)
        c_new_b = mf0.clone_cache(c_pre)
        mf0.run_segment(model, c_new_b, ids_e[:, p0:b_end], p0, max_chunk=1024)
        del c_pre

        delta_dict = mf2.compute_state_delta_fp64(c_old_b, c_new_b)
        delta_rec_list = [delta_dict[li].detach().cpu().clone() for li in sorted(delta_dict.keys())]
        rec_layer_ids = sorted(delta_dict.keys())
        remapped_pre_idx = {k for k, li in enumerate(rec_layer_ids) if li in pre_attn_idx}

        c_old = mf0.clone_cache(c_old_b)
        c_kvr = mf0.clone_cache(c_old_b)
        mf0.patch_kv_range(c_kvr, c_new_b, p0, b_end)
        c_new = mf0.clone_cache(c_new_b)

        g_sums: Dict[int, torch.Tensor] = {}
        chk_data: Dict[int, Dict[str, Any]] = {}
        eval_dir_map: Dict[int, Dict[str, Any]] = {}
        eval_br_map: Dict[int, Dict[str, Any]] = {}

        prev_d = 0
        for cur_d in l_grid:
            T_cur = b_end + cur_d
            if cur_d > prev_d:
                seg_o = ids_o[:, b_end + prev_d : b_end + cur_d]
                seg_e = ids_e[:, b_end + prev_d : b_end + cur_d]
                _, g_inc, _ = mf2.run_segment_with_mamba_decay(model, c_old, seg_o, b_end + prev_d, mixers, max_chunk=1024)
                for li, gv in g_inc.items():
                    g_sums[li] = gv if li not in g_sums else (g_sums[li] + gv)
                mf0.run_segment(model, c_new, seg_e, b_end + prev_d, max_chunk=1024)
                mf0.run_segment(model, c_kvr, seg_o, b_end + prev_d, max_chunk=1024)

            if cur_d == 0:
                H_m = mixers[0][1].num_heads
                g_pass = {li: torch.zeros((1, H_m), device=device, dtype=torch.float64) for li in rec_layer_ids}
            else:
                g_pass = g_sums

            gam_list = [torch.exp(g_pass[li]).detach().cpu() for li in rec_layer_ids]
            s_old_cur = [c_old.layers[li].recurrent_states[0].detach().cpu().double().clone() for li in rec_layer_ids]
            s_new_cur = [c_new.layers[li].recurrent_states[0].detach().cpu().double().clone() for li in rec_layer_ids]
            s_kvr_cur = [c_kvr.layers[li].recurrent_states[0].detach().cpu().double().clone() for li in rec_layer_ids]
            chk_data[int(cur_d)] = {"s_old": s_old_cur, "s_new": s_new_cur, "s_kvr": s_kvr_cur, "gam": gam_list}

            eval_dir_map[int(cur_d)] = _eval_mamba_ckpt(
                c_new, c_old, T_cur, ids_q_dir, delta_dict, g_pass, c_new_b,
                sample["new_val"], sample["old_val"], sample["f_e"], sample["f_o"],
            )
            if ids_q_br is not None and cur_d >= 256:
                eval_br_map[int(cur_d)] = _eval_mamba_ckpt(
                    c_new, c_old, T_cur, ids_q_br, delta_dict, g_pass, c_new_b,
                    sample["ans_new"], sample["ans_old"], sample["b_e"], sample["b_o"],
                )
            prev_d = cur_d

        max_d = max(l_grid)
        slow_masks = [g.reshape(-1) >= 0.1 for g in chk_data[int(max_d)]["gam"]]
        curve_points: List[Dict[str, Any]] = []
        s_b0_frob = 1.0
        for cur_d in l_grid:
            cd = chk_data[int(cur_d)]
            st_m = compute_prop1_state_metrics(
                cd["s_old"], cd["s_new"], cd["s_kvr"], delta_rec_list, cd["gam"], remapped_pre_idx, slow_masks
            )
            if cur_d == 0:
                s_b0_frob = st_m["S_old_frob"]
            st_m["S_old_vs_b"] = st_m["S_old_frob"] / max(1e-12, s_b0_frob)
            pt: Dict[str, Any] = {"horizon": int(cur_d), "state_metrics": st_m, "q_direct": eval_dir_map[int(cur_d)]}
            if int(cur_d) in eval_br_map:
                pt["q_bridge"] = eval_br_map[int(cur_d)]
            curve_points.append(pt)

        out_regimes[reg_name] = {
            "n_slow_heads": int(sum(int(m.sum().item()) for m in slow_masks)),
            "n_total_heads": int(sum(int(m.numel()) for m in slow_masks)),
            "curve": curve_points,
        }
        del c_old, c_new, c_kvr, c_old_b, c_new_b, chk_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"sample_index": int(sample["sample_index"]), "p0": p0, "b_end": b_end, "regimes": out_regimes}


@torch.no_grad()
def eval_mamba_impulse_sample(
    model: Any,
    cfg: Any,
    mixers: List[Tuple[int, Any]],
    sample: Dict[str, Any],
    pre_attn_idx: set[int],
    l_grid: List[int],
    eps_list: List[float],
    device: str,
) -> Dict[str, Any]:
    """Controlled impulse shock ||delta S_T|| / ||delta S_b|| on Mamba-2 (zero KV/token edit)."""
    p0, b_end = sample["p0"], sample["b_end"]
    ids_o = sample["ids_o_unc"]
    ids_e = sample["ids_e_unc"]
    rec_layer_ids = sorted(li for li, _ in mixers)
    remapped_pre = {k for k, li in enumerate(rec_layer_ids) if li in pre_attn_idx}

    c_pre = DynamicCache(config=cfg)
    if p0 > 0:
        mf0.run_segment(model, c_pre, ids_o[:, :p0], 0, max_chunk=1024)
    c_old_b = mf0.clone_cache(c_pre)
    mf0.run_segment(model, c_old_b, ids_o[:, p0:b_end], p0, max_chunk=1024)
    c_new_b = mf0.clone_cache(c_pre)
    mf0.run_segment(model, c_new_b, ids_e[:, p0:b_end], p0, max_chunk=1024)
    del c_pre
    delta_dict = mf2.compute_state_delta_fp64(c_old_b, c_new_b)
    del c_new_b

    real_rec = [delta_dict[li].detach().cpu().clone() for li in rec_layer_ids]
    old_rec_b = [c_old_b.layers[li].recurrent_states[0].detach().cpu().double().clone() for li in rec_layer_ids]
    norm_old_b = math.sqrt(sum(float((r * r).sum().item()) for r in old_rec_b))
    norm_real = math.sqrt(sum(float((r * r).sum().item()) for r in real_rec))
    dir_edit = [r * (norm_old_b / max(1e-12, norm_real)) for r in real_rec]
    gen = torch.Generator(device="cpu").manual_seed(9100 + int(sample["sample_index"]))
    raw_rand = [torch.randn(r.shape, generator=gen, dtype=torch.float64) for r in old_rec_b]
    norm_rand = math.sqrt(sum(float((r * r).sum().item()) for r in raw_rand))
    dir_rand = [r * (norm_old_b / max(1e-12, norm_rand)) for r in raw_rand]

    c_clean = mf0.clone_cache(c_old_b)
    clean_states = {0: [c_clean.layers[li].recurrent_states[0].detach().cpu().double().clone() for li in rec_layer_ids]}
    clean_logits: Dict[int, torch.Tensor] = {}
    H_m = mixers[0][1].num_heads
    gam_at: Dict[int, List[torch.Tensor]] = {0: [torch.ones((1, H_m), dtype=torch.float64) for _ in rec_layer_ids]}
    g_sums: Dict[int, torch.Tensor] = {}
    clean_logits[0] = model(input_ids=sample["ids_q_direct"], position_ids=mf0.make_position_ids(b_end, int(sample["ids_q_direct"].shape[1]), device), past_key_values=mf0.clone_cache(c_clean), use_cache=True).logits[:, -1, :].detach().cpu().float()

    prev_d = 0
    for cur_d in l_grid[1:]:
        seg = ids_o[:, b_end + prev_d : b_end + cur_d]
        _, g_inc, _ = mf2.run_segment_with_mamba_decay(model, c_clean, seg, b_end + prev_d, mixers, max_chunk=1024)
        for li, gv in g_inc.items():
            g_sums[li] = gv if li not in g_sums else (g_sums[li] + gv)
        gam_at[int(cur_d)] = [torch.exp(g_sums[li]).detach().cpu() for li in rec_layer_ids]
        clean_states[int(cur_d)] = [c_clean.layers[li].recurrent_states[0].detach().cpu().double().clone() for li in rec_layer_ids]
        T_cur = b_end + cur_d
        clean_logits[int(cur_d)] = model(input_ids=sample["ids_q_direct"], position_ids=mf0.make_position_ids(T_cur, int(sample["ids_q_direct"].shape[1]), device), past_key_values=mf0.clone_cache(c_clean), use_cache=True).logits[:, -1, :].detach().cpu().float()
        prev_d = cur_d
    del c_clean

    slow_masks = [g.reshape(-1) >= 0.1 for g in gam_at[int(max(l_grid))]]
    results: Dict[str, Any] = {}
    for dname, dvecs in (("edit_subspace", dir_edit), ("isotropic_random", dir_rand)):
        for eps in eps_list:
            c_pert = mf0.clone_cache(c_old_b)
            inj_list = [eps * dv for dv in dvecs]
            for k_l, li in enumerate(rec_layer_ids):
                dst = c_pert.layers[li].recurrent_states[0]
                dst.copy_((dst.double() + inj_list[k_l].to(device=dst.device)).to(dst.dtype))
            norm_inj0 = math.sqrt(sum(float((x * x).sum().item()) for x in inj_list))

            pts: List[Dict[str, Any]] = []
            prev_d = 0
            for cur_d in l_grid:
                if cur_d > prev_d:
                    seg = ids_o[:, b_end + prev_d : b_end + cur_d]
                    mf0.run_segment(model, c_pert, seg, b_end + prev_d, max_chunk=1024)
                sp = [c_pert.layers[li].recurrent_states[0].detach().cpu().double().clone() for li in rec_layer_ids]
                T_cur = b_end + cur_d
                lg_p = model(input_ids=sample["ids_q_direct"], position_ids=mf0.make_position_ids(T_cur, int(sample["ids_q_direct"].shape[1]), device), past_key_values=mf0.clone_cache(c_pert), use_cache=True).logits[:, -1, :].detach().cpu().float()
                kl_p = mf0.kl_div_from_logits(clean_logits[int(cur_d)], lg_p)

                tot_sq = slow_sq = fast_sq = slow_0_sq = fast_0_sq = gpred_sq = l0_sq = l0_0_sq = 0.0
                for idx_l, (sc, spp, iv, gm) in enumerate(zip(clean_states[int(cur_d)], sp, inj_list, gam_at[int(cur_d)])):
                    diff = spp - sc
                    dsq = float((diff * diff).sum().item())
                    ivsq = float((iv * iv).sum().item())
                    tot_sq += dsq
                    if idx_l == 0:
                        l0_sq = dsq
                        l0_0_sq = ivsq
                    gp = gm.reshape(1, -1, 1, 1) * iv
                    gpred_sq += float((gp * gp).sum().item())
                    sm = slow_masks[idx_l]
                    fm = ~sm
                    h_dsq = (diff * diff).sum(dim=(0, 2, 3))
                    h_isq = (iv * iv).sum(dim=(0, 2, 3))
                    if bool(sm.any()):
                        slow_sq += float(h_dsq[sm].sum().item())
                        slow_0_sq += float(h_isq[sm].sum().item())
                    if bool(fm.any()):
                        fast_sq += float(h_dsq[fm].sum().item())
                        fast_0_sq += float(h_isq[fm].sum().item())

                pts.append({
                    "horizon": int(cur_d),
                    "rel_pert_norm": math.sqrt(tot_sq) / max(1e-12, norm_inj0),
                    "gamma_pred_rel_norm": math.sqrt(gpred_sq) / max(1e-12, norm_inj0),
                    "layer0_rel_pert": math.sqrt(l0_sq) / max(1e-12, math.sqrt(l0_0_sq)) if l0_0_sq > 0 else None,
                    "slow_heads_rel_pert": math.sqrt(slow_sq) / max(1e-12, math.sqrt(slow_0_sq)) if slow_0_sq > 0 else None,
                    "fast_heads_rel_pert": math.sqrt(fast_sq) / max(1e-12, math.sqrt(fast_0_sq)) if fast_0_sq > 0 else None,
                    "kl_vs_clean": float(kl_p),
                })
                prev_d = cur_d
            results[f"{dname}:eps={eps:.2f}"] = pts
            del c_pert
    del c_old_b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sample_index": int(sample["sample_index"]), "norm_old_b": norm_old_b, "trajectories": results}


def prop1_analytical_risk_score(f: Dict[str, Any]) -> float:
    """
    Parameter-free analytical Prop-1 operational risk score (ZERO fitted ML weights):
      Combines (1) missing Delta-RoPE alignment when |Delta L| > 0,
               (2) state shock excess, and
               (3) realized query forcing dominance / stale anchoring.
    Higher score => higher failure risk (flag fallback / multi-span recompute).
    """
    return (
        2.0 * float(f.get("force_no_rope", 0.0))
        + 1.0 * float(f.get("stale_top1_prob_under_jl", 0.0))
        + 0.5 * (1.0 - float(f.get("top1_flipped_from_old", 1.0)))
        - 0.08 * min(float(f.get("shift_kl_old_vs_jl", 0.0)), 12.0)
        + 0.5 * max(0.0, float(f.get("rel_delta_s_mean", 0.0)) - 0.5)
    )


def prop1_analytical_flag(f: Dict[str, Any], attn_blk_max: Optional[float] = None) -> int:
    """Parameter-free binary Prop-1 sufficiency gate (`1` => escalate/fallback)."""
    if float(f.get("force_no_rope", 0.0)) > 0.0:
        return 1
    if float(f.get("rel_delta_s_mean", 0.0)) > 0.5:
        return 1
    if attn_blk_max is not None and attn_blk_max > 0.35 and float(f.get("shift_kl_old_vs_jl", 0.0)) < 1.0:
        return 1
    if float(f.get("top1_flipped_from_old", 1.0)) < 0.5 and float(f.get("stale_top1_prob_under_jl", 0.0)) >= 0.25:
        return 1
    if float(f.get("shift_kl_old_vs_jl", 0.0)) < 1.0:
        return 1
    return 0


def _auroc_np(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    return float(sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos) / (len(pos) * len(neg)))


def _pr_auc_np(y: np.ndarray, s: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return 0.0
    idx = np.argsort(-s)
    ys = y[idx]
    tp = np.cumsum(ys)
    prec = tp / (np.arange(len(ys)) + 1)
    return float(prec[ys == 1].sum() / n_pos)


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    if n <= 0 or k <= 0:
        return 0.0
    if k == n:
        return float((alpha / 2.0) ** (1.0 / n))
    # Pure Python binary search using incomplete beta via log-gamma / MC exact Beta CDF via regularized sum
    lo, hi = 0.0, float(k / n)
    # Binomial tail exact P(X >= k | n, p) = alpha/2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p_tail = sum(math.comb(n, j) * (mid ** j) * ((1.0 - mid) ** (n - j)) for j in range(k, n + 1))
        if p_tail < alpha / 2.0:
            lo = mid
        else:
            hi = mid
    return lo


def eval_osc_on_w11_and_w12(out_dir: Path, w12_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate parameter-free Proposition-1 Operational Sufficiency Criterion on W11 (2B/9B clean+T3) & W12."""
    rep: Dict[str, Any] = {"w11_splits": {}, "w12_coupled_vs_uncoupled": {}}
    for tag in ("2b", "9b"):
        for fn, kf in (
            (f"w11_detector_v2_clean_{tag}_records.jsonl", ":test:"),
            (f"w11_t3_organic_det_{tag}_records.jsonl", ":t3:"),
        ):
            p = out_dir / fn
            if not p.exists():
                continue
            recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            sub = [r for r in recs if kf in r.get("record_key", "")]
            if not sub:
                continue
            y = np.array([int(r["is_failure_jlocal"]) for r in sub])
            s = np.array([prop1_analytical_risk_score(r["features"]) for r in sub])
            flag = np.array([prop1_analytical_flag(r["features"]) for r in sub], dtype=int)
            tp = int(((flag == 1) & (y == 1)).sum())
            fp = int(((flag == 1) & (y == 0)).sum())
            tn = int(((flag == 0) & (y == 0)).sum())
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            tpr_lb = clopper_pearson_lower(tp, n_pos) if n_pos > 0 else 0.0
            tnr_lb = clopper_pearson_lower(tn, n_neg) if n_neg > 0 else 0.0
            rep["w11_splits"][f"{tag}:{fn}"] = {
                "n": len(sub),
                "n_fail": n_pos,
                "auroc": _auroc_np(y, s),
                "pr_auc": _pr_auc_np(y, s),
                "tpr": tp / max(1, n_pos),
                "fpr": fp / max(1, n_neg),
                "tpr_cp95_lb": tpr_lb,
                "tnr_cp95_lb": tnr_lb,
                "joint_auc_cp95_lb": max(0.0, tpr_lb + tnr_lb - 1.0),
                "raw_em": sum(bool(r["jlocal_em"]) for r in sub) / len(sub),
                "routed_em": sum(bool(r["fallback_em"] if fl else r["jlocal_em"]) for r, fl in zip(sub, flag)) / len(sub),
                "flag_rate": float(flag.mean()),
            }

    for summ in w12_summaries:
        mtag = summ.get("model_tag", "unknown")
        rows: List[Tuple[int, float, int, bool, bool]] = []
        for rec in summ.get("curve_records", []):
            # Evaluate at T - b >= 256 (uncoupled direct vs coupled bridge)
            for reg_n, qk in (("uncoupled_sniah", "q_direct"), ("coupled_bridge", "q_bridge")):
                if reg_n not in rec["regimes"]:
                    continue
                for pt in rec["regimes"][reg_n]["curve"]:
                    if pt["horizon"] < 256 or qk not in pt:
                        continue
                    qo = pt[qk]
                    f = dict(qo["o1_feats"])
                    f["rel_delta_s_mean"] = pt["state_metrics"]["delta_S_b_frob"] / max(1e-12, pt["state_metrics"]["S_old_frob"])
                    y_fail = int(not qo["jlocal"]["contains_new"])
                    r_sc = prop1_analytical_risk_score(f)
                    fl = prop1_analytical_flag(f, pt.get("attn_metrics", {}).get("attn_blk_max"))
                    rows.append((y_fail, r_sc, fl, bool(qo["jlocal"]["contains_new"]), bool(qo["recomp"]["contains_new"])))
        if rows:
            y = np.array([r[0] for r in rows])
            s = np.array([r[1] for r in rows])
            fl = np.array([r[2] for r in rows])
            tp = int(((fl == 1) & (y == 1)).sum())
            fp = int(((fl == 1) & (y == 0)).sum())
            rep["w12_coupled_vs_uncoupled"][mtag] = {
                "n_eval": len(rows),
                "n_fail": int((y == 1).sum()),
                "auroc": _auroc_np(y, s) if (y == 1).any() and (y == 0).any() else None,
                "tpr": tp / max(1, int((y == 1).sum())),
                "fpr": fp / max(1, int((y == 0).sum())),
                "raw_jlocal_em": sum(r[3] for r in rows) / len(rows),
                "routed_em": sum((r[4] if f_i else r[3]) for r, f_i in zip(rows, fl)) / len(rows),
            }
    return rep


def aggregate_curve_records(records: List[Dict[str, Any]], l_grid: List[int]) -> Dict[str, Any]:
    """Summarize median/mean/p95 across samples for each regime and horizon."""
    agg: Dict[str, Any] = {}
    if not records:
        return agg
    reg_names = list(records[0]["regimes"].keys())
    for reg in reg_names:
        rows_by_h: Dict[int, List[Dict[str, Any]]] = {h: [] for h in l_grid}
        slow_fracs = [
            r["regimes"][reg]["n_slow_heads"] / max(1, r["regimes"][reg]["n_total_heads"])
            for r in records
        ]
        for r in records:
            for pt in r["regimes"][reg]["curve"]:
                rows_by_h[int(pt["horizon"])].append(pt)

        h_summary: List[Dict[str, Any]] = []
        for h in l_grid:
            pts = rows_by_h[h]
            if not pts:
                continue

            def _stat(fn) -> Dict[str, float]:
                vals = [float(v) for v in (fn(p) for p in pts) if v is not None]
                if not vals:
                    return {}
                arr = np.array(vals, dtype=np.float64)
                return {
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "p95": float(np.percentile(arr, 95)),
                    "max": float(arr.max()),
                }

            entry: Dict[str, Any] = {
                "horizon": int(h),
                "n": len(pts),
                "identity_max_abs_err": max(p["state_metrics"]["identity_max_abs_err"] for p in pts),
                "S_old_frob": _stat(lambda p: p["state_metrics"]["S_old_frob"]),
                "S_old_vs_b": _stat(lambda p: p["state_metrics"]["S_old_vs_b"]),
                "E_hom_rel_b": _stat(lambda p: p["state_metrics"]["E_hom_rel_b"]),
                "gamma_pred_rel_b": _stat(lambda p: p["state_metrics"]["gamma_pred_rel_b"]),
                "E_forcing_frob": _stat(lambda p: p["state_metrics"]["E_forcing_frob"]),
                "E_forcing_rel_b": _stat(lambda p: p["state_metrics"]["E_forcing_rel_b"]),
                "E_forcing_rel_S": _stat(lambda p: p["state_metrics"]["E_forcing_rel_S"]),
                "E_res_jlocal_rel_b": _stat(lambda p: p["state_metrics"]["E_res_jlocal_rel_b"]),
                "E_res_jlocal_rel_S": _stat(lambda p: p["state_metrics"]["E_res_jlocal_rel_S"]),
                "E_res_stale_rel_b": _stat(lambda p: p["state_metrics"]["E_res_stale_rel_b"]),
                "pre_attn_E_hom_rel_b": _stat(lambda p: p["state_metrics"]["pre_attn_E_hom_rel_b"]),
                "pre_attn_E_forcing_frob": _stat(lambda p: p["state_metrics"]["pre_attn_E_forcing_frob"]),
                "layer0_E_hom_rel_b": _stat(lambda p: p["state_metrics"].get("layer0_E_hom_rel_b")),
                "layer0_gamma_pred_rel_b": _stat(lambda p: p["state_metrics"].get("layer0_gamma_pred_rel_b")),
                "layer0_E_forcing_frob": _stat(lambda p: p["state_metrics"].get("layer0_E_forcing_frob")),
                "post_attn_E_forcing_rel_b": _stat(lambda p: p["state_metrics"]["post_attn_E_forcing_rel_b"]),
                "slow_heads_E_hom_rel_b": _stat(lambda p: p["state_metrics"]["slow_heads_E_hom_rel_b"]),
                "slow_heads_gamma_pred_rel_b": _stat(lambda p: p["state_metrics"]["slow_heads_gamma_pred_rel_b"]),
                "fast_heads_E_hom_rel_b": _stat(lambda p: p["state_metrics"]["fast_heads_E_hom_rel_b"]),
                "attn_blk_max": _stat(lambda p: p.get("attn_metrics", {}).get("attn_blk_max")),
                "val_attn_max": _stat(lambda p: p.get("attn_metrics", {}).get("val_attn_max")),
                "val_attn_mean": _stat(lambda p: p.get("attn_metrics", {}).get("val_attn_mean")),
                "q_direct_jlocal_em": sum(p["q_direct"]["jlocal"]["contains_new"] for p in pts) / len(pts),
                "q_direct_kv_em": sum(p["q_direct"]["kv_only_blk"]["contains_new"] for p in pts) / len(pts),
                "q_direct_recomp_em": sum(p["q_direct"]["recomp"]["contains_new"] for p in pts) / len(pts),
                "q_direct_jlocal_kl": _stat(lambda p: p["q_direct"]["jlocal"]["kl_vs_oneshot"]),
                "q_direct_kv_kl": _stat(lambda p: p["q_direct"]["kv_only_blk"]["kl_vs_oneshot"]),
            }
            if all("q_bridge" in p for p in pts):
                entry["q_bridge_jlocal_em"] = sum(p["q_bridge"]["jlocal"]["contains_new"] for p in pts) / len(pts)
                entry["q_bridge_recomp_em"] = sum(p["q_bridge"]["recomp"]["contains_new"] for p in pts) / len(pts)
                entry["q_bridge_jlocal_kl"] = _stat(lambda p: p["q_bridge"]["jlocal"]["kl_vs_oneshot"])
            h_summary.append(entry)
        agg[reg] = {
            "slow_heads_frac_mean": float(np.mean(slow_fracs)),
            "by_horizon": h_summary,
        }
    return agg


def aggregate_impulse_records(records: List[Dict[str, Any]], l_grid: List[int]) -> Dict[str, Any]:
    if not records:
        return {}
    keys = list(records[0]["trajectories"].keys())
    out: Dict[str, Any] = {}
    for tk in keys:
        pts_by_h: Dict[int, List[Dict[str, Any]]] = {h: [] for h in l_grid}
        for r in records:
            for pt in r["trajectories"][tk]:
                pts_by_h[int(pt["horizon"])].append(pt)
        rows = []
        for h in l_grid:
            pts = pts_by_h[h]
            if not pts:
                continue

            def _mean(k):
                vals = [float(p[k]) for p in pts if p.get(k) is not None]
                return float(np.mean(vals)) if vals else None

            def _max(k):
                vals = [float(p[k]) for p in pts if p.get(k) is not None]
                return float(np.max(vals)) if vals else None

            rows.append({
                "horizon": int(h),
                "n": len(pts),
                "rel_pert_mean": _mean("rel_pert_norm"),
                "rel_pert_max": _max("rel_pert_norm"),
                "gamma_pred_mean": _mean("gamma_pred_rel_norm"),
                "layer0_rel_mean": _mean("layer0_rel_pert"),
                "pre_attn_rel_mean": _mean("pre_attn_rel_pert"),
                "post_attn_rel_mean": _mean("post_attn_rel_pert"),
                "slow_heads_rel_mean": _mean("slow_heads_rel_pert"),
                "fast_heads_rel_mean": _mean("fast_heads_rel_pert"),
                "kl_vs_clean_mean": _mean("kl_vs_clean"),
            })
        out[tk] = rows
    return out


def run_synthetic_dry_run(out_dir: Path, device: str = "cpu") -> Dict[str, Any]:
    """Tiny model unit test verifying exact telescoping identity, Layer-0 zero forcing, hooks, Mamba-2 & OSC."""
    seed_all(202612)
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

    tcfg = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention"],
        max_position_embeddings=1024,
        rope_parameters={"rope_theta": 10000.0},
    )
    qmod = Qwen3_5ForCausalLM(tcfg).to(device).eval()
    class _DummyTok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(str(x % 10) for x in ids[:4])

        def encode(self, txt, add_special_tokens=False):
            return [20 + (ord(ch) % 20) for ch in str(txt)[:4]] or [21]

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return str(msgs[-1]["content"]) + " "

    tok = _DummyTok()
    p0, p1, b_end = 12, 16, 28
    ids_o = torch.randint(10, 200, (1, b_end + 32), dtype=torch.long, device=device)
    ids_e = ids_o.clone()
    ids_e[:, p0:p1] = torch.tensor([[21, 22, 23, 24]], dtype=torch.long, device=device)
    syn_samp = {
        "sample_index": 0, "target": "Alice",
        "old_val": "1111", "new_val": "2222", "ans_old": "3333", "ans_new": "4444",
        "f_o": int(ids_o[0, p0].item()), "f_e": int(ids_e[0, p0].item()),
        "b_o": 15, "b_e": 16, "p0": p0, "p1": p1, "b_end": b_end, "bridge_span_rel": [8, 16],
        "ids_o_unc": ids_o, "ids_e_unc": ids_e, "ids_o_cup": ids_o, "ids_e_cup": ids_e,
        "ids_q_direct": torch.randint(10, 200, (1, 6), dtype=torch.long, device=device),
        "ids_q_bridge": torch.randint(10, 200, (1, 6), dtype=torch.long, device=device),
    }
    base_m = getattr(qmod, "model", qmod)
    pre_idx = {0, 1, 2}
    res_q = eval_qwen_prop1_sample(
        qmod, base_m, tok, [1], syn_samp, n_linear=4, d_k=16, w_conv=4,
        pre_attn_idx=pre_idx, l_grid=[0, 8, 16, 32], device=device, use_l2k=True, verify_restore=True,
    )
    res_imp = eval_qwen_impulse_sample(
        qmod, base_m, tok, [1], syn_samp, n_linear=4, d_k=16,
        pre_attn_idx=pre_idx, l_grid=[0, 8, 16, 32], eps_list=[0.1, 1.0], device=device, use_l2k=True,
    )
    # Check exact physical invariants on synthetic GDN!
    curve = res_q["regimes"]["uncoupled_sniah"]["curve"]
    max_id_err = max(pt["state_metrics"]["identity_max_abs_err"] for pt in curve)
    max_pre_forcing = max((pt["state_metrics"]["pre_attn_E_forcing_frob"] or 0.0) for pt in curve)
    post_forcing_end = curve[-1]["state_metrics"]["post_attn_E_forcing_rel_b"]
    assert max_id_err <= 1e-11, f"Telescoping identity failed: {max_id_err}"
    assert max_pre_forcing == 0.0, f"Pre-attn linear layers (0..2) must have ZERO KV forcing, got {max_pre_forcing}"

    # Verify Mamba-2 synthetic Falcon-H1 pipeline cleanly too (CPU or device)
    fcfg = FalconH1Config(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        mamba_d_ssm=128, mamba_n_heads=4, mamba_d_head=32, mamba_n_groups=1,
        mamba_d_state=16, mamba_d_conv=4, mamba_chunk_size=16,
        ssm_multipliers=[0.35, 0.25, 0.18, 0.5, 0.35],
    )
    fmod = FalconH1ForCausalLM(fcfg).to(device).eval()
    mixers = mf0.extract_mamba_mixers(fmod)
    res_f = eval_mamba_prop1_sample(fmod, fcfg, tok, mixers, [1], syn_samp, set(), [0, 8, 16, 32], device)
    res_f_imp = eval_mamba_impulse_sample(fmod, fcfg, mixers, syn_samp, set(), [0, 8, 16, 32], [0.1, 1.0], device)
    f_id_err = max(pt["state_metrics"]["identity_max_abs_err"] for pt in res_f["regimes"]["uncoupled_sniah"]["curve"])
    assert f_id_err <= 1e-11, f"Mamba2 telescoping failed: {f_id_err}"

    osc_w11 = eval_osc_on_w11_and_w12(out_dir, [{"model_tag": "syn_qwen", "curve_records": [res_q]}])
    out = {
        "status": "DRY_RUN_PASS",
        "qwen_identity_max_err": max_id_err,
        "qwen_pre_attn_layers_forcing_max": max_pre_forcing,
        "qwen_post_attn_forcing_end": post_forcing_end,
        "qwen_impulse_eps1_end_rel_pert": res_imp["trajectories"]["isotropic_random:eps=1.00"][-1]["rel_pert_norm"],
        "falcon_identity_max_err": f_id_err,
        "falcon_impulse_eps1_end_rel_pert": res_f_imp["trajectories"]["isotropic_random:eps=1.00"][-1]["rel_pert_norm"],
        "osc_on_existing_w11": osc_w11["w11_splits"],
    }
    return out


def run_qwen_model(
    model_id: str, tag: str, out_dir: Path, device: str,
    n_curve: int, n_impulse: int, n_warm: int,
) -> Dict[str, Any]:
    seed_all(202612)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device, trust_remote_code=True).eval()
    base_model = getattr(model, "model", model)
    t_cfg = getattr(base_model.config, "text_config", base_model.config)
    n_linear = qwen_r.count_linear_layers(t_cfg)
    d_k = int(t_cfg.linear_key_head_dim)
    w_conv = int(getattr(t_cfg, "linear_conv_kernel_dim", 4))
    ltypes = list(getattr(t_cfg, "layer_types", []))
    first_full = next((i for i, lt in enumerate(ltypes) if lt == "full_attention"), len(ltypes))
    linear_before_full = sum(1 for lt in ltypes[:first_full] if lt == "linear_attention")
    pre_attn_idx = set(range(linear_before_full))
    eos_ids = qwen_r.get_eos_ids(model, tok)
    use_l2k = qwen_r.has_logits_to_keep_arg(model)
    pool_d = qwen_r.build_digit4_pool()
    rng = random.Random(202612)

    jsonl_curve = out_dir / f"w12_prop1_curves_{tag}_records.jsonl"
    jsonl_imp = out_dir / f"w12_prop1_impulse_{tag}_records.jsonl"
    jsonl_warm = out_dir / f"w12_prop1_warm_{tag}_records.jsonl"
    done_c, done_i, done_w = load_done_keys(jsonl_curve), load_done_keys(jsonl_imp), load_done_keys(jsonl_warm)

    # 1. Curve suite (n_curve paired uncoupled vs coupled_bridge)
    s_idx = 0
    built_c = 0
    samples_cache: List[Dict[str, Any]] = []
    while built_c < max(n_curve, n_impulse) and s_idx < 300:
        samp = build_aligned_paired_sample(tok, "qwen", s_idx, rng, pool_d, n_linear, w_conv, QWEN_L_GRID, p0_target_fillers=2, device=device)
        s_idx += 1
        if samp is None:
            continue
        samples_cache.append(samp)
        built_c += 1

    for idx in range(min(n_curve, len(samples_cache))):
        rk = f"{tag}:curve:{idx}"
        if rk in done_c:
            continue
        t0 = sync_dev(device)
        rec = eval_qwen_prop1_sample(
            model, base_model, tok, eos_ids, samples_cache[idx], n_linear, d_k, w_conv,
            pre_attn_idx, QWEN_L_GRID, device, use_l2k, verify_restore=(idx == 0),
        )
        rec["record_key"] = rk
        rec["wall_s"] = sync_dev(device) - t0
        append_jsonl(jsonl_curve, rec)
        print(f"[{tag}] curve {idx+1}/{n_curve} wall={rec['wall_s']:.1f}s", flush=True)

    # 2. Controlled impulse perturbation injection
    for idx in range(min(n_impulse, len(samples_cache))):
        rk = f"{tag}:impulse:{idx}"
        if rk in done_i:
            continue
        t0 = sync_dev(device)
        rec = eval_qwen_impulse_sample(
            model, base_model, tok, eos_ids, samples_cache[idx], n_linear, d_k,
            pre_attn_idx, QWEN_L_GRID, [0.10, 0.50, 2.00], device, use_l2k,
        )
        rec["record_key"] = rk
        rec["wall_s"] = sync_dev(device) - t0
        append_jsonl(jsonl_imp, rec)
        print(f"[{tag}] impulse {idx+1}/{n_impulse} wall={rec['wall_s']:.1f}s", flush=True)

    # 3. Warm long-prefix check (p0 ~ 2048 -> check ||S_T^{old}||_F equilibrium & forcing)
    warm_i = 0
    attempts = 0
    while warm_i < n_warm and attempts < 100:
        rk = f"{tag}:warm:{warm_i}"
        attempts += 1
        if rk in done_w:
            warm_i += 1
            continue
        samp = build_aligned_paired_sample(tok, "qwen", 500 + attempts, rng, pool_d, n_linear, w_conv, WARM_L_GRID, p0_target_fillers=165, device=device)
        if samp is None or samp["p0"] < 1800:
            continue
        t0 = sync_dev(device)
        rec = eval_qwen_prop1_sample(
            model, base_model, tok, eos_ids, samp, n_linear, d_k, w_conv,
            pre_attn_idx, WARM_L_GRID, device, use_l2k, verify_restore=False,
        )
        rec["record_key"] = rk
        rec["wall_s"] = sync_dev(device) - t0
        append_jsonl(jsonl_warm, rec)
        print(f"[{tag}] warm_p0={samp['p0']} {warm_i+1}/{n_warm} wall={rec['wall_s']:.1f}s", flush=True)
        warm_i += 1

    c_recs = [json.loads(l) for l in jsonl_curve.read_text(encoding="utf-8").splitlines() if l.strip()]
    i_recs = [json.loads(l) for l in jsonl_imp.read_text(encoding="utf-8").splitlines() if l.strip()]
    w_recs = [json.loads(l) for l in jsonl_warm.read_text(encoding="utf-8").splitlines() if l.strip()] if jsonl_warm.exists() else []

    summary = {
        "model_id": model_id,
        "model_tag": tag,
        "family": "gdn",
        "n_linear": n_linear,
        "pre_attn_linear_indices": sorted(pre_attn_idx),
        "l_grid": QWEN_L_GRID,
        "curve_summary": aggregate_curve_records(c_recs, QWEN_L_GRID),
        "impulse_summary": aggregate_impulse_records(i_recs, QWEN_L_GRID),
        "warm_summary": aggregate_curve_records(w_recs, WARM_L_GRID) if w_recs else {},
        "curve_records": c_recs,
    }
    # Write lightweight summary (omit verbose full curve_records from JSON copy? Keep compact!)
    summ_to_save = {k: v for k, v in summary.items() if k != "curve_records"}
    (out_dir / f"w12_prop1_{tag}_summary.json").write_text(json.dumps(summ_to_save, indent=2), encoding="utf-8")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def run_mamba_model(
    model_id: str, tag: str, out_dir: Path, device: str,
    n_curve: int, n_impulse: int,
) -> Dict[str, Any]:
    seed_all(202612)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"[{tag}] Loading tokenizer & weights ({model_id})...", flush=True)
    t_ld = time.perf_counter()
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device, local_files_only=True).eval()
    except OSError:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device).eval()
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    print(f"[{tag}] Loaded in {time.perf_counter()-t_ld:.1f}s", flush=True)
    cfg = model.config
    mixers = mf0.extract_mamba_mixers(model)
    ltypes = list(getattr(cfg, "layers_block_type", []))
    first_hyb = next((i for i, t in enumerate(ltypes) if t in ("hybrid", "full_attention")), len(ltypes))
    pre_attn_idx = {li for li, _ in mixers if li < first_hyb}
    w_conv = int(getattr(cfg, "mamba_d_conv", getattr(cfg, "conv_kernel_size", 4)))
    eos_ids = mf0.get_eos_ids(model, tok)
    pool_d = qwen_r.build_digit4_pool()
    rng = random.Random(202612)

    jsonl_curve = out_dir / f"w12_prop1_curves_{tag}_records.jsonl"
    jsonl_imp = out_dir / f"w12_prop1_impulse_{tag}_records.jsonl"
    done_c, done_i = load_done_keys(jsonl_curve), load_done_keys(jsonl_imp)

    samples: List[Dict[str, Any]] = []
    s_idx = 0
    while len(samples) < max(n_curve, n_impulse) and s_idx < 300:
        samp = build_aligned_paired_sample(tok, tag, s_idx, rng, pool_d, len(mixers), w_conv, MAMBA_L_GRID, p0_target_fillers=2, device=device)
        s_idx += 1
        if samp is not None:
            samples.append(samp)
    print(f"[{tag}] Built {len(samples)} aligned samples (mixers={len(mixers)}, pre_attn={sorted(pre_attn_idx)})", flush=True)
    assert len(samples) > 0, f"[{tag}] Built 0 aligned samples!"

    for idx in range(min(n_curve, len(samples))):
        rk = f"{tag}:curve:{idx}"
        if rk in done_c:
            continue
        t0 = sync_dev(device)
        rec = eval_mamba_prop1_sample(model, cfg, tok, mixers, eos_ids, samples[idx], pre_attn_idx, MAMBA_L_GRID, device)
        rec["record_key"] = rk
        rec["wall_s"] = sync_dev(device) - t0
        append_jsonl(jsonl_curve, rec)
        print(f"[{tag}] curve {idx+1}/{n_curve} wall={rec['wall_s']:.1f}s", flush=True)

    for idx in range(min(n_impulse, len(samples))):
        rk = f"{tag}:impulse:{idx}"
        if rk in done_i:
            continue
        t0 = sync_dev(device)
        rec = eval_mamba_impulse_sample(model, cfg, mixers, samples[idx], pre_attn_idx, MAMBA_L_GRID, [0.10, 0.50, 2.00], device)
        rec["record_key"] = rk
        rec["wall_s"] = sync_dev(device) - t0
        append_jsonl(jsonl_imp, rec)
        print(f"[{tag}] impulse {idx+1}/{n_impulse} wall={rec['wall_s']:.1f}s", flush=True)

    c_recs = [json.loads(l) for l in jsonl_curve.read_text(encoding="utf-8").splitlines() if l.strip()]
    i_recs = [json.loads(l) for l in jsonl_imp.read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = {
        "model_id": model_id,
        "model_tag": tag,
        "family": "mamba2",
        "n_linear": len(mixers),
        "pre_attn_linear_indices": sorted(pre_attn_idx),
        "l_grid": MAMBA_L_GRID,
        "curve_summary": aggregate_curve_records(c_recs, MAMBA_L_GRID),
        "impulse_summary": aggregate_impulse_records(i_recs, MAMBA_L_GRID),
        "curve_records": c_recs,
    }
    summ_to_save = {k: v for k, v in summary.items() if k != "curve_records"}
    (out_dir / f"w12_prop1_{tag}_summary.json").write_text(json.dumps(summ_to_save, indent=2), encoding="utf-8")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-dry-run", action="store_true")
    p.add_argument("--eval-osc-only", action="store_true")
    p.add_argument("--models", type=str, default="qwen2b,falcon,zamba2")
    p.add_argument("--n-curve", type=int, default=8)
    p.add_argument("--n-impulse", type=int, default=5)
    p.add_argument("--n-warm", type=int, default=5)
    p.add_argument("--out-dir", type=str, default=str(PILOT_DIR))
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic_dry_run:
        res = run_synthetic_dry_run(out_dir, device=args.device)
        print(json.dumps(res, indent=2))
        return

    summaries: List[Dict[str, Any]] = []
    if not args.eval_osc_only:
        req = [m.strip().lower() for m in args.models.split(",") if m.strip()]
        for m in req:
            if m == "qwen2b":
                summaries.append(run_qwen_model("Qwen/Qwen3.5-2B", "2b", out_dir, args.device, args.n_curve, args.n_impulse, args.n_warm))
            elif m == "qwen9b":
                summaries.append(run_qwen_model("Qwen/Qwen3.5-9B", "9b", out_dir, args.device, args.n_curve, args.n_impulse, args.n_warm))
            elif m == "falcon":
                summaries.append(run_mamba_model("tiiuae/Falcon-H1-1.5B-Instruct", "falcon", out_dir, args.device, args.n_curve, args.n_impulse))
            elif m == "zamba2":
                summaries.append(run_mamba_model("Zyphra/Zamba2-1.2B-Instruct-v2", "zamba2", out_dir, args.device, args.n_curve, args.n_impulse))
            else:
                raise ValueError(f"Unknown model key: {m}")
    else:
        for tag in ("2b", "9b", "falcon", "zamba2"):
            jp = out_dir / f"w12_prop1_curves_{tag}_records.jsonl"
            if jp.exists():
                recs = [json.loads(l) for l in jp.read_text(encoding="utf-8").splitlines() if l.strip()]
                summaries.append({"model_tag": tag, "curve_records": recs})

    osc = eval_osc_on_w11_and_w12(out_dir, summaries)
    (out_dir / "w12_prop1_osc_summary.json").write_text(json.dumps(osc, indent=2), encoding="utf-8")
    print(json.dumps(osc, indent=2))


if __name__ == "__main__":
    main()
