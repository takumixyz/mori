"""Generate hash-pinned Stage-C & E5/E9 Evidence Bundle Manifest (`pilot/w12_c_evidence_manifest.json`).

Addresses Blocker B2 (raw JSONL provenance gap under `.gitignore:8-9`):
  - Computes exact byte sizes and SHA-256 digests for all Stage C, Wave 5-8, Wave 10-12 scripts,
    summaries, and raw JSONL logs (even though `*.jsonl` is gitignored).
  - Verifies exact byte-prefix provenance (`w8_2b_{16k,32k}_n50_records.jsonl` == prefix of
    `w12_2b_{16k,32k}_n200_records.jsonl`) and extracts `run_metadata` resume counts verbatim.
  - Pins ordered task/record-key digests, serving case-key reuse checks, git tracked vs ignored
    status, local Python/torch/transformers/FLA versions, and saved Colab runtime metadata verbatim
    from existing artifacts (never fabricates unrecorded metadata).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
PILOT_DIR = ROOT_DIR / "pilot"
REFINE_DIR = ROOT_DIR / "refine-logs"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_lf_bytes(path: Path) -> Dict[str, Any]:
    """Return both raw workspace bytes/sha256 and LF-normalized Git-blob-matching bytes/sha256."""
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return {
        "workspace_bytes": len(raw),
        "workspace_sha256": hashlib.sha256(raw).hexdigest(),
        "git_lf_bytes": len(lf),
        "git_lf_sha256": hashlib.sha256(lf).hexdigest(),
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_info() -> Dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True).stdout.strip()
    tracked_raw = subprocess.run(["git", "ls-files"], cwd=ROOT_DIR, capture_output=True, text=True).stdout.splitlines()
    tracked_set = set(tracked_raw)
    head_blobs: Dict[str, str] = {}
    if tracked_raw:
        batch_in = "".join(f"HEAD:{p}\n" for p in tracked_raw).encode("utf-8")
        proc = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT_DIR, input=batch_in, capture_output=True)
        raw = proc.stdout
        pos = 0
        for p in tracked_raw:
            nl = raw.find(b"\n", pos)
            if nl < 0:
                break
            hdr = raw[pos:nl].decode("utf-8", errors="replace").split()
            pos = nl + 1
            if len(hdr) >= 3 and hdr[1] == "blob":
                sz = int(hdr[2])
                content = raw[pos : pos + sz]
                head_blobs[p] = hashlib.sha256(content).hexdigest()
                pos += sz + 1  # skip trailing \n
    return {"head_at_generation": head, "tracked_set": tracked_set, "head_blob_sha256s": head_blobs}


RAW_JSONL_DRIVE_ARCHIVE: Dict[str, Any] = {
    "drive_folder_id": "1_PkgLg1_qoxz2VXtuwEnzmAFbXPsXK_T",
    "drive_folder_url": "https://drive.google.com/drive/folders/1_PkgLg1_qoxz2VXtuwEnzmAFbXPsXK_T",
    "drive_path": "Antigravity_Projects/focused-shannon/pilot",
    "files": {
        "pilot/w5_9b_e5_4k_o1_records.jsonl": "1Wibc4VaH10r940APcQMwHkoXQUDAGlbe",
        "pilot/w5_9b_e5_16k_o1_records.jsonl": "1i2EmZRkBXN1Z171IXoJ-YqMI2CqN9XtM",
        "pilot/w5_9b_e5_hybrid_n50_records.jsonl": "1wQlNGjKN5UpcJMJ_q9a_1UTkF1m7Kat-",
        "pilot/w6_9b_32k_probe_records.jsonl": "1ubzdn8LAzADMiPiyah1dskpEDKFMFCzp",
        "pilot/w6_9b_32k_n50_records.jsonl": "1vMFcmngT8-IlLvAyGmqB2_C2miu-Quum",
        "pilot/w7_e9_27b_results.jsonl": "1Mnm91sVYmCrl8oLWcJxe5efKeVb8nt3z",
        "pilot/w8_2b_16k_n50_records.jsonl": "1ML0wNpPy2uM5y_-jgLx2RGEIXhvNV9wI",
        "pilot/w8_2b_32k_n50_records.jsonl": "1rdsG8gupnzjLVHBXcx3dHqDtLAodS1za",
        "pilot/w11_crossfamily_e2_records.jsonl": "14yObVF44wqoaEL-tQg4HDSAelu1ty2Kb",
        "pilot/w11_crossfamily_e3_records.jsonl": "1xOHeemNVxKGYAhJWS_9oN3Q3OUYuXIEx",
        "pilot/w11_crossfamily_sb1_records.jsonl": "1dt980t-nCX71kTF6uCFHqlzitYT4qivc",
        "pilot/w11_detector_v2_clean_2b_records.jsonl": "11BylW1P3my_YTw45ReSkGIP3ND-gaq5r",
        "pilot/w11_detector_v2_clean_9b_records.jsonl": "12iQbKQsnzuB6929WJv1hSlrBU-J51Ksw",
        "pilot/w12_2b_16k_n200_records.jsonl": "1fI6Y4rhG4TDXqvAWx9lIFKgYFLKce8jL",
        "pilot/w12_2b_32k_n200_records.jsonl": "15T1mJf5oTjr5vRoTyH-ZauIfwBcgasn3",
        "pilot/w13_fallback_2b_osc_records.jsonl": "1JlgDg2SuIlAKUYIcs59oouFfEaVF5GuD",
        "pilot/w13_fallback_2b_16k_osc_records.jsonl": "1YpROd9L6r6ZIGvNy1HmXwIMTT1mMZrl-",
        "pilot/w13_fallback_9b_lr_records.jsonl": "19E-PLIpHwaIFOS2CmBTNAM4RVqlZmWlH",
        "pilot/w12_c_nvme_int8_2b_records.jsonl": "1Py9Dr-Owt2VnkJEg8Osz9lIEPQIWu6Jv",
        "pilot/w12_c_nvme_int8_9b_records.jsonl": "1tX8nC5misluKrQlRWHX1677ebIetAU3l",
        "pilot/w12_c_jamba3b_gate1_records.jsonl": "1FiH-FAtZyF03RWoFahWcDGneMcpwV3uV",
        "pilot/w14_optb_coupled_nvme_2b_l4_records.jsonl": "1vc9d6Fad_ovIdS94TPLYV4za2oqU9-nK",
        "pilot/w14_optb_coupled_nvme_9b_l4_records.jsonl": "16g-KrCq3YsxRqC347H5g_ialQuAM99uN",
        "pilot/w13_fallback_9b_osc_optb_records.jsonl": "1NBMcv-xp8ptb8WIF4ps0cuPD1EhqK4g2",
        "pilot/w13_fallback_9b_16k_lr_optb_records.jsonl": "1jlbfOilWlW8o6MSsWU2P7b-RdLQ1GZAn",
        "pilot/w13_fallback_9b_16k_osc_optb_records.jsonl": "1drp-72gGsO52UxOWCsDXaQGJx0yRW6wt",
        "pilot/w14_optb_live_16k_unique_9b_l4_records.jsonl": "1zXc4tCB3l52m41YAxf8dc01CYjFJ-Aj0",
    },
}


def _jsonl_digest_info(path: Path) -> Dict[str, Any]:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    keys: List[str] = []
    suites: Dict[str, int] = {}
    for idx, line in enumerate(lines):
        obj = json.loads(line)
        if "record_key" in obj:
            keys.append(str(obj["record_key"]))
        elif "sample_index" in obj and "depth" in obj:
            tlen = obj.get("target_length", obj.get("context_length", obj.get("T_ctx")))
            keys.append(f"{tlen}:{obj.get('sample_index')}:{obj.get('depth')}:{obj.get('tpl_id')}:{obj.get('old')}:{obj.get('new')}")
        elif "sample_idx" in obj and "cell" in obj:
            keys.append(f"{obj.get('suite')}:{obj.get('cell')}:{obj.get('sample_idx')}:{obj.get('depth')}")
        elif "sample_id" in obj:
            keys.append(f"{obj.get('suite', obj.get('task', 'rec'))}:{obj.get('sample_id')}:{obj.get('cell', idx)}")
        else:
            keys.append(str(idx))
        s = str(obj.get("suite") or obj.get("mode") or "default")
        suites[s] = suites.get(s, 0) + 1
    return {
        "line_count": len(lines),
        "unique_key_count": len(set(keys)),
        "ordered_keys_sha256": _sha256_text("\n".join(keys)),
        "suite_counts": suites,
    }


def _file_entry(path: Path, tracked_set: set, head_blob_hashes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    rel = path.relative_to(ROOT_DIR).as_posix()
    dig = _sha256_lf_bytes(path)
    git_blob_sha = (head_blob_hashes or {}).get(rel)
    rec: Dict[str, Any] = {
        "path": rel,
        "bytes": dig["workspace_bytes"],
        "sha256": dig["workspace_sha256"],
        "workspace_bytes": dig["workspace_bytes"],
        "workspace_sha256": dig["workspace_sha256"],
        "git_lf_bytes": dig["git_lf_bytes"],
        "git_lf_sha256": dig["git_lf_sha256"],
        "git_tracked": rel in tracked_set,
    }
    if git_blob_sha:
        rec["git_head_blob_sha256"] = git_blob_sha
        rec["git_head_matches_lf_or_raw"] = bool(
            git_blob_sha in (dig["workspace_sha256"], dig["git_lf_sha256"])
        )
    if path.suffix == ".jsonl":
        rec["jsonl"] = _jsonl_digest_info(path)
        did = RAW_JSONL_DRIVE_ARCHIVE["files"].get(rel)
        if did:
            rec["external_drive_file_id"] = did
            rec["external_drive_url"] = f"https://drive.google.com/file/d/{did}/view"
    return rec


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed_in_manifest_python"


def _probe_stateedit_env() -> Dict[str, Any]:
    py = Path(r"D:\environment\python\pyenvs\stateedit\python.exe")
    if not py.exists():
        return {"status": "not_found_at_expected_path"}
    code = (
        "import json, sys, importlib.metadata as m; "
        "pkgs=['torch','transformers','flash-linear-attention','triton-windows','bitsandbytes','numpy','scipy']; "
        "out={'python': sys.version.split()[0]}; "
        "[out.__setitem__(p.replace('-','_'), m.version(p)) if any(d.metadata['Name'] and d.metadata['Name'].lower()==p for d in m.distributions()) else out.__setitem__(p.replace('-','_'), 'not_installed') for p in pkgs]; "
        "print(json.dumps(out))"
    )
    res = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        return json.loads(res.stdout.strip())
    return {"status": "probe_failed", "stderr": res.stderr.strip()[:200]}


def _load_json_opt(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _extract_script_argparse_flags(script_path: Path) -> set[str]:
    """Parse all `--flag` option strings registered via `add_argument(...)` in a Python script via AST."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"):
                    flags.add(arg.value)
    return flags


def _validate_cli_commands_ast(cmds: Dict[str, str]) -> Dict[str, Any]:
    """Assert every CLI `--flag` in reconstructed_cli_commands exists in its target script's argparse definition."""
    checked = 0
    invalid: List[Dict[str, Any]] = []
    for key, cmd in cmds.items():
        if key.endswith("_NOTE"):
            continue
        # Handle PowerShell `$env:VAR='val'; python ...` prefix cleanly
        exec_part = cmd.split(";", 1)[-1].strip() if ";" in cmd else cmd.strip()
        toks = shlex.split(exec_part, posix=False)
        script_idx = next((i for i, t in enumerate(toks) if t.replace("\\", "/").startswith("pilot/") and t.endswith(".py")), None)
        if script_idx is None:
            continue
        script_rel = toks[script_idx].replace("\\", "/")
        script_path = ROOT_DIR / script_rel
        assert script_path.is_file(), f"Missing script in CLI command {key}: {script_rel}"
        allowed = _extract_script_argparse_flags(script_path)
        bad_flags = [t for t in toks[script_idx + 1 :] if t.startswith("--") and t not in allowed]
        checked += 1
        if bad_flags:
            invalid.append({"command_key": key, "script": script_rel, "invalid_flags": bad_flags})
    assert not invalid, f"Invalid CLI flags detected in reconstructed_cli_commands: {invalid}"
    return {"commands_validated": checked, "invalid_flags": invalid}


def main() -> None:
    ginfo = _git_info()
    tracked_set = ginfo["tracked_set"]

    artifact_globs = [
        "w5_9b_e5_runner.py",
        "w5_9b_e5_4k_o1_records.jsonl",
        "w5_9b_e5_4k_o1_summary.json",
        "w5_9b_e5_16k_o1_records.jsonl",
        "w5_9b_e5_16k_o1_summary.json",
        "w5_9b_e5_hybrid_n50_records.jsonl",
        "w5_9b_e5_hybrid_n50_summary.json",
        "w6_9b_32k_probe_records.jsonl",
        "w6_9b_32k_probe_summary.json",
        "w6_9b_32k_n50_records.jsonl",
        "w6_9b_32k_n50_summary.json",
        "w7_e9_27b_runner.py",
        "w7_e9_27b_results.jsonl",
        "w7_e9_27b_results_summary.json",
        "w7_e9_27b_fla_env_report.json",
        "w0_e8_suffix_recompute_bench.py",
        "w0_e8_2b_results.json",
        "w8_e8_2b_l4_results.json",
        "w8_2b_16k_n50_records.jsonl",
        "w8_2b_16k_n50_summary.json",
        "w8_2b_32k_n50_records.jsonl",
        "w8_2b_32k_n50_summary.json",
        "w10_p1_3_e4_e7_counterfact.py",
        "w11_cf_subset_for_colab.json",
        "w11_crossfamily_scaleup.py",
        "w11_crossfamily_scaleup_summary.json",
        "w11_crossfamily_e2_records.jsonl",
        "w11_crossfamily_e3_records.jsonl",
        "w11_crossfamily_sb1_records.jsonl",
        "w11_detector_v2_clean_2b_records.jsonl",
        "w11_detector_v2_clean_9b_records.jsonl",
        "w12_prop1_perturbation_decay.py",
        "w12_prop1_osc_summary.json",
        "w12_2b_16k_n200_records.jsonl",
        "w12_2b_16k_n200_summary.json",
        "w12_2b_32k_n200_records.jsonl",
        "w12_2b_32k_n200_summary.json",
        "w13_fallback_runner.py",
        "w13_fallback_2b_osc_records.jsonl",
        "w13_fallback_2b_osc_summary.json",
        "w13_fallback_2b_osc_tau_plateau.json",
        "w13_fallback_2b_16k_osc_records.jsonl",
        "w13_fallback_2b_16k_osc_summary.json",
        "w13_fallback_9b_lr_records.jsonl",
        "w13_fallback_9b_lr_summary.json",
        "w13_fallback_9b_lr_tau_plateau.json",
        "w12_c_nvme_int8_stress.py",
        "w12_c_nvme_int8_2b_records.jsonl",
        "w12_c_nvme_int8_2b_summary.json",
        "w12_c_nvme_int8_9b_records.jsonl",
        "w12_c_nvme_int8_9b_summary.json",
        "w12_c_offline_pairing_n200.py",
        "w12_c_offline_pairing_n200_results.json",
        "w12_c_third_family.py",
        "w12_c_jamba3b_gate1_records.jsonl",
        "w12_c_jamba3b_gate1_summary.json",
        "w12_c_posthoc_stats.py",
        "w12_c_posthoc_summary.json",
        "w12_c_evidence_manifest.py",
        "w0_e8_suffix_recompute_results.json",
        "w14_optb_e2e_routing_bench.py",
        "w14_optb_e2e_routing_2b_4060_summary.json",
        "w14_optb_e2e_routing_9b_l4_summary.json",
        "w14_optb_symmetric_baselines.py",
        "w14_optb_symmetric_baselines_local_4060_summary.json",
        "w14_optb_symmetric_baselines_small3_a100_summary.json",
        "w14_optb_symmetric_baselines_9b_l4_summary.json",
        "w14_optb_symmetric_baselines_9b_a100_summary.json",
        "w14_optb_coupled_nvme_eval.py",
        "w14_optb_coupled_nvme_2b_l4_summary.json",
        "w14_optb_coupled_nvme_2b_l4_records.jsonl",
        "w14_optb_coupled_nvme_9b_l4_summary.json",
        "w14_optb_coupled_nvme_9b_l4_records.jsonl",
        "w14_optb_jamba_stateful_probe.py",
        "w14_optb_jamba3b_summary.json",
        "w14_optb_jamba_28l_fp32_control_summary.json",
        "w13_fallback_9b_osc_optb_summary.json",
        "w13_fallback_9b_osc_optb_records.jsonl",
        "w13_fallback_9b_16k_lr_optb_summary.json",
        "w13_fallback_9b_16k_lr_optb_records.jsonl",
        "w13_fallback_9b_16k_osc_optb_summary.json",
        "w13_fallback_9b_16k_osc_optb_records.jsonl",
        "w14_optb_live_16k_unique_osc.py",
        "w14_optb_live_16k_unique_9b_l4_summary.json",
        "w14_optb_live_16k_unique_9b_l4_records.jsonl",
    ]
    head_blobs = ginfo["head_blob_sha256s"]
    files = []
    missing = []
    for name in artifact_globs:
        p = PILOT_DIR / name
        if p.exists():
            files.append(_file_entry(p, tracked_set, head_blobs))
        else:
            missing.append(name)

    extra_repo_paths = [
        ROOT_DIR / ".gitignore",
        REFINE_DIR / "STAGE_C_W13_EXPERIMENT_REGISTRY.md",
        REFINE_DIR / "STATEEDIT_J_PROPOSAL.md",
        REFINE_DIR / "STATEEDIT_J_EXPERIMENT_PLAN.md",
        REFINE_DIR / "NOVELTY_DOSSIER.md",
        REFINE_DIR / "STATEEDIT_J_SWARM_PROGRESS.md",
    ]
    for p in extra_repo_paths:
        if p.exists():
            files.append(_file_entry(p, tracked_set, head_blobs))
        else:
            missing.append(p.relative_to(ROOT_DIR).as_posix())

    # Exact prefix provenance check for W8 n=50 vs W12-5 n=200
    prefix_checks = {}
    for ctx in ("16k", "32k"):
        p50 = PILOT_DIR / f"w8_2b_{ctx}_n50_records.jsonl"
        p200 = PILOT_DIR / f"w12_2b_{ctx}_n200_records.jsonl"
        b50 = p50.read_bytes()
        b200 = p200.read_bytes()
        summ = _load_json_opt(PILOT_DIR / f"w12_2b_{ctx}_n200_summary.json") or {}
        prefix_checks[f"2b_{ctx}"] = {
            "n50_bytes": len(b50),
            "n200_bytes": len(b200),
            "n50_is_exact_byte_prefix_of_n200": b200.startswith(b50),
            "n200_run_metadata": summ.get("run_metadata"),
        }

    # Exact serving key reuse check (W13 2B 2K vs 16K)
    r2k = [json.loads(l) for l in (PILOT_DIR / "w13_fallback_2b_osc_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    r2k_serv = [r for r in r2k if r["suite"] == "w12_2_serving"]
    r16k_serv = [json.loads(l) for l in (PILOT_DIR / "w13_fallback_2b_16k_osc_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    k2k = [(r["regime"], r["record_key"].split(":", 1)[1]) for r in r2k_serv]
    k16k = [(r["regime"], r["record_key"].split(":", 1)[1]) for r in r16k_serv]

    fla_rep = _load_json_opt(PILOT_DIR / "w7_e9_27b_fla_env_report.json")
    s16_p = _load_json_opt(PILOT_DIR / "w8_2b_16k_n50_summary.json") or {}
    s_jamba = _load_json_opt(PILOT_DIR / "w12_c_jamba3b_gate1_summary.json") or {}

    tracked_entries = [f for f in files if f.get("git_tracked") and f.get("git_head_blob_sha256")]
    mismatched_head = [f["path"] for f in tracked_entries if not f.get("git_head_matches_lf_or_raw")]

    jamba_optb_summ = _load_json_opt(PILOT_DIR / "w14_optb_jamba3b_summary.json") or {}
    jamba_optb_eq = jamba_optb_summ.get("stateful_chunk_equivalence") or {}
    jamba_optb_fp32 = jamba_optb_summ.get("real_weights_fp32_equiv_check") or {}
    jamba_optb_og = jamba_optb_summ.get("oracle_gate") or {}
    jamba_optb_desc = (
        f"ai21labs/AI21-Jamba-Reasoning-3B (Stage-C raw 1-sample gate INCONCLUSIVE; Option-B stateful probe "
        f"bf16 pat_one KL={jamba_optb_eq.get('pat_one_vs_unpatched_mean_kl')}, "
        f"oracle_gate={jamba_optb_og.get('hits')}/{jamba_optb_og.get('n')}, "
        f"pat_chk_vs_unpatched @ {jamba_optb_summ.get('max_chunk')} mean_kl={jamba_optb_eq.get('pat_chk_vs_unpatched_mean_kl')} "
        f"strict={jamba_optb_eq.get('strict_kl_le_1e4')}/{jamba_optb_eq.get('n')} => gate_passed={jamba_optb_summ.get('gate_passed')}; "
        f"real_weights_fp32_equiv_check pat_chk mean_kl={jamba_optb_fp32.get('pat_chk_vs_unpatched_mean_kl')} "
        f"strict={jamba_optb_fp32.get('strict_kl_le_1e4')}/{jamba_optb_fp32.get('n')}; num_experts=1 dense)"
    )

    def _extract_runtime_meta(fname: str) -> Dict[str, Any]:
        d = _load_json_opt(PILOT_DIR / fname) or {}
        return {
            "device_name": d.get("device_name") or d.get("device"),
            "os_name": d.get("os_name"),
            "torch_version": d.get("torch_version"),
            "transformers_version": d.get("transformers_version"),
        }

    phase3_runtime_meta = {
        fname: _extract_runtime_meta(fname)
        for fname in (
            "w14_optb_e2e_routing_2b_4060_summary.json",
            "w14_optb_e2e_routing_9b_l4_summary.json",
            "w14_optb_symmetric_baselines_local_4060_summary.json",
            "w14_optb_symmetric_baselines_small3_a100_summary.json",
            "w14_optb_symmetric_baselines_9b_l4_summary.json",
            "w14_optb_symmetric_baselines_9b_a100_summary.json",
            "w14_optb_coupled_nvme_2b_l4_summary.json",
            "w14_optb_coupled_nvme_9b_l4_summary.json",
            "w14_optb_jamba3b_summary.json",
            "w14_optb_jamba_28l_fp32_control_summary.json",
            "w14_optb_live_16k_unique_9b_l4_summary.json",
        )
    }

    reconstructed_cli_commands = {
        "W12-2_2B_OSC_2K_OOD": "python pilot/w13_fallback_runner.py --model Qwen/Qwen3.5-2B --detector osc --tag 2b_osc --device cuda:0 --serving-ctx 2048 --suites both",
        "W12-2_2B_OSC_16K_SERVING": "python pilot/w13_fallback_runner.py --model Qwen/Qwen3.5-2B --detector osc --tag 2b_16k_osc --device cuda:0 --serving-ctx 16384 --suites serving",
        "W12-2_9B_LR_2K_OOD": "python pilot/w13_fallback_runner.py --model Qwen/Qwen3.5-9B --detector lr --tag 9b_lr --device cuda:0 --serving-ctx 2048 --suites both",
        "W12-4_2B_NVME_INT8": "python pilot/w12_c_nvme_int8_stress.py --model Qwen/Qwen3.5-2B --tag 2b --device cuda:0 --n-g1b 30 --n-col 30 --g1b-ctx 4096",
        "W12-4_9B_NVME_INT8": "python pilot/w12_c_nvme_int8_stress.py --model Qwen/Qwen3.5-9B --tag 9b --device cuda:0 --n-g1b 20 --n-col 20 --g1b-ctx 4096",
        "W12-5_2B_16K_N200": "python pilot/w5_9b_e5_runner.py --model Qwen/Qwen3.5-2B --mode o1 --block-horizon deep --max-chunk 2048 --n-samples 200 --context-lengths 16384 --out-jsonl pilot/w12_2b_16k_n200_records.jsonl --out-summary pilot/w12_2b_16k_n200_summary.json --resume",
        "W12-5_2B_32K_N200": "python pilot/w5_9b_e5_runner.py --model Qwen/Qwen3.5-2B --mode o1 --block-horizon deep --max-chunk 2048 --n-samples 200 --context-lengths 32768 --out-jsonl pilot/w12_2b_32k_n200_records.jsonl --out-summary pilot/w12_2b_32k_n200_summary.json --resume",
        "W12-5_2B_SEED_NOTE": "Out JSONLs seeded verbatim with 50-row prefix from pilot/w8_2b_16k_n50_records.jsonl and pilot/w8_2b_32k_n50_records.jsonl prior to --resume.",
        "W12-5_OFFLINE_PAIRING": "D:\\environment\\python\\pyenvs\\minimind\\python.exe pilot/w12_c_offline_pairing_n200.py",
        "W11-2_JAMBA3B_GATE": "python pilot/w12_c_third_family.py --model-key jamba3b --one-sample-gate --close-think",
        "OPTB_EXP1_2B_4060": "D:\\environment\\python\\pyenvs\\stateedit\\python.exe pilot/w14_optb_e2e_routing_bench.py --model Qwen/Qwen3.5-2B --tag 2b_4060 --detector osc --e8-suffixes 64,128,192,224,256,288,320,384,512,768,1024,1536,2048,3072,4096 --e2e-suffixes 128,256,512,1024,2048,4096,8192 --n-repeats-e8 21 --n-repeats-e2e 15 --n-warmup-e2e 5 --n-live-trials 40 --local-files-only",
        "OPTB_EXP1_9B_L4": "python3 pilot/w14_optb_e2e_routing_bench.py --model Qwen/Qwen3.5-9B --tag 9b_l4 --detector osc --e8-suffixes 64,128,192,224,256,288,320,384,512,768,1024,1536,2048,3072,4096 --e2e-suffixes 128,256,512,1024,2048,4096 --n-repeats-e8 15 --n-repeats-e2e 15 --n-warmup-e2e 5 --n-live-trials 40",
        "OPTB_EXP2_LOCAL_4060": "$env:PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128'; D:\\environment\\python\\pyenvs\\stateedit\\python.exe pilot/w14_optb_symmetric_baselines.py --models falcon-h1,zamba2,qwen2b --tag local_4060 --t-ctxs 512,1024,2048,4096,8192 --single-edit-steps 10 --local-files-only",
        "OPTB_EXP2_SMALL3_A100": "python3 pilot/w14_optb_symmetric_baselines.py --models falcon-h1,zamba2,qwen2b --tag small3_a100 --t-ctxs 512,1024,2048,4096,8192 --single-edit-steps 10",
        "OPTB_EXP2_9B_L4": "python3 pilot/w14_optb_symmetric_baselines.py --models qwen9b --tag 9b_l4 --t-ctxs 512,1024,2048,4096,8192 --single-edit-steps 10",
        "OPTB_EXP2_9B_A100": "python3 pilot/w14_optb_symmetric_baselines.py --models qwen9b --tag 9b_a100 --t-ctxs 512,1024,2048,4096,8192 --single-edit-steps 10",
        "OPTB_EXP3_COUPLED_2B_L4": "python3 pilot/w14_optb_coupled_nvme_eval.py --model Qwen/Qwen3.5-2B --tag 2b_l4 --ctx-len 8192 --n-samples 18 --out-dir /content/out --measure-prefix-kv-disk-cold --clean-out",
        "OPTB_EXP3_COUPLED_9B_L4": "python3 pilot/w14_optb_coupled_nvme_eval.py --model Qwen/Qwen3.5-9B --tag 9b_l4 --ctx-len 8192 --n-samples 12 --out-dir /content/out --measure-prefix-kv-disk-cold --clean-out",
        "OPTB_EXP4_JAMBA3B": "python3 pilot/w14_optb_jamba_stateful_probe.py --n-gate 5 --n-e2 30 --max-chunk 256 --clean-out --real-fp32-equiv-check",
        "OPTB_EXP4_28L_FP32_CONTROL": "D:\\environment\\python\\pyenvs\\stateedit\\python.exe pilot/w14_optb_jamba_stateful_probe.py --synthetic-28l-fp32-control --device cuda:0",
        "OPTB_EXP5_9B_OSC_2K": "python3 pilot/w13_fallback_runner.py --model Qwen/Qwen3.5-9B --detector osc --tag 9b_osc_optb",
        "OPTB_EXP5_9B_LR_16K": "python3 pilot/w13_fallback_runner.py --model Qwen/Qwen3.5-9B --detector lr --tag 9b_16k_lr_optb --suites serving --serving-ctx 16384",
        "OPTB_EXP5_9B_OSC_16K": "python3 pilot/w13_fallback_runner.py --model Qwen/Qwen3.5-9B --detector osc --tag 9b_16k_osc_optb --suites serving --serving-ctx 16384",
        "OPTB_EXP5_9B_OSC_16K_NOTE": "Legacy w13_fallback_runner.py serving traffic n=100 (n=93 deduplicated in posthoc); independent 100%-unique-entity diverse-context 16K run is OPTB_EXP5_UNIQUE_16K_9B_L4.",
        "OPTB_EXP5_UNIQUE_16K_9B_L4": "python3 pilot/w14_optb_live_16k_unique_osc.py --model Qwen/Qwen3.5-9B --tag 9b_l4 --seed 20261128 --ctx-len 16384 --n-clean-fact 40 --n-varlen-json 40 --n-coupled 20 --tau-osc 0.40 --clean-out --out-dir /content/out",
        "STAGE_C_POSTHOC": "D:\\environment\\python\\pyenvs\\minimind\\python.exe pilot/w12_c_posthoc_stats.py",
    }
    cli_ast_audit = _validate_cli_commands_ast(reconstructed_cli_commands)

    cf_sub_path = PILOT_DIR / "w11_cf_subset_for_colab.json"
    cf_provenance: Dict[str, Any] = {
        "pinned_subset_path": "pilot/w11_cf_subset_for_colab.json",
        "pinned_subset_exists": cf_sub_path.is_file(),
        "pinned_subset_sha256": _sha256_file(cf_sub_path) if cf_sub_path.is_file() else None,
        "pinned_subset_drive_id": "1f1JHRKs3BEyJoFOhlUF6zyMvwq29_GU0",
        "pinned_subset_rows": len(json.loads(cf_sub_path.read_text(encoding="utf-8"))) if cf_sub_path.is_file() else 0,
        "upstream_counterfact_full_rows": 21919,
        "upstream_counterfact_full_sha256": "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f",
        "gdn_lora_split_equivalence_note": "Verified identical cf_tr_0..23 and cf_te_25..48 splits between pilot/w11_cf_subset_for_colab.json (first 600 verbatim CounterFact rows) and full counterfact.json; Exp-2 GDN LoRA resolves pilot/w11_cf_subset_for_colab.json first.",
    }

    manifest = {
        "schema_version": "1.4",
        "purpose": "Hash-pinned Stage-C, Option-B (Phase-4) & E5/E9 evidence bundle for reproducible raw JSONL & dataset provenance.",
        "git_context": {
            "pinned_content_commit": ginfo["head_at_generation"],
            "git_tracked_blob_match_summary": {
                "total_tracked_in_manifest": len(tracked_entries),
                "matched_head_blob_raw_or_lf": sum(1 for f in tracked_entries if f.get("git_head_matches_lf_or_raw")),
                "mismatched_paths": mismatched_head,
            },
            "two_commit_pinning_protocol": (
                "C1 commits all code/JSON/doc updates; this manifest is generated right on C1 "
                "(recording pinned_content_commit=C1) and committed alone as C2 (`git diff C1 C2` modifies only w12_c_evidence_manifest.json)."
            ),
            "gitignore_rule_affecting_raw_logs": "pilot/*.jsonl and pilot/**/*.jsonl excluded from Git tree (pinned here via SHA-256 and Google Drive archive IDs).",
        },
        "external_raw_jsonl_archive": {
            "provider": "Google Drive (Antigravity_Projects/focused-shannon/pilot)",
            "folder_id": RAW_JSONL_DRIVE_ARCHIVE["drive_folder_id"],
            "folder_url": RAW_JSONL_DRIVE_ARCHIVE["drive_folder_url"],
            "archived_jsonl_count": len(RAW_JSONL_DRIVE_ARCHIVE["files"]),
        },
        "models_and_runtime_revisions": {
            "models": [
                "Qwen/Qwen3.5-2B",
                "Qwen/Qwen3.5-9B",
                "Qwen/Qwen3.5-27B (4-bit bnb NF4)",
                "tiiuae/Falcon-H1-1.5B-Instruct",
                "Zyphra/Zamba2-1.2B-Instruct-v2",
                jamba_optb_desc,
            ],
            "local_manifest_python_env_minimind": {
                "env_name": "minimind",
                "python": sys.version.split()[0],
                "torch": _pkg_version("torch"),
                "transformers": _pkg_version("transformers"),
                "flash_linear_attention": _pkg_version("flash-linear-attention"),
                "triton_windows": _pkg_version("triton-windows"),
                "triton": _pkg_version("triton"),
                "bitsandbytes": _pkg_version("bitsandbytes"),
                "numpy": _pkg_version("numpy"),
                "scipy": _pkg_version("scipy"),
            },
            "local_gpu_python_env_stateedit": _probe_stateedit_env(),
            "recorded_colab_a100_report": fla_rep,
            "recorded_colab_l4_2b_meta": s16_p.get("meta") or s16_p.get("config") or "see w8_2b_16k_n50_summary.json",
            "jamba_1sample_gate_meta": {
                "repo_id": s_jamba.get("repo_id"),
                "template_mode": s_jamba.get("template_mode"),
                "verdict": s_jamba.get("gate_verdict", {}).get("verdict"),
                "historical_colab_backend_pin_status": "not_recorded_in_artifact (only local stateedit transformers 5.22.2 modeling_jamba.py verified on disk)",
            },
            "option_b_jamba_stateful_meta": {
                "repo_id": "ai21labs/AI21-Jamba-Reasoning-3B",
                "runner": "pilot/w14_optb_jamba_stateful_probe.py",
                "summary": "pilot/w14_optb_jamba3b_summary.json",
            },
            "phase3_option_b_runtime_metadata": phase3_runtime_meta,
            "runtime_metadata_os_name_note": "Earlier Phase-1/2/3 summaries did not persist os_name (left null without fabrication); Phase-4 regenerated summaries record platform.platform() verbatim.",
            "counterfact_input_provenance": cf_provenance,
        },
        "cli_commands_ast_validation": cli_ast_audit,
        "reconstructed_cli_commands": reconstructed_cli_commands,
        "resume_and_reuse_provenance": {
            "w12_5_prefix_checks": prefix_checks,
            "w13_serving_2k_vs_16k_case_key_reuse": {
                "n_keys": len(k2k),
                "exact_tag_stripped_key_match": k2k == k16k,
            },
        },
        "missing_optional_files": missing,
        "pinned_artifacts": files,
    }

    out_path = PILOT_DIR / "w12_c_evidence_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Wrote {out_path} ({len(files)} artifacts pinned, missing={len(missing)})", flush=True)


if __name__ == "__main__":
    main()
