#!/usr/bin/env python3
"""Single entry point that runs the pipeline stages in order (Person 5 integration helper).

    python Scripts/10_run_pipeline.py --list                 # show the stages
    python Scripts/10_run_pipeline.py                        # default: evaluation only (fast, no docking)
    python Scripts/10_run_pipeline.py --stages 7 9           # pool builder + evaluation
    python Scripts/10_run_pipeline.py --stages all --dry-run # print the commands without running them

Only stage 9 (evaluation) is verified by Person 5.  Stages 1-8 belong to Persons 1-4; this runner just calls
them the way their scripts document, and it stops at the first failing stage.  Stages 1-6 need network access
and structural-biology packages; stage 8 needs the AutoDock Vina binary (set VINA_BIN) and is slow.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "Scripts"
CONFIG_DIR = ROOT / "Configuration and documentation"
WORK = ROOT / "pipeline_run"  # scratch dir for stages 1-6 (they read ./config.yaml and write ./outputs)
PY = sys.executable

# stage id -> (description, owner)
STAGES = {
    1: ("Fetch PDB structures", "P1"), 2: ("Prepare proteins", "P1"), 3: ("Define pockets", "P1"),
    4: ("Fetch ligands (ChEMBL/BindingDB)", "P1"), 5: ("Curate ligands + train/test split", "P1"),
    6: ("Ligand PDBQT preparation", "P1"), 7: ("Build combined EGFR+HER2 candidate pool", "P3"),
    8: ("Docking, reward, GA/RL (long)", "P4"), 9: ("Evaluation, benchmarking, figures", "P5"),
}
DEFAULT = [9]


def command(stage: int, vina_args: list[str]) -> tuple[list[str], Path, dict]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    if stage in (1, 2, 3, 4, 5, 6):
        script = next(SCRIPTS.glob(f"0{stage}_*.py"))
        return [PY, str(script)], WORK, env  # utils.load_config() reads ./config.yaml
    if stage == 7:
        return [PY, str(SCRIPTS / "07_build_dual_target_candidates.py")], ROOT, env
    if stage == 8:
        env["DRL_ROOT"] = str(ROOT)
        env["DRL_DATA"] = str(ROOT / "Outputs-git")
        script = SCRIPTS / "8_Docking,_Reward_Function_RL.py"
        return [PY, str(script), "all", *vina_args], ROOT, env
    return [PY, str(SCRIPTS / "09_evaluate_benchmark.py"), "--root", str(ROOT)], ROOT, env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=[str(s) for s in DEFAULT], help="stage numbers or 'all'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exhaustiveness", type=int, default=None, help="Vina exhaustiveness for stage 8 (recommend 8-32)")
    a = ap.parse_args()

    if a.list:
        for k, (d, o) in STAGES.items():
            print(f"  {k}  [{o}] {d}{'   <- default' if k in DEFAULT else ''}")
        return 0
    chosen = sorted(STAGES) if a.stages == ["all"] else sorted({int(s) for s in a.stages})
    bad = [s for s in chosen if s not in STAGES]
    if bad:
        sys.exit(f"unknown stage(s): {bad}")

    vina_args = ["--exhaustiveness", str(a.exhaustiveness)] if a.exhaustiveness else []
    if any(s <= 6 for s in chosen) and not a.dry_run:
        WORK.mkdir(exist_ok=True)
        shutil.copy(CONFIG_DIR / "config.yaml", WORK / "config.yaml")
    for s in chosen:
        cmd, cwd, env = command(s, vina_args)
        print(f"\n=== stage {s}: {STAGES[s][0]} [{STAGES[s][1]}]\n$ (cd {cwd.relative_to(ROOT) if cwd != ROOT else '.'}) {' '.join(cmd)}")
        if a.dry_run:
            continue
        r = subprocess.run(cmd, cwd=cwd, env=env)
        if r.returncode != 0:
            print(f"stage {s} failed (exit {r.returncode}); stopping.")
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
