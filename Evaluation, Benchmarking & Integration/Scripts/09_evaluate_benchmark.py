#!/usr/bin/env python3
"""Person 5 - evaluation & benchmarking for the EGFR/HER2 dual-target prototype.

Reads ONLY files already committed by Persons 1-4 and writes everything to
Outputs-git/evaluation/ (tables, figures, shortlist, QA checks).  No docking is run here,
so it needs just RDKit, pandas, numpy, scipy and matplotlib and finishes in about a minute.

    python Scripts/09_evaluate_benchmark.py            # from the repository root
    python Scripts/09_evaluate_benchmark.py --root .   # same, explicit

What is compared (see RESULTS_SUMMARY.md for the interpretation):
    EGFR-gen / HER2-gen / Pool   single-target generator output (38 molecules, 20 + 18) -> the
                                 "single-target generation, then filter" baseline
    GA-optimised                 genetic algorithm on the dual-docking reward (Person 4)
    RL-optimised                 REINVENT-style RL on the same reward (Person 4)
    Known actives (reference)    ChEMBL EGFR/HER2 actives (property reference only, not docked)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, Descriptors, Draw, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy import stats

RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------- criteria
# All thresholds live here so they can be changed in one place and are echoed in the outputs.
CRITERIA = {
    "dock_hit": -8.0,        # kcal/mol, per target (more negative = better). Vina scores are crude.
    "dock_hit_strict": -9.0,
    "balance_max": 1.0,      # |score_EGFR - score_HER2| <= this counts as "balanced"
    "qed_min": 0.5,
    "sa_max": 4.5,           # 1 = easy, 10 = hard to synthesise
    "novel_sim_max": 0.4,    # "novel_pct" metric: max Tanimoto (Morgan r2) to any known active < this
    "novelty_min": 0.3,      # lead / shortlist rule: novelty (1 - max Tanimoto) > this; same as Person 4's select_final
    "known_sim": 0.999,      # max Tanimoto >= this => the molecule IS a known active (rediscovered, not generated)
}
SEED = 42

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_LARGEST = rdMolStandardize.LargestFragmentChooser()


def _sascorer():
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return sascorer
    except Exception:
        from rdkit.Chem import RDConfig
        sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
        import sascorer
        return sascorer


sascorer = _sascorer()

# colour-blind-safe (Okabe-Ito) so the groups stay distinguishable
COL = {
    "EGFR-gen": "#0072B2", "HER2-gen": "#D55E00", "Pool": "#7F7F7F",
    "GA-optimised": "#009E73", "RL-optimised": "#CC79A7", "Known actives": "#000000",
}


# --------------------------------------------------------------------------- helpers
def canon(smiles) -> str | None:
    """Canonical SMILES of the largest fragment (drops salts); None if RDKit cannot parse it."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        m = _LARGEST.choose(m)
    except Exception:
        pass
    return Chem.MolToSmiles(m)


def fp_of(smi: str):
    return _MORGAN.GetFingerprint(Chem.MolFromSmiles(smi))


def max_sim(fp, ref_fps) -> float:
    return float(max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps))) if ref_fps else np.nan


def internal_diversity(smis: list[str]) -> float:
    """1 - mean pairwise Tanimoto (higher = more diverse)."""
    fps = [fp_of(s) for s in smis]
    if len(fps) < 2:
        return np.nan
    sims = [s for i in range(1, len(fps)) for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])]
    return float(1.0 - np.mean(sims))


def props(smi: str) -> dict:
    m = Chem.MolFromSmiles(smi)
    return {"qed": QED.qed(m), "sa_score": sascorer.calculateScore(m),
            "heavy_atoms": m.GetNumHeavyAtoms(), "mw": Descriptors.MolWt(m)}


# --------------------------------------------------------------------------- loading
def load_reference(out: Path):
    """Novelty reference = EGFR/HER2 train+test files (identical to Person 4's reward function)."""
    rows = []
    for t in ("EGFR", "HER2"):
        for k in ("train", "test"):
            p = out / f"{t}_{k}.csv"
            if p.exists():
                d = pd.read_csv(p)
                d["target"] = t
                rows.append(d)
    if not rows:
        sys.exit(f"No reference CSVs found in {out}")
    ref = pd.concat(rows, ignore_index=True)
    ref["canon"] = ref["canonical_smiles"].map(canon)
    ref = ref.dropna(subset=["canon"]).drop_duplicates("canon").reset_index(drop=True)

    # known dual actives = same ChEMBL molecule listed as an active for both targets
    sets = {t: set(ref.loc[ref.target == t, "molecule_chembl_id"]) for t in ("EGFR", "HER2")}
    full = pd.concat([pd.read_csv(p) for p in out.glob("*_curated_actives.csv")], ignore_index=True)
    full["canon"] = full["canonical_smiles"].map(canon)
    a = set(pd.read_csv(out / "EGFR_curated_actives.csv")["molecule_chembl_id"])
    b = set(pd.read_csv(out / "HER2_curated_actives.csv")["molecule_chembl_id"])
    dual_ids = a & b
    dual = full[full.molecule_chembl_id.isin(dual_ids)].dropna(subset=["canon"]).drop_duplicates("canon")
    ref_fps = [fp_of(s) for s in ref.canon]
    dual_fps = [fp_of(s) for s in dual.canon]
    return ref, ref_fps, dual, dual_fps


def _std_cols(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    d = pd.DataFrame({"canonical": df[source_col].map(canon)})
    for c in ("gate_reason", "dock_EGFR", "dock_HER2", "reward", "novelty"):
        d[c if c != "novelty" else "novelty_p4"] = df[c] if c in df else np.nan
    return d


def load_groups(root: Path, out: Path) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Return {group: per-molecule frame} and {group: parse-validity fraction (or nan)}."""
    groups: dict[str, pd.DataFrame] = {}
    validity: dict[str, float] = {}

    scored = pd.read_csv(out / "scored_candidates.csv")
    manifest = pd.read_csv(out / "dual_target_candidate_manifest.csv")
    scored = scored.merge(manifest[["candidate_id", "generation_sources"]], on="candidate_id", how="left")
    base = _std_cols(scored, "smiles")
    base["candidate_id"] = scored["candidate_id"].values
    base["source"] = scored["generation_sources"].values
    groups["EGFR-gen"] = base[base.source == "EGFR"].copy()
    groups["HER2-gen"] = base[base.source == "HER2"].copy()
    groups["Pool"] = base.copy()
    rep = pd.read_csv(root / "baseline_validation_report.csv")
    validity["EGFR-gen"] = rep.loc[rep.target == "EGFR", "rdkit_valid"].mean()
    validity["HER2-gen"] = rep.loc[rep.target == "HER2", "rdkit_valid"].mean()
    validity["Pool"] = rep["rdkit_valid"].mean()

    ga = _std_cols(pd.read_csv(out / "ga_all_scored.csv"), "smiles")
    ga["source"] = "GA"
    groups["GA-optimised"] = ga
    validity["GA-optimised"] = np.nan  # only valid molecules are logged; no sampling-validity record

    rl = _std_cols(pd.read_csv(out / "rl_all_scored.csv"), "smiles")
    rl["source"] = "RL"
    groups["RL-optimised"] = rl
    curve = pd.read_csv(out / "rl_training_curve.csv")
    validity["RL-optimised"] = float(curve["valid_frac"].mean())  # RDKit-parse rate of sampled SMILES
    return groups, validity


# --------------------------------------------------------------------------- per-molecule table
def build_molecule_table(groups, ref, ref_fps, dual_fps) -> pd.DataFrame:
    ref_by_fp = ref.reset_index(drop=True)
    rows = []
    for g, df in groups.items():
        if g == "Pool":  # the pool is EGFR-gen + HER2-gen; scoring it twice would be redundant
            continue
        rows.append(df.assign(group=g))
    # the Pool is evaluated at group level (metrics) but reuses the same per-molecule rows
    mol = pd.concat(rows, ignore_index=True)
    mol["parse_ok"] = mol["canonical"].notna()
    mol = mol[mol.parse_ok].copy()

    cache = {}
    for smi in mol.canonical.unique():
        fp = fp_of(smi)
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        j = int(np.argmax(sims))
        cache[smi] = {**props(smi), "max_sim_known": float(sims[j]),
                      "nearest_known_chembl": ref_by_fp.loc[j, "molecule_chembl_id"],
                      "max_sim_dual_known": max_sim(fp, dual_fps)}
    extra = pd.DataFrame([cache[s] for s in mol.canonical], index=mol.index)
    mol = pd.concat([mol, extra.drop(columns=[c for c in extra if c in mol.columns])], axis=1)
    mol["novelty"] = 1.0 - mol["max_sim_known"]
    mol["known_active"] = mol["max_sim_known"] >= CRITERIA["known_sim"]
    mol["novel_ok"] = mol["novelty"] > CRITERIA["novelty_min"]
    mol["gate_pass"] = mol["gate_reason"].isna()
    mol["docked"] = mol[["dock_EGFR", "dock_HER2"]].notna().all(axis=1)
    mol["dock_mean"] = mol[["dock_EGFR", "dock_HER2"]].mean(axis=1)
    mol["dock_diff"] = (mol.dock_EGFR - mol.dock_HER2).abs()
    mol["lig_eff"] = -mol.dock_mean / mol.heavy_atoms          # kcal/mol per heavy atom
    c = CRITERIA
    mol["dual_hit"] = mol.gate_pass & mol.docked & (mol.dock_EGFR <= c["dock_hit"]) & (mol.dock_HER2 <= c["dock_hit"])
    mol["dual_hit_strict"] = mol.gate_pass & mol.docked & (mol.dock_EGFR <= c["dock_hit_strict"]) & (mol.dock_HER2 <= c["dock_hit_strict"])
    mol["dual_hit_novel"] = mol.dual_hit & mol.novel_ok
    mol["dual_lead"] = (mol.dual_hit_novel & (mol.dock_diff <= c["balance_max"])
                        & (mol.qed >= c["qed_min"]) & (mol.sa_score <= c["sa_max"]))
    return mol.reset_index(drop=True)


# --------------------------------------------------------------------------- group metrics
def group_metrics(name, df, validity) -> dict:
    n = len(df)
    d = df[df.docked]
    r = {
        "group": name,
        "n_logged": n,
        "rdkit_valid_pct": 100 * validity.get(name, np.nan),
        "unique_pct": 100 * df.canonical.nunique() / n,
        "gate_pass_pct": 100 * df.gate_pass.mean(),
        "docked_pct": 100 * df.docked.mean(),
        "novel_pct": 100 * (df.max_sim_known < CRITERIA["novel_sim_max"]).mean(),
        "novelty_mean": df.novelty.mean(),
        "diversity": internal_diversity(df.canonical.tolist()),
        "qed_median": df.qed.median(),
        "sa_median": df.sa_score.median(),
        "heavy_atoms_median": df.heavy_atoms.median(),
        "sim_dual_known_median": df.max_sim_dual_known.median(),
        "n_docked": len(d),
        "dock_EGFR_median": d.dock_EGFR.median(),
        "dock_HER2_median": d.dock_HER2.median(),
        "dock_EGFR_best": d.dock_EGFR.min() if len(d) else np.nan,
        "dock_HER2_best": d.dock_HER2.min() if len(d) else np.nan,
        "dock_diff_median": d.dock_diff.median(),
        "lig_eff_median": d.lig_eff.median(),
        "n_known_rediscovered": int(df.known_active.sum()),
        "n_dual_hit": int(df.dual_hit.sum()),
        "dual_hit_pct_of_logged": 100 * df.dual_hit.mean(),
        "n_dual_hit_novel": int(df.dual_hit_novel.sum()),
        "n_dual_hit_strict": int(df.dual_hit_strict.sum()),
        "n_dual_lead": int(df.dual_lead.sum()),
        "dual_lead_pct_of_logged": 100 * df.dual_lead.mean(),
        "reward_median_docked": d.reward.median(),
        "reward_max": df.reward.max(),
    }
    return r


def reference_metrics(name, smis, dual_fps, ref_fps) -> dict:
    p = pd.DataFrame([props(s) for s in smis])
    return {"group": name, "n_logged": len(smis), "rdkit_valid_pct": 100.0,
            "unique_pct": 100 * len(set(smis)) / len(smis),
            "diversity": internal_diversity(smis),
            "qed_median": p.qed.median(), "sa_median": p.sa_score.median(),
            "heavy_atoms_median": p.heavy_atoms.median()}


# --------------------------------------------------------------------------- statistics
def stat_tests(mol: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mann-Whitney U vs. the filtered single-target baseline + size-bias correlations.

    Run twice for the optimisers: on everything they logged, and on novel molecules only
    (known actives rediscovered by the GA/RL are excluded in the second run)."""
    base = mol[(mol.group.isin(["EGFR-gen", "HER2-gen"])) & mol.docked & mol.gate_pass]
    rows = []
    for g in ("GA-optimised", "RL-optimised"):
        for subset in ("all logged", "novel only"):
            o = mol[(mol.group == g) & mol.docked]
            if subset == "novel only":
                o = o[o.novel_ok]
            for metric in ("dock_mean", "lig_eff", "qed", "sa_score"):
                a, b = o[metric].dropna(), base[metric].dropna()
                if len(a) < 3 or len(b) < 3:
                    continue
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                rows.append({"comparison": f"{g} vs filtered single-target baseline", "subset": subset, "metric": metric,
                             "n_a": len(a), "n_b": len(b), "median_a": a.median(), "median_b": b.median(),
                             "effect_a_gt_b": 2 * u / (len(a) * len(b)) - 1,  # +1: a always larger, -1: a always smaller
                             "p_value": p})
    tests = pd.DataFrame(rows)

    size = []
    for g, d in mol[mol.docked].groupby("group"):
        if len(d) >= 8:
            rho, p = stats.spearmanr(d.heavy_atoms, d.dock_mean)
            size.append({"group": g, "n": len(d), "spearman_heavy_atoms_vs_dock": rho, "p_value": p})
    return tests, pd.DataFrame(size)


# --------------------------------------------------------------------------- QA checks
def qa_checks(root: Path, out: Path, mol: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check, status, detail):
        rows.append({"check": check, "status": status, "detail": detail})

    # 1. co-crystal redocking
    p = out / "redocking_validation.csv"
    if p.exists():
        for _, r in pd.read_csv(p).iterrows():
            ok = bool(r.workflow_ok)
            add(f"Redocking {r.target}", "OK" if ok else "FAIL",
                f"top-pose RMSD {r.top_pose_rmsd:.2f} A (pass <= 2.0); best of {int(r.n_modes)} modes "
                f"{r.min_rmsd_over_modes:.2f} A; best score {r.best_score:.2f}")
    else:
        add("Redocking", "WARN", "redocking_validation.csv not found")

    # 2. docking boxes: must sit on the co-crystal ligand (committed by Person 1's script 03) and be comparable
    bp = out / "docking_boxes.json"
    res = {"EGFR": ("1XKK", "FMM"), "HER2": ("3PP0", "03Q")}  # primary PDB / ligand residue (config.yaml)
    if bp.exists():
        boxes = json.loads(bp.read_text())
        vols = {}
        for t, b in boxes.items():
            sides = [b["size_x"], b["size_y"], b["size_z"]]
            vols[t] = float(np.prod(sides))
            centre = np.array([b["center_x"], b["center_y"], b["center_z"]])
            pdb = out / f"{t}_{res[t][0]}_clean.pdb" if t in res else None
            lig = []
            if pdb is not None and pdb.exists():
                lig = [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in pdb.read_text().splitlines()
                       if l.startswith("HETATM") and l[17:20].strip() == res[t][1]]
            if lig:
                lig = np.array(lig)
                d = float(np.linalg.norm((lig.min(0) + lig.max(0)) / 2 - centre))
                add(f"Docking box {t} centred on crystal ligand", "OK" if d < 5 else "FAIL",
                    f"box centre is {d:.1f} A from the {res[t][1]} ligand centre; box sides {sides[0]:.0f} x {sides[1]:.0f} x {sides[2]:.0f} A"
                    + ("" if d < 5 else " (Person 1's docking_boxes.json is wrong for this target; Person 4's script uses docking_boxes_fixed.json instead)"))
        if len(vols) == 2 and max(vols.values()) / min(vols.values()) > 2:
            big, small = max(vols, key=vols.get), min(vols, key=vols.get)
            add("Box volume EGFR vs HER2 (committed json)", "WARN",
                f"{big} box is {vols[big] / vols[small]:.1f}x the volume of {small}; scores are only comparable if both targets were docked with similar boxes")
    else:
        add("Docking boxes", "WARN", "docking_boxes.json not found")
    if not (out / "docking_boxes_fixed.json").exists() and not (root / "docking_boxes_fixed.json").exists():
        add("docking_boxes_fixed.json committed", "WARN",
            "Person 4's script reads/writes docking_boxes_fixed.json but it is not in the repo, so the boxes behind the committed scores cannot be verified")

    # 2b. config anchor residues vs the renumbered clean PDBs (Person 1)
    clean = out / "EGFR_1XKK_clean.pdb"
    if clean.exists():
        nums = [int(l[22:26]) for l in clean.read_text().splitlines() if l.startswith("ATOM") and l[12:16].strip() == "CA"]
        if nums and max(nums) < 793:
            add("Anchor residue numbers usable (config.yaml)", "FAIL",
                f"cleaned PDB is renumbered 1..{max(nums)} so UniProt anchors (EGFR hinge 793...) match nothing; script 03's HER2->EGFR alignment is silently skipped and no aligned file exists")

    # 2c. geometry of the 3D molecules from Person 2
    bad = 0
    for f in ("egfr_mols_20.sdf", "her2_mols_valid.sdf"):
        if (root / f).exists():
            for m in Chem.SDMolSupplier(str(root / f), removeHs=False):
                if m is None:
                    continue
                x = m.GetConformer().GetPositions()
                if any(np.linalg.norm(x[b.GetBeginAtomIdx()] - x[b.GetEndAtomIdx()]) < 1.1 for b in m.GetBonds()):
                    bad += 1
    add("3D geometry of single-target molecules", "WARN" if bad else "OK",
        f"{bad} of 38 molecules contain a bond shorter than 1.1 A (physically impossible; e.g. O-O at 0.4 A); many more have stretched bonds")

    # 3. exhaustiveness
    for cand in (root / "8_Docking,_Reward_Function_RL.py", root / "Scripts" / "8_Docking,_Reward_Function_RL.py"):
        if cand.exists():
            m = re.search(r'"exhaustiveness"\s*:\s*(\d+)', cand.read_text())
            if m:
                ex = int(m.group(1))
                add("Vina exhaustiveness (script default)", "WARN" if ex < 8 else "OK",
                    f"default is {ex}; the script itself recommends 8-32 for final numbers. Confirm which value produced the "
                    f"committed scores (scores at low exhaustiveness are noisy)")
            break

    # 4. docking failures and gate
    pool = mol[mol.group.isin(["EGFR-gen", "HER2-gen"])]
    gate_pass = pool[pool.gate_pass]
    failed = gate_pass[~gate_pass.docked]
    add("Docking failures (baseline pool)", "WARN" if len(failed) else "OK",
        f"{len(failed)} of {len(gate_pass)} gate-passing baseline molecules have no docking score")

    # 5. internal consistency of file counts
    manifest = pd.read_csv(out / "dual_target_candidate_manifest.csv")
    n_sdf = 0
    sdf_path = out / "dual_target_candidates.sdf"
    if sdf_path.exists():
        n_sdf = sdf_path.read_text().count("$$$$")
    add("Pool counts consistent", "OK" if len(manifest) == n_sdf == len(pool) else "WARN",
        f"manifest {len(manifest)}, SDF {n_sdf}, scored {len(pool)}")

    # 6. documentation ID scheme
    doc = root / "Configuration and documentation" / "DUAL_TARGET.md"
    if doc.exists() and "DT_001" in doc.read_text() and manifest.candidate_id.iloc[0].startswith("P3_"):
        add("Candidate ID scheme", "WARN",
            "DUAL_TARGET.md says IDs are DT_001..DT_038 but the manifest/SDF/scores use P3_001..P3_038; update the doc")

    # 7. no known-ligand docking reference
    has_ref = any("known" in f.name.lower() or "reference_dock" in f.name.lower() for f in out.glob("*.csv"))
    if not has_ref:
        add("Known-inhibitor docking reference", "WARN",
            "no docking scores for known EGFR/HER2 inhibitors are committed (planned for Days 3-5); without them there "
            "is no way to say whether e.g. -9 kcal/mol is 'good' for this setup")

    # 8. optimisers start from known actives
    for g in ("GA-optimised", "RL-optimised"):
        d = mol[mol.group == g]
        add(f"Closeness to known actives - {g}", "INFO",
            f"median max Tanimoto to a known EGFR/HER2 active = {d.max_sim_known.median():.2f}; "
            f"{100 * (d.max_sim_known >= 0.7).mean():.0f}% have a close analogue (>= 0.7) and {int(d.known_active.sum())} of "
            f"{len(d)} are exact known actives (rediscovered, not newly generated)")

    # 9. Person 4's novelty vs ours
    both = mol.dropna(subset=["novelty_p4"])
    if len(both):
        diff = (both.novelty - both.novelty_p4).abs()
        add("Novelty cross-check vs reward-function value", "OK" if diff.max() < 0.05 else "WARN",
            f"max |difference| {diff.max():.3f} over {len(both)} molecules (ours strips salts from the reference set)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- shortlist
def build_shortlist(mol: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    cand = mol[mol.gate_pass & mol.docked & mol.novel_ok].copy()   # known actives are excluded
    cand = cand.sort_values("reward", ascending=False).drop_duplicates("canonical")
    cols = ["canonical", "group", "candidate_id", "reward", "dock_EGFR", "dock_HER2", "dock_diff", "qed", "sa_score",
            "heavy_atoms", "novelty", "max_sim_known", "nearest_known_chembl", "dual_hit_novel", "dual_lead"]
    out = cand[cols].head(n).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out


# --------------------------------------------------------------------------- figures
def _style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


def fig_funnel(metrics: pd.DataFrame, path: Path):
    m = metrics[metrics.group.isin(["Pool", "GA-optimised", "RL-optimised"])].set_index("group")
    steps = [("gate_pass_pct", "Pass chemistry gate"), ("docked_pct", "Docked on both targets"),
             ("dual_hit_pct_of_logged", f"Dual hit (both <= {CRITERIA['dock_hit']:.0f})"),
             ("dual_lead_pct_of_logged", "Dual lead (novel, balanced,\nQED ≥ 0.5, SA ≤ 4.5)")]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    h = 0.2
    for i, (g, color) in enumerate([("Pool", COL["Pool"]), ("GA-optimised", COL["GA-optimised"]), ("RL-optimised", COL["RL-optimised"])]):
        y = np.arange(len(steps)) + (i - 1) * h
        vals = [m.loc[g, s] for s, _ in steps]
        ax.barh(y, vals, height=h * 0.9, color=color, label=f"{g} (n={int(m.loc[g, 'n_logged'])})")
        for yy, v in zip(y, vals):
            ax.text(v + 1, yy, f"{v:.0f}%", va="center", fontsize=8)
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([l for _, l in steps])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of molecules generated / logged")
    ax.set_title("Yield: from generated molecules to dual-target leads")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def fig_scatter(mol: pd.DataFrame, path: Path):
    d = mol[mol.docked & mol.gate_pass]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    lo = min(d.dock_EGFR.min(), d.dock_HER2.min()) - 0.4
    hi = max(d.dock_EGFR.max(), d.dock_HER2.max()) + 0.4
    ax.plot([lo, hi], [lo, hi], color="#bbbbbb", lw=1, zorder=0)
    ax.fill_between([lo, hi], [lo - CRITERIA["balance_max"], hi - CRITERIA["balance_max"]],
                    [lo + CRITERIA["balance_max"], hi + CRITERIA["balance_max"]], color="#eeeeee", zorder=0,
                    label=f"|Δ| ≤ {CRITERIA['balance_max']:.0f} kcal/mol (balanced)")
    for g, mk in (("RL-optimised", "o"), ("GA-optimised", "s"), ("EGFR-gen", "^"), ("HER2-gen", "v")):
        s = d[d.group == g]
        ax.scatter(s.dock_EGFR, s.dock_HER2, s=26 if g != "RL-optimised" else 18, marker=mk, color=COL[g],
                   alpha=0.85 if "gen" in g else 0.6, edgecolor="white", linewidth=0.4, label=f"{g} (n={len(s)})")
    t = CRITERIA["dock_hit"]
    ax.axvline(t, color="#555", ls="--", lw=0.8)
    ax.axhline(t, color="#555", ls="--", lw=0.8)
    ax.set_xlim(hi, lo)
    ax.set_ylim(hi, lo)  # both axes reversed: better (more negative) is up/right
    ax.set_xlabel("EGFR docking score (kcal/mol)  ← better")
    ax.set_ylabel("HER2 docking score (kcal/mol)  ← better")
    ax.set_title("Dual docking: top-right of the dashed lines = good on both")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def fig_properties(mol: pd.DataFrame, ref_props: pd.DataFrame, path: Path):
    order = ["EGFR-gen", "HER2-gen", "GA-optimised", "RL-optimised", "Known actives"]
    data = {g: mol[mol.group == g] for g in order[:-1]}
    data["Known actives"] = ref_props
    panels = [("qed", "QED (higher = more drug-like)", CRITERIA["qed_min"]),
              ("sa_score", "SA score (lower = easier to make)", CRITERIA["sa_max"]),
              ("novelty", "Novelty = 1 − max Tanimoto to known actives", None),
              ("heavy_atoms", "Heavy atoms", None)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    rng = np.random.default_rng(SEED)
    for ax, (col, title, thr) in zip(axes, panels):
        for i, g in enumerate(order):
            if col not in data[g]:
                continue
            v = data[g][col].dropna().values
            if len(v) == 0:
                continue
            ax.boxplot(v, positions=[i], widths=0.55, showfliers=False, medianprops={"color": "black"},
                       boxprops={"color": COL[g]}, whiskerprops={"color": COL[g]}, capprops={"color": COL[g]})
            sub = v if len(v) <= 80 else rng.choice(v, 80, replace=False)
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(sub)), sub, s=6, color=COL[g], alpha=0.5)
        if thr is not None:
            ax.axhline(thr, color="#555", ls="--", lw=0.8)
        ax.set_xticks(range(len(order)))
        short = {"EGFR-gen": "EGFR\ngen", "HER2-gen": "HER2\ngen", "GA-optimised": "GA", "RL-optimised": "RL", "Known actives": "Known"}
        ax.set_xticklabels([f"{short[o]}\nn={len(data[o][col].dropna()) if col in data[o] else 0}" for o in order], fontsize=8)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Molecular properties by group (all logged molecules)", y=1.03)
    fig.savefig(path)
    plt.close(fig)


def fig_curves(out: Path, path: Path):
    ga = pd.read_csv(out / "ga_generations.csv")
    rl = pd.read_csv(out / "rl_training_curve.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(ga.generation, ga.best, "o-", color=COL["GA-optimised"], label="best")
    axes[0].plot(ga.generation, ga["mean"], "o--", color=COL["GA-optimised"], alpha=0.6, label="mean of new")
    axes[0].set(title="GA: reward by generation", xlabel="generation", ylabel="reward")
    axes[0].set_xticks(ga.generation)
    axes[0].legend(frameon=False)
    k = 5
    axes[1].plot(rl.step, rl.mean_reward, color=COL["RL-optimised"], alpha=0.35, label="mean (per step)")
    axes[1].plot(rl.step, rl.mean_reward.rolling(k, min_periods=1).mean(), color=COL["RL-optimised"], label=f"mean ({k}-step avg)")
    axes[1].plot(rl.step, rl.max_reward, color="#555", lw=0.8, ls=":", label="max in batch")
    axes[1].set(title="RL: reward by step", xlabel="step", ylabel="reward")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(rl.step, rl.valid_frac, color=COL["RL-optimised"])
    axes[2].set(title="RL: fraction of sampled SMILES that parse", xlabel="step", ylim=(0, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_top_grid(short: pd.DataFrame, path: Path, n: int = 12):
    top = short.head(n)
    mols = [Chem.MolFromSmiles(s) for s in top.canonical]
    legends = [f"#{r.rank} {r.group.split('-')[0]}  R={r.reward:.2f}\nEGFR {r.dock_EGFR:.1f} | HER2 {r.dock_HER2:.1f}"
               for r in top.itertuples()]
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(330, 250), legends=legends)
    if hasattr(img, "save"):          # PIL image (plain Python)
        img.save(str(path))
    else:                             # IPython Image (inside Jupyter) carries the PNG bytes
        Path(path).write_bytes(img.data)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="repository root")
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/Outputs-git/evaluation")
    ap.add_argument("--top", type=int, default=20, help="size of the final shortlist")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    outputs = root / "Outputs-git"
    ev = (a.out or outputs / "evaluation").resolve()
    (ev / "figures").mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    _style()

    print("[1/6] loading reference actives and candidate sets")
    ref, ref_fps, dual, dual_fps = load_reference(outputs)
    groups, validity = load_groups(root, outputs)
    print(f"      reference actives: {len(ref)} unique ({len(dual)} listed as active for both targets)")

    print("[2/6] computing per-molecule properties, novelty and docking descriptors")
    mol = build_molecule_table(groups, ref, ref_fps, dual_fps)
    mol.to_csv(ev / "per_molecule_table.csv", index=False)

    print("[3/6] group-level metrics")
    rows = []
    for g in ("EGFR-gen", "HER2-gen", "Pool", "GA-optimised", "RL-optimised"):
        df = mol[mol.group == g] if g != "Pool" else mol[mol.group.isin(["EGFR-gen", "HER2-gen"])]
        rows.append(group_metrics(g, df, validity))
    ref_props = pd.DataFrame([props(s) for s in ref.canon])
    rows.append(reference_metrics("Known actives", ref.canon.tolist(), dual_fps, ref_fps))
    rows.append(reference_metrics("Known dual actives", dual.canon.tolist(), dual_fps, ref_fps))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ev / "metrics_summary.csv", index=False)
    (ev / "metrics_summary.md").write_text(
        "Thresholds: " + json.dumps(CRITERIA) + "\n\n" + metrics.round(3).to_markdown(index=False) + "\n")

    print("[4/6] statistics, QA checks, shortlist")
    tests, size = stat_tests(mol)
    tests.to_csv(ev / "statistical_tests.csv", index=False)
    size.to_csv(ev / "size_bias.csv", index=False)
    qa = qa_checks(root, outputs, mol)
    qa.to_csv(ev / "qa_checks.csv", index=False)
    short = build_shortlist(mol, a.top)
    short.to_csv(ev / "final_candidate_shortlist.csv", index=False)

    print("[5/6] figures")
    fig_funnel(metrics, ev / "figures" / "fig1_yield_funnel.png")
    fig_scatter(mol, ev / "figures" / "fig2_dual_docking_scatter.png")
    ref_props["novelty"] = np.nan
    fig_properties(mol, ref_props, ev / "figures" / "fig3_property_distributions.png")
    fig_curves(outputs, ev / "figures" / "fig4_optimisation_curves.png")
    fig_top_grid(short, ev / "figures" / "fig5_top_candidates.png")

    print("[6/6] done ->", ev)
    print("\nQA flags:")
    print(qa[qa.status.isin(["FAIL", "WARN"])].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
