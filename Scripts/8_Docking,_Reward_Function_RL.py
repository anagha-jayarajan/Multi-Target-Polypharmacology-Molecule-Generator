#!/usr/bin/env python3
"""EGFR / HER2 dual-target docking, reward function and RL - single-file version.

Generated from the docking_rl package by tools/build_script.py (edit the package, not this file).

    python run_all.py prepare                 # fixed boxes + receptor PDBQT
    python run_all.py validate                # redock co-crystal ligands (RMSD <= 2.0 A = OK)
    python run_all.py score [--input FILE]    # dock + reward for the candidates
    python run_all.py ga                      # genetic algorithm
    python run_all.py rl [--chunk 10]         # resumable REINVENT-style RL (rerun to continue)
    python run_all.py final                   # final candidate set from the GA table
    python run_all.py all                     # everything above, in order

Needs: pip install -r requirements.txt, the AutoDock Vina binary (VINA_BIN or ./vina), and the inputs in data/
(or DRL_DATA). Environment: DRL_ROOT (project root), DRL_DATA, VINA_BIN.
"""

from __future__ import annotations


# =============================================================================
# config.py
# =============================================================================
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(os.environ.get("DRL_ROOT", Path.cwd())).resolve()
OUT_DIR = PROJECT_ROOT / "outputs"

TARGETS = {
    # target -> cleaned protein PDB, co-crystal ligand residue name inside that PDB, and an optional
    # SMILES of that ligand (only used to recover bond orders if the PDB ligand lacks hydrogens)
    "EGFR": {"pdb": "EGFR_1XKK_clean.pdb", "ligand_resname": "FMM",   # 1XKK (lapatinib)
             "ligand_smiles": "CS(=O)(=O)CCNCc1ccc(o1)-c1ccc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2c1"},
    "HER2": {"pdb": "HER2_3PP0_clean.pdb", "ligand_resname": "03Q",   # 3PP0 (TAK-285)
             "ligand_smiles": None},
}

REF_DRUGS = {
    "gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "lapatinib": "CS(=O)(=O)CCNCc1ccc(o1)-c1ccc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2c1",
    "erlotinib": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "weak_fragment": "CCCC(=O)NCC1CCCC(C)C1",
    "junk_OOOOO": "OOOOO",
}


def find_data_dir() -> Path:
    """DRL_DATA if set, else the first usual place that actually contains the receptor PDBs."""
    env = os.environ.get("DRL_DATA")
    if env:
        return Path(env).resolve()
    need = [t["pdb"] for t in TARGETS.values()]
    for cand in (PROJECT_ROOT / "data", PROJECT_ROOT, Path("/content/data"), Path("/content"), Path.cwd()):
        if all((cand / f).exists() for f in need):
            return cand.resolve()
    print(f"[config] could not find {need} - put them in {PROJECT_ROOT / 'data'} or set DRL_DATA")
    return (PROJECT_ROOT / "data").resolve()


def find_vina() -> str:
    env = os.environ.get("VINA_BIN")
    if env:
        return env
    for local in (PROJECT_ROOT / "vina", Path("/content/vina"), Path.cwd() / "vina"):
        if local.exists():
            return str(local.resolve())
    return shutil.which("vina") or "vina"


C = SimpleNamespace(
    PROJECT_ROOT=PROJECT_ROOT, DATA_DIR=find_data_dir(), OUT_DIR=OUT_DIR,
    RECEPTOR_DIR=OUT_DIR / "receptors", RESULTS_DIR=OUT_DIR / "results",
    DOCK_CACHE_DIR=OUT_DIR / "dock_cache", VINA_BIN=find_vina(),
    TARGETS=TARGETS,
    # "auto": Meeko for every target, and if it fails for ANY target, Open Babel for ALL targets
    # (both receptors must be prepared the same way or their Vina scores are not comparable).
    RECEPTOR_METHOD="auto",          # "auto" | "meeko" | "obabel"
    LIGAND_CHAIN_ONLY=True,          # dock into the chain that holds the co-crystal ligand only
    BOX_PADDING=8.0, BOX_MIN_SIDE=18.0, BOX_MAX_SIDE=30.0,
    # ONE docking setting for every stage (validation excepted: it uses its own exhaustiveness)
    # so that all reported scores are comparable. Raise exhaustiveness (8-32) for final numbers.
    DOCK={"exhaustiveness": 4, "num_modes": 5, "energy_range": 3, "cpu": 0, "seed": 42, "timeout_s": 1800},
    MAX_TORSIONS=20,                 # ligands with more rotatable bonds are skipped (Vina limit is 32)
    # Known actives used for novelty + GA seeds + RL prior. Missing files are skipped with a message.
    REFERENCE_CSVS={"EGFR": ["EGFR_train.csv", "EGFR_test.csv"],
                    "HER2": ["HER2_train.csv", "HER2_test.csv"]},
    # If a reference CSV is missing / not a real CSV the run STOPS unless this is True
    # (then state in the report that novelty / seeds / RL prior used an incomplete set).
    ALLOW_INCOMPLETE_REFERENCES=False,
    # kcal/mol, more negative = better. SCORE_BAD -> reward 0, SCORE_GOOD -> reward 1
    SCORE_GOOD=-11.0, SCORE_BAD=-7.0,
)


# =============================================================================
# receptors.py
# =============================================================================
import json
import subprocess
from typing import Optional

import numpy as np


_TWO_LETTER = {"CL", "BR", "FE", "ZN", "MG", "MN", "SE"}


def _guess_element(line: str) -> str:
    el = line[76:78].strip()
    if el:
        return el.upper()
    name = line[12:16]
    if name[0].isdigit():            # e.g. "1HB "
        return "H"
    if name[0] == " ":               # one-letter element: name starts in column 14
        return name[1].upper()
    two = name[:2].strip().upper()
    if line.startswith("HETATM") and two in _TWO_LETTER:
        return two
    return name[0].upper()


def _parse_atoms(pdb_path: Path):
    """ATOM/HETATM records. Alternate locations are resolved (first altloc seen per residue is kept,
    the altloc column is blanked) - keeping A *and* B copies gives atoms with impossible valences,
    which is what broke Meeko's receptor preparation and would break bond perception of the ligand."""
    atoms, chosen = [], {}
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        line = line.ljust(80)
        alt = line[16]
        if alt != " ":
            res_id = (line[21], line[22:26], line[26])
            if chosen.setdefault(res_id, alt) != alt:
                continue
            line = line[:16] + " " + line[17:]
        atoms.append({
            "record": line[:6].strip(),
            "name": line[12:16].strip(),
            "resname": line[17:20].strip(),
            "chain": line[21],
            "resseq": line[22:26].strip(),
            "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
            "element": _guess_element(line),
            "line": line.rstrip(),
        })
    return atoms


def ligand_atoms(atoms, resname: str):
    """Atoms of ONE copy of the ligand (first chain/resseq encountered)."""
    lig = [a for a in atoms if a["record"] == "HETATM" and a["resname"] == resname]
    if not lig:
        raise ValueError(f"Ligand residue '{resname}' not found. HETATM names: "
                         f"{sorted({a['resname'] for a in atoms if a['record'] == 'HETATM'})}")
    first = (lig[0]["chain"], lig[0]["resseq"])
    return [a for a in lig if (a["chain"], a["resseq"]) == first]


def compute_box(lig) -> dict:
    heavy = np.array([a["xyz"] for a in lig if a["element"] != "H"])
    lo, hi = heavy.min(0), heavy.max(0)
    center = (lo + hi) / 2.0
    size = np.clip((hi - lo) + C.BOX_PADDING, C.BOX_MIN_SIDE, C.BOX_MAX_SIDE)
    return {
        "center_x": round(float(center[0]), 3), "center_y": round(float(center[1]), 3),
        "center_z": round(float(center[2]), 3),
        "size_x": round(float(size[0]), 3), "size_y": round(float(size[1]), 3),
        "size_z": round(float(size[2]), 3),
    }


def choose_receptor_chains(atoms, lig, cutoff: float = 6.0):
    """Chains of the protein that actually form the pocket: every chain with protein heavy atoms within
    `cutoff` A of the co-crystal ligand. (The old code kept only the ligand's own chain id, which is often
    NOT the protein chain in a cleaned PDB -> an empty/near-empty receptor and meaningless Vina scores.)"""
    L = np.array([a["xyz"] for a in lig if a["element"] != "H"])
    counts = {}
    for a in atoms:
        if a["record"] != "ATOM" or a["element"] == "H":
            continue
        d = np.linalg.norm(L - np.array(a["xyz"]), axis=1).min()
        if d <= cutoff:
            counts[a["chain"]] = counts.get(a["chain"], 0) + 1
    if not counts:
        raise ValueError("No protein atoms within %.1f A of the ligand - wrong ligand_resname or file?" % cutoff)
    best = max(counts, key=counts.get)
    return [best], counts


def write_protein_only(pdb_path: Path, out_path: Path, chains=None) -> int:
    """Keep ATOM records only (drops ligand, phosphate, water), one altloc per residue, TER between chains.
    chains: iterable of chain ids to keep (None = all). Duplicate atom names inside a residue are dropped
    (a common cause of Meeko's 'explicit valence' error)."""
    keep, prev_chain, seen = [], None, set()
    chains = None if chains is None else set(chains)
    for a in _parse_atoms(pdb_path):
        if a["record"] != "ATOM":
            continue
        if chains is not None and a["chain"] not in chains:
            continue
        key = (a["chain"], a["resseq"], a["line"][26], a["name"])
        if key in seen:
            continue
        seen.add(key)
        if prev_chain is not None and a["chain"] != prev_chain:
            keep.append("TER")
        prev_chain = a["chain"]
        keep.append(a["line"])
    Path(out_path).write_text("\n".join(keep) + "\nTER\nEND\n")
    return sum(1 for l in keep if l != "TER")


MIN_RECEPTOR_ATOMS = 1500   # a kinase domain has ~2000+ heavy atoms; far fewer means the receptor is broken


def _find_pdbqt(prefix: Path) -> Optional[Path]:
    for cand in (Path(f"{prefix}.pdbqt"), Path(f"{prefix}_rigid.pdbqt")):
        if cand.exists():
            return cand
    others = [p for p in prefix.parent.glob(prefix.name + "*.pdbqt") if "flex" not in p.name]
    return others[0] if others else None


def _clear_old(prefix: Path):
    for stale in prefix.parent.glob(prefix.name + "*.pdbqt"):
        stale.unlink()


def _receptor_meeko(protein_pdb: Path, out_prefix: Path):
    """(path, "") on success, (None, error tail) on failure."""
    _clear_old(out_prefix)
    target = Path(f"{out_prefix}.pdbqt")
    exe = shutil.which("mk_prepare_receptor.py") or shutil.which("mk_prepare_receptor")
    if not exe:
        return None, "mk_prepare_receptor not found on PATH"
    last_err = ""
    for base in (["--read_pdb", str(protein_pdb)], ["-i", str(protein_pdb)]):
        for extra in ([], ["--allow_bad_res"]):
            cmd = [exe, *base, "-o", str(out_prefix), "-p", "--default_altloc", "A", *extra]
            res = subprocess.run(cmd, capture_output=True, text=True)
            found = _find_pdbqt(out_prefix)
            if res.returncode == 0 and found is not None:
                if found != target:
                    shutil.copy(found, target)
                return target, ""
            last_err = (res.stderr or res.stdout).strip()[-300:]
    return None, last_err


def _receptor_obabel(protein_pdb: Path, out_prefix: Path):
    _clear_old(out_prefix)
    target = Path(f"{out_prefix}.pdbqt")
    if shutil.which("obabel") is None:
        raise RuntimeError("Open Babel (obabel) is not installed: !apt-get -qq install -y openbabel")
    res = subprocess.run(["obabel", str(protein_pdb), "-O", str(target), "-xr", "-p", "7.4"],
                         capture_output=True, text=True)
    if not target.exists():
        raise RuntimeError(f"Receptor preparation failed:\n{res.stderr}")
    return target


def make_receptor_pdbqt(protein_pdb: Path, out_prefix: Path, method: str = "auto"):
    """Returns (pdbqt_path, method_actually_used)."""
    out_prefix = Path(out_prefix)
    if method in ("auto", "meeko"):
        path, err = _receptor_meeko(protein_pdb, out_prefix)
        if path is not None:
            return path, "meeko"
        if method == "meeko":
            raise RuntimeError(f"Meeko receptor preparation failed:\n{err}")
        print("   [Meeko failed ->", err.splitlines()[-1] if err else "?", "]")
    return _receptor_obabel(protein_pdb, out_prefix), "obabel"


def prepare_receptors(data_dir: Optional[Path] = None, method: Optional[str] = None):
    data_dir = Path(data_dir or C.DATA_DIR)
    method = method or C.RECEPTOR_METHOD
    C.RECEPTOR_DIR.mkdir(parents=True, exist_ok=True)
    boxes, prot_files, atoms_by_target = {}, {}, {}
    for target, info in C.TARGETS.items():
        pdb = data_dir / info["pdb"]
        if not pdb.exists():
            raise FileNotFoundError(f"{pdb} not found (set DRL_DATA or copy files into {data_dir})")
        atoms = _parse_atoms(pdb)
        lig = ligand_atoms(atoms, info["ligand_resname"])
        boxes[target] = compute_box(lig)
        prot_files[target] = C.RECEPTOR_DIR / f"{target}_protein_only.pdb"
        if C.LIGAND_CHAIN_ONLY:
            chains, counts = choose_receptor_chains(atoms, lig)
            print(f"[{target}] pocket chains (atoms within 6 A of ligand): {counts} -> keeping {chains}")
        else:
            chains = None
        n = write_protein_only(pdb, prot_files[target], chains=chains)
        if n < MIN_RECEPTOR_ATOMS:
            raise RuntimeError(f"[{target}] only {n} protein atoms were written to {prot_files[target]} "
                               f"(expected >= {MIN_RECEPTOR_ATOMS}). Docking into it would be meaningless - "
                               f"check the PDB chain ids / C.LIGAND_CHAIN_ONLY.")
        atoms_by_target[target] = (atoms, lig, n)

    used = {}
    for target in C.TARGETS:
        rec, used[target] = make_receptor_pdbqt(prot_files[target], C.RECEPTOR_DIR / f"{target}_receptor", method)
    if len(set(used.values())) > 1:
        print(f"[WARNING] receptors were prepared with different tools {used}; Vina scores would not be "
              f"comparable between targets -> re-preparing ALL with Open Babel.")
        for target in C.TARGETS:
            _, used[target] = make_receptor_pdbqt(prot_files[target], C.RECEPTOR_DIR / f"{target}_receptor", "obabel")

    for target, info in C.TARGETS.items():
        atoms, lig, n = atoms_by_target[target]
        box = boxes[target]
        P = np.array([a["xyz"] for a in atoms if a["record"] == "ATOM" and a["element"] != "H"])
        c = np.array([box[f"center_{k}"] for k in "xyz"]); s = np.array([box[f"size_{k}"] for k in "xyz"])
        inside = int((np.abs(P - c) <= s / 2).all(1).sum())
        # count what actually ended up in the receptor file that Vina reads (not just the input PDB)
        R = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                      for l in (C.RECEPTOR_DIR / (target + "_receptor.pdbqt")).read_text().splitlines()
                      if l.startswith(("ATOM", "HETATM"))])
        in_rec = int((np.abs(R - c) <= s / 2).all(1).sum()) if len(R) else 0
        print(f"[{target}] receptor PDBQT atoms: {len(R)}; inside box: {in_rec}")
        if len(R) < MIN_RECEPTOR_ATOMS or in_rec < 150:
            raise RuntimeError(f"[{target}] receptor PDBQT is broken ({len(R)} atoms, {in_rec} in box).")
        print(f"[{target}] ligand {info['ligand_resname']} ({len(lig)} atoms)  box={box}")
        print(f"[{target}] protein atoms: {n}; heavy atoms inside box: {inside}; "
              f"receptor ({used[target]}): {C.RECEPTOR_DIR / (target + '_receptor.pdbqt')}")
        if inside < 150:
            print(f"   [WARNING] only {inside} receptor atoms inside the box - check the structure/box.")
    out = C.OUT_DIR / "docking_boxes_fixed.json"
    out.write_text(json.dumps(boxes, indent=4))
    print(f"[SAVED] {out}")
    return boxes


# =============================================================================
# docking.py
# =============================================================================
import hashlib
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

try:  # Meeko >= 0.5
    from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
except Exception as exc:  # pragma: no cover
    raise ImportError("Meeko is required: pip install meeko") from exc


RDLogger.DisableLog("rdApp.*")


def standardize_smiles(smiles: str) -> Optional[str]:
    """Largest fragment, neutralised, canonical. None if RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def mol_to_pdbqt(mol: Chem.Mol) -> Optional[str]:
    """RDKit mol WITH explicit Hs and a 3D conformer -> PDBQT string (None on failure)."""
    try:
        setups = MoleculePreparation().prepare(mol)
        for setup in setups:
            pdbqt, ok, _err = PDBQTWriterLegacy.write_string(setup)
            if ok:
                m = re.search(r"TORSDOF\s+(\d+)", pdbqt)
                if m and int(m.group(1)) > C.MAX_TORSIONS:
                    return None
                return pdbqt
    except Exception:
        return None
    return None


def smiles_to_pdbqt(smiles: str, seed: int = 42, n_conf: int = 3) -> Optional[str]:
    """SMILES -> 3D (best of n_conf MMFF-minimised ETKDG conformers) -> PDBQT string (None on any failure)."""
    smi = standardize_smiles(smiles)
    if smi is None:
        return None
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params))
    if not cids:
        return None
    best, best_e = cids[0], float("inf")
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            for cid, (conv, e) in zip(cids, AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)):
                if e < best_e:
                    best, best_e = cid, e
        else:
            AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
    except Exception:
        return None          # force-field parameterisation failed: skip the molecule instead of crashing
    conf_mol = Chem.Mol(mol)
    conf_mol.RemoveAllConformers()
    conf_mol.AddConformer(mol.GetConformer(best), assignId=True)
    return mol_to_pdbqt(conf_mol)


def load_boxes(path: Optional[Path] = None) -> Dict[str, dict]:
    path = Path(path or (C.OUT_DIR / "docking_boxes_fixed.json"))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run prepare_receptors() first")
    return json.loads(path.read_text())


def receptor_path(target: str) -> Path:
    p = C.RECEPTOR_DIR / f"{target}_receptor.pdbqt"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run prepare_receptors() first")
    return p


VINA_ERRORS: List[str] = []   # last reasons Vina failed (inspect after a run)


def run_vina(pdbqt: str, target: str, box: dict, out_pdbqt: Path, exhaustiveness: int,
             num_modes: int, energy_range: int, cpu: int, seed: int, timeout: int):
    """Per-mode affinities (kcal/mol), best first; [] on failure."""
    out_pdbqt = Path(out_pdbqt)
    out_pdbqt.unlink(missing_ok=True)  # never read a stale pose file
    with tempfile.TemporaryDirectory() as td:
        lig = Path(td) / "lig.pdbqt"
        lig.write_text(pdbqt)
        cmd = [
            C.VINA_BIN, "--receptor", str(receptor_path(target)), "--ligand", str(lig),
            "--center_x", str(box["center_x"]), "--center_y", str(box["center_y"]),
            "--center_z", str(box["center_z"]),
            "--size_x", str(box["size_x"]), "--size_y", str(box["size_y"]), "--size_z", str(box["size_z"]),
            "--exhaustiveness", str(exhaustiveness), "--num_modes", str(num_modes),
            "--energy_range", str(energy_range), "--seed", str(seed),
            "--out", str(out_pdbqt),
        ]
        if cpu:
            cmd += ["--cpu", str(cpu)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            VINA_ERRORS.append(f"timeout after {timeout}s"); return []
        except FileNotFoundError:
            VINA_ERRORS.append(f"Vina binary not found: {C.VINA_BIN}"); return []
        if res.returncode != 0 or not out_pdbqt.exists():
            VINA_ERRORS.append((res.stderr or res.stdout).strip()[-300:] or f"exit code {res.returncode}")
            return []
    return [float(m.group(1)) for m in
            re.finditer(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)", out_pdbqt.read_text())]


_REC_TAG: Dict[tuple, str] = {}


def receptor_tag(target: str) -> str:
    """Short hash of the receptor PDBQT, so re-preparing a receptor invalidates its cached scores."""
    p = receptor_path(target)
    st = p.stat()
    k = (str(p), st.st_mtime_ns, st.st_size)
    if k not in _REC_TAG:
        _REC_TAG[k] = hashlib.sha1(p.read_bytes()).hexdigest()[:10]
    return _REC_TAG[k]


def _cache_key(smiles: str, target: str, settings: dict, box: dict) -> str:
    box_s = ",".join(f"{box[k]}" for k in sorted(box))
    raw = (f"{smiles}|{target}|{settings['exhaustiveness']}|{settings['num_modes']}|"
           f"{settings['energy_range']}|{settings['seed']}|{box_s}|{receptor_tag(target)}")
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def _dock_job(args):
    """Dock ONE ligand to ONE target. Returns a result dict."""
    smiles, target, box, settings = args
    C.DOCK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(smiles, target, settings, box)
    cache_file = C.DOCK_CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    result = {"smiles": smiles, "target": target, "score": None, "error": None, "pose": None}
    pdbqt = smiles_to_pdbqt(smiles, seed=settings["seed"])
    if pdbqt is None:
        result["error"] = "ligand_prep_failed"   # deterministic -> safe to cache
        cache_file.write_text(json.dumps(result))
        return result
    out = C.DOCK_CACHE_DIR / f"{key}_pose.pdbqt"
    scores = run_vina(pdbqt, target, box, out, settings["exhaustiveness"], settings["num_modes"],
                      settings["energy_range"], settings["cpu_per_job"], settings["seed"],
                      settings["timeout_s"])
    if scores:
        result["score"], result["pose"] = scores[0], str(out)
        cache_file.write_text(json.dumps(result))
    else:
        result["error"] = "vina_failed"          # maybe transient (timeout) -> NOT cached
    return result


class DualDocker:
    """Docks molecules to every target and returns {target: score_or_None}."""

    def __init__(self, targets: Iterable[str] = ("EGFR", "HER2"), exhaustiveness: int = None,
                 num_modes: int = None, n_workers: int = None, seed: int = None,
                 boxes_path: Optional[Path] = None):
        self.targets = tuple(targets)
        self.boxes = load_boxes(boxes_path)
        for t in self.targets:
            receptor_path(t)  # fail early
        self.n_workers = n_workers or os.cpu_count() or 1
        self.settings = {
            "exhaustiveness": exhaustiveness or C.DOCK["exhaustiveness"],
            "num_modes": num_modes or C.DOCK["num_modes"],
            "energy_range": C.DOCK["energy_range"],
            "seed": seed if seed is not None else C.DOCK["seed"],
            "timeout_s": C.DOCK["timeout_s"],
            "cpu_per_job": 1,   # parallelism is across ligands, not inside Vina
        }
        self._pool = None

    def _get_pool(self):
        # Threads (not processes): the heavy work is a Vina subprocess, and spawned
        # processes cannot import functions defined inside a notebook.
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.n_workers)
        return self._pool

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def dock_many(self, smiles_list: List[str]) -> List[Dict[str, Optional[float]]]:
        std = [standardize_smiles(s) for s in smiles_list]
        unique = list(dict.fromkeys(s for s in std if s is not None))   # dock each molecule once
        jobs = [(s, t, self.boxes[t], self.settings) for s in unique for t in self.targets]
        scores = {}
        if jobs:
            t0, done = time.time(), 0
            def _run(j):
                nonlocal done
                r = _dock_job(j)
                done += 1
                tag = f"score {r['score']}" if r["score"] is not None else f"FAILED ({r['error']})"
                print(f"   [dock {done}/{len(jobs)}] {j[1]} {j[0][:45]} -> {tag} | {time.time()-t0:.0f}s", flush=True)
                return r
            if self.n_workers > 1 and len(jobs) > 1:
                results = list(self._get_pool().map(_run, jobs))
            else:
                results = [_run(j) for j in jobs]
            if VINA_ERRORS:
                print("   [Vina errors, last 3]:", VINA_ERRORS[-3:])
            for (s, t, _, _), r in zip(jobs, results):
                scores[(s, t)] = r["score"]
        return [{t: (scores.get((s, t)) if s is not None else None) for t in self.targets} for s in std]

    def dock_smiles(self, smiles: str) -> Dict[str, Optional[float]]:
        return self.dock_many([smiles])[0]


# ---- validation: redock the co-crystal ligand -----------------------------------
def _xyz_block(lig, heavy_only: bool = False) -> str:
    atoms = [a for a in lig if not (heavy_only and a["element"] == "H")]
    return f"{len(atoms)}\n\n" + "\n".join(
        f"{a['element'].capitalize()} {a['xyz'][0]:.3f} {a['xyz'][1]:.3f} {a['xyz'][2]:.3f}" for a in atoms)


def _crystal_ligand_mol(pdb_path: Path, resname: str, smiles_template: Optional[str] = None) -> Chem.Mol:
    """Crystal ligand as an RDKit mol (with bond orders + 3D coords). Tries, in order:
    1. xyz2mol-style perception from ALL atoms (works only if the PDB ligand has complete hydrogens),
    2. heavy-atom connectivity + bond orders from a template SMILES (TARGETS[...]["ligand_smiles"]),
    3. Open Babel geometry-based perception on the heavy atoms."""
    from rdkit.Chem import rdDetermineBonds
    lig = ligand_atoms(_parse_atoms(pdb_path), resname)
    errors = []

    mol = Chem.MolFromXYZBlock(_xyz_block(lig))
    for charge in (0, 1, -1, 2, -2):
        try:
            m = Chem.Mol(mol)
            rdDetermineBonds.DetermineBonds(m, charge=charge)
            Chem.SanitizeMol(m)
            return m
        except Exception as exc:
            errors.append(f"all-atom perception (charge {charge}): {str(exc)[:80]}")

    heavy = Chem.MolFromXYZBlock(_xyz_block(lig, heavy_only=True))
    if smiles_template:
        try:
            m = Chem.Mol(heavy)
            rdDetermineBonds.DetermineConnectivity(m)
            m = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(smiles_template), m)
            for a in m.GetAtoms():          # XYZ-derived atoms are flagged "no implicit H": reset so Hs get added (and drop radicals)
                a.SetNoImplicit(False); a.SetNumExplicitHs(0); a.SetNumRadicalElectrons(0)
            Chem.SanitizeMol(m)
            return Chem.AddHs(m, addCoords=True)
        except Exception as exc:
            errors.append(f"template: {str(exc)[:80]}")

    if shutil.which("obabel"):
        try:
            with tempfile.TemporaryDirectory() as td:
                xyz, sdf = Path(td) / "lig.xyz", Path(td) / "lig.sdf"
                xyz.write_text(_xyz_block(lig, heavy_only=True))
                subprocess.run(["obabel", str(xyz), "-O", str(sdf), "-h"], capture_output=True, text=True)
                m = next(iter(Chem.SDMolSupplier(str(sdf), removeHs=False)))
                Chem.SanitizeMol(m)
                return m
        except Exception as exc:
            errors.append(f"obabel: {str(exc)[:80]}")
    raise RuntimeError(f"Could not perceive bonds for {resname}. Add a 'ligand_smiles' entry for it in "
                       f"C.TARGETS. Details: {errors[-3:]}")


def _topology_only(mol: Chem.Mol) -> Chem.Mol:
    rw = Chem.RWMol(Chem.RemoveHs(mol, sanitize=False))
    for a in rw.GetAtoms():
        a.SetIsAromatic(False); a.SetFormalCharge(0); a.SetNoImplicit(True); a.SetNumExplicitHs(0)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE); b.SetIsAromatic(False)
    return rw.GetMol()


def redock_reference(target: str, data_dir: Optional[Path] = None, exhaustiveness: int = 16) -> dict:
    """Re-dock the co-crystal ligand; RMSD (A, no re-alignment) <= 2.0 means the workflow is OK."""
    from rdkit.Chem import rdMolAlign
    data_dir = Path(data_dir or C.DATA_DIR)
    info = C.TARGETS[target]
    crystal = _crystal_ligand_mol(data_dir / info["pdb"], info["ligand_resname"], info.get("ligand_smiles"))
    pdbqt = mol_to_pdbqt(Chem.AddHs(crystal, addCoords=True))
    if pdbqt is None:
        raise RuntimeError("Meeko could not prepare the crystal ligand")
    C.DOCK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = C.DOCK_CACHE_DIR / f"redock_{target}.pdbqt"
    scores = run_vina(pdbqt, target, load_boxes()[target], out, exhaustiveness, 9, 3,
                      C.DOCK["cpu"], C.DOCK["seed"], 1800)
    if not scores:
        raise RuntimeError("Vina failed during redocking")
    pmol = PDBQTMolecule(out.read_text(), skip_typing=True)
    pose_mols = RDKitMolCreate.from_pdbqt_mol(pmol)
    ref = _topology_only(crystal)
    rmsds = []
    for pm in pose_mols:
        if pm is None:
            continue
        probe = _topology_only(pm)
        for conf in probe.GetConformers():
            try:
                rmsds.append(rdMolAlign.CalcRMS(probe, ref, prbId=conf.GetId()))
            except Exception:
                rmsds.append(float("nan"))
    top = rmsds[0] if rmsds else float("nan")
    return {"target": target, "best_score": scores[0],
            "top_pose_rmsd": top, "workflow_ok": bool(top <= 2.0),
            "min_rmsd_over_modes": float(np.nanmin(rmsds)) if rmsds and not np.all(np.isnan(rmsds)) else float("nan"),
            "n_modes": len(scores)}


def purge_prep_failures() -> int:
    """Delete cached 'ligand_prep_failed' results (e.g. left over from a run with other torsion limits).
    Good scores are kept. Returns the number of files removed."""
    n = 0
    for f in C.DOCK_CACHE_DIR.glob("*.json"):
        try:
            if json.loads(f.read_text()).get("error") == "ligand_prep_failed":
                f.unlink(); n += 1
        except Exception:
            pass
    return n


# =============================================================================
# reward.py
# =============================================================================
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Descriptors, RDConfig, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams



def _import_sascorer():
    try:
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        return sascorer
    except Exception:
        pass
    import urllib.request
    d = C.OUT_DIR / "sa_score"; d.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/"
    for f in ("sascorer.py", "fpscores.pkl.gz"):
        if not (d / f).exists():
            urllib.request.urlretrieve(base + f, d / f)
    sys.path.insert(0, str(d))
    import sascorer
    return sascorer


sascorer = _import_sascorer()
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

BAD_SMARTS = {
    "peroxide": "[OX2][OX2]",
    "allene": "[#6]=[#6]=[#6]",
    "cumulene_N": "[#6]=[#6]=[#7]",
    "sulfinic_sulfonic_NH": "[SX3](=O)([OH])",
    "sulfamic_acid": "[NX3][SX4](=O)(=O)[OH]",
    "thioacid": "[CX3](=O)[SX2H]",
    "halo_epoxide": "[#6]1([F,Cl,Br])O[#6]1",
    "N_hydroxylamine": "[NX3;!$(N-C=O);!$(N-S=O)][OH]",
    "S_without_oxo": "[#16;X3,X4;!$([#16]=[#8,#16,#7])]",
    "hypervalent_SH": "[SH]([#6])([#6])",
    "N-N-N chain": "[#7]-[#7]-[#7]",
}
_BAD = {k: Chem.MolFromSmarts(v) for k, v in BAD_SMARTS.items()}
_params = FilterCatalogParams()
_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS = FilterCatalog(_params)


def sanity_check(mol: Chem.Mol, min_heavy=12, max_heavy=55, max_small_rings=1) -> Optional[str]:
    """None if OK, otherwise a short reason string."""
    n = mol.GetNumHeavyAtoms()
    if n < min_heavy:
        return "too_small"
    if n > max_heavy:
        return "too_large"
    if len(Chem.GetMolFrags(mol)) > 1:
        return "fragmented"
    for name, patt in _BAD.items():
        if patt is not None and mol.HasSubstructMatch(patt):
            return f"alert:{name}"
    if _PAINS.HasMatch(mol):
        return "alert:PAINS"
    small = sum(1 for r in mol.GetRingInfo().AtomRings() if len(r) < 5)
    if small > max_small_rings:
        return "strained_rings"
    if rdMolDescriptors.CalcNumAromaticRings(mol) == 0 and rdMolDescriptors.CalcNumRings(mol) == 0:
        return "no_rings"
    return None


def dock_to_unit(score: Optional[float]) -> float:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return 0.0
    return float(np.clip((C.SCORE_BAD - score) / (C.SCORE_BAD - C.SCORE_GOOD), 0.0, 1.0))


def balanced_docking_term(score_a: Optional[float], score_b: Optional[float], sigma: float = 2.0) -> float:
    if score_a is None or score_b is None:
        return 0.0
    geo = math.sqrt(dock_to_unit(score_a) * dock_to_unit(score_b))
    balance = math.exp(-((score_a - score_b) ** 2) / (2 * sigma ** 2))
    return geo * (0.5 + 0.5 * balance)


def size_factor(n_heavy: int) -> float:
    if n_heavy <= 40:
        return 1.0
    return float(max(0.5, 1.0 - 0.5 * (n_heavy - 40) / 15.0))


def _as_list(x):
    return [x] if isinstance(x, (str, Path)) else list(x)


def _smiles_column(df: pd.DataFrame) -> pd.Series:
    for col in ("std_smiles", "canonical_smiles", "smiles", "SMILES"):
        if col in df.columns:
            return df[col].dropna()
    raise ValueError(f"No SMILES column found: {list(df.columns)}")


BAD_REFERENCE_FILES = []


def _read_reference_csv(p: Path):
    """Read a reference CSV. Returns None (and records the file) if it is not a real CSV,
    e.g. a GitHub web page saved as .csv (starts with <!DOCTYPE html>)."""
    head = Path(p).read_text(errors="ignore")[:2000].lstrip().lower()
    if head.startswith("<") or "<html" in head:
        BAD_REFERENCE_FILES.append(Path(p).name)
        print(f"[ERROR] {Path(p).name} is an HTML page, not a CSV - download the RAW file (see section 7.0)")
        return None
    return pd.read_csv(p)


def check_reference_files():
    """Report the state of every reference CSV; stop if any is unusable (unless explicitly allowed)."""
    problems = []
    for target, csvs in C.REFERENCE_CSVS.items():
        for csv in _as_list(csvs):
            p = C.DATA_DIR / csv
            if not p.exists():
                print(f"[MISSING] {csv} (skipped - fine if you do not have this split)"); continue
            df = _read_reference_csv(p)
            if df is None:
                problems.append(csv); continue
            try:
                n = len(_smiles_column(df)); print(f"[OK]      {csv}: {n} SMILES")
            except ValueError as e:
                print(f"[ERROR]   {csv}: {e}"); problems.append(csv)
    if problems and not C.ALLOW_INCOMPLETE_REFERENCES:
        raise RuntimeError(f"Unusable reference files: {problems}. Put the real CSVs in {C.DATA_DIR} "
                           "(download the RAW file from GitHub; delete the file if you do not have it), or set C.ALLOW_INCOMPLETE_REFERENCES = True "
                           "and say in the report that novelty / seeds / RL prior used an incomplete set.")
    return problems


def load_reference_fps(data_dir: Optional[Path] = None):
    data_dir = Path(data_dir or C.DATA_DIR)
    fps, seen = [], set()
    for csvs in C.REFERENCE_CSVS.values():
        for csv in _as_list(csvs):
            p = data_dir / csv
            if not p.exists():
                print(f"[reward] reference file missing: {p}")
                continue
            df = _read_reference_csv(p)
            if df is None:
                continue
            for smi in _smiles_column(df):
                m = Chem.MolFromSmiles(smi)
                if m is not None:
                    can = Chem.MolToSmiles(m)
                    if can not in seen:
                        seen.add(can)
                        fps.append(_MORGAN.GetFingerprint(m))
    return fps


@dataclass
class RewardWeights:
    dock: float = 0.60
    qed: float = 0.15
    sa: float = 0.15
    novelty: float = 0.10


@dataclass
class RewardFunction:
    docker: Optional[DualDocker] = None          # None -> property-only reward (fast debug mode)
    weights: RewardWeights = field(default_factory=RewardWeights)
    reference_fps: list = field(default_factory=list)
    targets: tuple = ("EGFR", "HER2")
    history: Dict[str, dict] = field(default_factory=dict)   # canonical smiles -> component dict

    @classmethod
    def build(cls, docker: Optional[DualDocker] = None, data_dir: Optional[Path] = None, **kw):
        return cls(docker=docker, reference_fps=load_reference_fps(data_dir), **kw)

    def novelty(self, mol: Chem.Mol) -> float:
        if not self.reference_fps:
            return 0.0
        return 1.0 - max(DataStructs.BulkTanimotoSimilarity(_MORGAN.GetFingerprint(mol), self.reference_fps))

    def properties(self, smiles: str) -> dict:
        can = standardize_smiles(smiles)
        out = {"smiles": smiles, "canonical": can, "gate_reason": "invalid_smiles"}
        if can is None:
            return out
        mol = Chem.MolFromSmiles(can)
        sa = sascorer.calculateScore(mol)
        out.update({
            "canonical": can, "gate_reason": sanity_check(mol), "heavy_atoms": mol.GetNumHeavyAtoms(),
            "mw": Descriptors.MolWt(mol), "qed": QED.qed(mol), "sa_score": sa,
            "sa_norm": float(np.clip((10.0 - sa) / 9.0, 0, 1)), "novelty": self.novelty(mol),
        })
        return out

    def __call__(self, smiles_list: List[str], return_details: bool = False):
        props = [self.properties(s) for s in smiles_list]
        to_dock = sorted({p["canonical"] for p in props if p["gate_reason"] is None
                          and p["canonical"] not in self.history})
        scores_by_smi = {}
        if self.docker is not None and to_dock:
            for smi, sc in zip(to_dock, self.docker.dock_many(to_dock)):
                scores_by_smi[smi] = sc

        w, rewards, details = self.weights, [], []
        for p in props:
            can = p["canonical"]
            if can in self.history:
                d = self.history[can]
            else:
                d = dict(p)
                if p["gate_reason"] is not None:
                    d.update(reward=0.0, dock_term=0.0)
                else:
                    sc = scores_by_smi.get(can, {t: None for t in self.targets})
                    for t in self.targets:
                        d[f"dock_{t}"] = sc.get(t)
                    a, b = (sc.get(t) for t in self.targets)
                    dterm = balanced_docking_term(a, b) if self.docker is not None else 0.0
                    d["dock_term"] = dterm
                    d["dock_diff"] = None if a is None or b is None else abs(a - b)
                    eps = 1e-3
                    if self.docker is None:   # property-only debug mode
                        base = (max(p["qed"], eps) ** w.qed * max(p["sa_norm"], eps) ** w.sa
                                * (0.5 + 0.5 * p["novelty"]) ** w.novelty) ** (1.0 / (1.0 - w.dock))
                    elif a is None or b is None:
                        base = 0.0            # docking failed -> no reward ...
                        d["dock_failed"] = True   # ... but do not memoise it (a timeout may be transient)
                    else:
                        base = (max(dterm, eps) ** w.dock * max(p["qed"], eps) ** w.qed
                                * max(p["sa_norm"], eps) ** w.sa * (0.5 + 0.5 * p["novelty"]) ** w.novelty)
                    d["reward"] = float(base * size_factor(p["heavy_atoms"]))
                if can is not None and not d.get("dock_failed"):
                    self.history[can] = d
            rewards.append(d["reward"])
            details.append(d)
        return (rewards, details) if return_details else rewards


# =============================================================================
# optimize.py
# =============================================================================
import copy
import itertools
import random
from typing import Callable, Iterable, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS


def history_frame(rf: RewardFunction) -> pd.DataFrame:
    rows = [dict(v, canonical=k) for k, v in rf.history.items() if k is not None]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("reward", ascending=False).reset_index(drop=True)


def export_candidates(rf: RewardFunction, out_csv: Path, top_n: int = 50,
                      df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Top-N molecules that passed the chemistry gate (df: use this table instead of rf.history)."""
    df = history_frame(rf) if df is None else df
    df = df[df["gate_reason"].isna()] if "gate_reason" in df else df
    df = df.sort_values("reward", ascending=False) if "reward" in df else df
    cols = [c for c in ["canonical", "reward", "dock_EGFR", "dock_HER2", "dock_diff", "dock_term",
                        "qed", "sa_score", "novelty", "mw", "heavy_atoms"] if c in df.columns]
    top = df[cols].head(top_n)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(out_csv, index=False)
    return top


_MUTATIONS = [
    "[cH:1]>>[c:1]C", "[cH:1]>>[c:1]F", "[cH:1]>>[c:1]Cl", "[cH:1]>>[c:1]OC", "[cH:1]>>[c:1]N",
    "[cH:1]>>[c:1]C#N", "[cH:1]>>[c:1]C(F)(F)F", "[cH:1]>>[c:1]O", "[cH:1]>>[n:1]",
    "[Cl:1]>>[Br:1]", "[Br:1]>>[Cl:1]", "[F:1]>>[Cl:1]", "[CH3:1][O:2]>>[CH2:1]([O:2])C",
    "[C;!R;H2:1]-[C;!R:2]>>[C:1](C)-[C:2]", "[NH2:1]>>[NH:1]C", "[OH:1]>>[O:1]C",
    "[c:1][NH2]>>[c:1]NC(=O)C", "[CH2;!R:1]>>[O:1]", "[CH2;!R:1]>>[NH:1]",
]
_REACTIONS = [AllChem.ReactionFromSmarts(s) for s in _MUTATIONS]
_REMOVE_TERMINAL = AllChem.ReactionFromSmarts("[*;D1;!$(*=*);!$([#1]):1]-[*;!D1:2]>>[*:2]")


def _clean(mol: Optional[Chem.Mol]) -> Optional[str]:
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        return smi if Chem.MolFromSmiles(smi) is not None else None
    except Exception:
        return None


def mutate(smiles: str, rng: random.Random) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    own = Chem.MolToSmiles(mol)
    rxns = _REACTIONS + [_REMOVE_TERMINAL]
    rng.shuffle(rxns)
    for rxn in rxns[:6]:
        try:
            prods = rxn.RunReactants((mol,))
        except Exception:
            continue
        outs = {s for s in (_clean(p[0]) for p in prods) if s and s != own}
        if outs:
            return rng.choice(sorted(outs))
    return None


def brics_crossover(a: str, b: str, rng: random.Random, n_try: int = 12) -> Optional[str]:
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if ma is None or mb is None:
        return None
    frags = sorted(set(BRICS.BRICSDecompose(ma)) | set(BRICS.BRICSDecompose(mb)))
    frag_mols = [f for f in (Chem.MolFromSmiles(x) for x in frags) if f is not None]
    if len(frag_mols) < 2:
        return None
    random.seed(rng.random())  # BRICSBuild shuffles with the global `random`
    try:
        built = list(itertools.islice(BRICS.BRICSBuild(frag_mols, onlyCompleteMols=True, scrambleReagents=True), n_try))
    except Exception:
        return None
    outs = [s for s in (_clean(m) for m in built) if s and s not in (a, b)]
    return rng.choice(outs) if outs else None


def ga_optimize(rf: RewardFunction, seeds: Iterable[str], generations: int = 6, pop_size: int = 24,
                n_offspring: int = 24, mutation_rate: float = 0.6, crossover_rate: float = 0.5,
                seed: int = 0, out_dir: Optional[Path] = None, log: Callable = print) -> pd.DataFrame:
    """Evolve molecules to maximise the dual-target reward."""
    rng = random.Random(seed)
    seeds = [s for s in dict.fromkeys(standardize_smiles(x) for x in seeds if x) if s]
    if not seeds:
        raise ValueError("No valid seed molecules for the GA.")
    rewards = rf(seeds)
    pop = sorted(zip(rewards, seeds), reverse=True)[:pop_size]
    log(f"[GA] initial population {len(pop)} | best {pop[0][0]:.3f} | mean {np.mean([r for r, _ in pop]):.3f}")
    gen_log = []
    for g in range(1, generations + 1):
        t0 = time.time()
        offspring, tries = [], 0
        while len(offspring) < n_offspring and tries < n_offspring * 40:
            tries += 1
            p1 = max(rng.sample(pop, min(3, len(pop))))[1]   # tournament selection (size 3)
            p2 = max(rng.sample(pop, min(3, len(pop))))[1]
            child = (brics_crossover(p1, p2, rng) if rng.random() < crossover_rate else p1) or p1
            if child and rng.random() < mutation_rate:
                child = mutate(child, rng) or child
            child = standardize_smiles(child) if child else None
            if not child or child in rf.history or child in offspring:
                continue
            if rf.properties(child)["gate_reason"] is not None:   # skip junk before docking
                continue
            offspring.append(child)
        child_rewards = rf(offspring) if offspring else []
        merged = {s: r for r, s in pop}
        merged.update(dict(zip(offspring, child_rewards)))
        pop = sorted(((r, s) for s, r in merged.items()), reverse=True)[:pop_size]
        gen_log.append({"generation": g, "n_new": len(offspring), "best": pop[0][0],
                        "mean": float(np.mean([r for r, _ in pop])), "seconds": time.time() - t0})
        log(f"[GA] gen {g}/{generations}: +{len(offspring)} new | best {pop[0][0]:.3f} "
            f"({pop[0][1][:60]}) | mean {gen_log[-1]['mean']:.3f} | {gen_log[-1]['seconds']:.0f}s")
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(gen_log).to_csv(out_dir / "ga_generations.csv", index=False)
        export_candidates(rf, out_dir / "ga_top_candidates.csv")
        history_frame(rf).to_csv(out_dir / "ga_all_scored.csv", index=False)
    return history_frame(rf)

# ---- RL (REINVENT-style) ------------------------------------------------------------
_TOKEN_RE = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")


def tokenize(smiles: str) -> List[str]:
    toks = _TOKEN_RE.findall(smiles)
    if "".join(toks) != smiles:
        raise ValueError(f"untokenisable: {smiles}")
    return toks


class Vocab:
    PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"

    def __init__(self, smiles_list: Iterable[str]):
        toks = sorted({t for s in smiles_list for t in tokenize(s)})
        self.itos = [self.PAD, self.BOS, self.EOS] + toks
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, smiles: str) -> List[int]:
        return [self.stoi[self.BOS]] + [self.stoi[t] for t in tokenize(smiles)] + [self.stoi[self.EOS]]

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            t = self.itos[i]
            if t == self.EOS:
                break
            if t not in (self.PAD, self.BOS):
                out.append(t)
        return "".join(out)


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def make_policy(vocab_size: int, emb: int = 128, hidden: int = 384, layers: int = 2):
    torch, nn = _torch()

    class SmilesRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
            self.rnn = nn.GRU(emb, hidden, layers, batch_first=True)
            self.out = nn.Linear(hidden, vocab_size)

        def forward(self, x, h=None):
            o, h = self.rnn(self.emb(x), h)
            return self.out(o), h

        def log_prob(self, seqs):
            """seqs: (B, T) starting with BOS. Summed log-prob of tokens 1..T (pad masked)."""
            logits, _ = self.forward(seqs[:, :-1])
            lp = torch.log_softmax(logits, -1).gather(2, seqs[:, 1:].unsqueeze(2)).squeeze(2)
            mask = (seqs[:, 1:] != 0).float()
            return (lp * mask).sum(1)

        @torch.no_grad()
        def sample(self, n, bos, eos, max_len=120, device="cpu"):
            x = torch.full((n, 1), bos, dtype=torch.long, device=device)
            seqs, h, done = [x], None, torch.zeros(n, dtype=torch.bool, device=device)
            for _ in range(max_len):
                logits, h = self.forward(x, h)
                nxt = torch.multinomial(torch.softmax(logits[:, -1], -1), 1)
                nxt[done] = 0
                seqs.append(nxt)
                done |= nxt.squeeze(1) == eos
                x = nxt
                if done.all():
                    break
            return torch.cat(seqs, 1)

    return SmilesRNN()


class RLFineTuner:
    def __init__(self, reward_fn: RewardFunction, train_smiles: List[str], device: str = "cpu", seed: int = 0):
        torch, _ = _torch()
        torch.manual_seed(seed); np.random.seed(seed)
        self.torch, self.rf, self.device = torch, reward_fn, device
        ok = []
        for s in train_smiles:
            try:
                tokenize(s); ok.append(s)
            except ValueError:
                pass
        if not ok:
            raise ValueError("No tokenisable training SMILES for the RL prior.")
        self.train_smiles = ok
        self.vocab = Vocab(ok)
        self.prior = make_policy(len(self.vocab.itos)).to(device)
        self.agent = None

    def _augmented_encoding(self):
        """Randomised (non-canonical) SMILES each epoch -> better validity and diversity."""
        enc = []
        for s in self.train_smiles:
            m = Chem.MolFromSmiles(s)
            r = Chem.MolToSmiles(m, doRandom=True) if m is not None else s
            try:
                enc.append(self.vocab.encode(r))
            except (KeyError, ValueError):
                enc.append(self.vocab.encode(s))
        return enc

    def pretrain(self, epochs: int = 60, batch_size: int = 64, lr: float = 1e-3, log=print):
        torch = self.torch
        opt = torch.optim.Adam(self.prior.parameters(), lr=lr)
        for ep in range(1, epochs + 1):
            enc = self._augmented_encoding()
            np.random.shuffle(enc)
            total = 0.0
            for i in range(0, len(enc), batch_size):
                batch = enc[i:i + batch_size]
                T = max(map(len, batch))
                x = torch.zeros(len(batch), T, dtype=torch.long)
                for j, e in enumerate(batch):
                    x[j, :len(e)] = torch.tensor(e)
                x = x.to(self.device)
                n_tok = (x[:, 1:] != 0).sum(1).clamp(min=1)          # real tokens only (not padding)
                loss = (-self.prior.log_prob(x) / n_tok).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.prior.parameters(), 5.0)
                opt.step(); total += loss.item() * len(batch)
            if ep % 5 == 0 or ep == epochs:
                log(f"[pretrain] epoch {ep}/{epochs} nll/token {total / len(enc):.3f}")
        self.prior.eval()
        for p in self.prior.parameters():
            p.requires_grad_(False)

    def save_prior(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({"state": self.prior.state_dict(), "vocab": self.vocab.itos}, path)

    def load_prior(self, path: Path):
        ckpt = self.torch.load(path, map_location=self.device)
        self.vocab.itos = ckpt["vocab"]
        self.vocab.stoi = {t: i for i, t in enumerate(self.vocab.itos)}
        self.prior = make_policy(len(self.vocab.itos)).to(self.device)
        self.prior.load_state_dict(ckpt["state"])
        self.prior.eval()
        for p in self.prior.parameters():
            p.requires_grad_(False)

    def _sample_smiles(self, model, n):
        seqs = model.sample(n, self.vocab.stoi[Vocab.BOS], self.vocab.stoi[Vocab.EOS], device=self.device)
        return seqs, [self.vocab.decode(s.tolist()) for s in seqs]

    def validity(self, n=200) -> dict:
        _, smi = self._sample_smiles(self.agent if self.agent is not None else self.prior, n)
        valid = [s for s in smi if s and Chem.MolFromSmiles(s) is not None]
        return {"valid_frac": len(valid) / n, "unique_frac": len(set(valid)) / max(1, len(valid))}

    # ---- policy-gradient fine-tuning (resumable) ------------------------------------------------
    def reinforce(self, steps: int = 30, batch_size: int = 16, lr: float = 5e-4, sigma: float = 20.0,
                  out_dir: Optional[Path] = None, chunk: Optional[int] = None, resume: bool = True,
                  log=print) -> pd.DataFrame:
        """REINVENT augmented-likelihood fine-tuning, checkpointed after every step.

        steps   total number of steps for the whole run
        chunk   run at most this many steps now, then stop (call again to continue); None = all
        resume  continue from out_dir/rl_checkpoint.pt if present; False deletes it and starts over
        A step that raises is skipped (logged) instead of killing the run. Resuming restores the
        optimiser and random-number state, but needs the same batch_size / lr / sigma.
        Files in out_dir (default: results folder): rl_checkpoint.pt, *_partial.csv while running;
        rl_training_curve.csv, rl_all_scored.csv, rl_top_candidates.csv, rl_policy.pt when finished.
        """
        torch = self.torch
        out_dir = Path(out_dir or C.RESULTS_DIR); out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_f, curve_f, hist_f = (out_dir / "rl_checkpoint.pt", out_dir / "rl_training_curve_partial.csv",
                                   out_dir / "rl_all_scored_partial.csv")
        cfg = {"batch_size": batch_size, "lr": lr, "sigma": sigma}
        if not resume:
            for f in (ckpt_f, curve_f, hist_f):
                f.unlink(missing_ok=True)

        self.agent = copy.deepcopy(self.prior)
        for p in self.agent.parameters():
            p.requires_grad_(True)
        self.agent.train()
        opt = torch.optim.Adam(self.agent.parameters(), lr=lr)

        start, curve = 0, []
        if ckpt_f.exists():
            ck = torch.load(ckpt_f, map_location="cpu")
            if ck.get("cfg", cfg) != cfg:
                raise ValueError(f"checkpoint was made with {ck['cfg']}, now {cfg}; "
                                 "use the same values or resume=False to start over")
            self.agent.load_state_dict(ck["agent"]); opt.load_state_dict(ck["opt"])
            if "rng" in ck:
                torch.set_rng_state(ck["rng"])
            start = int(ck["step"])
            curve = pd.read_csv(curve_f).to_dict("records") if curve_f.exists() else []
            log(f"[RL] resuming from step {start}/{steps}")
        else:
            log("[RL] starting a new run")

        end = steps if chunk is None else min(steps, start + chunk)
        for step in range(start + 1, end + 1):
            t0 = time.time()
            try:
                seqs, smiles = self._sample_smiles(self.agent, batch_size)
                rewards = self.rf(smiles)                       # invalid / junk -> 0, docked otherwise
                r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                agent_lp = self.agent.log_prob(seqs)
                with torch.no_grad():
                    prior_lp = self.prior.log_prob(seqs)
                loss = ((prior_lp + sigma * r - agent_lp) ** 2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 5.0)
                opt.step()
            except Exception as exc:
                log(f"[RL] step {step} failed ({type(exc).__name__}: {str(exc)[:120]}) - skipped")
                continue
            valid = sum(bool(s) and Chem.MolFromSmiles(s) is not None for s in smiles) / batch_size
            curve.append({"step": step, "mean_reward": float(np.mean(rewards)), "max_reward": float(np.max(rewards)),
                          "valid_frac": valid, "loss": loss.item(), "seconds": time.time() - t0})
            log(f"[RL] step {step}/{steps}: mean R {curve[-1]['mean_reward']:.3f} | max {curve[-1]['max_reward']:.3f} "
                f"| valid {valid:.2f} | {curve[-1]['seconds']:.0f}s")
            torch.save({"agent": self.agent.state_dict(), "opt": opt.state_dict(), "step": step,
                        "rng": torch.get_rng_state(), "cfg": cfg}, ckpt_f)
            pd.DataFrame(curve).to_csv(curve_f, index=False)
            self._save_history(hist_f)

        if end < steps:
            log(f"[RL] paused at step {end}/{steps}; call reinforce() again to continue")
            return pd.DataFrame(curve)

        pd.DataFrame(curve).to_csv(out_dir / "rl_training_curve.csv", index=False)
        self._save_history(hist_f)
        hist = pd.read_csv(hist_f) if hist_f.exists() else history_frame(self.rf)
        hist.to_csv(out_dir / "rl_all_scored.csv", index=False)
        export_candidates(self.rf, out_dir / "rl_top_candidates.csv", df=hist)
        torch.save({"agent": self.agent.state_dict(), "prior": self.prior.state_dict(),
                    "vocab": self.vocab.itos}, out_dir / "rl_policy.pt")
        log("[RL] finished: rl_training_curve.csv, rl_all_scored.csv, rl_top_candidates.csv, rl_policy.pt written")
        return pd.DataFrame(curve)

    def _save_history(self, path: Path):
        """Merge everything scored so far (this session + earlier sessions) into `path`."""
        h = history_frame(self.rf)
        if Path(path).exists():
            h = pd.concat([pd.read_csv(path), h], ignore_index=True)
        if h.empty:                      # nothing valid scored yet: do not write an unreadable empty CSV
            return
        h.drop_duplicates("canonical", keep="last").sort_values("reward", ascending=False).to_csv(path, index=False)


# =============================================================================
# pipeline.py
# =============================================================================
from typing import List, Optional





def read_smiles(path: Path):
    path = Path(path)
    if path.suffix.lower() == ".sdf":
        text = path.read_text(errors="ignore")
        n_records = text.count("$$$$")
        smi = [Chem.MolToSmiles(m) for m in Chem.SDMolSupplier(str(path)) if m is not None]
        if len(smi) < n_records:   # malformed records (e.g. extra blank header line): use the SMILES data field
            alt = [m.group(1).strip() for rec in text.replace("\r\n", "\n").split("$$$$")
                   for m in [re.search(r">\s*<(?:canonical_smiles|smiles|SMILES)>\s*\n([^\n]+)", rec)] if m]
            if len(alt) > len(smi):
                print(f"[read_smiles] RDKit read {len(smi)}/{n_records} SDF records; using the SMILES field ({len(alt)})")
                return alt
        return smi
    df = pd.read_csv(path)
    for col in ("canonical_smiles", "smiles", "std_smiles", "SMILES"):
        if col in df.columns:
            return df[col].dropna().tolist()
    raise ValueError(f"No SMILES column in {path}: {list(df.columns)}")


def read_candidates(path: Path) -> pd.DataFrame:
    """Candidates as a table with columns candidate_id, smiles (CSV manifest with candidate_id, or SDF)."""
    path = Path(path)
    smiles = read_smiles(path)
    ids = None
    if path.suffix.lower() != ".sdf":
        df = pd.read_csv(path)
        if "candidate_id" in df.columns and len(df) == len(smiles):
            ids = df["candidate_id"].tolist()
    return pd.DataFrame({"candidate_id": ids or [f"cand_{i + 1:03d}" for i in range(len(smiles))], "smiles": smiles})


def seed_smiles(n_per_target: Optional[int] = 15):
    """Most potent known actives of each target (None = all). Falls back to reference drugs."""
    seeds = []
    for target, csvs in C.REFERENCE_CSVS.items():
        frames = []
        for csv in _as_list(csvs):
            p = C.DATA_DIR / csv
            if p.exists():
                df_ = _read_reference_csv(p)
                if df_ is not None:
                    frames.append(df_)
            else:
                print(f"[seed_smiles] missing {p}")
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        if "standard_value" in df.columns:
            df = df.sort_values("standard_value")          # IC50-like: lowest = most potent
        smi = _smiles_column(df).drop_duplicates()
        seeds += smi.head(n_per_target).tolist() if n_per_target else smi.tolist()
    if not seeds:
        print("[seed_smiles] reference CSVs not found - using the 3 reference drugs as seeds")
        seeds = [REF_DRUGS[k] for k in ("gefitinib", "lapatinib", "erlotinib")]
    return seeds


def build_reward(no_dock: bool = False, exhaustiveness: Optional[int] = None, workers: Optional[int] = None) -> RewardFunction:
    """The ONE place a docker + reward function is created (so every stage uses the same settings)."""
    docker = None if no_dock else DualDocker(exhaustiveness=exhaustiveness, n_workers=workers)
    return RewardFunction.build(docker=docker)


def close_reward(rf: Optional[RewardFunction]) -> None:
    """Shut down the docker's worker threads (safe to call with None / property-only rewards)."""
    if rf is not None and getattr(rf, "docker", None) is not None:
        rf.docker.close()


def run_validate(exhaustiveness: int = 16) -> pd.DataFrame:
    """Re-dock the co-crystal ligands. Top-pose RMSD <= 2.0 A means the workflow reproduces the crystal pose."""
    rows = [redock_reference(t, exhaustiveness=exhaustiveness) for t in C.TARGETS]
    df = pd.DataFrame(rows)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RESULTS_DIR / "redocking_validation.csv", index=False)
    for r in rows:
        status = "PASS" if r["workflow_ok"] else "FAIL (report this; do not relax the threshold)"
        print(f"[validate] {r['target']}: top-pose RMSD {r['top_pose_rmsd']:.2f} A "
              f"(best of {r['n_modes']} modes {r['min_rmsd_over_modes']:.2f} A), score {r['best_score']:.1f} -> {status}")
    return df


def run_score(rf: RewardFunction, smiles: List[str], ids: Optional[List[str]] = None,
              out_name: str = "scored_candidates.csv") -> pd.DataFrame:
    _, details = rf(smiles, return_details=True)
    df = pd.DataFrame(details)
    if ids is not None:
        df.insert(0, "candidate_id", list(ids))
    df = df.sort_values("reward", ascending=False)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RESULTS_DIR / out_name, index=False)
    cols = [c for c in ["candidate_id", "canonical", "gate_reason", "dock_EGFR", "dock_HER2", "qed", "sa_score",
                        "novelty", "reward"] if c in df]
    print(df[cols].head(30).to_string(index=False))
    return df


def run_ga(rf, generations: int = 6, n_seeds: int = 15, pop: int = 24, offspring: int = 24, seed: int = 0,
           extra_smiles=None) -> pd.DataFrame:
    seeds = seed_smiles(n_seeds) + list(extra_smiles or [])
    return ga_optimize(rf, seeds, generations=generations, pop_size=pop, n_offspring=offspring,
                       seed=seed, out_dir=C.RESULTS_DIR)


def run_rl(rf, steps: int = 30, batch: int = 16, pretrain_epochs: int = 60, seed: int = 0,
           chunk: Optional[int] = None, resume: bool = True, prior_ckpt: Optional[Path] = None):
    """Resumable REINVENT-style fine-tuning (see RLFineTuner.reinforce). Returns (tuner, curve)."""
    prior_ckpt = Path(prior_ckpt or C.RESULTS_DIR / "prior.pt")
    train = [s for s in (standardize_smiles(x) for x in seed_smiles(None)) if s]
    tuner = RLFineTuner(rf, train, seed=seed)
    if prior_ckpt.exists():
        tuner.load_prior(prior_ckpt)
    else:
        tuner.pretrain(epochs=pretrain_epochs)
        tuner.save_prior(prior_ckpt)
    print("prior sampling quality:", tuner.validity())
    curve = tuner.reinforce(steps=steps, batch_size=batch, chunk=chunk, resume=resume, out_dir=C.RESULTS_DIR)
    print("agent sampling quality:", tuner.validity())
    return tuner, curve


def select_final(min_novelty: float = 0.3, max_dock: float = -8.0,
                 source: str = "ga_top_candidates.csv", out_name: str = "initial_optimised_candidates.csv") -> pd.DataFrame:
    """Final candidate set: novelty and BOTH docking scores must clear hand-chosen cutoffs
    (not tuned - state them in the report)."""
    top = pd.read_csv(C.RESULTS_DIR / source)
    final = top[(top.novelty > min_novelty) & (top.dock_EGFR < max_dock) & (top.dock_HER2 < max_dock)].copy()
    final.to_csv(C.RESULTS_DIR / out_name, index=False)
    print(f"{len(final)} of {len(top)} candidates pass novelty > {min_novelty} and docking < {max_dock} on both targets")
    return final


# =============================================================================== command line
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "validate", "score", "ga", "rl", "final", "all"])
    ap.add_argument("--input", type=Path, help="candidate file (.csv with SMILES column / .sdf); default: data/ manifest")
    ap.add_argument("--exhaustiveness", type=int, default=None, help="default: C.DOCK['exhaustiveness']")
    ap.add_argument("--validate-exhaustiveness", type=int, default=16)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-dock", action="store_true", help="property-only reward (debug)")
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=8, help="GA seeds per target (most potent actives)")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--offspring", type=int, default=12)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=None, help="RL: run at most this many steps, then stop")
    ap.add_argument("--restart", action="store_true", help="RL: ignore any checkpoint and start over")
    ap.add_argument("--pretrain-epochs", type=int, default=60)
    ap.add_argument("--min-novelty", type=float, default=0.3)
    ap.add_argument("--max-dock", type=float, default=-8.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    steps = ["prepare", "validate", "score", "ga", "rl", "final"] if a.cmd == "all" else [a.cmd]

    check_reference_files()
    if "prepare" in steps:
        prepare_receptors()
    if "validate" in steps:
        run_validate(a.validate_exhaustiveness)
    if not set(steps) & {"score", "ga", "rl"}:
        if "final" in steps:
            select_final(a.min_novelty, a.max_dock)
        return

    cand_file = a.input or next((p for p in (C.DATA_DIR / "dual_target_candidates.sdf",
                                             C.DATA_DIR / "dual_target_candidate_manifest.csv") if p.exists()), None)
    cands = read_candidates(cand_file) if cand_file else None
    rf = None
    try:
        if "score" in steps:
            if cands is None:
                raise SystemExit("no candidate file: pass --input or put one in the data folder")
            rf = build_reward(a.no_dock, a.exhaustiveness, a.workers)
            run_score(rf, cands.smiles.tolist(), ids=cands.candidate_id.tolist())
        if "ga" in steps:
            close_reward(rf); rf = build_reward(a.no_dock, a.exhaustiveness, a.workers)   # fresh history per stage
            extra = [REF_DRUGS["gefitinib"], REF_DRUGS["erlotinib"]]
            run_ga(rf, a.generations, a.n_seeds, a.pop, a.offspring, a.seed, extra)
        if "rl" in steps:
            close_reward(rf); rf = build_reward(a.no_dock, a.exhaustiveness, a.workers)
            run_rl(rf, a.steps, a.batch, a.pretrain_epochs, a.seed, chunk=a.chunk, resume=not a.restart)
        if "final" in steps:
            select_final(a.min_novelty, a.max_dock)
    finally:
        close_reward(rf)


if __name__ == "__main__":
    main()
