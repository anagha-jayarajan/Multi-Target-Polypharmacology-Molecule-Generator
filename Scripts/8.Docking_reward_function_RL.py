#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Docking reward function / dual-target (EGFR+HER2) RL molecule generator.

Converted from 8.Docking_Reward_Fuction_.ipynb (Colab). A GPU speeds up pre-training and RL sampling.
"""

# ---- Colab / plain-Python compatibility shims ----
try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)
try:
    from google.colab import drive, files
    IN_COLAB = True
except ImportError:
    drive = files = None
    IN_COLAB = False

# --- cell 0 ---
import subprocess, torch
try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
except FileNotFoundError:
    pass
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| CUDA", torch.version.cuda)
else:
    print("WARNING: no GPU found (Runtime > Change runtime type > T4 GPU). The notebook still runs on CPU, "
          "but pre-training and RL sampling are slower.")
torch.backends.cudnn.benchmark = True

# ======================================================================
# ## 1. Run settings

# --- cell 2 ---
# from google.colab import drive
import os
import pandas as pd

# Mount (safe to re-run)
if IN_COLAB:
    drive.mount('/content/drive')

# Point directly at the file on Drive
drive_path = '/content/drive/MyDrive/my_folder/my_file.csv'

# If the file doesn't exist, we will create dummy data so the pipeline doesn't break
if not os.path.exists(drive_path):
    print(f"File not found at {drive_path}. Creating a dummy template file...")
    os.makedirs(os.path.dirname(drive_path), exist_ok=True)
    dummy_df = pd.DataFrame({
        'smiles': ['c1ccccc1O', 'CC(C)(O)CC(=O)NCCn1ccc2ncnc(Nc3ccc(Oc4cccc(C(F)(F)F)c4)c(Cl)c3)c21'],
        'label': ['phenol', 'TAK-285']
    })
    dummy_df.to_csv(drive_path, index=False)

# Check the path before loading
print("File exists:", os.path.exists(drive_path))
print("Contents of MyDrive:", os.listdir('/content/drive/MyDrive'))

# Load it
df = pd.read_csv(drive_path)
print("\nLoaded DataFrame:")
print(df.head())

# ... do your work ...

# Save back to Drive
out_path = '/content/drive/MyDrive/my_folder/my_file_updated.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print("Saved to", out_path)

# --- cell 3 ---
# ----------------------------- RUN SETTINGS ----------------------------- #
import time
T0 = time.time()                 # the clock for the whole notebook starts here

TIME_BUDGET_MIN = 80             # wall-clock target for the WHOLE notebook (minutes). 80 leaves ~10 min of safety in a 90-min run.
DOCK_MODE  = "vina"              # "vina" = real AutoDock Vina | "mock" = fast fake scores (smoke test only, separate output folder)
QUICK_TEST = False               # True = ~15-minute smoke test of the whole pipeline

# ---- docking cost (Vina is CPU-only; in the earlier run exhaustiveness 4 took ~110 s per lapatinib-sized docking on one core) ----
SCREEN_EXH = 2                   # exhaustiveness for the benchmark, the RL loop and the final evaluation
REDOCK_EXH = 8                   # exhaustiveness for the two redocking checks (accuracy matters there)

# ---- RL (REINVENT-style).  Number of steps is NOT fixed: each mode runs until its share of the time budget is used ----
RL_MAX_STEPS  = 30               # upper limit on steps per mode (only reached if docking is fast)
RL_SAMPLE     = 32               # sequences sampled per step (cheap, GPU)
RL_DOCK_K     = 5                # at most this many valid, unique molecules are docked per step (docking is the bottleneck)
RL_NEG        = 4                # invalid samples added to each update with reward 0 (keeps SMILES validity up)
RL_SIGMA      = 60               # weight of the reward in the REINVENT augmented likelihood
RL_LR         = 5e-4             # original value; higher rates collapsed SMILES validity in my mock test
RL_EXH        = SCREEN_EXH
RL_SHARE      = {"dual": 0.50, "EGFR": 0.25, "HER2": 0.25}     # split of the RL time; dual costs 2 dockings per molecule

# ---- final evaluation ----
N_EVAL        = 500              # molecules sampled per model for validity / uniqueness / novelty / QED / SA (cheap)
EVAL_EXH      = SCREEN_EXH
MIN_EVAL_DOCK, MAX_EVAL_DOCK = 4, 40      # molecules per model that are docked on both targets (chosen from the time left)

# ---- time reserves (minutes) ----
EVAL_RESERVE_MIN   = 18          # kept free for the final evaluation docking
REPORT_RESERVE_MIN = 5           # kept free for tables, plots, report and zip

PRETRAIN_EPOCHS = 500
PRETRAIN_CSV  = None  # optional: path to a CSV with a 'smiles' column (e.g. ChEMBL EGFR/HER2 actives). None = built-in kinase-inhibitor seed set
HIT_THRESHOLD = -9.0  # kcal/mol; a molecule is a "dual hit" if both docking scores are <= this
SEED          = 42

if QUICK_TEST:
    TIME_BUDGET_MIN, REDOCK_EXH, PRETRAIN_EPOCHS = 15, 4, 200
    N_EVAL, MIN_EVAL_DOCK, MAX_EVAL_DOCK, RL_DOCK_K = 100, 2, 6, 3
    EVAL_RESERVE_MIN, REPORT_RESERVE_MIN = 4, 2

BUDGET_END = T0 + TIME_BUDGET_MIN * 60
def elapsed_min(): return (time.time() - T0) / 60
print(f"Time budget: {TIME_BUDGET_MIN} min | screening exhaustiveness {SCREEN_EXH} | redocking exhaustiveness {REDOCK_EXH}")

# ======================================================================
# ## 2. Install / check tools (Colab or Linux)

# --- cell 5 ---
import importlib, os, shutil, subprocess, sys

def _pip(*pk):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pk], check=False)

for mod, pkg in {"rdkit": "rdkit", "meeko": "meeko", "gemmi": "gemmi", "scipy": "scipy", "pandas": "pandas",
                 "tabulate": "tabulate", "torch": "torch", "matplotlib": "matplotlib"}.items():
    try:
        importlib.import_module(mod)
    except ImportError:
        print("installing", pkg); _pip(pkg)

if DOCK_MODE == "vina":
    if shutil.which("obabel") is None:
        subprocess.run("apt-get -qq update && apt-get -qq install -y openbabel", shell=True)
    if not os.environ.get("VINA_BIN") and shutil.which("vina") is None:
        url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
        os.makedirs(".bin", exist_ok=True)
        subprocess.run(["wget", "-q", "-O", ".bin/vina", url], check=False)
        os.chmod(".bin/vina", 0o755)
        os.environ["VINA_BIN"] = os.path.abspath(".bin/vina")
    print("obabel:", shutil.which("obabel"))
    print("vina  :", os.environ.get("VINA_BIN") or shutil.which("vina"))
    assert shutil.which("obabel"), "Open Babel (obabel) is missing - install it (apt-get install openbabel)."
else:
    print("MOCK MODE: no docking tools needed. Results are NOT real.")

# --- cell 6 ---
from rdkit import Chem, rdBase
print("RDKit", rdBase.rdkitVersion)
print(Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1O")))   # should print: Oc1ccccc1

# --- cell 7 ---
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ======================================================================
# ## 3. Imports, paths and input files

# --- cell 9 ---
import copy, hashlib, json, math, os, random, re, sys, tempfile, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger, RDConfig
from rdkit.Chem import AllChem, Descriptors, QED, rdDetermineBonds, rdFMCS, rdMolDescriptors
# Note: Please run `!pip install gemmi` before running this cell to avoid ModuleNotFoundError in meeko
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate

RDLogger.DisableLog("rdApp.*")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# Synthetic-accessibility score (RDKit Contrib)
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    try:
        import sascorer
    except ImportError:
        from rdkit.Contrib.SA_Score import sascorer
except Exception as e:
    sascorer = None
    print("WARNING: sascorer not available, SA scores will be NaN:", e)

PROJECT_ROOT = Path.cwd().resolve()
TAG = "evaluation" if DOCK_MODE == "vina" else "evaluation_MOCK"
OUT_DIR    = PROJECT_ROOT / ("outputs" if DOCK_MODE == "vina" else "outputs_MOCK")
EVAL_DIR   = PROJECT_ROOT / "Outputs-git" / TAG
MODEL_DIR  = PROJECT_ROOT / "Outputs-git" / ("models" if DOCK_MODE == "vina" else "models_MOCK")
CONFIG_DIR = PROJECT_ROOT / "Configuration and documentation"
for d in (OUT_DIR, EVAL_DIR, MODEL_DIR, CONFIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- locate the input files (unzip AIIMS_Docking.zip automatically if it is present) ----
def find_data_dir():
    for z in list(PROJECT_ROOT.glob("*AIIMS_Docking*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(PROJECT_ROOT)
    cands = [PROJECT_ROOT, PROJECT_ROOT / "AIIMS_Docking", Path("/content"), Path("/content/AIIMS_Docking"),
             Path("/content/drive/MyDrive/AIIMS_Docking")]
    cands += [p.parent for p in PROJECT_ROOT.rglob("3PP0.pdb")]
    for d in cands:
        if (d / "3PP0.pdb").exists() and any(d.glob("1XKK*.pdb")):
            return d
    return None

DATA_DIR = find_data_dir()
if DATA_DIR is None:
    print("\n=== IMPORTANT ===\nCould not find 3PP0.pdb and 1XKK*.pdb. Please upload AIIMS_Docking.zip to the files section on the left and re-run this cell.\n")
    RAW_PDB = {"EGFR": None, "HER2": None}
    TAK285_SDF = None
else:
    RAW_PDB = {"EGFR": next(DATA_DIR.glob("1XKK*.pdb")), "HER2": DATA_DIR / "3PP0.pdb"}
    TAK285_SDF = next(iter(DATA_DIR.glob("[Tt]ak-285*.sdf")), None)
    print("Data folder:", DATA_DIR); print("Raw PDBs   :", {k: v.name for k, v in RAW_PDB.items()}); print("TAK-285 SDF:", TAK285_SDF)

# ======================================================================
# ## 4. Targets and ligands (TAK-285 = HER2 ligand)
#
# * **EGFR** – PDB 1XKK, co-crystal ligand `FMM` = lapatinib.
# * **HER2** – PDB 3PP0, co-crystal ligand `03Q` = **TAK-285**.
#
# The SMILES of TAK-285 below is verified against your `Tak-285..sdf` (same molecule, C26H25ClF3N5O3).
# Note: in 3PP0 only **34 of TAK-285's 38 heavy atoms are modelled** (the hydroxy-amide tail is partly disordered), so redocking RMSD is computed over the atoms present in the crystal ligand (see section 6).

# --- cell 11 ---
TAK285_SMILES = "CC(C)(O)CC(=O)NCCn1ccc2ncnc(Nc3ccc(Oc4cccc(C(F)(F)F)c4)c(Cl)c3)c21"
LAPATINIB_SMILES = "CS(=O)(=O)CCNCc1ccc(o1)-c1ccc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2c1"

# sanity check against the supplied SDF
if TAK285_SDF is not None:
    _m = Chem.RemoveHs(Chem.MolFromMolFile(str(TAK285_SDF), removeHs=False))
    assert Chem.MolToSmiles(_m) == Chem.CanonSmiles(TAK285_SMILES), "TAK-285 SMILES does not match the SDF!"
    print("TAK-285 SMILES verified against", TAK285_SDF.name, "->", rdMolDescriptors.CalcMolFormula(_m))

TARGETS = {
    "EGFR": {"pdb": RAW_PDB["EGFR"], "clean": "EGFR_1XKK_clean.pdb", "ligand_resname": "FMM", "ligand_smiles": LAPATINIB_SMILES},
    "HER2": {"pdb": RAW_PDB["HER2"], "clean": "HER2_3PP0_clean.pdb", "ligand_resname": "03Q", "ligand_smiles": TAK285_SMILES},
}

REF_DRUGS = {
    "gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "lapatinib": LAPATINIB_SMILES,
    "erlotinib": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "TAK-285":   TAK285_SMILES,
}
# Small molecules that should NOT dock like kinase inhibitors (sanity check for the scoring set-up)
NEGATIVE_CONTROLS = {
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
}

C = SimpleNamespace(
    VINA_BIN=os.environ.get("VINA_BIN") or shutil.which("vina") or "vina",
    RECEPTOR_DIR=OUT_DIR / "receptors",
    BOX_PADDING=8.0, BOX_MIN_SIDE=18.0, BOX_MAX_SIDE=30.0,
    DOCK={"exhaustiveness": SCREEN_EXH, "num_modes": 5, "energy_range": 3, "cpu": 0, "seed": SEED, "timeout_s": 900},
    EXHAUSTIVENESS_LEVELS=[1, 2, 4] if not QUICK_TEST else [1, 2],      # reduced study (the full 4-32 study costs ~1 h of CPU)
    EXHAUSTIVENESS_SEEDS=[11, 22],
    RMSD_SUCCESS_A=2.0,
    MAX_TORSIONS=20, SCORE_GOOD=-11.0, SCORE_BAD=-7.0,
)
BOXES_JSON = CONFIG_DIR / "docking_boxes.json"

# run several Vina jobs in parallel (1 core each) when more than one CPU is available
N_WORKERS = 1 if DOCK_MODE == "mock" else max(1, os.cpu_count() or 1)
if N_WORKERS > 1:
    C.DOCK["cpu"] = 1
print("Parallel docking workers:", N_WORKERS)

# ======================================================================
# ## 5. Receptor preparation and docking boxes

# --- cell 13 ---
def make_clean_pdb(raw: Path, out: Path, drop=("HOH", "PO4", "SO4", "GOL", "EDO")):
    """Remove waters / buffer ions and CONECT records; keep altloc ' ' and 'A' only."""
    keep = []
    for line in Path(raw).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            if line[16] not in (" ", "A"):
                continue
            if line[17:20].strip() in drop:
                continue
            if line[16] == "A":
                line = line[:16] + " " + line[17:]
            keep.append(line)
        elif line.startswith(("TER", "END")):
            keep.append(line)
    Path(out).write_text("\n".join(keep) + "\n")


def split_complex(clean_pdb: Path, resname: str, protein_out: Path, ligand_out: Path):
    """Protein-only ATOM records of the ligand's chain + first copy of the ligand. Returns ligand heavy-atom coords."""
    lines = Path(clean_pdb).read_text().splitlines()
    lig_lines, chain, resid = [], None, None
    for ln in lines:
        if ln.startswith("HETATM") and ln[17:20].strip() == resname:
            key = (ln[21], ln[22:27])
            if chain is None:
                chain, resid = ln[21], ln[22:27]
            if key == (chain, resid):
                lig_lines.append(ln)
    if not lig_lines:
        raise ValueError(f"Ligand {resname} not found as HETATM in {clean_pdb}")
    prot_lines = [ln for ln in lines if ln.startswith("ATOM") and ln[21] == chain]
    if not prot_lines:
        prot_lines = [ln for ln in lines if ln.startswith("ATOM")]
    protein_out.write_text("\n".join(prot_lines) + "\nEND\n")
    ligand_out.write_text("\n".join(lig_lines) + "\nEND\n")
    coords = []
    for ln in lig_lines:
        element = (ln[76:78].strip() or ln[12:14].strip()).upper()
        if element.startswith("H"):
            continue
        coords.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return np.array(coords)


def compute_box(coords: np.ndarray):
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    center, extent = (lo + hi) / 2.0, hi - lo
    size = np.clip(extent + C.BOX_PADDING, C.BOX_MIN_SIDE, C.BOX_MAX_SIDE)
    box = {"center_x": round(float(center[0]), 3), "center_y": round(float(center[1]), 3), "center_z": round(float(center[2]), 3),
           "size_x": round(float(size[0]), 3), "size_y": round(float(size[1]), 3), "size_z": round(float(size[2]), 3)}
    if np.any(extent + 2.0 > size):
        print("  WARNING: crystal ligand extent is close to / larger than the capped box side; redocking may be clipped.")
    return box


def prepare_receptors():
    """Receptor PDBQTs (ligand + waters removed), crystal-ligand PDBs, and docking_boxes.json."""
    C.RECEPTOR_DIR.mkdir(parents=True, exist_ok=True)
    boxes = {}
    for target, cfg in TARGETS.items():
        if cfg["pdb"] is None:
            print(f"[{target}] Missing PDB input. Cannot prepare receptor yet.")
            continue
        clean = PROJECT_ROOT / cfg["clean"]
        make_clean_pdb(cfg["pdb"], clean)
        prot = C.RECEPTOR_DIR / f"{target}_protein.pdb"
        lig = C.RECEPTOR_DIR / f"{target}_crystal_ligand.pdb"
        coords = split_complex(clean, cfg["ligand_resname"], prot, lig)
        boxes[target] = compute_box(coords)
        pdbqt = C.RECEPTOR_DIR / f"{target}_receptor.pdbqt"
        if shutil.which("obabel"):
            subprocess.run(["obabel", str(prot), "-O", str(pdbqt), "-xr", "-p", "7.4"], check=True, capture_output=True)
        elif DOCK_MODE == "vina":
            raise RuntimeError("obabel is required to build receptor PDBQT files.")
        print(f"[{target}] {cfg['ligand_resname']}: {len(coords)} crystal-ligand heavy atoms. Box: {boxes[target]}")
    if boxes:
        BOXES_JSON.write_text(json.dumps(boxes, indent=2))
        print("Saved docking boxes ->", BOXES_JSON)
    return boxes

_BOXES = None
def load_boxes():
    global _BOXES
    if _BOXES is None:
        if not BOXES_JSON.exists():
            raise FileNotFoundError(f"{BOXES_JSON} not found - run prepare_receptors() first.")
        _BOXES = json.loads(BOXES_JSON.read_text())
    return _BOXES

BOXES = prepare_receptors()
_BOXES = BOXES

# ======================================================================
# ## 6. Docking engine (with a persistent cache)

# --- cell 15 ---
def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def smiles_to_pdbqt(smiles: str, seed: int = 42):
    smi = standardize_smiles(smiles)
    if smi is None:
        return None
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3(); params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    for setup in MoleculePreparation().prepare(mol):
        pdbqt, ok, _ = PDBQTWriterLegacy.write_string(setup)
        if ok:
            return pdbqt
    return None


def _mock_result(smi, target, exh, seed):
    """Deterministic fake score - ONLY for smoke-testing the notebook."""
    mol = Chem.MolFromSmiles(smi)
    ha, ar = mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumAromaticRings(mol)
    u = int(hashlib.md5(f"{smi}|{target}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    s = -3.0 - 0.17 * min(ha, 38) - 0.45 * min(ar, 4) - 1.2 * u
    return {"scores": [s], "best": s, "pdbqt": None, "runtime_s": 0.0}


def run_vina(smiles: str, target: str, exhaustiveness=None, seed=None, boxes=None):
    """Dock one SMILES. Returns dict(scores, best, pdbqt, runtime_s) or None on failure."""
    boxes = boxes or load_boxes()
    box = boxes.get(target)
    if not box:
        return None
    d = dict(C.DOCK)
    if exhaustiveness is not None:
        d["exhaustiveness"] = int(exhaustiveness)
    if seed is not None:
        d["seed"] = int(seed)
    if DOCK_MODE == "mock":
        smi = standardize_smiles(smiles)
        return _mock_result(smi, target, d["exhaustiveness"], d["seed"]) if smi else None

    pdbqt = smiles_to_pdbqt(smiles, seed=d["seed"])
    if not pdbqt:
        return None
    receptor = C.RECEPTOR_DIR / f"{target}_receptor.pdbqt"
    with tempfile.TemporaryDirectory() as td:
        lig_p, out_p = Path(td) / "lig.pdbqt", Path(td) / "out.pdbqt"
        lig_p.write_text(pdbqt)
        cmd = [C.VINA_BIN, "--receptor", str(receptor), "--ligand", str(lig_p), "--out", str(out_p),
               "--center_x", str(box["center_x"]), "--center_y", str(box["center_y"]), "--center_z", str(box["center_z"]),
               "--size_x", str(box["size_x"]), "--size_y", str(box["size_y"]), "--size_z", str(box["size_z"]),
               "--exhaustiveness", str(d["exhaustiveness"]), "--num_modes", str(d["num_modes"]),
               "--energy_range", str(d["energy_range"]), "--cpu", str(d["cpu"]), "--seed", str(d["seed"])]
        t0 = time.time()
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=d["timeout_s"])
        except subprocess.TimeoutExpired:
            return None
        runtime = time.time() - t0
        if res.returncode != 0 or not out_p.exists():
            return None
        text = out_p.read_text()
    scores = [float(l.split()[3]) for l in text.splitlines() if l.startswith("REMARK VINA RESULT:")]
    if not scores:
        return None
    return {"scores": scores, "best": scores[0], "pdbqt": text, "runtime_s": runtime}


# ---- persistent cache of best docking scores: (canonical SMILES, target, exhaustiveness, seed) -> score ----
CACHE_FILE = OUT_DIR / f"dock_cache_{DOCK_MODE}.json"
# failed dockings (None) are NOT kept: a stale None in an old cache silently turns every later lookup into "no score"
DOCK_CACHE = {k: v for k, v in (json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}).items() if v is not None}
DOCK_STATS = []     # (n_jobs, wall_seconds, exhaustiveness) for every dock_many call that ran Vina

def _key(smi, target, exh, seed):
    return f"{smi}|{target}|{exh}|{seed}"

def dock_many(smiles_list, targets, exhaustiveness=None):
    """Best Vina score for every (smiles, target); uses the cache and runs misses in parallel.
    Returns {(smiles, target): score_or_None} keyed by the *input* SMILES."""
    exh = int(exhaustiveness or C.DOCK["exhaustiveness"]); seed = C.DOCK["seed"]
    canon = {s: standardize_smiles(s) for s in set(smiles_list)}
    jobs = sorted({(canon[s], t) for s in canon if canon[s] for t in targets if _key(canon[s], t, exh, seed) not in DOCK_CACHE})
    def work(job):
        smi, t = job
        r = run_vina(smi, t, exhaustiveness=exh, seed=seed)
        return job, (r["best"] if r else None)
    if jobs:
        t_start = time.time()
        if N_WORKERS > 1:
            with ThreadPoolExecutor(N_WORKERS) as ex:
                results = list(ex.map(work, jobs))
        else:
            results = [work(j) for j in jobs]
        DOCK_STATS.append((len(jobs), time.time() - t_start, exh))
        for (smi, t), val in results:
            if val is not None:
                DOCK_CACHE[_key(smi, t, exh, seed)] = val
        CACHE_FILE.write_text(json.dumps(DOCK_CACHE))
    return {(s, t): (DOCK_CACHE.get(_key(canon[s], t, exh, seed)) if canon[s] else None)
            for s in canon for t in targets}

# ======================================================================
# ## 7. Redocking (validation task 1)
#
# Success = heavy-atom RMSD of the top-scoring pose to the crystal ligand ≤ 2.0 Å, **no re-alignment** (pose stays in the receptor frame).
#
# The RMSD is computed on the atoms that exist in *both* the crystal ligand and the docked molecule (maximum common substructure, symmetry-aware, bond orders ignored).
# Bonds of the crystal ligand are rebuilt from its coordinates, so no bond-order template is needed and the partially-modelled TAK-285 in 3PP0 works.

# --- cell 17 ---
def _crystal_mol(target):
    """Crystal ligand heavy atoms as an RDKit mol (connectivity from coordinates, element symbols from the PDB)."""
    lig_pdb = C.RECEPTOR_DIR / f"{target}_crystal_ligand.pdb"
    m = Chem.MolFromPDBFile(str(lig_pdb), removeHs=True, sanitize=False, proximityBonding=False)
    rdDetermineBonds.DetermineConnectivity(m)
    m.UpdatePropertyCache(strict=False); Chem.FastFindRings(m)
    return m


def pose_rmsd(ref_mol, pdbqt_text, max_ref_matches=20, max_pose_matches=5000):
    """In-place heavy-atom RMSD of the top Vina pose vs the crystal ligand over their common atoms.
    Returns (rmsd, n_matched_atoms, n_ref_atoms)."""
    pm = PDBQTMolecule(pdbqt_text, skip_typing=True)
    pose = Chem.RemoveHs(RDKitMolCreate.from_pdbqt_mol(pm)[0], sanitize=False)   # conformer 0 = best-scoring pose
    pose.UpdatePropertyCache(strict=False); Chem.FastFindRings(pose)
    mcs = rdFMCS.FindMCS([ref_mol, pose], atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareAny, ringMatchesRingOnly=False, timeout=30)
    q = Chem.MolFromSmarts(mcs.smartsString)
    ref_m = ref_mol.GetSubstructMatches(q, uniquify=False, maxMatches=max_ref_matches)
    pose_m = pose.GetSubstructMatches(q, uniquify=False, maxMatches=max_pose_matches)
    if not ref_m or not pose_m:
        return float("nan"), 0, ref_mol.GetNumAtoms()
    rc, pc = ref_mol.GetConformer().GetPositions(), pose.GetConformer(0).GetPositions()
    R = np.array([rc[list(m)] for m in ref_m])            # (K, k, 3)
    P = np.array([pc[list(m)] for m in pose_m])           # (M, k, 3)
    d2 = ((R[:, None] - P[None]) ** 2).sum(-1).mean(-1)   # (K, M)
    return float(np.sqrt(d2.min())), q.GetNumAtoms(), ref_mol.GetNumAtoms()


def redock_target(target, exhaustiveness=None, seed=None, boxes=None):
    cfg = TARGETS[target]
    res = run_vina(cfg["ligand_smiles"], target, exhaustiveness=exhaustiveness, seed=seed, boxes=boxes)
    if res is None:
        return None
    if DOCK_MODE == "mock":
        rmsd, nm, nr = float("nan"), 0, 0
    else:
        rmsd, nm, nr = pose_rmsd(_crystal_mol(target), res["pdbqt"])
    return {"target": target, "ligand": cfg["ligand_resname"],
            "exhaustiveness": exhaustiveness or C.DOCK["exhaustiveness"], "seed": seed or C.DOCK["seed"],
            "best_score": res["best"], "rmsd_A": round(rmsd, 3), "atoms_compared": nm, "crystal_heavy_atoms": nr,
            "success_le_2A": bool(rmsd <= C.RMSD_SUCCESS_A), "runtime_s": round(res["runtime_s"], 1)}


def run_redocking():
    names = list(TARGETS)
    print(f"Redocking {names} at exhaustiveness {REDOCK_EXH} (in parallel) ...")
    with ThreadPoolExecutor(max(1, min(len(names), N_WORKERS))) as pool:
        res = list(pool.map(lambda t: redock_target(t, exhaustiveness=REDOCK_EXH, boxes=BOXES), names))
    rows = [r for r in res if r]
    for r in rows:
        print(f"  {r['target']} ({r['ligand']}): score {r['best_score']:.2f} kcal/mol, RMSD {r['rmsd_A']} A over "
              f"{r['atoms_compared']}/{r['crystal_heavy_atoms']} atoms, {r['runtime_s']} s")
    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "redocking_results.csv", index=False)
    return df

redock_df = run_redocking()
print(f"[elapsed {elapsed_min():.1f} min]")
redock_df

# ======================================================================
# ## 8. Known-inhibitor benchmark (validation task 2)
# Known inhibitors (gefitinib, lapatinib, erlotinib, **TAK-285**) and three negative controls are docked to both targets. Inhibitors should score clearly better (more negative) than the controls.

# --- cell 19 ---
def run_benchmark():
    sets = [("known_inhibitor", REF_DRUGS), ("negative_control", NEGATIVE_CONTROLS)]
    allm = {n: s for _, m in sets for n, s in m.items()}
    scores = dock_many(list(allm.values()), list(TARGETS), SCREEN_EXH)
    rows = []
    for role, mols in sets:
        for name, smi in mols.items():
            row = {"name": name, "role": role, "smiles": smi}
            for t in TARGETS:
                row[f"dock_{t}"] = scores[(smi, t)]
            rows.append(row)
            print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in row.items() if k.startswith("dock_")))
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c.startswith("dock_")]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if len(cols) == 2:
        df["dock_mean"] = df[cols].mean(axis=1)
    for col in cols:
        df[f"rank_{col[5:]}"] = df[col].rank(method="min")
    df.to_csv(EVAL_DIR / "benchmark_results.csv", index=False)
    return df

benchmark_df = run_benchmark()
print(f"[elapsed {elapsed_min():.1f} min]")
benchmark_df

# ======================================================================
# ## 9. Exhaustiveness study (validation task 3)
# The EGFR crystal ligand is redocked at several exhaustiveness values × several seeds. The recommended value is the smallest one whose score SD ≤ 0.15, mean within 0.20 kcal/mol of the highest level, and mean RMSD ≤ 2 Å.

# --- cell 21 ---
def run_exhaustiveness():
    rows = []
    jobs = [(ex, seed) for ex in sorted(C.EXHAUSTIVENESS_LEVELS, reverse=True) for seed in C.EXHAUSTIVENESS_SEEDS]
    with ThreadPoolExecutor(max(1, N_WORKERS)) as pool:
        res = list(pool.map(lambda j: redock_target("EGFR", exhaustiveness=j[0], seed=j[1], boxes=BOXES), jobs))
    for r in res:
        if r:
            rows.append(r)
            print(f"  ex={r['exhaustiveness']:>2} seed={r['seed']}: {r['best_score']:.2f} kcal/mol, RMSD {r['rmsd_A']} A, {r['runtime_s']} s")
    df = pd.DataFrame(rows).sort_values(["exhaustiveness", "seed"]).reset_index(drop=True)
    df.to_csv(EVAL_DIR / "exhaustiveness_results.csv", index=False)
    summ = (df.groupby("exhaustiveness")
              .agg(score_mean=("best_score", "mean"), score_sd=("best_score", "std"),
                   rmsd_mean=("rmsd_A", "mean"), rmsd_sd=("rmsd_A", "std"),
                   runtime_mean_s=("runtime_s", "mean")).reset_index())
    top_mean = summ.iloc[-1]["score_mean"]
    use_rmsd = summ["rmsd_mean"].notna().all()          # (RMSD is NaN in mock mode)
    chosen, found = int(summ.iloc[-1]["exhaustiveness"]), False
    for _, s in summ.iterrows():
        ok = s["score_sd"] <= 0.15 and abs(s["score_mean"] - top_mean) <= 0.20
        if use_rmsd:
            ok = ok and s["rmsd_mean"] <= C.RMSD_SUCCESS_A
        if ok:
            chosen, found = int(s["exhaustiveness"]), True
            break
    rec = {"recommended_exhaustiveness": chosen, "criteria_met": found,
           "note": "" if found else "No tested level met the criteria; the highest level was returned. Test higher "
                                    "exhaustiveness or review the box/receptor before using it.",
           "rule": "smallest value with score SD <= 0.15, mean within 0.20 kcal/mol of the highest level, and mean "
                   "redocking RMSD <= 2.0 A (falls back to the highest level tested)",
           "levels_tested": C.EXHAUSTIVENESS_LEVELS, "seeds": C.EXHAUSTIVENESS_SEEDS,
           "vina_settings": C.DOCK, "summary": summ.round(3).to_dict(orient="records")}
    (EVAL_DIR / "exhaustiveness_recommended.json").write_text(json.dumps(rec, indent=2))
    print("Recommended exhaustiveness:", chosen, "" if found else "(criteria NOT met)")
    return summ, rec

exh_summary, exh_rec = run_exhaustiveness()
print(f"[elapsed {elapsed_min():.1f} min]")
exh_summary

# ======================================================================
# ## 10. Reward function
# `reward = (docking term)^0.6 × QED^0.4`, where each docking score is mapped linearly to 0–1 (−7 kcal/mol → 0, −11 kcal/mol → 1).
#
# * **dual** – geometric mean of the EGFR and HER2 terms (a molecule must dock well to *both*).
# * **EGFR** / **HER2** – single-target baselines, used to compare against the dual-target agent.
#
# Molecules with more than `MAX_TORSIONS` rotatable bonds get reward 0 and are not docked.

# --- cell 23 ---
def score_to_reward(score):
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return 0.0
    return float(np.clip((C.SCORE_BAD - score) / (C.SCORE_BAD - C.SCORE_GOOD), 0.0, 1.0))


def compute_reward(smiles, dock_egfr=None, dock_her2=None, mode="dual"):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or rdMolDescriptors.CalcNumRotatableBonds(mol) > C.MAX_TORSIONS:
        return 0.0
    qed = QED.qed(mol)
    re_, rh_ = score_to_reward(dock_egfr), score_to_reward(dock_her2)
    dock_term = {"dual": math.sqrt(re_ * rh_), "EGFR": re_, "HER2": rh_}[mode]
    return float((dock_term ** 0.6) * (qed ** 0.4))


def dock_and_reward(smiles, exhaustiveness=None, mode="dual"):
    """Real (not hard-coded) reward for one molecule."""
    targets = list(TARGETS) if mode == "dual" else [mode]
    sc = dock_many([smiles], targets, exhaustiveness)
    de, dh = sc.get((smiles, "EGFR")), sc.get((smiles, "HER2"))
    return {"smiles": smiles, "dock_EGFR": de, "dock_HER2": dh, "reward": compute_reward(smiles, de, dh, mode)}

# reference point: what do the known inhibitors score?
for n, s in REF_DRUGS.items():
    print(f"{n:10s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in dock_and_reward(s).items() if k != "smiles"})

# ======================================================================
# ## 11. SMILES tokenizer, vocabulary and generative model
# A GRU language model over SMILES tokens (REINVENT-style). It is **pre-trained** on a seed set of kinase inhibitors (EGFR/HER2/other TKIs, including TAK-285 and lapatinib) with randomised-SMILES augmentation, and then fine-tuned by RL.
#
# > The built-in seed set is small (~30 molecules). For publication-quality results set `PRETRAIN_CSV` to a larger set (e.g. ChEMBL EGFR/HER2 actives). Novelty is measured against whichever set is used.

# --- cell 25 ---
_TOKEN_RE = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")
def tokenize(smiles): return _TOKEN_RE.findall(smiles)

SEED_SMILES = [
    # EGFR / HER2 quinazolines, quinolines and related
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",                                  # gefitinib
    LAPATINIB_SMILES, TAK285_SMILES,
    "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",                                        # erlotinib
    "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1",                   # afatinib
    "CCOc1cc2ncc(C#N)c(Nc3ccc(OCc4ccccn4)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C",             # neratinib
    "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1",                  # osimertinib
    "C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1",                           # canertinib
    "COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1",                                   # vandetanib
    "C#Cc1cccc(Nc2ncnc3cc4c(cc23)OCCOCCOCCO4)c1",                                      # icotinib
    "COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC",                                               # PD153035
    "COc1cc2ncnc(Nc3cccc(Cl)c3)c2cc1OC",                                               # AG1478
    "CCOc1cc2ncc(C#N)c(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C",                       # pelitinib
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1",                          # dacomitinib
    "COc1cc(Nc2c(C#N)cnc3cc(OCCCN4CCN(C)CC4)c(OC)cc23)c(Cl)cc1Cl",                     # bosutinib
    # other tyrosine / protein kinase inhibitors
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",                     # imatinib
    "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",                       # sorafenib
    "CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\\C(=O)Nc3ccc(F)cc32)c1C",                        # sunitinib
    "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",                         # dasatinib
    "Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1",          # nilotinib
    "Cc1ccc(Nc2nccc(N(C)c3ccc4c(C)n(C)nc4c3)n2)cc1S(N)(=O)=O",                         # pazopanib
    "COc1cc2nccc(Oc3ccc(NC(=O)NC4CC4)c(Cl)c3)c2cc1C(N)=O",                             # lenvatinib
    "CNC(=O)c1ccccc1Sc1ccc2c(/C=C/c3ccccn3)n[nH]c2c1",                                 # axitinib
    "C=CC(=O)N1CCC[C@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",                    # ibrutinib
    "C[C@@H]1CCN(C(=O)CC#N)C[C@@H]1N(C)c1ncnc2[nH]ccc12",                              # tofacitinib
    "N#CCC(C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1",                                        # ruxolitinib
    "C[C@@H](Oc1cc(-c2cnn(C3CCNCC3)c2)cnc1N)c1c(Cl)ccc(F)c1Cl",                        # crizotinib
    "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1F",                 # vemurafenib
]

if PRETRAIN_CSV:
    _extra = pd.read_csv(PRETRAIN_CSV)["smiles"].dropna().tolist()
    SEED_SMILES = SEED_SMILES + _extra
TRAIN_CANON = sorted({Chem.MolToSmiles(m) for m in (Chem.MolFromSmiles(s) for s in SEED_SMILES) if m is not None})
print(f"Pre-training / novelty reference set: {len(TRAIN_CANON)} valid unique molecules "
      f"({len(SEED_SMILES) - len(TRAIN_CANON)} dropped as invalid/duplicate)")
TRAIN_SET = set(TRAIN_CANON)


def random_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, doRandom=True, canonical=False) if m else None


class Vocab:
    PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
    def __init__(self, smiles_list, n_aug=30):
        toks = set()
        for s in smiles_list:
            toks.update(tokenize(s))
            for _ in range(n_aug):
                r = random_smiles(s)
                if r: toks.update(tokenize(r))
        self.itos = [self.PAD, self.BOS, self.EOS] + sorted(toks)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
    def __len__(self): return len(self.itos)
    def encode(self, smiles):
        toks = tokenize(smiles)
        if any(t not in self.stoi for t in toks):
            return None
        return [self.stoi[self.BOS]] + [self.stoi[t] for t in toks] + [self.stoi[self.EOS]]
    def decode(self, ids):
        out = []
        for i in ids:
            t = self.itos[int(i)]
            if t == self.EOS: break
            if t not in (self.PAD, self.BOS): out.append(t)
        return "".join(out)

vocab = Vocab(TRAIN_CANON)
PAD_ID, BOS_ID, EOS_ID = vocab.stoi[vocab.PAD], vocab.stoi[vocab.BOS], vocab.stoi[vocab.EOS]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # T4 GPU on Colab (checked in section 0)
MAX_LEN = 160
print("Vocabulary size:", len(vocab), "| device:", DEVICE)


class SmilesRNN(nn.Module):
    def __init__(self, V, emb=128, hid=384, layers=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=PAD_ID)
        self.gru = nn.GRU(emb, hid, layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hid, V)

    def forward(self, x, h=None):
        o, h = self.gru(self.emb(x), h)
        return self.out(o), h

    def log_prob(self, seqs):
        """Sum of token log-probs per sequence. seqs: (B,T) incl. BOS ... EOS then PAD."""
        logits, _ = self(seqs[:, :-1])
        lp = torch.log_softmax(logits, -1).gather(-1, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
        return (lp * (seqs[:, 1:] != PAD_ID)).sum(1)

    @torch.no_grad()
    def sample(self, n, max_len=MAX_LEN):
        x = torch.full((n, 1), BOS_ID, dtype=torch.long, device=DEVICE)
        h, seqs, done = None, [x], torch.zeros(n, dtype=torch.bool, device=DEVICE)
        for _ in range(max_len):
            logits, h = self(x, h)
            x = torch.multinomial(torch.softmax(logits[:, -1], -1), 1)
            x = x.masked_fill(done.unsqueeze(1), PAD_ID)
            seqs.append(x)
            done |= (x.squeeze(1) == EOS_ID)
            if done.all(): break
        return torch.cat(seqs, 1)


def decode_batch(seqs):
    return [vocab.decode(s[1:]) for s in seqs.cpu().tolist()]

def pad(batch):
    L = max(len(b) for b in batch)
    return torch.tensor([b + [PAD_ID] * (L - len(b)) for b in batch], dtype=torch.long, device=DEVICE)

# ======================================================================
# ## 12. Pre-train the prior

# --- cell 27 ---
def pretrain(epochs=PRETRAIN_EPOCHS, bs=128, lr=2e-3, aug=10):   # larger batch to use the T4
    model = SmilesRNN(len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(1, epochs + 1):
        # every epoch = `aug` freshly randomised SMILES of every training molecule
        enc = [vocab.encode(random_smiles(s) or s) or vocab.encode(s) for s in TRAIN_CANON for _ in range(aug)]
        enc = [e for e in enc if e and len(e) <= MAX_LEN]
        random.shuffle(enc); losses = []
        for i in range(0, len(enc), bs):
            seqs = pad(enc[i:i + bs])
            loss = -model.log_prob(seqs).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(loss.item())
        hist.append(float(np.mean(losses)))
        if ep % max(1, epochs // 10) == 0 or ep == 1:
            print(f"epoch {ep:>4}/{epochs}  NLL/seq {hist[-1]:.2f}")
    return model, hist

prior, pretrain_hist = pretrain()
prior.eval()
torch.save(prior.state_dict(), MODEL_DIR / "prior.pt")
_s = decode_batch(prior.sample(200))
print("Prior sanity check - validity: %.0f%%" % (100 * np.mean([Chem.MolFromSmiles(s) is not None for s in _s])))
print(f"[elapsed {elapsed_min():.1f} min]")

# ======================================================================
# ## 13. Reinforcement learning (REINVENT-style)
# Each step: sample a batch from the agent → dock the valid, unique molecules → reward → update the agent with the augmented-likelihood loss
# `(log P_prior(x) + σ·reward(x) − log P_agent(x))²`. The prior stays frozen so the agent keeps generating drug-like, SMILES-valid molecules.
# Each mode runs until its share of the remaining time budget (`RL_SHARE`) is used, so the number of steps is chosen by the clock, not fixed. Run for three modes: **dual** (EGFR+HER2), **EGFR**-only and **HER2**-only.

# --- cell 29 ---
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# --- cell 30 ---
def _mean_or_nan(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def rl_train(mode, deadline, steps=RL_MAX_STEPS, n_sample=RL_SAMPLE, k_dock=RL_DOCK_K, n_neg=RL_NEG,
             sigma=RL_SIGMA, lr=RL_LR, exh=RL_EXH):
    """REINVENT-style RL that stops when `deadline` (epoch seconds) would be exceeded by the next step.
    Every step: sample n_sample SMILES (cheap, GPU) -> dock at most k_dock valid, unique, not-too-flexible molecules
    -> update the agent on the docked molecules plus a few invalid samples (reward 0)."""
    targets = list(TARGETS) if mode == "dual" else [mode]
    agent = copy.deepcopy(prior); agent.train()
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    log, step_times = [], []
    for step in range(1, steps + 1):
        t_step = time.time()
        if step > 1 and t_step + float(np.mean(step_times[-3:])) > deadline:
            print(f"[{mode}] time share used up after {step - 1} steps"); break
        seqs = agent.sample(n_sample)
        canon = [standardize_smiles(s) if s else None for s in decode_batch(seqs)]
        pick, seen = [], set()
        for i, c in enumerate(canon):
            if c is None or c in seen:
                continue
            seen.add(c)
            if len(pick) < k_dock and rdMolDescriptors.CalcNumRotatableBonds(Chem.MolFromSmiles(c)) <= C.MAX_TORSIONS:
                pick.append(i)
        to_dock = [canon[i] for i in pick]
        sc = dock_many(to_dock, targets, exh)
        neg = [i for i, c in enumerate(canon) if c is None][:n_neg]
        idx = pick + neg
        if not idx:
            step_times.append(time.time() - t_step); continue
        rewards = [compute_reward(canon[i], sc.get((canon[i], "EGFR")), sc.get((canon[i], "HER2")), mode) for i in pick] + [0.0] * len(neg)
        R = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        sel = seqs[torch.tensor(idx, dtype=torch.long, device=DEVICE)]
        with torch.no_grad():
            prior_lp = prior.log_prob(sel)
        agent_lp = agent.log_prob(sel)
        loss = ((prior_lp + sigma * R - agent_lp) ** 2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 1.0); opt.step()
        valid = [c for c in canon if c]
        row = {"mode": mode, "step": step, "loss": loss.item(), "n_docked": len(pick),
               "mean_reward": _mean_or_nan(rewards[:len(pick)]),
               "validity": len(valid) / n_sample, "unique_frac": len(set(valid)) / max(1, len(valid)),
               **{f"mean_dock_{t}": _mean_or_nan([sc.get((c, t)) for c in to_dock]) for t in TARGETS},
               "step_seconds": time.time() - t_step, "elapsed_min": elapsed_min()}
        log.append(row); step_times.append(row["step_seconds"])
        print(f"[{mode}] step {step:>2}  docked {len(pick)}  reward {row['mean_reward']:.3f}  valid {row['validity']:.2f}  "
              f"dockEGFR {row['mean_dock_EGFR']:.2f}  dockHER2 {row['mean_dock_HER2']:.2f}  "
              f"{row['step_seconds']:.0f}s  [elapsed {row['elapsed_min']:.1f} min]")
    agent.eval()
    torch.save(agent.state_dict(), MODEL_DIR / f"agent_{mode}.pt")
    return agent, pd.DataFrame(log)


RL_END = BUDGET_END - (EVAL_RESERVE_MIN + REPORT_RESERVE_MIN) * 60
print(f"RL window: {max(0, RL_END - time.time()) / 60:.1f} min (elapsed so far {elapsed_min():.1f} min)")
agents, rl_logs, todo = {}, [], ["dual", "EGFR", "HER2"]
for mode in ["dual", "EGFR", "HER2"]:
    now = time.time()
    deadline = now + max(0.0, RL_END - now) * RL_SHARE[mode] / sum(RL_SHARE[m] for m in todo)
    print(f"\n=== RL {mode}: time share {(deadline - now) / 60:.1f} min ===")
    agents[mode], lg = rl_train(mode, deadline)
    todo.remove(mode); rl_logs.append(lg)
    torch.cuda.empty_cache()
rl_log = pd.concat(rl_logs, ignore_index=True)
rl_log.to_csv(EVAL_DIR / "rl_training_log.csv", index=False)
print(f"[elapsed {elapsed_min():.1f} min]  RL steps done:", rl_log.groupby("mode")["step"].max().to_dict())

# --- cell 31 ---
import matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for mode, g in rl_log.groupby("mode"):
    ax[0].plot(g["step"], g["mean_reward"], label=mode)
    ax[1].plot(g["step"], g["mean_dock_EGFR"], label=mode)
    ax[2].plot(g["step"], g["mean_dock_HER2"], label=mode)
for a, t in zip(ax, ["Mean reward", "Mean EGFR docking score (kcal/mol)", "Mean HER2 docking score (kcal/mol)"]):
    a.set_title(t); a.set_xlabel("RL step"); a.grid(alpha=.3)
ax[0].legend(); plt.tight_layout(); plt.savefig(EVAL_DIR / "rl_training_curves.png", dpi=150); plt.show()

# ======================================================================
# ## 14. Evaluate every model
# For each model (prior, dual, EGFR-only, HER2-only) we sample `N_EVAL` molecules and compute validity, uniqueness, novelty (vs. the pre-training set), QED, SA score and docking scores on **both** targets.
# All per-molecule data goes into one master table, from which every result file is derived.

# --- cell 33 ---
import warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')          # hides SMILES parse errors
warnings.filterwarnings("ignore", message=".*RNN module weights.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- cell 34 ---
def run_benchmark():
    sets = [("known_inhibitor", REF_DRUGS), ("negative_control", NEGATIVE_CONTROLS)]
    allm = {n: s for _, m in sets for n, s in m.items()}
    scores = dock_many(list(allm.values()), list(TARGETS), SCREEN_EXH)
    rows = []
    for role, mols in sets:
        for name, smi in mols.items():
            row = {"name": name, "role": role, "smiles": smi}
            for t in TARGETS:
                row[f"dock_{t}"] = scores[(smi, t)]
            rows.append(row)
            print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in row.items() if k.startswith("dock_")))
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c.startswith("dock_")]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Mean docking score across all targets (works for 1, 2, 3+ targets).
    # skipna=False -> NaN if any target failed, so a failed docking isn't hidden.
    df["dock_mean"] = df[cols].mean(axis=1, skipna=False)

    for col in cols:
        df[f"rank_{col[5:]}"] = df[col].rank(method="min")
    df["rank_mean"] = df["dock_mean"].rank(method="min")

    df.to_csv(EVAL_DIR / "benchmark_results.csv", index=False)
    return df

benchmark_df = run_benchmark()
print(f"[elapsed {elapsed_min():.1f} min]")
benchmark_df

# --- cell 35 ---
import numpy as np, pandas as pd, shutil, os
from pathlib import Path

N_SAMPLED = 300   # <-- set to the number of molecules you actually sampled per model

print("=== 1. Redocking (success = RMSD <= 2 A) ===")
print(redock_df[["target", "best_score", "rmsd_A", "success_le_2A"]].to_string(index=False))
if not redock_df["success_le_2A"].all():
    print("WARNING: some targets failed redocking; state this as a limitation.")

print("\n=== 2. Benchmark: known inhibitors vs negative controls ===")
dock_cols = [c for c in benchmark_df.columns if c.startswith("dock_") and c != "dock_mean"]
print("NaN per column:\n", benchmark_df[dock_cols].isna().sum().to_string())
print("Any positive scores:", bool((benchmark_df[dock_cols] > 0).any().any()))
print(benchmark_df.groupby("role")[dock_cols + ["dock_mean"]].mean().round(2).to_string())
g = benchmark_df.groupby("role")["dock_mean"].mean()
if "known_inhibitor" in g and "negative_control" in g:
    print("Inhibitors score better than controls:", g["known_inhibitor"] < g["negative_control"])

print("\n=== 3. Generation metrics as rates ===")
gen = pd.DataFrame({
    "valid":  [250, 269, 270, 275],
    "unique": [162, 190, 177, 175],
    "novel":  [136, 166, 150, 150]},
    index=["prior", "agent_EGFR", "agent_HER2", "agent_dual"])
rates = pd.DataFrame({
    "validity_%": gen["valid"] / N_SAMPLED * 100,
    "uniqueness_% (of valid)": gen["unique"] / gen["valid"] * 100,
    "novelty_% (of unique)": gen["novel"] / gen["unique"] * 100}).round(1)
print(rates.to_string())

print("\n=== 4. Save outputs to Drive ===")
dest = Path("/content/drive/MyDrive/docking_results")
dest.mkdir(parents=True, exist_ok=True)
for f in Path(EVAL_DIR).glob("*"):
    if f.is_file():
        shutil.copy(f, dest / f.name)
rates.to_csv(dest / "generation_rates.csv")
print("Saved to", dest, "->", sorted(os.listdir(dest)))

# --- cell 36 ---
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def sa_score(m):
    return float(sascorer.calculateScore(m)) if sascorer else float("nan")


def eval_dock_plan(n_models=4):
    """How many molecules per model can be docked (on both targets) in the time that is left, from the measured Vina speed."""
    rows = [(n, w) for n, w, e in DOCK_STATS if e == EVAL_EXH and n >= 4]
    eff = (sum(w for _, w in rows) / sum(n for n, _ in rows)) if rows else 40.0      # effective wall-seconds per docking
    left = BUDGET_END - REPORT_RESERVE_MIN * 60 - time.time()
    n = int(left / (eff * len(TARGETS) * n_models))
    n = max(MIN_EVAL_DOCK, min(MAX_EVAL_DOCK, n))
    print(f"Evaluation docking: ~{eff:.0f} s per docking (wall, all workers), {left / 60:.1f} min left -> {n} molecules/model x {len(TARGETS)} targets")
    return n


def evaluate_model(name, model, n_dock, deadline, n=N_EVAL):
    raw = sum([decode_batch(model.sample(min(256, n - i))) for i in range(0, n, 256)], [])
    rows, seen = [], set()
    for i, s in enumerate(raw):
        mol = Chem.MolFromSmiles(s) if s else None
        valid = mol is not None and mol.GetNumHeavyAtoms() > 0
        r = {"condition": name, "idx": i, "smiles": s, "valid": valid, "canonical": None, "unique": False, "novel": False}
        if valid:
            can = Chem.MolToSmiles(mol)
            r.update(canonical=can, unique=can not in seen, novel=(can not in seen) and (can not in TRAIN_SET),
                     qed=QED.qed(mol), sa=sa_score(mol), mw=Descriptors.MolWt(mol),
                     rot_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol), heavy_atoms=mol.GetNumHeavyAtoms())
            seen.add(can)
        rows.append(r)
    df = pd.DataFrame(rows)
    for col in ["qed", "sa", "mw", "rot_bonds", "heavy_atoms"]:      # keep the schema even if nothing was valid
        if col not in df: df[col] = np.nan
    # docking is only affordable for a random subset of the unique, dockable molecules (same subset rule for every model)
    elig = df[df["unique"] & (df["rot_bonds"] <= C.MAX_TORSIONS)]["canonical"].tolist()
    random.Random(SEED).shuffle(elig)
    elig, sc, chunk = elig[:n_dock], {}, max(2, N_WORKERS)
    for i in range(0, len(elig), chunk):
        if i > 0 and time.time() > deadline:
            print(f"  [{name}] time share used up: docked {i}/{len(elig)} molecules"); elig = elig[:i]; break
        sc.update(dock_many(elig[i:i + chunk], list(TARGETS), EVAL_EXH))
    docked = set(elig)
    for t in TARGETS:
        df[f"dock_{t}"] = [sc.get((c, t)) if (u and c in docked) else np.nan for c, u in zip(df["canonical"], df["unique"])]
        df[f"dock_{t}"] = pd.to_numeric(df[f"dock_{t}"], errors="coerce")
    df["dock_mean"] = df[[f"dock_{t}" for t in TARGETS]].mean(axis=1)
    need = {"dual": ["dock_EGFR", "dock_HER2"], "EGFR": ["dock_EGFR"], "HER2": ["dock_HER2"]}
    for mode, cols in need.items():
        df[f"reward_{mode}"] = [compute_reward(c, e, h, mode) if (u and c and all(pd.notna(x) for x in vals)) else np.nan
                                for c, e, h, u, vals in zip(df["canonical"], df["dock_EGFR"], df["dock_HER2"], df["unique"],
                                                            zip(*[df[k] for k in cols]))]
    df["dual_hit"] = (df["dock_EGFR"] <= HIT_THRESHOLD) & (df["dock_HER2"] <= HIT_THRESHOLD)
    return df


N_DOCK_EVAL = eval_dock_plan()
EVAL_END = BUDGET_END - REPORT_RESERVE_MIN * 60
models = {"prior": prior, "agent_dual": agents["dual"], "agent_EGFR": agents["EGFR"], "agent_HER2": agents["HER2"]}
frames = []
for k, (name, m) in enumerate(models.items()):
    now = time.time()
    frames.append(evaluate_model(name, m, N_DOCK_EVAL, now + max(0.0, EVAL_END - now) / (len(models) - k)))
    print(f"  evaluated {name}  [elapsed {elapsed_min():.1f} min]")
master = pd.concat(frames, ignore_index=True)
print(master.groupby("condition")[["valid", "unique", "novel"]].sum())

# --- cell 37 ---
master.to_csv(EVAL_DIR / "master_eval.csv", index=False)
import shutil; shutil.copy(EVAL_DIR / "master_eval.csv", "/content/drive/MyDrive/docking_results/")

# ======================================================================
# ## 15. Metrics and result files

# --- cell 39 ---
def internal_diversity(smiles, cap=200):
    smiles = smiles[:cap]
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in smiles]
    if len(fps) < 2: return float("nan")
    sims = [1 - np.mean(DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1:])) for i, fp in enumerate(fps[:-1])]
    return float(np.mean(sims))

def condition_metrics(g):
    n, nv = len(g), int(g["valid"].sum())
    u = g[g["unique"]]
    docked = u.dropna(subset=["dock_EGFR", "dock_HER2"])
    return {
        "n_sampled": n, "n_valid": nv, "n_unique": len(u), "n_novel": int(u["novel"].sum()), "n_docked": len(docked),
        "validity_pct": 100 * nv / n if n else np.nan,
        "uniqueness_pct": 100 * len(u) / nv if nv else np.nan,
        "novelty_pct": 100 * u["novel"].sum() / len(u) if len(u) else np.nan,
        "internal_diversity": internal_diversity(u["canonical"].tolist()) if len(u) else np.nan,
        "qed_mean": u["qed"].mean(), "qed_median": u["qed"].median(),
        "sa_mean": u["sa"].mean(), "sa_median": u["sa"].median(),
        "dock_EGFR_mean": docked["dock_EGFR"].mean(), "dock_EGFR_best": docked["dock_EGFR"].min(),
        "dock_HER2_mean": docked["dock_HER2"].mean(), "dock_HER2_best": docked["dock_HER2"].min(),
        "dock_mean_both": docked["dock_mean"].mean(),
        "dock_gap_mean": (docked["dock_EGFR"] - docked["dock_HER2"]).abs().mean(),
        "dual_hit_pct": 100 * docked["dual_hit"].sum() / len(docked) if len(docked) else np.nan,
        "reward_dual_mean": u["reward_dual"].mean(), "reward_dual_max": u["reward_dual"].max(),
    }

order = ["prior", "agent_dual", "agent_EGFR", "agent_HER2"]
summary = pd.DataFrame({c: condition_metrics(master[master["condition"] == c]) for c in order}).T
summary.index.name = "condition"; summary = summary.reset_index()

# ---- 1. evaluation_results.csv : every sampled molecule with every property
master.to_csv(EVAL_DIR / "evaluation_results.csv", index=False)

# ---- 2-4. validity / uniqueness / novelty
summary[["condition", "n_sampled", "n_valid", "validity_pct"]].assign(n_invalid=lambda d: d.n_sampled - d.n_valid) \
    .to_csv(EVAL_DIR / "molecular_validity.csv", index=False)
summary[["condition", "n_valid", "n_unique", "uniqueness_pct"]].to_csv(EVAL_DIR / "uniqueness_results.csv", index=False)
summary[["condition", "n_unique", "n_novel", "novelty_pct"]].to_csv(EVAL_DIR / "novelty_results.csv", index=False)

# ---- 5-6. QED / SA (per unique valid molecule)
uv = master[master["unique"]]
uv[["condition", "canonical", "qed"]].rename(columns={"canonical": "smiles"}).to_csv(EVAL_DIR / "qed_results.csv", index=False)
uv[["condition", "canonical", "sa"]].rename(columns={"canonical": "smiles"}).to_csv(EVAL_DIR / "sa_results.csv", index=False)

# ---- 7. dual-target results (agent trained on EGFR+HER2)
cols = ["condition", "canonical", "qed", "sa", "mw", "rot_bonds", "dock_EGFR", "dock_HER2", "dock_mean", "reward_dual", "dual_hit", "novel"]
dual = uv[uv["condition"] == "agent_dual"][cols].rename(columns={"canonical": "smiles"}).sort_values("reward_dual", ascending=False)
dual.to_csv(EVAL_DIR / "dual_target_results.csv", index=False)

# ---- 8. single-target results (EGFR-only and HER2-only agents, scored on both targets)
single = uv[uv["condition"].isin(["agent_EGFR", "agent_HER2"])].copy()
single["optimized_target"] = single["condition"].str.replace("agent_", "")
single["reward_single"] = np.where(single["optimized_target"] == "EGFR", single["reward_EGFR"], single["reward_HER2"])
single = single[["condition", "optimized_target", "canonical", "qed", "sa", "mw", "rot_bonds", "dock_EGFR", "dock_HER2",
                 "dock_mean", "reward_single", "dual_hit", "novel"]].rename(columns={"canonical": "smiles"})
single.sort_values(["optimized_target", "reward_single"], ascending=[True, False]).to_csv(EVAL_DIR / "single_target_results.csv", index=False)

# ---- 9. comparison: dual vs single-target agents vs prior vs known inhibitors
cmp_cols = ["validity_pct", "uniqueness_pct", "novelty_pct", "qed_mean", "sa_mean", "dock_EGFR_mean", "dock_HER2_mean",
            "dock_mean_both", "dock_gap_mean", "dual_hit_pct", "reward_dual_mean"]
cmp_df = summary[["condition"] + cmp_cols].copy()
ref = benchmark_df[benchmark_df["role"] == "known_inhibitor"]
cmp_df.loc[len(cmp_df)] = {"condition": "known_inhibitors", "dock_EGFR_mean": ref["dock_EGFR"].mean(),
                           "dock_HER2_mean": ref["dock_HER2"].mean(), "dock_mean_both": ref["dock_mean"].mean(),
                           "dock_gap_mean": (ref["dock_EGFR"] - ref["dock_HER2"]).abs().mean()}
base = cmp_df[cmp_df["condition"] == "prior"].iloc[0]
for c in cmp_cols:
    cmp_df[f"delta_vs_prior_{c}"] = cmp_df[c] - base[c]
dual_row = cmp_df[cmp_df["condition"] == "agent_dual"].iloc[0]
for c in ["dock_EGFR_mean", "dock_HER2_mean", "dock_mean_both", "dock_gap_mean", "dual_hit_pct", "reward_dual_mean"]:
    cmp_df[f"delta_vs_dual_{c}"] = cmp_df[c] - dual_row[c]
cmp_df.to_csv(EVAL_DIR / "comparison_results.csv", index=False)

# ---- 10. metrics_summary.csv
summary["dock_mode"] = DOCK_MODE; summary["eval_exhaustiveness"] = EVAL_EXH
_steps = {f"agent_{m}": int(g["step"].max()) for m, g in rl_log.groupby("mode")}
summary["rl_steps"] = summary["condition"].map(_steps)
summary["rl_sample_per_step"], summary["rl_dock_per_step"] = RL_SAMPLE, RL_DOCK_K
summary["n_dock_per_model_target"] = N_DOCK_EVAL
summary["n_pretrain_molecules"] = len(TRAIN_CANON)
summary.to_csv(EVAL_DIR / "metrics_summary.csv", index=False)

summary.round(3)

# --- cell 40 ---
cmp_df.round(3)

# --- cell 41 ---
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
x = np.arange(len(order)); w = 0.38
ax[0].bar(x - w/2, summary["dock_EGFR_mean"], w, label="EGFR"); ax[0].bar(x + w/2, summary["dock_HER2_mean"], w, label="HER2")
ax[0].set_xticks(x); ax[0].set_xticklabels(order, rotation=20); ax[0].set_title("Mean docking score (more negative = better)"); ax[0].legend()
ax[1].bar(order, summary["dual_hit_pct"]); ax[1].set_title(f"Dual hits (both <= {HIT_THRESHOLD} kcal/mol), %"); ax[1].tick_params(axis="x", rotation=20)
ax[2].bar(order, summary["reward_dual_mean"]); ax[2].set_title("Mean dual reward"); ax[2].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(EVAL_DIR / "model_comparison.png", dpi=150); plt.show()

# ======================================================================
# ## 15b. Person 5 evaluation (validity / uniqueness / novelty / QED / SA, GA vs RL, dual vs single-target)
#
# Runs **after** all docking is finished and uses only CSVs/DataFrames already in memory, so it takes well under a minute and does not change the 90-minute budget (the 5-minute end reserve covers it).
# It recomputes every molecular metric **from the raw SMILES** and writes the `Person5/` folder (`evaluation/`, `figures/`, `scripts/`) that is included in the results ZIP.
#
# * **Dual vs single-target** comparison uses the notebook's own agents (`agent_dual`, `agent_EGFR`, `agent_HER2`).
# * **GA vs RL** is added only if a GA molecules CSV is present (`GA_CSV` below, or a file such as `*ga*result*.csv` next to the input data); otherwise it is skipped and the report says so.
# * Novelty here is measured against the pre-training set **plus** any ChEMBL EGFR/HER2 train/test CSVs found, plus TAK-285, so it can be lower than the `novelty_pct` in `metrics_summary.csv` (which uses the pre-training set only).
# * A failure in this section never blocks the report and ZIP cells (it is wrapped in `try/except`).

# --- cell 43 ---
# --- writes scripts/integrated_evaluation.py (self-contained, no external file needed) ---
P5 = PROJECT_ROOT / "Person5"
for _d in ("evaluation", "figures", "scripts", "input", "notebooks"):
    (P5 / _d).mkdir(parents=True, exist_ok=True)
(P5 / "scripts" / "integrated_evaluation.py").write_text(r'''#!/usr/bin/env python3
"""
Person 5 - integrated evaluation (GA vs RL, dual vs single-target).

Everything is RECOMPUTED from the raw SMILES in the supplied files (validity, uniqueness, novelty, QED, SA) -
nothing is copied from earlier result tables.

Usage (from the Person5 folder):
    python scripts/integrated_evaluation.py --input input --ga input/ga_results.csv --rl input/evaluation_results.csv
Run  python scripts/integrated_evaluation.py -h  for all options.
"""
import argparse, json, os, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, fisher_exact
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import QED, Descriptors, AllChem, RDConfig
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
HIT_THRESHOLD = -9.0                       # kcal/mol: a "dual hit" has BOTH docking scores <= this
RL_DUAL, RL_SINGLE, RL_PRIOR = "agent_dual", ("agent_EGFR", "agent_HER2"), "prior"
GA_LABEL = "GA"
NOTES = []                                  # caveats collected while running; written to the summary report


def note(msg):
    print("  NOTE:", msg); NOTES.append(msg)


# ------------------------------------------------------------------ helpers
def _load_sascorer():
    try:
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        return sascorer
    except Exception:
        pass
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return sascorer
    except Exception:
        return None


SASCORER = _load_sascorer()


def canon(smiles):
    """Canonical, non-isomeric SMILES (stereo ignored so stereo-variants count as one molecule). None if invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    m = Chem.MolFromSmiles(smiles.strip())
    if m is None or m.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=False)


def find_file(input_dir, names, required=False):
    """Case-insensitive recursive search for the first file whose name equals / matches one of `names`."""
    input_dir = Path(input_dir)
    allf = [p for p in input_dir.rglob("*") if p.is_file()]
    for n in names:
        for p in sorted(allf):
            if p.name.lower() == n.lower() or ("*" in n and p.match(n.lower()) or ("*" in n and p.match(n))):
                return p
    if required:
        raise FileNotFoundError(f"none of {names} found under {input_dir}")
    return None


def _find_col(df, kind):
    cols = list(df.columns)
    if kind == "smiles":
        for c in cols:
            if c.lower() in ("smiles", "canonical", "canonical_smiles", "std_smiles", "smi", "molecule"):
                return c
        return next((c for c in cols if "smiles" in c.lower()), None)
    target = kind.lower()                                     # "egfr" / "her2"
    skip = re.compile(r"reward|pred|sd_|std|prob|rank", re.I)
    pat = re.compile(rf"(dock|vina|score|affin|energy).*{target}|{target}.*(dock|vina|score|affin|energy)", re.I)
    for c in cols:
        if pat.search(c) and not skip.search(c):
            return c
    return next((c for c in cols if c.lower() == target), None)


def load_table(path, default_label):
    """Read a CSV and map it to: condition, smiles, dock_EGFR, dock_HER2."""
    df = pd.read_csv(path)
    sc = _find_col(df, "smiles")
    if sc is None:
        raise ValueError(f"{path}: no SMILES column found (columns: {list(df.columns)})")
    out = pd.DataFrame({"smiles": df[sc]})
    out["condition"] = df["condition"] if "condition" in df.columns else default_label
    for t in ("EGFR", "HER2"):
        c = _find_col(df, t)
        out[f"dock_{t}"] = pd.to_numeric(df[c], errors="coerce") if c else np.nan
        if c is None:
            note(f"{Path(path).name}: no {t} docking column found - docking metrics for {t} are empty.")
    out["source_file"] = Path(path).name
    print(f"  loaded {Path(path).name}: {len(out)} rows, conditions {sorted(out['condition'].unique())}, smiles col '{sc}'")
    return out


def load_reference(input_dir, extra=(), extra_smiles=()):
    """Reference set for novelty = all ChEMBL train/test molecules + TAK-285 (+ any extra CSV/SDF)."""
    ref, srcs = set(), []
    for t in ("EGFR", "HER2"):
        for s in ("train", "test"):
            p = find_file(input_dir, [f"{t}_{s}.csv"])
            if p:
                d = pd.read_csv(p); c = _find_col(d, "smiles")
                if c:
                    ref |= {canon(x) for x in d[c]}; srcs.append(p.name)
    for p in extra:
        p = Path(p)
        if p.suffix.lower() == ".csv":
            d = pd.read_csv(p); c = _find_col(d, "smiles"); ref |= {canon(x) for x in d[c]}
        elif p.suffix.lower() == ".sdf":
            ref |= {canon(Chem.MolToSmiles(m)) for m in Chem.SDMolSupplier(str(p)) if m}
        srcs.append(p.name)
    ref |= {canon(x) for x in extra_smiles}; srcs += ["in-memory set"] if len(extra_smiles) else []
    ref |= {canon("CC(C)(O)CC(=O)NCCn1ccc2ncnc(Nc3ccc(Oc4cccc(C(F)(F)F)c4)c(Cl)c3)c21")}       # TAK-285
    ref.discard(None)
    if not srcs:
        note("no ChEMBL train/test CSVs found - novelty is measured against TAK-285 only (not meaningful).")
    print(f"  novelty reference: {len(ref)} molecules from {srcs or 'TAK-285 only'}")
    return ref


# ------------------------------------------------------------------ per-molecule annotation
def annotate(df, ref):
    """Adds canonical, valid, unique (within condition), novel, qed, sa, mw, heavy_atoms."""
    df = df.copy()
    df["canonical"] = df["smiles"].map(canon)
    df["valid"] = df["canonical"].notna()
    df["unique"] = df["valid"] & ~df.duplicated(["condition", "canonical"], keep="first")
    df["novel"] = df["unique"] & ~df["canonical"].isin(ref)
    q, s, mw, ha = [], [], [], []
    for can, v in zip(df["canonical"], df["valid"]):
        if not v:
            q.append(np.nan); s.append(np.nan); mw.append(np.nan); ha.append(np.nan); continue
        m = Chem.MolFromSmiles(can)
        q.append(QED.qed(m)); mw.append(Descriptors.MolWt(m)); ha.append(m.GetNumHeavyAtoms())
        s.append(float(SASCORER.calculateScore(m)) if SASCORER else np.nan)
    df["qed"], df["sa"], df["mw"], df["heavy_atoms"] = q, s, mw, ha
    if SASCORER is None:
        note("RDKit SA_Score module not available - SA values are NaN (it ships in RDKit Contrib; check the RDKit install).")
    df["dock_mean"] = df[["dock_EGFR", "dock_HER2"]].mean(axis=1)
    df["dual_hit"] = (df["dock_EGFR"] <= HIT_THRESHOLD) & (df["dock_HER2"] <= HIT_THRESHOLD)
    return df


def internal_diversity(smiles, cap=300):
    smiles = list(smiles)[:cap]
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in smiles]
    if len(fps) < 2:
        return np.nan
    return float(np.mean([1 - np.mean(DataStructs.BulkTanimotoSimilarity(f, fps[i + 1:])) for i, f in enumerate(fps[:-1])]))


def metrics_table(df):
    rows = []
    for cond, g in df.groupby("condition", sort=False):
        n, nv = len(g), int(g["valid"].sum()); u = g[g["unique"]]
        dk = u.dropna(subset=["dock_EGFR", "dock_HER2"])
        r = dict(condition=cond, n_rows=n, n_valid=nv, validity_pct=100 * nv / n if n else np.nan,
                 n_unique=len(u), uniqueness_pct=100 * len(u) / nv if nv else np.nan,
                 n_novel=int(u["novel"].sum()), novelty_pct=100 * u["novel"].sum() / len(u) if len(u) else np.nan,
                 internal_diversity=internal_diversity(u["canonical"]) if len(u) else np.nan,
                 qed_mean=u["qed"].mean(), qed_median=u["qed"].median(), qed_sd=u["qed"].std(),
                 sa_mean=u["sa"].mean(), sa_median=u["sa"].median(), sa_sd=u["sa"].std(),
                 mw_mean=u["mw"].mean(), n_docked=len(dk),
                 dock_EGFR_mean=dk["dock_EGFR"].mean(), dock_EGFR_best=dk["dock_EGFR"].min(),
                 dock_HER2_mean=dk["dock_HER2"].mean(), dock_HER2_best=dk["dock_HER2"].min(),
                 dock_mean_both=dk["dock_mean"].mean(),
                 dock_gap_mean=(dk["dock_EGFR"] - dk["dock_HER2"]).abs().mean(),
                 dual_hit_n=int(dk["dual_hit"].sum()), dual_hit_pct=100 * dk["dual_hit"].mean() if len(dk) else np.nan)
        r["validity_note"] = ("all rows valid: the file may contain only valid molecules, so validity is not informative"
                              if nv == n and n > 0 else "")
        rows.append(r)
    m = pd.DataFrame(rows)
    for _, r in m.iterrows():
        if r["validity_note"]:
            note(f"[{r['condition']}] {r['validity_note']}.")
        if 0 < r["n_docked"] < 10:
            note(f"[{r['condition']}] only {int(r['n_docked'])} docked molecules - docking comparisons have very low power.")
    return m


# ------------------------------------------------------------------ statistics
def _boot_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed); a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def compare_pair(df, a, b, labels=None):
    """Continuous properties (Mann-Whitney U + bootstrap CI of the mean difference) and rates (Fisher exact)."""
    A, B = df[df.condition == a], df[df.condition == b]; rows = []

    def pool(g, col, kind):
        u = g[g["unique"]]
        return (u[col].dropna() if kind == "unique" else u.dropna(subset=["dock_EGFR", "dock_HER2"])[col])

    for col, name, kind in [("qed", "QED", "unique"), ("sa", "SA score (lower = easier)", "unique"), ("mw", "Molecular weight", "unique"),
                            ("dock_EGFR", "Docking EGFR (kcal/mol)", "docked"), ("dock_HER2", "Docking HER2 (kcal/mol)", "docked"),
                            ("dock_mean", "Docking mean of both (kcal/mol)", "docked")]:
        pa, pb = pool(A, col, kind), pool(B, col, kind)
        row = dict(metric=name, kind="continuous", mean_a=np.nan, mean_b=np.nan, n_a=len(pa), n_b=len(pb),
                   diff_a_minus_b=np.nan, ci_low=np.nan, ci_high=np.nan, p_value=np.nan, note="")
        if len(pa) >= 2 and len(pb) >= 2:
            lo, hi = _boot_diff(pa, pb)
            row.update(mean_a=pa.mean(), mean_b=pb.mean(), diff_a_minus_b=pa.mean() - pb.mean(), ci_low=lo, ci_high=hi,
                       p_value=mannwhitneyu(pa, pb, alternative="two-sided").pvalue, note="Mann-Whitney U; CI = bootstrap of mean difference")
        else:
            row["note"] = "too few values"
        rows.append(row)

    def rate(name, ka, na, kb, nb):
        rows.append(dict(metric=name, kind="rate (%)", mean_a=100 * ka / na if na else np.nan, mean_b=100 * kb / nb if nb else np.nan,
                         n_a=na, n_b=nb, diff_a_minus_b=(100 * ka / na - 100 * kb / nb) if na and nb else np.nan, ci_low=np.nan, ci_high=np.nan,
                         p_value=fisher_exact([[ka, na - ka], [kb, nb - kb]])[1] if na and nb else np.nan, note="Fisher exact"))

    for name, fn in [("Validity", lambda g: (g["valid"].sum(), len(g))),
                     ("Uniqueness", lambda g: (g["unique"].sum(), g["valid"].sum())),
                     ("Novelty", lambda g: (g["novel"].sum(), g["unique"].sum()))]:
        (ka, na), (kb, nb) = fn(A), fn(B); rate(name, int(ka), int(na), int(kb), int(nb))
    da = A[A["unique"]].dropna(subset=["dock_EGFR", "dock_HER2"]); db = B[B["unique"]].dropna(subset=["dock_EGFR", "dock_HER2"])
    rate(f"Dual hits (both <= {HIT_THRESHOLD})", int(da["dual_hit"].sum()), len(da), int(db["dual_hit"].sum()), len(db))
    la, lb = labels or (a, b)
    out = pd.DataFrame(rows).rename(columns={"mean_a": la, "mean_b": lb, "n_a": f"n_{la}", "n_b": f"n_{lb}",
                                              "diff_a_minus_b": f"diff_{la}_minus_{lb}"})
    for c in (f"n_{la}", f"n_{lb}"):
        out[c] = out[c].astype(int)
    return out


# ------------------------------------------------------------------ figures
def _bar(metrics, col, title, ylabel, fname, figdir, order):
    d = metrics.set_index("condition").reindex(order).dropna(subset=[col])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(d) + 2, 4)); bars = ax.bar(d.index, d[col])
    for b, v in zip(bars, d[col]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=20); ax.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(figdir / fname, dpi=150); plt.close(fig)


def _dist(df, col, title, fname, figdir, order):
    data = [(c, df[(df.condition == c) & df["unique"]][col].dropna().values) for c in order]
    data = [(c, v) for c, v in data if len(v) > 1]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(data) + 2, 4.2))
    ax.boxplot([v for _, v in data], tick_labels=[c for c, _ in data], showmeans=True)
    rng = np.random.default_rng(0)
    for i, (_, v) in enumerate(data, 1):
        ax.scatter(rng.normal(i, .05, len(v)), v, s=8, alpha=.4)
    ax.set_title(title); ax.set_ylabel(col.upper()); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(figdir / fname, dpi=150); plt.close(fig)


def make_molecular_figures(df, metrics, figdir, order):
    figdir.mkdir(exist_ok=True, parents=True)
    _bar(metrics, "validity_pct", "Validity", "% valid of generated", "validity_comparison.png", figdir, order)
    _bar(metrics, "uniqueness_pct", "Uniqueness", "% unique of valid", "uniqueness_comparison.png", figdir, order)
    _bar(metrics, "novelty_pct", "Novelty (vs ChEMBL train/test + TAK-285)", "% novel of unique", "novelty_comparison.png", figdir, order)
    _dist(df, "qed", "QED distribution (unique valid molecules)", "qed_distribution.png", figdir, order)
    _dist(df, "sa", "SA score distribution (lower = easier to make)", "sa_distribution.png", figdir, order)


def _panel(ax, vals, labels, title, fmt="{:.2f}"):
    ax.bar(labels, vals)
    for i, v in enumerate(vals):
        if pd.notna(v):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_title(title, fontsize=10); ax.tick_params(axis="x", rotation=15)


def make_ga_rl_figure(metrics, ga, rl, figdir):
    d = metrics.set_index("condition")
    if ga not in d.index or rl not in d.index:
        return
    panels = [("validity_pct", "Validity %"), ("uniqueness_pct", "Uniqueness %"), ("novelty_pct", "Novelty %"),
              ("qed_mean", "QED mean"), ("sa_mean", "SA mean (lower better)"), ("dock_mean_both", "Docking mean (kcal/mol)"),
              ("dual_hit_pct", f"Dual hits % (<= {HIT_THRESHOLD})")]
    fig, axs = plt.subplots(2, 4, figsize=(16, 7)); axs = axs.ravel()
    for ax, (c, t) in zip(axs, panels):
        _panel(ax, [d.loc[ga, c], d.loc[rl, c]], [ga, rl], t)
    for ax in axs[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{ga} vs {rl}"); fig.tight_layout(); fig.savefig(figdir / "ga_vs_rl_comparison.png", dpi=150); plt.close(fig)


def make_dual_single_figure(metrics, conds, figdir):
    d = metrics.set_index("condition").reindex(conds).dropna(subset=["n_docked"])
    if d.empty or d["n_docked"].max() == 0:
        return
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2)); x = np.arange(len(d)); w = .38
    axs[0].bar(x - w / 2, d["dock_EGFR_mean"], w, label="EGFR"); axs[0].bar(x + w / 2, d["dock_HER2_mean"], w, label="HER2")
    axs[0].set_xticks(x); axs[0].set_xticklabels(d.index, rotation=15); axs[0].legend()
    axs[0].set_title("Mean docking score (more negative = better)")
    _panel(axs[1], d["dual_hit_pct"].values, list(d.index), f"Dual hits % (both <= {HIT_THRESHOLD})")
    _panel(axs[2], d["dock_gap_mean"].values, list(d.index), "Mean |EGFR - HER2| score gap")
    fig.tight_layout(); fig.savefig(figdir / "dual_vs_single_comparison.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ docking validation (existing CSVs -> one table)
def docking_validation_summary(input_dir):
    rows = []
    p = find_file(input_dir, ["redocking_results.csv"])
    if p:
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append(dict(check="Redocking", target=r.get("target"), ligand=r.get("ligand"), value=r.get("rmsd_A"),
                             unit="A (top pose)", criterion="<= 2.0",
                             passed=bool(r.get("success_le_2A")) if "success_le_2A" in d else np.nan,
                             extra=f"best-of-N RMSD {r.get('rmsd_best_of_N_A')}" if "rmsd_best_of_N_A" in d else ""))
    p = find_file(input_dir, ["benchmark_results.csv"])
    if p:
        d = pd.read_csv(p)
        if {"role", "dock_EGFR", "dock_HER2"} <= set(d.columns):
            for t in ("EGFR", "HER2"):
                inh, ctl = d[d.role == "known_inhibitor"][f"dock_{t}"], d[d.role != "known_inhibitor"][f"dock_{t}"]
                if len(inh) and len(ctl):
                    gap = ctl.min() - inh.max()
                    rows.append(dict(check="Known-inhibitor benchmark", target=t, ligand="all", value=round(gap, 3),
                                     unit="kcal/mol (best control - worst inhibitor)", criterion="> 0", passed=bool(gap > 0),
                                     extra=f"worst inhibitor {inh.max():.2f}; best control {ctl.min():.2f}"))
    for name in ("exhaustiveness_recommended.json", "docking_settings.json", "docking_boxes.json"):
        p = find_file(input_dir, [name])
        rows.append(dict(check=f"File: {name}", target="", ligand="", value="present" if p else "MISSING", unit="",
                         criterion="required", passed=bool(p), extra=str(p.relative_to(input_dir)) if p else ""))
    p = find_file(input_dir, ["exhaustiveness_results.csv"])
    if p:
        rows.append(dict(check="File: exhaustiveness_results.csv", target="", ligand="", value="present", unit="", criterion="",
                         passed=True, extra=""))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report
def _md(df, floats=2):
    if df is None or len(df) == 0:
        return "_not available_"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else (f"{v:.{floats}f}" if abs(v) >= 0.01 or v == 0 else f"{v:.2g}"))
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.astype(str).values)


def write_report(path, metrics, ga_rl, dual_single, valid_tbl, ga, rl, settings):
    L = ["# Person 5 - Evaluation results summary", "",
         "_Generated by `scripts/integrated_evaluation.py`. All molecular metrics were recomputed from the raw SMILES; "
         "docking numbers are taken from the supplied docking outputs._", "", "## 1. Settings", ""]
    L += [f"- {k}: {v}" for k, v in settings.items()]
    L += ["", "Definitions: **validity** = RDKit-parsable / rows supplied; **uniqueness** = distinct canonical SMILES (stereo ignored) / valid, "
          "per condition; **novelty** = unique molecules absent from the reference set / unique; QED = RDKit QED; SA = Ertl-Schuffenhauer "
          f"score (1 easy - 10 hard); **dual hit** = both docking scores <= {HIT_THRESHOLD} kcal/mol.", "", "## 2. Molecular metrics (all conditions)", ""]
    show = ["condition", "n_rows", "validity_pct", "n_unique", "uniqueness_pct", "novelty_pct", "internal_diversity", "qed_mean", "sa_mean",
            "n_docked", "dock_EGFR_mean", "dock_HER2_mean", "dual_hit_pct"]
    L += [_md(metrics[[c for c in show if c in metrics]]), "", f"## 3. {ga} vs {rl}", ""]
    L += [_md(ga_rl) if ga_rl is not None else "_GA and/or RL file not supplied._", "", "## 4. Dual-target vs single-target agents", ""]
    L += [_md(dual_single) if dual_single is not None else
          "_single-target agents not found in the RL file (needs a `condition` column with agent_EGFR / agent_HER2)._", "",
          "## 5. Docking validation", "", _md(valid_tbl), "", "## 6. Caveats", ""]
    L += [f"- {n}" for n in dict.fromkeys(NOTES)]
    L += ["- Comparisons between conditions are only fair if all molecules were docked with the same receptor, box and exhaustiveness.",
          "- Vina scores are not binding free energies; differences of a few tenths of kcal/mol are within seed-to-seed noise.",
          "- With few docked molecules per condition, p-values are indicative only."]
    Path(path).write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ main pipeline
def load_all(input_dir, ga_path=None, rl_path=None, extra_ref=(), extra_smiles=(), auto_ga=True):
    """Locate and load the GA / RL molecule files and the novelty reference set."""
    rl_path = Path(rl_path) if rl_path else find_file(input_dir, ["evaluation_results.csv", "rl_all_scored.csv"])
    ga_path = Path(ga_path) if ga_path else (find_file(input_dir, ["*ga*result*.csv", "*ga*eval*.csv", "*ga*scored*.csv", "*ga*.csv"]) if auto_ga else None)
    frames = []
    if rl_path:
        frames.append(load_table(rl_path, "RL"))
    else:
        note("no RL molecule file found (evaluation_results.csv / rl_all_scored.csv).")
    if ga_path:
        t = load_table(ga_path, GA_LABEL); t["condition"] = GA_LABEL; frames.append(t)
    else:
        note("no GA molecule file found - GA vs RL comparison skipped (pass --ga path/to/ga.csv).")
    if not frames:
        raise SystemExit("No input molecule files found.")
    return pd.concat(frames, ignore_index=True), load_reference(input_dir, extra_ref, extra_smiles), rl_path, ga_path


def run(input_dir, ga_path=None, rl_path=None, extra_ref=(), out_root=ROOT, extra_smiles=(), auto_ga=True):
    global NOTES
    NOTES.clear()
    input_dir, out_root = Path(input_dir), Path(out_root)
    ev, fg = out_root / "evaluation", out_root / "figures"
    ev.mkdir(exist_ok=True, parents=True); fg.mkdir(exist_ok=True, parents=True)
    print("1/6 locating inputs + 2/6 novelty reference")
    raw, ref, rl_path, ga_path = load_all(input_dir, ga_path, rl_path, extra_ref, extra_smiles, auto_ga)
    print("3/6 recomputing validity / uniqueness / novelty / QED / SA"); df = annotate(raw, ref)
    metrics = metrics_table(df)
    conds = list(metrics["condition"]); known = [GA_LABEL, RL_PRIOR, "RL", RL_DUAL, *RL_SINGLE]
    order = [c for c in known if c in conds] + [c for c in conds if c not in known]
    metrics = metrics.set_index("condition").loc[order].reset_index()
    metrics.to_csv(ev / "molecular_metrics.csv", index=False)
    df.drop(columns=["smiles"]).rename(columns={"canonical": "smiles"}).to_csv(ev / "molecule_level_annotations.csv", index=False)

    print("4/6 GA vs RL"); rl_name = RL_DUAL if RL_DUAL in conds else ("RL" if "RL" in conds else None); ga_rl = None
    if GA_LABEL in conds and rl_name:
        ga_rl = compare_pair(df, GA_LABEL, rl_name); ga_rl.to_csv(ev / "ga_vs_rl_comparison.csv", index=False)
        make_ga_rl_figure(metrics, GA_LABEL, rl_name, fg)
    print("5/6 dual vs single"); dual_single = None
    singles = [c for c in RL_SINGLE if c in conds]
    if RL_DUAL in conds and singles:
        parts = []
        for s in singles:
            c = compare_pair(df, RL_DUAL, s, labels=("dual", "single")); c.insert(0, "comparison", f"{RL_DUAL} vs {s}"); parts.append(c)
        dual_single = pd.concat(parts, ignore_index=True)
        dual_single.to_csv(ev / "dual_vs_single_target_comparison.csv", index=False)
        make_dual_single_figure(metrics, [RL_PRIOR, RL_DUAL, *singles], fg)
    else:
        note("dual vs single-target comparison skipped: the RL file needs conditions agent_dual plus agent_EGFR and/or agent_HER2 "
             "(the notebook's evaluation_results.csv has them).")
    print("6/6 figures + report"); make_molecular_figures(df, metrics, fg, order)
    valid_tbl = docking_validation_summary(input_dir); valid_tbl.to_csv(ev / "docking_validation_summary.csv", index=False)
    settings = {"RL file": rl_path.name if rl_path else "-", "GA file": ga_path.name if ga_path else "-",
                "reference molecules": len(ref), "SA scorer": "available" if SASCORER else "NOT available",
                "dual-hit threshold (kcal/mol)": HIT_THRESHOLD}
    p = find_file(input_dir, ["docking_settings.json"])
    if p:
        try:
            settings["docking settings"] = json.dumps(json.loads(p.read_text()))[:300]
        except Exception:
            pass
    write_report(ev / "PERSON5_RESULTS_SUMMARY.md", metrics, ga_rl, dual_single, valid_tbl, GA_LABEL, rl_name or "RL", settings)
    print("\nDone. Outputs in", ev, "and", fg)
    return dict(df=df, metrics=metrics, ga_rl=ga_rl, dual_single=dual_single, validation=valid_tbl)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(ROOT / "input"), help="folder with the unzipped results (searched recursively)")
    ap.add_argument("--ga", help="CSV with the GA molecules (needs a SMILES column; docking columns optional)")
    ap.add_argument("--rl", help="CSV with the RL molecules (evaluation_results.csv from the notebook, or rl_all_scored.csv)")
    ap.add_argument("--ref", nargs="*", default=[], help="extra reference files for novelty (.csv with smiles, or .sdf)")
    ap.add_argument("--out", default=str(ROOT), help="output root (evaluation/ and figures/ are created inside)")
    a = ap.parse_args()
    run(a.input, a.ga, a.rl, a.ref, a.out)
''')
print("wrote", P5 / "scripts" / "integrated_evaluation.py")

# --- cell 44 ---
import sys, json, importlib, shutil, traceback
GA_CSV = None        # path to the GA molecules CSV (needs a SMILES column; docking columns optional). None = look for one in the data folder.

try:
    # ---- docking_settings.json + docking_boxes.json go next to the other validation files (they are required deliverables) ----
    _settings = {k: v for k, v in vars(C).items() if k.isupper() or k == "DOCK"}
    _settings.update(dict(dock_mode=DOCK_MODE, screen_exhaustiveness=SCREEN_EXH, redock_exhaustiveness=REDOCK_EXH, seed=SEED,
                          hit_threshold_kcal_mol=HIT_THRESHOLD, n_eval_sampled=N_EVAL, targets=list(TARGETS)))
    (EVAL_DIR / "docking_settings.json").write_text(json.dumps(_settings, indent=2, default=str))
    if Path(BOXES_JSON).exists(): shutil.copy(BOXES_JSON, EVAL_DIR / "docking_boxes.json")

    # ---- optional GA file ----
    if GA_CSV is None:
        _pats = ["*ga*result*.csv", "*ga*eval*.csv", "*ga*scored*.csv", "*genetic*.csv", "ga_*.csv"]
        for _root in (DATA_DIR, PROJECT_ROOT, Path("/content")):
            _hit = next((p for pat in _pats for p in sorted(Path(_root).glob(pat)) if EVAL_DIR not in p.parents), None)
            if _hit: GA_CSV = _hit; break
    print("GA file:", GA_CSV if GA_CSV else "none found -> GA vs RL comparison will be skipped")

    # ---- ChEMBL train/test CSVs (novelty reference), if supplied ----
    _chembl = [p for t in ("EGFR", "HER2") for s in ("train", "test") for p in [DATA_DIR / f"{t}_{s}.csv"] if p.exists()]

    sys.path.insert(0, str(P5 / "scripts"))
    import integrated_evaluation as ie; importlib.reload(ie)
    ie.HIT_THRESHOLD = HIT_THRESHOLD
    p5 = ie.run(EVAL_DIR, ga_path=GA_CSV, rl_path=EVAL_DIR / "evaluation_results.csv", extra_ref=_chembl,
                out_root=P5, extra_smiles=TRAIN_CANON, auto_ga=False)
    display(p5["metrics"][["condition", "n_rows", "validity_pct", "uniqueness_pct", "novelty_pct", "qed_mean", "sa_mean",
                           "n_docked", "dock_EGFR_mean", "dock_HER2_mean", "dual_hit_pct"]].round(2))
    for _f in ("dual_vs_single_comparison", "ga_vs_rl_comparison", "qed_distribution"):
        _p = P5 / "figures" / f"{_f}.png"
        if _p.exists():
            from IPython.display import Image; display(Image(str(_p)))
    print(f"Person 5 section finished [elapsed {elapsed_min():.1f} min]")
except Exception:
    print("Person 5 section failed (the rest of the notebook is unaffected):"); traceback.print_exc()

# ======================================================================
# ## 16. Validation report

# --- cell 46 ---
def _md(df):
    return "_no results_" if df is None or df.empty else df.to_markdown(index=False)

def write_report():
    red = pd.read_csv(EVAL_DIR / "redocking_results.csv"); bench = pd.read_csv(EVAL_DIR / "benchmark_results.csv")
    exh = pd.read_csv(EVAL_DIR / "exhaustiveness_results.csv")
    s = exh.groupby("exhaustiveness").agg(score_mean=("best_score", "mean"), score_sd=("best_score", "std"),
                                          rmsd_mean=("rmsd_A", "mean"), runtime_mean_s=("runtime_s", "mean")).round(3).reset_index()
    out = ["# Validation report update", "",
           f"Docking mode: **{DOCK_MODE}**" + ("  (MOCK - scores are not real!)" if DOCK_MODE == "mock" else ""),
           f"Vina settings used: {json.dumps(C.DOCK)}", "",
           f"**Time-budgeted profile** (target {TIME_BUDGET_MIN} min): screening exhaustiveness {SCREEN_EXH} (benchmark, RL, evaluation), "
           f"redocking exhaustiveness {REDOCK_EXH}, RL steps per mode {rl_log.groupby('mode')['step'].max().to_dict()}, "
           f"{N_DOCK_EVAL} molecules per model docked on both targets in the evaluation.", "",
           "## 1. EGFR (1XKK / lapatinib) and HER2 (3PP0 / TAK-285) redocking", "",
           f"Success criterion: heavy-atom RMSD of the top-scoring pose <= {C.RMSD_SUCCESS_A} A (no re-alignment). "
           "RMSD is computed over the atoms common to the crystal ligand and the docked molecule (see `atoms_compared`); in 3PP0 only 34 of TAK-285's 38 heavy atoms are modelled.", "", _md(red), "",
           "## 2. Known-inhibitor benchmark", "", _md(bench), "",
           "Known inhibitors should score clearly better (more negative) than the negative controls on both targets.", "",
           "## 3. Exhaustiveness", "", _md(s), "",
           f"**Recommended exhaustiveness: {exh_rec['recommended_exhaustiveness']}**" + ("" if exh_rec["criteria_met"] else " (criteria NOT met)"),
           "", exh_rec.get("note", ""), f"Rule: {exh_rec['rule']}.", "",
           "## 4. Generative model evaluation", "",
           "Validity, uniqueness, novelty, QED and SA use all `n_sampled` molecules of each model. Docking-based columns use only the "
           "random subset counted in `n_docked` and are therefore directional, not statistically tight. RL ran for a few steps only.", "",
           _md(summary.round(3)), ""]
    (EVAL_DIR / "validation_report_update.md").write_text("\n".join(out))
    print("Report written ->", EVAL_DIR / "validation_report_update.md")

write_report()
print("\nFiles in", EVAL_DIR); [print("  ", p.name) for p in sorted(EVAL_DIR.iterdir())]

# ======================================================================
# ### Optional: download everything (Colab)

# --- cell 48 ---
# shutil.make_archive("results", "zip", EVAL_DIR); from google.colab import files; files.download("results.zip")

# --- cell 49 ---
print(f"Total notebook time: {elapsed_min():.1f} min (target {TIME_BUDGET_MIN} min)")

import shutil, os
from pathlib import Path
# from google.colab import files

# folders/files to include (missing ones are skipped)
INCLUDE = [EVAL_DIR,                        # Outputs-git/evaluation (redocking, benchmark, TAK-285 results, plots)
           Path("outputs/poses"),           # saved docking poses (.pdbqt)
           Path("outputs/receptors"),       # receptor PDBQTs + crystal ligands
           BOXES_JSON,                    # docking_boxes.json
           PROJECT_ROOT / 'Person5']      # Person 5 evaluation package                      # docking_boxes.json

STAGE = Path("/content/results_bundle")
shutil.rmtree(STAGE, ignore_errors=True); STAGE.mkdir(parents=True)
for p in INCLUDE:
    p = Path(p)
    if p.is_dir():
        shutil.copytree(p, STAGE / p.name, dirs_exist_ok=True)
    elif p.is_file():
        shutil.copy(p, STAGE / p.name)
    else:
        print("skipped (not found):", p)

zip_path = shutil.make_archive("/content/results", "zip", STAGE)
print(f"created {zip_path}  ({os.path.getsize(zip_path)/1024:.0f} KB)")

# keep a copy on Drive so it survives a runtime restart
drive_dir = Path("/content/drive/MyDrive/person4")
if drive_dir.is_dir():
    shutil.copy(zip_path, drive_dir / "results.zip"); print("copy saved to", drive_dir / "results.zip")

if IN_COLAB:
    files.download(zip_path)                    # downloads to your computer

