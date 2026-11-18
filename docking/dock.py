
import json, yaml, subprocess, tempfile, os, re
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy

CFG = yaml.safe_load(open("p4_config.yaml"))
VINA_EXE = r"C:\tools\vina.exe"

def smiles_to_pdbqt(smiles, seed=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed or CFG["seed"]) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    setups = MoleculePreparation().prepare(mol)
    pdbqt, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
    return pdbqt if ok else None

def dock_pdbqt(pdbqt, target, return_pose=False):
    t = CFG["targets"][target]
    box = json.load(open(t["box"]))
    with tempfile.TemporaryDirectory() as d:
        lig, out = os.path.join(d, "lig.pdbqt"), os.path.join(d, "out.pdbqt")
        open(lig, "w").write(pdbqt)
        cx, cy, cz = box["center"]; sx, sy, sz = box["size"]
        cmd = [VINA_EXE, "--receptor", os.path.abspath(t["receptor"]), "--ligand", lig,
               "--out", out,
               "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
               "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
               "--seed", str(CFG["seed"]),
               "--exhaustiveness", str(CFG["vina"]["exhaustiveness"]),
               "--num_modes", str(CFG["vina"]["n_poses"]),
               "--cpu", str(CFG["vina"]["cpu"])]
        subprocess.run(cmd, check=True, capture_output=True)
        txt = open(out).read()
    best = float(re.search(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)", txt).group(1))
    return (best, txt) if return_pose else best

def dock_smiles(smiles, target):
    try:
        pdbqt = smiles_to_pdbqt(smiles)
        return None if pdbqt is None else dock_pdbqt(pdbqt, target)
    except Exception:
        return None
