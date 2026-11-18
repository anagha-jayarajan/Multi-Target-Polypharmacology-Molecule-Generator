
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from meeko import PDBQTMolecule, RDKitMolCreate
from dock import CFG, smiles_to_pdbqt, dock_pdbqt

def redock(target):
    ref = Chem.SDMolSupplier(CFG["targets"][target]["ref_ligand"], removeHs=False)[0]
    ref = Chem.RemoveHs(ref)
    smi = Chem.MolToSmiles(ref)
    pdbqt = smiles_to_pdbqt(smi)
    score, poses = dock_pdbqt(pdbqt, target, return_pose=True)

    pm = PDBQTMolecule(poses, skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pm)
    mol = Chem.RemoveHs(mols[0])
    rmsds = []
    for cid in range(mol.GetNumConformers()):
        # symmetry-aware RMSD with no realignment: both are in the receptor frame
        rmsds.append(rdMolAlign.CalcRMS(mol, ref, prbId=cid))
    return score, rmsds[0], min(rmsds)

if __name__ == "__main__":
    for t in CFG["targets"]:
        s, top, best = redock(t)
        status = "PASS" if top < 2 else "FAIL"
        print(f"{t}: score {s:.2f} kcal/mol, top-pose RMSD {top:.2f} A, "
              f"best-of-poses RMSD {best:.2f} A, {status} (top pose)")
