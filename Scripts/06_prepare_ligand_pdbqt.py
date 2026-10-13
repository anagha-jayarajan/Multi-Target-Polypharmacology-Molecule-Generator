import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from utils import load_config


def smiles_to_pdbqt(smiles, output_pdbqt_path):
    """Generates 3D conformer and converts RDKit mol to PDBQT using Meeko v0.5+ API."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # Add explicit hydrogens and generate 3D embedding
        mol = Chem.AddHs(mol)
        embed_result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if embed_result != 0:
            # Fallback embedding if ETKDGv3 fails
            embed_result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if embed_result != 0:
                return False

        # MMFF Force Field Optimization
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)

        # Meeko v0.5+ PDBQT preparation
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)

        writer = PDBQTWriterLegacy()
        for setup in mol_setups:
            pdbqt_string, is_ok, _ = writer.write_string(setup)
            if is_ok:
                with open(output_pdbqt_path, "w") as f:
                    f.write(pdbqt_string)
                return True
        return False

    except Exception:
        return False


def prepare_ligands_pdbqt():
    cfg = load_config()
    lig_dir = Path(cfg["output_dir"]) / "ligands"
    pdbqt_base_dir = Path(cfg["output_dir"]) / "ligands_pdbqt"
    pdbqt_base_dir.mkdir(parents=True, exist_ok=True)

    for target in cfg["targets"].keys():
        curated_csv = lig_dir / f"{target}_curated_actives.csv"
        if not curated_csv.exists():
            print(f"[SKIP] Missing curated CSV for {target}")
            continue

        target_pdbqt_dir = pdbqt_base_dir / target
        target_pdbqt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Generating 3D PDBQT Conformers for {target} ---")
        df = pd.read_csv(curated_csv)

        success_count = 0
        for idx, row in df.iterrows():
            chembl_id = row["molecule_chembl_id"]
            smiles = row["std_smiles"]
            out_file = target_pdbqt_dir / f"{chembl_id}.pdbqt"

            if smiles_to_pdbqt(smiles, out_file):
                success_count += 1

            if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
                print(f"[{target}] Processed {idx + 1}/{len(df)} compounds ({success_count} successful)...")

        print(f"[SUCCESS] Saved {success_count} PDBQT ligand files to {target_pdbqt_dir}")


if __name__ == "__main__":
    prepare_ligands_pdbqt()