import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import SaltRemover
from sklearn.model_selection import train_test_split
from utils import load_config


def standardize_smiles(smiles, remover):
    """Parses, strips salts, and canonicalizes a SMILES string."""
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Remove common salt counterions
        mol = remover.StripMol(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def curate_and_split_ligands():
    cfg = load_config()
    lig_dir = Path(cfg["output_dir"]) / "ligands"
    max_act = cfg["ligand_curation"]["max_activity_nm"]
    split_ratio = cfg["ligand_curation"]["split_ratio"]

    remover = SaltRemover.SaltRemover()

    for target in cfg["targets"].keys():
        raw_csv = lig_dir / f"{target}_raw_bioactivity.csv"
        if not raw_csv.exists():
            print(f"[SKIP] Missing raw bioactivity CSV for {target}")
            continue

        print(f"\n--- Curating Ligands for {target} ---")
        df = pd.read_csv(raw_csv)
        initial_count = len(df)

        # 1. Standardize SMILES
        df["std_smiles"] = df["canonical_smiles"].apply(lambda s: standardize_smiles(s, remover))
        df = df.dropna(subset=["std_smiles"])
        print(f"[{target}] Standardized {len(df)}/{initial_count} SMILES structures.")

        # 2. Filter active compounds (standard_value <= 1000 nM)
        df_active = df[df["standard_value"] <= max_act].copy()
        print(f"[{target}] Filtered to {len(df_active)} active compounds (<= {max_act} nM).")

        # 3. Deduplicate by canonical SMILES (keep minimum standard_value)
        df_active = df_active.sort_values("standard_value").groupby("std_smiles", as_index=False).first()
        print(f"[{target}] Unique active compounds after deduplication: {len(df_active)}")

        # Save complete curated active dataset
        curated_csv = lig_dir / f"{target}_curated_actives.csv"
        df_active.to_csv(curated_csv, index=False)

        # 4. Perform Train/Test Split
        if len(df_active) >= 10:
            train_df, test_df = train_test_split(df_active, train_size=split_ratio, random_state=42)
            train_csv = lig_dir / f"{target}_train.csv"
            test_csv = lig_dir / f"{target}_test.csv"

            train_df.to_csv(train_csv, index=False)
            test_df.to_csv(test_csv, index=False)
            print(f"[{target}] Saved Train Split: {len(train_df)} | Test Split: {len(test_df)}")


if __name__ == "__main__":
    curate_and_split_ligands()