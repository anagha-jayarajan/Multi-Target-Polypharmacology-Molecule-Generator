from pathlib import Path
from pdbfixer import PDBFixer
from openmm.app import PDBFile, Modeller
from utils import load_config

def prepare_proteins():
    cfg = load_config()
    raw_dir = Path(cfg["output_dir"]) / "raw"
    prot_dir = Path(cfg["output_dir"]) / "proteins"
    prot_dir.mkdir(parents=True, exist_ok=True)

    for target, info in cfg["targets"].items():
        primary_pdb = info["primary_pdb"]
        chain_id = cfg["structures"][primary_pdb]["chain"]
        raw_pdb_path = raw_dir / f"{primary_pdb}.pdb"

        if not raw_pdb_path.exists():
            print(f"[SKIP] Missing structure file: {raw_pdb_path}")
            continue

        print(f"\n--- Preparing {target} ({primary_pdb}, Chain {chain_id}) ---")
        
        fixer = PDBFixer(filename=str(raw_pdb_path))
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

        # Direct OpenMM Modeller chain removal
        modeller = Modeller(fixer.topology, fixer.positions)
        chains_to_delete = [c for c in modeller.topology.chains() if c.id != chain_id]
        if chains_to_delete:
            modeller.delete(chains_to_delete)
            fixer.topology = modeller.topology
            fixer.positions = modeller.positions

        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(cfg["protein_prep"]["add_hydrogens_ph"])

        clean_pdb_path = prot_dir / f"{target}_{primary_pdb}_clean.pdb"
        with open(clean_pdb_path, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        print(f"[SUCCESS] Saved cleaned protein: {clean_pdb_path}")

if __name__ == "__main__":
    prepare_proteins()
