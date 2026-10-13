import shutil
from pathlib import Path
from Bio.PDB import PDBList
from utils import load_config


def fetch_and_organize_structures():
    cfg = load_config()

    # Paths
    data_prep_dir = Path(".")
    existing_raw_dir = data_prep_dir / "raw"
    outputs_raw_dir = Path(cfg["output_dir"]) / "raw"
    outputs_raw_dir.mkdir(parents=True, exist_ok=True)

    pdbl = PDBList()

    # Gather all PDB IDs listed in config
    all_pdbs = []
    for target_name, info in cfg["targets"].items():
        all_pdbs.append(info["primary_pdb"])
        all_pdbs.extend(info.get("backup_pdbs", []))

    print("--- Checking and Fetching PDB Structures ---")

    for pdb_id in all_pdbs:
        pdb_id_upper = pdb_id.upper()
        target_file = outputs_raw_dir / f"{pdb_id_upper}.pdb"

        # 1. Check if file already exists in outputs/raw
        if target_file.exists():
            print(f"[FOUND] {pdb_id_upper}.pdb already exists in {outputs_raw_dir}")
            continue

        # 2. Check if file exists in data_prep/raw
        local_file = existing_raw_dir / f"{pdb_id_upper}.pdb"
        if local_file.exists():
            print(f"[COPYING] Copying {pdb_id_upper}.pdb from raw/ to {outputs_raw_dir}")
            shutil.copy(local_file, target_file)
            continue

        # 3. If missing, download from RCSB PDB
        print(f"[DOWNLOADING] {pdb_id_upper} from RCSB PDB...")
        downloaded_path = pdbl.retrieve_pdb_file(
            pdb_id_upper, pdir=str(outputs_raw_dir), file_format="pdb"
        )
        
        # Rename downloaded pdb file (BioPython downloads as pdb1xkk.ent)
        downloaded_file = Path(downloaded_path)
        if downloaded_file.exists():
            downloaded_file.rename(target_file)

    print(f"\nAll structures are ready in: {outputs_raw_dir.resolve()}")


if __name__ == "__main__":
    fetch_and_organize_structures()