import time
import requests
import pandas as pd
from pathlib import Path
from utils import load_config


def fetch_chembl_data(chembl_id, activity_types, max_records=2000):
    """Fetches bioactivity records directly via ChEMBL REST API with chunking and retries."""
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    records = []
    limit = 500  # Fetch in smaller chunks of 500 to prevent connection resets
    offset = 0

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    while len(records) < max_records:
        params = {
            "target_chembl_id": chembl_id,
            "standard_type__in": ",".join(activity_types),
            "standard_units": "nM",
            "limit": limit,
            "offset": offset
        }

        success = False
        for attempt in range(3):  # Retry up to 3 times per page
            try:
                res = requests.get(base_url, params=params, headers=headers, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    page_activities = data.get("activities", [])
                    if not page_activities:
                        print(f"   Reached end of available records ({len(records)} fetched).")
                        return records
                    records.extend(page_activities)
                    print(f"   Fetched {len(records)} records so far...")
                    offset += limit
                    success = True
                    break
            except (requests.exceptions.RequestException, ConnectionError) as e:
                print(f"   [RETRY {attempt + 1}/3] Connection interrupted ({e}). Retrying in 3s...")
                time.sleep(3)

        if not success:
            print("   [WARNING] Skipping further pages due to repeated connection drops.")
            break

        time.sleep(0.5)  # Pause briefly to prevent rate limiting

    return records


def fetch_ligands():
    cfg = load_config()
    lig_dir = Path(cfg["output_dir"]) / "ligands"
    lig_dir.mkdir(parents=True, exist_ok=True)

    activity_types = cfg["ligand_curation"]["activity_types"]

    for target, info in cfg["targets"].items():
        chembl_id = info["chembl_id"]
        print(f"\n--- Fetching bioactive ligands for {target} ({chembl_id}) ---")

        raw_records = fetch_chembl_data(chembl_id, activity_types, max_records=2500)

        if not raw_records:
            print(f"[WARNING] No activity records found for {target}")
            continue

        df = pd.DataFrame(raw_records)

        # Retain key columns
        keep_cols = ["molecule_chembl_id", "standard_type", "standard_value", "standard_units", "canonical_smiles"]
        available_cols = [c for c in keep_cols if c in df.columns]
        df = df[available_cols]

        # Clean missing SMILES and numerical activity values
        df = df.dropna(subset=["canonical_smiles", "standard_value"])
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
        df = df.dropna(subset=["standard_value"])

        out_csv = lig_dir / f"{target}_raw_bioactivity.csv"
        df.to_csv(out_csv, index=False)
        print(f"[SUCCESS] Saved {len(df)} bioactivity records to {out_csv}")


if __name__ == "__main__":
    fetch_ligands()