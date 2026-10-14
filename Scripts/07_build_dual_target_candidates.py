"""Build the Person 3 generate-and-filter candidate pool from both baselines.

The EGFR-conditioned and HER2-conditioned molecules are kept as one shared
pool for Person 4 to dock against both targets. Exact canonical-SMILES
duplicates are written once, with provenance from both input sets retained.
Requires RDKit (available in the DiffSBDD environment used by Person 2).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def canonical_smiles(mol: Chem.Mol) -> str:
    """Return canonical isomeric SMILES without explicit hydrogens."""
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)


def load_candidates(path: Path, target: str, pool: OrderedDict[str, dict]) -> tuple[int, int]:
    """Read valid SDF records and accumulate unique molecules by structure."""
    if not path.is_file():
        raise FileNotFoundError(f"Input SDF not found: {path}")
    records = 0
    invalid = 0
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    for index, mol in enumerate(supplier, start=1):
        records += 1
        if mol is None:
            invalid += 1
            continue
        try:
            smiles = canonical_smiles(mol)
        except Exception:
            invalid += 1
            continue
        if not smiles:
            invalid += 1
            continue

        source_id = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        if not source_id:
            source_id = f"{target}_candidate_{index:02d}"
        if smiles not in pool:
            pool[smiles] = {
                "mol": Chem.Mol(mol),
                "source_ids": [],
                "sources": [],
            }
        entry = pool[smiles]
        if source_id not in entry["source_ids"]:
            entry["source_ids"].append(source_id)
        if target not in entry["sources"]:
            entry["sources"].append(target)
    return records, invalid


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egfr-sdf", type=Path, default=project_root / "egfr_mols_20.sdf")
    parser.add_argument("--her2-sdf", type=Path, default=project_root / "her2_mols_valid.sdf")
    parser.add_argument("--output-sdf", type=Path, default=project_root / "Outputs-git" / "dual_target_candidates.sdf")
    parser.add_argument("--manifest", type=Path, default=project_root / "Outputs-git" / "dual_target_candidate_manifest.csv")
    args = parser.parse_args()

    pool: OrderedDict[str, dict] = OrderedDict()
    try:
        egfr_records, egfr_invalid = load_candidates(args.egfr_sdf, "EGFR", pool)
        her2_records, her2_invalid = load_candidates(args.her2_sdf, "HER2", pool)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not pool:
        print("Error: no valid molecules were found in the input SDF files.", file=sys.stderr)
        return 2

    args.output_sdf.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    writer = Chem.SDWriter(str(args.output_sdf))
    try:
        for number, (smiles, entry) in enumerate(pool.items(), start=1):
            candidate_id = f"P3_{number:03d}"
            mol = Chem.Mol(entry["mol"])
            generation_sources = "+".join(entry["sources"])
            source_candidate_ids = ";".join(entry["source_ids"])
            mol.SetProp("_Name", candidate_id)
            mol.SetProp("candidate_id", candidate_id)
            mol.SetProp("generation_sources", generation_sources)
            mol.SetProp("source_candidate_ids", source_candidate_ids)
            mol.SetProp("canonical_smiles", smiles)
            writer.write(mol)
            manifest_rows.append({
                "candidate_id": candidate_id,
                "generation_sources": generation_sources,
                "source_candidate_ids": source_candidate_ids,
                "canonical_smiles": smiles,
                "heavy_atom_count": rdMolDescriptors.CalcNumHeavyAtoms(Chem.RemoveHs(mol)),
                "has_3d_coordinates": str(bool(mol.GetNumConformers() and mol.GetConformer().Is3D())).lower(),
            })
    finally:
        writer.close()

    with args.manifest.open("w", newline="", encoding="utf-8") as stream:
        fields = ["candidate_id", "generation_sources", "source_candidate_ids", "canonical_smiles", "heavy_atom_count", "has_3d_coordinates"]
        csv_writer = csv.DictWriter(stream, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(manifest_rows)

    duplicate_count = egfr_records + her2_records - egfr_invalid - her2_invalid - len(pool)
    print(f"EGFR input: {egfr_records - egfr_invalid} valid / {egfr_records} records")
    print(f"HER2 input: {her2_records - her2_invalid} valid / {her2_records} records")
    print(f"Unique shared candidate pool: {len(pool)} (collapsed exact duplicates: {duplicate_count})")
    print(f"SDF for Person 4 to dock against both targets: {args.output_sdf}")
    print(f"Candidate/provenance manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
