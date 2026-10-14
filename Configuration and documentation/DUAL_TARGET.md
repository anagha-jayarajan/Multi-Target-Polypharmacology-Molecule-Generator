# Dual-target candidate pool

A single candidate pool was assembled from the available EGFR and HER2 molecule files:

- `egfr_mols_20.sdf`: 20 valid molecules
- `her2_mols_valid.sdf`: 18 valid molecules

The structures were compared by canonical isomeric SMILES. The resulting pool contains 38 unique molecules, with no exact canonical-SMILES duplicates across the two input sets. Each molecule was assigned a stable ID (`DT_001` through `DT_038`) and labeled with its source set.

## Files produced

- `Outputs-git/dual_target_candidates.sdf` contains the combined 3D structures and their candidate IDs, source labels, original IDs, and canonical SMILES.
- `Outputs-git/dual_target_candidate_manifest.csv` contains the candidate IDs, source labels, canonical SMILES, molecular weights, heavy-atom counts, and 3D-coordinate indicators.
- `Scripts/07_build_dual_target_candidates.py` recreates the combined SDF and manifest from the two input SDF files. It requires RDKit.

Run the builder from the repository root in an environment with RDKit installed:

```powershell
python Scripts/07_build_dual_target_candidates.py
```

## Scope

The input molecules were generated separately using EGFR and HER2 structures. The combined pool is a generate-and-filter starting set; it is not the result of simultaneous two-protein conditioning. No docking scores or binding conclusions are included in these files.

