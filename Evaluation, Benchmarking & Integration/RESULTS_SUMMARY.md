# Results summary: EGFR + HER2 dual-target prototype (Person 5)

Produced by `Scripts/09_evaluate_benchmark.py` (reproducible, ~10 s, no docking). Tables and figures are in `Outputs-git/evaluation/`; the walk-through is `Evaluation_Notebook.ipynb`.

## What was actually compared

| Group | n | What it is |
|---|---|---|
| EGFR-gen / HER2-gen / Pool | 20 / 18 / 38 | Pretrained single-target generator output, pooled (Person 3's generate-and-filter set) |
| GA-optimised | 66 | Genetic algorithm on the dual-docking reward (Person 4) |
| RL-optimised | 237 | REINVENT-style RL on the same reward (Person 4) |
| Known actives | 1,616 | ChEMBL EGFR/HER2 actives, properties only (never docked) |

**Limits that shape every conclusion**
1. **No true dual-conditioned generator output exists in the repo** (`DUAL_TARGET.md` says the pool is generate-and-filter). So "dual conditioning vs. single-target + filter" cannot be tested; what can be compared is the filtered baseline vs. the optimisers.
2. **GA/RL molecules are 2D SMILES derived from known EGFR/HER2 actives**, not output of the 3D diffusion model. 18 of the 66 GA and 22 of the 237 RL molecules *are* known actives (rediscovered); they are excluded from "novel" counts and the shortlist.
3. **The optimisers were trained on the same docking reward that scores them**, so beating the baseline on docking is expected by construction.

## Headline numbers

| | Pool (38) | GA (66) | RL (237) |
|---|---|---|---|
| RDKit parse validity | 100% | not logged | 54% (sampled SMILES) |
| Pass chemistry gate | 39% (15/38) | 100% | 95% |
| Docked on both targets | 34% (13/38) | 100% | 95% |
| Median SA / QED | 4.54 / 0.35 | 2.89 / 0.52 | 2.41 / 0.60 |
| Median score EGFR / HER2 (kcal/mol) | -7.4 / -8.3 | -9.8 / -10.1 | -8.6 / -9.3 |
| Dual hits (both <= -8), novel only | 5 | 35 | 112 |
| **Dual leads** (novel, both <= -8, \|Δ\| <= 1, QED >= 0.5, SA <= 4.5) | **2 (5%)** | **13 (20%)** | **39 (16%)** |
| Median ligand efficiency (kcal/mol/heavy atom) | 0.36 | 0.34 | 0.36 |

All uniqueness values are 100%. Novelty is trivially ~0.83 for the pool because most of its molecules are chemically implausible, so it says nothing positive about them.

## Interpretation

- **The single-target generator output is poor.** 23 of 38 molecules fail the chemistry gate (allenes, strained rings, PAINS, peroxides...), median SA is 4.5 and QED 0.35. Filtering leaves 13 docked survivors, 2 of which meet the lead criteria.
- **GA and RL give far better candidates by the reward's own metrics**: better docking (GA median mean-score -9.96 vs -8.18, p < 0.001), far easier synthesis (SA 2.9 / 2.4 vs 3.7 among survivors). Caution: these are largely kinase-inhibitor scaffolds close to known drugs (median max Tanimoto to a known active 0.58 GA / 0.56 RL; 39% / 29% have an analogue >= 0.7).
- **Ligand efficiency is the same as the baseline survivors** (0.34-0.36, p > 0.8), and docking score correlates with size (Spearman -0.45 GA, -0.50 RL). Much of the raw score gain therefore reflects larger, more kinase-like molecules rather than tighter per-atom binding.
- **RL did not show sustained improvement** in 30 steps: mean reward rose to ~0.40 around step 11 then fell to ~0.2, and the parse rate fell from its peak (Fig. 4). The RL set is best read as samples from the pretrained prior, not evidence that RL fine-tuning worked. GA improved steadily (best 0.814 -> 0.835; mean of new molecules 0.69 -> 0.77) but over only 4 generations.
- Dual-target balance is similar across groups (median |Δscore| 0.7-0.9).

## Issues Person 4 / the team should check before the final demo (`qa_checks.csv`)

1. **EGFR redocking fails the 2.0 Å criterion** (top pose 2.47 Å; best of 9 modes 1.34 Å). HER2 passes (0.62 Å). EGFR scores are less validated.
2. **The committed HER2 docking box is wrong** (written by Person 1's `03_define_pockets.py`): its centre is 25 Å from the HER2 crystal ligand (EGFR: 0.7 Å), it is 11x the volume of EGFR's, and it exceeds Person 4's own 30 Å cap. Person 4's script uses a different file, `docking_boxes_fixed.json`, which is **not committed**, so which box produced the committed HER2 scores cannot be verified. Commit that file.
3. **Vina exhaustiveness defaults to 4**; the script recommends 8-32 for final numbers. Confirm what produced the committed scores.
4. **No docking of known EGFR/HER2 inhibitors** is committed, so there is no yardstick for whether -9 or -11 kcal/mol is good here. Docking ~20 known actives would fix this.
5. 2 of 15 gate-passing baseline molecules have no docking score (failures).
6. `DUAL_TARGET.md` says IDs `DT_001-038`; files use `P3_001-038`. `p3_candidates_scored.csv` duplicates `scored_candidates.csv` (same values, different row order). The docking script exists twice (repo root and `Scripts/`, identical).

## Review of Persons 1-4 (what is solid, what is not)

| Person | Solid | Problems found |
|---|---|---|
| **P1 data prep** | Ligand data is clean: no duplicate molecules, no train/test overlap, consistent <= 1000 nM cut-off, 80/20 split, EGFR 769 / HER2 956 actives, 106 shared actives. Crystal ligands and pockets are correct. | (a) `docking_boxes.json` is wrong for HER2 (above). (b) Cleaned PDBs are renumbered 1..N, so the anchor residue numbers in `config.yaml` match nothing; script 03's HER2->EGFR alignment is **silently skipped** (no `else`), so no aligned structure exists and "aligned-frame conditioning" had no input. HER2 anchor numbers also look offset (hinge/gatekeeper 805/797; verify against the UniProt numbering). (c) Plan said ChEMBL **and** BindingDB; only ChEMBL was used. (d) IC50 and Ki are mixed in one potency cut-off; 16-17 molecules sit in one target's train and the other's test. (e) `*_clean.pdb` still contains the ligand, waters and phosphate (OK for the docking script, which strips them, but the name is misleading). (f) The test split is never used for evaluation. |
| **P2 baseline generator** | 38 molecules with 3D coordinates, correctly placed in their own pockets (EGFR 20/20 inside the box; HER2 17/18 inside the very large box). | (a) **No generator code, config, checkpoint, seed or sampling settings in the repo**: the deliverable cannot be reproduced, and nothing shows the model really was TargetDiff/DiffSBDD. SDFs sit loose in the repo root. (b) Only valid molecules were kept (`her2_mols_valid.sdf`), so the **true validity rate of the generator is unknown**. (c) Output quality is poor: 23/38 fail the chemistry gate and 7/38 have impossible bond lengths (e.g. O-O 0.4 Å, C=O 0.86 Å). Likely a sampling/bond-inference problem worth fixing before presenting. |
| **P3 dual-target** | Script 07 is clean, deterministic and documented, dedupes correctly, and `DUAL_TARGET.md` is honest that this is generate-and-filter. | (a) **The core deliverable (dual-target conditioning) was not attempted**; the fallback (pool + filter) is only the simplest baseline and no "baseline comparison" was written. (b) Doc/ID mismatch (`DT_` vs `P3_`). (c) Extra duplicate `p3_candidates_scored.csv`. (d) Inherits the missing alignment from P1. |
| **P4 docking/reward/RL** | The most substantial work: sound reward design (geometric mean of normalised scores, balance term, QED/SA/novelty, chemistry gate, size penalty), docking cache, co-crystal redocking, GA and resumable RL, committed result tables. Novelty values reproduce exactly (max diff 0.000). | (a) EGFR redocking **fails** its own 2.0 Å criterion (2.47 Å). (b) `docking_boxes_fixed.json` not committed; exhaustiveness default 4 (validation used 16); confirm what the scores used. (c) **Known inhibitors were never docked**, so scores have no calibration. (d) GA/RL start from known actives (18 + 22 exact known actives in the logged sets) and use train **and** test sets, leaving no held-out data. (e) RL ran only 30 steps with no sustained gain. (f) Script is duplicated (root and `Scripts/`) and the file name contains commas, so it is awkward to run; scripts 01-06 import `utils`/`config.yaml` from folders they are not in. |

**Nothing here looks like fabricated or manipulated data**; the problems are gaps, silent failures and unverified settings. The two things most likely to cost marks are the missing dual-target conditioning (P3, which should be stated plainly) and the missing/unreproducible generator (P2).

## Suggested framing for the final report

Present the system as a **proof-of-concept pipeline** (data prep -> candidate generation -> dual docking -> reward -> evaluation) in which a reward-driven optimiser produces dual-docking leads far more efficiently than filtering raw single-target generations. State plainly that dual-target conditioning of the diffusion model, 3D generation by the optimisers, and RL fine-tuning gains were **not** demonstrated, and that docking scores are unvalidated predictions (no experimental or MD confirmation).

## Reproduce
```
pip install -r requirements-eval.txt
python Scripts/09_evaluate_benchmark.py          # or: python Scripts/10_run_pipeline.py
```
