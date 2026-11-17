# Validation report update

Docking mode: **vina**
Vina settings used: {"exhaustiveness": 2, "num_modes": 5, "energy_range": 3, "cpu": 1, "seed": 42, "timeout_s": 900}

**Time-budgeted profile** (target 80 min): screening exhaustiveness 2 (benchmark, RL, evaluation), redocking exhaustiveness 8, RL steps per mode {'EGFR': 4, 'HER2': 4, 'dual': 5}, 4 molecules per model docked on both targets in the evaluation.

## 1. EGFR (1XKK / lapatinib) and HER2 (3PP0 / TAK-285) redocking

Success criterion: heavy-atom RMSD of the top-scoring pose <= 2.0 A (no re-alignment). RMSD is computed over the atoms common to the crystal ligand and the docked molecule (see `atoms_compared`); in 3PP0 only 34 of TAK-285's 38 heavy atoms are modelled.

| target   | ligand   |   exhaustiveness |   seed |   best_score |   rmsd_A |   atoms_compared |   crystal_heavy_atoms | success_le_2A   |   runtime_s |
|:---------|:---------|-----------------:|-------:|-------------:|---------:|-----------------:|----------------------:|:----------------|------------:|
| EGFR     | FMM      |                8 |     42 |      -11.094 |    2.954 |               40 |                    40 | False           |       399.4 |
| HER2     | 03Q      |                8 |     42 |      -11.771 |    0.466 |               29 |                    34 | True            |       385.7 |

## 2. Known-inhibitor benchmark

| name      | role             | smiles                                                               |   dock_EGFR |   dock_HER2 |   dock_mean |   rank_EGFR |   rank_HER2 |   rank_mean |
|:----------|:-----------------|:---------------------------------------------------------------------|------------:|------------:|------------:|------------:|------------:|------------:|
| gefitinib | known_inhibitor  | COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1                       |      -8.372 |      -9.134 |     -8.753  |           3 |           2 |           3 |
| lapatinib | known_inhibitor  | CS(=O)(=O)CCNCc1ccc(o1)-c1ccc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2c1 |     -11.054 |      -7.183 |     -9.1185 |           1 |           4 |           2 |
| erlotinib | known_inhibitor  | COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC                             |      -7.749 |      -8.516 |     -8.1325 |           4 |           3 |           4 |
| TAK-285   | known_inhibitor  | CC(C)(O)CC(=O)NCCn1ccc2ncnc(Nc3ccc(Oc4cccc(C(F)(F)F)c4)c(Cl)c3)c21   |     -10.632 |     -11.771 |    -11.2015 |           2 |           1 |           1 |
| caffeine  | negative_control | Cn1cnc2c1c(=O)n(C)c(=O)n2C                                           |      -5.654 |      -6.291 |     -5.9725 |           7 |           6 |           6 |
| aspirin   | negative_control | CC(=O)Oc1ccccc1C(=O)O                                                |      -6.262 |      -6.432 |     -6.347  |           5 |           5 |           5 |
| glucose   | negative_control | OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O                               |      -5.684 |      -5.265 |     -5.4745 |           6 |           7 |           7 |

Known inhibitors should score clearly better (more negative) than the negative controls on both targets.

## 3. Exhaustiveness

|   exhaustiveness |   score_mean |   score_sd |   rmsd_mean |   runtime_mean_s |
|-----------------:|-------------:|-----------:|------------:|-----------------:|
|                1 |      -10.988 |      0.301 |       2.916 |            54.75 |
|                2 |      -10.995 |      0.03  |       2.796 |           101.2  |
|                4 |      -10.995 |      0.03  |       2.796 |           200.2  |

**Recommended exhaustiveness: 4** (criteria NOT met)

No tested level met the criteria; the highest level was returned. Test higher exhaustiveness or review the box/receptor before using it.
Rule: smallest value with score SD <= 0.15, mean within 0.20 kcal/mol of the highest level, and mean redocking RMSD <= 2.0 A (falls back to the highest level tested).

## 4. Generative model evaluation

Validity, uniqueness, novelty, QED and SA use all `n_sampled` molecules of each model. Docking-based columns use only the random subset counted in `n_docked` and are therefore directional, not statistically tight. RL ran for a few steps only.

| condition   |   n_sampled |   n_valid |   n_unique |   n_novel |   n_docked |   validity_pct |   uniqueness_pct |   novelty_pct |   internal_diversity |   qed_mean |   qed_median |   sa_mean |   sa_median |   dock_EGFR_mean |   dock_EGFR_best |   dock_HER2_mean |   dock_HER2_best |   dock_mean_both |   dock_gap_mean |   dual_hit_pct |   reward_dual_mean |   reward_dual_max | dock_mode   |   eval_exhaustiveness |   rl_steps |   rl_sample_per_step |   rl_dock_per_step |   n_dock_per_model_target |   n_pretrain_molecules |
|:------------|------------:|----------:|-----------:|----------:|-----------:|---------------:|-----------------:|--------------:|---------------------:|-----------:|-------------:|----------:|------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|---------------:|-------------------:|------------------:|:------------|----------------------:|-----------:|---------------------:|-------------------:|--------------------------:|-----------------------:|
| prior       |         500 |       244 |        151 |       124 |          2 |           48.8 |           61.885 |        82.119 |                0.799 |      0.485 |        0.467 |     2.814 |       2.686 |           -9.36  |           -9.4   |           -8.683 |           -9.098 |           -9.021 |           0.676 |             50 |              0.455 |             0.475 | vina        |                     2 |        nan |                   32 |                  5 |                         4 |                     28 |
| agent_dual  |         500 |       255 |        155 |       130 |          2 |           51   |           60.784 |        83.871 |                0.797 |      0.504 |        0.494 |     2.793 |       2.67  |           -8.636 |           -9.696 |           -7.889 |           -8.454 |           -8.262 |           0.748 |              0 |              0.301 |             0.404 | vina        |                     2 |          5 |                   32 |                  5 |                         4 |                     28 |
| agent_EGFR  |         500 |       258 |        173 |       149 |          2 |           51.6 |           67.054 |        86.127 |                0.774 |      0.497 |        0.485 |     2.773 |       2.651 |           -9.997 |          -10.139 |          -10.514 |          -10.781 |          -10.255 |           0.517 |            100 |              0.617 |             0.671 | vina        |                     2 |          4 |                   32 |                  5 |                         4 |                     28 |
| agent_HER2  |         500 |       271 |        167 |       143 |          2 |           54.2 |           61.624 |        85.629 |                0.789 |      0.468 |        0.446 |     2.825 |       2.703 |           -9.994 |          -11.054 |           -7.877 |           -8.571 |           -8.935 |           2.117 |              0 |              0.296 |             0.392 | vina        |                     2 |          4 |                   32 |                  5 |                         4 |                     28 |
