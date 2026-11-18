import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, RDConfig
import os, sys
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer
from dock import CFG, dock_smiles

R = CFG["reward"]

def norm_score(s):
    # -12 kcal/mol -> 1, -4 -> 0
    return float(np.clip((R["score_worst"] - s) / (R["score_worst"] - R["score_best"]), 0, 1))

def reward(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return R["failure_reward"], {}
    e = dock_smiles(smiles, "EGFR")
    h = dock_smiles(smiles, "HER2")
    if e is None or h is None:
        return R["failure_reward"], {}
    ne, nh = norm_score(e), norm_score(h)
    dual = min(ne, nh)                      # molecule must do well on both targets
    qed = QED.qed(mol)
    sa = 1 - (sascorer.calculateScore(mol) - 1) / 9   # 1 = easy to make
    imbalance = abs(ne - nh)
    r = (R["w_dock"] * dual + R["w_qed"] * qed + R["w_sa"] * sa
         - R["imbalance_penalty"] * imbalance)
    return r, {"egfr": e, "her2": h, "qed": qed, "sa": sa, "reward": r}