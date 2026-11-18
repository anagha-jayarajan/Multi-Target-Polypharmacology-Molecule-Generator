import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from dock import dock_smiles

def run(target, actives_csv, decoys_smi, n_decoys=500, out="enrich.csv"):
    act = pd.read_csv(actives_csv)["smiles"].tolist()      # check Person 1's column name
    dec = [l.split()[0] for l in open(decoys_smi)][:n_decoys]
    rows = [(s, 1) for s in act] + [(s, 0) for s in dec]
    df = pd.DataFrame(rows, columns=["smiles", "label"])
    df["score"] = [dock_smiles(s, target) for s in df.smiles]
    df = df.dropna()
    y, s = df.label.values, -df.score.values               # higher = better
    auc = roc_auc_score(y, s)
    top = max(1, int(0.01 * len(df)))
    ef1 = (y[np.argsort(-s)][:top].mean()) / y.mean()
    df.to_csv(out, index=False)
    print(f"{target}: ROC-AUC {auc:.2f}, EF1% {ef1:.1f}, n={len(df)}")

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])