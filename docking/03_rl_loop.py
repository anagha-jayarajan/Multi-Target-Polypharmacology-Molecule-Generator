import csv, random, numpy as np
from concurrent.futures import ProcessPoolExecutor
from reward import reward

class Generator:
    """Placeholder. Replace with the wrapper Persons 2 and 3 provide
    (TargetDiff/DiffSBDD sampling conditioned on the pocket)."""
    def sample(self, n):            # -> list of SMILES
        raise NotImplementedError
    def update(self, smiles, rewards):   # reward-weighted fine-tune or guidance
        raise NotImplementedError

def score_batch(smiles_list, workers=4):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(reward, smiles_list))

def run(gen, iters=20, batch=64, log="rl_log.csv", mode="dual"):
    random.seed(42); np.random.seed(42)
    with open(log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "smiles", "reward", "egfr", "her2", "qed", "sa"])
        for it in range(iters):
            smis = list(set(gen.sample(batch)))
            res = score_batch(smis)
            rewards = [r for r, _ in res]
            for s, (r, d) in zip(smis, res):
                w.writerow([it, s, r, d.get("egfr"), d.get("her2"),
                            d.get("qed"), d.get("sa")])
            gen.update(smis, rewards)
            print(f"iter {it}: mean reward {np.mean(rewards):.3f}, "
                  f"valid-docked {sum(1 for _, d in res if d)/len(res):.0%}")

if __name__ == "__main__":
    run(Generator())