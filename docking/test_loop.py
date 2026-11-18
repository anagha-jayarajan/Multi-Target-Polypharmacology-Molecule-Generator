
import random
from importlib import import_module

if __name__ == "__main__":
    loop = import_module("03_rl_loop")

    class Dummy:
        pool = ["c1ccccc1", "CCO",
                "CS(=O)(=O)CCNCc1ccc(o1)-c1ccc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2c1"]
        def sample(self, n): return [random.choice(self.pool) for _ in range(n)]
        def update(self, smiles, rewards): pass

    loop.run(Dummy(), iters=2, batch=6, log="test_log.csv")

