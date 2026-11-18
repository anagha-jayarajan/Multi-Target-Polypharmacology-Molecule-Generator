from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from reward import reward

lap = "CS(=O)(=O)CCNCc1ccc(o1)-c1ccc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2c1"
print("lapatinib formula:", CalcMolFormula(Chem.MolFromSmiles(lap)), "(expected C29H26ClFN4O4S)")

for name, smi in [("lapatinib", lap), ("benzene", "c1ccccc1"), ("invalid", "not_a_smiles")]:
    r, d = reward(smi)
    print(name, "reward", round(r, 3), d)