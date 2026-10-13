import json
from pathlib import Path
from Bio.PDB import PDBParser, Superimposer, PDBIO, Select
from utils import load_config


class PocketSelect(Select):
    """Selects protein residues with atoms within distance_cutoff of ligand atoms."""
    def __init__(self, ligand_atoms, cutoff=10.0):
        self.ligand_atoms = ligand_atoms
        self.cutoff = cutoff

    def accept_residue(self, residue):
        if residue.id[0] != " ":  # Exclude non-protein residues/waters from output pocket
            return False
        for atom in residue:
            for lig_atom in self.ligand_atoms:
                if atom - lig_atom <= self.cutoff:
                    return True
        return False


def find_ligand_atoms(structure, raw_pdb_path, target_resname):
    """Searches for ligand heavy atoms across all residues, falling back to HETATM record scanning."""
    ligand_atoms = []
    target_resname = target_resname.strip().upper()
    common_solvents = {"HOH", "WAT", "DOD", "EDO", "DMS", "SO4", "PO4", "GOL", "PEG", "CL", "NA"}

    # 1. Direct search across all residues in BioPython model
    all_het_candidates = {}
    for residue in structure.get_residues():
        res_name = residue.resname.strip().upper()
        if res_name not in common_solvents:
            heavy_atoms = [a for a in residue if a.element != "H"]
            if heavy_atoms:
                if res_name not in all_het_candidates:
                    all_het_candidates[res_name] = []
                all_het_candidates[res_name].extend(heavy_atoms)

    if target_resname in all_het_candidates:
        return all_het_candidates[target_resname]

    # 2. Inspect raw PDB lines to check actual HETATM residue names
    raw_het_names = set()
    with open(raw_pdb_path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip().upper()
                if res_name not in common_solvents:
                    raw_het_names.add(res_name)

    print(f"   [DEBUG] Target '{target_resname}' missing. Non-solvent HETATMs in file: {raw_het_names or set(all_het_candidates.keys())}")

    # 3. Fallback: select largest non-protein/non-solvent residue group (> 10 atoms)
    for res_name, atoms in sorted(all_het_candidates.items(), key=lambda x: len(x[1]), reverse=True):
        if res_name not in {"ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}:
            if len(atoms) >= 10:
                print(f"   [AUTO-DETECT] Using detected co-crystal ligand '{res_name}' ({len(atoms)} heavy atoms)")
                return atoms

    return []


def define_pockets_and_align():
    cfg = load_config()
    raw_dir = Path(cfg["output_dir"]) / "raw"
    prot_dir = Path(cfg["output_dir"]) / "proteins"
    pocket_dir = Path(cfg["output_dir"]) / "pockets"
    pocket_dir.mkdir(parents=True, exist_ok=True)

    parser = PDBParser(QUIET=True)
    box_configs = {}

    print("--- 1. Defining Pockets & Computing Docking Boxes ---")
    for target, info in cfg["targets"].items():
        primary_pdb = info["primary_pdb"]
        target_ligand = cfg["structures"][primary_pdb]["ligand_resname"]

        raw_path = raw_dir / f"{primary_pdb}.pdb"
        clean_path = prot_dir / f"{target}_{primary_pdb}_clean.pdb"

        if not raw_path.exists() or not clean_path.exists():
            print(f"[SKIP] Missing files for {target}")
            continue

        raw_struct = parser.get_structure(primary_pdb, str(raw_path))
        ligand_atoms = find_ligand_atoms(raw_struct, raw_path, target_ligand)

        if not ligand_atoms:
            print(f"[WARNING] Could not locate any ligand in {primary_pdb}")
            continue

        # Calculate bounding box center and size (padding = 8.0 A)
        coords = [a.get_coord() for a in ligand_atoms]
        min_xyz = [min(c[i] for c in coords) for i in range(3)]
        max_xyz = [max(c[i] for c in coords) for i in range(3)]

        center = [float(round((min_xyz[i] + max_xyz[i]) / 2.0, 3)) for i in range(3)]
        size = [float(round((max_xyz[i] - min_xyz[i]) + 8.0, 3)) for i in range(3)]

        box_configs[target] = {
            "center_x": center[0],
            "center_y": center[1],
            "center_z": center[2],
            "size_x": size[0],
            "size_y": size[1],
            "size_z": size[2],
        }

        # Extract binding pocket from clean protein structure
        clean_struct = parser.get_structure(f"{target}_clean", str(clean_path))
        io = PDBIO()
        io.set_structure(clean_struct)
        pocket_path = pocket_dir / f"{target}_pocket.pdb"
        io.save(str(pocket_path), PocketSelect(ligand_atoms, cutoff=10.0))

        print(f"[{target}] Box Center: {center}, Size: {size}")
        print(f"[{target}] Saved pocket PDB: {pocket_path}")

    # Save docking box configurations JSON
    box_json_path = pocket_dir / "docking_boxes.json"
    with open(box_json_path, "w") as f:
        json.dump(box_configs, f, indent=4)
    print(f"\n[SAVED] Docking box settings: {box_json_path}")

    print("\n--- 2. Structural Alignment (Superimposing HER2 onto EGFR) ---")
    egfr_pdb = cfg["targets"]["EGFR"]["primary_pdb"]
    her2_pdb = cfg["targets"]["HER2"]["primary_pdb"]

    egfr_clean = prot_dir / f"EGFR_{egfr_pdb}_clean.pdb"
    her2_clean = prot_dir / f"HER2_{her2_pdb}_clean.pdb"

    if egfr_clean.exists() and her2_clean.exists():
        struct_ref = parser.get_structure("EGFR", str(egfr_clean))
        struct_mobile = parser.get_structure("HER2", str(her2_clean))

        ref_atoms = []
        mobile_atoms = []

        egfr_anchors = cfg["structures"][egfr_pdb]["anchor_residues"]
        her2_anchors = cfg["structures"][her2_pdb]["anchor_residues"]

        for key in ["hinge", "gatekeeper", "catalytic_lys", "dfg_asp"]:
            ref_res_num = egfr_anchors[key]
            mob_res_num = her2_anchors[key]

            for res in struct_ref.get_residues():
                if res.id[1] == ref_res_num and "CA" in res:
                    ref_atoms.append(res["CA"])
            for res in struct_mobile.get_residues():
                if res.id[1] == mob_res_num and "CA" in res:
                    mobile_atoms.append(res["CA"])

        if len(ref_atoms) == len(mobile_atoms) and len(ref_atoms) > 0:
            super_imposer = Superimposer()
            super_imposer.set_atoms(ref_atoms, mobile_atoms)
            super_imposer.apply(struct_mobile.get_atoms())

            print(f"[ALIGNMENT] RMSD across ATP-binding anchors: {float(super_imposer.rms):.3f} Å")

            aligned_path = prot_dir / f"HER2_{her2_pdb}_aligned_to_EGFR.pdb"
            io = PDBIO()
            io.set_structure(struct_mobile)
            io.save(str(aligned_path))
            print(f"[SAVED] Aligned HER2 structure: {aligned_path}")


if __name__ == "__main__":
    define_pockets_and_align()