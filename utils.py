from pathlib import Path
import yaml


def load_config(config_path="config.yaml"):
    """Loads and returns the project YAML configuration dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {path.resolve()}"
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # Test loading the config file
    cfg = load_config()
    print("Project Name:", cfg["project"])
    print("EGFR Primary PDB:", cfg["targets"]["EGFR"]["primary_pdb"])
    print(
        "HER2 Hinge Residue:",
        cfg["alignment"]["anchor_residues"]["hinge"]["HER2"],
    )