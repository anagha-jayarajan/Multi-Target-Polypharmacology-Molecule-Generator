# Data Preparation Module (`data_prep/`)

Welcome to the **Data Preparation** stage for our Multi-Target Polypharmacology Project targeting **EGFR** and **HER2**. 

This module automates target protein cleaning, binding pocket definition, docking box calculation, HER2-to-EGFR superimposition, ChEMBL bioactivity curation, and 3D PDBQT ligand conformer generation.

---

## 🚀 Quick Start Guide for Team Members

### Step 1: Environment Setup
Ensure you are in the `data_prep` directory and have activated the Python virtual environment:

```powershell
# Navigate to data_prep folder
cd "path/to/AIMS CAPSTONE/data_prep"

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install biopython rdkit meeko pandas requests scikit-learn

## 📂 Output Artifacts & Handoff Guide

All output files are saved inside the **`outputs/`** directory. Use the guide below to locate deliverables for your role:

### 🟢 Person 2: Baseline 3D Molecule Generator
* **Binding Pockets:** `outputs/pockets/EGFR_pocket.pdb` & `outputs/pockets/HER2_pocket.pdb`
* **Pocket Center & Size:** `outputs/pockets/docking_boxes.json`
* **Reference Ligands:** `outputs/ligands/EGFR_curated_actives.csv` & `outputs/ligands/HER2_curated_actives.csv`

### 🟢 Person 3: Dual-Target Conditioning
* **Superimposed Protein:** `outputs/proteins/HER2_aligned.pdb` (HER2 aligned onto EGFR space)
* **Dual Target Box Coordinates:** `outputs/pockets/docking_boxes.json`

### 🟢 Person 4: Docking, Reward Function & RL
* **Clean Receptor PDBs:** `outputs/proteins/EGFR_clean.pdb` & `outputs/proteins/HER2_clean.pdb`
* **Docking Grid Box:** `outputs/pockets/docking_boxes.json`
* **Known Active Conformer Benchmarks:** `outputs/ligands_pdbqt/EGFR/` & `outputs/ligands_pdbqt/HER2/` (Download `ligands_pdbqt.zip` from team drive and extract to `outputs/`)

### 🟢 Person 5: Evaluation & Benchmarking
* **Test Datasets for Novelty/Similarity:** `outputs/ligands/EGFR_test.csv` & `outputs/ligands/HER2_test.csv`
* **Complete Curated Sets:** `outputs/ligands/*_curated_actives.csv`

---

## ⚙️ Reproducing the Pipeline (Sequential Execution)

To re-run the entire pipeline from scratch, execute the following scripts in order:

### Script 01: Fetch Raw PDB Structures
Downloads raw PDB structures (`1XKK` for EGFR, `3PP0` for HER2) into `outputs/raw/`.
```powershell
python 01_fetch_structures.py

### Script 02: Prepare Proteins
Cleans protein structures, adds missing heavy atoms/loops via OpenMM/PDBFixer, and protonates residues at pH 7.4.
```powershell
python 02_prepare_proteins.py

### Script 03: Define Pockets & Superimpose
Identifies native ligands (`FMM`, `03Q`), extracts 10 Å binding pockets, calculates docking bounding boxes (`docking_boxes.json`), and superimposes HER2 onto EGFR[cite: 1].
```powershell
python 03_define_pockets.py

### Script 04: Fetch Bioactivity Data
Queries the ChEMBL API for EGFR (`CHEMBL203`) and HER2 (`CHEMBL1824`) bioactivities and exports raw CSV records[cite: 1].
```powershell
python 04_fetch_ligands.py

### Script 05: Curate Ligands & Train/Test Split
Standardizes SMILES (salt stripping, canonicalization), filters active compounds ($\le 1000\text{ nM}$), deduplicates structures, and saves 80/20 train/test splits[cite: 1].
```powershell
python 05_curate_ligands.py

### Script 06: Generate 3D PDBQT Conformers
Generates low-energy 3D ETKDGv3/MMFF conformers and writes `.pdbqt` files via Meeko for docking validation[cite: 1].
```powershell
python 06_prepare_ligand_pdbqt.py

## 📊 Dataset Statistics Summary

| Target | ChEMBL ID | PDB ID | Unique Actives ($\le 1000\text{ nM}$) | Train Split | Test Split | 3D Conformer PDBQTs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EGFR** | CHEMBL203 | 1XKK | 769 | 615 | 154 | 763 |
| **HER2** | CHEMBL1824 | 3PP0 | 956 | 764 | 192 | 956 |