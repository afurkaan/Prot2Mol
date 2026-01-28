#!/usr/bin/env python
"""
Stage 1: Prepare processed datasets for downstream analysis.

- Read raw generated molecule CSVs (Phase I/II/III, AKT1/CDK2).
- Read raw Papyrus filtered dataset.
- Decode SELFIES to SMILES where needed.
- Canonicalize SMILES and compute RDKit descriptors (MW, HBD, HBA, etc.).
- Normalize column names (sa, qed, logp, predicted_pchembl, etc.).
- Save processed datasets under data/processed/...

Raw datasets are NEVER modified; only new processed copies are written.

You can run this script from anywhere, e.g.:

    python analysis/prepare_datasets.py \
        --raw-dir /path/to/Prot2Mol/data/raw \
        --processed-dir /path/to/Prot2Mol/data/processed

If you omit the flags, defaults are inferred relative to the repository root.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski

# -------------------------------------------------------------------------
# Make sure we can import the prot2mol package regardless of CWD
# -------------------------------------------------------------------------

# Repo root = parent of this "analysis" directory
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# Now we can import utils from prot2mol
from prot2mol.utils import (  # type: ignore
    get_mol,
    canonic_smiles,
    sascorer_calculation,
    qed_calculation,
    logp_calculation,
)

RDLogger.DisableLog("rdApp.*")

# -------------------------------------------------------------------------
# Static configuration (non-path)
# -------------------------------------------------------------------------

# Mapping from protein CHEMBL ID to a human-readable target symbol.
# !!! IMPORTANT: update these IDs to match your dataset exactly. !!!
TARGET_CHEMBL_TO_SYMBOL: Dict[str, str] = {
    "CHEMBL4282": "AKT1",  # example for AKT1
    "CHEMBL301": "CDK2",  # example for CDK2
}

# Raw CSV reading configuration defaults
DEFAULT_GENERATED_CSV_SEP = "\t"   # generation output often uses tab-separated format
DEFAULT_PAPYRUS_CSV_SEP = ","      # Papyrus filtered CSV is usually comma-separated

# -------------------------------------------------------------------------
# Helper functions (path-independent)
# -------------------------------------------------------------------------

def safe_decode_selfies(selfies_str: Optional[str]) -> Optional[str]:
    """Decode SELFIES string to SMILES. Return None if decoding fails."""
    if selfies_str is None or (isinstance(selfies_str, float) and np.isnan(selfies_str)):
        return None
    try:
        return sf.decoder(selfies_str)
    except Exception:
        return None

def add_common_rdkit_descriptors(
    df: pd.DataFrame,
    smiles_col: str = "smiles_raw",
    recompute_sa_qed_logp: bool = False,
    existing_sa_col: Optional[str] = None,
    existing_qed_col: Optional[str] = None,
    existing_logp_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add common RDKit-based descriptors to the DataFrame:
      - smiles_canonical
      - is_valid
      - mw (molecular weight)
      - hbd (H-bond donors)
      - hba (H-bond acceptors)
      - sa, qed, logp (either copied from existing columns or recomputed)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with at least a column smiles_col.
    smiles_col : str
        Column name containing SMILES strings to canonicalize and featurize.
    recompute_sa_qed_logp : bool
        If True, recompute SA/QED/logP using RDKit functions.
        If False, try to use existing columns (existing_sa_col, existing_qed_col, existing_logp_col).
    existing_sa_col, existing_qed_col, existing_logp_col : Optional[str]
        Column names to copy SA/QED/LogP from if recompute_sa_qed_logp is False.

    Returns
    -------
    pd.DataFrame
        DataFrame with new descriptor columns added.
    """
    # Canonical SMILES
    df["smiles_canonical"] = df[smiles_col].apply(
        lambda s: canonic_smiles(s) if isinstance(s, str) and len(s) > 0 else None
    )

    # Build RDKit Mol objects once to reuse
    mols: List[Optional[Chem.Mol]] = [get_mol(smi) for smi in df["smiles_canonical"]]
    df["is_valid"] = [mol is not None for mol in mols]

    # MW, HBD, HBA (NaN for invalid molecules)
    df["mw"] = [
        float(Descriptors.MolWt(mol)) if mol is not None else np.nan
        for mol in mols
    ]
    df["hbd"] = [
        float(Lipinski.NumHDonors(mol)) if mol is not None else np.nan
        for mol in mols
    ]
    df["hba"] = [
        float(Lipinski.NumHAcceptors(mol)) if mol is not None else np.nan
        for mol in mols
    ]

    # SA, QED, LogP
    if (not recompute_sa_qed_logp) and all(
        col in df.columns
        for col in [existing_sa_col, existing_qed_col, existing_logp_col]
    ):
        # Copy from existing columns
        df["sa"] = df[existing_sa_col]          # type: ignore[index]
        df["qed"] = df[existing_qed_col]        # type: ignore[index]
        df["logp"] = df[existing_logp_col]      # type: ignore[index]
    else:
        # Recompute from RDKit Mols
        df["sa"] = sascorer_calculation(mols)
        df["qed"] = qed_calculation(mols)
        df["logp"] = logp_calculation(mols)

    return df

# -------------------------------------------------------------------------
# Generated dataset processing
# -------------------------------------------------------------------------

def process_single_generated_dataset(
    raw_path: Path,
    processed_path: Path,
    phase_label: str,
    target_label: str,
    csv_sep: str = DEFAULT_GENERATED_CSV_SEP,
) -> None:
    """
    Process a single raw generated dataset (Phase I/II/III for a given target)
    into a processed parquet file with standardized columns and descriptors.

    Parameters
    ----------
    raw_path : Path
        Path to the raw generation CSV file.
    processed_path : Path
        Path where the processed parquet file will be saved.
    phase_label : str
        Phase label (e.g., "I", "II", "III").
    target_label : str
        Human-readable target label (e.g., "AKT1", "CDK2").
    csv_sep : str
        Column separator used in the raw CSV file (default: tab).
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw generated file not found: {raw_path}")

    print(f"[INFO] Processing generated dataset: {raw_path.name} "
          f"(phase={phase_label}, target={target_label})")

    df = pd.read_csv(raw_path, sep=csv_sep)

    # Basic metadata
    df["source_type"] = "generated"
    df["phase"] = phase_label
    df["target_id"] = target_label

    # Protein CHEMBL ID and mapping to target symbol (if desired)
    df["protein_chembl_id"] = df["Protein_ID"]
    df["target_symbol"] = df["protein_chembl_id"].map(
        TARGET_CHEMBL_TO_SYMBOL
    ).fillna(df["protein_chembl_id"])

    # Normalize/rename key columns (keep original names too if you like)
    df["generated_selfies"] = df["Generated_SELFIES"]
    df["smiles_raw"] = df["Generated_SMILES"]
    df["encoder"] = df["Protein_Encoder"]
    df["model_name"] = df["Model_Name"]
    df["predicted_pchembl"] = df["Predicted_pChEMBL"]

    df["generation_temperature"] = df["Generation_Temperature"]
    df["generation_top_p"] = df["Generation_Top_p"]
    df["max_length"] = df["Max_Length"]
    df["batch_size"] = df["Batch_Size"]
    df["generation_timestamp"] = df["Generation_Timestamp"]

    df["test_similarity"] = df["Test_Similarity"]
    df["train_similarity"] = df["Train_Similarity"]

    # Add RDKit descriptors; SA/QED/LogP are already present in columns
    df = add_common_rdkit_descriptors(
        df,
        smiles_col="smiles_raw",
        recompute_sa_qed_logp=False,
        existing_sa_col="SA_Score",
        existing_qed_col="QED_Score",
        existing_logp_col="LogP_Score",
    )

    # Keep only the columns we care about (optional, but keeps files clean)
    keep_cols = [
        # metadata
        "source_type",
        "phase",
        "target_id",
        "protein_chembl_id",
        "target_symbol",
        "encoder",
        "model_name",
        "generation_temperature",
        "generation_top_p",
        "max_length",
        "batch_size",
        "generation_timestamp",
        # sequences
        "generated_selfies",
        "smiles_raw",
        "smiles_canonical",
        "is_valid",
        # scores & similarities
        "predicted_pchembl",
        "sa",
        "qed",
        "logp",
        "test_similarity",
        "train_similarity",
        # RDKit descriptors
        "mw",
        "hbd",
        "hba",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    print(f"[INFO] Saved processed generated dataset to: {processed_path}")

def process_all_generated_datasets(
    raw_generated_dir: Path,
    proc_generated_dir: Path,
    generated_sep: str = DEFAULT_GENERATED_CSV_SEP,
) -> None:
    """
    Run processing for all generated datasets with a fixed naming scheme.

    This assumes the following potential raw filenames inside raw_generated_dir:

        phase1_akt1.csv
        phase1_cdk2.csv
        phase2_akt1.csv
        phase2_cdk2.csv
        phase3_akt1.csv
        phase3_cdk2.csv

    For any of these that do NOT exist on disk, the script will just print a warning
    and skip them instead of raising an error.
    """
    config = [
        {"name": "phase1_akt1", "phase": "I",   "target_label": "AKT1"},
        {"name": "phase1_cdk2", "phase": "I",   "target_label": "CDK2"},
        {"name": "phase2_akt1", "phase": "II",  "target_label": "AKT1"},
        {"name": "phase2_cdk2", "phase": "II",  "target_label": "CDK2"},
        {"name": "phase3_akt1", "phase": "III", "target_label": "AKT1"},
        {"name": "phase3_cdk2", "phase": "III", "target_label": "CDK2"},
    ]

    any_processed = False

    for cfg in config:
        raw_path = raw_generated_dir / f"{cfg['name']}.csv"
        proc_path = proc_generated_dir / f"{cfg['name']}_processed.parquet"

        if not raw_path.exists():
            print(f"[WARN] Skipping {raw_path.name}: file not found in {raw_generated_dir}")
            continue

        process_single_generated_dataset(
            raw_path=raw_path,
            processed_path=proc_path,
            phase_label=cfg["phase"],
            target_label=cfg["target_label"],
            csv_sep=generated_sep,
        )
        any_processed = True

    if not any_processed:
        print("[WARN] No generated CSV files were found; nothing was processed.")

# -------------------------------------------------------------------------
# Papyrus / reference dataset processing
# -------------------------------------------------------------------------

def process_papyrus_reference(
    raw_path: Path,
    out_all_path: Path,
    out_akt1_path: Path,
    out_cdk2_path: Path,
    papyrus_sep: str = DEFAULT_PAPYRUS_CSV_SEP,
    drop_duplicate_smiles: bool = True,
) -> None:
    """
    Process Papyrus filtered dataset into:
      - papyrus_all_processed.parquet
      - akt1_ref_processed.parquet
      - cdk2_ref_processed.parquet

    This step:
      - decodes SELFIES to SMILES,
      - canonicalizes SMILES,
      - computes RDKit descriptors (SA, QED, LogP, MW, HBD, HBA),
      - optionally drops duplicate SMILES within each subset.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw Papyrus file not found: {raw_path}")

    print(f"[INFO] Processing Papyrus reference dataset: {raw_path.name}")

    df = pd.read_csv(raw_path, sep=papyrus_sep)

    # Basic metadata
    df["source_type"] = "ref_papyrus_all"
    df["target_chembl_id"] = df["Target_CHEMBL_ID"]
    df["target_symbol"] = df["target_chembl_id"].map(
        TARGET_CHEMBL_TO_SYMBOL
    ).fillna(df["target_chembl_id"])

    df["target_fasta"] = df["Target_FASTA"]
    df["selfies"] = df["Compound_SELFIES"]

    # Decode SELFIES -> SMILES
    df["smiles_raw"] = df["selfies"].apply(safe_decode_selfies)

    # Add RDKit descriptors; for Papyrus we recompute SA/QED/LogP
    df = add_common_rdkit_descriptors(
        df,
        smiles_col="smiles_raw",
        recompute_sa_qed_logp=True,
    )

    out_all_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_all_path, index=False)
    print(f"[INFO] Saved papyrus_all_processed to: {out_all_path}")

    # Build AKT1 and CDK2 subsets
    akt1_df = df[df["target_symbol"] == "AKT1"].copy()
    cdk2_df = df[df["target_symbol"] == "CDK2"].copy()

    akt1_df["source_type"] = "ref_akt1"
    cdk2_df["source_type"] = "ref_cdk2"

    if drop_duplicate_smiles:
        if "smiles_canonical" in akt1_df.columns:
            akt1_df = akt1_df.drop_duplicates(subset=["smiles_canonical"])
        if "smiles_canonical" in cdk2_df.columns:
            cdk2_df = cdk2_df.drop_duplicates(subset=["smiles_canonical"])

    akt1_df.to_parquet(out_akt1_path, index=False)
    cdk2_df.to_parquet(out_cdk2_path, index=False)
    print(f"[INFO] Saved akt1_ref_processed to: {out_akt1_path}")
    print(f"[INFO] Saved cdk2_ref_processed to: {out_cdk2_path}")

# -------------------------------------------------------------------------
# CLI & main
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the dataset preparation script.
    """
    parser = argparse.ArgumentParser(
        description="Prepare processed datasets (generated + Papyrus reference) "
                    "for Prot2Mol analysis."
    )

    default_raw_dir = REPO_ROOT / "data" / "raw"
    default_processed_dir = REPO_ROOT / "data" / "processed"

    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(default_raw_dir),
        help=f"Base directory for raw data (default: {default_raw_dir})",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(default_processed_dir),
        help=f"Base directory for processed data (default: {default_processed_dir})",
    )
    parser.add_argument(
        "--generated-sep",
        type=str,
        default=DEFAULT_GENERATED_CSV_SEP,
        help=f"CSV separator for generated datasets (default: '{DEFAULT_GENERATED_CSV_SEP}')",
    )
    parser.add_argument(
        "--papyrus-sep",
        type=str,
        default=DEFAULT_PAPYRUS_CSV_SEP,
        help=f"CSV separator for Papyrus dataset (default: '{DEFAULT_PAPYRUS_CSV_SEP}')",
    )
    parser.add_argument(
        "--papyrus-file",
        type=str,
        default=None,
        help="Path to Papyrus filtered CSV. "
             "If not provided, defaults to <raw-dir>/reference/papyrus_filtered.csv",
    )

    return parser.parse_args()

def main() -> None:
    """Entry point for the Stage 1 dataset preparation pipeline."""
    args = parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    proc_dir = Path(args.processed_dir).resolve()

    raw_generated_dir = raw_dir / "generated"
    raw_reference_dir = raw_dir / "reference"

    proc_generated_dir = proc_dir / "generated"
    proc_reference_dir = proc_dir / "reference"

    proc_generated_dir.mkdir(parents=True, exist_ok=True)
    proc_reference_dir.mkdir(parents=True, exist_ok=True)

    # Papyrus file path (either user-specified or default)
    papyrus_path = (
        Path(args.papyrus_file).resolve()
        if args.papyrus_file is not None
        else (raw_reference_dir / "papyrus_filtered.csv")
    )

    papyrus_all_proc_path = proc_reference_dir / "papyrus_all_processed.parquet"
    akt1_ref_proc_path = proc_reference_dir / "akt1_ref_processed.parquet"
    cdk2_ref_proc_path = proc_reference_dir / "cdk2_ref_processed.parquet"

    print("[INFO] Repository root   :", REPO_ROOT)
    print("[INFO] Raw data dir      :", raw_dir)
    print("[INFO] Processed data dir:", proc_dir)
    print("[INFO] Raw generated dir :", raw_generated_dir)
    print("[INFO] Raw reference dir :", raw_reference_dir)
    print("[INFO] Papyrus raw file  :", papyrus_path)

    print("[INFO] Starting Stage 1 dataset preparation...")

    # 1) Process generated datasets (Phase I/II/III, AKT1/CDK2)
    process_all_generated_datasets(
        raw_generated_dir=raw_generated_dir,
        proc_generated_dir=proc_generated_dir,
        generated_sep=args.generated_sep,
    )

    # 2) Process Papyrus reference dataset and its AKT1/CDK2 subsets
    process_papyrus_reference(
        raw_path=papyrus_path,
        out_all_path=papyrus_all_proc_path,
        out_akt1_path=akt1_ref_proc_path,
        out_cdk2_path=cdk2_ref_proc_path,
        papyrus_sep=args.papyrus_sep,
    )

    print("[INFO] Stage 1 dataset preparation completed.")

if __name__ == "__main__":
    main()
