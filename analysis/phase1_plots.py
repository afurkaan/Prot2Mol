#!/usr/bin/env python
"""
Phase 1 analysis: 1D distribution plots for generated vs. reference molecules.

Goal:
- Check whether generated molecules physically resemble the known inhibitors
  in the reference dataset (AKT1 or CDK2).
- Properties to analyze:
    - Molecular Weight (MW)
    - LogP (Lipophilicity)
    - Hydrogen Bond Donors (HBD)
    - Hydrogen Bond Acceptors (HBA)
    - QED
    - SA (Synthetic Accessibility)

Visualization:
- Overlay distribution plots (histograms or KDE curves) for:
    - Multiple generated sets (e.g., Phase I, II, III for a given target)
    - Reference training dataset (e.g., akt1_ref_processed.parquet)

Usage example:

    python analysis/phase1_plots.py \
        --generated data/processed/generated/phase1_akt1_processed.parquet \
        --generated data/processed/generated/phase2_akt1_processed.parquet \
        --generated data/processed/generated/phase3_akt1_processed.parquet \
        --reference data/processed/reference/akt1_ref_processed.parquet \
        --target AKT1 \
        --properties mw logp hbd hba qed sa \
        --plot-type kde
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# clean, paper-friendly style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# Property configuration: column name in processed parquet and pretty label
PROPERTY_INFO: Dict[str, Dict[str, str]] = {
    "mw":   {"col": "mw",   "label": "Molecular weight (Da)"},
    "logp": {"col": "logp", "label": "LogP"},
    "hbd":  {"col": "hbd",  "label": "H-bond donors"},
    "hba":  {"col": "hba",  "label": "H-bond acceptors"},
    "qed":  {"col": "qed",  "label": "QED score"},
    "sa":   {"col": "sa",   "label": "SA score"},
}

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def infer_generated_label(df: pd.DataFrame, fallback: str) -> str:
    """Label generated dataset as 'Phase X (TARGET)' if possible."""
    phase = None
    target = None

    if "phase" in df.columns and df["phase"].notna().any():
        vals = df["phase"].dropna().unique()
        if len(vals) == 1:
            phase = str(vals[0])

    if "target_id" in df.columns and df["target_id"].notna().any():
        vals = df["target_id"].dropna().unique()
        if len(vals) == 1:
            target = str(vals[0])

    if phase is not None and target is not None:
        return f"Phase {phase} ({target})"
    if phase is not None:
        return f"Phase {phase}"
    return fallback

def infer_reference_label(df_ref: pd.DataFrame, fallback: str = "Reference") -> str:
    """Label reference dataset as 'Reference (TARGET)' if possible."""
    label = fallback
    if "target_symbol" in df_ref.columns and df_ref["target_symbol"].notna().any():
        vals = df_ref["target_symbol"].dropna().unique()
        if len(vals) == 1:
            label = f"Reference ({vals[0]})"
    return label

def parse_properties(props: List[str]) -> List[str]:
    props = [p.lower() for p in props]
    if "all" in props:
        return list(PROPERTY_INFO.keys())

    unknown = [p for p in props if p not in PROPERTY_INFO]
    if unknown:
        raise ValueError(
            f"Unknown properties requested: {unknown}. "
            f"Valid options: {list(PROPERTY_INFO.keys())} or 'all'."
        )
    return props

def filter_df_for_target_and_encoder(
    df: pd.DataFrame,
    target: Optional[str],
    encoder: Optional[str],
    is_reference: bool = False,
) -> pd.DataFrame:
    """
    Filter a dataframe by target and encoder if those filters are provided.

    - For generated sets:
        * target name (AKT1/CDK2) -> column 'target_id'
        * CHEMBL id (CHEMBL4282)   -> column 'protein_chembl_id'
    - For reference sets:
        * target name -> 'target_symbol'
        * CHEMBL id   -> 'target_chembl_id'
    """
    filtered = df.copy()

    if target is not None:
        t_upper = target.upper()
        is_chembl = t_upper.startswith("CHEMBL")

        col_to_use = None
        if is_reference:
            if is_chembl and "target_chembl_id" in filtered.columns:
                col_to_use = "target_chembl_id"
            elif "target_symbol" in filtered.columns:
                col_to_use = "target_symbol"
            elif "target_id" in filtered.columns:
                col_to_use = "target_id"
        else:
            if is_chembl and "protein_chembl_id" in filtered.columns:
                col_to_use = "protein_chembl_id"
            elif "target_id" in filtered.columns:
                col_to_use = "target_id"
            elif "target_symbol" in filtered.columns:
                col_to_use = "target_symbol"

        if col_to_use is not None:
            filtered = filtered[
                filtered[col_to_use].astype(str).str.upper() == t_upper
            ]

    if encoder is not None and not is_reference:
        if "encoder" in filtered.columns:
            filtered = filtered[filtered["encoder"] == encoder]

    return filtered

def ensure_non_empty(df: pd.DataFrame, label: str, prop_col: str) -> np.ndarray:
    """Return non-NaN values for the given column (as float array)."""
    if prop_col not in df.columns:
        print(f"[WARN] Column '{prop_col}' not found in dataset '{label}'. Skipping.")
        return np.array([])

    values = df[prop_col].to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        print(f"[WARN] No valid values for '{prop_col}' in dataset '{label}'. Skipping.")
    return values

# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def _compute_x_limits(arrays: List[np.ndarray]) -> Optional[tuple]:
    """
    Compute x-limits from all value arrays.

    We use a wide percentile range (0.5–99.5) and a bit more padding so that
    the right tail of the KDE can smoothly decay towards zero instead of
    being cut in the middle of the slope.
    """
    non_empty = [a for a in arrays if a.size > 0]
    if not non_empty:
        return None

    concat = np.concatenate(non_empty)
    q_low, q_high = np.percentile(concat, [0.5, 99.5])

    if q_low == q_high:
        # fallback if all values identical
        return q_low - 1.0, q_high + 1.0

    # slightly larger padding than before
    pad = 0.10 * (q_high - q_low)
    return q_low - pad, q_high + pad

def plot_property_distribution(
    prop_key: str,
    reference: Dict,
    generated_list: List[Dict],
    plot_type: str,
    output_dir: Path,
    target: Optional[str] = None,
    encoder: Optional[str] = None,
    bins: int = 60,
) -> None:
    """
    Plot overlayed 1D distributions (hist or kde) for a single property.

    KDE uses seaborn.kdeplot, with filled reference and line-only generated sets.
    Legend labels include mean value like DrugGEN figures.
    """
    info = PROPERTY_INFO[prop_key]
    col = info["col"]
    xlabel = info["label"]

    ref_vals = reference["values"]
    ref_label = reference["label"]

    # Prepare all values for x-limits
    all_vals = [ref_vals] + [g["values"] for g in generated_list]
    x_limits = _compute_x_limits(all_vals)

    fig, ax = plt.subplots(figsize=(8, 6))

    # ---- reference ----
    if ref_vals.size > 0:
        mean_ref = float(ref_vals.mean())
        label_ref = f"{ref_label} ({mean_ref:.2f})"

        if plot_type == "hist":
            ax.hist(
                ref_vals,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.4,
                label=label_ref,
            )
        elif plot_type == "kde":
            sns.kdeplot(
            x=ref_vals,
            ax=ax,
            fill=True,
            alpha=0.35,
            linewidth=2,
            label=label_ref,
            bw_adjust=1.2,
            gridsize=512,
            cut=0,
        )

        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")

    # ---- generated sets ----
    for gen in generated_list:
        vals = gen["values"]
        if vals.size == 0:
            continue
        mean_gen = float(vals.mean())
        label_gen = f"{gen['label']} ({mean_gen:.2f})"

        if plot_type == "hist":
            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                alpha=0.9,
                label=label_gen,
            )
        elif plot_type == "kde":
            sns.kdeplot(
            x=vals,
            ax=ax,
            fill=True,
            linewidth=2,
            label=label_gen,
            bw_adjust=1.2,
            gridsize=512,
            cut=0,
        )

    # ---- axis labels / title / legend ----
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    title_parts = [f"Distribution of {xlabel}"]
    if target is not None:
        title_parts.append(f"(Target: {target})")
    if encoder is not None:
        title_parts.append(f"[Encoder: {encoder}]")
    ax.set_title(" ".join(title_parts))

    if x_limits is not None:
        ax.set_xlim(*x_limits)

    ax.legend(frameon=True)

    fig.tight_layout()

    # filename
    target_part = f"_target-{target}" if target is not None else ""
    encoder_part = f"_encoder-{encoder}" if encoder is not None else ""
    fname = f"{prop_key}_{plot_type}{target_part}{encoder_part}.png"
    out_path = output_dir / fname

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] Saved plot: {out_path} (property={col}, type={plot_type})")

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: plot 1D distributions (generated vs reference) "
                    "for MW, LogP, HBD/HBA, QED, SA."
    )

    default_output_dir = REPO_ROOT / "plots" / "phase1"

    parser.add_argument(
        "--generated",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to processed generated parquet files "
             "(e.g., phase1_akt1_processed.parquet phase2_akt1_processed.parquet ...).",
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to processed reference parquet file "
             "(e.g., akt1_ref_processed.parquet or cdk2_ref_processed.parquet).",
    )
    parser.add_argument(
        "--properties",
        type=str,
        nargs="+",
        default=["mw", "logp", "hbd", "hba", "qed", "sa"],
        help="Properties to plot. Options: mw logp hbd hba qed sa or 'all'. "
             "Default: mw logp hbd hba qed sa",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["hist", "kde"],
        default="kde",
        help="Type of 1D distribution plot: 'hist' or 'kde' (default: kde).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help=f"Directory to save plots (default: {default_output_dir})",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional target filter (e.g., AKT1, CDK2 or CHEMBL4282).",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Optional encoder filter for generated datasets "
             "(e.g., 'prot_t5' or 'esm2').",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of bins for histogram (default: 60).",
    )

    return parser.parse_args()

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    # 1) reference
    ref_path = Path(args.reference).resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference parquet file not found: {ref_path}")

    print(f"[INFO] Loading reference dataset: {ref_path}")
    df_ref = pd.read_parquet(ref_path)
    df_ref = filter_df_for_target_and_encoder(df_ref, args.target, encoder=None, is_reference=True)
    ref_label = infer_reference_label(df_ref)
    print(f"[INFO] Reference label: {ref_label}, rows after filtering: {len(df_ref)}")

    # 2) generated
    generated_datasets: List[Dict] = []
    for gpath_str in args.generated:
        gpath = Path(gpath_str).resolve()
        if not gpath.exists():
            print(f"[WARN] Generated parquet not found, skipping: {gpath}")
            continue

        print(f"[INFO] Loading generated dataset: {gpath}")
        df_gen = pd.read_parquet(gpath)
        df_gen = filter_df_for_target_and_encoder(
            df_gen,
            target=args.target,
            encoder=args.encoder,
            is_reference=False,
        )

        stem = gpath.stem
        if stem.endswith("_processed"):
            stem = stem[:-10]
        gen_label = infer_generated_label(df_gen, fallback=stem)

        print(f"[INFO] Generated label: {gen_label}, rows after filtering: {len(df_gen)}")

        generated_datasets.append({"label": gen_label, "df": df_gen})

    if len(generated_datasets) == 0:
        print("[WARN] No generated datasets loaded; nothing to plot.")
        return

    # 3) properties
    props = parse_properties(args.properties)
    print(f"[INFO] Properties to plot: {props}")
    print(f"[INFO] Plot type: {args.plot_type}")
    if args.target:
        print(f"[INFO] Target filter: {args.target}")
    if args.encoder:
        print(f"[INFO] Encoder filter: {args.encoder}")
    print(f"[INFO] Output directory: {output_dir}")

    # 4) plot each property
    for prop_key in props:
        col = PROPERTY_INFO[prop_key]["col"]

        ref_vals = ensure_non_empty(df_ref, ref_label, col)
        ref_dict = {"label": ref_label, "values": ref_vals}

        gen_value_dicts: List[Dict] = []
        for item in generated_datasets:
            label = item["label"]
            df_gen = item["df"]
            vals = ensure_non_empty(df_gen, label, col)
            if vals.size > 0:
                gen_value_dicts.append({"label": label, "values": vals})

        if (ref_vals.size == 0) and (len(gen_value_dicts) == 0):
            print(f"[WARN] No valid values for property '{col}' across all datasets; skipping plot.")
            continue

        plot_property_distribution(
            prop_key=prop_key,
            reference=ref_dict,
            generated_list=gen_value_dicts,
            plot_type=args.plot_type,
            output_dir=output_dir,
            target=args.target,
            encoder=args.encoder,
            bins=args.bins,
        )

    print("[INFO] Phase 1 distribution plotting completed.")

if __name__ == "__main__":
    main()

