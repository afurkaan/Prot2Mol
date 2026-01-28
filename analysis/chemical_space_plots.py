#!/usr/bin/env python
"""
Chemical space visualization for a SINGLE generated set.

Each run plots:
    - Papyrus background (50k random subset)
    - All inhibitors for a given target (AKT1 / CDK2)
    - Generated molecules from ONE phase (I/II/III),
      for that target, optionally filtered by encoder.

Usage examples:

    # AKT1, Phase III, ProtT5
    python analysis/chemical_space_plots.py \
        --target AKT1 \
        --phase 3 \
        --encoder prot_t5 \
        --include-tsne

    # CDK2, Phase I, all encoders together
    python analysis/chemical_space_plots.py \
        --target CDK2 \
        --phase I

Output:
    plots/chemical_space/umap_AKT1_phaseIII_prot_t5.png
    plots/chemical_space/tsne_AKT1_phaseIII_prot_t5.png (if --include-tsne)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# nice paper-style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

# -------------------------------------------------------------------------
# Import project utilities
# -------------------------------------------------------------------------

# add repo root to path (as you requested)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prot2mol.utils_fps import generate_morgan_fingerprints_parallel  # type: ignore

try:
    import umap  # umap-learn
except ImportError as e:
    raise ImportError("Please install 'umap-learn' for UMAP.") from e

try:
    from sklearn.manifold import TSNE
except ImportError as e:
    raise ImportError("Please install 'scikit-learn' for t-SNE.") from e

REPO_ROOT = Path(__file__).resolve().parents[1]


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    print(f"[INFO] Loading: {path}")
    return pd.read_parquet(path)


def normalize_phase(phase_str: str) -> Tuple[str, str]:
    """
    Normalize user phase arg to:
        - numeric: "1" / "2" / "3"   (for filenames)
        - roman:   "I" / "II" / "III" (for labels)
    """
    s = phase_str.strip().upper()
    if s in {"1", "I", "PHASE1", "PHASE_I"}:
        return "1", "I"
    if s in {"2", "II", "PHASE2", "PHASE_II"}:
        return "2", "II"
    if s in {"3", "III", "PHASE3", "PHASE_III"}:
        return "3", "III"
    raise ValueError(f"Unrecognized phase: {phase_str}")


def load_background_smiles(
    processed_ref_dir: Path, n_background: int, random_state: int
) -> List[str]:
    """Sample Papyrus background subset."""
    all_path = processed_ref_dir / "papyrus_all_processed.parquet"
    df = _load_parquet(all_path)

    if "is_valid" in df.columns:
        df = df[df["is_valid"]]

    smiles = df["smiles_canonical"].dropna().astype(str)
    if len(smiles) <= n_background:
        print(
            f"[WARN] Requested {n_background} background; "
            f"only {len(smiles)} available. Using all."
        )
        subset = smiles
    else:
        subset = smiles.sample(n=n_background, random_state=random_state)

    print(f"[INFO] Background molecules: {len(subset)}")
    return subset.tolist()


def load_inhibitor_smiles(processed_ref_dir: Path, target: str) -> List[str]:
    """Load all inhibitors for a given target (AKT1 / CDK2)."""
    target_lower = target.lower()
    ref_path = processed_ref_dir / f"{target_lower}_ref_processed.parquet"
    df = _load_parquet(ref_path)

    if "is_valid" in df.columns:
        df = df[df["is_valid"]]

    smiles = df["smiles_canonical"].dropna().astype(str)
    print(f"[INFO] {target} inhibitors: {len(smiles)} molecules")
    return smiles.tolist()


def load_generated_smiles_single(
    processed_gen_dir: Path,
    target: str,
    phase_num: str,
    encoder: str = None,
    max_generated: int = None,
) -> List[str]:
    """
    Load generated molecules for ONE phase + ONE target.
    Optionally filter by encoder; optionally subsample.
    """
    target_lower = target.lower()
    path = processed_gen_dir / f"phase{phase_num}_{target_lower}_processed.parquet"
    df = _load_parquet(path)

    mask = pd.Series(True, index=df.index)

    # safety: filter again by target / encoder / validity
    if "target_id" in df.columns:
        mask &= df["target_id"].astype(str).str.upper() == target.upper()
    if encoder is not None and "encoder" in df.columns:
        mask &= df["encoder"].astype(str) == encoder
    if "is_valid" in df.columns:
        mask &= df["is_valid"]

    df = df[mask]
    smiles = df["smiles_canonical"].dropna().astype(str)

    if max_generated is not None and len(smiles) > max_generated:
        smiles = smiles.sample(n=max_generated, random_state=42)

    print(
        f"[INFO] Generated set – target={target}, phase={phase_num}, "
        f"encoder={encoder if encoder else 'ALL'}, "
        f"mols={len(smiles)}"
    )
    return smiles.tolist()


def build_fps_and_labels(
    bg_smiles: List[str],
    inh_smiles: List[str],
    gen_smiles: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build fingerprint matrix + labels for three datasets:
      0: Background
      1: Reference inhibitors
      2: Generated (single set)
    """
    datasets = [
        ("Background", bg_smiles),
        ("Reference inhibitors", inh_smiles),
        ("Generated", gen_smiles),
    ]

    smiles_all: List[str] = []
    labels_idx: List[int] = []

    for idx, (_, smiles) in enumerate(datasets):
        smiles_all.extend(smiles)
        labels_idx.extend([idx] * len(smiles))

    print(f"[INFO] Total molecules in embedding: {len(smiles_all)}")

    fps = generate_morgan_fingerprints_parallel(
        smiles_all, radius=2, nBits=1024, n_jobs=None
    )
    labels_idx_arr = np.array(labels_idx, dtype=np.int32)
    label_names = [name for name, _ in datasets]

    return fps, labels_idx_arr, label_names


def compute_umap(fps: np.ndarray, random_state: int) -> np.ndarray:
    """2D UMAP (Dice distance)."""
    reducer = umap.UMAP(
        n_neighbors=50,
        min_dist=0.8,
        metric="dice",
        random_state=random_state,
    )
    return reducer.fit_transform(fps)


def compute_tsne(fps: np.ndarray, random_state: int) -> np.ndarray:
    """2D t-SNE (Jaccard distance over binary fingerprints)."""
    tsne = TSNE(
        n_components=2,
        metric="jaccard",
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    return tsne.fit_transform(fps)


def plot_embedding(
    coords: np.ndarray,
    labels_idx: np.ndarray,
    label_names: List[str],
    target: str,
    phase_roman: str,
    encoder: str,
    method: str,
    output_dir: Path,
) -> None:
    """Scatter plot for one embedding (UMAP or t-SNE)."""
    encoder_label = encoder.upper() if encoder is not None else "ALL"
    method_lower = method.lower()

    # colors similar to DrugGEN style
    color_bg = "#B0B0B0"     # light grey
    color_ref = "#F6A800"    # orange
    color_gen = "#1f77b4"    # blue

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, name in enumerate(label_names):
        mask = labels_idx == idx
        if not np.any(mask):
            continue
        x = coords[mask, 0]
        y = coords[mask, 1]

        if name == "Background":
            ax.scatter(
                x,
                y,
                s=3,
                c=color_bg,
                alpha=0.15,
                linewidth=0,
                label="Papyrus background (50k)",
            )
        elif name == "Reference inhibitors":
            ax.scatter(
                x,
                y,
                s=12,
                c=color_ref,
                alpha=0.9,
                linewidth=0.3,
                edgecolors="black",
                label=f"{target} inhibitors",
            )
        else:  # Generated
            gen_label = f"Generated – Phase {phase_roman}"
            if encoder is not None:
                gen_label += f" ({encoder_label})"
            ax.scatter(
                x,
                y,
                s=10,
                c=color_gen,
                alpha=0.9,
                linewidth=0,
                label=gen_label,
            )

    if method_lower == "umap":
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
    else:
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")

    title = f"{method} projection – {target}, Phase {phase_roman}"
    if encoder is not None:
        title += f" ({encoder_label})"
    ax.set_title(title)

    ax.legend(
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fname_parts = [
        method_lower,
        target.upper(),
        f"phase{phase_roman}",
    ]
    if encoder is not None:
        fname_parts.append(encoder)
    fname = "_".join(fname_parts) + ".png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] Saved {method} plot to: {out_path}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UMAP / t-SNE chemical space plot for a SINGLE generated set."
    )

    default_proc_dir = REPO_ROOT / "data" / "processed"
    default_out_dir = REPO_ROOT / "plots" / "chemical_space"

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target to analyze, e.g. AKT1 or CDK2.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        help="Phase of generated set: 1 / 2 / 3 or I / II / III.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Optional encoder filter (e.g. 'prot_t5', 'esm2'). "
             "If omitted, all encoders in that phase file are used.",
    )
    parser.add_argument(
        "--n-background",
        type=int,
        default=50000,
        help="Number of Papyrus background molecules (default: 50000).",
    )
    parser.add_argument(
        "--max-generated",
        type=int,
        default=None,
        help="Optional cap on number of generated molecules "
             "from this phase (e.g. 5000). Default: use all.",
    )
    parser.add_argument(
        "--include-tsne",
        action="store_true",
        help="Also compute and save t-SNE plot.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(default_proc_dir),
        help=f"Base processed data dir (default: {default_proc_dir}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_out_dir),
        help=f"Directory to save plots (default: {default_out_dir}).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling / embeddings (default: 42).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    proc_dir = Path(args.processed_dir).resolve()
    processed_ref_dir = proc_dir / "reference"
    processed_gen_dir = proc_dir / "generated"
    output_dir = Path(args.output_dir).resolve()

    target = args.target.upper()
    phase_num, phase_roman = normalize_phase(args.phase)
    encoder = args.encoder

    print("[INFO] Repository root :", REPO_ROOT)
    print("[INFO] Processed dir   :", proc_dir)
    print("[INFO] Target          :", target)
    print("[INFO] Phase           :", phase_roman)
    print("[INFO] Encoder filter  :", encoder if encoder else "ALL")

    # 1) Load datasets
    bg_smiles = load_background_smiles(
        processed_ref_dir,
        n_background=args.n_background,
        random_state=args.random_state,
    )
    inh_smiles = load_inhibitor_smiles(processed_ref_dir, target)
    gen_smiles = load_generated_smiles_single(
        processed_gen_dir,
        target=target,
        phase_num=phase_num,
        encoder=encoder,
        max_generated=args.max_generated,
    )

    if len(gen_smiles) == 0:
        print("[WARN] No generated molecules found for this configuration.")
        return

    # 2) Fingerprints + labels
    fps, labels_idx, label_names = build_fps_and_labels(
        bg_smiles, inh_smiles, gen_smiles
    )

    # 3) UMAP
    print("[INFO] Computing UMAP embedding...")
    umap_coords = compute_umap(fps, random_state=args.random_state)
    plot_embedding(
        coords=umap_coords,
        labels_idx=labels_idx,
        label_names=label_names,
        target=target,
        phase_roman=phase_roman,
        encoder=encoder,
        method="UMAP",
        output_dir=output_dir,
    )

    # 4) t-SNE (optional)
    if args.include_tsne:
        print("[INFO] Computing t-SNE embedding (may be slow)...")
        tsne_coords = compute_tsne(fps, random_state=args.random_state)
        plot_embedding(
            coords=tsne_coords,
            labels_idx=labels_idx,
            label_names=label_names,
            target=target,
            phase_roman=phase_roman,
            encoder=encoder,
            method="tSNE",
            output_dir=output_dir,
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

