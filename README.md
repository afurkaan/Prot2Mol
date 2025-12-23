# 🧬 Prot2Mol

> **De novo drug design using protein language models and generative AI.**

**Prot2Mol** is a state-of-the-art generative model designed to create novel drug candidates tailored to specific protein targets. By leveraging the power of deep learning and protein language models (like ESM and ProtT5), Prot2Mol translates protein embeddings into molecular structures (represented as SELFIES), enabling the discovery of new potential therapeutics for drug-resistant diseases.

## ✨ Key Features

-   **Target-Specific Generation**: Generates molecules conditioned on the specific physical properties of target proteins.
-   **Advanced Architectures**: Utilizes Transformer-based architectures (GPT-2) with Cross-Attention mechanisms.
-   **Robust Representation**: Uses SELFIES for 100% valid molecular generation.
-   **Multi-Model Support**: Compatible with various protein embeddings including ESM-2, ESM-3, ProtT5, and AlphaFold2.
-   **Dual Mode**: Supports both **De Novo Molecule Generation** and **pChEMBL Prediction** for existing compounds.

## 🚀 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/atabeyunlu/Prot2Mol.git
    cd Prot2Mol
    ```

2.  **Install Dependencies**
    Ensure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

Prot2Mol provides a unified interface for both training and molecule generation.

### 1. Pre-training
Train the model on a large dataset of protein-molecule pairs.

```bash
python prot2mol/pretrain.py \
    --selfies_path data/your_dataset.csv \
    --prot_emb_model esm3 \
    --epoch 30 \
    --train_batch_size 32
```

### 2. Inference & Generation

The `produce_molecules.py` script is your main entry point for using the trained model. It supports two modes: `generation` and `prediction`.

#### 🧪 Mode 1: Molecule Generation
Generate novel molecules for a specific target protein.

```bash
python prot2mol/produce_molecules.py \
    --mode generation \
    --model_file ./saved_models/your_model \
    --prot_emb_model esm2 \
    --prot_id CHEMBL4282 \
    --num_samples 1000 \
    --output_file ./results/generated_mols.csv
```

**Key Arguments:**
-   `--mode generation`: Selects generation mode.
-   `--model_file`: Path to your trained model.
-   `--prot_id`: target ID (e.g. ChEMBL ID).
-   `--num_samples`: How many molecules to generate.

#### 🔮 Mode 2: pChEMBL Prediction
Predict the binding affinity (pChEMBL) of existing molecules against a target protein.

```bash
python prot2mol/produce_molecules.py \
    --mode prediction \
    --model_file ./saved_models/your_model \
    --input_molecules ./data/candidates.csv \
    --output_file ./results/predictions.csv
```

**Key Arguments:**
-   `--mode prediction`: Selects prediction mode.
-   `--input_molecules`: CSV file containing molecules (must have `smiles` or `selfies` column) and optionally `Target_FASTA` if not inferable.

## 📂 Project Structure

```
Prot2Mol/
├── prot2mol/               # Core source code
│   ├── model.py            # Model architecture definitions
│   ├── pretrain.py         # Training script
│   ├── produce_molecules.py # Inference and generation script
│   └── ...
├── data_processing/        # Scripts for data preparation (Embeddings, etc.)
├── data/                   # Data storage
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## 📜 Citation

If you use **Prot2Mol** in your research, please cite:

```bibtex
Ünlü, A., & Çevrim, E., & Doğan, T. (2024). Prot2Mol: Target based molecule generation using protein embeddings and SELFIES molecule representation. GitHub. https://github.com/HUBioDataLab/Prot2Mol
```
