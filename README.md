# Scaling Laws for SVG Transformers

A comprehensive study of neural scaling laws applied to SVG (Scalable Vector Graphics) generation using decoder-only GPT-style transformers. We train a family of 5 models (1.3M to 88M parameters), compare Standard Parameterization (SP) vs Maximal Update Parameterization (µP), and evaluate a best model on unconditional and conditional SVG generation.

For a detailed analysis of the results, architectures, and findings, please read the Final Report [FINAL_REPORT.md](FINAL_REPORT.md).

---

## Key Results

| Metric | Value |
|--------|-------|
| Model family | 5 sizes: Tiny (1.3M) → XL (88.1M) |
| Best model | Large (33.6M params, 10 epochs) |
| Test perplexity | **2.25** |
| XML validity | **71.4%** (15/21 samples) |
| SVG render rate | **52.4%** (11/21 samples) |
| Training time (best) | 59 min on A100 |

**Main findings**:
- Fixed learning rate scaling fails when depth varies across models (R²=0.18 power law fit)
- µP enables 33× higher learning rates and provides partial protection against instability
- SVG models achieve very low perplexity due to the structured, low-entropy nature of the domain

---

## Project Structure

```
├── config.py                    # Centralized configuration (paths, hyperparams)
├── model.py                     # GPT model (Standard Parameterization)
├── model_mup.py                 # GPT model (µP Parameterization)
├── train.py                     # Training loop (SP)
├── train_mup.py                 # Training loop (µP)
├── requirements.txt             # Python dependencies
│
├── scripts/                     # Numbered pipeline scripts (run in order)
│   ├── 01_download_data.py      # Download SVG datasets from HuggingFace
│   ├── 02_clean_normalize.py    # XML cleaning, normalization, deduplication
│   ├── 03_train_tokenizer.py    # Train Byte-Level BPE tokenizer (vocab=4096)
│   ├── 04_tokenize_split.py     # Tokenize corpus + train/val/test split
│   ├── 05_statistics.py         # Compute corpus statistics and render examples
│   ├── 06_lr_sweep.py           # SP learning rate sweep (Tiny model)
│   ├── 07_train_all_models.py   # Train all 5 SP models (1 epoch each)
│   ├── 08_scaling_analysis.py   # Power law fitting and scaling plots
│   ├── 09_mup_lr_sweep.py       # µP learning rate sweep
│   ├── 10_mup_train_all.py      # Train all 5 µP models (1 epoch each)
│   ├── 11_mup_comparison.py     # SP vs µP comparison plots and analysis
│   ├── 13_generate_samples.py   # Generate SVG samples from best model
│   └── 14_evaluate_model.py     # Evaluate: perplexity, XML/render validity, grids
│
├── utils/
│   ├── data_utils.py            # Dataset class and data loading
│   └── svg_utils.py             # SVG parsing, cleaning, rendering utilities
│
├── data/
│   ├── raw/                     # Downloaded SVG files (gitignored)
│   ├── cleaned/                 # Normalized SVG files (gitignored)
│   ├── tokenizer/               # Trained BPE tokenizer (svg_bpe_4096.json)
│   ├── splits/                  # train.npy, val.npy, test.npy
│   └── stats/                   # Corpus statistics, histograms, example renders
│
├── checkpoints/
│   ├── lr_sweep/                # SP LR sweep results and plot
│   ├── scaling/                 # SP scaling study (5 models) + training curves
│   ├── mup_lr_sweep/            # µP LR sweep results and plot
│   ├── mup_scaling/             # µP scaling study (5 models)
│   ├── comparison/              # SP vs µP comparison plots, extrapolation
│   └── best_model/              # Final 33.6M model weights (model.pt)
│
├── outputs/samples/             # Generated SVG files, render PNGs, eval metrics
│
├── colab_*.py / colab_*.ipynb   # Google Colab versions of GPU-heavy scripts
│
├── FINAL_REPORT.pdf             # Full project report (PDF)
└── REPORT_Part{1-4}.md          # Individual part reports (Markdown)
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for training; A100 recommended)
- GTK3 runtime (for CairoSVG rendering on Windows)

### Installation

```bash
# Clone the repository
git clone https://github.com/chirantan2003/Scaling-Laws-for-SVG-Transformers.git
cd Scaling-Laws-for-SVG-Transformers

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or
.\venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Windows-specific: GTK3 for CairoSVG

CairoSVG requires native Cairo libraries. On Windows, install MSYS2 and GTK3:

```powershell
# Install MSYS2 from https://www.msys2.org/
# Then in MSYS2 terminal:
pacman -S mingw-w64-x86_64-gtk3
```

The evaluation script automatically adds `C:\msys64\mingw64\bin` to the DLL search path.

---

## Usage

### Full Pipeline (Sequential)

Run scripts in numerical order. Steps 01–05 run on CPU; steps 06+ require a GPU.

```bash
# Data preparation (CPU)
python scripts/01_download_data.py
python scripts/02_clean_normalize.py
python scripts/03_train_tokenizer.py
python scripts/04_tokenize_split.py
python scripts/05_statistics.py

# SP scaling study (GPU)
python scripts/06_lr_sweep.py
python scripts/07_train_all_models.py
python scripts/08_scaling_analysis.py

# µP scaling study (GPU)
python scripts/09_mup_lr_sweep.py
python scripts/10_mup_train_all.py
python scripts/11_mup_comparison.py

# Best model training + evaluation (GPU)
# (Use colab_12_train_best_model.py for extended training on Colab)
python scripts/13_generate_samples.py
python scripts/14_evaluate_model.py
```

### Google Colab

GPU-intensive scripts (µP sweeps, best model training) have Colab notebook equivalents:

| Notebook | Purpose |
|----------|---------|
| `colab_09_mup_lr_sweep.ipynb` | µP learning rate sweep |
| `colab_10_mup_train_all.ipynb` | Train all µP models |
| `colab_11_mup_comparison.ipynb` | SP vs µP analysis |
| `colab_12_train_best_model.ipynb` | Train Large model for 10 epochs |

Upload the `data/` directory to your Colab Drive and update paths in the notebook.

---

## Model Architecture

Decoder-only GPT with weight-tied embeddings, no bias, and Flash Attention support:

| Name | Params | d_model | Layers | Heads | d_ff |
|------|--------|---------|--------|-------|------|
| Tiny | 1.3M | 128 | 4 | 4 | 512 |
| Small | 3.4M | 192 | 6 | 6 | 768 |
| Medium | 12.2M | 384 | 6 | 6 | 1,536 |
| Large | 33.6M | 512 | 10 | 8 | 2,048 |
| XL | 88.1M | 768 | 12 | 12 | 3,072 |

---

## Dataset

284,548 SVGs from three HuggingFace StarVector datasets, preprocessed to 273,400 files (150.4M tokens):

| Source | Files |
|--------|------:|
| `starvector/svg-icons-simple` | 80,434 |
| `starvector/svg-emoji-simple` | 4,114 |
| `starvector/svg-fonts-simple` | 200,000 |

**Tokenizer**: Byte-Level BPE, vocabulary size 4,096, context window 2,048 tokens.

---

## References

- Kaplan et al. (2020). *Scaling Laws for Neural Language Models*. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models*. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Karpathy (2022). [nanoGPT](https://github.com/karpathy/nanoGPT)

---
