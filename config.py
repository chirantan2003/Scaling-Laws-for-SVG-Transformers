"""
Centralized configuration for the SVG Scaling Laws project.
All paths, hyperparameters, and thresholds are defined here.
"""
import os
from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
SPLITS_DIR = DATA_DIR / "splits"
STATS_DIR = DATA_DIR / "stats"
EXAMPLES_DIR = STATS_DIR / "examples"

# Create all directories
for d in [RAW_DIR, CLEANED_DIR, TOKENIZER_DIR, SPLITS_DIR, STATS_DIR, EXAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Dataset Configuration
# ============================================================
DATASETS = {
    "svg-icons-simple": {
        "hf_name": "starvector/svg-icons-simple",
        "svg_column": "Svg",           # Column containing SVG code
        "filename_column": "Filename",  # Column containing filename (if any)
        "subsample": None,             # Use all (~89K)
        "primary": True,
    },
    "svg-emoji-simple": {
        "hf_name": "starvector/svg-emoji-simple",
        "svg_column": "svg",
        "filename_column": None,
        "subsample": None,             # Use all (~14.5 MB, relatively small)
        "primary": False,
    },
    "svg-fonts-simple": {
        "hf_name": "starvector/svg-fonts-simple",
        "svg_column": "svg",
        "filename_column": None,
        "subsample": 200_000,          # Subsample from the 2.38 GB dataset
        "primary": False,
    },
}

# ============================================================
# SVG Cleaning / Normalization
# ============================================================
MIN_SVG_LENGTH = 50         # Minimum character length (discard trivial SVGs)
MAX_SVG_LENGTH = 50_000     # Maximum character length (pre-tokenization safety)
COORD_PRECISION = 1         # Round floating-point coordinates to this many decimals
CANONICALIZE_ATTRS = True   # Sort attributes alphabetically within each element
VALIDATE_RENDER = False     # Attempt CairoSVG render validation (slower but thorough)

# ============================================================
# Tokenizer
# ============================================================
VOCAB_SIZE = 4096
TOKENIZER_PATH = TOKENIZER_DIR / "svg_bpe_4096.json"
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2

# ============================================================
# Tokenization & Splitting
# ============================================================
MAX_SEQ_LENGTH = 2048       # Maximum token sequence length per SVG
TRAIN_RATIO = 0.98
VAL_RATIO = 0.01
TEST_RATIO = 0.01
SPLIT_SEED = 42             # Random seed for reproducible splits

# ============================================================
# GPU Configuration
# ============================================================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[config] Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"[config] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[config] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
