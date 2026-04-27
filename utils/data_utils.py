"""
Data loading, splitting, and serialization utilities.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def load_svg_files(directory: Path) -> Dict[str, str]:
    """Load all SVG files from a directory.
    
    Returns:
        dict mapping filename -> svg_string
    """
    svgs = {}
    svg_files = sorted(directory.glob("*.svg"))
    for f in svg_files:
        try:
            svgs[f.name] = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  [warn] Could not read {f.name}: {e}")
    return svgs


def save_svg_file(svg_string: str, filepath: Path) -> bool:
    """Save an SVG string to a file."""
    try:
        filepath.write_text(svg_string, encoding="utf-8")
        return True
    except Exception as e:
        print(f"  [warn] Could not write {filepath}: {e}")
        return False


def split_files(
    filenames: List[str],
    train_ratio: float = 0.98,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split filenames into train/val/test sets by file.
    
    Returns:
        (train_files, val_files, test_files)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    rng = np.random.RandomState(seed)
    shuffled = list(filenames)
    rng.shuffle(shuffled)
    
    n = len(shuffled)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test
    
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    
    return train, val, test


def save_token_array(tokens: List[int], filepath: Path):
    """Save a list of token IDs as a numpy array."""
    arr = np.array(tokens, dtype=np.uint16)  # uint16 supports up to 65535 vocab
    np.save(filepath, arr)
    print(f"  [save] {filepath.name}: {len(tokens):,} tokens, {arr.nbytes / 1e6:.1f} MB")


def load_token_array(filepath: Path) -> np.ndarray:
    """Load a token array from a numpy file."""
    return np.load(filepath)


def save_json(data: dict, filepath: Path):
    """Save a dictionary as a JSON file."""
    filepath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_json(filepath: Path) -> dict:
    """Load a JSON file."""
    return json.loads(filepath.read_text(encoding="utf-8"))
