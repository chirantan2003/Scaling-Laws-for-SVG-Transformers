import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

import config


def download_dataset(name: str, ds_config: dict):
    """Download a single dataset and save SVGs as individual files."""
    print(f"\n{'='*60}")
    print(f"Downloading: {ds_config['hf_name']}")
    print(f"{'='*60}")
    
    output_dir = config.RAW_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    existing = list(output_dir.glob("*.svg"))
    if len(existing) > 100:
        print(f"  [skip] Already have {len(existing)} SVGs in {output_dir}")
        return len(existing)
    
    # Load dataset from HuggingFace
    print(f"  Loading from HuggingFace...")
    try:
        dataset = load_dataset(ds_config["hf_name"], split="train")
    except Exception as e:
        print(f"  [error] Failed to load dataset: {e}")
        # Try without specifying split
        try:
            dataset = load_dataset(ds_config["hf_name"])
            # Get the first available split
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
            print(f"  [info] Using split: {split_name}")
        except Exception as e2:
            print(f"  [error] Could not load dataset at all: {e2}")
            return 0
    
    print(f"  Total rows: {len(dataset):,}")
    
    # Subsample if configured
    indices = list(range(len(dataset)))
    if ds_config["subsample"] is not None and ds_config["subsample"] < len(dataset):
        random.seed(config.SPLIT_SEED)
        indices = random.sample(indices, ds_config["subsample"])
        print(f"  Subsampling to {ds_config['subsample']:,} entries")
    
    # Determine column names
    svg_col = ds_config["svg_column"]
    fname_col = ds_config.get("filename_column")
    
    # Check actual column names
    actual_columns = dataset.column_names
    print(f"  Columns: {actual_columns}")
    
    # Try to find SVG column (case-insensitive)
    if svg_col not in actual_columns:
        for col in actual_columns:
            if col.lower() == svg_col.lower():
                svg_col = col
                break
            if col.lower() in ("svg", "code", "svg_code"):
                svg_col = col
                break
    
    if svg_col not in actual_columns:
        print(f"  [error] SVG column '{svg_col}' not found. Available: {actual_columns}")
        return 0
    
    print(f"  Using SVG column: '{svg_col}'")
    
    # Extract and save SVGs
    count = 0
    for i in tqdm(indices, desc=f"  Saving {name}", unit="svg"):
        row = dataset[i]
        svg_string = row[svg_col]
        
        if svg_string is None or len(str(svg_string).strip()) == 0:
            continue
        
        # Determine filename
        if fname_col and fname_col in row and row[fname_col]:
            fname = str(row[fname_col])
            if not fname.endswith(".svg"):
                fname += ".svg"
        else:
            fname = f"{name}_{i:07d}.svg"
        
        # Clean filename
        fname = fname.replace("/", "_").replace("\\", "_").replace(" ", "_")
        
        # Save
        filepath = output_dir / fname
        try:
            filepath.write_text(str(svg_string), encoding="utf-8")
            count += 1
        except Exception as e:
            pass  # Skip problematic files silently
    
    print(f"  [OK] Saved {count:,} SVGs to {output_dir}")
    return count


def main():
    print("=" * 60)
    print("STEP 1: Download SVG Datasets from HuggingFace")
    print("=" * 60)
    
    total = 0
    for name, ds_config in config.DATASETS.items():
        count = download_dataset(name, ds_config)
        total += count
    
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE: {total:,} total SVG files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
