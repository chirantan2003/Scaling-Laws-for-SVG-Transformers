import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import config
from utils.svg_utils import clean_svg, validate_render


def process_single_svg(args):
    """Process a single SVG file. Designed for multiprocessing."""
    filepath, source_dataset = args
    
    try:
        svg_string = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    
    cleaned, stats = clean_svg(
        svg_string,
        coord_precision=config.COORD_PRECISION,
        canonicalize=config.CANONICALIZE_ATTRS,
        min_length=config.MIN_SVG_LENGTH,
        max_length=config.MAX_SVG_LENGTH,
    )
    
    # Optional render validation
    if cleaned and config.VALIDATE_RENDER:
        stats["render_ok"] = validate_render(cleaned)
    
    result = {
        "filename": filepath.name,
        "source_dataset": source_dataset,
        "original_length": stats["original_length"],
        "cleaned_length": stats["cleaned_length"],
        "valid_xml": stats["valid_xml"],
        "render_ok": stats["render_ok"],
        "reject_reason": stats["reject_reason"],
        "cleaned_svg": cleaned,
    }
    
    return result


def main():
    print("=" * 60)
    print("STEP 2: Clean and Normalize SVG Files")
    print("=" * 60)
    
    # Collect all raw SVG files
    all_files = []
    raw_counts = {}
    for name in config.DATASETS.keys():
        raw_dir = config.RAW_DIR / name
        if raw_dir.exists():
            files = list(raw_dir.glob("*.svg"))
            all_files.extend([(f, name) for f in files])
            raw_counts[name] = len(files)
            print(f"  {name}: {len(files):,} raw files")
    
    print(f"\n  Total raw files: {len(all_files):,}")
    
    if not all_files:
        print("  [error] No raw SVG files found! Run 01_download_data.py first.")
        return
    
    # Clear old cleaned files
    if config.CLEANED_DIR.exists():
        old_files = list(config.CLEANED_DIR.glob("*.svg"))
        if old_files:
            print(f"  Clearing {len(old_files):,} old cleaned files...")
            for f in old_files:
                f.unlink()
    
    config.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    
    # PASS 1: Clean all SVGs, collect results in memory
    print(f"\n--- PASS 1: Cleaning & Normalizing ---")
    num_workers = min(os.cpu_count() or 4, 8)
    print(f"  Processing with {num_workers} workers...")
    
    all_results = []
    pass1_rejected = 0
    pass1_reject_reasons = {}
    cleaning_examples = []  # Collect a few before/after examples
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_svg, args): args
            for args in all_files
        }
        
        pbar = tqdm(total=len(all_files), desc="  Cleaning", unit="svg")
        for future in as_completed(futures):
            pbar.update(1)
            result = future.result()
            
            if result is None:
                pass1_rejected += 1
                continue
            
            if result["cleaned_svg"] is not None:
                all_results.append(result)
                
                # Collect first 3 examples where cleaning made a difference
                if len(cleaning_examples) < 3:
                    if result["original_length"] != result["cleaned_length"]:
                        cleaning_examples.append(result)
            else:
                pass1_rejected += 1
                reason = result.get("reject_reason", "unknown")
                pass1_reject_reasons[reason] = pass1_reject_reasons.get(reason, 0) + 1
        
        pbar.close()
    
    print(f"\n  Pass 1 results:")
    print(f"    Cleaned successfully: {len(all_results):,}")
    print(f"    Rejected (pass 1):    {pass1_rejected:,}")
    if pass1_reject_reasons:
        for reason, count in sorted(pass1_reject_reasons.items(), key=lambda x: -x[1]):
            print(f"      {reason}: {count:,}")
    
    # Show cleaning examples to prove it's working
    print(f"\n  --- Cleaning Examples (before -> after) ---")
    for i, ex in enumerate(cleaning_examples[:3]):
        pct_change = (1 - ex["cleaned_length"] / ex["original_length"]) * 100
        print(f"    Example {i+1}: {ex['filename']}")
        print(f"      Original: {ex['original_length']:,} chars -> Cleaned: {ex['cleaned_length']:,} chars ({pct_change:.1f}% reduction)")
        # Show a snippet of the cleaned SVG
        snippet = ex["cleaned_svg"][:120] + "..."
        print(f"      Preview:  {snippet}")
    
    # PASS 2: Percentile-based filtering
    print(f"\n--- PASS 2: Percentile-Based Filtering ---")
    
    cleaned_lengths = np.array([r["cleaned_length"] for r in all_results])
    
    # Compute statistics
    p1 = np.percentile(cleaned_lengths, 1)
    p25 = np.percentile(cleaned_lengths, 25)
    p50 = np.percentile(cleaned_lengths, 50)
    p75 = np.percentile(cleaned_lengths, 75)
    p99 = np.percentile(cleaned_lengths, 99)
    
    print(f"  Cleaned SVG character length distribution:")
    print(f"    Min:  {cleaned_lengths.min():,}")
    print(f"    P1:   {p1:,.0f}")
    print(f"    P25:  {p25:,.0f}")
    print(f"    P50:  {p50:,.0f}")
    print(f"    P75:  {p75:,.0f}")
    print(f"    P99:  {p99:,.0f}")
    print(f"    Max:  {cleaned_lengths.max():,}")
    
    # Filter: remove SVGs < 50 chars AND top 1 percentile
    min_chars = config.MIN_SVG_LENGTH   # 50
    max_chars = p99                      # 99th percentile
    
    print(f"\n  Filtering criteria:")
    print(f"    Min char length: {min_chars}")
    print(f"    Max char length (P99): {max_chars:,.0f}")
    
    accepted_results = []
    pass2_rejected = 0
    pass2_reasons = {}
    
    for result in all_results:
        length = result["cleaned_length"]
        
        if length < min_chars:
            pass2_rejected += 1
            pass2_reasons["too_short"] = pass2_reasons.get("too_short", 0) + 1
            result["reject_reason"] = "too_short_pass2"
        elif length > max_chars:
            pass2_rejected += 1
            pass2_reasons["top_1pct"] = pass2_reasons.get("top_1pct", 0) + 1
            result["reject_reason"] = "top_1_percentile"
        else:
            accepted_results.append(result)
    
    print(f"\n  Pass 2 results:")
    print(f"    Accepted:             {len(accepted_results):,}")
    print(f"    Rejected (too short): {pass2_reasons.get('too_short', 0):,}")
    print(f"    Rejected (top 1%):    {pass2_reasons.get('top_1pct', 0):,}")
    
    print(f"\n--- Saving cleaned SVGs ---")
    for result in tqdm(accepted_results, desc="  Saving", unit="svg"):
        out_name = f"{result['source_dataset']}_{result['filename']}"
        out_path = config.CLEANED_DIR / out_name
        out_path.write_text(result["cleaned_svg"], encoding="utf-8")
    
    manifest_path = config.CLEANED_DIR / "manifest.csv"
    all_manifest_entries = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "cleaned_svg"}
        all_manifest_entries.append(entry)
    
    if all_manifest_entries:
        fieldnames = list(all_manifest_entries[0].keys())
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_manifest_entries)
    
    print(f"\n--- Validating saved SVGs ---")
    from utils.svg_utils import validate_xml
    saved_files = list(config.CLEANED_DIR.glob("*.svg"))
    valid_count = 0
    invalid_count = 0
    invalid_examples = []
    
    for f in tqdm(saved_files, desc="  Validating", unit="svg"):
        try:
            content = f.read_text(encoding="utf-8")
            if validate_xml(content):
                valid_count += 1
            else:
                invalid_count += 1
                if len(invalid_examples) < 5:
                    invalid_examples.append(f.name)
        except Exception:
            invalid_count += 1
    
    total_rejected = pass1_rejected + pass2_rejected
    total_accepted = len(accepted_results)
    
    print(f"\n{'='*60}")
    print(f"CLEANING COMPLETE")
    print(f"{'='*60}")
    print(f"  Raw files:              {len(all_files):,}")
    print(f"  After cleaning (pass1): {len(all_results):,}")
    print(f"  After filtering (pass2):{total_accepted:,}")
    print(f"  Total rejected:         {total_rejected:,}")
    print(f"  Acceptance rate:        {total_accepted/len(all_files)*100:.1f}%")
    
    print(f"\n  Rejection breakdown:")
    all_reasons = {}
    all_reasons.update(pass1_reject_reasons)
    all_reasons.update(pass2_reasons)
    for reason, count in sorted(all_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count:,}")
    
    print(f"\n  XML Validation:")
    print(f"    Valid:   {valid_count:,}")
    print(f"    Invalid: {invalid_count:,}")
    if invalid_examples:
        print(f"    Invalid examples: {invalid_examples}")
    
    print(f"\n  Manifest saved to: {manifest_path}")
    print(f"  Cleaned SVGs in:   {config.CLEANED_DIR}")
    print(f"  Total cleaned files on disk: {len(saved_files):,}")


if __name__ == "__main__":
    main()
