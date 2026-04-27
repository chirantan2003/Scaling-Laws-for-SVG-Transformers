"""
Step 4: Tokenize the cleaned SVG corpus, filter by sequence length,
and create train/val/test splits.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, numpy as np
from pathlib import Path
from tqdm import tqdm
from tokenizers import Tokenizer
import config
from utils.data_utils import split_files, save_token_array, save_json


def main():
    print("=" * 60)
    print("STEP 4: Tokenize Corpus & Create Train/Val/Test Splits")
    print("=" * 60)

    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_PATH))
    bos_id, eos_id = config.BOS_TOKEN_ID, config.EOS_TOKEN_ID

    svg_files = sorted(config.CLEANED_DIR.glob("*.svg"))
    print(f"  Found {len(svg_files):,} cleaned SVG files")
    if not svg_files:
        print("  [error] No cleaned SVGs! Run step 02 first."); return

    # Tokenize all files
    print(f"\n  Tokenizing all files...")
    file_data = {}
    filtered_long = filtered_empty = 0

    for f in tqdm(svg_files, desc="  Tokenizing", unit="svg"):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            ids = tokenizer.encode(text).ids
            if not ids:
                filtered_empty += 1; continue
            full_seq = [bos_id] + list(ids) + [eos_id]
            if len(full_seq) > config.MAX_SEQ_LENGTH:
                filtered_long += 1; continue
            file_data[f.name] = full_seq
        except Exception:
            pass

    print(f"    Accepted: {len(file_data):,}  Filtered(long): {filtered_long:,}  Filtered(empty): {filtered_empty:,}")

    # Split by file
    filenames = list(file_data.keys())
    train_f, val_f, test_f = split_files(filenames, config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO, config.SPLIT_SEED)
    print(f"    Train: {len(train_f):,}  Val: {len(val_f):,}  Test: {len(test_f):,}")

    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    seq_len_data = {}

    for name, flist in [("train", train_f), ("val", val_f), ("test", test_f)]:
        tokens, lengths = [], []
        for fn in flist:
            seq = file_data[fn]; lengths.append(len(seq)); tokens.extend(seq)
        save_token_array(tokens, config.SPLITS_DIR / f"{name}.npy")
        seq_len_data[name] = lengths
        split_stats[name] = {
            "num_files": len(flist), "total_tokens": len(tokens),
            "mean_len": float(np.mean(lengths)), "median_len": float(np.median(lengths)),
            "min_len": int(np.min(lengths)), "max_len": int(np.max(lengths)),
        }

    meta = {
        "vocab_size": config.VOCAB_SIZE, "max_seq_length": config.MAX_SEQ_LENGTH,
        "split_seed": config.SPLIT_SEED,
        "filtering": {"total_raw": len(svg_files), "accepted": len(file_data),
                       "filtered_long": filtered_long, "filtered_empty": filtered_empty},
        "splits": split_stats,
    }
    save_json(meta, config.SPLITS_DIR / "split_metadata.json")
    save_json(seq_len_data, config.SPLITS_DIR / "seq_lengths.json")

    print(f"\n{'='*60}\nSPLIT COMPLETE\n{'='*60}")
    for n, s in split_stats.items():
        print(f"  {n:5s}: {s['num_files']:>8,} files | {s['total_tokens']:>12,} tokens | mean={s['mean_len']:.0f}")
    t = split_stats["train"]["total_tokens"]
    if t >= 100_000_000:
        print(f"\n  [OK] Train tokens ({t:,}) >= 100M target")
    else:
        print(f"\n  [WARN] Train tokens ({t:,}) < 100M. Increase fonts subsample in config.py")

if __name__ == "__main__":
    main()
