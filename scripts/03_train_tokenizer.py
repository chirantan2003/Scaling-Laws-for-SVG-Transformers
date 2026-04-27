"""
Step 3: Train a BPE tokenizer on the cleaned SVG corpus.

Uses HuggingFace tokenizers library with ByteLevel BPE.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from tokenizers.normalizers import NFC
import json

import config


def main():
    print("=" * 60)
    print("STEP 3: Train BPE Tokenizer")
    print("=" * 60)
    
    # Collect all cleaned SVG files
    svg_files = sorted(config.CLEANED_DIR.glob("*.svg"))
    print(f"  Found {len(svg_files):,} cleaned SVG files")
    
    if not svg_files:
        print("  [error] No cleaned SVG files found! Run 02_clean_normalize.py first.")
        return
    
    # Initialize ByteLevel BPE tokenizer
    print(f"\n  Initializing ByteLevel BPE tokenizer...")
    print(f"  Vocab size: {config.VOCAB_SIZE}")
    print(f"  Special tokens: {config.SPECIAL_TOKENS}")
    
    tokenizer = Tokenizer(models.BPE())
    
    # ByteLevel pre-tokenizer — handles all characters including special XML chars
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Unicode NFC normalization
    tokenizer.normalizer = NFC()
    
    # Decoder for ByteLevel
    tokenizer.decoder = decoders.ByteLevel()
    
    # Post-processor: add <bos> and <eos> tokens
    # We'll handle this manually during tokenization for more control
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=config.VOCAB_SIZE,
        min_frequency=2,
        special_tokens=config.SPECIAL_TOKENS,
        show_progress=True,
    )
    
    # Train on SVG files
    print(f"\n  Training tokenizer on {len(svg_files):,} files...")
    file_paths = [str(f) for f in svg_files]
    tokenizer.train(file_paths, trainer)
    
    # Save tokenizer
    config.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(config.TOKENIZER_PATH))
    print(f"\n  [OK] Tokenizer saved to: {config.TOKENIZER_PATH}")
    
    # ---- Tokenizer Statistics ----
    vocab = tokenizer.get_vocab()
    print(f"\n{'='*60}")
    print(f"TOKENIZER STATISTICS")
    print(f"{'='*60}")
    print(f"  Vocabulary size: {len(vocab):,}")
    
    # Show special tokens
    print(f"\n  Special tokens:")
    for tok in config.SPECIAL_TOKENS:
        tid = vocab.get(tok, "N/A")
        print(f"    {tok} -> {tid}")
    
    # Test encoding on a sample SVG
    print(f"\n  Sample encoding test:")
    test_svg = '<svg width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'
    encoded = tokenizer.encode(test_svg)
    print(f"    Input:  {test_svg[:80]}...")
    print(f"    Tokens: {len(encoded.ids)} IDs")
    print(f"    First 20 tokens: {encoded.tokens[:20]}")
    
    decoded = tokenizer.decode(encoded.ids)
    print(f"    Decoded: {decoded[:80]}...")
    
    # Token frequency analysis — encode a sample of files and count
    print(f"\n  Analyzing token frequency on sample...")
    from collections import Counter
    token_counts = Counter()
    sample_size = min(5000, len(svg_files))
    
    for f in svg_files[:sample_size]:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            enc = tokenizer.encode(text)
            token_counts.update(enc.tokens)
        except Exception:
            pass
    
    print(f"\n  Top 30 most frequent tokens:")
    for tok, count in token_counts.most_common(30):
        # Escape for display
        display_tok = repr(tok)
        print(f"    {display_tok:30s} : {count:,}")
    
    # Save tokenizer stats
    stats = {
        "vocab_size": len(vocab),
        "special_tokens": {tok: vocab.get(tok) for tok in config.SPECIAL_TOKENS},
        "top_50_tokens": [(tok, count) for tok, count in token_counts.most_common(50)],
        "test_input": test_svg,
        "test_token_count": len(encoded.ids),
    }
    stats_path = config.TOKENIZER_DIR / "tokenizer_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  [OK] Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
