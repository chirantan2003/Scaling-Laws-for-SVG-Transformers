"""
Step 5: Compute dataset statistics and render SVG examples.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import config
from utils.svg_utils import render_svg_to_png
from utils.data_utils import load_json


def plot_seq_length_histogram(seq_lengths, output_path):
    """Plot histogram of sequence lengths across all splits."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"train": "#4CAF50", "val": "#2196F3", "test": "#FF9800"}
    for ax, (split, lengths) in zip(axes, seq_lengths.items()):
        ax.hist(lengths, bins=80, color=colors.get(split, "#999"), alpha=0.85, edgecolor="white")
        ax.set_title(f"{split.upper()} (n={len(lengths):,})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Sequence Length (tokens)")
        ax.set_ylabel("Count")
        ax.axvline(np.median(lengths), color="red", linestyle="--", label=f"median={np.median(lengths):.0f}")
        ax.legend()
    plt.suptitle("Token Sequence Length Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Histogram saved to {output_path}")


def plot_file_counts(output_path):
    """Plot file counts per dataset before/after filtering."""
    before, after = {}, {}
    for name in config.DATASETS:
        raw_dir = config.RAW_DIR / name
        before[name] = len(list(raw_dir.glob("*.svg"))) if raw_dir.exists() else 0
    # Count cleaned files by source prefix
    for f in config.CLEANED_DIR.glob("*.svg"):
        for name in config.DATASETS:
            if f.name.startswith(name):
                after[name] = after.get(name, 0) + 1; break

    names = list(config.DATASETS.keys())
    b_vals = [before.get(n, 0) for n in names]
    a_vals = [after.get(n, 0) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, b_vals, 0.35, label="Before Filtering", color="#42A5F5")
    ax.bar(x + 0.2, a_vals, 0.35, label="After Filtering", color="#66BB6A")
    ax.set_xticks(x); ax.set_xticklabels([n.replace("svg-","") for n in names])
    ax.set_ylabel("Number of Files"); ax.set_title("File Counts: Before vs After Filtering")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (b, a) in enumerate(zip(b_vals, a_vals)):
        ax.text(i - 0.2, b + 200, f"{b:,}", ha="center", fontsize=8)
        ax.text(i + 0.2, a + 200, f"{a:,}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] File counts chart saved to {output_path}")


def render_examples():
    """Render SVGs at various complexity percentiles."""
    print(f"\n  Rendering SVG examples at complexity percentiles...")
    config.EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load sequence lengths to determine percentiles
    seq_path = config.SPLITS_DIR / "seq_lengths.json"
    if not seq_path.exists():
        print("  [skip] seq_lengths.json not found"); return
    seq_data = load_json(seq_path)
    train_lengths = seq_data.get("train", [])
    if not train_lengths:
        print("  [skip] No train lengths"); return

    # Get all cleaned files and their token lengths (approx by char length)
    svg_files = sorted(config.CLEANED_DIR.glob("*.svg"))
    file_lengths = []
    for f in svg_files:
        try:
            content = f.read_text(encoding="utf-8")
            file_lengths.append((f, len(content)))
        except Exception:
            pass

    file_lengths.sort(key=lambda x: x[1])
    percentiles = [10, 25, 50, 75, 90]
    examples_info = []

    for p in percentiles:
        idx = int(len(file_lengths) * p / 100)
        idx = min(idx, len(file_lengths) - 1)
        f, length = file_lengths[idx]
        svg_content = f.read_text(encoding="utf-8")
        png_name = f"example_p{p}.png"
        png_path = config.EXAMPLES_DIR / png_name
        ok = render_svg_to_png(svg_content, str(png_path), width=512)
        # Also save the SVG source
        svg_out = config.EXAMPLES_DIR / f"example_p{p}.svg"
        svg_out.write_text(svg_content, encoding="utf-8")
        examples_info.append({
            "percentile": p, "filename": f.name,
            "char_length": length, "rendered": ok, "png": png_name
        })
        status = "OK" if ok else "FAIL"
        print(f"    P{p:2d}: {length:>6,} chars | {f.name[:40]:40s} | render: {status}")

    info_path = config.EXAMPLES_DIR / "examples_info.json"
    with open(info_path, "w") as fh:
        json.dump(examples_info, fh, indent=2)


def main():
    print("=" * 60)
    print("STEP 5: Dataset Statistics & Visualization")
    print("=" * 60)

    config.STATS_DIR.mkdir(parents=True, exist_ok=True)

    # Load split metadata
    meta_path = config.SPLITS_DIR / "split_metadata.json"
    if not meta_path.exists():
        print("  [error] split_metadata.json not found! Run step 04 first."); return
    meta = load_json(meta_path)

    # ---- Summary Table ----
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Vocabulary size:  {meta['vocab_size']:,}")
    print(f"  Max seq length:   {meta['max_seq_length']:,}")
    print(f"  Split seed:       {meta.get('split_seed', 'N/A')}")
    filt = meta.get("filtering", {})
    print(f"\n  Files before filtering: {filt.get('total_raw', 'N/A'):,}")
    print(f"  Files after filtering:  {filt.get('accepted', 'N/A'):,}")
    print(f"  Filtered (too long):    {filt.get('filtered_long', 'N/A'):,}")
    print(f"  Filtered (empty):       {filt.get('filtered_empty', 'N/A'):,}")

    print(f"\n  {'Split':<8} {'Files':>10} {'Tokens':>14} {'Mean Len':>10} {'Med Len':>10} {'Min':>6} {'Max':>6}")
    print(f"  {'-'*66}")
    total_tokens = 0
    for name, s in meta["splits"].items():
        ml = s.get("mean_len", s.get("mean_seq_length", 0))
        md = s.get("median_len", s.get("median_seq_length", 0))
        mn = s.get("min_len", s.get("min_seq_length", 0))
        mx = s.get("max_len", s.get("max_seq_length", 0))
        print(f"  {name:<8} {s['num_files']:>10,} {s['total_tokens']:>14,} {ml:>10.0f} {md:>10.0f} {mn:>6} {mx:>6}")
        total_tokens += s["total_tokens"]
    print(f"  {'TOTAL':<8} {'':>10} {total_tokens:>14,}")

    # ---- Save summary as markdown ----
    md_lines = ["# Dataset Statistics\n"]
    md_lines.append(f"| Metric | Value |")
    md_lines.append(f"|--------|-------|")
    md_lines.append(f"| Vocabulary Size | {meta['vocab_size']:,} |")
    md_lines.append(f"| Max Sequence Length | {meta['max_seq_length']:,} |")
    md_lines.append(f"| Files Before Filtering | {filt.get('total_raw', 'N/A'):,} |")
    md_lines.append(f"| Files After Filtering | {filt.get('accepted', 'N/A'):,} |")
    md_lines.append(f"| Total Tokens | {total_tokens:,} |")
    md_lines.append(f"\n## Split Details\n")
    md_lines.append(f"| Split | Files | Tokens | Mean Len | Median Len |")
    md_lines.append(f"|-------|-------|--------|----------|------------|")
    for name, s in meta["splits"].items():
        ml = s.get("mean_len", s.get("mean_seq_length", 0))
        md_val = s.get("median_len", s.get("median_seq_length", 0))
        md_lines.append(f"| {name} | {s['num_files']:,} | {s['total_tokens']:,} | {ml:.0f} | {md_val:.0f} |")
    (config.STATS_DIR / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    # ---- Plots ----
    seq_path = config.SPLITS_DIR / "seq_lengths.json"
    if seq_path.exists():
        seq_data = load_json(seq_path)
        plot_seq_length_histogram(seq_data, config.STATS_DIR / "seq_length_histogram.png")
    plot_file_counts(config.STATS_DIR / "file_counts.png")

    # ---- Render Examples ----
    render_examples()

    print(f"\n{'='*60}")
    print(f"STATISTICS COMPLETE — outputs in {config.STATS_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
