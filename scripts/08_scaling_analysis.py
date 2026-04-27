"""
Step 8: Scaling Analysis — fit power law and generate plots.

Creates:
  - Scaling plot (params vs val loss) with power law fit
  - Training loss curves for all models
  - Model summary table
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


def power_law(N, a, alpha, c):
    """Power law: L = a * N^(-alpha) + c"""
    return a * np.power(N, -alpha) + c


def main():
    print("=" * 60)
    print("STEP 8: Scaling Analysis")
    print("=" * 60)

    scaling_dir = Path("checkpoints/scaling")
    results_path = scaling_dir / "all_results.json"

    if not results_path.exists():
        print("  [error] all_results.json not found! Run 07_train_all_models.py first.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    print(f"  Loaded results for {len(all_results)} models")

    # Extract data
    names = [r["model_name"] for r in all_results]
    params = np.array([r["n_params"] for r in all_results], dtype=float)
    val_losses = np.array([r["best_val_loss"] for r in all_results], dtype=float)

    # Sort by params
    sort_idx = np.argsort(params)
    names = [names[i] for i in sort_idx]
    params = params[sort_idx]
    val_losses = val_losses[sort_idx]

    # ================================================================
    # 1. Fit Power Law: L = a * N^(-alpha) + c
    # ================================================================
    print(f"\n  Fitting power law: L = a * N^(-alpha) + c")

    try:
        popt, pcov = curve_fit(
            power_law, params, val_losses,
            p0=[10.0, 0.1, min(val_losses) * 0.9],  # Initial guess
            bounds=([0, 0, 0], [1e6, 2.0, max(val_losses)]),
            maxfev=10000,
        )
        a, alpha, c = popt
        perr = np.sqrt(np.diag(pcov))

        # Compute R^2
        y_pred = power_law(params, *popt)
        ss_res = np.sum((val_losses - y_pred) ** 2)
        ss_tot = np.sum((val_losses - np.mean(val_losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"  Fitted parameters:")
        print(f"    a     = {a:.4f} +/- {perr[0]:.4f}")
        print(f"    alpha = {alpha:.4f} +/- {perr[1]:.4f}")
        print(f"    c     = {c:.4f} +/- {perr[2]:.4f}")
        print(f"    R^2   = {r_squared:.4f}")

        fit_success = True
    except Exception as e:
        print(f"  [warn] Power law fit failed: {e}")
        print(f"  Falling back to log-linear fit")
        fit_success = False

    # ================================================================
    # 2. Scaling Plot
    # ================================================================
    print(f"\n  Creating scaling plot...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Data points
    ax.scatter(params, val_losses, s=120, c='#E53935', zorder=5, edgecolors='white', linewidth=2)

    # Label each point
    for name, p, vl in zip(names, params, val_losses):
        ax.annotate(f"  {name}\n  ({p/1e6:.1f}M)", (p, vl),
                   fontsize=9, fontweight='bold',
                   ha='left', va='center')

    # Fitted curve
    if fit_success:
        x_fit = np.logspace(np.log10(params.min() * 0.5), np.log10(params.max() * 2), 200)
        y_fit = power_law(x_fit, *popt)
        ax.plot(x_fit, y_fit, '--', color='#1E88E5', linewidth=2,
                label=f'$L = {a:.2f} \\cdot N^{{-{alpha:.4f}}} + {c:.4f}$\n$R^2 = {r_squared:.4f}$')
        ax.legend(fontsize=12, loc='upper right')

    ax.set_xscale('log')
    ax.set_xlabel("Number of Parameters", fontsize=13)
    ax.set_ylabel("Validation Loss (1 Epoch)", fontsize=13)
    ax.set_title("SVG Language Model Scaling Law", fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    plot_path = scaling_dir / "scaling_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    # ================================================================
    # 3. Training Loss Curves (overlaid)
    # ================================================================
    print(f"  Creating training curves plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E53935', '#9C27B0']

    for i, (r, color) in enumerate(zip(all_results, colors)):
        name = r["model_name"]
        # Training loss
        if r["train_losses"]:
            steps_t, losses_t = zip(*r["train_losses"])
            ax1.plot(steps_t, losses_t, color=color, linewidth=1.5,
                    label=f'{name} ({r["n_params"]/1e6:.1f}M)', alpha=0.85)

        # Validation loss
        if r["val_losses"]:
            steps_v, losses_v = zip(*r["val_losses"])
            ax2.plot(steps_v, losses_v, 'o-', color=color, linewidth=1.5,
                    markersize=4, label=f'{name} ({r["n_params"]/1e6:.1f}M)', alpha=0.85)

    ax1.set_xlabel("Optimizer Step", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Training Loss Curves", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Optimizer Step", fontsize=12)
    ax2.set_ylabel("Validation Loss", fontsize=12)
    ax2.set_title("Validation Loss Curves", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = scaling_dir / "training_curves.png"
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {curves_path}")

    # ================================================================
    # 4. Summary Table (Markdown)
    # ================================================================
    print(f"\n  Generating summary table...")

    md_lines = ["# Part 2: Model Architectures and Training Statistics\n"]
    md_lines.append("## Model Configurations\n")
    md_lines.append("| Model | Params | d_model | Layers | Heads | d_ff |")
    md_lines.append("|-------|--------|---------|--------|-------|------|")
    for r in all_results:
        cfg = r["config"]
        md_lines.append(
            f"| {r['model_name']} | {r['n_params']:,} | {cfg['n_embd']} | "
            f"{cfg['n_layer']} | {cfg['n_head']} | {cfg.get('d_ff', 'N/A')} |"
        )

    md_lines.append("\n## Training Results\n")
    md_lines.append("| Model | Params | Val Loss | Train Loss | Time (s) | Tok/s | GPU MB |")
    md_lines.append("|-------|--------|----------|------------|----------|-------|--------|")
    for r in all_results:
        md_lines.append(
            f"| {r['model_name']} | {r['n_params']:,} | {r['best_val_loss']:.4f} | "
            f"{r['final_train_loss']:.4f} | {r['wall_time_seconds']:.0f} | "
            f"{r['tokens_per_second']:,.0f} | {r['peak_gpu_memory_mb']:,.0f} |"
        )

    if fit_success:
        md_lines.append("\n## Scaling Law Fit\n")
        md_lines.append(f"**Power Law:** L = {a:.4f} * N^(-{alpha:.4f}) + {c:.4f}\n")
        md_lines.append(f"- Scaling exponent alpha = {alpha:.4f}")
        md_lines.append(f"- Irreducible loss c = {c:.4f}")
        md_lines.append(f"- R-squared = {r_squared:.4f}")

    table_path = scaling_dir / "model_summary.md"
    with open(table_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"  Saved: {table_path}")

    # ================================================================
    # 5. Final Summary
    # ================================================================
    print(f"\n{'='*60}")
    print(f"SCALING ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Models: {', '.join(names)}")
    print(f"  Param range: {params.min()/1e6:.2f}M - {params.max()/1e6:.2f}M")
    print(f"  Val loss range: {val_losses.min():.4f} - {val_losses.max():.4f}")
    if fit_success:
        print(f"  Scaling exponent (alpha): {alpha:.4f}")
        print(f"  R^2: {r_squared:.4f}")
    print(f"\n  Outputs in: {scaling_dir}")


if __name__ == "__main__":
    main()
