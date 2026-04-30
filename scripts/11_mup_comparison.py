"""
Step 11: Compare SP vs muP scaling + Extrapolation analysis.

Creates:
  - Dual scaling plot (SP vs muP on same axes)
  - LR sweep comparison
  - Power law fits for both
  - Scaling law extrapolation with confidence intervals
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
    """L = a * N^(-alpha) + c"""
    return a * np.power(N, -alpha) + c


def load_results(path):
    """Load training results from JSON."""
    with open(path) as f:
        data = json.load(f)
    names = [r["model_name"] for r in data]
    params = np.array([r["n_params"] for r in data], dtype=float)
    val_losses = np.array([r["best_val_loss"] for r in data], dtype=float)
    # Sort by params
    idx = np.argsort(params)
    return [names[i] for i in idx], params[idx], val_losses[idx], [data[i] for i in idx]


def fit_power_law(params, val_losses, label=""):
    """Fit power law and return params + stats."""
    try:
        popt, pcov = curve_fit(
            power_law, params, val_losses,
            p0=[10.0, 0.1, min(val_losses) * 0.9],
            bounds=([0, 0, 0], [1e6, 2.0, max(val_losses)]),
            maxfev=10000,
        )
        a, alpha, c = popt
        perr = np.sqrt(np.diag(pcov))
        y_pred = power_law(params, *popt)
        ss_res = np.sum((val_losses - y_pred) ** 2)
        ss_tot = np.sum((val_losses - np.mean(val_losses)) ** 2)
        r2 = 1 - ss_res / ss_tot

        print(f"  {label} Power Law: L = {a:.4f} * N^(-{alpha:.4f}) + {c:.4f}")
        print(f"    alpha = {alpha:.4f} +/- {perr[1]:.4f}")
        print(f"    c     = {c:.4f} +/- {perr[2]:.4f}")
        print(f"    R^2   = {r2:.4f}")

        return {"a": a, "alpha": alpha, "c": c, "perr": perr.tolist(),
                "r2": r2, "popt": popt.tolist(), "pcov": pcov.tolist()}, True
    except Exception as e:
        print(f"  {label} Power law fit failed: {e}")
        return {}, False


def main():
    print("=" * 60)
    print("STEP 11: SP vs muP Comparison & Extrapolation")
    print("=" * 60)

    out_dir = Path("checkpoints/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load SP results ----
    sp_path = Path("checkpoints/scaling/all_results.json")
    mup_path = Path("checkpoints/mup_scaling/all_results.json")

    if not sp_path.exists():
        print("[error] SP results not found! Run 07_train_all_models.py first.")
        return
    if not mup_path.exists():
        print("[error] muP results not found! Run 10_mup_train_all.py first.")
        return

    sp_names, sp_params, sp_losses, sp_data = load_results(sp_path)
    mup_names, mup_params, mup_losses, mup_data = load_results(mup_path)

    print(f"\n  SP models:  {sp_names}")
    print(f"  muP models: {mup_names}")

    # ---- Fit power laws ----
    print(f"\n  Fitting power laws...")
    sp_fit, sp_ok = fit_power_law(sp_params, sp_losses, "SP")
    mup_fit, mup_ok = fit_power_law(mup_params, mup_losses, "muP")

    # ================================================================
    # PLOT 1: Dual Scaling Plot (SP vs muP)
    # ================================================================
    print(f"\n  Creating dual scaling plot...")
    fig, ax = plt.subplots(figsize=(10, 7))

    # SP data + fit
    ax.scatter(sp_params, sp_losses, s=120, c='#2196F3', zorder=5,
              edgecolors='white', linewidth=2, label='SP (Standard)')
    if sp_ok:
        x_fit = np.logspace(np.log10(sp_params.min() * 0.5),
                            np.log10(sp_params.max() * 2), 200)
        y_sp = power_law(x_fit, *sp_fit["popt"])
        ax.plot(x_fit, y_sp, '--', color='#2196F3', linewidth=2, alpha=0.7,
                label=f'SP: $\\alpha$={sp_fit["alpha"]:.4f}, R$^2$={sp_fit["r2"]:.4f}')

    # muP data + fit
    ax.scatter(mup_params, mup_losses, s=120, c='#E53935', zorder=5,
              edgecolors='white', linewidth=2, marker='D', label='muP')
    if mup_ok:
        x_fit = np.logspace(np.log10(mup_params.min() * 0.5),
                            np.log10(mup_params.max() * 2), 200)
        y_mup = power_law(x_fit, *mup_fit["popt"])
        ax.plot(x_fit, y_mup, '--', color='#E53935', linewidth=2, alpha=0.7,
                label=f'muP: $\\alpha$={mup_fit["alpha"]:.4f}, R$^2$={mup_fit["r2"]:.4f}')

    # Labels for each point
    for name, p, vl in zip(sp_names, sp_params, sp_losses):
        ax.annotate(f"  {name}", (p, vl), fontsize=8, color='#2196F3', ha='left')
    for name, p, vl in zip(mup_names, mup_params, mup_losses):
        ax.annotate(f"  {name}", (p, vl), fontsize=8, color='#E53935', ha='left')

    ax.set_xscale('log')
    ax.set_xlabel("Number of Parameters", fontsize=13)
    ax.set_ylabel("Validation Loss (1 Epoch)", fontsize=13)
    ax.set_title("Scaling Laws: SP vs muP", fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    plt.savefig(out_dir / "sp_vs_mup_scaling.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: sp_vs_mup_scaling.png")

    # ================================================================
    # PLOT 2: LR Sweep Comparison
    # ================================================================
    print(f"  Creating LR sweep comparison...")
    sp_sweep_path = Path("checkpoints/lr_sweep/sweep_results.json")
    mup_sweep_path = Path("checkpoints/mup_lr_sweep/sweep_results.json")

    fig, ax = plt.subplots(figsize=(8, 5))

    if sp_sweep_path.exists():
        with open(sp_sweep_path) as f:
            sp_sweep = json.load(f)
        sp_lrs = [r["lr"] for r in sp_sweep["results"]]
        sp_vl = [r["best_val_loss"] for r in sp_sweep["results"]]
        ax.semilogx(sp_lrs, sp_vl, 'o-', color='#2196F3', markersize=8,
                   linewidth=2, label=f'SP (best: {sp_sweep["best_lr"]:.1e})')

    if mup_sweep_path.exists():
        with open(mup_sweep_path) as f:
            mup_sweep = json.load(f)
        mup_lrs = [r["lr"] for r in mup_sweep["results"] if r["best_val_loss"] < float('inf')]
        mup_vl = [r["best_val_loss"] for r in mup_sweep["results"] if r["best_val_loss"] < float('inf')]
        ax.semilogx(mup_lrs, mup_vl, 'D-', color='#E53935', markersize=8,
                   linewidth=2, label=f'muP (best: {mup_sweep["best_lr"]:.1e})')

    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Best Validation Loss", fontsize=12)
    ax.set_title("LR Sweep: SP vs muP (Tiny Model)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_dir / "lr_sweep_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: lr_sweep_comparison.png")

    # ================================================================
    # PLOT 3: Training Curves Comparison
    # ================================================================
    print(f"  Creating training curves comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E53935', '#9C27B0']

    for i, (sp_r, mup_r, color) in enumerate(zip(sp_data, mup_data, colors)):
        name = sp_r["model_name"]
        # SP training curves (solid)
        if sp_r["val_losses"]:
            steps, losses = zip(*sp_r["val_losses"])
            axes[0].plot(steps, losses, 'o-', color=color, linewidth=1.5,
                        markersize=4, alpha=0.85,
                        label=f'{name} ({sp_r["n_params"]/1e6:.1f}M)')
        # muP training curves (dashed)
        if mup_r["val_losses"]:
            steps, losses = zip(*mup_r["val_losses"])
            axes[1].plot(steps, losses, 'D-', color=color, linewidth=1.5,
                        markersize=4, alpha=0.85,
                        label=f'{name} ({mup_r["n_params"]/1e6:.1f}M)')

    axes[0].set_xlabel("Optimizer Step", fontsize=12)
    axes[0].set_ylabel("Validation Loss", fontsize=12)
    axes[0].set_title("SP Validation Curves", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Optimizer Step", fontsize=12)
    axes[1].set_ylabel("Validation Loss", fontsize=12)
    axes[1].set_title("muP Validation Curves", fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: training_curves_comparison.png")

    # ================================================================
    # PLOT 4: Per-Model SP vs muP Bar Chart
    # ================================================================
    print(f"  Creating per-model comparison bar chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sp_names))
    width = 0.35

    bars_sp = ax.bar(x - width/2, sp_losses, width, label='SP', color='#2196F3', alpha=0.8)
    bars_mup = ax.bar(x + width/2, mup_losses, width, label='muP', color='#E53935', alpha=0.8)

    # Add value labels
    for bar in bars_sp:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
               f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars_mup:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
               f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Best Validation Loss", fontsize=12)
    ax.set_title("SP vs muP: Per-Model Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({p/1e6:.1f}M)" for n, p in zip(sp_names, sp_params)])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    plt.savefig(out_dir / "per_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_model_comparison.png")

    # ================================================================
    # SCALING LAW EXTRAPOLATION
    # ================================================================
    print(f"\n{'='*60}")
    print(f"SCALING LAW EXTRAPOLATION")
    print(f"{'='*60}")

    # Use the better fit (higher R^2)
    if sp_ok and mup_ok:
        best_label = "muP" if mup_fit["r2"] > sp_fit["r2"] else "SP"
        best_fit = mup_fit if best_label == "muP" else sp_fit
        best_params = mup_params if best_label == "muP" else sp_params
    elif sp_ok:
        best_label, best_fit, best_params = "SP", sp_fit, sp_params
    elif mup_ok:
        best_label, best_fit, best_params = "muP", mup_fit, mup_params
    else:
        print("  No valid power law fits! Cannot extrapolate.")
        return

    print(f"\n  Using {best_label} fit for extrapolation (R^2 = {best_fit['r2']:.4f})")

    # Predict at 10x XL
    xl_params = best_params.max()
    target_10x = xl_params * 10

    a, alpha, c = best_fit["popt"]
    pcov = np.array(best_fit["pcov"])
    predicted_loss = power_law(target_10x, a, alpha, c)

    # Confidence interval via error propagation (delta method)
    # Jacobian of L(N) = a*N^(-alpha) + c w.r.t. [a, alpha, c]
    N = target_10x
    J = np.array([
        N**(-alpha),                    # dL/da
        -a * N**(-alpha) * np.log(N),   # dL/dalpha
        1.0                             # dL/dc
    ])
    var_L = J @ pcov @ J
    std_L = np.sqrt(max(var_L, 0))
    ci_95 = 1.96 * std_L

    print(f"\n  Largest trained model: {xl_params/1e6:.1f}M params")
    print(f"  Extrapolation target: {target_10x/1e6:.1f}M params (10x)")
    print(f"  Predicted loss: {predicted_loss:.4f} +/- {ci_95:.4f} (95% CI)")
    print(f"  95% CI: [{predicted_loss - ci_95:.4f}, {predicted_loss + ci_95:.4f}]")

    # Also predict at 2x and 5x for discussion
    target_2x = xl_params * 2
    target_5x = xl_params * 5
    pred_2x = power_law(target_2x, a, alpha, c)
    pred_5x = power_law(target_5x, a, alpha, c)
    print(f"\n  Additional predictions:")
    print(f"    2x ({target_2x/1e6:.1f}M): {pred_2x:.4f}")
    print(f"    5x ({target_5x/1e6:.1f}M): {pred_5x:.4f}")
    print(f"   10x ({target_10x/1e6:.1f}M): {predicted_loss:.4f}")

    # ---- Extrapolation Plot ----
    print(f"\n  Creating extrapolation plot...")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Trained range
    ax.scatter(best_params, sp_losses if best_label == "SP" else mup_losses,
              s=120, c='#2196F3', zorder=5, edgecolors='white', linewidth=2,
              label=f'{best_label} trained models')

    # Fit curve (trained range)
    x_trained = np.logspace(np.log10(best_params.min() * 0.5),
                           np.log10(best_params.max() * 1.1), 200)
    y_trained = power_law(x_trained, a, alpha, c)
    ax.plot(x_trained, y_trained, '-', color='#2196F3', linewidth=2, alpha=0.8)

    # Extrapolation range (dashed)
    x_extrap = np.logspace(np.log10(best_params.max()),
                          np.log10(target_10x * 1.5), 200)
    y_extrap = power_law(x_extrap, a, alpha, c)
    ax.plot(x_extrap, y_extrap, '--', color='#FF9800', linewidth=2, alpha=0.8,
           label='Extrapolation')

    # Confidence band on extrapolation
    y_upper = []
    y_lower = []
    for N_pt in x_extrap:
        J_pt = np.array([N_pt**(-alpha), -a * N_pt**(-alpha) * np.log(N_pt), 1.0])
        var_pt = J_pt @ pcov @ J_pt
        ci_pt = 1.96 * np.sqrt(max(var_pt, 0))
        pred_pt = power_law(N_pt, a, alpha, c)
        y_upper.append(pred_pt + ci_pt)
        y_lower.append(pred_pt - ci_pt)
    ax.fill_between(x_extrap, y_lower, y_upper, alpha=0.15, color='#FF9800',
                   label='95% CI')

    # Prediction markers
    ax.scatter([target_10x], [predicted_loss], s=200, c='#E53935', zorder=6,
              marker='*', edgecolors='white', linewidth=2,
              label=f'10x prediction: {predicted_loss:.4f}')

    ax.set_xscale('log')
    ax.set_xlabel("Number of Parameters", fontsize=13)
    ax.set_ylabel("Validation Loss", fontsize=13)
    ax.set_title(f"Scaling Law Extrapolation ({best_label})", fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    plt.savefig(out_dir / "extrapolation_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: extrapolation_plot.png")

    # ================================================================
    # SAVE SUMMARY
    # ================================================================
    summary = {
        "sp_fit": sp_fit if sp_ok else None,
        "mup_fit": mup_fit if mup_ok else None,
        "best_approach": best_label,
        "extrapolation": {
            "xl_params": float(xl_params),
            "target_10x_params": float(target_10x),
            "predicted_loss": float(predicted_loss),
            "ci_95": float(ci_95),
            "ci_lower": float(predicted_loss - ci_95),
            "ci_upper": float(predicted_loss + ci_95),
        },
        "sp_results": {n: {"params": float(p), "val_loss": float(l)}
                      for n, p, l in zip(sp_names, sp_params, sp_losses)},
        "mup_results": {n: {"params": float(p), "val_loss": float(l)}
                       for n, p, l in zip(mup_names, mup_params, mup_losses)},
    }

    with open(out_dir / "comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- Print final comparison table ----
    print(f"\n{'='*60}")
    print(f"SP vs muP COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':<8} | {'SP Loss':>10} | {'muP Loss':>10} | {'Diff':>10} | {'muP Better?':>12}")
    print(f"  {'-'*60}")
    for n, sp_l, mup_l in zip(sp_names, sp_losses, mup_losses):
        diff = mup_l - sp_l
        better = "YES" if diff < 0 else "no"
        print(f"  {n:<8} | {sp_l:>10.4f} | {mup_l:>10.4f} | {diff:>+10.4f} | {better:>12}")

    if sp_ok and mup_ok:
        print(f"\n  SP  scaling exponent alpha = {sp_fit['alpha']:.4f}")
        print(f"  muP scaling exponent alpha = {mup_fit['alpha']:.4f}")
        steeper = "muP" if mup_fit['alpha'] > sp_fit['alpha'] else "SP"
        print(f"  Steeper scaling: {steeper}")

    print(f"\n  All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
