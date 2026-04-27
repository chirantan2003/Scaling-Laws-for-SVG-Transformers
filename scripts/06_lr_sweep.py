"""
Step 6: Learning Rate Sweep on the Tiny model.

Tests 7 learning rates on a log scale, trains for 1 epoch each,
and selects the best LR based on final validation loss.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from train import train, TrainConfig


# Learning rates to sweep (log scale)
LR_VALUES = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def main():
    print("=" * 60)
    print("STEP 6: Learning Rate Sweep (Tiny Model)")
    print("=" * 60)
    print(f"  Testing {len(LR_VALUES)} learning rates: {LR_VALUES}")

    sweep_dir = Path("checkpoints/lr_sweep")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, lr in enumerate(LR_VALUES):
        print(f"\n{'#'*60}")
        print(f"  LR Sweep [{i+1}/{len(LR_VALUES)}]: lr = {lr}")
        print(f"{'#'*60}")

        cfg = TrainConfig(
            model_name="tiny",
            learning_rate=lr,
            micro_batch_size=32,
            gradient_accum_steps=8,
            max_epochs=1,
            warmup_steps=100,
            eval_interval=50,
            eval_batches=20,
            log_interval=10,
            checkpoint_dir=str(sweep_dir / f"lr_{lr:.0e}"),
            save_checkpoint=False,  # Don't save checkpoints for sweep
        )

        metrics = train(cfg)

        results.append({
            "lr": lr,
            "final_val_loss": metrics["final_val_loss"],
            "best_val_loss": metrics["best_val_loss"],
            "final_train_loss": metrics["final_train_loss"],
            "wall_time": metrics["wall_time_seconds"],
            "tokens_per_second": metrics["tokens_per_second"],
        })

    # ---- Find Best LR ----
    best = min(results, key=lambda r: r["best_val_loss"])
    best_lr = best["lr"]

    print(f"\n{'='*60}")
    print(f"LR SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"  {'LR':>10} | {'Val Loss':>10} | {'Train Loss':>10} | {'Time (s)':>10}")
    print(f"  {'-'*50}")
    for r in results:
        marker = " <-- BEST" if r["lr"] == best_lr else ""
        print(f"  {r['lr']:>10.1e} | {r['best_val_loss']:>10.4f} | {r['final_train_loss']:>10.4f} | {r['wall_time']:>10.1f}{marker}")

    print(f"\n  Best learning rate: {best_lr}")
    print(f"  Best validation loss: {best['best_val_loss']:.4f}")

    # ---- Save Results ----
    results_path = sweep_dir / "sweep_results.json"
    with open(results_path, 'w') as f:
        json.dump({"results": results, "best_lr": best_lr}, f, indent=2)

    # ---- Plot LR vs Val Loss ----
    fig, ax = plt.subplots(figsize=(8, 5))
    lrs = [r["lr"] for r in results]
    val_losses = [r["best_val_loss"] for r in results]

    ax.semilogx(lrs, val_losses, 'o-', color='#2196F3', markersize=8, linewidth=2)
    ax.axvline(best_lr, color='#4CAF50', linestyle='--', alpha=0.7, label=f'Best LR: {best_lr:.1e}')
    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Best Validation Loss", fontsize=12)
    ax.set_title("Learning Rate Sweep (Tiny Model, 1 Epoch)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = sweep_dir / "lr_sweep_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {plot_path}")

    # ---- Also plot training curves for all LRs ----
    # (We'd need to save per-step losses for this — done via metrics)

    print(f"\n  Results saved to: {results_path}")
    print(f"  Use best_lr = {best_lr} for all subsequent training runs.")


if __name__ == "__main__":
    main()
