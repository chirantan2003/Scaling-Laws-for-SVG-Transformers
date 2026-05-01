import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from train import TrainConfig
from train_mup import train_mup


# muP typically allows higher learning rates since updates are width-normalized
LR_VALUES = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


def main():
    print("=" * 60)
    print("STEP 9: muP Learning Rate Sweep (Tiny Model)")
    print("=" * 60)
    print(f"  Testing {len(LR_VALUES)} learning rates: {LR_VALUES}")

    sweep_dir = Path("checkpoints/mup_lr_sweep")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, lr in enumerate(LR_VALUES):
        print(f"\n{'#'*60}")
        print(f"  muP LR Sweep [{i+1}/{len(LR_VALUES)}]: lr = {lr}")
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
            save_checkpoint=False,
        )

        try:
            metrics = train_mup(cfg)
            results.append({
                "lr": lr,
                "final_val_loss": metrics["final_val_loss"],
                "best_val_loss": metrics["best_val_loss"],
                "final_train_loss": metrics["final_train_loss"],
                "wall_time": metrics["wall_time_seconds"],
                "tokens_per_second": metrics["tokens_per_second"],
            })
        except Exception as e:
            print(f"  [error] LR={lr} failed: {e}")
            results.append({
                "lr": lr,
                "final_val_loss": float('inf'),
                "best_val_loss": float('inf'),
                "final_train_loss": float('inf'),
                "wall_time": 0,
                "tokens_per_second": 0,
            })

    # ---- Find Best LR ----
    valid_results = [r for r in results if r["best_val_loss"] < float('inf')]
    if not valid_results:
        print("[error] All LR runs failed!")
        return

    best = min(valid_results, key=lambda r: r["best_val_loss"])
    best_lr = best["lr"]

    print(f"\n{'='*60}")
    print(f"muP LR SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"  {'LR':>10} | {'Val Loss':>10} | {'Train Loss':>10} | {'Time (s)':>10}")
    print(f"  {'-'*50}")
    for r in results:
        marker = " <-- BEST" if r["lr"] == best_lr else ""
        vl = f"{r['best_val_loss']:.4f}" if r['best_val_loss'] < float('inf') else "FAILED"
        tl = f"{r['final_train_loss']:.4f}" if r['final_train_loss'] < float('inf') else "FAILED"
        print(f"  {r['lr']:>10.1e} | {vl:>10} | {tl:>10} | {r['wall_time']:>10.1f}{marker}")

    print(f"\n  Best muP learning rate: {best_lr}")
    print(f"  Best validation loss: {best['best_val_loss']:.4f}")

    # ---- Save Results ----
    results_path = sweep_dir / "sweep_results.json"
    with open(results_path, 'w') as f:
        json.dump({"results": results, "best_lr": best_lr}, f, indent=2)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    lrs = [r["lr"] for r in valid_results]
    val_losses = [r["best_val_loss"] for r in valid_results]

    ax.semilogx(lrs, val_losses, 'o-', color='#E53935', markersize=8, linewidth=2, label='muP')
    ax.axvline(best_lr, color='#4CAF50', linestyle='--', alpha=0.7, label=f'Best LR: {best_lr:.1e}')
    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Best Validation Loss", fontsize=12)
    ax.set_title("muP Learning Rate Sweep (Tiny Model, 1 Epoch)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = sweep_dir / "mup_lr_sweep_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {plot_path}")
    print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
