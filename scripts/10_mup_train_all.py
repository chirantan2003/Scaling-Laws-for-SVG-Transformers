import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from pathlib import Path

from train import TrainConfig
from train_mup import train_mup


# Same per-model batch sizes as SP (Part 2), tuned for 8GB VRAM
MODEL_TRAINING_CONFIGS = {
    "tiny":   {"micro_batch_size": 32, "gradient_accum_steps": 8},
    "small":  {"micro_batch_size": 16, "gradient_accum_steps": 16},
    "medium": {"micro_batch_size": 8,  "gradient_accum_steps": 32},
    "large":  {"micro_batch_size": 4,  "gradient_accum_steps": 64},
    "xl":     {"micro_batch_size": 1,  "gradient_accum_steps": 256},
}


def main():
    print("=" * 60)
    print("STEP 10: Train All muP Model Sizes")
    print("=" * 60)

    # Load best muP LR from sweep
    sweep_path = Path("checkpoints/mup_lr_sweep/sweep_results.json")
    if sweep_path.exists():
        with open(sweep_path) as f:
            sweep_data = json.load(f)
        best_lr = sweep_data["best_lr"]
        print(f"  Using best muP LR from sweep: {best_lr}")
    else:
        best_lr = 1e-2  # Fallback (muP typically uses higher LR)
        print(f"  [WARN] No muP sweep results found. Using default LR: {best_lr}")
        print(f"  Run 09_mup_lr_sweep.py first for proper LR selection.")

    scaling_dir = Path("checkpoints/mup_scaling")
    scaling_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_name, batch_cfg in MODEL_TRAINING_CONFIGS.items():
        print(f"\n{'#'*60}")
        print(f"  muP Training: {model_name.upper()}")
        print(f"{'#'*60}")

        cfg = TrainConfig(
            model_name=model_name,
            learning_rate=best_lr,
            micro_batch_size=batch_cfg["micro_batch_size"],
            gradient_accum_steps=batch_cfg["gradient_accum_steps"],
            max_epochs=1,
            warmup_steps=200,
            eval_interval=50,
            eval_batches=20,
            log_interval=10,
            checkpoint_dir=str(scaling_dir),
            save_checkpoint=True,
        )

        metrics = train_mup(cfg)
        all_results.append(metrics)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"ALL muP MODELS TRAINED")
    print(f"{'='*60}")
    print(f"  {'Model':<8} | {'Params':>10} | {'Val Loss':>10} | {'Time':>8} | {'Tok/s':>10} | {'GPU MB':>8}")
    print(f"  {'-'*66}")
    for r in all_results:
        print(f"  {r['model_name']:<8} | {r['n_params']:>10,} | {r['best_val_loss']:>10.4f} | "
              f"{r['wall_time_seconds']:>7.0f}s | {r['tokens_per_second']:>10,.0f} | "
              f"{r['peak_gpu_memory_mb']:>8,.0f}")

    combined_path = scaling_dir / "all_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Combined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
