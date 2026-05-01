import os
import sys
import time
import math
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_mup import MuGPT, create_mup_model
from mup.optim import MuAdamW
from train import TrainConfig, get_batch, evaluate, get_lr


def train_mup(cfg: TrainConfig) -> Dict:
    """
    Training function for µP models.

    Similar to train() but uses MuAdamW and relative LR scheduling.
    """
    print("=" * 60)
    print(f"TRAINING (muP): {cfg.model_name}")
    print("=" * 60)

    # ---- Setup ----
    device = cfg.device
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    pt_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)
    ctx = (torch.amp.autocast(device_type="cuda", dtype=pt_dtype)
           if device == "cuda"
           else torch.amp.autocast(device_type="cpu", dtype=pt_dtype))

    # ---- Load Data ----
    print(f"\n  Loading data...")
    train_data = np.load(cfg.train_data_path, mmap_mode='r')
    val_data = np.load(cfg.val_data_path, mmap_mode='r')
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens:   {len(val_data):,}")

    tokens_per_step = cfg.micro_batch_size * cfg.gradient_accum_steps * cfg.block_size
    total_steps = (len(train_data) // tokens_per_step) * cfg.max_epochs
    print(f"  Tokens per optimizer step: {tokens_per_step:,}")
    print(f"  Total optimizer steps: {total_steps:,}")

    # ---- Create µP Model ----
    print(f"\n  Creating muP model '{cfg.model_name}'...")
    model = create_mup_model(cfg.model_name)
    n_params = model.get_num_params()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    model = model.to(device)

    # ---- µP Optimizer (MuAdamW) ----
    print(f"\n  Configuring MuAdamW optimizer...")
    print(f"  Learning rate: {cfg.learning_rate}")

    # MuAdamW automatically scales LR per layer based on width ratio
    # We still separate weight-decay vs no-weight-decay groups
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': cfg.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in nodecay_params)
    print(f"  Decayed params: {len(decay_params)} tensors, {num_decay:,} parameters")
    print(f"  Non-decayed params: {len(nodecay_params)} tensors, {num_nodecay:,} parameters")

    optimizer = MuAdamW(optim_groups, lr=cfg.learning_rate, betas=cfg.betas)

    # GradScaler for float16
    scaler = torch.amp.GradScaler(enabled=(cfg.dtype == "float16"))

    min_lr = cfg.learning_rate * cfg.min_lr_ratio

    # Store the initial per-group LRs set by MuAdamW (it adjusts them internally)
    initial_lrs = [pg['lr'] for pg in optimizer.param_groups]

    # ---- Metrics Storage ----
    model_config = model.config
    metrics = {
        "model_name": cfg.model_name,
        "n_params": n_params,
        "config": {
            "n_layer": model_config.n_layer,
            "n_head": model_config.n_head,
            "n_embd": model_config.n_embd,
            "d_ff": model_config.d_ff or 4 * model_config.n_embd,
        },
        "learning_rate": cfg.learning_rate,
        "parameterization": "mup",
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "train_losses": [],
        "val_losses": [],
        "lr_history": [],
        "wall_time_seconds": 0,
        "peak_gpu_memory_mb": 0,
        "tokens_per_second": 0,
        "final_train_loss": 0,
        "final_val_loss": 0,
    }

    # ---- Training Loop ----
    print(f"\n  Starting training...")
    print(f"  {'Step':>6} | {'LR':>10} | {'Train Loss':>10} | {'Val Loss':>10} | {'Tok/s':>10} | {'GPU MB':>8}")
    print(f"  {'-'*70}")

    model.train()
    t0 = time.time()
    total_tokens_processed = 0
    best_val_loss = float('inf')
    running_loss = 0.0
    running_count = 0

    for step in range(total_steps):
        # Cosine LR schedule — apply as ratio to preserve mup's per-layer scaling
        base_lr = get_lr(step, cfg.warmup_steps, total_steps, cfg.learning_rate, min_lr)
        lr_ratio = base_lr / cfg.learning_rate  # Ratio relative to initial LR

        for pg, init_lr in zip(optimizer.param_groups, initial_lrs):
            pg['lr'] = init_lr * lr_ratio

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(cfg.gradient_accum_steps):
            x, y = get_batch(train_data, cfg.block_size, cfg.micro_batch_size, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.gradient_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()
            total_tokens_processed += x.numel()

        # Gradient clipping
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += accum_loss
        running_count += 1

        metrics["lr_history"].append((step, base_lr))

        # Logging
        if (step + 1) % cfg.log_interval == 0:
            avg_loss = running_loss / running_count
            elapsed = time.time() - t0
            tok_per_sec = total_tokens_processed / elapsed if elapsed > 0 else 0
            gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0

            metrics["train_losses"].append((step, avg_loss))
            print(f"  {step+1:>6} | {base_lr:>10.6f} | {avg_loss:>10.4f} | {'':>10} | {tok_per_sec:>10,.0f} | {gpu_mem:>8,.0f}")

            running_loss = 0.0
            running_count = 0

        # Validation
        if (step + 1) % cfg.eval_interval == 0 or step == total_steps - 1:
            val_loss = evaluate(model, val_data, cfg.block_size,
                              cfg.micro_batch_size, cfg.eval_batches, device, ctx)
            metrics["val_losses"].append((step, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            elapsed = time.time() - t0
            tok_per_sec = total_tokens_processed / elapsed if elapsed > 0 else 0
            gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0

            print(f"  {step+1:>6} | {base_lr:>10.6f} | {'':>10} | {val_loss:>10.4f} | {tok_per_sec:>10,.0f} | {gpu_mem:>8,.0f}")

    # ---- Final Metrics ----
    total_time = time.time() - t0
    metrics["wall_time_seconds"] = total_time
    metrics["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0
    metrics["tokens_per_second"] = total_tokens_processed / total_time if total_time > 0 else 0
    metrics["final_train_loss"] = metrics["train_losses"][-1][1] if metrics["train_losses"] else 0
    metrics["final_val_loss"] = metrics["val_losses"][-1][1] if metrics["val_losses"] else 0
    metrics["best_val_loss"] = best_val_loss

    print(f"\n  {'='*60}")
    print(f"  TRAINING COMPLETE (muP): {cfg.model_name}")
    print(f"  {'='*60}")
    print(f"  Wall time:        {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Final train loss: {metrics['final_train_loss']:.4f}")
    print(f"  Final val loss:   {metrics['final_val_loss']:.4f}")
    print(f"  Best val loss:    {best_val_loss:.4f}")
    print(f"  Tokens/sec:       {metrics['tokens_per_second']:,.0f}")
    print(f"  Peak GPU memory:  {metrics['peak_gpu_memory_mb']:,.0f} MB")

    # ---- Save Checkpoint ----
    if cfg.save_checkpoint:
        ckpt_dir = Path(cfg.checkpoint_dir) / cfg.model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = ckpt_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        ckpt_path = ckpt_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': asdict(model.config),
            'train_config': asdict(cfg),
            'metrics': metrics,
        }, ckpt_path)

        print(f"  Checkpoint saved to: {ckpt_dir}")

    return metrics
