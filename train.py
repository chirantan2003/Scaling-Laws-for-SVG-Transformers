"""
Core training loop for GPT models on SVG data.

Supports:
  - Memory-mapped data loading from numpy arrays
  - Cosine LR schedule with linear warmup
  - Mixed precision (bfloat16) training
  - Gradient accumulation for large effective batch sizes
  - Periodic validation evaluation
  - Metrics tracking (loss, time, memory, throughput)
"""

import os
import sys
import time
import math
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Data
    train_data_path: str = "data/splits/train.npy"
    val_data_path: str = "data/splits/val.npy"

    # Model
    model_name: str = "medium"

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_epochs: int = 1
    grad_clip: float = 1.0

    # Batch sizing
    block_size: int = 2048        # Context window (sequence length)
    micro_batch_size: int = 8     # Sequences per GPU forward pass
    gradient_accum_steps: int = 32  # Accumulate before optimizer step
    # Effective batch = micro_batch_size * gradient_accum_steps * block_size tokens

    # LR Schedule
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1     # Min LR = min_lr_ratio * learning_rate

    # Evaluation
    eval_interval: int = 50       # Evaluate every N optimizer steps
    eval_batches: int = 20        # Number of val batches to average

    # Logging & Checkpointing
    log_interval: int = 10        # Print every N optimizer steps
    checkpoint_dir: str = "checkpoints"
    save_checkpoint: bool = True

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"       # bfloat16 or float16
    compile_model: bool = False   # torch.compile (can be slower on Windows)


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str):
    """Get a random batch of data.

    Samples random starting positions and creates input/target pairs
    for next-token prediction.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def evaluate(model, data: np.ndarray, block_size: int, batch_size: int,
             eval_batches: int, device: str, ctx):
    """Evaluate model on validation data."""
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def get_lr(step: int, warmup_steps: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # Cosine decay
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(cfg: TrainConfig) -> Dict:
    """
    Main training function.

    Returns a dict with training results and metrics.
    """
    print("=" * 60)
    print(f"TRAINING: {cfg.model_name}")
    print("=" * 60)

    # ---- Setup ----
    device = cfg.device
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    pt_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)
    ctx = torch.amp.autocast(device_type="cuda", dtype=pt_dtype) if device == "cuda" else torch.amp.autocast(device_type="cpu", dtype=pt_dtype)

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

    # ---- Create Model ----
    print(f"\n  Creating model '{cfg.model_name}'...")
    from model import MODEL_CONFIGS
    if cfg.model_name in MODEL_CONFIGS:
        model_config = MODEL_CONFIGS[cfg.model_name]
    else:
        raise ValueError(f"Unknown model: {cfg.model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

    model = GPT(model_config)
    n_params = model.get_num_params()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    model = model.to(device)

    if cfg.compile_model and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # ---- Optimizer ----
    print(f"\n  Configuring optimizer...")
    print(f"  Learning rate: {cfg.learning_rate}")
    optimizer = model.configure_optimizers(
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.learning_rate,
        betas=cfg.betas,
        device_type=device,
    )

    # GradScaler for float16 (not needed for bfloat16)
    scaler = torch.amp.GradScaler(enabled=(cfg.dtype == "float16"))

    min_lr = cfg.learning_rate * cfg.min_lr_ratio

    # ---- Metrics Storage ----
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
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "train_losses": [],       # (step, loss)
        "val_losses": [],         # (step, loss)
        "lr_history": [],         # (step, lr)
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
        # Set learning rate
        lr = get_lr(step, cfg.warmup_steps, total_steps, cfg.learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(cfg.gradient_accum_steps):
            x, y = get_batch(train_data, cfg.block_size, cfg.micro_batch_size, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.gradient_accum_steps  # Scale loss for accumulation

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

        # Record LR
        metrics["lr_history"].append((step, lr))

        # Logging
        if (step + 1) % cfg.log_interval == 0:
            avg_loss = running_loss / running_count
            elapsed = time.time() - t0
            tok_per_sec = total_tokens_processed / elapsed if elapsed > 0 else 0
            gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0

            metrics["train_losses"].append((step, avg_loss))

            print(f"  {step+1:>6} | {lr:>10.6f} | {avg_loss:>10.4f} | {'':>10} | {tok_per_sec:>10,.0f} | {gpu_mem:>8,.0f}")

            running_loss = 0.0
            running_count = 0

        # Validation evaluation
        if (step + 1) % cfg.eval_interval == 0 or step == total_steps - 1:
            val_loss = evaluate(model, val_data, cfg.block_size,
                              cfg.micro_batch_size, cfg.eval_batches, device, ctx)
            metrics["val_losses"].append((step, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            elapsed = time.time() - t0
            tok_per_sec = total_tokens_processed / elapsed if elapsed > 0 else 0
            gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0

            print(f"  {step+1:>6} | {lr:>10.6f} | {'':>10} | {val_loss:>10.4f} | {tok_per_sec:>10,.0f} | {gpu_mem:>8,.0f}")

    # ---- Final Metrics ----
    total_time = time.time() - t0
    metrics["wall_time_seconds"] = total_time
    metrics["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0
    metrics["tokens_per_second"] = total_tokens_processed / total_time if total_time > 0 else 0
    metrics["final_train_loss"] = metrics["train_losses"][-1][1] if metrics["train_losses"] else 0
    metrics["final_val_loss"] = metrics["val_losses"][-1][1] if metrics["val_losses"] else 0
    metrics["best_val_loss"] = best_val_loss

    print(f"\n  {'='*60}")
    print(f"  TRAINING COMPLETE: {cfg.model_name}")
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

        # Save metrics
        metrics_path = ckpt_dir / "metrics.json"
        # Convert tuples to lists for JSON serialization
        metrics_json = {k: v for k, v in metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2, default=str)

        # Save model checkpoint
        ckpt_path = ckpt_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': asdict(model_config),
            'train_config': asdict(cfg),
            'metrics': metrics,
        }, ckpt_path)

        print(f"  Checkpoint saved to: {ckpt_dir}")

    return metrics
