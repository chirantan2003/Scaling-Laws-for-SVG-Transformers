"""Quick verification: instantiate all muP models and check param counts + forward pass."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import MODEL_CONFIGS
from model_mup import MuGPT, create_mup_model

print("muP Model Parameter Counts")
print("=" * 60)
print(f"{'Name':<8} {'Params':>12} {'d_model':>8} {'Layers':>7} {'Heads':>6} {'d_ff':>6}")
print("-" * 60)

for name in MODEL_CONFIGS:
    model = create_mup_model(name)
    n = model.get_num_params()
    cfg = model.config
    d_ff = cfg.d_ff or 4 * cfg.n_embd
    print(f"{name:<8} {n:>12,} {cfg.n_embd:>8} {cfg.n_layer:>7} {cfg.n_head:>6} {d_ff:>6}")

    # Check that infshape is set (muP is configured)
    has_infshape = hasattr(list(model.parameters())[0], 'infshape')
    print(f"         muP infshape set: {has_infshape}")
    del model

# Test forward pass with tiny model
print("\nForward pass test (muP Tiny)...")
model = create_mup_model("tiny")
x = torch.randint(0, 4096, (2, 64))
logits, loss = model(x, x)
print(f"  Input:  {x.shape}")
print(f"  Logits: {logits.shape}")
print(f"  Loss:   {loss.item():.4f}")
print("  [OK] muP forward pass works!")
