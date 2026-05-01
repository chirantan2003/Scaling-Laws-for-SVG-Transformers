import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import GPT, MODEL_CONFIGS

print("Model Parameter Counts")
print("=" * 60)
print(f"{'Name':<8} {'Params':>12} {'d_model':>8} {'Layers':>7} {'Heads':>6} {'d_ff':>6}")
print("-" * 60)

for name, config in MODEL_CONFIGS.items():
    model = GPT(config)
    n = model.get_num_params()
    d_ff = config.d_ff or 4 * config.n_embd
    print(f"{name:<8} {n:>12,} {config.n_embd:>8} {config.n_layer:>7} {config.n_head:>6} {d_ff:>6}")
    del model

# Test forward pass with tiny model
print("\nForward pass test (Tiny)...")
config = MODEL_CONFIGS["tiny"]
model = GPT(config)
x = torch.randint(0, config.vocab_size, (2, 64))
logits, loss = model(x, x)
print(f"  Input:  {x.shape}")
print(f"  Logits: {logits.shape}")
print(f"  Loss:   {loss.item():.4f}")
print("  [OK] Forward pass works!")
