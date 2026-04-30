import json
data = json.load(open('checkpoints/scaling/all_results.json'))
for r in data:
    print(f"{r['model_name']:8s} | params={r['n_params']:>12,} | val={r['best_val_loss']:.4f} | train={r['final_train_loss']:.4f} | time={r['wall_time_seconds']:.0f}s | tok/s={r['tokens_per_second']:,.0f} | gpu={r['peak_gpu_memory_mb']:,.0f}MB")
