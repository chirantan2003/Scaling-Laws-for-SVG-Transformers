import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from model import GPT, GPTConfig
from tokenizers import Tokenizer
import matplotlib.pyplot as plt

def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Tokenizer
    tok_path = Path("data/tokenizer/svg_bpe_4096.json")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    
    # Load Model Checkpoint
    ckpt_path = Path("checkpoints/best_model/model.pt")
    if not ckpt_path.exists():
        print(f"WARNING: Checkpoint {ckpt_path} not found. Using untrained Large model for testing.")
        config = GPTConfig(n_layer=10, n_head=8, n_embd=512, d_ff=2048)
        model = GPT(config).to(device)
    else:
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        config = GPTConfig(n_layer=10, n_head=8, n_embd=512, d_ff=2048)
        model = GPT(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model, tokenizer, device

def generate(model, tokenizer, device, prompt, max_new_tokens=1024, temperature=1.0, top_k=None, top_p=None):
    # Encode prompt
    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if device=="cuda" else torch.float32):
            out_idx = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    
    # Decode
    out_tokens = out_idx[0].tolist()
    # truncate at eos or end
    if 2 in out_tokens: # assuming 2 is EOS
        out_tokens = out_tokens[:out_tokens.index(2)+1]
    
    return tokenizer.decode(out_tokens, skip_special_tokens=True)

def main():
    print("=" * 60)
    print("STEP 13: SVG Generation (Part 4)")
    print("=" * 60)
    
    model, tokenizer, device = load_model_and_tokenizer()
    out_dir = Path("outputs/samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    samples_record = []
    
    # 1. Unconditional Samples
    print("\nGenerating Unconditional Samples (prefix = '<svg')...")
    prefix = "<svg"
    for i in range(10):
        print(f"  Unconditional #{i+1} (Temp=0.8, Top_p=0.95)...")
        svg_code = generate(model, tokenizer, device, prefix, max_new_tokens=2000, temperature=0.8, top_p=0.95)
        path = out_dir / f"unconditional_{i+1}.svg"
        with open(path, "w") as f: f.write(svg_code)
        samples_record.append({"type": "unconditional", "id": i+1, "file": path.name, "prompt": prefix, "temperature": 0.8, "top_p": 0.95})
        
    # 2. Prefix-Conditioned Samples
    print("\nGenerating Prefix-Conditioned Samples...")
    
    prefixes = [
        # 1. Partial face
        {"name": "partial_face", "prompt": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">\n<circle cx="50" cy="50" r="40" fill="yellow" />\n<circle cx="35" cy="35" r="5" fill="black" />\n'},
        # 2. Open path
        {"name": "open_path", "prompt": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">\n<path d="M20,20 L80,20 L80,80" stroke="black" stroke-width="5" fill="none"'},
        # 3. Group with one shape
        {"name": "group_shapes", "prompt": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">\n<g fill="blue">\n<rect x="10" y="10" width="20" height="20" />\n'},
        # 4. Color theme (red/orange)
        {"name": "color_theme", "prompt": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">\n<rect x="2" y="2" width="20" height="20" fill="#FF5733" />\n<circle cx="12" cy="12" r="5" fill="#FFC300" />\n'},
        # 5. Icon skeleton
        {"name": "icon_skeleton", "prompt": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">\n<path d="M256,0 C114.6,0 0,114.6 0,256'}
    ]
    
    for p in prefixes:
        print(f"  Prefix: {p['name']}...")
        svg_code = generate(model, tokenizer, device, p['prompt'], max_new_tokens=1800, temperature=0.8, top_p=0.95)
        path = out_dir / f"prefix_{p['name']}.svg"
        with open(path, "w") as f: f.write(svg_code)
        
        # Save prompt context
        with open(out_dir / f"prefix_{p['name']}_PROMPT.svg", "w") as f:
            f.write(p['prompt'] + "\n</svg>")
            
        samples_record.append({"type": "prefix", "id": p['name'], "file": path.name, "prompt": p['prompt'], "temperature": 0.8, "top_p": 0.95})

    # 3. Temperature / Sampling Tests
    print("\nGenerating Temperature Tests...")
    prompt = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">\n<path d="'
    
    for temp in [0.5, 0.8, 1.0]:
        print(f"  Temp: {temp}...")
        svg_code = generate(model, tokenizer, device, prompt, max_new_tokens=1500, temperature=temp, top_p=0.95)
        path = out_dir / f"temp_{temp}.svg"
        with open(path, "w") as f: f.write(svg_code)
        samples_record.append({"type": "temperature", "id": str(temp), "file": path.name, "prompt": prompt, "temperature": temp, "top_p": 0.95})
        
    for top_k in [10, 50, 100]:
        print(f"  Top-K: {top_k}...")
        svg_code = generate(model, tokenizer, device, prompt, max_new_tokens=1500, temperature=0.8, top_k=top_k)
        path = out_dir / f"topk_{top_k}.svg"
        with open(path, "w") as f: f.write(svg_code)
        samples_record.append({"type": "top_k", "id": str(top_k), "file": path.name, "prompt": prompt, "temperature": 0.8, "top_k": top_k})

    # Save manifest
    with open(out_dir / "samples_manifest.json", "w") as f:
        json.dump(samples_record, f, indent=2)
        
    print(f"\nSaved {len(samples_record)} samples to {out_dir}/")

if __name__ == "__main__":
    main()
