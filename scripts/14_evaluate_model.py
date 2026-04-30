"""
Step 14: Evaluate Model (Part 4)

Calculates test perplexity and evaluates the generated SVG samples
for XML validity, structural validity, and renderability (via CairoSVG).
"""
import sys, os, json, time, re, threading
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from lxml import etree
import matplotlib.pyplot as plt

if os.name == 'nt':
    os.add_dll_directory(r"C:\msys64\mingw64\bin")

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"WARNING: CairoSVG failed to load ({e}).")
    CAIROSVG_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig


def _render_with_timeout(svg_bytes, png_path, timeout=10):
    """Run CairoSVG in a thread with a timeout to avoid infinite hangs."""
    result = {"ok": False}
    def _do_render():
        try:
            cairosvg.svg2png(bytestring=svg_bytes, write_to=str(png_path))
            result["ok"] = True
        except Exception:
            pass
    t = threading.Thread(target=_do_render, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result["ok"]


def _clean_svg(raw):
    """Try to produce a well-formed SVG from potentially truncated model output.

    Strategy:
      1. If the raw text already has </svg>, take everything up to and including
         the LAST </svg>.
      2. Otherwise, truncate at the last fully-closed tag (/>  or </…>),
         then append </svg>.
    """
    raw = raw.strip()

    # Case 1: already has a closing </svg>
    idx = raw.rfind("</svg>")
    if idx != -1:
        return raw[:idx + len("</svg>")]

    # Case 2: find the last self-closing (/>) or element-closing (</…>) tag
    #         so we don't leave a half-written attribute inside an open tag.
    last_selfclose = raw.rfind("/>")
    last_close = raw.rfind("</")
    if last_close != -1:
        end = raw.find(">", last_close)
        if end != -1:
            last_close = end
        else:
            last_close = -1

    cut = max(last_selfclose + 2 if last_selfclose != -1 else -1,
              last_close + 1 if last_close != -1 else -1)

    if cut > 0:
        return raw[:cut] + "\n</svg>"

    # Fallback: just slap on a closing tag
    return raw + "\n</svg>"


def calc_perplexity():
    print("\n--- 1. Test Set Perplexity ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path("checkpoints/best_model/model.pt")
    if not ckpt_path.exists():
        print("Model checkpoint not found. Skipping perplexity.")
        return None

    print("Loading model for perplexity evaluation...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = GPTConfig(n_layer=10, n_head=8, n_embd=512, d_ff=2048)
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_path = Path("data/splits/test.npy")
    if not test_path.exists():
        print("test.npy not found.")
        return None

    test_data = np.load(test_path, mmap_mode='r')
    block_size = 2048
    batch_size = 8
    eval_iters = min(100, len(test_data) // (block_size * batch_size))

    if eval_iters == 0:
        print("Not enough test data.")
        return None

    losses = []
    ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)

    print(f"Evaluating {eval_iters} batches...")
    with torch.no_grad():
        for i in range(eval_iters):
            ix = torch.randint(len(test_data) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy(test_data[j:j+block_size].astype(np.int64)) for j in ix]).to(device)
            y = torch.stack([torch.from_numpy(test_data[j+1:j+1+block_size].astype(np.int64)) for j in ix]).to(device)
            with ctx:
                _, loss = model(x, y)
                losses.append(loss.item())

    avg_loss = np.mean(losses)
    ppl = np.exp(avg_loss)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Perplexity: {ppl:.4f}")
    return ppl


def evaluate_samples():
    print("\n--- 2. Sample Evaluation ---")
    samples_dir = Path("outputs/samples")
    if not samples_dir.exists():
        print("No samples found."); return

    manifest_path = samples_dir / "samples_manifest.json"
    if not manifest_path.exists():
        print("Manifest not found."); return

    with open(manifest_path, "r") as f:
        samples = json.load(f)

    renders_dir = samples_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    stats = {"total": len(samples), "xml_valid": 0, "structural_valid": 0, "render_valid": 0}

    for s in samples:
        svg_file = samples_dir / s["file"]
        if not svg_file.exists(): continue

        raw = open(svg_file, "r").read()

        # --- Structural check (on raw output) ---
        has_svg_tags = raw.strip().startswith("<svg") and "</svg>" in raw
        if has_svg_tags:
            stats["structural_valid"] += 1

        # --- Clean / truncate for XML + render ---
        svg_content = _clean_svg(raw)

        # --- XML validity ---
        xml_valid = False
        try:
            etree.fromstring(svg_content.encode('utf-8'))
            xml_valid = True
            stats["xml_valid"] += 1
        except etree.XMLSyntaxError:
            pass

        # --- Render (only if XML valid, with 10-s timeout) ---
        render_valid = False
        if CAIROSVG_AVAILABLE and xml_valid:
            png_out = renders_dir / f"{svg_file.stem}.png"
            render_valid = _render_with_timeout(svg_content.encode('utf-8'), png_out, timeout=10)
            if render_valid:
                stats["render_valid"] += 1
            else:
                # Clean up partial file if timeout
                if png_out.exists(): png_out.unlink()

        render_status = 'PASS' if render_valid else 'FAIL'
        if not CAIROSVG_AVAILABLE: render_status = 'SKIPPED'
        print(f"  {s['file']}: XML={'PASS' if xml_valid else 'FAIL'} | Render={render_status}")

    print(f"\n--- Final Metrics ---")
    print(f"Total Evaluated: {stats['total']}")
    print(f"Structural Validity (<svg> root): {stats['structural_valid']} / {stats['total']} ({stats['structural_valid']/stats['total']*100:.1f}%)")
    print(f"XML Validity: {stats['xml_valid']} / {stats['total']} ({stats['xml_valid']/stats['total']*100:.1f}%)")
    if CAIROSVG_AVAILABLE:
        print(f"Render Success: {stats['render_valid']} / {stats['total']} ({stats['render_valid']/stats['total']*100:.1f}%)")
    else:
        print(f"Render Success: SKIPPED (CairoSVG unavailable)")

    with open(samples_dir / "evaluation_metrics.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def generate_sample_grid():
    """Create a rendered grid of unconditional samples."""
    print("\n--- 3. Generating Sample Grid ---")
    renders_dir = Path("outputs/samples/renders")
    out_dir = Path("outputs/samples")

    # Unconditional grid
    uncond_pngs = sorted(renders_dir.glob("unconditional_*.png"))
    if not uncond_pngs:
        print("No rendered unconditional PNGs found. Skipping grid.")
        return

    from PIL import Image
    imgs = []
    for p in uncond_pngs:
        try:
            img = Image.open(p).convert("RGBA")
            # Resize to uniform size
            img = img.resize((200, 200), Image.LANCZOS)
            imgs.append((p.stem, img))
        except Exception:
            pass

    if not imgs:
        print("No valid PNGs to grid."); return

    # Create grid: 2 rows x 5 cols
    cols = min(5, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    cell = 220
    grid = Image.new("RGBA", (cols * cell, rows * cell), (255, 255, 255, 255))

    for i, (name, img) in enumerate(imgs):
        r, c = i // cols, i % cols
        grid.paste(img, (c * cell + 10, r * cell + 10))

    grid_path = out_dir / "unconditional_grid.png"
    grid.save(str(grid_path))
    print(f"Unconditional grid saved: {grid_path}")

    # Temperature comparison grid
    temp_pngs = []
    for t in ["0.5", "0.8", "1.0"]:
        p = renders_dir / f"temp_{t}.png"
        if p.exists(): temp_pngs.append((f"T={t}", p))
    for k in ["10", "50", "100"]:
        p = renders_dir / f"topk_{k}.png"
        if p.exists(): temp_pngs.append((f"K={k}", p))

    if temp_pngs:
        n = len(temp_pngs)
        tgrid = Image.new("RGBA", (n * cell, cell), (255, 255, 255, 255))
        for i, (label, p) in enumerate(temp_pngs):
            try:
                img = Image.open(p).convert("RGBA").resize((200, 200), Image.LANCZOS)
                tgrid.paste(img, (i * cell + 10, 10))
            except Exception:
                pass
        tgrid_path = out_dir / "temperature_grid.png"
        tgrid.save(str(tgrid_path))
        print(f"Temperature grid saved: {tgrid_path}")

    # Prefix completion grid
    prefix_pngs = sorted(renders_dir.glob("prefix_*.png"))
    if prefix_pngs:
        n = len(prefix_pngs)
        pgrid = Image.new("RGBA", (n * cell, cell), (255, 255, 255, 255))
        for i, p in enumerate(prefix_pngs):
            try:
                img = Image.open(p).convert("RGBA").resize((200, 200), Image.LANCZOS)
                pgrid.paste(img, (i * cell + 10, 10))
            except Exception:
                pass
        pgrid_path = out_dir / "prefix_grid.png"
        pgrid.save(str(pgrid_path))
        print(f"Prefix grid saved: {pgrid_path}")


def main():
    print("=" * 60)
    print("STEP 14: Model Evaluation (Part 4)")
    print("=" * 60)

    calc_perplexity()
    evaluate_samples()
    generate_sample_grid()

if __name__ == "__main__":
    main()
