"""
Paint a target image with brush-stroke tokens (differentiable stroke rendering).

Pipeline:
  1. Optimize a set of colored capsule brush strokes (coarse->fine layers) to
     reconstruct a target raster (default: the Mona Lisa, public domain).
  2. VQ-quantize the optimized strokes into a discrete BRUSH-TOKEN codebook
     (k-means, K codes) and re-render -> the image from a finite brush vocab.
  3. Save target | continuous | tokenized comparison + per-layer progression +
     the token-id sequence and codebook.

This is the "paint with brush tokens" thread: a painting = sequence of discrete
brush-token IDs drawn from a small learned vocabulary.

Run:
    modal run paint.py                       # Mona Lisa, defaults
    modal run paint.py --steps 400 --codes 256
Pull artifacts:
    modal volume get brush-paint /out ./paint_out
"""

import modal

app = modal.App("brush-paint")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "requests", "matplotlib", "pillow")
)

volume = modal.Volume.from_name("brush-paint", create_if_missing=True)
OUT = "/out"

# Public-domain Mona Lisa (Wikimedia Commons). Wikimedia only serves a fixed
# set of thumbnail widths, so we try several standard ones, then the original.
_ML_FILE = "Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"
_ML_THUMB = ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/"
             f"{_ML_FILE}/{{w}}px-{_ML_FILE}")
MONA_LISA_CANDIDATES = [_ML_THUMB.format(w=w) for w in (800, 640, 1024, 320)]
MONA_LISA_CANDIDATES.append(
    f"https://upload.wikimedia.org/wikipedia/commons/e/ec/{_ML_FILE}")
MONA_LISA = MONA_LISA_CANDIDATES[0]


@app.function(image=image, volumes={OUT: volume}, gpu="T4", timeout=3600)
def paint(url: str = MONA_LISA, res: int = 224, steps: int = 300,
          codes: int = 256, seed: int = 0):
    import io
    import json
    import numpy as np
    import requests
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev).manual_seed(seed)
    print(f"device={dev} res={res} steps={steps} codes={codes}")

    # ---- load target ----
    hdr = {"User-Agent": "brush-paint/0.1 (research; contact juqu@dtu.dk)"}
    candidates = [url] if url != MONA_LISA else MONA_LISA_CANDIDATES
    content = None
    for cu in candidates:
        try:
            r = requests.get(cu, headers=hdr, timeout=120)
            r.raise_for_status()
            content = r.content
            print(f"fetched target from {cu}")
            break
        except Exception as e:
            print(f"fetch failed ({cu}): {e}")
    if content is None:
        raise RuntimeError("could not fetch target image from any candidate")
    img = Image.open(io.BytesIO(content)).convert("RGB")
    # center-crop to square, resize
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((res, res), Image.LANCZOS)
    target = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    target = target.permute(2, 0, 1).to(dev)  # (3,H,W)
    print(f"target loaded {img.size}")

    # normalized pixel grid in [0,1]
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, res, device=dev),
        torch.linspace(0, 1, res, device=dev),
        indexing="ij",
    )

    def seg_dist(x0, y0, x1, y1):
        # distance from every pixel to the segment (x0,y0)-(x1,y1)
        vx, vy = x1 - x0, y1 - y0
        wx, wy = xs - x0, ys - y0
        L2 = vx * vx + vy * vy + 1e-6
        t = ((wx * vx + wy * vy) / L2).clamp(0, 1)
        px, py = x0 + t * vx, y0 + t * vy
        return torch.sqrt((xs - px) ** 2 + (ys - py) ** 2 + 1e-9)

    def blur(t, k):
        c = t.shape[0]
        ker = torch.ones(c, 1, k, k, device=dev) / (k * k)
        return F.conv2d(t.unsqueeze(0), ker, padding=k // 2, groups=c).squeeze(0)

    # ---- stroke parameterization (raw -> constrained) ----
    class Layer:
        def __init__(self, n, scale):
            # init stroke centers at random pixels, color sampled from target
            cx = torch.rand(n, generator=g, device=dev)
            cy = torch.rand(n, generator=g, device=dev)
            ang = torch.rand(n, generator=g, device=dev) * 6.283
            half = scale * 0.5
            dx, dy = torch.cos(ang) * half, torch.sin(ang) * half
            self.p = {
                "x0": (cx - dx), "y0": (cy - dy),
                "x1": (cx + dx), "y1": (cy + dy),
                "w_raw": torch.full((n,), _inv_sig(scale * 0.5), device=dev),
                "col_raw": _sample_color(target, cx, cy, res),
                "a_raw": torch.full((n,), 1.5, device=dev),
            }
            for v in self.p.values():
                v.requires_grad_(True)
            self.scale = scale

        def params(self):
            return list(self.p.values())

        def strokes(self):
            w = torch.sigmoid(self.p["w_raw"]) * self.scale + 2.0 / res
            col = torch.sigmoid(self.p["col_raw"])           # (n,3)
            a = torch.sigmoid(self.p["a_raw"])
            return (self.p["x0"], self.p["y0"], self.p["x1"], self.p["y1"],
                    w, col, a)

    def render(canvas, strokes):
        x0, y0, x1, y1, w, col, a = strokes
        n = x0.shape[0]
        for i in range(n):
            d = seg_dist(x0[i], y0[i], x1[i], y1[i])
            cov = torch.sigmoid((w[i] * 0.5 - d) / (1.2 / res))  # (H,W)
            alpha = (a[i] * cov).unsqueeze(0)                     # (1,H,W)
            canvas = canvas * (1 - alpha) + col[i].view(3, 1, 1) * alpha
        return canvas

    # underpainting: heavily blurred target
    base = blur(target, 31).detach()

    # coarse-to-fine layers: (n_strokes, stroke scale)
    plan = [(80, 0.45), (140, 0.22), (220, 0.10)]
    layers = []
    snapshots = []
    canvas0 = base.clone()

    for li, (n, scale) in enumerate(plan):
        layer = Layer(n, scale)
        opt = torch.optim.Adam(layer.params(), lr=0.02)
        frozen = canvas0.detach()
        for step in range(steps):
            canvas = render(frozen.clone(), layer.strokes())
            loss = F.mse_loss(canvas, target)
            # coarse guidance term at low res
            loss = loss + 0.5 * F.mse_loss(blur(canvas, 9), blur(target, 9))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print(f"layer{li} step{step} loss={loss.item():.5f}")
        with torch.no_grad():
            canvas0 = render(canvas0, layer.strokes()).clamp(0, 1)
        layers.append(layer)
        snapshots.append(canvas0.detach().cpu().clone())
        print(f"layer{li} done ({n} strokes, scale={scale})")

    continuous = canvas0.clamp(0, 1)
    final_mse = F.mse_loss(continuous, target).item()
    print(f"continuous MSE={final_mse:.5f}")

    # ---- collect all strokes -> feature matrix for VQ ----
    feats, meta = [], []
    for layer in layers:
        x0, y0, x1, y1, w, col, a = [t.detach() for t in layer.strokes()]
        for i in range(x0.shape[0]):
            feats.append([x0[i].item(), y0[i].item(), x1[i].item(),
                          y1[i].item(), w[i].item(),
                          col[i, 0].item(), col[i, 1].item(), col[i, 2].item(),
                          a[i].item()])
            meta.append(len(meta))
    feats = torch.tensor(feats, device=dev)  # (N,9)
    N = feats.shape[0]
    print(f"total strokes N={N}")

    # standardize, k-means codebook
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    fz = (feats - mu) / sd
    K = min(codes, N)
    cb = fz[torch.randperm(N, generator=g, device=dev)[:K]].clone()
    for _ in range(30):
        d = torch.cdist(fz, cb)
        assign = d.argmin(1)
        for k in range(K):
            m = assign == k
            if m.any():
                cb[k] = fz[m].mean(0)
    token_ids = assign.tolist()
    used = len(set(token_ids))
    print(f"brush-token codebook: {used}/{K} codes used across {N} strokes")

    # de-standardize codebook -> stroke params, re-render tokenized painting
    cb_params = cb * sd + mu
    tok_strokes = cb_params[assign]  # (N,9) each stroke snapped to its code
    with torch.no_grad():
        tcanvas = base.clone()
        x0, y0, x1, y1 = (tok_strokes[:, 0], tok_strokes[:, 1],
                          tok_strokes[:, 2], tok_strokes[:, 3])
        w = tok_strokes[:, 4].clamp(min=2.0 / res)
        col = tok_strokes[:, 5:8].clamp(0, 1)
        a = tok_strokes[:, 8].clamp(0, 1)
        tcanvas = render(tcanvas, (x0, y0, x1, y1, w, col, a)).clamp(0, 1)
    tok_mse = F.mse_loss(tcanvas, target).item()
    print(f"tokenized MSE={tok_mse:.5f} (K={K} brush tokens)")

    # ---- save artifacts ----
    def to_np(t):
        return t.detach().cpu().permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(to_np(target)); ax[0].set_title("target")
    ax[1].imshow(to_np(continuous))
    ax[1].set_title(f"continuous (N={N})\nMSE={final_mse:.4f}")
    ax[2].imshow(to_np(tcanvas))
    ax[2].set_title(f"brush tokens (K={used})\nMSE={tok_mse:.4f}")
    for a_ in ax:
        a_.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/compare.png", dpi=130)

    fig2, ax2 = plt.subplots(1, len(snapshots) + 1, figsize=(4 * (len(snapshots) + 1), 4))
    ax2[0].imshow(to_np(base)); ax2[0].set_title("underpaint")
    for i, snap in enumerate(snapshots):
        ax2[i + 1].imshow(snap.permute(1, 2, 0).numpy())
        ax2[i + 1].set_title(f"after layer {i}")
    for a_ in ax2:
        a_.axis("off")
    fig2.tight_layout()
    fig2.savefig(f"{OUT}/progression.png", dpi=110)

    Image.fromarray((to_np(continuous) * 255).astype(np.uint8)).save(
        f"{OUT}/painting.png")
    Image.fromarray((to_np(tcanvas) * 255).astype(np.uint8)).save(
        f"{OUT}/painting_tokens.png")

    with open(f"{OUT}/tokens.json", "w") as f:
        json.dump({
            "token_ids": token_ids,
            "codebook_size": K,
            "codes_used": used,
            "n_strokes": N,
            "feature_order": ["x0", "y0", "x1", "y1", "w",
                              "r", "g", "b", "alpha"],
            "codebook": cb_params.detach().cpu().tolist(),
            "continuous_mse": final_mse,
            "tokenized_mse": tok_mse,
        }, f)

    volume.commit()
    print("saved compare.png progression.png painting*.png tokens.json")
    return {"n_strokes": N, "codes_used": used, "codebook_size": K,
            "continuous_mse": final_mse, "tokenized_mse": tok_mse}


def _inv_sig(y):
    import math
    y = min(max(y, 1e-4), 1 - 1e-4)
    return math.log(y / (1 - y))


def _sample_color(target, cx, cy, res):
    import torch
    ix = (cx * (res - 1)).long().clamp(0, res - 1)
    iy = (cy * (res - 1)).long().clamp(0, res - 1)
    col = target[:, iy, ix].t()  # (n,3)
    return torch.log(col.clamp(1e-3, 1 - 1e-3)
                     / (1 - col.clamp(1e-3, 1 - 1e-3)))  # inverse sigmoid


@app.local_entrypoint()
def main(res: int = 224, steps: int = 300, codes: int = 256):
    result = paint.remote(res=res, steps=steps, codes=codes)
    print("RESULT:", result)
    print("pull: modal volume get brush-paint /out ./paint_out")
