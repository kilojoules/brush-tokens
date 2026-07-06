"""
Learn a SHARED brush-token vocabulary and paint held-out images with it.

The per-image k-means in paint.py is not a vocabulary (a different image -> a
different codebook, near-uniform code usage, ~1/3 singletons). This makes the
claim real:

  Stage A (build vocabulary): paint a CORPUS of public-domain paintings with a
    differentiable 2-segment stroke painter; collect each stroke's canonical
    SHAPE descriptor (length, width, bend) -- translation/rotation invariant,
    color excluded; k-means -> K shared brush codes. A brush is a mark, not a
    color; you dip the same brush in any paint.

  Stage B (use it): paint HELD-OUT images (e.g. the Mona Lisa, never seen in the
    corpus). Solve continuous, snap each stroke's shape to the nearest of the K
    fixed brushes, then refine placement (position/orientation/color) only. The
    reconstruction uses ONLY the shared alphabet.

Reported: the rendered brush alphabet, held-out reconstructions, and vocabulary
quality (strokes/code, code-usage entropy, singletons, cross-image reuse).

Run:
    modal run vocab.py
    modal volume get brush-vocab /out ./vocab_out
"""

import modal

app = modal.App("brush-vocab")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "requests", "matplotlib", "pillow")
)

volume = modal.Volume.from_name("brush-vocab", create_if_missing=True)
OUT = "/out"

# Wikimedia Special:FilePath 302-redirects to the original file (arbitrary thumb
# widths are blocked), so we fetch originals by filename. Corpus = training set;
# held-out = never in the corpus, the real test of a shared vocabulary.
FILEPATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"
CORPUS = [
    "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    "1665_Girl_with_a_Pearl_Earring.jpg",
    "Grant_Wood_-_American_Gothic_-_Google_Art_Project.jpg",
    "A_Sunday_on_La_Grande_Jatte,_Georges_Seurat,_1884.jpg",
    "The_Scream.jpg",
    "Claude_Monet,_Impression,_soleil_levant.jpg",
]
HELDOUT = [
    "Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg",
    "Vermeer-view-of-delft.jpg",
]


@app.function(image=image, volumes={OUT: volume}, gpu="A10G", timeout=5400)
def run(k_brushes: int = 32, corpus_res: int = 160, heldout_res: int = 200,
        corpus_steps: int = 120, heldout_steps: int = 180, seed: int = 0):
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
    torch.manual_seed(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    print(f"device={dev} K={k_brushes}")
    hdr = {"User-Agent": "brush-vocab/0.1 (research; juqu@dtu.dk)"}

    Image.MAX_IMAGE_PIXELS = None  # originals are gigapixel; guard off
    def fetch(fname, res):
        r = requests.get(FILEPATH + fname, headers=hdr, timeout=300,
                         allow_redirects=True)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        img.draft("RGB", (res * 3, res * 3))  # decode JPEG at reduced scale
        img = img.convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((res, res), Image.LANCZOS)
        t = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)
        return t.to(dev)

    def grids(res):
        ys, xs = torch.meshgrid(torch.linspace(0, 1, res, device=dev),
                                torch.linspace(0, 1, res, device=dev),
                                indexing="ij")
        return xs, ys

    def seg_dist(xs, ys, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        wx, wy = xs - ax, ys - ay
        L2 = vx * vx + vy * vy + 1e-6
        t = ((wx * vx + wy * vy) / L2).clamp(0, 1)
        px, py = ax + t * vx, ay + t * vy
        return torch.sqrt((xs - px) ** 2 + (ys - py) ** 2 + 1e-9)

    def blur(t, k):
        c = t.shape[0]
        ker = torch.ones(c, 1, k, k, device=dev) / (k * k)
        return F.conv2d(t.unsqueeze(0), ker, padding=k // 2, groups=c).squeeze(0)

    # world points of a 3-point (2-segment) stroke from instance+shape params
    def world_points(o, theta, L, bx, by):
        c, s = torch.cos(theta), torch.sin(theta)
        p0 = o                                            # (N,2)
        p1 = o + torch.stack([L * c, L * s], -1)
        mx, my = L * 0.5 + bx, by
        pm = o + torch.stack([mx * c - my * s, mx * s + my * c], -1)
        return p0, pm, p1

    def render(canvas, xs, ys, p0, pm, p1, w, col, a):
        n = p0.shape[0]
        soft = 1.2 / canvas.shape[-1]
        for i in range(n):
            d1 = seg_dist(xs, ys, p0[i, 0], p0[i, 1], pm[i, 0], pm[i, 1])
            d2 = seg_dist(xs, ys, pm[i, 0], pm[i, 1], p1[i, 0], p1[i, 1])
            d = torch.minimum(d1, d2)
            cov = torch.sigmoid((w[i] * 0.5 - d) / soft)
            alpha = (a[i] * cov).unsqueeze(0)
            canvas = canvas * (1 - alpha) + col[i].view(3, 1, 1) * alpha
        return canvas

    # a set of strokes with raw (unconstrained) params -> constrained getters
    class Strokes:
        def __init__(self, n, scale, target, res, shape_fixed=None):
            xs, ys = grids(res)
            ox = torch.rand(n, generator=g, device=dev)
            oy = torch.rand(n, generator=g, device=dev)
            self.o = torch.stack([ox, oy], -1)
            self.theta = torch.rand(n, generator=g, device=dev) * 6.283
            # color init sampled from target at origin
            ix = (ox * (res - 1)).long(); iy = (oy * (res - 1)).long()
            col0 = target[:, iy, ix].t().clamp(1e-3, 1 - 1e-3)
            self.col_raw = torch.log(col0 / (1 - col0))
            self.a_raw = torch.full((n,), 1.2, device=dev)
            self.shape_fixed = shape_fixed  # (n,3) L,w,bend? -> (L,w,bx,by) via 4
            if shape_fixed is None:
                self.L_raw = torch.full((n,), _isp(scale), device=dev)
                self.w_raw = torch.full((n,), _isp(scale * 0.5), device=dev)
                self.bx = torch.zeros(n, device=dev)
                self.by = torch.zeros(n, device=dev)
            for t in self._trainable():
                t.requires_grad_(True)
            self.res = res
            self.xs, self.ys = xs, ys

        def _trainable(self):
            base = [self.o, self.theta, self.col_raw, self.a_raw]
            if self.shape_fixed is None:
                base += [self.L_raw, self.w_raw, self.bx, self.by]
            return base

        def shape(self):
            if self.shape_fixed is not None:
                L, w, bx, by = (self.shape_fixed[:, 0], self.shape_fixed[:, 1],
                                self.shape_fixed[:, 2], self.shape_fixed[:, 3])
            else:
                L = F.softplus(self.L_raw) + 3.0 / self.res
                w = F.softplus(self.w_raw) + 2.0 / self.res
                bx, by = self.bx, self.by
            return L, w, bx, by

        def forward(self, canvas):
            L, w, bx, by = self.shape()
            p0, pm, p1 = world_points(self.o, self.theta, L, bx, by)
            col = torch.sigmoid(self.col_raw)
            a = torch.sigmoid(self.a_raw)
            return render(canvas, self.xs, self.ys, p0, pm, p1, w, col, a)

        def descriptors(self):  # canonical shape: (n,4) L,w,bx,by
            L, w, bx, by = self.shape()
            return torch.stack([L, w, bx, by], -1).detach()

    def paint(target, plan, steps, res, shape_codebook=None, assign_from=None):
        # solve strokes for one image; if shape_codebook given, freeze each
        # stroke's shape to its assigned code and refine placement only.
        base = torch.ones(3, res, res, device=dev)
        canvas0 = base.clone()
        all_strokes = []
        for li, (n, scale) in enumerate(plan):
            if shape_codebook is None:
                st = Strokes(n, scale, target, res)
            else:
                # init continuous, snap shape to nearest code, freeze it
                tmp = Strokes(n, scale, target, res)
                opt = torch.optim.Adam(tmp._trainable(), lr=0.02)
                for _ in range(max(30, steps // 3)):
                    canvas = tmp.forward(canvas0.detach().clone())
                    loss = F.mse_loss(canvas, target)
                    opt.zero_grad(); loss.backward(); opt.step()
                desc = tmp.descriptors()
                idx = torch.cdist((desc - cb_mu) / cb_sd, shape_codebook).argmin(1)
                fixed = (shape_codebook[idx] * cb_sd + cb_mu)
                st = Strokes(n, scale, target, res, shape_fixed=fixed)
                # warm-start placement from tmp
                with torch.no_grad():
                    st.o.copy_(tmp.o); st.theta.copy_(tmp.theta)
                    st.col_raw.copy_(tmp.col_raw); st.a_raw.copy_(tmp.a_raw)
                for t in st._trainable():
                    t.requires_grad_(True)
                st._assigned = idx
            opt = torch.optim.Adam(st._trainable(), lr=0.02)
            frozen = canvas0.detach()
            for step in range(steps):
                canvas = st.forward(frozen.clone())
                loss = F.mse_loss(canvas, target)
                loss = loss + 0.5 * F.mse_loss(blur(canvas, 9), blur(target, 9))
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                canvas0 = st.forward(canvas0).clamp(0, 1)
            all_strokes.append(st)
        return canvas0.clamp(0, 1), all_strokes

    # ---------------- Stage A: build the shared vocabulary ----------------
    corpus_plan = [(120, 0.5), (180, 0.27), (240, 0.12)]
    descs, per_img = [], []
    loaded_corpus = []
    for fname in CORPUS:
        try:
            tgt = fetch(fname, corpus_res)
        except Exception as e:
            print(f"corpus fetch failed {fname}: {e}"); continue
        rec, strokes = paint(tgt, corpus_plan, corpus_steps, corpus_res)
        mse = F.mse_loss(rec, tgt).item()
        for st in strokes:
            d = st.descriptors()
            descs.append(d)
            per_img.append(torch.full((d.shape[0],), len(loaded_corpus),
                                      device=dev))
        loaded_corpus.append(fname)
        print(f"corpus painted {fname[:30]} strokes={sum(s.o.shape[0] for s in strokes)} MSE={mse:.4f}")
    if len(loaded_corpus) < 2:
        raise RuntimeError("need >=2 corpus images")
    descs = torch.cat(descs, 0)          # (M,4)
    img_of = torch.cat(per_img, 0)       # (M,)
    M = descs.shape[0]
    print(f"corpus strokes M={M} across {len(loaded_corpus)} images")

    cb_mu, cb_sd = descs.mean(0), descs.std(0) + 1e-6
    dz = (descs - cb_mu) / cb_sd
    K = min(k_brushes, M)
    cb = dz[torch.randperm(M, generator=g, device=dev)[:K]].clone()
    for _ in range(60):
        assign = torch.cdist(dz, cb).argmin(1)
        for k in range(K):
            m = assign == k
            if m.any():
                cb[k] = dz[m].mean(0)
    # vocabulary quality
    from collections import Counter
    cnt = Counter(assign.tolist())
    counts = torch.tensor([cnt.get(k, 0) for k in range(K)], dtype=torch.float)
    probs = counts / counts.sum()
    ent = float(-(probs[probs > 0] * probs[probs > 0].log2()).sum())
    singles = int((counts == 1).sum())
    reuse = [len(set(img_of[assign == k].tolist())) for k in range(K)]
    cross = sum(1 for r in reuse if r >= 2)
    print(f"VOCAB K={K}: entropy {ent:.2f}/{np.log2(K):.2f} bits, "
          f"strokes/code median {int(counts.median())}, singletons {singles}, "
          f"codes reused across >=2 images: {cross}/{K}")

    # ---------------- Stage B: paint held-out with fixed vocabulary ----------
    heldout_plan = [(140, 0.5), (200, 0.27), (300, 0.12), (340, 0.06)]
    results = []
    for fname in HELDOUT:
        try:
            tgt = fetch(fname, heldout_res)
        except Exception as e:
            print(f"heldout fetch failed {fname}: {e}"); continue
        # free reconstruction (upper bound) then vocabulary-constrained
        rec_free, _ = paint(tgt, heldout_plan, heldout_steps, heldout_res)
        rec_vocab, vstrokes = paint(tgt, heldout_plan, heldout_steps,
                                    heldout_res, shape_codebook=cb)
        used = sorted(set(int(i) for st in vstrokes for i in st._assigned))
        mse_free = F.mse_loss(rec_free, tgt).item()
        mse_vocab = F.mse_loss(rec_vocab, tgt).item()
        print(f"HELDOUT {fname[:28]}: free MSE {mse_free:.4f} | "
              f"vocab MSE {mse_vocab:.4f} | brushes used {len(used)}/{K}")
        results.append((fname, tgt.cpu(), rec_free.cpu(), rec_vocab.cpu(),
                        mse_free, mse_vocab, len(used)))

    # ---------------- render the brush alphabet ----------------
    tile = 64
    cb_real = cb * cb_sd + cb_mu   # (K,4) L,w,bx,by
    xs, ys = grids(tile)
    cols = 8
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = np.array(axes).reshape(-1)
    for k in range(K):
        L, w, bx, by = [float(v) for v in cb_real[k]]
        L = max(L, 0.2); w = max(w, 0.02)
        o = torch.tensor([[0.5 - L / 2, 0.5]], device=dev)
        th = torch.zeros(1, device=dev)
        Lt = torch.tensor([L], device=dev); bxt = torch.tensor([bx], device=dev)
        byt = torch.tensor([by], device=dev)
        p0, pm, p1 = world_points(o, th, Lt, bxt, byt)
        canvas = torch.ones(3, tile, tile, device=dev)
        canvas = render(canvas, xs, ys, p0, pm, p1,
                        torch.tensor([w], device=dev),
                        torch.tensor([[0.1, 0.1, 0.1]], device=dev),
                        torch.tensor([1.0], device=dev))
        axes[k].imshow(canvas.permute(1, 2, 0).cpu().numpy())
        axes[k].set_title(str(k), fontsize=6); axes[k].axis("off")
    for j in range(K, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Shared brush alphabet (K={K})", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/brush_alphabet.png", dpi=130)

    # held-out comparison figure
    if results:
        fig2, ax2 = plt.subplots(len(results), 3,
                                 figsize=(9, 3 * len(results)))
        ax2 = np.array(ax2).reshape(len(results), 3)
        for i, (fn, tgt, rf, rv, mf, mv, nu) in enumerate(results):
            for j, (im, title) in enumerate([
                    (tgt, "target"), (rf, f"free MSE {mf:.3f}"),
                    (rv, f"vocab MSE {mv:.3f} ({nu} brushes)")]):
                ax2[i, j].imshow(im.permute(1, 2, 0).numpy())
                ax2[i, j].set_title(title, fontsize=8); ax2[i, j].axis("off")
        fig2.tight_layout()
        fig2.savefig(f"{OUT}/heldout_vocab.png", dpi=120)

    with open(f"{OUT}/vocab.json", "w") as f:
        json.dump({
            "k_brushes": K,
            "descriptor_order": ["length", "width", "bend_x", "bend_y"],
            "codebook_standardized": cb.cpu().tolist(),
            "codebook_mu": cb_mu.cpu().tolist(),
            "codebook_sd": cb_sd.cpu().tolist(),
            "corpus": loaded_corpus,
            "corpus_strokes": M,
            "usage_entropy_bits": ent,
            "max_entropy_bits": float(np.log2(K)),
            "singletons": singles,
            "codes_reused_across_images": cross,
            "heldout": [{"file": fn, "free_mse": mf, "vocab_mse": mv,
                         "brushes_used": nu}
                        for (fn, _, _, _, mf, mv, nu) in results],
        }, f)

    volume.commit()
    print("saved brush_alphabet.png heldout_vocab.png vocab.json")
    return {"K": K, "corpus_images": len(loaded_corpus), "corpus_strokes": M,
            "entropy_bits": round(ent, 2), "singletons": singles,
            "cross_image_codes": cross,
            "heldout": [(fn[:20], round(mv, 4)) for (fn, _, _, _, _, mv, _)
                        in results]}


def _isp(y):  # inverse softplus
    import math
    return math.log(math.expm1(max(y, 1e-3)))


@app.local_entrypoint()
def main(k_brushes: int = 32):
    print("RESULT:", run.remote(k_brushes=k_brushes))
    print("pull: modal volume get brush-vocab /out ./vocab_out")
