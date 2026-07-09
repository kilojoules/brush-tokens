"""
Vocabulary v2: scale-invariant brushes + scale bins + shared color palette,
learned over a MIXED corpus (paintings AND line-drawing task states), evaluated
as a full 6-slot token round-trip on held-out images from BOTH domains.

Changes vs vocab.py (Phase 0 of docs/training-program.md):

  1. Scale invariance. A brush shape is now the RATIO descriptor
     (width/length, bend_x/length, bend_y/length); length itself becomes a
     separate discrete SCALE slot (16 log-spaced bins). One brush, many sizes.
  2. Shared color palette. k-means over corpus stroke RGBA -> 256 codes.
     Brush and color stay independent tokens (a brush is a mark, not a paint).
  3. Mixed corpus. The downstream tasks are mazes/folds (line drawings), not
     Vermeer: corpus = 6 paintings + 4 synthetic mazes + 2 fold diagrams,
     held-out = Mona Lisa + Vermeer + 1 unseen maze + 1 unseen fold.
  4. Token round-trip eval. Held-out strokes are quantized on ALL six slots
     (brush 32, scale 16, x 64, y 64, theta 32, color 256) and re-rendered.
     Reported per image: free MSE (continuous upper bound), vocab MSE (shape+
     scale snapped, placement/color refined), token MSE (everything snapped).

Gate (docs/training-program.md Phase 0): held-out vocab MSE within ~1.5x of
the free fit on BOTH a painting and a maze render.

Run:
    modal run vocab2.py
    modal volume get brush-vocab /out ./vocab_out
"""

import modal

app = modal.App("brush-vocab2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "requests", "matplotlib", "pillow")
)

volume = modal.Volume.from_name("brush-vocab", create_if_missing=True)
OUT = "/out"

FILEPATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"
CORPUS_PAINTINGS = [
    "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    "1665_Girl_with_a_Pearl_Earring.jpg",
    "Grant_Wood_-_American_Gothic_-_Google_Art_Project.jpg",
    "A_Sunday_on_La_Grande_Jatte,_Georges_Seurat,_1884.jpg",
    "The_Scream.jpg",
    "Claude_Monet,_Impression,_soleil_levant.jpg",
]
HELDOUT_PAINTINGS = [
    "Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg",
    "Vermeer-view-of-delft.jpg",
]
# synthetic line-drawing task states: (kind, seed, size)
CORPUS_TASKS = [("maze", 1, 6), ("maze", 2, 7), ("maze", 3, 8), ("maze", 4, 9),
                ("fold", 5, 0), ("fold", 6, 0)]
HELDOUT_TASKS = [("maze", 777, 8), ("fold", 888, 0)]

K_BRUSHES = 32
K_SCALE = 16
K_PALETTE = 256
K_THETA = 32
GRID = 64


@app.function(image=image, volumes={OUT: volume}, gpu="A10G", timeout=10800)
def run(corpus_res: int = 160, heldout_res: int = 200,
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
    print(f"device={dev} K={K_BRUSHES} scale_bins={K_SCALE} palette={K_PALETTE}")
    hdr = {"User-Agent": "brush-vocab/0.2 (research; juqu@dtu.dk)"}

    Image.MAX_IMAGE_PIXELS = None
    def fetch(fname, res):
        r = requests.get(FILEPATH + fname, headers=hdr, timeout=300,
                         allow_redirects=True)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        img.draft("RGB", (res * 3, res * 3))
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

    def world_points(o, theta, L, bx, by):
        c, s = torch.cos(theta), torch.sin(theta)
        p0 = o
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

    # ---------------- synthetic task-state targets (we own the generator) ----
    def draw_segments(segs, res, width=0.011, shade=None):
        """Render line segments (in [0,1]^2) black-on-white; optional shaded
        polygon underneath (fold diagrams)."""
        xs, ys = grids(res)
        canvas = torch.ones(3, res, res, device=dev)
        if shade is not None:  # shade: (poly_pts, gray) painted as fat strokes
            pts, gray = shade
            for i in range(len(pts) - 1):
                a = torch.tensor(pts[i], device=dev)
                b = torch.tensor(pts[i + 1], device=dev)
                d = seg_dist(xs, ys, a[0], a[1], b[0], b[1])
                cov = torch.sigmoid((0.04 - d) / (1.2 / res))
                canvas = canvas * (1 - cov * 0.5) + gray * (cov * 0.5)
        for (a, b) in segs:
            a = torch.tensor(a, device=dev, dtype=torch.float32)
            b = torch.tensor(b, device=dev, dtype=torch.float32)
            d = seg_dist(xs, ys, a[0], a[1], b[0], b[1])
            cov = torch.sigmoid((width * 0.5 - d) / (1.2 / res))
            ink = torch.full((3,), 0.08, device=dev)
            canvas = canvas * (1 - cov) + ink.view(3, 1, 1) * cov
        return canvas.clamp(0, 1)

    def maze_target(seed_, n, res):
        rng = np.random.default_rng(seed_)
        visited = np.zeros((n, n), bool)
        wh = np.ones((n + 1, n), bool)   # horizontal walls
        wv = np.ones((n, n + 1), bool)   # vertical walls
        stack = [(0, 0)]
        visited[0, 0] = True
        while stack:
            x, y = stack[-1]
            nbrs = [(x + dx, y + dy, dx, dy)
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= x + dx < n and 0 <= y + dy < n
                    and not visited[y + dy, x + dx]]
            if not nbrs:
                stack.pop()
                continue
            nx, ny, dx, dy = nbrs[rng.integers(len(nbrs))]
            if dx == 1:
                wv[y, x + 1] = False
            elif dx == -1:
                wv[y, x] = False
            elif dy == 1:
                wh[y + 1, x] = False
            else:
                wh[y, x] = False
            visited[ny, nx] = True
            stack.append((nx, ny))
        m, span = 0.08, 0.84
        def pt(i, j):
            return (m + span * i / n, m + span * j / n)
        segs = []
        for j in range(n + 1):
            for i in range(n):
                if wh[j, i]:
                    segs.append((pt(i, j), pt(i + 1, j)))
        for j in range(n):
            for i in range(n + 1):
                if wv[j, i]:
                    segs.append((pt(i, j), pt(i, j + 1)))
        return draw_segments(segs, res)

    def fold_target(seed_, res):
        rng = np.random.default_rng(seed_)
        m = 0.12
        sq = [(m, m), (1 - m, m), (1 - m, 1 - m), (m, 1 - m), (m, m)]
        segs = [(sq[i], sq[i + 1]) for i in range(4)]
        # 1-2 fold lines: chords between random points on different edges
        edges = [(sq[i], sq[i + 1]) for i in range(4)]
        for _ in range(int(rng.integers(1, 3))):
            e1, e2 = rng.choice(4, 2, replace=False)
            t1, t2 = rng.uniform(0.15, 0.85, 2)
            def lerp(e, t):
                (ax, ay), (bx, by) = edges[e]
                return (ax + t * (bx - ax), ay + t * (by - ay))
            segs.append((lerp(e1, t1), lerp(e2, t2)))
        # folded corner: shaded triangle
        c = int(rng.integers(4))
        (cx, cy) = sq[c]
        (px, py) = sq[(c + 1) % 4]
        (qx, qy) = sq[(c - 1) % 4]
        f = rng.uniform(0.25, 0.45)
        p2 = (cx + f * (px - cx), cy + f * (py - cy))
        q2 = (cx + f * (qx - cx), cy + f * (qy - cy))
        segs.append((p2, q2))
        return draw_segments(segs, res, shade=([p2, q2, (cx, cy), p2], 0.65))

    # ---------------- strokes / painter (as vocab.py, shape via getters) -----
    class Strokes:
        def __init__(self, n, scale, target, res, shape_fixed=None):
            xs, ys = grids(res)
            ox = torch.rand(n, generator=g, device=dev)
            oy = torch.rand(n, generator=g, device=dev)
            self.o = torch.stack([ox, oy], -1)
            self.theta = torch.rand(n, generator=g, device=dev) * 6.283
            ix = (ox * (res - 1)).long(); iy = (oy * (res - 1)).long()
            col0 = target[:, iy, ix].t().clamp(1e-3, 1 - 1e-3)
            self.col_raw = torch.log(col0 / (1 - col0))
            self.a_raw = torch.full((n,), 1.2, device=dev)
            self.shape_fixed = shape_fixed  # (n,4) L,w,bx,by frozen
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
                w = F.softplus(self.w_raw) + 1.5 / self.res
                bx, by = self.bx, self.by
            return L, w, bx, by

        def forward(self, canvas):
            L, w, bx, by = self.shape()
            p0, pm, p1 = world_points(self.o, self.theta, L, bx, by)
            col = torch.sigmoid(self.col_raw)
            a = torch.sigmoid(self.a_raw)
            return render(canvas, self.xs, self.ys, p0, pm, p1, w, col, a)

        def descriptors(self):
            """v2 canonical shape: RATIOS (w/L, bx/L, by/L) + separate L."""
            L, w, bx, by = self.shape()
            ratios = torch.stack([w / L, bx / L, by / L], -1).detach()
            return ratios, L.detach()

        def colors_rgba(self):
            col = torch.sigmoid(self.col_raw).detach()
            a = torch.sigmoid(self.a_raw).detach().unsqueeze(-1)
            return torch.cat([col, a], -1)  # (n,4)

    def snap_shape_scale(ratios, L):
        """ratios (n,3), L (n,) -> frozen (n,4) L,w,bx,by from codebooks."""
        idx = torch.cdist((ratios - cb_mu) / cb_sd, cb).argmin(1)
        r = cb[idx] * cb_sd + cb_mu                       # (n,3) w/L,bx/L,by/L
        sbin = (torch.log(L.clamp_min(1e-4)).unsqueeze(1)
                - scale_bins.log().unsqueeze(0)).abs().argmin(1)
        Lq = scale_bins[sbin]
        fixed = torch.stack([Lq, r[:, 0] * Lq, r[:, 1] * Lq, r[:, 2] * Lq], -1)
        return fixed, idx, sbin

    def paint(target, plan, steps, res, constrain=False):
        base = torch.ones(3, res, res, device=dev)
        canvas0 = base.clone()
        all_strokes = []
        for (n, scale) in plan:
            if not constrain:
                st = Strokes(n, scale, target, res)
            else:
                tmp = Strokes(n, scale, target, res)
                opt = torch.optim.Adam(tmp._trainable(), lr=0.02)
                for _ in range(max(30, steps // 3)):
                    canvas = tmp.forward(canvas0.detach().clone())
                    loss = F.mse_loss(canvas, target)
                    opt.zero_grad(); loss.backward(); opt.step()
                ratios, L = tmp.descriptors()
                fixed, idx, sbin = snap_shape_scale(ratios, L)
                st = Strokes(n, scale, target, res, shape_fixed=fixed)
                with torch.no_grad():
                    st.o.copy_(tmp.o); st.theta.copy_(tmp.theta)
                    st.col_raw.copy_(tmp.col_raw); st.a_raw.copy_(tmp.a_raw)
                for t in st._trainable():
                    t.requires_grad_(True)
                st._assigned = idx
                st._sbin = sbin
            opt = torch.optim.Adam(st._trainable(), lr=0.02)
            frozen = canvas0.detach()
            for _ in range(steps):
                canvas = st.forward(frozen.clone())
                loss = F.mse_loss(canvas, target)
                loss = loss + 0.5 * F.mse_loss(blur(canvas, 9), blur(target, 9))
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                canvas0 = st.forward(canvas0).clamp(0, 1)
            all_strokes.append(st)
        return canvas0.clamp(0, 1), all_strokes

    def token_render(strokes_list, res):
        """Full 6-slot round-trip: shape+scale already frozen to codes; snap
        x,y to GRID, theta to K_THETA bins, RGBA to palette. Re-render."""
        xs, ys = grids(res)
        canvas = torch.ones(3, res, res, device=dev)
        with torch.no_grad():
            for st in strokes_list:
                L, w, bx, by = st.shape()
                o = st.o.clamp(0, 1)
                oq = (o * (GRID - 1)).round() / (GRID - 1)
                th = torch.remainder(st.theta, 2 * np.pi)
                thq = (th / (2 * np.pi) * K_THETA).round() % K_THETA
                thq = thq * (2 * np.pi / K_THETA)
                rgba = st.colors_rgba()
                pidx = torch.cdist(rgba, palette).argmin(1)
                rgbaq = palette[pidx]
                p0, pm, p1 = world_points(oq, thq, L, bx, by)
                canvas = render(canvas, xs, ys, p0, pm, p1, w,
                                rgbaq[:, :3], rgbaq[:, 3])
        return canvas.clamp(0, 1)

    # ---------------- assemble corpus / held-out targets ----------------
    paint_plan = [(120, 0.5), (180, 0.27), (240, 0.12)]
    line_plan = [(80, 0.45), (160, 0.18), (280, 0.07)]
    ho_paint_plan = [(140, 0.5), (200, 0.27), (300, 0.12), (340, 0.06)]
    ho_line_plan = [(90, 0.45), (180, 0.18), (320, 0.07)]

    corpus = []
    for fname in CORPUS_PAINTINGS:
        try:
            corpus.append((fname[:24], "paint", fetch(fname, corpus_res)))
        except Exception as e:
            print(f"corpus fetch failed {fname}: {e}")
    for (kind, sd_, sz) in CORPUS_TASKS:
        tgt = maze_target(sd_, sz, corpus_res) if kind == "maze" \
            else fold_target(sd_, corpus_res)
        corpus.append((f"{kind}{sd_}", "line", tgt))
    if len(corpus) < 4:
        raise RuntimeError("need >=4 corpus images")

    heldout = []
    for fname in HELDOUT_PAINTINGS:
        try:
            heldout.append((fname[:24], "paint", fetch(fname, heldout_res)))
        except Exception as e:
            print(f"heldout fetch failed {fname}: {e}")
    for (kind, sd_, sz) in HELDOUT_TASKS:
        tgt = maze_target(sd_, sz, heldout_res) if kind == "maze" \
            else fold_target(sd_, heldout_res)
        heldout.append((f"{kind}{sd_}", "line", tgt))

    # ---------------- Stage A: paint corpus, build codebooks ----------------
    all_ratios, all_L, all_rgba, per_img = [], [], [], []
    for i, (name, dom, tgt) in enumerate(corpus):
        plan = paint_plan if dom == "paint" else line_plan
        rec, strokes = paint(tgt, plan, corpus_steps, corpus_res)
        mse = F.mse_loss(rec, tgt).item()
        for st in strokes:
            r, L = st.descriptors()
            all_ratios.append(r); all_L.append(L)
            all_rgba.append(st.colors_rgba())
            per_img.append(torch.full((r.shape[0],), i, device=dev))
        print(f"corpus [{dom}] {name}: MSE={mse:.4f}")
    ratios = torch.cat(all_ratios, 0)
    Ls = torch.cat(all_L, 0)
    rgba = torch.cat(all_rgba, 0)
    img_of = torch.cat(per_img, 0)
    M = ratios.shape[0]
    print(f"corpus strokes M={M} across {len(corpus)} images")

    def kmeans(x, K, iters=60):
        mu, sd = x.mean(0), x.std(0) + 1e-6
        z = (x - mu) / sd
        c = z[torch.randperm(z.shape[0], generator=g, device=dev)[:K]].clone()
        for _ in range(iters):
            a = torch.cdist(z, c).argmin(1)
            for k in range(K):
                m = a == k
                if m.any():
                    c[k] = z[m].mean(0)
        return c, mu, sd, a

    # brush shapes: k-means over scale-invariant ratios
    cb, cb_mu, cb_sd, assign = kmeans(ratios, K_BRUSHES)
    # scale bins: log-spaced between corpus percentiles
    lo = float(torch.quantile(Ls, 0.02)); hi = float(torch.quantile(Ls, 0.98))
    scale_bins = torch.tensor(
        np.geomspace(max(lo, 1e-3), hi, K_SCALE), device=dev,
        dtype=torch.float32)
    # color palette: k-means over RGBA (un-standardized space for snapping)
    pal_c, pal_mu, pal_sd, pal_assign = kmeans(rgba, K_PALETTE)
    palette = (pal_c * pal_sd + pal_mu).clamp(0, 1)

    from collections import Counter
    cnt = Counter(assign.tolist())
    counts = torch.tensor([cnt.get(k, 0) for k in range(K_BRUSHES)],
                          dtype=torch.float)
    probs = counts / counts.sum()
    ent = float(-(probs[probs > 0] * probs[probs > 0].log2()).sum())
    singles = int((counts == 1).sum())
    reuse = [len(set(img_of[assign == k].tolist())) for k in range(K_BRUSHES)]
    cross = sum(1 for r in reuse if r >= 2)
    # domain reuse: brushes used by BOTH paintings and line drawings
    n_paint = sum(1 for (_, d, _) in corpus if d == "paint")
    dom_of = (img_of < n_paint)  # True=painting
    both_dom = sum(1 for k in range(K_BRUSHES)
                   if (assign == k).any()
                   and dom_of[assign == k].any()
                   and (~dom_of[assign == k]).any())
    pal_counts = Counter(pal_assign.tolist())
    pal_used = sum(1 for k in range(K_PALETTE) if pal_counts.get(k, 0) > 0)
    print(f"VOCAB2 K={K_BRUSHES}: entropy {ent:.2f}/{np.log2(K_BRUSHES):.2f}, "
          f"strokes/code median {int(counts.median())}, singletons {singles}, "
          f"cross-image {cross}/{K_BRUSHES}, cross-DOMAIN {both_dom}/{K_BRUSHES}, "
          f"palette used {pal_used}/{K_PALETTE}")

    # ---------------- Stage B: held-out, vocab-constrained + tokenized ------
    results = []
    for (name, dom, tgt) in heldout:
        plan = ho_paint_plan if dom == "paint" else ho_line_plan
        rec_free, _ = paint(tgt, plan, heldout_steps, heldout_res)
        rec_vocab, vstrokes = paint(tgt, plan, heldout_steps, heldout_res,
                                    constrain=True)
        rec_token = token_render(vstrokes, heldout_res)
        used = sorted(set(int(i) for st in vstrokes for i in st._assigned))
        mse_f = F.mse_loss(rec_free, tgt).item()
        mse_v = F.mse_loss(rec_vocab, tgt).item()
        mse_t = F.mse_loss(rec_token, tgt).item()
        ratio = mse_v / max(mse_f, 1e-8)
        gate = "PASS" if ratio <= 1.5 else "FAIL"
        print(f"HELDOUT [{dom}] {name}: free {mse_f:.4f} | vocab {mse_v:.4f} "
              f"(x{ratio:.2f} {gate}) | token {mse_t:.4f} | "
              f"brushes {len(used)}/{K_BRUSHES}")
        results.append(dict(name=name, dom=dom, tgt=tgt.cpu(),
                            free=rec_free.cpu(), vocab=rec_vocab.cpu(),
                            token=rec_token.cpu(), mse_f=mse_f, mse_v=mse_v,
                            mse_t=mse_t, ratio=ratio, used=len(used)))

    # ---------------- figures ----------------
    tile = 64
    cb_real = cb * cb_sd + cb_mu   # (K,3) ratios
    xs_t, ys_t = grids(tile)
    cols_n = 8
    rows_n = (K_BRUSHES + cols_n - 1) // cols_n
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n, rows_n))
    axes = np.array(axes).reshape(-1)
    L_show = 0.55  # scale-invariant: render every brush at the same length
    for k in range(K_BRUSHES):
        wr, bxr, byr = [float(v) for v in cb_real[k]]
        L = L_show
        w = max(wr * L, 0.02)
        o = torch.tensor([[0.5 - L / 2, 0.5]], device=dev)
        th = torch.zeros(1, device=dev)
        p0, pm, p1 = world_points(o, th, torch.tensor([L], device=dev),
                                  torch.tensor([bxr * L], device=dev),
                                  torch.tensor([byr * L], device=dev))
        canvas = torch.ones(3, tile, tile, device=dev)
        canvas = render(canvas, xs_t, ys_t, p0, pm, p1,
                        torch.tensor([w], device=dev),
                        torch.tensor([[0.1, 0.1, 0.1]], device=dev),
                        torch.tensor([1.0], device=dev))
        axes[k].imshow(canvas.permute(1, 2, 0).cpu().numpy())
        axes[k].set_title(str(k), fontsize=6); axes[k].axis("off")
    for j in range(K_BRUSHES, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Scale-invariant brush alphabet (K={K_BRUSHES}, "
                 f"rendered at one scale)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/brush_alphabet_v2.png", dpi=130)

    if results:
        fig2, ax2 = plt.subplots(len(results), 4,
                                 figsize=(12, 3 * len(results)))
        ax2 = np.array(ax2).reshape(len(results), 4)
        for i, r in enumerate(results):
            panels = [(r["tgt"], "target"),
                      (r["free"], f"free {r['mse_f']:.4f}"),
                      (r["vocab"], f"vocab {r['mse_v']:.4f} (x{r['ratio']:.2f})"),
                      (r["token"], f"token {r['mse_t']:.4f}")]
            for j, (im, title) in enumerate(panels):
                ax2[i, j].imshow(im.permute(1, 2, 0).numpy())
                ax2[i, j].set_title(f"[{r['dom']}] {title}" if j == 0 else title,
                                    fontsize=8)
                ax2[i, j].axis("off")
        fig2.tight_layout()
        fig2.savefig(f"{OUT}/heldout_vocab_v2.png", dpi=120)

    with open(f"{OUT}/vocab2.json", "w") as f:
        json.dump({
            "k_brushes": K_BRUSHES, "k_scale": K_SCALE,
            "k_palette": K_PALETTE, "k_theta": K_THETA, "grid": GRID,
            "descriptor_order": ["width/length", "bend_x/length",
                                 "bend_y/length"],
            "codebook_standardized": cb.cpu().tolist(),
            "codebook_mu": cb_mu.cpu().tolist(),
            "codebook_sd": cb_sd.cpu().tolist(),
            "scale_bins": scale_bins.cpu().tolist(),
            "palette_rgba": palette.cpu().tolist(),
            "corpus": [f"{n} [{d}]" for (n, d, _) in corpus],
            "corpus_strokes": M,
            "usage_entropy_bits": ent,
            "max_entropy_bits": float(np.log2(K_BRUSHES)),
            "singletons": singles,
            "codes_reused_across_images": cross,
            "codes_reused_across_domains": both_dom,
            "palette_codes_used": pal_used,
            "heldout": [{"name": r["name"], "domain": r["dom"],
                         "free_mse": r["mse_f"], "vocab_mse": r["mse_v"],
                         "token_mse": r["mse_t"], "vocab_over_free": r["ratio"],
                         "gate_1p5x": r["ratio"] <= 1.5,
                         "brushes_used": r["used"]} for r in results],
        }, f)

    volume.commit()
    print("saved brush_alphabet_v2.png heldout_vocab_v2.png vocab2.json")
    return {"K": K_BRUSHES, "corpus_strokes": M, "entropy": round(ent, 2),
            "singletons": singles, "cross_image": cross,
            "cross_domain": both_dom, "palette_used": pal_used,
            "heldout": [(r["name"], r["dom"], round(r["mse_v"], 4),
                         f"x{r['ratio']:.2f}",
                         "PASS" if r["ratio"] <= 1.5 else "FAIL")
                        for r in results]}


def _isp(y):
    import math
    return math.log(math.expm1(max(y, 1e-3)))


@app.local_entrypoint()
def main():
    print("RESULT:", run.remote())
    print("pull: modal volume get brush-vocab /out ./vocab_out")
