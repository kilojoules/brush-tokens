"""
Stroke VQ-VAE tokenizer — build step #1 for the CoT-brush-token idea.

Trains a VQ-VAE over QuickDraw pen-stroke sequences (SketchRNN stroke-3 format).
Output = a discrete stroke codebook: each sketch -> sequence of code indices.
That codebook is the visual vocabulary we later graft onto an LLM.

Clean licensing: QuickDraw data (Google, CC-BY 4.0), scratch implementation.
No StrokeNUWA code (unlicensed).

Run:
    modal run modal_vq_stroke.py                 # download + train few categories
    modal run modal_vq_stroke.py --categories cat,face,apple --epochs 30

Artifacts land in the `stroke-vq` volume: checkpoints + recon PNGs.
Pull them with:
    modal volume get stroke-vq /out ./out
"""

import modal

app = modal.App("stroke-vq-tokenizer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "requests", "matplotlib")
)

volume = modal.Volume.from_name("stroke-vq", create_if_missing=True)
DATA_DIR = "/data"

# QuickDraw sketchrnn npz set. Keep small for the first smoke run.
DEFAULT_CATEGORIES = ["cat", "face", "apple", "bicycle", "fish"]
MAX_LEN = 150  # truncate/pad stroke sequences to this many points


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@app.function(image=image, volumes={DATA_DIR: volume}, timeout=1800)
def download_data(categories: list[str]):
    import os
    import requests

    os.makedirs(f"{DATA_DIR}/raw", exist_ok=True)
    for c in categories:
        path = f"{DATA_DIR}/raw/{c}.npz"
        if os.path.exists(path):
            print(f"have {c}")
            continue
        url = f"https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{c}.npz"
        print(f"download {c} <- {url}")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    volume.commit()
    print("download done")


# ----------------------------------------------------------------------------
# Training (runs on GPU)
# ----------------------------------------------------------------------------
@app.function(image=image, volumes={DATA_DIR: volume}, gpu="T4", timeout=3600)
def train(categories: list[str], epochs: int = 20, batch_size: int = 128,
          codebook_size: int = 512, code_dim: int = 64, d_model: int = 128,
          lr: float = 3e-4, beta: float = 0.25):
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} categories={categories}")

    # ---- load + convert stroke-3 -> stroke-5, normalize deltas ----
    def stroke3_to_5(s):
        # s: (N,3) int [dx, dy, pen_lift]; -> (N+1,5) [dx,dy,p_down,p_up,p_end]
        n = len(s)
        out = np.zeros((n + 1, 5), dtype=np.float32)
        out[:n, 0:2] = s[:, 0:2]
        lift = s[:, 2]
        out[:n, 2] = 1.0 - lift  # pen down (drawing)
        out[:n, 3] = lift        # pen up (stroke break)
        out[n, 4] = 1.0          # end of sketch
        return out

    raw = []
    for c in categories:
        d = np.load(f"{DATA_DIR}/raw/{c}.npz", encoding="latin1", allow_pickle=True)
        for s in d["train"]:
            s = np.asarray(s, dtype=np.float32)
            if 2 < len(s) <= MAX_LEN - 1:
                raw.append(stroke3_to_5(s))
    print(f"loaded {len(raw)} sketches")

    # global std over dx,dy for normalization (SketchRNN convention)
    alld = np.concatenate([s[:, 0:2] for s in raw], axis=0)
    scale = float(alld.std() + 1e-6)
    print(f"delta scale={scale:.3f}")

    class StrokeDS(Dataset):
        def __init__(self, seqs):
            self.seqs = seqs

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, i):
            s = self.seqs[i].copy()
            s[:, 0:2] /= scale
            L = len(s)
            pad = np.zeros((MAX_LEN, 5), dtype=np.float32)
            pad[:L] = s
            mask = np.zeros(MAX_LEN, dtype=np.float32)
            mask[:L] = 1.0
            return torch.from_numpy(pad), torch.from_numpy(mask)

    ds = StrokeDS(raw)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                    num_workers=2)

    # ---- model: GRU encoder -> per-step EMA-VQ -> GRU decoder ----
    # EMA codebook updates + dead-code revival to prevent collapse
    # (v1 with loss-based VQ collapsed to ~20/512 codes).
    class VQEMA(nn.Module):
        def __init__(self, K, D, decay=0.99, eps=1e-5):
            super().__init__()
            emb = torch.randn(K, D)
            self.register_buffer("emb", emb)
            self.register_buffer("emb_avg", emb.clone())
            self.register_buffer("cluster_size", torch.zeros(K))
            self.K, self.D, self.decay, self.eps = K, D, decay, eps

        def forward(self, z):  # z: (B,T,D)
            B, T, D = z.shape
            flat = z.reshape(-1, D)
            d = (flat.pow(2).sum(1, keepdim=True)
                 - 2 * flat @ self.emb.t()
                 + self.emb.pow(2).sum(1))
            idx = d.argmin(1)
            oh = F.one_hot(idx, self.K).type(flat.dtype)
            zq = (oh @ self.emb).view(B, T, D)
            if self.training:
                # detach + no_grad: EMA buffers persist across batches, so any
                # retained graph on them accumulates unboundedly -> OOM.
                with torch.no_grad():
                    flat_d = flat.detach()
                    n = oh.sum(0)
                    self.cluster_size.mul_(self.decay).add_(
                        n, alpha=1 - self.decay)
                    dw = oh.t() @ flat_d
                    self.emb_avg.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                    total = self.cluster_size.sum()
                    cs = ((self.cluster_size + self.eps)
                          / (total + self.K * self.eps) * total)
                    self.emb.copy_(self.emb_avg / cs.unsqueeze(1))
            commit_loss = F.mse_loss(z, zq.detach())
            zq_st = z + (zq - z).detach()  # straight-through
            return zq_st, commit_loss, idx.view(B, T)

        @torch.no_grad()
        def revive(self, z_flat, threshold=1.0):
            # reset dead codes to random live encoder outputs
            dead = self.cluster_size < threshold
            ndead = int(dead.sum())
            if ndead == 0:
                return 0
            pick = torch.randint(0, z_flat.size(0), (ndead,),
                                 device=z_flat.device)
            self.emb[dead] = z_flat[pick]
            self.emb_avg[dead] = z_flat[pick]
            self.cluster_size[dead] = 1.0
            return ndead

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(5, d_model)
            self.enc = nn.GRU(d_model, d_model, batch_first=True,
                              bidirectional=True)
            self.to_code = nn.Linear(2 * d_model, code_dim)
            self.vq = VQEMA(codebook_size, code_dim)
            self.from_code = nn.Linear(code_dim, d_model)
            self.dec = nn.GRU(d_model, d_model, batch_first=True)
            self.out_delta = nn.Linear(d_model, 2)
            self.out_pen = nn.Linear(d_model, 3)

        def encode(self, x):  # pre-VQ latents, for dead-code revival
            h = self.in_proj(x)
            h, _ = self.enc(h)
            return self.to_code(h)

        def forward(self, x):
            z = self.encode(x)
            zq, commit_loss, idx = self.vq(z)
            g = self.from_code(zq)
            g, _ = self.dec(g)
            return self.out_delta(g), self.out_pen(g), commit_loss, idx

    model = Model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(f"{DATA_DIR}/out", exist_ok=True)
    for ep in range(epochs):
        model.train()
        tot = {"delta": 0.0, "pen": 0.0, "vq": 0.0, "n": 0}
        code_hist = torch.zeros(codebook_size)
        last_z = None
        for x, mask in dl:
            x, mask = x.to(dev), mask.to(dev)
            pred_d, pred_p, commit_loss, idx = model(x)
            m = mask.unsqueeze(-1)
            delta_loss = ((pred_d - x[:, :, 0:2]) ** 2 * m).sum() / m.sum()
            pen_tgt = x[:, :, 2:5].argmax(-1)
            pen_loss = (F.cross_entropy(pred_p.reshape(-1, 3),
                                        pen_tgt.reshape(-1), reduction="none")
                        * mask.reshape(-1)).sum() / mask.sum()
            # EMA updates the codebook; only commitment enters the loss
            loss = delta_loss + pen_loss + beta * commit_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot["delta"] += delta_loss.item()
            tot["pen"] += pen_loss.item()
            tot["vq"] += commit_loss.item()
            tot["n"] += 1
            code_hist += torch.bincount(idx.reshape(-1).cpu(),
                                        minlength=codebook_size).float()
        # revive dead codes using a fresh batch's encoder outputs
        with torch.no_grad():
            xb, _ = next(iter(dl))
            zf = model.encode(xb.to(dev)).reshape(-1, code_dim)
        revived = model.vq.revive(zf)
        n = tot["n"]
        used = int((code_hist > 0).sum())
        print(f"ep{ep:02d} delta={tot['delta']/n:.4f} pen={tot['pen']/n:.4f} "
              f"commit={tot['vq']/n:.4f} codes_used={used}/{codebook_size} "
              f"revived={revived}")

    # ---- save checkpoint ----
    ckpt = {
        "model": model.state_dict(),
        "scale": scale,
        "config": {"codebook_size": codebook_size, "code_dim": code_dim,
                   "d_model": d_model, "max_len": MAX_LEN,
                   "categories": categories},
    }
    torch.save(ckpt, f"{DATA_DIR}/out/stroke_vq.pt")

    # ---- reconstruction plots (orig vs recon) ----
    def render(ax, seq5, title):
        # seq5: (T,5) normalized deltas
        xy = np.cumsum(seq5[:, 0:2] * scale, axis=0)
        pen_up = seq5[:, 3] > 0.5
        end = seq5[:, 4] > 0.5
        start = 0
        for i in range(len(xy)):
            if pen_up[i] or end[i]:
                ax.plot(xy[start:i + 1, 0], -xy[start:i + 1, 1],
                        "k-", linewidth=1.5)
                start = i + 1
            if end[i]:
                break
        ax.set_title(title, fontsize=8)
        ax.axis("equal")
        ax.axis("off")

    model.eval()
    with torch.no_grad():
        x, mask = next(iter(dl))
        x = x.to(dev)
        pred_d, pred_p, _, idx = model(x)
        pen_oh = F.one_hot(pred_p.argmax(-1), 3).float()
        recon = torch.cat([pred_d, pen_oh], dim=-1).cpu().numpy()
        orig = x.cpu().numpy()
    ncol = 6
    fig, axes = plt.subplots(2, ncol, figsize=(2 * ncol, 4))
    for j in range(ncol):
        render(axes[0, j], orig[j], "orig")
        render(axes[1, j], recon[j], "recon")
    fig.tight_layout()
    fig.savefig(f"{DATA_DIR}/out/recon.png", dpi=120)
    print("saved ckpt + recon.png")

    volume.commit()
    return {"codes_used": used, "codebook_size": codebook_size,
            "n_sketches": len(raw)}


@app.local_entrypoint()
def main(categories: str = ",".join(DEFAULT_CATEGORIES), epochs: int = 20):
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    download_data.remote(cats)
    result = train.remote(cats, epochs=epochs)
    print("RESULT:", result)
    print("pull artifacts: modal volume get stroke-vq /out ./out")
