# Brush-GPT: an autoregressive model over brush-token sequences

**Goal.** Turn the *fitting* pipeline (`vocab.py`, which reconstructs one image by
gradient optimization) into a *generative* one: a model that emits a painting as
a sequence of brush tokens, stroke by stroke. Image-GPT, but over strokes from
the shared 32-brush vocabulary instead of pixels.

Two payoffs beyond "it generates":
- **Amortized painting** — one forward pass paints any image as brush tokens,
  vs. minutes of per-image optimization (100–1000× faster).
- **A real drawing animation** — the earlier `drawing.gif` replayed *optimized*
  strokes in an arbitrary order (the review correctly called it theater). A model
  that autoregressively emits strokes genuinely decides them one at a time; the
  animation becomes an honest visualization of the model's process.

This is also the bridge to the reasoning goal: a model that can *emit* brush
tokens is a model that can *draw its reasoning* (condition on a task, score with
a verifier).

---

## 1. What is a token

A stroke instance = a shape from the vocabulary + how it's placed. Prerequisite
refinement to `vocab.py`: make brush shapes **scale-invariant** (normalize each
descriptor to unit length; store `(width/length, bend_x/length, bend_y/length)`)
and add a separate **scale** attribute, so one brush is reusable at many sizes —
otherwise 32 brushes must also span every size and the vocabulary bloats.

Per stroke, six attributes, each a discrete token with its own ID range:

| slot | attribute | source | vocab size |
|------|-----------|--------|-----------:|
| 0 | `brush` | shared shape codebook (`vocab.py`) | 32 |
| 1 | `scale` | log-spaced size bins | 16 |
| 2 | `x` | position, grid bin | 64 |
| 3 | `y` | position, grid bin | 64 |
| 4 | `theta` | orientation, angle bin | 32 |
| 5 | `color` | shared palette codebook (k-means over corpus colors) | 256 |

Plus specials: `BOS`, `EOS`, `PAD`. Total token vocabulary ≈ **470**. Color
leaves the brush codebook (a brush is a mark, not a paint) and becomes its own
shared palette — so "brush" and "color" are independent tokens, as in real
painting.

## 2. Sequence format

Flatten a painting into `BOS, [brush,scale,x,y,theta,color]×N, EOS`, padded to a
max stroke count. ~400 strokes × 6 = 2400 tokens + specials → fits a 3–4k
context. (Efficiency option: **factored** prediction — one transformer step per
stroke with six output heads — cuts length 6× at the cost of modeling
intra-stroke dependencies with small per-head conditioning. Start flattened.)

**Canonical order (critical).** AR needs a deterministic order; the optimizer
lays strokes down in coarse→fine layers but *random within a layer*. Impose:
(1) layer coarse→fine, (2) within a layer, raster scan by `(y_bin, x_bin)`. Then
**re-render in this order** to define the target image the sequence produces
(reordering within a coarse→fine layering barely moves the image). Variable
stroke counts are handled by `EOS` + padding; the model learns when to stop.

## 3. Model

Decoder-only transformer (GPT), ~10–30M params (12 layers, d=384, 6 heads is
plenty for ~470 vocab). Two cheap structural aids:
- **Typed slot embedding.** Add an embedding for *which of the 6 slots* a
  position is (brush/scale/x/y/theta/color). The token-type pattern is periodic;
  telling the model the slot prevents type errors and speeds convergence.
- **Constrained decoding.** At slot *s* only that slot's ID range is legal — mask
  logits accordingly. Guarantees every sampled sequence decodes to valid strokes.

**Conditioning (image → strokes, the primary mode).** Encode a 32×32 target
thumbnail with a small CNN into a short prefix of conditioning vectors; the
decoder cross-attends (or consumes them as a prefix). This makes Brush-GPT an
*amortized painter*: feed an image, get its brush-token sequence in one pass.
Unconditional and class-conditional (prepend a style/corpus-id token) fall out as
special cases and are useful stepping stones.

## 4. Training data

The `vocab.py` optimizer is the **labeler** (bootstrap, à la SPIRAL / Learning to
Paint): run it, with the *frozen shared vocabulary*, over a corpus of images to
produce `(image, canonical token sequence)` pairs.

- Source: a few hundred → few thousand images (WikiArt subset, painting
  datasets, or crops/augmentations to multiply the corpus).
- **Cost is the real constraint**: each label is minutes of optimization.
  Mitigate with Modal `.map()` — fan the labeler across many containers so
  wall-clock is short even if GPU-hours aren't. Start at ~500 images (≈$5–10),
  scale once the pipeline works.

Loss: teacher-forced cross-entropy over the flattened sequence (slot-masked).
**Optional v2**: a rendering loss — decode predicted tokens, render, compare to
target — through the differentiable renderer. The discrete sampling blocks
gradients, so use straight-through / Gumbel or REINFORCE. Base model is pure CE;
add this only if CE reconstructions look structurally off.

## 5. Inference & evaluation

- **Sampling**: `BOS` → sample slot by slot (temperature / top-k / nucleus,
  slot-masked) → decode each stroke tuple → render incrementally → stop at `EOS`.
  Incremental render = the honest drawing animation.
- **Metrics**:
  - reconstruction (image-conditioned): MSE / LPIPS vs target on **held-out**
    images; compare against per-image optimization (upper bound) and speed.
  - generation (unconditional/class): FID of rendered samples vs real; qualitative.
  - health: valid token-type rate (≈100% with masking), brush-vocab coverage,
    `EOS` calibration (does stroke count match image complexity).

## 6. Build phases

1. **Vocab v2** — scale-invariant brushes + scale bins + shared color palette;
   re-verify held-out reconstruction still holds. *(small change to `vocab.py`)*
2. **Labeler** — `vocab.py` → save canonical token sequences; `.map()` over ~500
   images. Sanity: re-render sequences, confirm MSE ≈ the un-ordered fit.
3. **Brush-GPT v1** — flattened GPT, typed slots, slot-masked decoding,
   image-conditioned. Train on the labeled set.
4. **Eval + animation** — held-out reconstruction vs optimization; real
   stroke-by-stroke generation GIF.
5. **(later) reasoning** — swap image-conditioning for task/text conditioning;
   score drawn solutions with the think-visually fold/maze verifiers.

## 7. Prior art to stay honest about

Autoregressive stroke/vector generation exists — SketchRNN, DeepSVG,
Sketchformer, StrokeNUWA, Paint Transformer. Brush-GPT over *painterly* strokes
from a shared shape vocabulary is a reasonable engineering contribution but **not
novel on its own**. The defensible novelty remains the reasoning claim (phase 5):
does emitting brush tokens help a model solve verifiable spatial tasks that it
fails in text. Brush-GPT is the vehicle, not the paper.
