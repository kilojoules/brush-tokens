# Brush-GPT: end-to-end training program

**Vision, split honestly in two:**

1. *"AR alternative to diffusion"* — **not the headline.** Autoregressive image
   generation already works at industrial scale (VAR — NeurIPS 2024 best paper,
   next-scale AR beating diffusion FID; LlamaGen; Infinity; Janus; GPT-4o /
   Gemini native image gen are AR). A 10–30M-param stroke GPT will not compete
   on image quality. Stroke-AR specifically is also prior art (SketchRNN,
   Sketchformer, StrokeNUWA, Paint Transformer). We keep the generative painter
   as a **demo asset** (the drawing animation sells), not as a claim.

2. *"Scratchpad for LLM thinking"* — **the actual bet.** Nearby prior art:
   MVoT (interleaved VQ image patches), Visual Sketchpad (external tool calls,
   no learned tokens), Whiteboard-of-Thought (code→matplotlib), Chameleon/Anole
   (interleaved generation, opaque patch tokens), o3-style "thinking with
   images" (crop/zoom tools, not drawing). None of them have **legible,
   tiny-vocab, delta-editable stroke tokens with a programmatic verifier**.
   That is the open slice this program targets.

## Thesis (write this on the wall)

**Strokes beat image patches as a reasoning substrate because the canvas is
persistent state.** Patch-token visual CoT must re-emit an entire image to
change one line; a stroke model appends one token — the same incremental
structure that makes text CoT work. Plus:

- **~470-token vocabulary** (vs 8k–64k patch codes) — graftable onto a 1.5B
  LLM's embedding table without drowning it.
- **Human-readable tokens** — every step of the visual reasoning is
  inspectable; interpretability comes for free.
- **Verifier-friendly** — fold/maze verifiers read stroke geometry directly.
  No vision model in the reward loop, no reward hacking through pixels.

**One claim to win:** on verifiable spatial tasks, a small open LLM with a
brush-token scratchpad + verifier RL beats the same model with text-only CoT
at a **matched token budget**. A clean negative result is also publishable —
it is the direct sequel to think-visually ("a monitor cannot rescue what the
model cannot produce").

---

## Phase 0 — Vocabulary v2 (~2 days)

Prerequisites from `docs/ar-brush-gpt.md` §1, plus one strategic change:

- **Scale-invariant brushes**: normalize shape descriptors to unit length,
  store `(width/length, bend_x/length, bend_y/length)`; add a discrete
  `scale` slot (16 log-spaced bins). One brush, many sizes.
- **Shared color palette**: k-means over corpus colors → 256 codes. Brush and
  color stay independent tokens.
- **Line-drawing corpus alongside paintings.** The downstream tasks are mazes
  and folds — line drawings, not Vermeer. Either retrain the codebook on a
  mixed corpus (paintings + rendered task states + QuickDraw rasters) or
  keep two sub-alphabets (32 painterly + 32 line brushes). Verify held-out
  reconstruction still holds on *both* domains.

**Gate:** held-out MSE within ~1.5× of the unconstrained fit on both a
painting and a maze render.

## Phase 1 — Labeler (~3 days, ~$10–30 Modal)

The `vocab.py` optimizer (frozen vocabulary) becomes the labeler, fanned out
with Modal `.map()`. Two datasets:

- **(a) Painter pretrain**: ~500 paintings/photos (WikiArt subset + crops/
  augmentations) → `(image, canonical token sequence)` pairs. Canonical order:
  coarse→fine layers, raster within layer; re-render in that order to define
  the target.
- **(b) Task states**: ~5k synthetic maze boards / fold diagrams from the
  think-visually generators. **Cheap shortcut: we own the generator** — emit
  ground-truth stroke sequences directly (walls, paths, fold lines are already
  line segments). Most of (b) may need no optimizer at all.

**Gate:** re-render labeled sequences; MSE ≈ the unordered fit. Spot-check 20
by eye.

## Phase 2 — Brush-GPT pretrain (~3 days, ~$20)

Per `docs/ar-brush-gpt.md`: decoder-only, 10–30M params, flattened
`[brush,scale,x,y,theta,color]` slots, typed slot embeddings, slot-masked
decoding, image-conditioned via 32×32 CNN thumbnail prefix. Teacher-forced CE.

Deliverables: amortized painter (one forward pass vs minutes of optimization),
honest stroke-by-stroke drawing animation, held-out reconstruction vs the
optimization upper bound. **This is demo + sanity check, not the result.**

**Gate:** held-out reconstruction qualitatively recognizable; valid-token rate
≈100%; EOS calibrated to image complexity.

## Phase 3 — Graft onto Qwen2.5-1.5B (~1 week)

- Extend the embedding table by ~470 rows (brush vocab + `<draw>`/`</draw>`
  span markers). Initialize new rows near the mean of existing embeddings.
- SFT on **synthetic interleaved traces**: text plan → `<draw>` stroke tokens
  `</draw>` → text answer. Example (maze): restate goal in text, draw the
  candidate path as strokes, read off the move sequence.
- **v1: no render feedback** — the model emits and attends to its own stroke
  tokens only. **v2 (only if v1 shows signal):** render the canvas, re-encode,
  feed back as conditioning.
- Trace quality is the whole game. think-visually failed because the model
  could not *produce* useful drawings; SFT data must teach production, not
  reading. Generate traces programmatically from ground-truth solutions so
  the drawn content is always correct and always used by the subsequent text.

**Gate:** SFT model emits well-formed draw spans on held-out prompts;
rendered drawings match task semantics (verifier-checkable) >80% of the time.

## Phase 4 — The experiment (~1–2 weeks)

Verifier-reward RL (GRPO) on fold/maze, reusing think-visually verifiers.

**Conditions (all at matched total token budget):**

| condition | CoT content |
|---|---|
| text-only | plain text CoT, budget-matched (padded/longer, not shorter) |
| brush-CoT | interleaved text + stroke tokens |
| shuffled-stroke ablation | same budget, stroke tokens with randomized geometry |

**Confounds designed against, up front:**

1. **Compute confound** — extra tokens = extra thinking regardless of content
   (pause-token effect). Hence budget-matched text baseline, never a shorter
   one.
2. **Content vs filler** — if brush-CoT's gain survives the shuffled-stroke
   ablation, the gain was compute, not drawing. The claim dies; say so.
3. **Task selection** — start where drawing *provably* helps: maze
   path-drawing is near-tautological (the answer *is* a drawing). Then fold,
   where the mapping is less direct. Report both; do not cherry-pick.

**Victory condition:** brush-CoT > text-only AND brush-CoT > shuffled, on
held-out tasks, with error bars (≥3 seeds). Anything else is a (still
publishable) negative result.

## Budget & timeline

~4–6 weeks total, <$200 Modal compute. Phases 0–2 are parallelizable with
other work; phases 3–4 are the focus block.

## Kill criteria

- Phase 0 gate fails on line-drawing domain → the painterly vocabulary is the
  wrong substrate; fall back to `modal_vq_stroke.py`-style polyline tokens for
  the reasoning phases and keep the painter as a standalone demo.
- Phase 3 gate fails (<50% verifier-valid drawings after SFT iteration) →
  1.5B can't produce; try 7B once before concluding.
- Phase 4 shuffled ablation ties brush-CoT → write the negative result,
  cite think-visually, move on.
