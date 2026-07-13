# Stroke-CoT × difficulty × scale: when does drawing beat writing?

**Research question.** On verifiable labyrinth tasks, does a stroke-token
scratchpad in the CoT beat budget-matched text CoT — and how does the
advantage move with (a) task difficulty and (b) model size?

**Headline deliverable: one figure.** The advantage
`A(d, s) = solve_rate(stroke) − solve_rate(text)` as a function of difficulty
`d`, one curve per model size `s`, with the shuffled-stroke ablation flat at
~0. Every design choice below exists to make that figure trustworthy. A
single-difficulty, single-size win is a bar chart; the interaction surface is
a finding.

This supersedes Phases 3–4 of [training-program.md](training-program.md) as
the focus block. Phases 0–2 (the painter) are **not prerequisites**: the
random-codebook control (`vocab2.py --n-rand 3`, 2026-07-12) showed the
learned shape codebook ties random valid shapes on held-out reconstruction
(learned/random advantage ×0.94–×1.04, mean ×0.99). Shape learning is not
load-bearing; a fixed lattice alphabet suffices for labyrinths. The painter
stays a demo track.

## 1. Pre-registered hypotheses

- **H1 (mechanism: persistent state).** Text CoT must re-serialize or
  re-derive spatial state as it reasons, so its effective cost grows
  super-linearly with solution length; a stroke scratchpad appends
  (delta-editable), growing linearly. Prediction: `A(d, s)` **increases in
  d**. If `A` is positive but flat in `d`, drawing helps for some other
  reason and this mechanism story is wrong — report that.
- **H2a (scaffold).** Externalized state is a crutch small models need and
  large models internalize. Prediction: `A` **decreases in s**.
- **H2b (capability unlock).** Small models cannot *produce* useful drawings
  (think-visually: "a monitor cannot rescue what the model cannot produce").
  Prediction: `A` **increases in s**, at least until saturation.
- H2a and H2b predict opposite slopes, so the size axis is informative
  whatever it shows. A bump (nothing at 0.5B, peak at mid-scale, shrinking
  later) = unlock-then-outgrow, the most interesting outcome.
- **H0.** `A ≈` shuffled ablation everywhere: the stroke tokens are filler
  and any gain is compute. Publishable negative, direct sequel to
  think-visually.

## 2. Task & difficulty axis

Labyrinths from the existing generator (DFS backtracker, as in `vocab2.py`),
n×n for n = 4…16. **Primary difficulty knob: solution length** `L_sol`
(secondary: dead-end count). Bins, e.g. `L_sol ∈ {6–10, 11–20, 21–35, 36–60,
61–100}`.

- Input serialization **identical across conditions** (compact wall list +
  start/goal). Final answer format identical (move sequence in text). Only
  the scratchpad medium differs.
- Train on mixed difficulty up to bin 3; **evaluate all bins** — report
  interpolation and extrapolation columns separately. H1 bites hardest in
  extrapolation.
- Maze path-drawing is near-tautological (the answer *is* a drawing). That is
  deliberate — it maximizes the chance of detecting the mechanism — but it
  caps generality. Keep one task with an indirect drawing→answer mapping
  (fold, or maze-with-counting) as a secondary result. Report both; no
  cherry-picking.

## 3. Conditions

All: same base model, same SFT recipe, same per-item token budget.

| condition | scratchpad content |
|---|---|
| text-CoT | strongest text format we can build (explicit frontier/visited coordinate lists, BFS-style) |
| stroke-CoT | interleaved text + `<draw>` stroke tokens `</draw>`, **no render feedback** |
| shuffled-stroke | same token count, stroke geometry randomized (filler control) |
| stroke+feedback *(phase 2, only if stroke-CoT shows signal)* | canvas rendered and re-encoded after each draw span |

**Say it plainly:** without render feedback the model never *sees* a canvas —
it attends to its own emitted tokens. v1 therefore tests **structured
geometric notation vs free-form text**, not vision. The feedback condition is
where "seeing" enters; difficulty scaling may separate the two (notation
suffices on small mazes, feedback matters on large ones).

## 4. Stroke token set (fixed, not learned)

Lattice segments + cell marks; ~75 new embeddings, verifier-readable
directly:

| tokens | count | role |
|---|---|---|
| `<draw>`, `</draw>` | 2 | span markers |
| `X0…X31`, `Y0…Y31` | 64 | lattice coordinates (covers extrapolation to 32×32) |
| `D_N D_E D_S D_W` | 4 | segment direction; a segment = `[X, Y, D]` |
| `MARK_VISITED`, `MARK_DEAD`, `MARK_START`, `MARK_GOAL` | 4 | cell annotations = `[MARK_*, X, Y]` |

**Parameter parity:** the text-condition models get the same ~75 rows added
to the embedding table, unused. No condition differs in parameter count.

## 5. Data & training

- **Traces are programmatic** (we own the generator; no optimizer in the
  loop). Stroke traces: draw walls context → mark start/goal → extend path,
  mark dead ends on backtrack → read off moves. Text traces: BFS/DFS with
  explicit frontier and visited lists, generated from the same ground truth
  **with matched care** — an under-engineered text baseline is the
  reviewer-bait failure mode. Pilot: hand-inspect 50 traces per condition.
- ~50k traces per condition, difficulty-mixed. Models: **Qwen2.5 base 0.5B /
  1.5B / 7B** (same family/tokenizer; base + identical SFT to control
  post-training differences across sizes).
- **Stage 1: SFT the full grid** (cheap). **Stage 2: GRPO** with verifier
  reward, only in cells that pass the production gate below. Runs =
  conditions × sizes × seeds (difficulty is an eval axis, not a training
  axis — this is what keeps the matrix affordable).

**Budget-matching rule (per difficulty):** `B(d) = 1.25 × p95` stroke-trace
token length at difficulty `d`; every condition gets `B(d)` (text padded /
longer, never shorter). Matching must scale with `d` or the compute confound
reappears exactly where the effect is claimed to live.

## 6. Metrics

- **Primary:** exact-solve rate (verifier: legal moves, reaches goal) per
  (condition, size, difficulty bin); ≥3 seeds; bootstrap CIs over eval items.
  Trend test: logistic regression `solve ~ condition × L_sol` per size;
  `condition × size` interaction at fixed difficulty.
- **Mechanism probe:** consistency between the *drawn* path and the
  *answered* move sequence. High solve + low consistency = drawings are
  decorative, not used — kills the interpretation even if the win is real.
- **Health:** well-formed span rate, verifier-valid drawing rate, budget
  compliance, answer-extraction rate.

## 7. Confounds designed against, up front

1. **Compute/filler** (pause-token effect) → budget matched upward per
   difficulty + shuffled-stroke ablation.
2. **Text-baseline strawman** → strongest text format; pilot ≥2 text
   serializations per size, keep the better one.
3. **Trace-quality asymmetry** → both trace generators built from the same
   ground truth with the same effort; hand-inspection pilot.
4. **Parameter count** → unused embedding rows added to text models.
5. **Post-training differences across sizes** → base checkpoints + one shared
   SFT recipe.
6. **Difficulty leakage** → eval mazes seed-disjoint; extrapolation bins
   never trained on.
7. **RL variance** → ≥3 seeds per cell, all runs reported; SFT-only grid
   published alongside RL results.

## 8. Gates & kill criteria

- **Production gate (per size, after SFT):** >80% verifier-valid drawings on
  held-out prompts. Fails at every size ≤7B → the production bottleneck
  persists at small scale; one 14B attempt, then write it up as the
  capability-unlock boundary.
- **Kill:** `A ≤` shuffled everywhere after RL → negative result, cite
  think-visually, publish.
- **Kill:** drawn/answered consistency <50% in winning cells → the win is
  not *from drawing*; investigate before claiming anything.
- Every branch of the outcome map is a paper: mechanism-supported
  (A > shuffled, rising in d), drawing-helps-otherwise (A > shuffled, flat),
  filler-effect (A ≈ shuffled > 0), or clean negative (A ≈ 0 with production
  gate passed).

## 9. Prior art to stay honest about

MVoT is the closest — interleaved visual tokens in CoT, evaluated on
maze-like tasks — but with opaque patch tokens, one model size, no token
budget matching. Visual Sketchpad and Whiteboard-of-Thought are tool-based on
frontier models; Anole/Chameleon interleave opaque codes; pause/filler-token
work explains compute-only gains. **The open slice:** legible tiny-vocab
delta-editable stroke tokens, budget-matched with a filler control, mapped
over the size × difficulty surface. The interaction map is the contribution;
no single cell of it is novel on its own.

## 10. Budget & timeline

- SFT grid: 3 sizes × 3 conditions × 3 seeds = 27 runs (LoRA on 7B, full on
  smaller), short sequences, ~50k examples → **~$100–300**.
- Eval sweeps: batch inference, negligible.
- GRPO: gated cells only; the 9 7B runs dominate → **~$500–1500**.
- Ceiling **≈ $2k**, staged so an early kill is a cheap kill. This exceeds
  the repo's original <$200 framing — that's what the gates are for.
- ~4–8 weeks part-time: wk 1 tokens + generators + traces; wk 2 SFT grid +
  gates; wks 3–5 RL; wk 6 analysis and writeup.
