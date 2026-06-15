# Lab Rat AI — Python evaluation weaknesses & fixes (Round 1)

Model: Gemma 4 E4B-it Q4_K_M, served via llama-server.exe on `start.py`.
Question bank: `python_lab04_questions.md` (8 questions).
Answers: `python_lab04_answers.md`.

## Score

5/8 clean passes · 1 minor nit · 2 failures · 1 critical duplication bug.

## Per-question findings

| Q | Verdict | Finding | Fixed in commit |
|---|---|---|---|
| Q1 RGB/HSV channels | PASS | Clean. `img_as_float`, six titled subplots, all greyscale. | — |
| Q2 Colour harmonies | PASS | Correct hue wrap with `np.mod`, swatch rectangles. | — |
| Q3 Manual violin | PASS | `gaussian_kde` + `fill_betweenx` arg order correct. | — |
| Q4 Divergent heatmap | **CRITICAL: duplicate answer** | Model produced TWO complete Assumptions/code/Notes triples in one response. Code itself was correct both times. | `Add single-answer rule to Python prompt` + `Mirror single-answer rule into R prompt` |
| Q5 Four outlier techniques | PASS with nit | Alpha panel was `imshow(M, alpha=0.85)` over a blank axes — looked like a faded heatmap, didn't demonstrate what alpha is for. | `Strengthen alpha-overlay recipe so demo actually shows what alpha does` |
| Q6 Quantile-anchored cmap | **FAIL: context loss** | Model invented a fresh `np.random.rand(10,10)*10-5` matrix instead of "the same M from Q5" — each question is a fresh chat. cmap logic was correct. Eval-design bug, not a prompt bug. | `Make Q6 self-contained: inline the matrix definition instead of referencing Q5` |
| Q7 Ridgeline 2×4 grid | **FAIL: semantics** | Refused joypy correctly. But put one symmetric KDE per subplot — that's 8 independent density plots, not a ridgeline. The question was internally inconsistent ("ridgeline in 2×4 grid"). | `Rewrite Q7 to ask for the canonical stacked-KDE ridgeline (single axes)` |
| Q8 GM + CHM | PASS | Both formulas straight from the prompt, four-decimal output. | — |

## Root causes & decisions

1. **Duplicate-answer drift (Q4).** Highest-priority fix. The prompt had no explicit STOP rule. Small CPU-quantised models without strong sampling discipline can drift into a second draft when they treat "Notes:" as a soft delimiter. Fix is two reinforcing rules — one in RESPONSE FORMAT, one in ANTI-PATTERNS — so the model sees the same instruction twice and cannot miss it. Applied to both `ui/index.html` and `ui/r_lang.html` because R mode runs the same model and has the same risk.

2. **Shallow alpha demonstration (Q5).** The prompt taught the technique but not the demonstration. The recipe now shows two valid patterns (base layer or visible grid) and explicitly says "pick ONE" so the model doesn't produce a Frankenstein hybrid.

3. **Context loss (Q6).** Not a model failure. The chat UI does not thread Q6 to Q5, so "the same matrix M" had no referent. Fix is in the question, not the prompt.

4. **Ridgeline ambiguity (Q7).** Question fault — a real ridgeline lives on one set of axes. The original phrasing collided with the stacked-KDE fallback rule in the prompt. Question is now canonical and the "no joypy" trick aspect is preserved.

## What was NOT changed and why

- Q1, Q2, Q3, Q8 already passed cleanly — no prompt edits.
- The base Lab 04 recipes (violin, heatmap, divergent cmap, multi-checkpoint cmap) are calibrated correctly — the model executed them when it didn't hallucinate.
- No model swap. The research finding from the original plan still holds: Gemma 4 E4B is the right pick for this hardware in June 2026, and the failures observed were prompt-level, not capacity-level.

## Re-evaluation gate

Before tagging `v2026.06-final-ready`, re-run the 8 Python questions on the
updated prompt and confirm:
- Q4 produces exactly one Assumptions/code/Notes triple.
- Q5 alpha panel shows a visible base layer or grid through the transparency.
- Q6 uses the inlined `M` definition without inventing a new matrix.
- Q7 produces one axes with 8 stacked KDE ridges, not 8 subplots.

## Round 2 results (re-eval against the fixed prompt + questions)

All four targeted questions PASS. Logged in `PYTHON_LAB04_REEVALUATE.MD`.

| Q | Verdict | Detail |
|---|---|---|
| Q4 | PASS | One triple, stops cleanly after Notes. The single-answer rule held. |
| Q5 | PASS | Used Pattern B (white grid behind alpha imshow) and explicitly attributed the choice in Notes. The alpha panel now demonstrates what the technique is for. |
| Q6 | PASS | Used the inlined seed/normal/`+= 20` block verbatim. cmap stops anchored at the 5/50/95 quantiles, normalised positions, deduplication present. |
| Q7 | PASS | Single axes, 8 stacked ridges, y-ticks set to group names. The "no joypy" trick aspect held. |

### Residual nits (intentionally NOT fixed — over-tuning risk)

- Q7 stacked ridges: model used raw `dens = kde(xs)` instead of the
  `dens = dens / dens.max() * 0.4` normalisation the prompt's fallback recipe
  shows. Ridges may be unevenly scaled. Structure is correct, answer earns
  marks. Prompt already teaches the normalisation — pushing further risks
  over-fitting.
- Q7 Notes prose: model wrote "plt.fill_between_x" (no such function — it is
  `plt.fill_betweenx`). Inside Notes prose, not the code, so harmless.

Python half of the eval-iterate loop is now CLOSED. Tag `v2026.06-final-ready`
can be applied once the R half of the loop reaches the same closed state.
