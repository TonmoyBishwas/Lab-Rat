# Lab Rat AI — dress-rehearsal audit (final scorecard)

Source paper: `data_viz_final_docs/another_section_questions/{1.jpeg, 2.jpeg}` —
DS 3522 Data Visualization Lab Final, same teacher, same difficulty as the
student's upcoming paper. 30 marks total: 22 R + 8 Python.

Question bank: `another_section_final_questions.md`.
Answers + RStudio / Jupyter outputs: `ANOSTER_SEC.MD`.

## Final scorecard

5/5 PASS. All questions produce runnable code matching the marks scheme.

| Q | Lang | Marks | Verdict | Notes |
|---|---|---|---|---|
| Q1 airquality line chart | R | 6 | PASS (Round 2) | Fixed by adding built-in-datasets recipe to R prompt. |
| Q2 iris scatter + per-species trend | R | 8 | PASS (Round 1) | Clean from the start. |
| Q3 mtcars facet + trend | R | 8 | PASS (Round 1) | Clean from the start. |
| Q4 colour-harmony palettes | Python | 4 | PASS (Round 2) | Fixed by adding "ALL palettes start with base hue h" rule. |
| Q5 RGB + HSI channels | Python | 4 | PASS (Round 2) | Fixed by adding explicit img_as_float import + HSI=(R+G+B)/3 recipe. |

## Round-by-round failure modes

### Round 1 failures and fixes

| Q | Failure | Fix committed in |
|---|---|---|
| Q1 | Filtered `Day >= 62 & Day <= 92` thinking Day was day-of-year; tried to convert via `lubridate::days()`; errored on empty vector. | `f677f9f` — added BUILT-IN R DATASETS section to R prompt with explicit airquality column semantics. |
| Q4 | Wrote `h_comp = [h_base + 0.5]` — only one swatch (the complement), dropped the base hue. | `7e6ebd3` — strengthened hue arithmetic recipe with "ALL THREE palettes ALWAYS start with the base hue h itself as the FIRST element. Never drop h." |
| Q5 | `NameError: name 'img_as_float' is not defined`. Model wrote `import skimage.util` but called bare `img_as_float()`. | `7e6ebd3` — added explicit `from skimage.util import img_as_float` to the prompt's imports block with the exact NameError text quoted. |

### Round 2 verifications

All three re-runs produced clean, runnable code that matches the marks
scheme. Specifically:

- Q1: `airquality |> filter(Month == 7)` then `ggplot(... aes(x = Day, y = Temp))` —
  no lubridate, no synthetic data, plot renders.
- Q4: complementary palette is `[h, h + 0.5]` — exactly 2 swatches as the
  spec requires. Triadic (3) and analogous (3) also correct.
- Q5: model now correctly computes HSI Intensity as
  `I = img_float[..., 0:3].mean(axis=-1)` — the proper (R+G+B)/3 channel,
  not the V channel from rgb2hsv. Improvement over the
  acceptable-but-imprecise V fallback the prompt offers.

## What this means for the real exam

The student walks in with prompts that have been:
- Calibrated against the same teacher's previous final paper.
- Stress-tested across two evaluation rounds and three dress-rehearsal
  rounds (Python: round 1, round 2 + Q4/Q5 fix; R: round 1, round 2 + Q1
  fix).
- Anti-pattern-armoured against every failure mode actually observed.

No known failure modes remain on the question shapes that appear in the
paper. The system is ready.

## Tag

This state is tagged `v2026.06-exam-ready` on origin/main.
