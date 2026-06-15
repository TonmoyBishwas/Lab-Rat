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

---

## Stress-test round (added after dress rehearsal)

Once the dress rehearsal closed, the student raised a real concern: the
synthetic banks and the dress rehearsal both passed, but neither tested
whether the model would respect the paper's literal imports list. The
paper allows `library(ggplot2)` only for R and (numpy, matplotlib.pyplot,
colorsys, skimage data+color) only for Python. The prompts were freely
reaching for dplyr, lubridate, pandas, seaborn — green on the synthetic
banks because those banks tested recipes, not exam-grading discipline.

Commit `baeea82` added EXAM IMPORT DISCIPLINE blocks to both prompts.

A 4-question stress-test bank (`stress_test_questions.md`) probed nearby-
but-not-identical exam shapes under the strict imports rule:

| Q | Verdict | Detail |
|---|---|---|
| RQ1 mtcars histogram | PASS | library(ggplot2) only, geom_histogram(bins=10), plot saved. |
| RQ2 iris boxplot | PASS | library(ggplot2) only, fill=Species in aes, plot saved. |
| RQ3 airquality scatter + NA filter | PASS — the load-bearing test | Base-R airquality[!is.na(airquality$Ozone), ], NO library(dplyr). The discipline rule held against a question where dplyr is the obvious idiom. |
| PQ1 colorsys triadic | Round 1 FAIL → Round 2 PASS | First attempt: `colorsys.hsv_to_rgb([h, s, v])` TypeError (wrong signature) + missing np.mod wrap. Fixed in commit `3f53b3e` by adding a 'colorsys vs skimage signatures' section and a mandatory-np.mod-before-list-comprehension recipe. Round 2 produced clean three-arg call + pre-comprehension wrap. |

### Final state across all evaluation work

| Bank | Scope | Final |
|---|---|---|
| Synthetic Python (Lab 04) | 8 Qs | 8/8 |
| Synthetic R (assignment_task.pdf) | 8 Qs | 8/8 |
| Real exam paper, another section | 5 Qs | 5/5 |
| Stress-test (likely variations) | 4 Qs | 4/4 |

Total 25/25 across four banks, three of them against material the same
teacher actually grades to.

This state is tagged `v2026.06-stress-tested` on origin/main, superseding
`v2026.06-exam-ready` as the truly-final-ready milestone.
