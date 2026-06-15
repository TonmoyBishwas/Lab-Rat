# Lab Rat AI — R evaluation weaknesses & fixes (Round 1)

Model: Gemma 4 E4B-it Q4_K_M, served via llama-server.exe on `start_r.py`.
Question bank: `r_questions.md` (8 questions).
Answers: `r_answers.md` (includes actual RStudio console output per question).

## Score

5/8 clean passes · 2 runtime failures (environment) · 1 logic failure.

## Per-question findings

| Q | Verdict | Finding | Fixed in |
|---|---|---|---|
| Q1 Line chart | RUNTIME FAIL | `Error in library(tidyverse) : there is no package called 'tidyverse'`. Code logic was fine. Two distinct issues: environment lacks tidyverse, and `library(tidyverse)` is a bad practice anyway. | Environment install + prompt anti-pattern |
| Q2 Vector stats | PASS (nit) | All six outputs correct (Mean 5, Median 4, SD 2.915, Min 2, Max 9, values>mean = 7,9). Minor: model added `library(base)` which is a no-op. | Prompt anti-pattern |
| Q3 iris scatter | PASS | `library(ggplot2)` only, ran successfully, saved Rplot. Textbook `geom_smooth(method="lm", se=FALSE)`. | — |
| Q4 mtcars facet | PASS | `facet_wrap(~ cyl)` + per-panel trend. Ran successfully, saved Rplot.png. Informational `geom_smooth() using formula 'y ~ x'` message printed. | — |
| Q5 dplyr pipeline | RUNTIME FAIL | `Error in library(dplyr) : there is no package called 'dplyr'`. dplyr code itself was correct (group_by → summarise(.groups="drop") → arrange(desc) → select). | Environment install |
| Q6 Vectorised FizzBuzz | LOGIC FAIL | Model wrote `x \|> mutate(result = case_when(...)) \|> pull(result)`. `x` is a bare vector — `mutate()` only operates on data frame columns. Would error at runtime even with dplyr installed. Prompt has the correct `ifelse` recipe in anti-patterns; model used `case_when` instead and bolted on a bogus `mutate()`. | Prompt anti-pattern |
| Q7 Base-R filter | PASS (nit) | `mtcars[mtcars$mpg > 20, ]` correct, 14 rows printed exactly right. Over-engineered with `if (!exists("mtcars"))` defensive fallback. Also had pointless `library(base)`. | Prompt anti-pattern |
| Q8 0-to-4 for-loop | PASS | `for (i in 0:4) print(i)` — held the trick perfectly. Printed 0,1,2,3,4 exactly. | — |

## Root causes & decisions

There are **two distinct categories** of failures here, which need separate
remedies.

### Category A: Environment (user action, not prompt)

The R install used for evaluation has `ggplot2` and `base` but is missing
`dplyr`, `tidyr`, `readr`, `lubridate`. This is why Q1, Q5, Q6 errored at
`library(...)`. **User must run, once, in RStudio:**

```r
install.packages(c("dplyr", "tidyr", "readr", "lubridate", "tidyverse"))
```

After this is done, re-run Q1 and Q5 — they should pass on the existing
generated code without any prompt change. Q6 still needs the prompt fix
because the code is logically wrong, not just import-wrong.

### Category B: Prompt — four anti-patterns added

Committed in `cf270d5` to `ui/r_lang.html`:

1. **`library(tidyverse)` is banned.** Always library() individual packages.
   Hides dependencies, slows startup, fails silently when only part of
   tidyverse is installed. Fixes the structural cause of Q1.
2. **`library(base)` is banned.** Base R is auto-loaded. Writing
   `library(base)` is a no-op tell. Cleans up Q2 and Q7 noise.
3. **`mutate()` on a bare vector is banned.** mutate() works only on data
   frame columns. For a vector `x`, write `result <- case_when(...)` or
   `result <- ifelse(...)` directly. Same rule applies to summarise(),
   select(), filter(), arrange(). **Fixes the actual logic bug in Q6.**
4. **Defensive `exists()` checks for built-in datasets are banned.**
   mtcars, iris, airquality, etc. are always available via the `datasets`
   package. Cleans up Q7 over-engineering.

## What was NOT changed and why

- The "Libraries available:" header line was left as-is. It correctly names
  what the prompt expects (tidyverse + lubridate); the fix for "but the user
  doesn't have them" is to install them, not to lie to the model.
- The vectorised-FizzBuzz `ifelse` recipe in anti-patterns is correct as-is.
  Q6 failed because the model chose `case_when` over `ifelse` and then
  misused `mutate`, not because the recipe was wrong.
- No model swap. Same reasoning as Python: failures are prompt-level or
  environment-level, not capacity-level.

## Re-evaluation gate

Before tagging `v2026.06-final-ready`, complete two steps:

1. **Install missing R packages** (one-time, in RStudio):
   ```r
   install.packages(c("dplyr", "tidyr", "readr", "lubridate", "tidyverse"))
   ```
2. **Re-run only Q1, Q5, Q6** in R mode against the updated prompt. Confirm:
   - Q1 uses `library(ggplot2)` + `library(lubridate)` (not `library(tidyverse)`)
     and the line chart renders.
   - Q5 uses `library(dplyr)` and the pipeline runs end-to-end.
   - Q6 writes `result <- case_when(...)` (or `ifelse(...)`) on the bare
     vector `x` with NO `mutate()` and NO pipe into mutate. Output is the
     correct 30-element character vector.

## Round 2 results (logged in `R_REEVALUATE.MD`)

Mixed. One win, one regression with a new failure mode, one re-test was
mis-pasted under the wrong heading.

| Q | Verdict | Detail |
|---|---|---|
| Q1 | NOT TESTED | The FizzBuzz answer was pasted under the Q1 heading by mistake. Q1 line chart was not actually re-run. |
| Q5 | PASS | `library(dplyr)` + clean group_by → summarise(.groups="drop") → arrange(desc) pipeline. Output is the correct virginica/versicolor/setosa tibble. The install fixed it. |
| Q6 | STILL FAILING (new mode) | Model learned to drop `mutate()` but discovered a different wrong pattern: `numbers \|> case_when(...)`. Errors with "Case 1 (`numbers`) must be a two-sided formula, not an integer vector." The Round 1 anti-pattern banned mutate-on-vector but did not explicitly ban piping into case_when. |

## Round 2 fix (committed in `2c61f9d`)

The R prompt's dplyr section was reworked to make the data-frame vs.
bare-vector split explicit, with a copy-this-verbatim FizzBuzz template for
the bare-vector case. A second anti-pattern bullet was added that quotes the
exact runtime error message the model just produced, so the rule is
semantically anchored to the failure.

Also removed a stale "(or library(tidyverse))" hint from the recipes intro
that contradicted the existing "never library(tidyverse)" anti-pattern.

## Round 3 gate (next re-evaluation)

1. **Hard-refresh the browser** (Ctrl+F5 on `localhost:8080` after starting
   `python start_r.py`). The system prompt is set when the page loads — an
   already-open tab keeps the OLD prompt in its textarea even though the
   server file changed. Verify by opening Settings and confirming the
   prompt contains the "BARE VECTOR context" subsection.
2. **Re-run Q1 and Q6 only** (Q5 already passed Round 2). Confirm:
   - Q1: line chart renders, uses individual library() calls, dates format
     as `Jun 01`.
   - Q6: `result <- case_when(x %% 15 == 0 ~ "FizzBuzz", ...)` with NO pipe
     and NO mutate. Output is the correct 30-element character vector
     beginning `[1] "1" "2" "Fizz" "4" "Buzz" "Fizz" "7" "8" "Fizz" "Buzz"`.

## Round 3 results (logged in `R_REEVALUATE.MD`)

Both targets PASS. R half of the eval-iterate loop is now CLOSED.

| Q | Verdict | Detail |
|---|---|---|
| Q1 | PASS | Three targeted `library()` calls (ggplot2, lubridate, dplyr), `data.frame` + `ymd` + `seq("day")`, `scale_x_date(date_labels = "%b %d")`, `print(p)` rendered the plot. No `library(tidyverse)`, no `library(base)`. |
| Q6 | PASS | Clean bare-vector pattern: `library(dplyr); x <- 1:30; result <- case_when(x %% 15 == 0 ~ "FizzBuzz", ...)`. NO pipe, NO mutate. Output element-by-element correct, ending `...28 29 FizzBuzz`. |

### Residual nits (NOT fixed)

- Q1 imports `library(dplyr)` but never calls a dplyr function. Harmless dead
  import. Pushing a "only library() what you actually use" rule would risk
  regressing harder cases — accepted as-is.

### Final R scorecard

8/8 pass after three rounds:
- Q2, Q3, Q4, Q7, Q8 — passed Round 1 cleanly.
- Q5 — passed Round 2 after the package install.
- Q1, Q6 — passed Round 3 after the bare-vector case_when patch.

Tagged `v2026.06-final-ready` once Python + R loops both close.
