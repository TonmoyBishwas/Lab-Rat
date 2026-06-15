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
