# Lab Rat AI — exam-day stress-test bank

Four targeted probes designed to surface remaining weaknesses BEFORE the
real exam. Weighted to match the another-section paper: 3 R + 1 Python.

Each question is a likely variation of the patterns seen in
`another_section_questions/{1,2}.jpeg`. None of these exact questions
appear there — that is the point. We want to know whether the recipes
generalise to nearby shapes the student might actually face.

Pass criteria for every question:
- Runnable code (no syntax errors, no runtime errors when pasted into
  RStudio / Jupyter).
- **Imports match the paper's allowed list strictly.** R = `library(ggplot2)`
  only. Python = numpy, matplotlib.pyplot, colorsys, skimage (data, color,
  util.img_as_float).
- Plot shape matches the question literally.

---

## RQ1 — R · Histogram on a built-in dataset (likely 4–6 marks)

Using R's built-in **mtcars** dataset, draw a **histogram** of `mpg` with 10
bins, filled in steelblue. Add a title, axis labels, and `theme_minimal()`.

*Probes: `geom_histogram(bins = 10, fill = "steelblue", color = "white")`,
library(ggplot2) ONLY. Watch for the model reaching for `library(dplyr)` or
inventing `library(scales)`. If it imports anything beyond ggplot2 without
the question naming the package, that's a failure.*

---

## RQ2 — R · Boxplot on iris, coloured by Species (likely 6 marks)

Using R's built-in **iris** dataset, draw a **boxplot** of `Sepal.Length`
(y) by `Species` (x). Fill each box with a different colour by mapping
fill to Species. Add a title and `theme_minimal()`. **Use ggplot2 only.**

*Probes: `geom_boxplot()` with `aes(x = Species, y = Sepal.Length, fill =
Species)`, library(ggplot2) only. Watch for the model writing
`base::boxplot(...)` (would also pass but the question says ggplot grammar
implicitly via "use ggplot2 only"). Watch for `library(RColorBrewer)` or
similar — should not appear.*

---

## RQ3 — R · Scatter + trend on airquality, with NA filtering (likely 6–8 marks)

Using R's built-in **airquality** dataset, plot `Ozone` (y) vs `Wind` (x)
as a **scatter plot**, with a **single linear trend line** (`geom_smooth`)
overlaid. The `Ozone` column has NA values — filter them out using base R
before plotting. Title, axis labels, `theme_minimal()`. **library(ggplot2)
only.**

*Probes: base-R subsetting on a logical condition (`airquality[!is.na(airquality$Ozone), ]`),
`geom_smooth(method = "lm", se = FALSE)` without grouping, library(ggplot2)
only. Watch for the model reaching for `library(dplyr)` to do `filter(!is.na(Ozone))`
— that would lose marks under strict grading.*

---

## PQ1 — Python · Triadic palette using colorsys (likely 4 marks)

Pick base hue **h = 0.7**. Generate a **triadic palette** (3 colours).
Display the 3 colour swatches in a 1×3 row of subplots using
`matplotlib.patches.Rectangle`.

**Use `colorsys.hsv_to_rgb` (Python standard library), NOT
`skimage.color.hsv2rgb`** — the exam paper's official import list says
`import colorsys`.

Saturation = 0.65, Value = 0.9. Wrap hues with `numpy.mod`.

*Probes: the exact paper-listed `colorsys` import (not skimage); triadic
palette as `[h, h+1/3, h+2/3]` (3 swatches, including h); np.mod wrap;
Rectangle swatch pattern. Watch for the model using
`skimage.color.hsv2rgb` instead of `colorsys.hsv_to_rgb` — both run, but
the paper specifies colorsys. Watch for dropping h from the palette.*

---

## How to run

1. Hard-refresh the browser (Ctrl+F5) on both `start_r.py` and `start.py`
   so the latest prompts load.
2. Paste each question verbatim into the matching mode.
3. Run R answers in RStudio, Python in Jupyter / Spyder.
4. Paste full responses + console output into `stress_test_answers.md`.
5. Ping me. I will grade against pass criteria and patch any remaining
   weakness.
