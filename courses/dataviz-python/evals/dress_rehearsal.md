# Another-section final exam — OCR'd from the question paper

**Course:** DS 3522 Data Visualization Lab — Final
**Duration:** 1 hour 30 minutes
**Total marks:** 30
**Student on paper:** Nishat Jahan (0152410024)
**Source:** `data_viz_final_docs/another_section_questions/{1.jpeg, 2.jpeg}`

## Imports allowed (from page 2 of the paper)

R-Studio:
- `library(ggplot2)`

Python:
- `import numpy`
- `import matplotlib.pyplot`
- `import colorsys`
- `from skimage import data, color`

Note: the official Python imports list uses **colorsys** (Python stdlib) for
the hue-arithmetic question, not `skimage.color.hsv2rgb`. The model's answer
should still produce correct output either way, but if the marker grades to
the imports list literally, `colorsys.hsv_to_rgb` is the expected call.

## Mark distribution

| Q | Language | Marks |
|---|---|---|
| 1 | R | 6 |
| 2 | R | 8 |
| 3 | R | 8 |
| 4 | Python | 4 |
| 5 | Python | 4 |
| **Total** | | **30** |

R = 22 marks (73 %), Python = 8 marks (27 %). R mode matters more.

---

## Q1 — R · Line chart of daily temperature (6 marks)

An environmental analyst is reviewing daily readings from New York's summer
of 1973 (R's built-in air quality dataset) and wants to chart how
temperature changed across one month.

Using R-programming, draw a **line chart** of daily temperature across one
month.

*Hint for the system: this uses the built-in `airquality` dataset
(`?airquality`), which has columns `Ozone`, `Solar.R`, `Wind`, `Temp`,
`Month`, `Day`. Pick one month (e.g. `Month == 7` for July) and plot
`Day` vs `Temp`. NOT synthetic data — use the real built-in.*

---

## Q2 — R · iris scatter with per-species trend lines (8 marks)

Your earlier petal-length sketch caught the eye of the journal's figure
editor, who now wants a polished, colour-coded figure suitable for print
using the ggplot2 grammar of graphics.

Using R-programming and the built-in **iris** data, draw a **scatterplot**
of petal length vs petal width, colour-grouped by species, with per-group
trend lines.

*Expected: `ggplot(iris, aes(x = Petal.Width, y = Petal.Length,
color = Species)) + geom_point() + geom_smooth(method = "lm", se = FALSE)`.*

---

## Q3 — R · Faceted mtcars scatter with trend lines (8 marks)

The automotive analyst from the first assignment is back. This time they
want the mpg-versus-weight relationship for each cylinder class shown in
its own panel, side by side, so the engine classes can be compared at a
glance.

Using R-programming and the built-in **mtcars** data, draw a **faceted,
one-panel-per-cylinder-class scatter plot** with trend lines.

*Expected: `ggplot(mtcars, aes(x = wt, y = mpg)) + geom_point() +
geom_smooth(method = "lm", se = FALSE) + facet_wrap(~ cyl)`.*

---

## Q4 — Python · Colour-harmony palettes (4 marks)

Pick a base hue h ∈ [0, 1]. Generate complementary, triadic, and analogous
palettes from that hue and display the swatches for all three.

*Imports allowed: `numpy`, `matplotlib.pyplot`, `colorsys`. The official
import list uses `colorsys.hsv_to_rgb(h, s, v)` — same H/S/V semantics as
`skimage.color.hsv2rgb` but returns an `(R, G, B)` tuple instead of an
array. Either is acceptable.*

---

## Q5 — Python · RGB and HSI channel display (4 marks)

Load a bundled scikit-image sample image (`skimage.data.coffee()`). Show
its R, G, B channels as three greyscale images, and its H, S, I channels
as three more.

*IMPORTANT — read literally: the question says **HSI**, not HSV.*

- The `H` and `S` channels are the same between HSI and HSV (`color.rgb2hsv`
  gives both correctly).
- The `I` (Intensity) channel is **NOT** the V channel from `rgb2hsv`. HSI
  intensity is the average of the three RGB channels:
  `I = (R + G + B) / 3` — call this on `img_as_float(img)` so values stay in
  `[0, 1]`.

The model needs to either (a) compute I = mean of R/G/B explicitly, or
(b) treat HSI as a typo/synonym for HSV and use the V channel directly. If
it does (b), flag it but it usually earns marks because most undergrad
texts use HSV/HSI interchangeably.

---

## How to evaluate

1. Run Q1–Q3 in **R mode** (`python start_r.py` → hard-refresh browser to
   `localhost:8080`). Paste each question text verbatim. Paste the
   response into `another_section_final_answers.md` under the matching
   heading.
2. Run the R code in RStudio. Confirm the plot renders. Paste any error
   messages or output above the failing block.
3. Run Q4–Q5 in **Python mode** (`python start.py`). Paste responses.
   Run the Python code in Jupyter / Spyder / wherever you have matplotlib
   to confirm the plot renders.
4. When all 5 answers + R Studio / Jupyter outputs are in, ping me. I'll
   grade each against the marks scheme:
   - Plot type matches the question
   - Code runs without error
   - All required elements present (axes labels, title, colour grouping,
     trend lines, etc.)
   - No phantom imports beyond the official list
