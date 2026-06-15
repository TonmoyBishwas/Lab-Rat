# Lab Rat AI — R programming evaluation bank

Eight questions. Paste each one into the R launcher (`launch-r.bat` → opens at
`http://localhost:8080`) one at a time. Copy the full model response into
`r_answers.md` under the matching heading.

The bank is calibrated against:
- The visible R questions in `data_viz_final_docs/another_section_questions/`
  (line chart, grouped scatter + `geom_smooth`, faceted scatter + trend)
- The R syllabus in `data_viz_final_docs/assignment_task.pdf`
- Predicted small-model failure modes for R (Python idioms, zero-indexing,
  missing `na.rm`, `&` vs `&&`, etc.)

Difficulty legend: **EASY** / **MED** / **HARD** / **TRICK**.

For every R question, render the model's code in RStudio to verify it actually
runs. A correctly-typed answer that errors at runtime is still a failure.

---

## Q1 — EASY · Line chart of one month of daily temperature

Using `ggplot2`, plot a line chart of daily temperature across one month
(30 days). Generate synthetic data: dates from `2026-06-01` to `2026-06-30`,
temperatures around 30 °C with `rnorm` noise. Add points on top of the line.
Title the plot, label the axes, format the x-axis dates as `Jun 01`, `Jun 05`,
etc.

*Mirrors another-section Q1. Probes the `aes(x=date, y=temp) + geom_line()` +
`labs()` recipe and `scale_x_date(date_labels = "%b %d")`.*

---

## Q2 — EASY · Vector statistics with one NA, ignoring NA correctly

For the vector `x <- c(3, 7, 2, NA, 9, 4)`, compute and print:

1. mean
2. median
3. standard deviation
4. minimum
5. maximum
6. the values of x that are strictly greater than the mean

All aggregates must ignore the NA. The final filter must also ignore the NA.

*Probes: `na.rm = TRUE` on every aggregate. Common failure: omitting `na.rm`,
in which case the model's output silently becomes NA and the student loses
marks.*

---

## Q3 — MED · iris scatter with per-species linear trend lines

Using `ggplot2` on the built-in `iris` dataset, draw a scatter plot of
`Petal.Length` (y) vs `Petal.Width` (x), colour-coded by `Species`. Overlay a
**per-species linear regression line** using `geom_smooth(method = "lm",
se = FALSE)`. Include a title and properly-labelled axes.

*Mirrors another-section Q2. Probes the grouped scatter + `geom_smooth` recipe
and the `se = FALSE` rule.*

---

## Q4 — MED · Faceted mtcars scatter with per-panel trend

Using `ggplot2` on `mtcars`, draw a scatter plot of `mpg` (y) vs `wt` (x),
**faceted by cylinder count** so there is one panel per value of `cyl`. In each
panel, overlay a linear regression line. Title the figure.

*Mirrors another-section Q3. Probes `facet_wrap(~ cyl)` and per-panel
`geom_smooth`.*

---

## Q5 — MED · dplyr pipeline — group, summarise, arrange

Using `dplyr` on `iris`, build a single pipeline that:

1. Groups by `Species`
2. Computes the mean `Petal.Length` per species
3. Arranges the result in descending order by that mean
4. Returns a data frame with two columns: `Species` and `mean_pl`

Use the native pipe `|>` and close the `group_by` chain correctly so no
"grouped output" warning is printed.

*Probes: the `group_by → summarise(.groups = "drop") → arrange(desc(...))`
recipe and the `|>` preference.*

---

## Q6 — HARD · Vectorised FizzBuzz, 1 through 30

Print the FizzBuzz sequence from 1 to 30:

- Multiples of 15 print `"FizzBuzz"`
- Multiples of 3 (but not 5) print `"Fizz"`
- Multiples of 5 (but not 3) print `"Buzz"`
- Everything else prints the number itself

**Do not use a for-loop.** Use only vectorised operations (`ifelse` or
`case_when`) and `%%`.

*Mirrors assignment_task practice task 4. Probes vectorisation discipline.*

---

## Q7 — TRICK · Filter mtcars without dplyr or the pipe

Print every row of `mtcars` where `mpg > 20`. Do not use `dplyr`,
`subset()`, or any pipe (`|>` or `%>%`). Use only **base-R indexing**.

*Trick: the prompt teaches `df |> filter(...)` and `subset()`. Expected answer:
`mtcars[mtcars$mpg > 20, ]` — note the trailing comma. Common failure: writing
`mtcars[mtcars$mpg > 20]` (no comma — collapses to vector indexing on the
first column) or silently using `subset()` anyway.*

---

## Q8 — TRICK · Print 0 through 4 inclusive with a for-loop in R

Write an R for-loop that prints the integers `0, 1, 2, 3, 4` — i.e. five
numbers starting from zero, not one. Use `for` and `print`.

*Trick: R is 1-indexed but this question explicitly demands a zero-based
range. Expected answer: `for (i in 0:4) print(i)`. Common failure: the model
"corrects" the range to `1:5` because it learned that R loops use 1-indexed
sequences. The right behaviour is to follow the user's spec literally — the
range here is data, not an index into a vector.*

---

## How to record results

After running each question, paste the full model output into
`r_answers.md` under the same heading. **Render each answer in RStudio** to
confirm it executes without error and produces the expected plot/output. Flag
any runtime errors with a `RUNTIME ERROR:` line above the failing block.

When all eight are done, send me a message and I'll read the answers, write
`weaknesses_r.md`, and propose a targeted prompt fix for each failure.
