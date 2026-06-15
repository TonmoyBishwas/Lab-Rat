# Lab Rat AI — Python (Lab 04) evaluation bank

Eight questions. Paste each one into the General launcher (`launch-qwen.bat` or
`launch-gemma4.bat`) one at a time. Copy the full model response into
`python_lab04_answers.md` under the matching heading.

The bank is calibrated against:
- The visible questions in `data_viz_final_docs/another_section_questions/`
- Lab 04 Parts A (spread plots), B (colour theory), C (heatmaps + outliers)
- Known small-model failure modes recorded in `system_prompt_calibration.md`

Difficulty legend: **EASY** / **MED** / **HARD** / **TRICK**.

---

## Q1 — EASY · RGB and HSV channel display

Load `skimage.data.coffee()`. Show its three RGB channels and three HSV channels
as six greyscale subplots arranged in a 2×3 grid. Title each subplot with its
channel name (R, G, B, H, S, V).

*Mirrors another-section Q5. Probes the `img_as_float` + `rgb2hsv` recipe and
the `cmap='gray'` convention.*

---

## Q2 — EASY · Three colour-harmony palettes from one hue

Take base hue `h = 0.55`. Build and display three colour palettes as horizontal
swatch rows in a 3×1 subplot grid:

1. Complementary (2 swatches)
2. Triadic (3 swatches)
3. Analogous (3 swatches, ±1/12 from h)

Use `skimage.color.hsv2rgb` and saturation/value of `(0.65, 0.9)`. Wrap hues
into `[0, 1]` with `np.mod`.

*Mirrors another-section Q4. Probes the hue-wrap rule and the swatch-row
template.*

---

## Q3 — MED · Manual violin plot of `iris` petal length

Draw a violin plot of `iris` petal length per species — but **do not use
`seaborn.violinplot` or `seaborn.violin`**. Build it manually with
`scipy.stats.gaussian_kde` and `fill_betweenx`. One violin per species
positioned at `x = 0, 1, 2`. Use the species names as x-tick labels.

*Probes: the manual-violin recipe, the `fill_betweenx(xs, i - dens, i + dens)`
argument order, and that the model does not silently fall back to
`sns.violinplot`.*

---

## Q4 — MED · Divergent heatmap of the iris correlation matrix

Build a correlation matrix of the four numeric columns of `iris`. Plot it as a
heatmap with a **divergent colormap** centred at zero (blue → white → red).
Annotate each cell with the correlation value. Build the colormap as a variable
with `LinearSegmentedColormap.from_list`; do not pass a bare string.

*Probes: divergent colormap construction, `center=0`, and the
`numeric_only=True` / `select_dtypes` guard against the species column.*

---

## Q5 — HARD · All four outlier-handling techniques side-by-side

Generate a noisy 50×50 matrix:
```
np.random.seed(0)
M = np.random.normal(0, 1, (50, 50))
M[::5, ::5] += 20   # planted outliers
```
Plot a 2×2 subplot grid showing the **four outlier-handling techniques** from
Lab 04 Part C, each applied to `M`:

1. Clip to the 2nd–98th quantile
2. Quantile-rank normalisation
3. Signed log1p (`np.sign(M) * np.log1p(np.abs(M))`)
4. Alpha overlay (`ax.imshow(M, alpha=0.85)`)

Title each subplot with its technique name.

*The inferred high-weight Q6 from the other section. Probes the full
outlier-handling block, including the alpha-overlay rule that was just added
to the prompt.*

---

## Q6 — HARD · Multi-checkpoint quantile-anchored colormap

Generate this noisy matrix (paste the block verbatim along with the question
text — the launcher does NOT remember Q5):
```python
np.random.seed(0)
M = np.random.normal(0, 1, (50, 50))
M[::5, ::5] += 20
```
Build a custom **piecewise colormap** whose colour stops are anchored at the
5th, 50th, and 95th quantiles of `M` (colours: navy → white → darkred).
Normalise the stop positions to `[0, 1]` using `(q - vmin) / (vmax - vmin)`.
Deduplicate stops that are within `1e-6` of each other. Pass the cmap to
`sns.heatmap` with the same `vmin`/`vmax`.

*Probes: the worked-example structure of multi-checkpoint
`LinearSegmentedColormap.from_list` — specifically that the model uses the
COMPUTED normalised positions and does not silently fall back to the bare
colour list. Self-contained so the model cannot drift to a fresh matrix.*

---

## Q7 — TRICK · Ridgeline of 8 groups inside a 2×4 subplot grid

Generate synthetic data for 8 groups (A–H), each with 200 samples drawn from a
normal distribution with a different mean. Plot a **ridgeline (joypy-style)
visualisation** with each ridge inside its own slot of a `2×4 plt.subplots`
grid.

*Trick: this should NOT use `joypy.joyplot`. The prompt mandates the manual
stacked-KDE fallback whenever the question lives inside a subplot grid. If the
model uses joypy in any slot, mark this as a failure.*

---

## Q8 — TRICK · Geometric mean and contra-harmonic mean of a vector

For `x = np.array([1, 2, 4, 8])`, compute and print:

1. The geometric mean
2. The contra-harmonic mean (order p = 1)

Use the calibrated formulas. Output values to four decimal places.

*Probes: the formula calibration — GM = `np.exp(np.log(x).mean())`,
CHM = `(x**2).sum() / x.sum()`. Common failure: harmonic-mean / geometric-mean
confusion, or wrong CHM definition. Lab 02 carryover that may appear on the
final.*

---

## How to record results

After running each question, paste the full model output into
`python_lab04_answers.md` under the same heading. When all eight are done, send
me a message and I'll read the answers, write `weaknesses_python.md`, and
propose a targeted prompt fix for each failure.
