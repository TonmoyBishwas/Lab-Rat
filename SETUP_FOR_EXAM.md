# Setup for the DS 3522 Data Visualization Lab Final

What to install on your exam laptop **before** the paper is in front of you.

Two tiers per language:
- **Exam essentials** — the literal imports list on the paper's page 2. If
  you only install these, every question on the paper can be answered.
- **Broader syllabus** — everything Lab Rat AI may produce based on the full
  Lab 04 notebook and R assignment. Install these too if you have time —
  they unlock any nearby variations the student's section could face.

---

## Python — for Lab 04 questions

### Exam essentials (paper page 2)

```
import numpy
import matplotlib.pyplot
import colorsys                          # Python stdlib — no install
from skimage import data, color
```

`colorsys` is in the Python standard library and needs no install. The other
three need `pip`.

### One-line install

**In a terminal (Windows PowerShell, macOS, Linux, or VS Code's terminal):**
```powershell
pip install numpy matplotlib scikit-image
```

**Inside a Jupyter Notebook cell:**
```python
!pip install numpy matplotlib scikit-image
```

The skimage package is installed via `pip install scikit-image` — note the
`scikit-image` package name vs the `skimage` import name. They are different.

### Broader Lab 04 syllabus (recommended for safety)

```powershell
pip install numpy pandas matplotlib seaborn scipy scikit-image joypy palettable
```

What each package gives you:
- `numpy` — arrays, math, `np.mod`, `np.random`, `np.quantile`, `np.clip`
- `pandas` — DataFrames, `iris.corr()`, `select_dtypes`
- `matplotlib` — `plt.plot`, `plt.imshow`, `plt.subplots`, `Rectangle` patches
- `seaborn` — `sns.heatmap`, `sns.boxplot`, `sns.violinplot`, `sns.load_dataset`
- `scipy` — `scipy.stats.gaussian_kde` for manual violins and ridgelines
- `scikit-image` (imported as `skimage`) — image loading and colour conversion
- `joypy` — ridgeline plots (standalone, not inside subplot grids)
- `palettable` — `Set2_8.mpl_colors` for the seaborn heatmap palette recipe

### Verify the install worked

In a Jupyter cell or Python REPL:
```python
import numpy, pandas, matplotlib, seaborn, scipy, skimage, joypy, palettable, colorsys
print("all good")
```

If any line errors with `ModuleNotFoundError`, run `pip install` on the
package name. If `seaborn` errors, run `pip install seaborn`. Same for any
other.

### VS Code specifics

- Open the project folder.
- Press `` Ctrl + ` `` to open the integrated terminal.
- Run the `pip install ...` command above.
- If VS Code complains "Python interpreter not selected", click the version
  in the bottom-right and pick the Python you ran pip with.
- For `.ipynb` notebook files, install the **Jupyter** extension once. Then
  VS Code can run notebook cells directly. Use the same install commands
  prefixed with `!` inside a cell if you prefer.

### Jupyter Notebook specifics

If you launch `jupyter notebook` from a terminal where you ran `pip install`,
the imports just work. If they don't, run the install commands inside a
notebook cell with the `!` prefix:
```python
!pip install numpy matplotlib scikit-image
```

---

## R — for R-programming questions

### Exam essentials (paper page 2)

```r
library(ggplot2)
```

That is the entire allowed surface on the paper. Strict markers will dock
marks for any other library that appears in the answer unless the question
text explicitly names the package.

### One-line install in RStudio

Open RStudio, then paste this into the Console (the pane at the bottom-left)
and press Enter:
```r
install.packages("ggplot2")
```

It will download from CRAN, compile, and finish. You only need to do this
ONCE per machine. After that, every R session can `library(ggplot2)` without
re-installing.

### Broader R syllabus (recommended for safety)

Some of the synthetic evaluations and the dress-rehearsal Q5 used `dplyr`,
`lubridate`, etc. The system prompt now defaults to base R for exam answers,
but installing the wider set is cheap insurance in case a question explicitly
names tidyverse / dplyr / lubridate.

```r
install.packages(c("tidyverse", "lubridate"))
```

`tidyverse` is a meta-package — installing it pulls in `ggplot2`, `dplyr`,
`tidyr`, `readr`, `purrr`, `tibble`, `stringr`, `forcats`. So `tidyverse` +
`lubridate` covers every R package this course's syllabus mentions.

If you want to install one-by-one instead:
```r
install.packages(c("ggplot2", "dplyr", "tidyr", "readr", "lubridate"))
```

### Verify the install worked

In the RStudio Console:
```r
library(ggplot2); library(dplyr); library(tidyr); library(readr); library(lubridate)
cat("all good\n")
```

If any `library(...)` errors with `there is no package called 'xxx'`, run
`install.packages("xxx")` for that one.

### Built-in datasets — no install needed

These ship with base R and `library(datasets)` is auto-loaded at startup:

| Dataset | Columns | Used in |
|---|---|---|
| `iris` | Sepal.Length, Sepal.Width, Petal.Length, Petal.Width, Species | Scatter + per-species trend questions |
| `mtcars` | mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb | Faceted scatter, histograms |
| `airquality` | Ozone, Solar.R, Wind, Temp, Month, Day | Line chart, scatter + NA filter |
| `cars` | speed, dist | Simple scatter probes |
| `ToothGrowth`, `PlantGrowth`, `chickwts` | various | Alternative shapes |

You do not install these. Just type the name (`iris`, `mtcars`,
`airquality`) and R prints the data.

---

## Python — for Data Analytics Lab questions (`launch-da.bat`)

### One-line install

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn
```

That is the entire surface the Data Analytics Lab uses. `scikit-learn` is the
pip name; you import it as `sklearn`. The course only needs `MinMaxScaler`,
`StandardScaler` and `train_test_split` from it.

### CRITICAL — pre-warm the seaborn dataset cache

Every DA Lab question starts with `sns.load_dataset('titanic')` or
`sns.load_dataset('diamonds')`. **`load_dataset` downloads from GitHub the
first time it is called** and caches the CSV locally. With the exam-hall
internet off, an un-warmed machine raises `URLError` and you cannot even load
the data.

Run this ONCE, at home, on the exam laptop, while you still have internet:

```powershell
python -c "import seaborn as sns; [sns.load_dataset(n) for n in ['titanic','diamonds','tips','iris','penguins','mpg','flights']]; print('cache warmed')"
```

The CSVs land in `%LOCALAPPDATA%\seaborn\seaborn\Cache` and stay there. After
that, `sns.load_dataset` works with the network unplugged.

**Belt and braces:** this repo also ships the same CSVs in `data\`. Copy them
next to your notebook on exam day, and the fallback the AI writes into every
answer will pick them up automatically:

```python
try:
    df = sns.load_dataset('titanic')
except Exception:
    df = pd.read_csv('titanic.csv')
```

### Exam-day tip: warm it up BEFORE the clock starts

The Data Analytics system prompt is large (~8.5k tokens). llama-server must
process it once, which costs roughly a minute on a lab PC — but it then caches
that prefix, so every later question skips it. Observed: 98 s for the first
question on a cold server, 31 s and 45 s for the next two.

So on exam day: launch `launch-da.bat` as soon as you sit down, and ask it
something trivial ("hi") while the paper is still being handed out. Every real
question after that is fast. Do not launch it for the first time on question 1.

### Verify

```python
import numpy, pandas, matplotlib, seaborn, sklearn
import seaborn as sns
print(sns.load_dataset('titanic').shape)    # (891, 15)
print(sns.load_dataset('diamonds').shape)   # (53940, 10)
print("all good")
```

Then pull the network cable / turn off Wi-Fi and run it again. If it still
prints both shapes, you are genuinely offline-ready.

---

## Final checklist before exam day

Print this out. Tick each.

- [ ] `pip install numpy pandas matplotlib seaborn scipy scikit-image joypy palettable` runs without errors.
- [ ] `import numpy, pandas, matplotlib, seaborn, scipy, skimage, joypy, palettable, colorsys` in a Jupyter cell shows no error.
- [ ] `install.packages(c("tidyverse", "lubridate"))` runs in RStudio without errors.
- [ ] `library(ggplot2); library(dplyr); library(tidyr); library(readr); library(lubridate)` in RStudio shows no error.
- [ ] `pip install numpy pandas matplotlib seaborn scikit-learn` runs without errors (Data Analytics Lab).
- [ ] Seaborn cache pre-warmed: the one-liner above printed `cache warmed`.
- [ ] **Wi-Fi OFF**, then `sns.load_dataset('titanic')` and `sns.load_dataset('diamonds')` still return their shapes. This is the one that bites on exam day.
- [ ] Copied `data\*.csv` next to the notebook you will work in.
- [ ] `launch-da.bat` (= `python start.py da-python`) launches the Data Analytics mode and opens `localhost:8080`.
- [ ] `launch-python.bat` (= `python start.py dataviz-python`) launches the Python mode and opens `localhost:8080` in the browser.
- [ ] `launch-r.bat` (= `python start.py dataviz-r`) launches the R mode and opens `localhost:8080` in the browser.
- [ ] After launching, hard-refreshed the browser (Ctrl+F5) so the latest
  system prompt loads. The Settings textarea should contain the line
  "EXAM IMPORT DISCIPLINE" (Python mode) — it is served from
  `courses\<id>\prompt.md` now, not baked into the HTML.

If every box is ticked, you are ready.
