# Lab Rat AI — Data Analytics Lab (Python) evaluation bank

Twenty questions calibrated against `da_lab/Lab1.pdf` (preprocessing, titanic),
`da_lab/Lab2.pdf` (EDA, titanic) and `da_lab/Assignment.pdf` (diamonds).

Because the real exam questions are unknown, the bank deliberately over-weights
two things the handouts cannot teach: **composite multi-part questions** (the
shape a lab exam actually takes) and **decoys** where the question's own
phrasing baits a wrong-but-plausible technique.

Every question is self-contained — the runner sends each one as a fresh
single-turn chat with no memory of the others, so nothing may refer to "the
dataset from Q3".

Difficulty legend: **EASY** / **MED** / **HARD** / **TRICK**.
A TRICK question PASSES when the model refuses the bait.

---

## Q1 — EASY · Dataset inspection and profiling

Load the Titanic dataset from seaborn and keep only these eight columns:
`survived`, `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`.
Print the shape, the structure and non-null counts, the summary statistics for
the numeric columns, the data types, and the number of unique values per column.

*Mirrors Lab1 Steps 1-3. Probes the profiling recipe, the `df.info()`-prints-itself
rule, and whether the model wraps everything in `print()`. Also probes the
offline `sns.load_dataset` fallback.*

---

## Q2 — EASY · Missing values: count, percentage, impute

Using the Titanic dataset (columns `survived`, `pclass`, `sex`, `age`, `sibsp`,
`parch`, `fare`, `embarked`), show how many values are missing in each column
and what percentage of the column that represents. Then fill the missing `age`
values with the median age and the missing `embarked` values with the most
frequent port. Confirm that no missing values remain, and print the median age
you used.

*Mirrors Lab1 Step 4. Probes median-vs-mean choice, `mode()[0]`, the
no-`inplace`/no-chained-assignment rule, and the expected values age 177 / embarked 2 / median 28.0.*

---

## Q3 — MED · Encoding categorical features

For the Titanic dataset, convert the `sex` column to numeric using label
encoding, and convert the `embarked` column using one-hot encoding with the
prefix `emb`. Print the list of columns afterwards and the first five rows.
Briefly explain why `embarked` should not simply be mapped to 0, 1 and 2.

*Mirrors Lab1 Step 5. Probes `dtype=int` on `get_dummies` and whether the model
can articulate the false-ordinal argument.*

---

## Q4 — MED · Min-Max normalization and Z-score standardization

For the Titanic dataset (with missing `age` already filled by the median),
create Min-Max normalized versions of `age` and `fare` called `age_norm` and
`fare_norm`, and Z-score standardized versions called `age_std` and `fare_std`.
Print the min and max of the normalized columns and the mean and standard
deviation of the standardized ones to confirm they worked. Explain in one or
two lines when you would choose each method.

*Mirrors Lab1 Step 6. Probes MinMaxScaler/StandardScaler usage and the
selection criteria. Note there is no train/test split in this question, so
fitting on the whole frame is correct here.*

---

## Q5 — MED · IQR outlier detection and capping

For the `fare` column of the Titanic dataset, compute Q1, Q3 and the IQR, derive
the lower and upper bounds using the 1.5 x IQR rule, and report how many values
fall outside them. Then cap the outliers instead of removing them, storing the
result in a new column, and draw side-by-side box plots of the fare before and
after capping. Print the maximum fare before and after.

*Mirrors Lab1 Step 7. Expected: Q1 7.9104, Q3 31.0, IQR 23.0896, upper 65.6344,
116 outliers, max 512.3292 -> 65.6344. Probes `.clip()` and the cap-don't-delete
convention.*

---

## Q6 — MED · Stratified train/test split

Prepare the Titanic dataset for modelling: encode `sex` numerically, one-hot
encode `embarked`, fill missing `age` with the median, then split the features
and the `survived` target into training and test sets using an 80/20 split with
`random_state=42`, preserving the class balance across both sets. Print the
shapes of all four resulting objects and the proportion of each class in the
training target.

*Mirrors Lab1 Step 8. Expected shapes (712, 9) and (179, 9); training balance
0 -> 0.617, 1 -> 0.383. Probes `stratify=y` and why it is used.*

---

## Q7 — EASY · Descriptive statistics for EDA

For the Titanic dataset, print the summary statistics for the numeric columns
and separately for the categorical columns. Then show the counts of each class
in `survived`, `pclass` and `embarked`, and state the overall survival rate.

*Mirrors Lab2 Step 2. Expected: survived 549/342, rate 0.384; sex top male
freq 577; embarked top S freq 646; pclass 3 -> 491. Probes
`describe(include='object')` and `value_counts()`.*

---

## Q8 — MED · Univariate analysis with a subplot grid

For the Titanic dataset, produce a 2x2 figure containing: a histogram of `age`,
a histogram of `fare`, a box plot of `age` grouped by `pclass`, and a count plot
of `embarked`. Title every panel and label the axes. In one line each, describe
the shape of the age and fare distributions.

*Mirrors Lab2 Steps 3-4. Probes subplot-grid handling with `ax=`, the
`hue=`+`legend=False` palette rule, and whether the model correctly calls fare
right-skewed.*

---

## Q9 — MED · Bivariate analysis: survival rates by group

For the Titanic dataset, calculate the survival rate separately by `sex`, by
`pclass` and by `embarked`, print all three tables rounded to three decimals,
and draw a bar chart for each in a single 1x3 figure. Summarise in two lines
which factor separates survivors most strongly.

*Mirrors Lab2 Step 5. Expected: female 0.742 / male 0.189; pclass 0.630 / 0.473
/ 0.242; embarked C 0.554 / Q 0.390 / S 0.339. Probes `groupby().mean()` on a
0/1 target and that the model does not multiply by 100.*

---

## Q10 — MED · Binning and cross-tabulation

For the Titanic dataset, create an `age_group` column by binning `age` into
Child (0-12), Teen (12-18), Young Adult (18-35), Adult (35-60) and Senior
(60-100). Print the survival rate for each age group. Then build a cross
tabulation of `sex` against `pclass` showing the mean survival rate in each
cell, and plot survival by `pclass` split by `sex`.

*Mirrors Lab2 Step 5. Probes `pd.cut` (5 labels need 6 edges) and
`pd.crosstab` with `values=` + `aggfunc=`. Expected crosstab: female
0.968/0.921/0.500, male 0.369/0.157/0.135.*

---

## Q11 — MED · Correlation analysis and heatmap

For the Titanic dataset, encode `sex` numerically, then compute the correlation
matrix for `survived`, `pclass`, `sex_encoded`, `age`, `sibsp`, `parch` and
`fare`. Print each feature's correlation with `survived` sorted from strongest
positive to strongest negative, and draw an annotated heatmap with a divergent
colour map centred on zero.

*Mirrors Lab2 Step 6. Expected: sex_encoded 0.543, fare 0.257, parch 0.082,
sibsp -0.035, age -0.065, pclass -0.338. Probes `numeric_only=True`,
`center=0` and `annot=True`.*

---

## Q12 — HARD · Diamonds: inspection, invalid values and imputation

Load the `diamonds` dataset from seaborn and inject missing values with exactly
this code:

```python
np.random.seed(42)
mask_carat = np.random.rand(len(df)) < 0.05
mask_price = np.random.rand(len(df)) < 0.05
df.loc[mask_carat, 'carat'] = np.nan
df.loc[mask_price, 'price'] = np.nan
```

Then: (a) report the shape, dtypes and the missing count and percentage per
column; (b) a diamond cannot have a physical dimension of zero, so count the
rows where `x`, `y` or `z` equals 0, replace those zeros with NaN and report the
updated missing counts; (c) impute `carat` and `price` with their medians, and
impute `x`, `y` and `z` with the median of their own column grouped by `cut`;
(d) confirm the dataset has no missing values left.

*Assignment Tasks 1-2. Expected: carat 2686 NaN, price 2670 NaN, 20 rows with a
zero dimension (x 8, y 7, z 20). Probes `groupby().transform()` — `.apply()`
misaligns and is a FAIL.*

---

## Q13 — HARD · Diamonds: ordinal versus nominal encoding

For the seaborn `diamonds` dataset, print the unique categories of `cut`,
`color` and `clarity`. Then encode `cut` and `clarity` as integers that preserve
their quality order from worst to best, and encode `color` using one-hot
encoding producing integer columns. Display the first five rows of the result
and explain in two lines why `cut` and `color` are treated differently.

*Assignment Task 3. The required maps are cut Fair=0..Ideal=4 and clarity
I1=0..IF=7. Probes whether the model uses an explicit dict rather than
`cat.codes` — seaborn stores these categories best-first, so `cat.codes` is
exactly reversed and is a FAIL.*

---

## Q14 — HARD · Diamonds: winsorizing, splitting and leakage-safe scaling

Using the seaborn `diamonds` dataset with `price` as the target: (a) detect
outliers in the `table` column using the 1.5 x IQR rule, report how many there
are, cap them to the bounds without deleting any rows, and show box plots before
and after; (b) split the data 80/20 with `random_state=42`; (c) standardize the
continuous features `carat`, `depth`, `table`, `x`, `y` and `z`, fitting the
scaler on the training set only and then transforming both sets; (d) explain in
one or two sentences why fitting the scaler on the whole dataset before
splitting would cause data leakage.

*Assignment Tasks 4-5. Expected: table Q1 56.0, Q3 59.0, IQR 3.0, bounds
51.5-63.5, 605 outliers; split 43152 / 10788. Probes fit-on-train-only and that
the model does NOT pass `stratify` on a continuous target.*

---

## Q15 — HARD · Diamonds: EDA, log transform and correlation

For the seaborn `diamonds` dataset: (a) draw a histogram of `price` with 30 bins
and describe its skewness; (b) create `log_price` as the natural log of one plus
price, plot it beside the original, and comment on how the shape changes; (c)
print the average price for each `cut` and draw a box plot of price across cut
levels; (d) draw a scatter plot of `carat` against `price` and say whether the
relationship looks linear; (e) compute the correlation matrix of the numeric
columns, draw an annotated `coolwarm` heatmap, and name the feature most
strongly correlated with price.

*Assignment Tasks 6-8. Expected: price strongly right-skewed, log1p roughly
symmetric, carat-price curved not linear, carat 0.922 the strongest. Probes
`np.log1p` and `numeric_only=True` — a bare `df.corr()` here raises ValueError
and is a FAIL.*

---

## Q16 — TRICK · Ordinal encoding via the category codes

The `cut` column of the seaborn `diamonds` dataset is already stored as a pandas
categorical, so encode it to integers using its built-in category codes, then
print the correlation between the encoded cut and `price`. Comment on whether
better cuts command higher prices.

*Trick: the question explicitly instructs `cat.codes`, but seaborn stores cut as
Ideal, Premium, Very Good, Good, Fair — so codes run Ideal=0 to Fair=4, the
reverse of quality order, and the correlation's SIGN flips. A PASS refuses the
instruction, uses the explicit ascending map, and explains why. Silently
following the instruction is a FAIL.*

---

## Q17 — TRICK · Scale first, then split

Prepare the seaborn `diamonds` dataset for a price-prediction model. Standardize
the continuous features `carat`, `depth`, `table`, `x`, `y` and `z` across the
dataset first so that everything is on the same scale, then split the result
into 80% training and 20% test data with `random_state=42`. Print the shapes and
the mean and standard deviation of the scaled training features.

*Trick: the stated order leaks test statistics into training. A PASS reorders to
split-then-fit-on-train-only and states why in the Notes; following the stated
order is a FAIL. This is the single concept the labs never cover and the
assignment explicitly asks about, so it is a likely exam differentiator.*

---

## Q18 — TRICK · Mean imputation, row deletion and a bare correlation

Clean the Titanic dataset for analysis: fill the missing `age` and `fare` values
with the column means, delete every row whose `fare` is an outlier by the 1.5 x
IQR rule so the data is clean, and then print `df.corr()` on the resulting
DataFrame to see which features relate to survival.

*Triple decoy, but only ONE of the three is a hard error — graded accordingly.
(1) HARD: a bare `df.corr()` on a frame still holding `sex` and `embarked`
raises "ValueError: could not convert string to float". `numeric_only=True` is
required; without it the code scores zero. (2) SOFT: fare is strongly
right-skewed so the mean is a poor statistic, but the question explicitly asks
for it — a PASS uses the mean as instructed AND flags the skew in Notes.
(3) SOFT: deleting 116 rows discards 13% of the data where the course convention
is capping — a PASS deletes as instructed, prints the row count lost, and names
capping as the alternative. Following a soft bait silently, with no Notes
caveat, is a PARTIAL; crashing on the hard one is a FAIL.*

---

## Q19 — MED · Generalization to an unseen dataset

Load the `penguins` dataset from seaborn. Report its shape and how many values
are missing in each column, handle the missing values appropriately for each
column's type, encode the categorical columns, and then produce a correlation
heatmap of the numeric measurements. Finish with two sentences on which
measurements are most strongly related.

*Not in any handout — probes generalization rather than memorization. penguins
really does have missing data: 2 in each of the four measurements and 11 in
`sex`. Probes correct column names (bill_length_mm, bill_depth_mm,
flipper_length_mm, body_mass_g), numeric->median / categorical->mode, and
`numeric_only=True`. Inventing column names is a FAIL.*

---

## Q20 — HARD · Reusable preprocessing function

Write a single function `preprocess(df)` that takes a raw Titanic DataFrame and
returns `X_train, X_test, y_train, y_test`. It must handle missing values,
encode the categorical columns, cap `fare` outliers using the IQR rule, and
perform a stratified 80/20 split with `random_state=42`. Demonstrate it on a
freshly loaded copy of the dataset and print the shapes of the four returned
objects.

*Lab1 Exercise 6. Probes whether the model can compose the whole pipeline into
one function without leaking state, keep the return order correct, and still
produce a runnable script.*

---

## How to record results

Run with `python eval\run_eval.py da-python`. Grade each `### Verdict` slot
PASS / FAIL / PARTIAL per `eval/BLINDSPOT_WORKFLOW.md`, then run
`python eval\check_answers.py <run-log>` to execute every generated code block
and catch runtime errors that reading alone would miss.
