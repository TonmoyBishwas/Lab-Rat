# Eval run - da-python

- **Date:** 20260728-2322
- **Model:** (already-running server)
- **Bank:** courses\da-python\evals\questions.md
- **Temperature:** 0.3  |  **Max tokens:** 4096
- **Prompt:** courses/da-python/prompt.md (34067 chars)

Grading: see eval/BLINDSPOT_WORKFLOW.md. Verdict per question: PASS / FAIL / PARTIAL + failure-mode notes.

---

## Q1

### Question

Load the Titanic dataset from seaborn and keep only these eight columns:
`survived`, `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`.
Print the shape, the structure and non-null counts, the summary statistics for
the numeric columns, the data types, and the number of unique values per column.

### Answer  (142s, 679 completion tokens, 4.8 tok/s)

Assumptions:
- Dataset: titanic (loaded via sns.load_dataset).
- Columns kept: survived, pclass, sex, age, sibsp, parch, fare, embarked.
- No specific imputation or transformation is requested, so the data is profiled as is.

```python
import pandas as pd
import seaborn as sns
import numpy as np

# --- Load Dataset ---
try:
    df = sns.load_dataset('titanic')
except Exception:
    # Offline fallback
    df = pd.read_csv('titanic.csv')

# --- Select required columns ---
cols_to_keep = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
df = df[cols_to_keep]

# --- Profile the dataset ---
print("Shape:", df.shape)

# Structure and non-null counts (df.info() prints itself)
print("\n--- Data Structure and Non-Null Counts ---")
df.info()

# Non-null counts explicitly
print("\nMissing values per column:\n", df.isnull().sum())

# Summary statistics for numeric columns
print("\n--- Numeric Summary Statistics ---")
print(df.describe())

# Data types
print("\nData Types per column:\n", df.dtypes)

# Number of unique values per column
print("\nUnique values per column:\n", df.nunique())
```

Notes:
The `df.info()` method provides the structure and non-null counts simultaneously, showing the data types and memory usage. The `df.describe()` output confirms the distribution of numeric data, while `df.nunique()` confirms the cardinality of each column.

### Verdict

**PASS** - one code block, Assumptions + Notes present. Executes clean with ZERO warnings (was the point of this round). Correct shape (891, 8), correct dtypes reported as 'str' not 'object', nunique correct. Notes point at what the printed output shows without inventing values.

---

## Q7

### Question

For the Titanic dataset, print the summary statistics for the numeric columns
and separately for the categorical columns. Then show the counts of each class
in `survived`, `pclass` and `embarked`, and state the overall survival rate.

### Answer  (90s, 893 completion tokens, 10.0 tok/s)

Assumptions:
- Dataset: titanic (loaded using offline fallback).
- Numeric columns for summary: all numeric columns in the subset (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`).
- Categorical columns for summary: all string/categorical columns in the subset (`sex`, `embarked`).
- Target for counts: `survived` (0/1), `pclass` (1/2/3), `embarked` (C/Q/S).
- Split ratio: Not applicable, as this is a descriptive statistics request.

```python
import pandas as pd
import seaborn as sns

# --- Offline Dataset Loading ---
try:
    df = sns.load_dataset('titanic')
except Exception:
    df = pd.read_csv('titanic.csv')

print("--- Dataset Profile ---")
print("Shape:", df.shape)
print(df.head())

# --- Part (a): Summary Statistics ---
print("\n--- Numeric Column Summary Statistics ---")
print(df.describe())

print("\n--- Categorical Column Summary Statistics ---")
print(df.describe(include='str'))

# --- Part (b): Class Counts ---
print("\n--- Counts per Class in 'survived' ---")
print(df['survived'].value_counts())

print("\n--- Counts per Class in 'pclass' ---")
print(df['pclass'].value_counts())

print("\n--- Counts per Class in 'embarked' ---")
print(df['embarked'].value_counts())

# --- Part (c): Overall Survival Rate ---
overall_survival_rate = df['survived'].mean()
print("\n--- Overall Survival Rate ---")
print(f"Overall Survival Rate: {overall_survival_rate:.3f}")
```

Notes:
The printed summary statistics provide a quick overview of the distribution and central tendency for both numeric and categorical features. The overall survival rate is calculated by taking the mean of the binary 'survived' column, which is the standard method for calculating proportions.

### Verdict

**PASS** - the patch landed: emitted `df.describe(include='str')`, no Pandas4Warning. All numbers verified independently: survived 549/342, rate 0.384, sex top male 577, embarked top S 644, pclass 3 -> 491. (NOTE: the bank's own expected value said 646 for embarked S - that was a stale note of mine, corrected to 644 in questions.md this round. The MODEL was right and my key was wrong.)

---

## Q12

### Question

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

### Answer  (142s, 1378 completion tokens, 9.7 tok/s)

Assumptions:
- The dataset used is `diamonds` from seaborn.
- The target columns for imputation are `carat` and `price` (overall median).
- The physical dimension columns (`x`, `y`, `z`) are imputed using the median of their respective group (`cut`).
- The missing value injection uses the provided `np.random.seed(42)` preamble.

```python
import numpy as np
import pandas as pd
import seaborn as sns

# Load the diamonds dataset with offline fallback
try:
    df = sns.load_dataset('diamonds')
except Exception:
    df = pd.read_csv('diamonds.csv')

# Inject missing values using the specified code
np.random.seed(42)
mask_carat = np.random.rand(len(df)) < 0.05
mask_price = np.random.rand(len(df)) < 0.05
df.loc[mask_carat, 'carat'] = np.nan
df.loc[mask_price, 'price'] = np.nan

print("--- Part (a): Initial Profiling ---")
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values count per column:\n", df.isnull().sum())
print("\nMissing values percentage per column:\n", (df.isnull().mean() * 100).round(2))

# --- Part (b): Handling Zero Dimensions ---
print("\n--- Part (b): Handling Zero Dimensions ---")
# Count rows where x, y, or z equals 0
zero_rows = ((df[['x','y','z']] == 0).any(axis=1)).sum()
print("Rows with a zero dimension (x, y, or z):", zero_rows)

# Replace zeros with NaN
df[['x','y','z']] = df[['x','y','z']].replace(0, np.nan)

# Report updated missing counts after zero replacement
print("\nMissing values count after zero replacement:\n", df.isnull().sum())

# --- Part (c): Imputation ---
print("\n--- Part (c): Imputation ---")

# Impute carat and price with their overall medians
carat_median = df['carat'].median()
price_median = df['price'].median()
df['carat'] = df['carat'].fillna(carat_median)
df['price'] = df['price'].fillna(price_median)
print(f"Imputed carat with median: {carat_median:.2f}")
print(f"Imputed price with median: {price_median:.2f}")

# Impute x, y, z with the median of their own column grouped by cut
for c in ['x', 'y', 'z']:
    df[c] = df.groupby('cut', observed=True)[c].transform(lambda s: s.fillna(s.median()))
print("Imputed physical dimensions (x, y, z) using group median by 'cut'.")

# --- Part (d): Final Check ---
print("\n--- Part (d): Final Missing Value Check ---")
print("Total remaining missing values:", df.isnull().sum().sum())
```

Notes:
The initial profiling shows missing values in `carat` and `price` due to the injection step. The zero dimensions were successfully converted to NaN, and subsequent imputation using both overall medians and group-wise medians ensures the dataset is complete. The final check confirms that all missing values have been successfully addressed.

### Verdict

**PASS** - hardest question in the round and it held. Verified independently: 20 rows with a zero dimension, carat missing 2686 / price 2670 after injection, carat median 0.70, price median 2403.0, x/y/z NaN counts 8/7/20, 0 missing at the end. Group-median imputation by `cut` done with transform. The second ```python block in this entry is the QUESTION's own quoted injection code, not a second draft - the answer itself is one block.

---

## Q19

### Question

Load the `penguins` dataset from seaborn. Report its shape and how many values
are missing in each column, handle the missing values appropriately for each
column's type, encode the categorical columns, and then produce a correlation
heatmap of the numeric measurements. Finish with two sentences on which
measurements are most strongly related.

### Answer  (150s, 1474 completion tokens, 9.8 tok/s)

Assumptions:
- The dataset is `penguins` loaded via `sns.load_dataset`.
- The target variables for correlation are the numeric measurements: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`.
- Missing values will be handled using the median for numeric columns and the mode for categorical columns.
- Categorical columns (`sex`, `species`, `island`) will be encoded using binary mapping or one-hot encoding.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# --- Step 1: Load and Profile Data ---
try:
    df = sns.load_dataset('penguins')
except Exception:
    df = pd.read_csv('penguins.csv')

print("--- Data Profile ---")
print("Shape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# --- Step 2: Handle Missing Values ---
print("\n--- Handling Missing Values ---")
# Impute sex (categorical) with mode
sex_mode = df['sex'].mode()[0]
df['sex'] = df['sex'].fillna(sex_mode)
print("Sex imputed with mode:", sex_mode)

# Impute numeric measurements with median
numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"{col} imputed with median: {median_val:.2f}")

print("\nRemaining missing values per column:\n", df.isnull().sum())

# --- Step 3: Encode Categorical Columns ---
print("\n--- Encoding Categorical Columns ---")

# (a) Binary Encoding: sex
df['sex_encoded'] = df['sex'].map({'male': 0, 'female': 1})
print("Sex encoded successfully.")

# (b) Nominal Encoding: species (One-Hot)
df = pd.get_dummies(df, columns=['species'], prefix='species', dtype=int)
print("Species one-hot encoded successfully.")

# (c) Nominal Encoding: island (One-Hot)
df = pd.get_dummies(df, columns=['island'], prefix='island', dtype=int)
print("Island one-hot encoded successfully.")

# --- Step 4: Correlation Heatmap ---
print("\n--- Correlation Analysis ---")
# Calculate correlation matrix, ensuring only numeric columns are used
corr_matrix = df.corr(numeric_only=True)
print("Correlation matrix head:\n", corr_matrix.head())

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Heatmap of Penguin Measurements')
plt.tight_layout()
plt.show()

# Identify strongest relationships by looking at the absolute correlation values
# We check the correlation of all measurements against body_mass_g as a representative measure
body_mass_corr = corr_matrix['body_mass_g'].drop('body_mass_g').abs().sort_values(ascending=False)
print("\nAbsolute correlations with body_mass_g (sorted):\n", body_mass_corr.round(3))

# --- Interpretation ---
print("\n--- Interpretation ---")
print("The printed correlation values show that flipper_length_mm and bill_length_mm are the most strongly related measurements to body_mass_g.")
print("This suggests that larger flippers and longer bills are associated with heavier penguins.")
```

Notes:
The printed correlation values show that flipper_length_mm and bill_length_mm are the most strongly related measurements to body_mass_g.
This suggests that larger flippers and longer bills are associated with heavier penguins.

### Verdict

**PASS** - unseen dataset, no memorization to fall back on. Correct column names (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g), no invention. numeric_only=True used. Q19 explicitly ASKS for two sentences of interpretation, so naming the strongest relationship is required here rather than a Notes-discipline violation - and the claim is true: flipper 0.871 and bill_length 0.595 are the top two by magnitude against body_mass_g.
