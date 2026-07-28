# Class Test 1 — the real paper (another section, 2026-07-28)

A copy of another section's Class Test 1. Same course, same dataset; the
student's own paper is expected to match its shape and difficulty with
different wording. This is the highest-value regression bank in the project —
it is not synthetic, and every failure it exposed was real.

Grade it with `eval\grade_class_test.py`, which executes the answer and checks
25 specific properties, rather than by reading.

**Key difference from the assignment:** the paper supplies its own loader,
`pd.read_csv('diamonds.csv')`, reading a file from the lab drive. After
read_csv the text columns are `str`, not `category` — the opposite of
`sns.load_dataset('diamonds')`.

Verified values (from `data/diamonds.csv`, 53940 x 10):
zero-dimension rows 20 (x:8, y:7, z:20) · table Q1 56.0 / Q3 59.0 / IQR 3.0 /
bounds 51.5–63.5 / 605 outliers · carat median 0.70 · price median 2401.0 ·
80/20 split 43152 / 10788 · price skew 1.618, log1p skew 0.116 ·
**highest correlation with price = carat, 0.9216** · mean price by cut:
Premium 4584 > Fair 4359 > Very Good 3982 > Good 3929 > Ideal 3458.

---

## Q1 — the complete paper, pasted as one message

Class Test 1: Data Preprocessing & Exploratory Data Analysis
Course: Data Analytics Lab | Time: 60 Minutes | Full Marks: 30

Instructions & Dataset Setup
- Environment: Open a fresh Jupyter Notebook.
- Network Policy: Internet access is disabled during the exam. Use the diamonds.csv dataset provided in your local lab drive.
- Reproducibility: Set random_state = 42 wherever applicable.
- Dataset Loading: Execute the following code block to load the dataset:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the local dataset file
df = pd.read_csv('diamonds.csv')

Part 1: Data Preprocessing [18 Marks]

Task 1 [5 Marks]
a. Display the shape, non-null counts, and data types of the dataset using .info(). Display summary statistics using .describe(). [2 Marks]
b. Inspect the physical dimension columns x (length), y (width), and z (depth). Identify how many rows contain an invalid dimension of 0 mm across x, y, or z. [1 Mark]
c. Replace all 0 values in x, y, and z with NaN. Impute these missing values using the median of each column grouped by cut. Verify that zero missing values remain. [2 Marks]

Task 2 [7 Marks]
a. If any numeric values in carat or price are missing, impute them using their respective column medians. [2 Marks]
b. cut and clarity are ordinal categorical variables. Convert them into numeric format using explicit integer mapping:
   - cut: {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
   - clarity: {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
   [2 Marks]
c. color is a nominal variable. Apply One-Hot Encoding using pd.get_dummies() with dtype=int. Display the first 5 rows of the updated dataset. [3 Marks]

Task 3 [6 Marks]
a. Compute Q1, Q3, and IQR for the table column. Identify the boundary thresholds:
   Lower Bound = Q1 - 1.5 x IQR    Upper Bound = Q3 + 1.5 x IQR
   Cap (clip) any outliers in table to these boundary values. [2 Marks]
b. Perform an 80/20 train/test split using train_test_split() with random_state = 42, setting price as target (y) and all other columns as predictors (X). [2 Marks]
c. Apply Standardization (Z-score scaling) to continuous features (carat, depth, table, x, y, z). Fit the scaler only on X_train and transform both X_train and X_test. Briefly state why fitting on the entire dataset beforehand causes data leakage. [2 Marks]

Part 2: Exploratory Data Analysis [12 Marks]

Task 4 [6 Marks]
a. Plot a histogram with 30 bins for price in the training set. Describe its distribution shape (skewness) in one sentence. [2 Marks]
b. Create a count plot displaying the frequency distribution of cut categories. [2 Marks]
c. Apply a Log Transformation (ln(1 + price)) to create log_price. Plot its histogram next to the original price distribution and explain how the shape changes. [2 Marks]

Task 5 [6 Marks]
a. Compute the mean price grouped by cut using .groupby() and print the result. [2 Marks]
b. Generate a box plot comparing price across different cut categories. [2 Marks]
c. Compute the correlation matrix for all numeric columns in X_train along with price using .corr(). Generate an annotated seaborn heatmap using the coolwarm palette. Identify which feature has the highest correlation with price. [2 Marks]

-- End of Question Paper --

*The four traps, all of which drew blood before the 2026-07-28 tuning: (1) the
target must be excluded from X with .drop(columns=['price']), not a hand-built
list; (2) "highest correlation with price" must not answer "price" at 1.000;
(3) Part 2 reads the target from y_train, since the split already happened;
(4) Task 5c needs X_train.assign(price=y_train) to put the target back.*
