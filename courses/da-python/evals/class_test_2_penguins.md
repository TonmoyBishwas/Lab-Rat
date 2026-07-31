# Class Test 1 — Data Preprocessing & Exploratory Data Analysis (penguins.csv)

The REAL paper sat by the student on 2026-07-29, transcribed verbatim from the
question sheet. Course: Data Analytics Lab. 60 minutes, 30 marks.

This is the SECOND real paper in the bank. The first (`class_test_1.md`) used
diamonds.csv with a regression target; this one uses penguins.csv with a
CLASSIFICATION target, has REAL missing values, and asks for a derived feature.
Tuning against only one of them overfits — the July 2026 round drove
`class_test_1` to 24/25 and then scored roughly 15/30 on this paper, because
every defect it exposed was in a step diamonds does not have.

Known failure modes this bank exists to catch (all observed in the real sitting):

1. `.map({'Male':0,'Female':1})` on a file containing MALE/FEMALE — returns NaN
   for every row, raises nothing, destroys the column.
2. Hand-typed one-hot names `island_Boscoe` / `island_Targersen` — KeyError.
3. A four-column impute list emitted with three columns (`bill_depth_mm` lost).
4. `train_test_split` / `MinMaxScaler` used without importing them, in a
   follow-up turn — NameError.
5. `.plot(kind='bar')` substituted where the paper names `sns.barplot()`.
6. The top correlated PAIR printed twice, mirrored.
7. `palette=` without `hue=` on countplot.

---

## Q1

Instructions & Dataset Setup

- Environment: Open a fresh Jupyter Notebook.
- Network Policy: Internet access is disabled during the exam. Use the penguins.csv dataset provided in your local lab drive.
- Reproducibility: Set random_state = 42 wherever applicable.
- Dataset Loading: Execute the following code block to load the dataset:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the local dataset file
df = pd.read_csv('penguins.csv')
```

Part 1: Data Preprocessing [18 Marks]

Task 1 [5 Marks]
a. Display the shape, column data types, and count of missing values per column using .info() and .isnull().sum(). [2 Marks]
b. For missing values in categorical column sex, impute them using the mode (most frequent value) of the column. [1 Mark]
c. For numeric features (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g), perform group-based median imputation where missing values are filled using the median of their respective species group. Confirm zero missing values remain. [2 Marks]

Task 2 [7 Marks]
a. The sex column contains two values (Male and Female). Perform Label Encoding by mapping 'Male': 0 and 'Female': 1. [2 Marks]
b. island contains nominal categories (Biscoe, Dream, Torgersen). Convert it into numeric format using One-Hot Encoding (pd.get_dummies()) with dtype=int. [2 Marks]
c. Create a new derived feature called culmen_ratio calculated as: culmen_ratio = bill_length_mm / bill_depth_mm [2 Marks]
Display the first 5 rows of the updated DataFrame containing species, sex, island dummies, and culmen_ratio. [3 Marks]

Task 3 [6 Marks]
a. Calculate Q1, Q3, and IQR for the newly created culmen_ratio column. Cap any outliers using: Lower Bound = Q1 - 1.5 x IQR, Upper Bound = Q3 + 1.5 x IQR [2 Marks]
b. Split the dataset into features X (all columns except target) and target y (species) using an 80/20 train/test split with random_state = 42. [2 Marks]
c. Apply Min-Max Normalization (scaling features between 0 and 1) to continuous numerical predictors (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g). Fit the MinMaxScaler only on X_train and transform both X_train and X_test. [2 Marks]

Part 2: Exploratory Data Analysis [12 Marks]

Task 4 [6 Marks]
a. Plot a histogram with 25 bins for body_mass_g using plt.hist(). Comment on its distribution shape. [2 Marks]
b. Create a count plot using sns.countplot() displaying the distribution of target species. Identify whether the classes are balanced or imbalanced. [2 Marks]
c. Generate box plots comparing flipper_length_mm across the three species using sns.boxplot(). State which species has the largest average flipper length. [2 Marks]

Task 5 [6 Marks]
a. Compute the average body_mass_g grouped by species using .groupby() and plot a bar chart using kind='bar'. [2 Marks]
b. Plot a grouped bar chart using sns.barplot() to compare mean bill_length_mm across species. Describe what you observe. [2 Marks]
c. Compute the correlation matrix for all numeric columns in X_train. Generate an annotated heatmap using sns.heatmap() with cmap='coolwarm'. Identify which two continuous features share the highest correlation. [2 Marks]

---

# MULTI-TURN VARIANT

The same paper, task by task, as the student actually typed it under time
pressure — with their real spelling mistakes preserved, and with the dataset
mentioned ONLY in the first message. This is the harder test: it exercises the
follow-up rules (do not re-derive, always repeat imports).

## T1

i got this csv named penguins.csv that is:
species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex
Adelie,Torgersen,39.1,18.7,181,3750,MALE
Adelie,Torgersen,39.5,17.4,186,3800,FEMALE
Adelie,Torgersen,40.3,18,195,3250,FEMALE

task1: a. display the shape, column data types, and count of missing values per column using .info() and .isnull().sum(). b. for missing values in catergorical column sex, impute them using the mode of the column c. for numeric features (bill_lenght_mm, bill_depth_mm, flippter_lenght_mm, body_mass_g) per form group-basd median imputation were missing values are filled using the median of their respesctive species group. confirms zero missing vale remain.

## T2

task 2: a. the sex column contains two values (Male and Female). perform label encoding by mapping 'Male': 0 and 'Female': 1. b. island contains nominal categories (Biscoe, Dream, Torgersen). convert it into numeric format using one-hot encoding (pd.get_dummies()) with dtype=int. c. create a new derived feature called culmen_ratio calculated as culmen_ratio = bill_length_mm / bill_depth_mm. display the first 5 rows of the updated dataframe containing species, sex, island dummies, and culmen_ratio.

## T3

task 3: a. calculate Q1, Q3 and IQR for the newly created culmen_ratio column. cap any outliers using lower bound = Q1 - 1.5 x IQR, upper bound = Q3 + 1.5 x IQR. b. split the dataset into features X (all columns except target) and target y (species) using an 80/20 train/test split with random_state = 42. c. apply min-max normalization (scaling features between 0 and 1) to continuous numerical predictors (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g). fit the MinMaxScaler only on X_train and transform both X_train and X_test.

## T4

task 4: a. plot a histogram with 25 bins for body_mass_g using plt.hist(). comment on its distribution shape. b. create a count plot using sns.countplot() displaying the distribution of target species. identify whether the classes are balanced or imbalanced. c. generate box plots comparing flipper_length_mm across the three species using sns.boxplot(). state which species has the largest average flipper length.

## T5

task 5: a. compute the average body_mass_g grouped by species using .groupby() and plot a bar chart using kind='bar'. b. plot a grouped bar chart using sns.barplot() to compare mean bill_length_mm across species. describe what you observe. c. compute the correlation matrix for all numeric columns in X_train. generate an annotated heatmap using sns.heatmap() with cmap='coolwarm'. identify which two continuous features share the highest correlation.

---

# VERIFIED ANSWER KEY (computed from data/penguins.csv, not from the model)

    shape                      (344, 7)
    missing before             bill_* 2 each, flipper 2, body_mass 2, sex 11 (19 total)
    mode of sex                'MALE'
    missing after imputation   0
    sex encoding               MALE->0 (179 rows), FEMALE->1 (165 rows), 0 unmapped
    one-hot columns            island_Biscoe, island_Dream, island_Torgersen
    culmen_ratio head          2.0909, 2.2701, 2.2389, 2.1087, 1.9016
    culmen_ratio Q1/Q3/IQR     2.1594 / 3.0995 / 0.9402
    bounds                     0.7491 .. 4.5098   -> 0 outliers to cap
    split 80/20                X_train 275 x 9, X_test 69 x 9
    X columns (9)              bill_length_mm bill_depth_mm flipper_length_mm
                               body_mass_g sex island_Biscoe island_Dream
                               island_Torgersen culmen_ratio
    species counts             Adelie 152, Gentoo 124, Chinstrap 68  -> IMBALANCED
    mean flipper by species    Gentoo 217.18 > Chinstrap 195.82 > Adelie 189.95
                               -> largest: GENTOO
    mean body_mass by species  Gentoo 5075.40 > Chinstrap 3733.09 > Adelie 3700.66
    mean bill_length by species Chinstrap 48.83 > Gentoo 47.50 > Adelie 38.79
    top correlated PAIR in X_train
                               body_mass_g & flipper_length_mm   r = +0.8783
      runners-up: culmen_ratio & flipper_length_mm +0.8160
                  bill_length_mm & culmen_ratio    +0.7802
                  bill_depth_mm & culmen_ratio     -0.7796
