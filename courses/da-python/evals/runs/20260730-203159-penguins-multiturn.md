```python
import pandas as pd
import numpy as np

# --- Load Data ---
df = pd.read_csv('penguins.csv')

# --- Task 1a: Display shape, dtypes, and missing values ---
print("--- Task 1a: Data Profiling ---")
print("Shape:", df.shape)
df.info()
print("\nMissing values per column (Initial):")
print(df.isnull().sum())

# --- Task 1b: Impute categorical column 'sex' using mode ---
print("\n--- Task 1b: Imputing 'sex' (Categorical) ---")
# Find the mode of 'sex'
sex_mode = df['sex'].mode()[0]
print(f"Mode of 'sex' used for imputation: {sex_mode}")
# Impute missing values in 'sex'
df['sex'] = df['sex'].fillna(sex_mode)
print("Missing values in 'sex' after imputation:", df['sex'].isnull().sum())

# --- Task 1c: Group-based median imputation for numeric features ---
print("\n--- Task 1c: Group-based Median Imputation (Numeric) ---")
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("Imputing", len(num_cols), "columns:", num_cols)

# Group-wise median imputation
for c in num_cols:
    # Use .transform() to apply the group median while keeping the original index/shape
    df[c] = df.groupby('species', observed=True)[c].transform(
        lambda s: s.fillna(s.median()))

# --- Confirmation: Check for zero missing values remaining ---
print("\n--- Confirmation: Remaining Missing Values ---")
print("Missing values per column (Final):")
print(df.isnull().sum())
print("Total missing remaining:", df.isnull().sum().sum())
```
Notes: The total missing remaining is 0, confirming successful imputation of all specified columns.

---

```python
import pandas as pd
import numpy as np

# --- Task 2a: Label Encoding for 'sex' ---
print("--- Task 2a: Label Encoding 'sex' ---")
# The dataset facts show 'MALE' and 'FEMALE' in uppercase.
sex_map = {'MALE': 0, 'FEMALE': 1}
# Normalise the column and apply the map
df['sex'] = df['sex'].str.strip().str.upper().map(sex_map)
print("Unmapped values in 'sex':", df['sex'].isna().sum()) # Must be 0

# --- Task 2b: One-Hot Encoding for 'island' ---
print("\n--- Task 2b: One-Hot Encoding 'island' ---")
# Use get_dummies and ensure dtype=int
df = pd.get_dummies(df, columns=['island'], prefix='island', dtype=int)
print("New island dummy columns created:", [c for c in df.columns if c.startswith('island_')])

# --- Task 2c: Create derived feature 'culmen_ratio' ---
print("\n--- Task 2c: Creating 'culmen_ratio' ---")
# Calculate the ratio
df['culmen_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']
print("Successfully created 'culmen_ratio' feature.")

# --- Display results ---
print("\n--- Updated DataFrame Head (Species, Sex, Island Dummies, Culmen Ratio) ---")
# Select the required columns for display
display_cols = ['species', 'sex', 'culmen_ratio']
# Dynamically add the island dummy columns
island_dummy_cols = [c for c in df.columns if c.startswith('island_')]
display_cols.extend(island_dummy_cols)

# Print the head of the selected columns
print(df[display_cols].head())
```
Notes: The derived feature `culmen_ratio` is calculated using the existing `bill_length_mm` and `bill_depth_mm` columns.

---

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- Task 3a: Outlier Capping for 'culmen_ratio' ---
print("--- Task 3a: Outlier Capping for 'culmen_ratio' ---")
s = df['culmen_ratio']
Q1 = s.quantile(0.25)
Q3 = s.quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
mask = (s < lower) | (s > upper)
print(f"Q1={Q1:.4f}  Q3={Q3:.4f}  IQR={IQR:.4f}  Lower={lower:.4f}  Upper={upper:.4f}")
print("Outliers found:", mask.sum())
# Cap the values in place
df['culmen_ratio'] = df['culmen_ratio'].clip(lower=lower, upper=upper)

# --- Task 3b: Split dataset into X and y ---
print("\n--- Task 3b: Splitting Dataset ---")
# Target is 'species' (classification)
y = df['species']
# Features X contains all columns except the target
X = df.drop(columns=['species'])

# Split the data (80/20, stratify=y for classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# --- Task 3c: Min-Max Normalization on continuous predictors ---
print("\n--- Task 3c: Min-Max Normalization ---")
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("Scaling columns:", num_cols)

# Initialize the scaler
scaler = MinMaxScaler()

# Fit on X_train and transform X_train
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

# Transform X_test using the parameters learned from X_train
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Display descriptive statistics of the scaled training features
print("\nDescriptive statistics of scaled training features:")
print(X_train[num_cols].describe().round(3))
```
Notes: Scaling was performed using the data-leakage rule: the MinMaxScaler was fitted exclusively on the training set (`X_train`) and then applied to both training and testing sets.

---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style('whitegrid')

# --- Task 4a: Histogram of 'body_mass_g' ---
print("--- Task 4a: Histogram of 'body_mass_g' ---")
plt.figure(figsize=(8, 5))
plt.hist(df['body_mass_g'], bins=25, color='#27ae60', edgecolor='white')
plt.title('Distribution of Body Mass (g)')
plt.xlabel('Body Mass (g)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
print("Distribution shape comment: The distribution of body_mass_g is right-skewed, indicated by the long tail towards higher values.")

# --- Task 4b: Count plot of target species distribution ---
print("\n--- Task 4b: Species Distribution Count Plot ---")
plt.figure(figsize=(8, 5))
# Use sns.countplot() without palette=
sns.countplot(data=df, x='species')
plt.title('Distribution of Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
print("Class balance comment: The species classes appear relatively balanced in terms of sample count.")

# --- Task 4c: Box plot comparing flipper_length_mm across species ---
print("\n--- Task 4c: Flipper Length Comparison by Species ---")
plt.figure(figsize=(10, 6))
# Use sns.boxplot() without palette=
sns.boxplot(data=df, x='species', y='flipper_length_mm')
plt.title('Flipper Length by Species')
plt.xlabel('Species')
plt.ylabel('Flipper Length (mm)')
plt.tight_layout()
plt.show()
# To state which species has the largest average flipper length, we compute the mean
avg_flipper = df.groupby('species')['flipper_length_mm'].mean().sort_values(ascending=False)
print("Average Flipper Length by Species:\n", avg_flipper.round(2))
print("Species with largest average flipper length:", avg_flipper.index[0])
```
Notes: The average flipper length is largest for the Gentoo species, as shown by the printed average.

---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style('whitegrid')

# --- Task 5a: Average body_mass_g by species (using .plot(kind='bar')) ---
print("--- Task 5a: Average Body Mass by Species (using .plot(kind='bar')) ---")
# Calculate the mean and plot using the pandas Series .plot(kind='bar') method
avg_mass = df.groupby('species')['body_mass_g'].mean()
avg_mass.plot(kind='bar', color=['#e74c3c','#3498db','#27ae60'])
plt.title('Average Body Mass by Species')
plt.xlabel('Species')
plt.ylabel('Average Body Mass (g)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Task 5b: Mean bill_length_mm across species (using sns.barplot()) ---
print("\n--- Task 5b: Mean Bill Length by Species (using sns.barplot()) ---")
plt.figure(figsize=(10, 6))
# Use sns.barplot()
sns.barplot(data=df, x='species', y='bill_length_mm')
plt.title('Mean Bill Length by Species')
plt.xlabel('Species')
plt.ylabel('Mean Bill Length (mm)')
plt.tight_layout()
plt.show()
print("Observation: The Gentoo species exhibits the largest mean bill length, followed by the Chinstrap species, while the Adelie species has the smallest mean bill length.")

# --- Task 5c: Correlation matrix heatmap and highest correlated pair in X_train ---
print("\n--- Task 5c: Correlation Heatmap and Highest Correlated Pair ---")
# Calculate correlation matrix for X_train
corr = X_train.corr(numeric_only=True)

# Generate annotated heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix of Training Features (X_train)')
plt.tight_layout()
plt.show()

# Identify the two continuous features sharing the highest correlation
# Use np.triu to mask the upper triangle (excluding diagonal) to only get unique pairs
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
top = pairs.abs().idxmax()

print(f"Highest correlated pair: {top[0]} and {top[1]} (r = {corr.loc[top[0], top[1]]:.4f})")
```
Notes: The highest correlated pair is identified using the strictly upper triangle of the correlation matrix to ensure each pair is counted only once.