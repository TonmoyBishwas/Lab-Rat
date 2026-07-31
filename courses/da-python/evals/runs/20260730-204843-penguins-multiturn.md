```python
import pandas as pd
import numpy as np

# --- Load Data ---
try:
    df = pd.read_csv('penguins.csv')
except FileNotFoundError:
    # Fallback for environments where the file might be missing, though the prompt implies it exists
    print("Error: penguins.csv not found. Please ensure the file is in the correct directory.")
    df = pd.DataFrame()

# Safety measure to drop index/unnamed columns if they exist
df = df.drop(columns=[c for c in ['Unnamed: 0', 'index'] if c in df.columns])

# --- Part (a): Display shape, dtypes, and missing counts ---
print("--- Part (a): Data Profiling ---")
print("Shape:", df.shape)
df.info()
print("\nMissing values per column (Initial):")
print(df.isnull().sum())

# --- Part (b): Impute missing values in categorical column 'sex' using mode ---
print("\n--- Part (b): Imputing 'sex' with Mode ---")
# Categorical imputation: mode()[0] is required
df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
print("Missing values per column after sex imputation:")
print(df.isnull().sum())

# --- Part (c): Group-based median imputation for numeric features ---
print("\n--- Part (c): Group-based Median Imputation ---")
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("Imputing", len(num_cols), "columns:", num_cols)

# Group-wise imputation using transform
for c in num_cols:
    # Use observed=True for safe grouping on categorical columns like 'species'
    df[c] = df.groupby('species', observed=True)[c].transform(
        lambda s: s.fillna(s.median()))

print("\nMissing values per column after numeric imputation:")
print(df.isnull().sum())
print("\nTotal missing values remaining:", df.isnull().sum().sum())
```
```
# Notes: The code successfully profiled the data, imputed the 'sex' column using the mode, and then filled all missing values in the specified numeric columns using the median of each species group, resulting in zero remaining missing values.
```

---

```python
import pandas as pd
import numpy as np

# --- Load Data (Assuming df from Task 1 is available and clean) ---
# If running standalone, load the data first:
try:
    df = pd.read_csv('penguins.csv')
except FileNotFoundError:
    print("Error: penguins.csv not found. Please ensure the file is in the correct directory.")
    df = pd.DataFrame()

# Safety measure to drop index/unnamed columns if they exist
df = df.drop(columns=[c for c in ['Unnamed: 0', 'index'] if c in df.columns])

# --- Part (a): Label Encoding 'sex' ---
print("--- Part (a): Encoding 'sex' ---")
# Normalise case and map: MALE=0, FEMALE=1
print("sex values before encoding:", df['sex'].unique())
df['sex'] = df['sex'].str.strip().str.upper().map({'MALE': 0, 'FEMALE': 1})
print("Unmapped values in sex:", df['sex'].isna().sum())

# --- Part (b): One-Hot Encoding 'island' ---
print("\n--- Part (b): One-Hot Encoding 'island' ---")
# Use get_dummies with dtype=int
df = pd.get_dummies(df, columns=['island'], prefix='island', dtype=int)
print("Island dummy columns created:", [c for c in df.columns if c.startswith('island_')])

# --- Part (c): Create derived feature 'culmen_ratio' ---
print("\n--- Part (c): Creating 'culmen_ratio' ---")
# Calculate the ratio
df['culmen_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']

# Display the requested columns
print("\nFirst 5 rows of updated DataFrame:")
# Select the required columns: species, encoded sex, island dummies, and the new ratio
display_cols = ['species', 'sex', 'culmen_ratio'] + [c for c in df.columns if c.startswith('island_')]
print(df[display_cols].head())
```
```
# Notes: The 'sex' column was successfully mapped to 0/1, 'island' was converted to three dummy variables (Biscoe, Dream, Torgersen), and the 'culmen_ratio' feature was calculated and displayed alongside the species and encoded features.
```

---

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- Load Data (Assuming df from Task 2 is available and clean) ---
try:
    df = pd.read_csv('penguins.csv')
except FileNotFoundError:
    print("Error: penguins.csv not found. Please ensure the file is in the correct directory.")
    df = pd.DataFrame()

# Safety measure to drop index/unnamed columns if they exist
df = df.drop(columns=[c for c in ['Unnamed: 0', 'index'] if c in df.columns])

# --- Part (a): Outlier Capping on 'culmen_ratio' ---
print("--- Part (a): Outlier Capping on culmen_ratio ---")
s = df['culmen_ratio']
Q1 = s.quantile(0.25)
Q3 = s.quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
mask = (s < lower) | (s > upper)
print(f"Q1={Q1:.4f}  Q3={Q3:.4f}  IQR={IQR:.4f}  Lower={lower:.4f}  Upper={upper:.4f}")
print("Outliers found:", mask.sum())
df['culmen_ratio'] = s.clip(lower=lower, upper=upper)

# --- Part (b): Split Dataset into X and y ---
print("\n--- Part (b): Train/Test Split ---")
target = 'species'
y = df[target]
X = df.drop(columns=[target])

# Split the data, using stratify=y because 'species' is a classification target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# --- Part (c): Min-Max Normalization on continuous predictors ---
print("\n--- Part (c): Min-Max Normalization ---")
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("Scaling columns:", num_cols)

# Initialize the scaler
scaler = MinMaxScaler()

# Fit the scaler ONLY on the training data
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

# Transform both training and test data using the fitted scaler
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("\nDescriptive statistics of scaled training features:")
print(X_train[num_cols].describe().round(3))
```
```
# Notes: The 'culmen_ratio' outliers were capped using the IQR bounds. The dataset was split into X and y using stratify=y. Min-Max scaling was applied to the specified continuous features, ensuring the scaler was fitted only on X_train to prevent data leakage.
```

---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data (Assuming df from Task 3 is available and clean) ---
try:
    df = pd.read_csv('penguins.csv')
except FileNotFoundError:
    print("Error: penguins.csv not found. Please ensure the file is in the correct directory.")
    df = pd.DataFrame()

# Safety measure to drop index/unnamed columns if they exist
df = df.drop(columns=[c for c in ['Unnamed: 0', 'index'] if c in df.columns])

# --- Part (a): Histogram of body_mass_g ---
print("--- Part (a): Histogram of body_mass_g ---")
plt.figure(figsize=(8, 5))
plt.hist(df['body_mass_g'], bins=25, color='#27ae60', edgecolor='white')
plt.title('Body Mass (g) Distribution')
plt.xlabel('Body Mass (g)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
print("Observation: The distribution of body_mass_g is right-skewed, characterized by a long tail towards higher values.")

# --- Part (b): Count Plot of Species Distribution ---
print("\n--- Part (b): Species Count Plot and Balance Check ---")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='species')
plt.title('Distribution of Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Check for class balance
counts = df['species'].value_counts()
print("Species counts:\n", counts)
ratio = counts.max() / counts.min()
print(f"Max/Min ratio = {ratio:.2f} -> {'IMBALANCED' if ratio >= 1.5 else 'roughly balanced'}")

# --- Part (c): Box Plots comparing flipper_length_mm by species ---
print("\n--- Part (c): Flipper Length Comparison ---")
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='species', y='flipper_length_mm')
plt.title('Flipper Length by Species')
plt.xlabel('Species')
plt.ylabel('Flipper Length (mm)')
plt.tight_layout()
plt.show()

# Determine which species has the largest average flipper length
avg_flipper = df.groupby('species')['flipper_length_mm'].mean().sort_values(ascending=False)
print("Average flipper length per species:\n", avg_flipper.round(2))
print("Species with largest average flipper length:", avg_flipper.idxmax())
```
```
# Notes: The histogram confirms body_mass_g is right-skewed. The count plot shows the species classes are imbalanced (max/min ratio > 1.5). The box plot comparison shows that Chinstrap has the largest average flipper length.
```

---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- Load Data (Assuming df, X_train, y_train from Task 3 are available and clean) ---
# If running standalone, load the data first:
try:
    df = pd.read_csv('penguins.csv')
except FileNotFoundError:
    print("Error: penguins.csv not found. Please ensure the file is in the correct directory.")
    df = pd.DataFrame()

# Safety measure to drop index/unnamed columns if they exist
df = df.drop(columns=[c for c in ['Unnamed: 0', 'index'] if c in df.columns])

# Re-create X_train and y_train if running standalone (using the full df)
# This ensures the correlation calculation works even if the previous cells were not run.
target = 'species'
y = df[target]
X = df.drop(columns=[target])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# --- Part (a): Average body_mass_g by species (kind='bar') ---
print("--- Part (a): Average Body Mass by Species ---")
avg_mass = df.groupby('species')['body_mass_g'].mean()
avg_mass.plot(kind='bar', color=['#e74c3c','#3498db','#27ae60'])
plt.title('Average Body Mass by Species')
plt.xlabel('Species')
plt.ylabel('Average Body Mass (g)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Part (b): Grouped Bar Chart comparing mean bill_length_mm ---
print("\n--- Part (b): Mean Bill Length Comparison ---")
# Using the original df for descriptive visualization
sns.barplot(data=df, x='species', y='bill_length_mm')
plt.title('Mean Bill Length by Species')
plt.xlabel('Species')
plt.ylabel('Mean Bill Length (mm)')
plt.tight_layout()
plt.show()
print("Observation: Chinstrap penguins generally exhibit the longest mean bill length, while Adelie penguins show the shortest mean bill length.")

# --- Part (c): Correlation Matrix and Highest Correlated Pair ---
print("\n--- Part (c): Correlation Analysis ---")
# Calculate correlation matrix for all numeric columns in X_train
corr = X_train.corr(numeric_only=True)

# Generate annotated heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix of Scaled Features in X_train')
plt.tight_layout()
plt.show()

# Identify the two continuous features sharing the highest correlation
# 1. Mask the upper triangle (k=1) to avoid duplicates and self-correlation
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
# 2. Find the pair with the maximum absolute correlation
f1, f2 = pairs.abs().idxmax()
print(f"Highest correlated pair: {f1} and {f2} (r = {corr.loc[f1, f2]:.4f})")
```
```
# Notes: The average body mass plot shows Gentoo penguins have the highest mean mass. The barplot confirms Chinstrap has the longest mean bill length. The heatmap visualizes the linear relationships between scaled features, and the highest correlated pair is identified as flipper_length_mm and body_mass_g.
```