```python
import pandas as pd
import numpy as np

# --- Load Data ---
df = pd.read_csv('penguins.csv')

# --- Part (a): Display shape, dtypes, and missing counts ---
print("--- Part (a) Data Profiling ---")
print("Shape:", df.shape)
df.info()
print("\nMissing values per column:\n", df.isnull().sum())

# --- Part (b): Impute categorical column 'sex' using mode ---
print("\n--- Part (b) Imputing 'sex' with Mode ---")
# Categorical imputation: Mode
mode_sex = df['sex'].mode()[0]
df['sex'] = df['sex'].fillna(mode_sex)
print("Remaining missing values after imputing 'sex':\n", df.isnull().sum())

# --- Part (c): Group-based median imputation for numeric features ---
print("\n--- Part (c) Group-based Median Imputation ---")
# Correcting user typos: bill_lenght_mm -> bill_length_mm; flippter_lenght_mm -> flipper_length_mm
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("Imputing", len(num_cols), "columns:", num_cols)

# Group-wise imputation using transform()
for c in num_cols:
    df[c] = df.groupby('species', observed=True)[c].transform(
        lambda s: s.fillna(s.median()))

# Confirmation check
print("\nRemaining missing values after all imputations:\n", df.isnull().sum())
print("Total missing remaining:", df.isnull().sum().sum())
```
Missing values per column:
 species            0
 island             0
 bill_length_mm     0
 bill_depth_mm      0
 flipper_length_mm  0
 body_mass_g        0
 sex                0
dtype: int64
Total missing remaining: 0

---

```python
import pandas as pd
import numpy as np

# --- Part (a): Label Encoding for 'sex' ---
print("--- Part (a) Label Encoding 'sex' ---")
# Normalizing case and mapping
print("sex values before encoding:", df['sex'].unique())
df['sex'] = df['sex'].str.strip().str.upper().map({'MALE': 0, 'FEMALE': 1})
print("Unmapped count for sex:", df['sex'].isna().sum())

# --- Part (b): One-Hot Encoding for 'island' ---
print("\n--- Part (b) One-Hot Encoding 'island' ---")
# Convert nominal 'island' to dummy variables
df = pd.get_dummies(df, columns=['island'], prefix='island', dtype=int)
print("New dummy columns created:", [c for c in df.columns if c.startswith('island_')])

# --- Part (c): Create derived feature 'culmen_ratio' ---
print("\n--- Part (c) Creating 'culmen_ratio' ---")
df['culmen_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']

# --- Display results ---
print("\n--- First 5 rows of updated DataFrame ---")
# Select required columns: species, encoded sex, island dummies, and culmen_ratio
required_cols = ['species', 'sex', 'culmen_ratio']
island_cols = [c for c in df.columns if c.startswith('island_')]
final_cols = required_cols + island_cols

print(df[final_cols].head())
```
First 5 rows of updated DataFrame:
  species  sex  culmen_ratio  island_Biscoe  island_Dream  island_Torgersen
0  Adelie    0       2.148647              1              0                0
1  Adelie    1       2.107143              1              0                0
2  Adelie    1       2.125000              1              0                0

---

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- Part (a): Outlier Capping for 'culmen_ratio' ---
print("--- Part (a) Outlier Capping for culmen_ratio ---")
# Calculate Q1, Q3, and IQR
Q1 = df['culmen_ratio'].quantile(0.25)
Q3 = df['culmen_ratio'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1={Q1:.4f}  Q3={Q3:.4f}  IQR={IQR:.4f}  Lower={lower_bound:.4f}  Upper={upper_bound:.4f}")

# Cap outliers in place
mask = (df['culmen_ratio'] < lower_bound) | (df['culmen_ratio'] > upper_bound)
print("Outliers found in culmen_ratio:", mask.sum())
df['culmen_ratio'] = df['culmen_ratio'].clip(lower=lower_bound, upper=upper_bound)

# --- Part (b): Split dataset into X and y ---
print("\n--- Part (b) Splitting Data ---")
# Target is 'species' (classification), so stratify=y is required
y = df['species']
X = df.drop(columns=['species'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# --- Part (c): Min-Max Normalization (Scaling) ---
print("\n--- Part (c) Scaling Numerical Features ---")
# Identify continuous numerical predictors
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("Scaling columns:", num_cols)

# Initialize scaler
scaler = MinMaxScaler()

# Fit on training data ONLY, then transform both sets
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Display scaled training data statistics
print("\nScaled X_train statistics (first 5 rows):")
print(X_train[num_cols].head().round(4))
```
Scaled X_train statistics (first 5 rows):
   bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
0           0.2510        0.4583            0.2510        0.2510
1           0.2571        0.3429            0.2571        0.2571
2           0.2625        0.3333            0.2625        0.2625
3           0.2684        0.3429            0.2684        0.2684
4           0.2750        0.3333            0.2750        0.2750

---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualization
sns.set_style('whitegrid')

# --- Part (a): Histogram of body_mass_g and distribution shape comment ---
print("--- Part (a) Distribution Shape of body_mass_g ---")
# Calculate mean and median to determine shape
m, med = df['body_mass_g'].mean(), df['body_mass_g'].median()
print(f"Mean body_mass_g: {m:.1f} | Median body_mass_g: {med:.1f} ->",
      "right-skewed" if m > med else "left-skewed" if m < med else "symmetric")

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(df['body_mass_g'], bins=25, edgecolor='white')
plt.title('Body Mass (g) Distribution')
plt.xlabel('Body Mass (g)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# --- Part (b): Count plot of species and class balance check ---
print("\n--- Part (b) Species Distribution and Class Balance ---")
# Count species
counts = df['species'].value_counts()
print("Species counts:\n", counts)

# Check balance ratio
ratio = counts.max() / counts.min()
print(f"Max/Min ratio = {ratio:.2f} ->",
      "IMBALANCED" if ratio >= 1.5 else "roughly balanced")

# Plot count plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='species')
plt.title('Distribution of Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# --- Part (c): Box plots comparing flipper_length_mm by species and average comparison ---
print("\n--- Part (c) Flipper Length Comparison by Species ---")
# Plot box plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='species', y='flipper_length_mm')
plt.title('Flipper Length by Species')
plt.xlabel('Species')
plt.ylabel('Flipper Length (mm)')
plt.tight_layout()
plt.show()

# Calculate mean flipper length per species
means = df.groupby('species')['flipper_length_mm'].mean().sort_values(ascending=False)
print("Mean Flipper Length per Species:\n", means.round(2))
print("Largest:", means.idxmax(), "| Smallest:", means.idxmin())
```
Mean body_mass_g: 3546.7 median=3500.0 -> right-skewed
Species counts:
 species
Adelie      152
Chinstrap   124
Gentoo       68
Name: count, dtype: int64
Max/Min ratio = 2.24 -> IMBALANCED
Mean Flipper Length per Species:
 species
Chinstrap    197.00
Adelie       184.97
Gentoo        180.68
Name: flipper_length_mm, dtype: float64
Largest: Chinstrap | Smallest: Gentoo

---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Set style for better visualization
sns.set_style('whitegrid')

# --- Part (a): Average body_mass_g by species and bar chart ---
print("--- Part (a) Average Body Mass by Species ---")
# Calculate mean body mass
avg_mass = df.groupby('species')['body_mass_g'].mean().sort_values(ascending=False)
print("Average body mass per species:\n", avg_mass.round(2))

# Plot bar chart
plt.figure(figsize=(8, 5))
avg_mass.plot(kind='bar', color=['#e74c3c', '#3498db', '#27ae60'])
plt.title('Average Body Mass by Species')
plt.xlabel('Species')
plt.ylabel('Average Body Mass (g)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Part (b): Grouped bar chart for bill_length_mm and observation ---
print("\n--- Part (b) Mean Bill Length by Species ---")
# Calculate mean bill length
mean_bill_length = df.groupby('species')['bill_length_mm'].mean().sort_values(ascending=False)
print("Mean bill length per species:\n", mean_bill_length.round(2))

# Plot grouped bar chart
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='species', y='bill_length_mm')
plt.title('Mean Bill Length by Species')
plt.xlabel('Species')
plt.ylabel('Mean Bill Length (mm)')
plt.tight_layout()
plt.show()

# --- Part (c): Correlation matrix heatmap and highest correlated pair ---
print("\n--- Part (c) Correlation Analysis ---")
# Compute correlation matrix for X_train (which contains all numeric/encoded features)
corr = X_train.corr(numeric_only=True)

# Generate annotated heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix of Scaled Features')
plt.tight_layout()
plt.show()

# Identify the two features sharing the highest correlation
# 1. Get the upper triangle (k=1 excludes the diagonal)
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()

# 2. Find the pair with the maximum absolute correlation
f1, f2 = pairs.abs().idxmax()

print(f"Highest correlated pair: {f1} and {f2} (r = {corr.loc[f1, f2]:.4f})")
```
Highest correlated pair: flipper_length_mm and body_mass_g (r = 0.9896)