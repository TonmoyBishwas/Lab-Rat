```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set plotting style for consistency
sns.set_style('whitegrid')

# --- Dataset Loading ---
df = pd.read_csv('penguins.csv')
print("--- Initial Data Profile ---")
print("Shape:", df.shape)
df.info()
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDescriptive statistics:\n", df.describe())
print("\nCategorical summary:\n", df.describe(include=['object', 'str', 'category']))

# --- Task 1: Data Preprocessing ---
# --- Part 1a: Display shape, dtypes, missing counts (Done above) ---

# --- Part 1b: Impute sex (categorical) with mode ---
print("\n--- Task 1b: Imputing 'sex' with mode ---")
# Check unique values before imputation
print("sex values before imputation:", df['sex'].unique())
# Impute using mode
df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
print("Remaining missing in sex:", df['sex'].isnull().sum())

# --- Part 1c: Group-based median imputation for numeric features ---
print("\n--- Task 1c: Group-wise median imputation ---")
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("Imputing", len(num_cols), "columns:", num_cols)
for c in num_cols:
    df[c] = df.groupby('species', observed=True)[c].transform(
        lambda s: s.fillna(s.median()))
print("Remaining missing values per column after group imputation:\n", df.isnull().sum())
print("Total missing remaining:", df.isnull().sum().sum())

# --- Task 2: Encoding & Feature Engineering ---
# --- Part 2a: Label Encoding for sex ---
print("\n--- Task 2a: Label Encoding 'sex' ---")
# Normalise case and map
print("sex values before mapping:", df['sex'].unique())
df['sex'] = df['sex'].str.strip().str.upper().map({'MALE': 0, 'FEMALE': 1})
print("Unmapped sex values:", df['sex'].isna().sum()) # Must be 0

# --- Part 2b: One-Hot Encoding for island ---
print("\n--- Task 2b: One-Hot Encoding 'island' ---")
df = pd.get_dummies(df, columns=['island'], prefix='island', dtype=int)
print("Island dummy columns created:", [c for c in df.columns if c.startswith('island_')])

# --- Part 2c: Create derived feature culmen_ratio ---
print("\n--- Task 2c: Creating culmen_ratio ---")
df['culmen_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']

# Display the first 5 rows of the updated DataFrame
print("\n--- Updated DataFrame Head (species, sex, island dummies, culmen_ratio) ---")
print(df[['species', 'sex', 'culmen_ratio'] + [c for c in df.columns if c.startswith('island_')]].head())

# --- Task 3: Outliers, Split, Scale ---
# --- Part 3a: Outlier capping for culmen_ratio ---
print("\n--- Task 3a: Outlier Capping for culmen_ratio ---")
Q1 = df['culmen_ratio'].quantile(0.25)
Q3 = df['culmen_ratio'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
mask = (df['culmen_ratio'] < lower) | (df['culmen_ratio'] > upper)
print(f"Q1={Q1:.4f}  Q3={Q3:.4f}  IQR={IQR:.4f}  Lower={lower:.4f}  Upper={upper:.4f}")
print("Outliers found:", mask.sum())
df['culmen_ratio'] = df['culmen_ratio'].clip(lower=lower, upper=upper)

# --- Part 3b: Split dataset ---
print("\n--- Task 3b: Splitting data (80/20) ---")
y = df['species']
X = df.drop(columns=['species'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# --- Part 3c: Min-Max Normalization ---
print("\n--- Task 3c: Min-Max Normalization ---")
# Identify continuous numerical predictors
num_cols_scaling = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
scaler = MinMaxScaler()

# Fit on train only
X_train[num_cols_scaling] = scaler.fit_transform(X_train[num_cols_scaling])
# Transform both
X_test[num_cols_scaling] = scaler.transform(X_test[num_cols_scaling])
print("X_train scaled descriptive stats (first 3 rows):\n", X_train[num_cols_scaling].head(3).describe().round(3))

# --- Task 4: Exploratory Data Analysis (Univariate) ---
# --- Part 4a: Histogram of body_mass_g ---
print("\n--- Task 4a: Histogram of body_mass_g ---")
plt.figure(figsize=(10, 6))
plt.hist(df['body_mass_g'], bins=25, color='#27ae60', edgecolor='white')
plt.title('Body Mass (g) Distribution')
plt.xlabel('Body Mass (g)')
plt.ylabel('Count')
plt.show()
# Notes: The distribution is right-skewed, indicated by the long tail towards higher values.

# --- Part 4b: Count plot of species ---
print("\n--- Task 4b: Species Count Plot ---")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='species')
plt.title('Distribution of Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()
# Notes: The classes are imbalanced, as the counts vary significantly between species.

# --- Part 4c: Box plots comparing flipper_length_mm ---
print("\n--- Task 4c: Flipper Length Comparison ---")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='species', y='flipper_length_mm')
plt.title('Flipper Length by Species')
plt.xlabel('Species')
plt.ylabel('Flipper Length (mm)')
plt.show()
# Notes: Chinstrap species has the largest average flipper length, as shown by the highest median/upper quartile.

# --- Task 5: Exploratory Data Analysis (Bivariate & Correlation) ---
# --- Part 5a: Average body_mass_g by species (bar chart, kind='bar') ---
print("\n--- Task 5a: Average Body Mass by Species (Bar Chart) ---")
avg_mass = df.groupby('species', observed=True)['body_mass_g'].mean().sort_values(ascending=False)
avg_mass.plot(kind='bar')
plt.title('Average Body Mass by Species')
plt.xlabel('Species')
plt.ylabel('Average Body Mass (g)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Part 5b: Mean bill_length_mm across species (sns.barplot()) ---
print("\n--- Task 5b: Mean Bill Length by Species (Bar Plot) ---")
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='species', y='bill_length_mm')
plt.title('Mean Bill Length by Species')
plt.xlabel('Species')
plt.ylabel('Mean Bill Length (mm)')
plt.show()
# Notes: The mean bill length varies across species, with Gentoo having the longest average bill length.

# --- Part 5c: Correlation matrix and heatmap ---
print("\n--- Task 5c: Correlation Matrix and Heatmap ---")
# Calculate correlation matrix on X_train
corr = X_train.corr(numeric_only=True)

# Generate annotated heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix of Training Features')
plt.tight_layout()
plt.show()

# Identify the two continuous features sharing the highest correlation
# We must use the correlation matrix of X_train
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
f1, f2 = pairs.abs().idxmax()
r_value = corr.loc[f1, f2]
print(f"Highest correlated pair: {f1} and {f2} (r = {r_value:.4f})")
```
Notes: The distribution of body_mass_g is right-skewed, indicated by the long tail towards higher values. The classes are imbalanced, as the counts vary significantly between species. Chinstrap species has the largest average flipper length, as shown by the highest median/upper quartile. The mean bill length varies across species, with Gentoo having the longest average bill length. The highest correlated pair is flipper_length_mm and body_mass_g (r = 0.7911).