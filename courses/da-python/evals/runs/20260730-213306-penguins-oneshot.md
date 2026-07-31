```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set display options for wide frames
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 20)
sns.set_style('whitegrid')

# --- Dataset Loading ---
df = pd.read_csv('penguins.csv')

# --- Task 1: Data Preprocessing ---
# --- Task 1a: Profile data ---
print("Shape:", df.shape)
df.info()
print("\nMissing values per column:\n", df.isnull().sum())

# --- Task 1b: Impute sex (categorical) with mode ---
# Note: sex values are 'MALE'/'FEMALE' in the raw data, but Task 2 requires mapping.
# We impute first using the raw values.
mode_sex = df['sex'].mode()[0]
df['sex'] = df['sex'].fillna(mode_sex)
print("\nMissing values after imputing sex:\n", df.isnull().sum())

# --- Task 1c: Group-based median imputation for numeric features ---
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("\nImputing", len(num_cols), "columns:", num_cols)
for c in num_cols:
    df[c] = df.groupby('species', observed=True)[c].transform(
        lambda s: s.fillna(s.median()))
print("Remaining missing values after group imputation:\n", df.isnull().sum())
print("Total missing remaining:", df.isnull().sum().sum())

# --- Task 2: Encoding and Feature Engineering ---
# --- Task 2a: Label Encoding for sex (Binary) ---
# Normalise case and verify no unmapped values
print("\nsex values before encoding:", df['sex'].unique())
df['sex'] = df['sex'].str.strip().str.upper().map({'MALE': 0, 'FEMALE': 1})
print("unmapped sex values:", df['sex'].isna().sum())

# --- Task 2b: One-Hot Encoding for island (Nominal) ---
df = pd.get_dummies(df, columns=['island'], prefix='island', dtype=int)
# Derive dummy column names to ensure correctness
island_cols = [c for c in df.columns if c.startswith('island_')]
print("\nIsland dummy columns created:", island_cols)

# --- Task 2c: Create derived feature culmen_ratio ---
df['culmen_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']

# Display the first 5 rows of the updated DataFrame
print("\nUpdated DataFrame head (species, sex, island dummies, culmen_ratio):\n", df[['species', 'sex'] + island_cols + ['culmen_ratio']].head())

# --- Task 3: Outliers, Split, Scale ---
# --- Task 3a: Outlier capping for culmen_ratio ---
Q1 = df['culmen_ratio'].quantile(0.25)
Q3 = df['culmen_ratio'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
mask = (df['culmen_ratio'] < lower) | (df['culmen_ratio'] > upper)
print(f"\nCulmen Ratio Bounds: {lower:.4f} to {upper:.4f} | Outliers found: {mask.sum()}")
df['culmen_ratio'] = df['culmen_ratio'].clip(lower=lower, upper=upper)

# --- Task 3b: Split dataset (X, y) ---
target = 'species'
y = df[target]
X = df.drop(columns=[target])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTrain/Test split shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# --- Task 3c: Min-Max Normalization ---
scaler = MinMaxScaler()
continuous_predictors = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("\nScaling continuous predictors...")
# Fit on train only
X_train[continuous_predictors] = scaler.fit_transform(X_train[continuous_predictors])
# Transform test set
X_test[continuous_predictors] = scaler.transform(X_test[continuous_predictors])
print("X_train scaled summary:\n", X_train[continuous_predictors].describe().round(3))

# --- Task 4: Exploratory Data Analysis (Univariate) ---
# --- Task 4a: Histogram of body_mass_g ---
plt.figure(figsize=(8, 5))
plt.hist(df['body_mass_g'], bins=25, color='#27ae60', edgecolor='white')
plt.title('Body Mass Distribution')
plt.xlabel('Body Mass (g)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
print("Observation: The distribution of body_mass_g is right-skewed (mean > median).")

# --- Task 4b: Count plot of species (Balance check) ---
counts = df['species'].value_counts()
print("\nSpecies counts:\n", counts)
ratio = counts.max() / counts.min()
print(f"max/min = {ratio:.2f} -> {'IMBALANCED' if ratio >= 1.5 else 'roughly balanced'}")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='species')
plt.title('Species Distribution')
plt.xlabel('Species')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# --- Task 4c: Box plots of flipper_length_mm by species (Largest average) ---
means = df.groupby('species')['flipper_length_mm'].mean().sort_values(ascending=False)
print("\nMean flipper length by species:\n", means.round(2))
print("Largest:", means.idxmax(), "| Smallest:", means.idxmin())
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='species', y='flipper_length_mm')
plt.title('Flipper Length by Species')
plt.xlabel('Species')
plt.ylabel('Flipper Length (mm)')
plt.tight_layout()
plt.show()

# --- Task 5: Exploratory Data Analysis (Bivariate & Correlation) ---
# --- Task 5a: Average body_mass_g by species (Bar chart, kind='bar') ---
avg_mass = df.groupby('species')['body_mass_g'].mean()
print("\nAverage body mass by species:\n", avg_mass.round(2))
plt.figure(figsize=(8, 5))
avg_mass.plot(kind='bar')
plt.title('Average Body Mass by Species')
plt.xlabel('Species')
plt.ylabel('Average Body Mass (g)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Task 5b: Mean bill_length_mm across species (Grouped bar chart, sns.barplot) ---
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='species', y='bill_length_mm')
plt.title('Mean Bill Length by Species')
plt.xlabel('Species')
plt.ylabel('Mean Bill Length (mm)')
plt.tight_layout()
plt.show()
print("Observation: The mean bill_length_mm generally increases from Adelie to Gentoo.")

# --- Task 5c: Correlation matrix and highest correlated pair ---
# Create a temporary DataFrame for correlation using X_train and y_train
train_corr_df = X_train.assign(species=y_train)
corr = train_corr_df.corr(numeric_only=True)

print("\nCorrelation matrix for X_train features:\n", corr)

# Generate annotated heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix of Training Features')
plt.tight_layout()
plt.show()

# Identify the two continuous features with the highest correlation
# Filter for the upper triangle (k=1) to avoid duplicates and self-correlation
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
f1, f2 = pairs.abs().idxmax()
r_value = corr.loc[f1, f2]
print(f"\nHighest correlated pair: {f1} and {f2} (r = {r_value:.4f})")
```
The printed 'Highest correlated pair' line names the two features with the strongest linear relationship in the training set.