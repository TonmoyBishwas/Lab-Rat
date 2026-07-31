```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the local dataset file
df = pd.read_csv('penguins.csv')

# --- Task 1: Data Preprocessing ---
# --- Part 1a: Display shape, dtypes, and missing counts ---
print("--- Task 1a: Data Profiling ---")
print("Shape:", df.shape)
df.info()
print("\nMissing values per column:\n", df.isnull().sum())

# --- Part 1b: Impute sex mode ---
print("\n--- Task 1b: Imputing sex mode ---")
sex_mode = df['sex'].mode()[0]
df['sex'] = df['sex'].fillna(sex_mode)
print("Remaining missing values in sex:", df['sex'].isnull().sum())

# --- Part 1c: Group-based median imputation for numeric features ---
print("\n--- Task 1c: Group-wise Median Imputation ---")
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("Imputing", len(num_cols), "columns:", num_cols)
for c in num_cols:
    df[c] = df.groupby('species', observed=True)[c].transform(
        lambda s: s.fillna(s.median()))
print("Remaining missing values after group imputation:\n", df.isnull().sum())
print("Total missing remaining:", df.isnull().sum().sum())

# --- Task 2: Encoding and Feature Engineering ---
# --- Part 2a: Label Encoding sex ---
print("\n--- Task 2a: Label Encoding sex ---")
# Normalise case and verify mapping
print("sex values before mapping:", df['sex'].unique())
df['sex'] = df['sex'].str.strip().str.upper().map({'MALE': 0, 'FEMALE': 1})
print("Unmapped sex values:", df['sex'].isna().sum())

# --- Part 2b: One-Hot Encoding island ---
print("\n--- Task 2b: One-Hot Encoding island ---")
df = pd.get_dummies(df, columns=['island'], prefix='island', dtype=int)
print("Island dummy columns created successfully.")

# --- Part 2c: Create culmen_ratio ---
print("\n--- Task 2c: Creating culmen_ratio ---")
df['culmen_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']
print("First 5 rows of updated DataFrame:")
print(df[['species', 'sex', 'island_Biscoe', 'island_Dream', 'island_Torgersen', 'culmen_ratio']].head())

# --- Task 3: Outliers, Split, and Scaling ---
# --- Part 3a: Outlier Capping for culmen_ratio ---
print("\n--- Task 3a: Outlier Capping for culmen_ratio ---")
Q1 = df['culmen_ratio'].quantile(0.25)
Q3 = df['culmen_ratio'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
mask = (df['culmen_ratio'] < lower_bound) | (df['culmen_ratio'] > upper_bound)
print(f"Q1={Q1:.4f}  Q3={Q3:.4f}  IQR={IQR:.4f}  Lower={lower_bound:.4f}  Upper={upper_bound:.4f}")
print("Outliers found:", mask.sum())
df['culmen_ratio'] = df['culmen_ratio'].clip(lower=lower_bound, upper=upper_bound)

# --- Part 3b: Split dataset ---
print("\n--- Task 3b: Splitting Dataset ---")
y = df['species']
X = df.drop(columns=['species'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# --- Part 3c: Min-Max Normalization ---
print("\n--- Task 3c: Min-Max Scaling ---")
scaler = MinMaxScaler()
continuous_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Fit on train only
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
# Transform both
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])
print("X_train scaled feature summary (after scaling):")
print(X_train[continuous_cols].describe().round(3))

# --- Part 2: Exploratory Data Analysis ---
# --- Task 4: Univariate Analysis ---
# --- Part 4a: Histogram of body_mass_g ---
print("\n--- Task 4a: Body Mass Distribution ---")
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.hist(df['body_mass_g'], bins=25, color='#27ae60', edgecolor='white')
ax.set_title('Body Mass Distribution'); ax.set_xlabel('Body Mass (g)'); ax.set_ylabel('Count')
plt.tight_layout(); plt.show()
m, med = df['body_mass_g'].mean(), df['body_mass_g'].median()
print(f"mean={m:.1f} median={med:.1f} ->",
      "right-skewed" if m > med else "left-skewed" if m < med else "symmetric")

# --- Part 4b: Count plot of species (Balance check) ---
print("\n--- Task 4b: Species Count Plot and Balance Check ---")
sns.countplot(data=df, x='species')
counts = df['species'].value_counts()
print("\nSpecies counts:\n", counts)
ratio = counts.max() / counts.min()
print(f"max/min = {ratio:.2f} ->",
      "IMBALANCED" if ratio >= 1.5 else "roughly balanced")

# --- Part 4c: Box plots of flipper_length_mm by species (Largest average) ---
print("\n--- Task 4c: Flipper Length Comparison ---")
sns.boxplot(data=df, x='species', y='flipper_length_mm')
plt.title('Flipper Length by Species'); plt.xlabel('Species'); plt.ylabel('Flipper Length (mm)')
plt.tight_layout(); plt.show()
means = df.groupby('species')['flipper_length_mm'].mean().sort_values(ascending=False)
print("Mean Flipper Length by Species:\n", means.round(2))
print("Largest:", means.idxmax(), "| Smallest:", means.idxmin())

# --- Task 5: Bivariate Analysis ---
# --- Part 5a: Grouped bar chart of mean body_mass_g by species (kind='bar') ---
print("\n--- Task 5a: Mean Body Mass by Species (Bar Chart) ---")
df.groupby('species')['body_mass_g'].mean().plot(kind='bar')
plt.title('Average Body Mass by Species'); plt.xlabel('Species'); plt.ylabel('Mean Body Mass (g)')
plt.xticks(rotation=0); plt.tight_layout(); plt.show()

# --- Part 5b: Grouped bar chart of mean bill_length_mm by species (sns.barplot()) ---
print("\n--- Task 5b: Mean Bill Length by Species (Seaborn Barplot) ---")
sns.barplot(data=df, x='species', y='bill_length_mm')
plt.title('Mean Bill Length by Species'); plt.xlabel('Species'); plt.ylabel('Mean Bill Length (mm)')
plt.tight_layout(); plt.show()
print("Observation: The printed bar chart shows the mean bill length for each species.")

# --- Part 5c: Correlation Matrix and Highest Correlated Pair ---
print("\n--- Task 5c: Correlation Matrix and Highest Pair ---")
# Calculate correlation on the training set features
corr = X_train.corr(numeric_only=True)
print("Correlation of X_train features with themselves (partial view):\n", corr.round(3))

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix of Training Features'); plt.tight_layout(); plt.show()

# Identify the two continuous features with the highest correlation pair
# We must ensure we only check the upper triangle (k=1) and exclude the diagonal (1.0)
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
f1, f2 = pairs.abs().idxmax()
print(f"Highest correlated pair: {f1} and {f2} (r = {corr.loc[f1, f2]:.4f})")
```
The printed 'Highest correlated pair' line names the two features with the strongest linear relationship.