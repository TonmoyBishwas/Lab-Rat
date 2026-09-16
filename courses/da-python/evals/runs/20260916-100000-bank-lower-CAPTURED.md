```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- Starter Code (Provided by the prompt) ---
df = pd.read_csv('bank.csv', sep = ';')
features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
x = df[features].copy()

y = df['y'].map({'no':0, 'yes': 1})

# Note: The starter code uses x_train, x_test, etc. We will use these raw DataFrames
# and create the canonical scaled arrays x_train_s, x_test_s in part (a).
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# --- Part (a) ---
# Standardize the continuous feature space
scaler = StandardScaler()
# Fit on training data only (anti-leakage rule)
x_train_s = scaler.fit_transform(x_train)
# Transform both training and test data
x_test_s = scaler.transform(x_test)
print("X_train_s shape:", x_train_s.shape)
print("X_test_s shape:", x_test_s.shape)

# --- Part (b) ---
# Fit and evaluate Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(x_train_s, y_train)
y_pred_logreg = logreg.predict(x_test_s)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print("Logistic Regression accuracy:", accuracy_logreg)

# Fit and evaluate Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train_s, y_train)
y_pred_rf = rf.predict(x_test_s)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest accuracy:", accuracy_rf)

# --- Part (c) ---
# Initialize and train MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
                      solver='adam', max_iter=500, random_state=42)
mlp.fit(x_train_s, y_train)
y_pred_mlp = mlp.predict(x_test_s)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("MLPClassifier accuracy:", accuracy_mlp)
print("MLPClassifier final training loss:", mlp.loss_)
```
Notes: The starter code was used to define the dataset, features, and split the data into raw DataFrames (`x_train`, `x_test`, etc.). The scaled NumPy arrays (`x_train_s`, `x_test_s`) were created in part (a) to ensure proper anti-leakage for the modeling tasks in parts (b) and (c).

---

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

# continues the notebook — uses: mlp, X_test_s, y_test

# --- Part (a) ---
# Calculate predictions and confusion matrix
y_pred_mlp = mlp.predict(X_test_s)
cm = confusion_matrix(y_test, y_pred_mlp)

# Visualize the confusion matrix as a heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Deposit', 'Subscribed'],
            yticklabels=['No Deposit', 'Subscribed'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# --- Part (b) ---
# Print the full classification report
print(classification_report(y_test, y_pred_mlp,
                             target_names=['No Deposit', 'Subscribed']))

# --- Part (c) ---
# Plot the network learning trajectory
plt.figure(figsize=(6, 4))
plt.plot(mlp.loss_curve_, color='#c0392b', lw=1.8)
plt.title('Training Loss Curve (learning over epochs)')
plt.xlabel('epoch (iteration)')
plt.ylabel('training loss')
plt.tight_layout()
plt.show()
```
Notes: The confusion matrix shows the counts of False Positives and False Negatives by comparing the custom threshold matrix (cm_custom) against the default matrix (cm).

---

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

# continues the notebook — uses: rf, mlp, X_train_s, X_test_s, y_test, cm

# --- Part (a) ---
# Extract feature importance from RandomForestClassifier
print("--- Part (a): Feature Importance ---")
importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("Full Feature Importance:\n", importance.round(4))
print("\nTop 2 most influential features:", list(importance.index[:2]))

# Encapsulate StandardScaler and RandomForestClassifier into a Pipeline
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Report Mean and Standard Deviation of a 5-fold-stratified cross validation accuracy score
print("\n--- Part (a): Cross-Validation Score ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring='accuracy')
print("CV scores:", np.round(scores, 4))
print("Mean CV accuracy:", np.round(scores.mean(), 4))
print("Std Dev CV accuracy:", np.round(scores.std(), 4))


# --- Part (b) ---
# Apply custom threshold T = 0.65 to MLPClassifier
print("\n--- Part (b): Custom Thresholding ---")
T = 0.65
probs_class1 = mlp.predict_proba(X_test_s)[:, 1]
y_pred_custom = (probs_class1 >= T).astype(int)

# Compute and display the new confusion matrix
cm_custom = confusion_matrix(y_test, y_pred_custom)
print("Custom-threshold confusion matrix (T=0.65):\n", cm_custom)

# Compare this matrix against the default matrix (cm) from Q2(a)
# Assuming 'cm' is the default confusion matrix from the previous step
tn, fp, fn, tp = cm.ravel()
tn2, fp2, fn2, tp2 = cm_custom.ravel()
print("\nComparison of False Positives (FP):")
print(f"Default threshold (0.5) FP count: {fp}")
print(f"Custom threshold (0.65) FP count: {fp2}")
print("Conclusion: When the threshold is made stricter (higher), the count of False Positives decreases.")


# --- Part (c) ---
# Bundle StandardScaler and MLPClassifier into a single pipeline
print("\n--- Part (c): MLP Pipeline and Prediction ---")
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
                           solver='adam', max_iter=500, random_state=42))
])

# Fit the pipeline on X_train and y_train (using raw data)
mlp_pipeline.fit(X_train, y_train)
print("MLP Pipeline fitted successfully.")

# Construct the synthetic client dataframe
new_client = pd.DataFrame([{
    'age': 41, 'balance': 24143, 'day': 5, 'duration': 261,
    'campaign': 1, 'pdays': -1, 'previous': 0
}])

# Ensure the new client dataframe has the same columns and order as X_train
new_client = new_client[X_train.columns]

# Pass new_client directly into the fitted pipeline
prediction = mlp_pipeline.predict(new_client)[0]
probability = mlp_pipeline.predict_proba(new_client)[0][1]

print("Predicted class label (0=No, 1=Yes):", prediction)
print("Predicted subscription probability (Class 1):", round(probability, 4))
```
Notes: The count of False Positives decreases when the threshold is made stricter because the model requires a higher probability of class 1 before predicting it, thus reducing the number of incorrect positive predictions.