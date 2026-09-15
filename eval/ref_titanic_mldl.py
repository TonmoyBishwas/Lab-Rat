r"""Ground truth for the titanic ML/DL generalization paper.

    python eval\ref_titanic_mldl.py

This paper exists to answer one question: is the prompt tuned to the bank.csv
Class Test, or to the SYLLABUS? Every axis that could be memorised from
bank.csv is deliberately different - dataset, models, architecture, fold count,
which MLP attribute is asked for, and above all the THRESHOLD DIRECTION
(T=0.35, below 0.5, so False Positives must RISE where the bank paper's 0.65
made them fall).

Dev box only - needs pandas + scikit-learn.
"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
raw = pd.read_csv(os.path.join(ROOT, 'data', 'titanic.csv'))[cols].copy()

df = raw.copy()
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['sex'] = df['sex'].str.strip().str.lower().map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['embarked'], prefix='emb', dtype=int)
print("Q1a shape", df.shape, "| missing", int(df.isnull().sum().sum()))
print("Q1a columns", list(df.columns))

X = df.drop(columns=['survived']); y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
print("Q1b", X_train.shape, X_test.shape)

dtree = DecisionTreeClassifier(random_state=42).fit(X_train_s, y_train)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_s, y_train)
print("Q1c dtree %.4f  knn %.4f" % (
    accuracy_score(y_test, dtree.predict(X_test_s)),
    accuracy_score(y_test, knn.predict(X_test_s))))

mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam',
                    max_iter=1000, random_state=42).fit(X_train_s, y_train)
y_pred_mlp = mlp.predict(X_test_s)
print("Q2a mlp acc %.4f  n_layers_ %d  loss_ %.4f  curve %d" % (
    accuracy_score(y_test, y_pred_mlp), mlp.n_layers_, mlp.loss_, len(mlp.loss_curve_)))
cm = confusion_matrix(y_test, y_pred_mlp)
print("Q2b cm\n", cm, "\n  TN,FP,FN,TP =", cm.ravel())
print("Q2c report\n", classification_report(y_test, y_pred_mlp,
      target_names=['Died', 'Survived']))

print("Q3a architectures:")
for name, hl in {'(8,)': (8,), '(16,)': (16,), '(16, 8)': (16, 8),
                 '(32, 16, 8)': (32, 16, 8)}.items():
    m = MLPClassifier(hidden_layer_sizes=hl, activation='relu', solver='adam',
                      max_iter=1000, random_state=42).fit(X_train_s, y_train)
    print("   %-12s %.4f" % (name, accuracy_score(y_test, m.predict(X_test_s))))

pipe_mlp = Pipeline([('scaler', StandardScaler()),
                     ('mlp', MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu',
                                           solver='adam', max_iter=1000, random_state=42))])
sc = cross_val_score(pipe_mlp, X, y, cv=StratifiedKFold(10, shuffle=True, random_state=42))
print("Q3a 10-fold CV mean %.4f std %.4f" % (sc.mean(), sc.std()))
sc2 = cross_val_score(pipe_mlp, X, y, cv=10)
print("Q3a 10-fold (plain cv=10) mean %.4f std %.4f" % (sc2.mean(), sc2.std()))

probs = mlp.predict_proba(X_test_s)[:, 1]
y_low = (probs >= 0.35).astype(int)
cm_low = confusion_matrix(y_test, y_low)
print("Q3b T=0.35 cm\n", cm_low)
print("Q3b FP %d -> %d   FN %d -> %d" % (cm[0,1], cm_low[0,1], cm[1,0], cm_low[1,0]))
assert cm_low[0,1] > cm[0,1], "a LOWER threshold must RAISE false positives"
print("   direction: FP RISES, FN FALLS  (opposite of the bank paper)")

num_f = ['age', 'sibsp', 'parch', 'fare']
cat_f = ['pclass', 'sex', 'embarked']
pre = ColumnTransformer([
    ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                      ('scale', StandardScaler())]), num_f),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                      ('encode', OneHotEncoder(handle_unknown='ignore'))]), cat_f)])
Xr = raw.drop(columns=['survived']); yr = raw['survived']
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    Xr, yr, test_size=0.2, random_state=42, stratify=yr)
clf = Pipeline([('preprocess', pre),
                ('model', MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu',
                                        solver='adam', max_iter=1000, random_state=42))])
clf.fit(Xr_tr, yr_tr)
print("Q3c full pipeline accuracy %.4f" % clf.score(Xr_te, yr_te))
new_p = pd.DataFrame([{'pclass': 1, 'sex': 'female', 'age': 28, 'sibsp': 0,
                       'parch': 0, 'fare': 80, 'embarked': 'C'}])
print("Q3c new passenger ->", clf.predict(new_p)[0],
      "P(survive) %.4f" % clf.predict_proba(new_p)[0][1])
