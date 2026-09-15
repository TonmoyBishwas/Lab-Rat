r"""Ground truth for the bank.csv Class Test 02 paper.

    python eval\ref_bank_ct2.py

Every number in the VERIFIED ANSWER KEY of
courses/da-python/evals/class_test_3_bank.md comes from running this. Nothing
in that key is recalled or estimated. Re-run it after a scikit-learn upgrade:
if a number moves, the key moves with it.

Dev box only - needs pandas + scikit-learn.
"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, 'data', 'bank.csv'), sep=';')
features = ['age','balance','day','duration','campaign','pdays','previous']
X = df[features].copy()
y = df['y'].map({'no':0,'yes':1})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("X_train", X_train.shape, "X_test", X_test.shape)
print("train dist\n", y_train.value_counts(normalize=True).round(3))

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_s, y_train)
mlp = MLPClassifier(hidden_layer_sizes=(32,16), activation='relu', solver='adam',
                    max_iter=500, random_state=42).fit(X_train_s, y_train)

print("Q1b logreg acc %.4f" % accuracy_score(y_test, logreg.predict(X_test_s)))
print("Q1b rf     acc %.4f" % accuracy_score(y_test, rf.predict(X_test_s)))
y_pred_mlp = mlp.predict(X_test_s)
print("Q1c mlp    acc %.4f  loss_ %.4f  n_iter %d" % (
    accuracy_score(y_test, y_pred_mlp), mlp.loss_, mlp.n_iter_))

cm = confusion_matrix(y_test, y_pred_mlp)
print("Q2a default cm\n", cm)
print("Q2a  TN,FP,FN,TP =", cm.ravel())
print("Q2b report\n", classification_report(y_test, y_pred_mlp,
      target_names=['No Deposit','Subscribed']))
print("Q2c loss_curve_ len", len(mlp.loss_curve_), "first %.4f last %.4f" % (
    mlp.loss_curve_[0], mlp.loss_curve_[-1]))

imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("Q3a importances\n", imp.round(4))
print("Q3a TOP 2 =", list(imp.index[:2]))

pipe_rf = Pipeline([('scaler', StandardScaler()),
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sc = cross_val_score(pipe_rf, X, y, cv=cv, scoring='accuracy')
print("Q3a CV(shuffle)  mean %.4f std %.4f" % (sc.mean(), sc.std()))
sc2 = cross_val_score(pipe_rf, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
print("Q3a CV(cv=5)     mean %.4f std %.4f" % (sc2.mean(), sc2.std()))

probs_class1 = mlp.predict_proba(X_test_s)[:, 1]
y_pred_custom = (probs_class1 >= 0.65).astype(int)
cm2 = confusion_matrix(y_test, y_pred_custom)
print("Q3b custom cm T=0.65\n", cm2)
print("Q3b FP default =", cm[0,1], " FP custom =", cm2[0,1])
print("Q3b FN default =", cm[1,0], " FN custom =", cm2[1,0])

pipe_mlp = Pipeline([('scaler', StandardScaler()),
                     ('mlp', MLPClassifier(hidden_layer_sizes=(32,16), activation='relu',
                                           solver='adam', max_iter=500, random_state=42))])
pipe_mlp.fit(X_train, y_train)
new_client = pd.DataFrame([{'age':41,'balance':2143,'day':5,'duration':261,
                            'campaign':1,'pdays':-1,'previous':0}])
print("Q3c pred", pipe_mlp.predict(new_client)[0],
      "proba %.4f" % pipe_mlp.predict_proba(new_client)[0][1])
