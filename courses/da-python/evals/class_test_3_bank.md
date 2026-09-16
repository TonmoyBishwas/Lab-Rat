# Class Test 02 — DS 4892 Data Analytics Lab (bank.csv, ML + DL pipeline)

The THIRD real paper in the bank, and the first from the NEW syllabus
(Parts 3-4: Machine Learning and Deep Learning). Obtained from another section
2026-09-15. 30 marks, 1 hour.

Shape differences from CT-1 that matter:

  * The paper SUPPLIES STARTER CODE. Load, feature list, target encoding and the
    stratified split are already written and already run. Q1(a) starts at the
    scaler. Re-emitting the starter block is a mark-losing padding error.
  * `bank.csv` is SEMICOLON-DELIMITED. `pd.read_csv('bank.csv')` without
    `sep=';'` returns one column and every task dies.
  * The target is heavily imbalanced (~88.5/11.5), so accuracy is a trap and
    the paper leans on the confusion matrix, the classification report and a
    shifted decision threshold.
  * Nothing is preprocessed by hand — everything is scikit-learn estimators,
    Pipelines and ColumnTransformers.

HOW THE STUDENT SITS IT (this is what the grader must reproduce):
each `## Q<n>` block below is pasted into a **brand-new, empty chat**. Q2 and Q3
arrive with NO conversation history, no dataset description and no starter code.
The model has to infer "this continues a notebook" from the question text alone
and must not rebuild anything.

## Q1

The Bank dataset contains 4,521 phone records from a Portuguese bank to predict
whether a client subscribes to a long-term deposit (y = 1) or declines (y = 0).
The numerical attributes capture client demographics, account balances, and call
campaign interactions across widely different scales. Because only ~ 11.5% of
contacts subscribed, the target is heavily imbalanced, requiring feature
standardization and diagnostic metrics beyond simple accuracy.

Starter Code:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. Load dataset with the correct delimiter (sep=';')
df = pd.read_csv('bank.csv', sep=';')

# 2. Select numerical features for continuous pipeline processing
features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
X = df[features].copy()

# 3. Binary target encoding: 'yes' -> 1 (Subscribed), 'no' -> 0 (No Deposit)
y = df['y'].map({'no': 0, 'yes': 1})

# 4. Stratified 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data ready: X_train {X_train.shape}, X_test {X_test.shape}")
print(f"Target distribution:\n{y_train.value_counts(normalize=True).round(3)}")
```

(a) Standardize the continuous feature space using StandardScaler. Ensure the
scaler is fit strictly on X_train and then applied to transform both X_train and
X_test to avoid data leakage. [3 Marks]

(b) Fit both a baseline LogisticRegression(max_iter=1000) and a
RandomForestClassifier(n_estimators=100, random_state=42) on the scaled training
features. Compute and display their respective test set accuracies. [3 Marks]

(c) Initialize and train an MLPClassifier with two hidden layers of sizes
(32, 16), activation='relu', solver='adam', max_iter=500, and random_state=42.
Report its test accuracy and print its final convergence training loss via the
.loss_ attribute. [4 Marks]

## Q2

(a) Generate and visualize the Confusion Matrix for your trained MLPClassifier
using a Seaborn heatmap with axis tick labels ['No Deposit', 'Subscribed'].
[4 Marks]

(b) Print the full Classification Report for the MLPClassifier. [3 Marks]

(c) Plot the network's learning trajectory across epochs using the .loss_curve_
attribute. Include clear axis labels and a title. [3 Marks]

## Q3

Context: In the training partition, the positive class (Subscribed) constitutes
only ~ 11.5% of records. In telemarketing campaigns, customer calls incur fixed
operational and staffing costs; dialing non-converting leads (False Positives)
wastes limited operational budget.

(a) Extract the feature importances from your trained RandomForestClassifier and
list the top 2 most influential features. Then, encapsulate StandardScaler and
RandomForestClassifier into a scikit-learn Pipeline and report the Mean and
Standard Deviation of a 5-fold stratified cross-validation accuracy score.
[3 Marks]

(b) In default scikit-learn classifiers, .predict() assigns a sample to Class 1
whenever its predicted probability is >= 0.5. In high-cost marketing or risk
analysis, you can shift this decision threshold manually using probabilities
from .predict_proba():

```python
# 1. Extract probabilities for Class 1 (Subscribed)
probs_class1 = model.predict_proba(X_test_s)[:, 1]

# 2. Apply custom threshold T (yields 1 if >= T, else 0)
y_pred_custom = (probs_class1 >= T).astype(int)
```

- Apply the technique above to your trained MLPClassifier on X_test_s with a
  strict threshold of T = 0.65. [1 Mark]
- Compute and display the new Confusion Matrix using
  confusion_matrix(y_test, y_pred_custom). [1 Mark]
- Compare this matrix against your default matrix from Q2(a). What happened to
  the count of False Positives (clients predicted to subscribe who actually did
  not)? Why is this shift critical when marketing campaign call budgets are
  strictly limited? [2 Marks]

(c) Bundle StandardScaler and an MLPClassifier(hidden_layer_sizes=(32, 16),
activation='relu', solver='adam', max_iter=500, random_state=42) into a single
Pipeline object and fit it on X_train and y_train. Construct the following
synthetic client DataFrame:

```python
new_client = pd.DataFrame([{
    'age': 41, 'balance': 2143, 'day': 5, 'duration': 261,
    'campaign': 1, 'pdays': -1, 'previous': 0
}])
```

Pass new_client directly into your fitted pipeline to print its predicted class
label and predicted subscription probability via .predict_proba() without manual
preprocessing. [3 Marks]

---

# Rushed-typing variant (T1-T3)

The same paper as the student actually types it under time pressure: no starter
code pasted (they run it from the paper), lowercase, abbreviated, and with the
spelling mistakes seen in the July 2026 sitting. Same three-fresh-chats flow.

## T1

i have a csv bank.csv, the target is y which is yes/no and its imbalanced only
11.5% yes. features are age, balance, day, duration, campaign, pdays, previous.
i already ran the starter code so X_train X_test y_train y_test exist (80/20
stratified random_state 42).

a. standerdize the continuous features using StandardScaler, fit only on X_train
then transfrom both to avoid data lekage
b. fit a baseline LogisticRegression(max_iter=1000) and a
RandomForestClassifer(n_estimators=100, random_state=42) on the scaled traning
features, show there test accuracies
c. initilize and train a MLPClassifier with two hiden layers (32,16),
activation relu, solver adam, max_iter 500, random_state 42. report test accuray
and print the final convergance traning loss using .loss_

## T2

a. generate and visualise the confusin matrix for the trained MLPClassifier
using seaborn heatmap with tick labels ['No Deposit','Subscribed']
b. print the full clasification report for the MLPClassifer
c. plot the netwroks learning trajectry accross epochs using .loss_curve_, with
clear axis labels and a title

## T3

a. extract feature importances from the trained RandomForestClassifer and list
the top 2 most influencial featurs. then encapsulate StandardScaler and
RandomForestClassifer into a sklearn Pipeline and report mean and std of a
5 fold stratifed cross validation accuray
b. apply a custom decision threshold T=0.65 to the trained MLPClassifier on
X_test_s using predict_proba, show the new confusion matrix, and compare it
against the default matrix from Q2a — what happend to the false positives and
why does that mater when the call budget is limited
c. bundle StandardScaler and MLPClassifier(hidden_layer_sizes=(32,16),
activation relu, solver adam, max_iter 500, random_state 42) into one Pipeline,
fit on X_train and y_train, then build new_client = pd.DataFrame([{'age':41,
'balance':2143,'day':5,'duration':261,'campaign':1,'pdays':-1,'previous':0}])
and pass it directly to the pipeline to print the predicted class and the
subscription probabilty from predict_proba, no manual preprocesing

---

# VERIFIED ANSWER KEY

Computed by `eval/ref_bank_ct2.py` against `data/bank.csv` (the real UCI file,
4521 rows), scikit-learn 1.9.0. Every number below was executed, not recalled.

    starter     X_train (3616, 7)  X_test (905, 7)   train dist 0.885 / 0.115
    Q1(b)       LogisticRegression 0.8862   RandomForest 0.8729
    Q1(c)       MLP (32,16) accuracy 0.8762   .loss_ 0.2034   n_iter_ 496
                (converges before 500 — no ConvergenceWarning expected)
    Q2(a)       cm = [[768, 33], [79, 25]]     TN 768  FP 33  FN 79  TP 25
    Q2(b)       No Deposit  P 0.91 R 0.96 F1 0.93 (support 801)
                Subscribed  P 0.43 R 0.24 F1 0.31 (support 104)
                accuracy 0.88
    Q2(c)       len(loss_curve_) == 496, falls 1.0042 -> 0.2034
    Q3(a)       importances: duration .3604, balance .1753, age .1560,
                day .1317, pdays .0874, campaign .0519, previous .0372
                TOP 2 = ['duration', 'balance']
                Pipeline 5-fold stratified CV, shuffle+random_state=42:
                    mean 0.8854  std 0.0063
                plain StratifiedKFold(5) / cv=5:  mean 0.8832  std 0.0051
                (either is acceptable — the question does not pin shuffling)
    Q3(b)       T=0.65 cm = [[783, 18], [90, 14]]
                False Positives 33 -> 18   (FELL)
                False Negatives 79 -> 90   (ROSE)
    Q3(c)       new_client -> predicted class 0, P(subscribe) 0.1096

THE DIRECTION IN Q3(b) IS ARITHMETIC, NOT LUCK: raising the threshold above 0.5
demands more confidence before predicting Class 1, so fewer positives are
predicted — FP must fall and FN must rise. An answer claiming the opposite is
wrong regardless of the split.


---

## L1

<!--
THE LOWERCASE-STARTER BANK. Captured 2026-09-16 from the student typing the
other section's paper by hand, at exam speed, with their own spelling. It is
NOT a third paper - it is the SAME paper as Q1-Q3, retyped.

Its whole reason to exist is one character: the printed paper says `X = df[...]`
and a student under time pressure types `x`. Block 1 can see that; blocks 2 and
3 are fresh chats that cannot, and fall back to the canonical `X`. The first
capture ran Q1 to four exactly-correct figures and then died on
`NameError: name 'X_test_s' is not defined. Did you mean: 'x_test_s'?`

Neither the clean bank nor the typo bank could find this: both feed the
paper's printed uppercase X. Run this mode with:  python eval\grade_bank_ct2.py lower
-->

the bank dataset contains 4521 phone records from a portuguese bank to predict whether a clinet subscribes toa long term deposit (y = 1) or declines (y = 0). the numerical attributes capture client demographics account balances and call campaign interactions across widely different scales. because only equavelnt of 11.5% of contacs subcribed the target is heavily imbalanced, requiring feature standardization and diagnostic metrics beyond simple accuracy.
starter code that is given:
import numpy, pandast, matplotlib seabord sklearn etc as their default alisas

df = pd.read_csv('bank.csv', sep = ';')
features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
x =df[features].copy()

y = df['y'].map({'no':0, 'yes': 1}0

x_train,  x_test, y_train, y_test = train_test_split( x,y, test_size=0.2, random_state=42, stratify=y)



Q1)
a) standardize the conitnous feature space using standardscaler. ensure the scaler is fit strictly on x_train and then applied to transform both x_train and x_test to avoid data leakage.

b) fit both baseline logisticregression(max_iter=1000) and random forest classcifier (n_estimators=100, random_state=42) on the scaled training features. compute and display their respective test set accuancies.

c) initializie and train and MLPClassifier with two hidden layers of size (32,16) , activation = 'relu', solver = 'adam', max_itar=500, and random_state=42. report its test accuracy and print its final coverance training loss via the .loss via the .loss_attribute.

## L2

q2) 
a)generate and visualize confusion matrix for your trained MLP classifier using a seaborn heatmap with axis tick labels ['No Deposit', 'Subscribed'].
b) print the full classification report for the MLPClassfier.
c) Plot the networks learnign trajectory across epochs usign the .loss_curve_attribute. Include clear axis labels and a title.

## L3

q3) 
a)extract the feature important from your trained RandomForestClassifier and list the top 2 most influential features. Then, encapsulate StandardScaler and RanDomForestClassfifier into a scikit-learn Pipeline and report the Mean and Standard Deviationof a 5 fold-stratified cross validation accuracty score.

b) In default scikit-learn classifiers,. predi(0 assigns a simple to class1 whenever its predicted probability is >= 0.5. In high cost marketing or risk analysis you cna shift this decision threshold manually using probabilyt from .predict_proba():
#methodology: custom decision boundary
probs_class1 = model.predict_proba(x_test_s)[:, 1]
y_pred_custom = (probs_class1 >= T).astype(int)

- appply tehcnique above to your trained MLPClassififer on x_test_s with a strict threshold of T = 0.65.
-compute and display the new confusion matrix using confusino_matrix(y_test, y_pred_custom).
-compare this matrix againt your default matrix from q2(a). what will happend to the ocunt of false positives (clinets predicted to subcre the actua ly did not?) wju os tjos sjoft crotoa; wjem,arlettogm ca[aogm ca;; nidget are stricl limited?

c) bundle standardscaler and an mlp classfifier(hidden_layers_sizes=(32,16), activation= 'relu", solver = 'adam', max_iter=500, random_state=42) into a single pipeline object and fit it on x_train and y_train. construct the followign syntehci clinet datafram:
new_clinet = pd.DataFrame([{
    'age':41, 'balance': 24143, 'day':5, 'duration': 261,
    'campaign': 1, 'pdays': -1, 'previous': 0
}])
pass new_clinet directly into your fitted pipeline to print its predicted class label and predicted subscription probailty via .predict_proba(0 without manual preprocessing.
