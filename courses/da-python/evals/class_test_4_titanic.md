# GENERALIZATION PAPER — titanic ML/DL (not a real sitting)

This paper is NOT from a real Class Test. It exists to answer one question:

    Is the prompt tuned to the bank.csv paper, or to the SYLLABUS?

The bank paper and its typo variant are the same paper typed two ways, so
56/56 on both is evidence about ONE paper. Written from the Part 3 and Part 4
handouts (the teacher's own material), this one deliberately differs on every
axis a model could memorise:

| axis                | bank CT-2                  | this paper                       |
|---------------------|----------------------------|----------------------------------|
| dataset             | bank.csv, 7 numeric cols   | titanic, text + REAL missing     |
| preprocessing       | none (starter did it)      | impute, map, one-hot — in Q1(a)  |
| classic models      | LogisticRegression, RF     | DecisionTree, KNN                |
| MLP architecture    | (32, 16), max_iter=500     | (16, 8), max_iter=1000           |
| MLP attribute asked | `.loss_`                   | `.n_layers_` AND `.loss_`        |
| cross-validation    | 5-fold                     | **10-fold**                      |
| heatmap palette     | Blues                      | Purples                          |
| architecture sweep  | not asked                  | asked (4 designs)                |
| ColumnTransformer   | must NOT be used           | **must be used**                 |
| threshold           | T=0.65, **FP falls**       | **T=0.35, FP RISES**             |

The last two rows are the sharpest tests.

  * **The threshold direction is inverted.** A model that memorised "raising
    the threshold reduces False Positives" from the bank paper will state the
    wrong direction here, where the threshold is LOWERED. Only reasoning gets
    this right.
  * **ColumnTransformer flips from banned to required.** The prompt gained a
    rule saying not to reach for it when X is already a chosen numeric list.
    Q3(c) explicitly asks for it, so this checks the gate suppresses the
    reflex without suppressing the capability.

Same three-fresh-chats flow: each `## Q<n>` block is pasted into a brand-new
empty chat.

## Q1

The Titanic dataset records 891 passengers and whether each one survived
(survived = 1) or did not (survived = 0). It mixes numeric attributes with text
columns, and two columns arrive with missing values, so it must be cleaned
before any model can read it.

Starter Code:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
df = sns.load_dataset('titanic')[cols].copy()
print("Loaded:", df.shape)
print(df.isnull().sum())
```

(a) Clean the frame so that every column is numeric and nothing is missing:
fill age with its median, fill embarked with its mode, encode sex as
male -> 0 and female -> 1, and one-hot encode embarked with the prefix 'emb'.
Print the resulting shape and confirm the total missing count is zero.
[3 Marks]

(b) Separate the target survived from the features, split 80/20 with
random_state=42 and stratification, then standardize the features with
StandardScaler fitted strictly on the training set. Print both split shapes.
[3 Marks]

(c) Train a DecisionTreeClassifier(random_state=42) and a
KNeighborsClassifier(n_neighbors=5) on the scaled training features and
display their test set accuracies. [4 Marks]

## Q2

(a) Initialize and train an MLPClassifier with two hidden layers of sizes
(16, 8), activation='relu', solver='adam', max_iter=1000 and random_state=42.
Report its test accuracy, the number of layers via the .n_layers_ attribute,
and its final training loss via .loss_. [4 Marks]

(b) Generate and visualize the Confusion Matrix for the MLPClassifier using a
Seaborn heatmap with cmap='Purples' and axis tick labels ['Died', 'Survived'].
[3 Marks]

(c) Print the full Classification Report for the MLPClassifier using
target_names=['Died', 'Survived'], and plot the network's training loss across
epochs from the .loss_curve_ attribute with clear axis labels and a title.
[3 Marks]

## Q3

Context: Only about 38% of passengers survived, so the classes are uneven. Read
this as a rescue-screening problem instead of a historical one: a passenger the
model flags as a likely survivor is one a rescue team would search for first.
Here, missing a real survivor (a False Negative) is far more costly than
searching for someone who turns out not to need it (a False Positive).

(a) Compare four network architectures — (8,), (16,), (16, 8) and (32, 16, 8) —
by training each with activation='relu', solver='adam', max_iter=1000,
random_state=42 and printing the test accuracy of each. Then encapsulate
StandardScaler and an MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu',
solver='adam', max_iter=1000, random_state=42) into a scikit-learn Pipeline and
report the Mean and Standard Deviation of a 10-fold stratified cross-validation
accuracy score. [4 Marks]

(b) Using probabilities from .predict_proba(), apply a RELAXED decision
threshold of T = 0.35 to your trained MLPClassifier on the scaled test set, so
that a passenger is predicted to survive whenever the predicted probability is
>= 0.35. Compute and display the new Confusion Matrix. Compare it against your
default matrix from Q2(b): what happened to the count of False Positives, and
what happened to False Negatives? Given the rescue-screening context above,
explain why a lower threshold may be the right choice here. [4 Marks]

(c) Build a single Pipeline that takes the RAW, uncleaned Titanic frame
(text columns and missing values included) and handles everything itself. Use a
ColumnTransformer with two branches: for the numeric columns
['age', 'sibsp', 'parch', 'fare'] a SimpleImputer(strategy='median') followed by
StandardScaler, and for the categorical columns ['pclass', 'sex', 'embarked'] a
SimpleImputer(strategy='most_frequent') followed by
OneHotEncoder(handle_unknown='ignore'). Chain that to an
MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam',
max_iter=1000, random_state=42), fit it on a raw 80/20 stratified split
(random_state=42), and report its test accuracy. Then construct the following
new passenger and pass it straight into the fitted pipeline, with no manual
preprocessing, to print the predicted class and the survival probability:

```python
new_passenger = pd.DataFrame([{
    'pclass': 1, 'sex': 'female', 'age': 28, 'sibsp': 0,
    'parch': 0, 'fare': 80, 'embarked': 'C'
}])
```

[6 Marks]

---

# VERIFIED ANSWER KEY

Computed by `eval/ref_titanic_mldl.py`. Every value below was executed. Where
the Part 3/4 handouts print the same quantity, the handout AGREES — an
independent check on the reference.

    Q1(a)   shape (891, 10), total missing 0
            columns: survived pclass sex age sibsp parch fare emb_C emb_Q emb_S
    Q1(b)   X_train (712, 9)   X_test (179, 9)
    Q1(c)   DecisionTree 0.8212   KNN(5) 0.8156
    Q2(a)   MLP (16,8) accuracy 0.7709   n_layers_ 4   loss_ 0.3058
            (handout Part 4 prints 0.7709 / 4 / 0.3058 — exact match)
    Q2(b)   cm = [[91, 19], [22, 47]]    TN 91  FP 19  FN 22  TP 47
            (handout Part 4 prints the same matrix)
    Q2(c)   Died 0.81/0.83/0.82 (110), Survived 0.71/0.68/0.70 (69), acc 0.77
    Q3(a)   (8,) 0.8045 · (16,) 0.7989 · (16,8) 0.7709 · (32,16,8) 0.8045
            (handout Part 4 prints exactly these four)
            10-fold CV on the Pipeline: shuffle+seed  mean 0.8069 std 0.0349
                                        plain cv=10   mean 0.8160 std 0.0340
            (either is acceptable — the question does not pin shuffling)
    Q3(b)   T=0.35 cm = [[85, 25], [15, 54]]
            False Positives 19 -> 25   (RISES)
            False Negatives 22 -> 15   (FALLS)
    Q3(c)   full ColumnTransformer pipeline accuracy 0.7933
            (handout Part 3 prints 0.7933 for the RF version of this pipeline)
            new_passenger -> predicted class 1, P(survive) 0.9997

THE DIRECTION IN Q3(b) IS THE WHOLE POINT. LOWERING the threshold below 0.5
means less confidence is required before predicting the positive class, so MORE
positives are predicted: False Positives RISE and False Negatives FALL. This is
the exact opposite of the bank paper's T=0.65. An answer that says false
positives fell has recited the other paper instead of reasoning about this one.
