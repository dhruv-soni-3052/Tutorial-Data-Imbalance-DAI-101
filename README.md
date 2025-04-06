import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv("creditcard.csv")

print("Dataset shape:", data.shape)
print("\nClass distribution:\n", data['Class'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=data, palette="Set2")
plt.title("Distribution of Fraud vs Non-Fraud")
plt.xticks([0,1], ['Non-Fraud (0)', 'Fraud (1)'])
plt.show()

X = data.drop(columns=['Class'])
y = data['Class']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)

under = RandomUnderSampler(random_state=7)
X_under, y_under = under.fit_resample(X_tr, y_tr)

over = RandomOverSampler(random_state=7)
X_over, y_over = over.fit_resample(X_tr, y_tr)

smt = SMOTE(random_state=7)
X_sm, y_sm = smt.fit_resample(X_tr, y_tr)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_sm, y_sm)

y_pred = clf.predict(X_te)

print("\nClassification Report on Test Set:")
print(classification_report(y_te, y_pred))

cm = confusion_matrix(y_te, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu")
plt.title("Confusion Matrix (Test Data)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

proba = clf.predict_proba(X_te)[:,1]
auc = roc_auc_score(y_te, proba)
fpr, tpr, _ = roc_curve(y_te, proba)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.2f})")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='f1', n_jobs=-1)
search.fit(X_sm, y_sm)

print("\nBest hyperparameters found:", search.best_params_)
