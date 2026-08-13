"""
Credit Risk Classification Using Decision Trees
Dataset: Statlog (German Credit Data) / credit-g.csv
Model: DecisionTreeClassifier with preprocessing, hyperparameter tuning, and evaluation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "credit-g.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load and inspect data
df = pd.read_csv(DATA_PATH)

eda_summary = {
    "shape": df.shape,
    "columns": df.columns.tolist(),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "missing_values_total": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "class_distribution": df["class"].value_counts().to_dict(),
    "class_percentages": (df["class"].value_counts(normalize=True) * 100).round(2).to_dict(),
}

numeric_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
categorical_cols = [c for c in df.select_dtypes(include=["object"]).columns.tolist() if c != "class"]

numeric_summary = df[numeric_cols].describe().round(2)
categorical_summary = pd.DataFrame({
    "variable": categorical_cols,
    "unique_categories": [df[c].nunique() for c in categorical_cols],
    "most_common_value": [df[c].mode()[0] for c in categorical_cols],
    "most_common_count": [int(df[c].value_counts().iloc[0]) for c in categorical_cols],
})

numeric_summary.to_csv(os.path.join(OUTPUT_DIR, "numeric_summary.csv"))
categorical_summary.to_csv(os.path.join(OUTPUT_DIR, "categorical_summary.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "eda_summary.json"), "w") as f:
    json.dump(eda_summary, f, indent=4)

# 2. Exploratory visualizations
fig, ax = plt.subplots(figsize=(7, 4.5))
counts = df["class"].value_counts().reindex(["good", "bad"])
ax.bar(counts.index, counts.values)
ax.set_title("Target Class Distribution: Credit Risk")
ax.set_xlabel("Credit risk class")
ax.set_ylabel("Number of applicants")
for i, value in enumerate(counts.values):
    ax.text(i, value + 10, str(value), ha="center")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=220)
plt.close(fig)

for col in ["duration", "credit_amount", "age"]:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df[col], bins=25, edgecolor="black")
    ax.set_title(f"Distribution of {col.replace('_', ' ').title()}")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"hist_{col}.png"), dpi=220)
    plt.close(fig)

for col in ["duration", "credit_amount", "age"]:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = [df.loc[df["class"] == label, col] for label in ["good", "bad"]]
    ax.boxplot(groups, tick_labels=["good", "bad"], showfliers=True)
    ax.set_title(f"{col.replace('_', ' ').title()} by Credit Risk Class")
    ax.set_xlabel("Credit risk class")
    ax.set_ylabel(col.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"box_{col}_by_class.png"), dpi=220)
    plt.close(fig)

for col in ["checking_status", "credit_history", "savings_status", "employment", "housing"]:
    ct = pd.crosstab(df[col], df["class"], normalize="index")[["good", "bad"]] * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ct.plot(kind="bar", ax=ax)
    ax.set_title(f"Credit Risk Share by {col.replace('_', ' ').title()}")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Percent within category")
    ax.legend(title="Class")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"bar_{col}_by_class.png"), dpi=220)
    plt.close(fig)

# 3. Preprocess and split data
X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# 4. Baseline decision tree
baseline_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]
)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_proba = baseline_model.predict_proba(X_test)[:, list(baseline_model.named_steps["tree"].classes_).index("bad")]

# 5. Hyperparameter tuning
param_grid = {
    "tree__criterion": ["gini", "entropy"],
    "tree__max_depth": [3, 4, 5, 6],
    "tree__min_samples_leaf": [5, 15],
    "tree__class_weight": [None, "balanced"],
    "tree__ccp_alpha": [0.0, 0.005],
}

tuned_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("tree", DecisionTreeClassifier(random_state=RANDOM_STATE, min_samples_split=10)),
    ]
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
grid_search = GridSearchCV(
    estimator=tuned_pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=1,
    return_train_score=True,
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
tuned_pred = best_model.predict(X_test)
tuned_proba = best_model.predict_proba(X_test)[:, list(best_model.named_steps["tree"].classes_).index("bad")]

# 6. Evaluation
def model_metrics(name, model, y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=["good", "bad"])
    false_bad_as_good = int(cm[1, 0])
    false_good_as_bad = int(cm[0, 1])
    total_cost = false_good_as_bad * 1 + false_bad_as_good * 5
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_bad": precision_score(y_true, y_pred, pos_label="bad", zero_division=0),
        "recall_bad": recall_score(y_true, y_pred, pos_label="bad", zero_division=0),
        "f1_bad": f1_score(y_true, y_pred, pos_label="bad", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc_bad": roc_auc_score((y_true == "bad").astype(int), y_proba),
        "false_good_as_bad": false_good_as_bad,
        "false_bad_as_good": false_bad_as_good,
        "cost_sensitive_error": total_cost,
        "tree_depth": model.named_steps["tree"].get_depth(),
        "number_of_leaves": model.named_steps["tree"].get_n_leaves(),
    }

metrics = pd.DataFrame([
    model_metrics("Baseline unpruned tree", baseline_model, y_test, baseline_pred, baseline_proba),
    model_metrics("Tuned/pruned tree", best_model, y_test, tuned_pred, tuned_proba),
])

metrics_rounded = metrics.copy()
for col in ["accuracy", "precision_bad", "recall_bad", "f1_bad", "macro_f1", "roc_auc_bad"]:
    metrics_rounded[col] = metrics_rounded[col].round(3)
metrics_rounded.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)

baseline_cm = confusion_matrix(y_test, baseline_pred, labels=["good", "bad"])
tuned_cm = confusion_matrix(y_test, tuned_pred, labels=["good", "bad"])

pd.DataFrame(baseline_cm, index=["Actual good", "Actual bad"], columns=["Predicted good", "Predicted bad"]).to_csv(
    os.path.join(OUTPUT_DIR, "baseline_confusion_matrix.csv")
)
pd.DataFrame(tuned_cm, index=["Actual good", "Actual bad"], columns=["Predicted good", "Predicted bad"]).to_csv(
    os.path.join(OUTPUT_DIR, "tuned_confusion_matrix.csv")
)

with open(os.path.join(OUTPUT_DIR, "classification_reports.txt"), "w") as f:
    f.write("Baseline unpruned tree classification report\n")
    f.write(classification_report(y_test, baseline_pred, digits=3))
    f.write("\n\nTuned/pruned tree classification report\n")
    f.write(classification_report(y_test, tuned_pred, digits=3))

feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
feature_importance = pd.Series(
    best_model.named_steps["tree"].feature_importances_, index=feature_names
).sort_values(ascending=False)
feature_importance.head(20).to_csv(os.path.join(OUTPUT_DIR, "top_feature_importance.csv"), header=["importance"])

rules = export_text(best_model.named_steps["tree"], feature_names=list(feature_names), max_depth=6)
with open(os.path.join(OUTPUT_DIR, "decision_tree_rules.txt"), "w") as f:
    f.write(rules)

with open(os.path.join(OUTPUT_DIR, "grid_search_results.txt"), "w") as f:
    f.write("Best parameters:\n")
    f.write(str(grid_search.best_params_))
    f.write("\n\nBest cross-validation macro F1:\n")
    f.write(str(grid_search.best_score_))

# 7. Model visualizations
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=tuned_cm, display_labels=["good", "bad"])
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title("Tuned Decision Tree Confusion Matrix")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "tuned_confusion_matrix.png"), dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 5))
RocCurveDisplay.from_predictions((y_test == "bad").astype(int), tuned_proba, ax=ax)
ax.set_title("ROC Curve for Bad Credit Risk Classification")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5.5))
top_imp = feature_importance.head(10).sort_values()
ax.barh([x.replace("cat__", "").replace("num__", "") for x in top_imp.index], top_imp.values)
ax.set_title("Top 10 Decision Tree Feature Importances")
ax.set_xlabel("Importance")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(22, 12))
plot_tree(
    best_model.named_steps["tree"],
    feature_names=[name.replace("cat__", "").replace("num__", "") for name in feature_names],
    class_names=list(best_model.named_steps["tree"].classes_),
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
)
ax.set_title("Tuned/Pruned Decision Tree for German Credit Risk")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "decision_tree_plot.png"), dpi=220)
plt.close(fig)

print("Analysis complete.")
print(metrics_rounded.to_string(index=False))
print("Best parameters:", grid_search.best_params_)
print("Outputs saved to", OUTPUT_DIR)
