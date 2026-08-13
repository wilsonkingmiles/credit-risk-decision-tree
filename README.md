# Credit Risk Decision Tree

A supervised machine-learning project that classifies credit applicants as good or bad credit risk while emphasizing interpretability and the unequal cost of classification errors.

## What this project demonstrates
- Exploratory data analysis
- Mixed numeric and categorical preprocessing
- One-hot encoding with `ColumnTransformer`
- Decision-tree classification
- Hyperparameter tuning with `GridSearchCV`
- Model pruning and interpretability
- Feature-importance analysis
- Cost-sensitive evaluation

## Selected results
The tuned/pruned model improved the bad-risk recall from **0.322 to 0.644** and reduced the cost-sensitive error from **350 to 231**. The tuned model used a shallower tree with **depth 6** and **11 leaves**, making it substantially easier to interpret than the baseline tree.

## Repository structure
```text
src/credit_risk_tree.py
results/
  model_metrics.csv
  top_feature_importance.csv
  decision_tree_rules.txt
data/README.md
```

## Run locally
Install the requirements, place `credit-g.csv` in the project root, and run:

`python src/credit_risk_tree.py`

## Portfolio note
This is a cleaned academic portfolio project. It is intended to demonstrate analytical methodology and explainable classification, not to make real lending decisions.
