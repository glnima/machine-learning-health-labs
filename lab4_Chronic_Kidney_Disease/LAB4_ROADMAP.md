# Lab 4 roadmap: chronic kidney disease classification

## What the professor asks for

1. Read and clean the corrected kidney-disease dataset.
2. Understand the supplied LLS-based missing-value imputation, which produces `x_new` from rows with at least 19 known values.
3. Create `y_new`: all 400 rows of the original data, replacing missing values with medians computed from the complete `Xtrain` rows.
4. Train a CART decision tree on complete `Xtrain`, then evaluate it on `x_new` and `y_new` with accuracy and a confusion matrix.
5. Repeat the comparison with random forests of 100 and 1000 trees. Inspect their feature importances.
6. Shuffle `y_new`, make a 50/50 train/test split with no fixed random seed, and evaluate a random forest and a CART tree on the test half only.

## Run it

From the repository root:

```zsh
source .venv/bin/activate
python lab4_Chronic_Kidney_Disease/data/Kidney_for_students_2025.py
```

Every run creates `lab4_Chronic_Kidney_Disease/data/lab4_results/`. The last 50/50 split intentionally differs on every run.

## How to read the confusion matrix

The saved table uses rows as true classes and columns as predictions:

| | Predicted not CKD | Predicted CKD |
| --- | --- | --- |
| True not CKD | TN | FP |
| True CKD | FN | TP |

For a health-screening task, false negatives (FN: a CKD patient classified as healthy) are particularly important to examine.

## Files to inspect

- `performance_summary.csv`: all requested accuracies and confusion-matrix values.
- `decision_tree_complete_training.png`: the CART tree trained on complete records.
- `random_forest_100_importance.png` and `random_forest_1000_importance.png`: feature importance rankings.
- `x_new_lls_imputed.csv` and `y_new_median_imputed.csv`: the two completed datasets.
