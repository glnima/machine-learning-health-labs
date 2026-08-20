# Lab 5 roadmap: ROC curves for COVID-19 serological tests

## What the professor asks for

1. Remove uncertain swab results (`COVID_swab_res = 1`) and use the remaining swab outcome as ground truth.
2. Min-max normalize the two marker levels only for DBSCAN, find outliers, and remove them from the later analysis.
3. For each serological test, calculate sensitivity and specificity across thresholds based on the sorted test values.
4. Plot sensitivity/specificity against threshold and the ROC curve: sensitivity versus false-positive rate.
5. Calculate AUC manually using the trapezoidal rule and confirm it with scikit-learn.
6. Select the threshold that maximizes Youden's J statistic: sensitivity minus false-positive rate.
7. Compare Test1 and Test2 using AUC and their ROC curves.

## Run it

```zsh
source .venv/bin/activate
python "lab5_Covid/lab_ROC - forstudents.py"
```

The script saves results in `lab5_Covid/lab5_results/`.

## How to interpret the result

- Higher AUC means better separation between positive and negative swabs.
- The Youden threshold balances sensitivity and specificity equally.
- For COVID screening, a threshold chosen only by Youden's J may not be ideal: missing an infected person (a false negative) can be more harmful than an additional false positive. A lower threshold may be preferable when sensitivity is the priority.

## Files to inspect

- `dbscan_outliers.png` and `dbscan_removed_outliers.csv`: outlier detection.
- `Test1_roc.png` and `Test2_roc.png`: ROC comparison and Youden-J points.
- `roc_summary.csv`: manual/scikit-learn AUC values and selected thresholds.
- `Test1_sensitivity_specificity.png` and `Test2_sensitivity_specificity.png`: threshold trade-offs.
