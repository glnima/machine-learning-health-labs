# Lab 2 roadmap: K-nearest-neighbour LLS regression

## What the professor asks for

1. Reuse the Parkinson dataset and seed `358768`.
2. Use the 17 regressors listed in the Lab 2 PDF; `total_UPDRS` is the target.
3. Shuffle and split the data into 40% true training, 20% validation, and 40% test data.
4. Calculate mean and standard deviation only from the true-training subset, then normalize all subsets with those values.
5. Manually find the K closest training points for every validation/test point. Fit a local LLS model with ridge term `1e-8 * I` for each point.
6. Select K using the smallest validation MSE, then compare KNN-LLS with standard LLS on exactly the same 40% test set.

## Run the script

From the repository root:

```zsh
source .venv/bin/activate
python lab2_Parkinsons/lab2_knn_lls.py
```

The script reads the shared CSV from `lab1_Parkinsons/` and saves all figures and CSV tables in `lab2_Parkinsons/lab2_results/`.

## What to inspect

- `validation_mse_vs_k.png`: identify `Kopt`, the K with the minimum validation MSE.
- `performance_summary.csv`: compare test MSE, R-squared, and correlation for KNN-LLS and standard LLS.
- `*_test_hist.png`: inspect error distribution and outliers.
- `*_test_yhat_vs_y.png`: points near the red diagonal are more accurate.

KNN-LLS development metrics are included because the PDF asks whether they can be measured. They are optimistic because every development point can select itself as its nearest neighbour; use the test metrics for the meaningful comparison.
