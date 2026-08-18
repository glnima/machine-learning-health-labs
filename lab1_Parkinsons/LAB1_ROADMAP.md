# Lab 1 roadmap: Parkinson's regression

## What the professor asks for

1. Use matricola `358768` as the random seed.
2. Always exclude `Jitter:DDP` and `Shimmer:DDA` to reduce collinearity.
3. Predict `total_UPDRS` with both least linear squares (LLS) and steepest descent.
4. Compare the training and test errors, MSE, R-squared, and correlation coefficient for the two methods.
5. Run with and without `motor_UPDRS`, then explain that the voice-only model should perform worse because motor UPDRS is strongly related to total UPDRS.
6. Run with and without shuffled rows, and compare the results. Shuffling distributes measurements from patients throughout both subsets; without it, the split depends on the CSV row order.

## How to run it

From the repository root:

```zsh
source .venv/bin/activate
python lab1_Parkinsons/lab1_2025_26_forstudents.py
```

The default experiment is **with Motor UPDRS and shuffled data**. Its output is saved in `lab1_Parkinsons/lab1_results/with_motor_shuffled/`.

## How to test the required cases

At the top of `lab1_2025_26_forstudents.py`, change only these flags, run the script again, and compare `performance_summary.csv` files:

| Case | `INCLUDE_MOTOR_UPDRS` | `SHUFFLE_DATA` |
| --- | --- | --- |
| Default | `True` | `True` |
| Voice only | `False` | `True` |
| No shuffle | `True` | `False` |
| Voice only, no shuffle | `False` | `False` |

For every run, check that LLS and steepest descent give nearly identical metrics, the training mean error is close to zero, and the test performance is not dramatically worse than training performance. Inspect the saved histogram and `y_hat` versus `y` figure for unusual errors or systematic bias.
