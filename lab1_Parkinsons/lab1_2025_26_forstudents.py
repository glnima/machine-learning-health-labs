# -*- coding: utf-8 -*-
"""Laboratory 1 - predict total UPDRS from Parkinson's data.

Implements every task listed on slide 42 of the laboratory PDF.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Change only these two flags when asked to test the required cases.
INCLUDE_MOTOR_UPDRS = True
SHUFFLE_DATA = True
RANDOM_SEED = 358768  # matricola

DATA_FILE = Path(__file__).with_name("parkinsons_updrs_av.csv")
OUTPUT_ROOT = Path(__file__).with_name("lab1_results")
TARGET = "total_UPDRS"
DROP_ALWAYS = ["subject#", "Jitter:DDP", "Shimmer:DDA"]


def error_statistics(y_true, y_pred):
    """Return the error statistics requested in the slides."""
    error = y_true - y_pred
    mse = np.mean(error**2)
    return {
        "min": error.min(),
        "max": error.max(),
        "mean": error.mean(),
        "std": error.std(),
        "MSE": mse,
        "R^2": 1 - mse / np.var(y_true),
        "corr_coeff": np.corrcoef(y_true, y_pred)[0, 1],
    }


def fit_lls(x_train, y_train):
    """Least-linear-squares solution of min ||Xw-y||^2."""
    return np.linalg.solve(x_train.T @ x_train, x_train.T @ y_train)


def fit_steepest_descent(x_train, y_train, tolerance=1e-9, max_iterations=150_000):
    """Minimize MSE with a stable data-dependent step and stop on weight updates."""
    n_samples = len(y_train)
    largest_eigenvalue = np.linalg.eigvalsh(x_train.T @ x_train).max()
    learning_rate = 0.9 * n_samples / (2 * largest_eigenvalue)
    weights = np.zeros(x_train.shape[1])

    for iteration in range(1, max_iterations + 1):
        gradient = 2 / n_samples * x_train.T @ (x_train @ weights - y_train)
        new_weights = weights - learning_rate * gradient
        if np.linalg.norm(new_weights - weights) <= tolerance:
            return new_weights, iteration, learning_rate
        weights = new_weights
    raise RuntimeError("Steepest descent did not converge; increase max_iterations.")


def save_figures(output_dir, regressors, weights, predictions, targets):
    """Save the required weights, error, and prediction plots."""
    x = np.arange(len(regressors))
    plt.figure(figsize=(8, 4))
    plt.plot(x, weights["LLS"], "-o", label="LLS")
    plt.plot(x, weights["Steepest descent"], "--s", label="Steepest descent")
    plt.xticks(x, regressors, rotation=90)
    plt.ylabel("weight")
    plt.title("Optimized regression weights")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "weights_comparison.png", dpi=150)

    for method in weights:
        train_error = targets["train"] - predictions[method]["train"]
        test_error = targets["test"] - predictions[method]["test"]
        filename = method.lower().replace(" ", "_")

        plt.figure(figsize=(6, 4))
        plt.hist([train_error, test_error], bins=50, density=True,
                 label=["training", "test"])
        plt.xlabel("e = y - y_hat")
        plt.ylabel("density")
        plt.title(f"{method} error histograms")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename}_hist.png", dpi=150)

        plt.figure(figsize=(5, 5))
        plt.plot(targets["test"], predictions[method]["test"], ".")
        low, high = plt.xlim()
        plt.plot([low, high], [low, high], "r", linewidth=2)
        plt.xlabel("y: true total_UPDRS")
        plt.ylabel("y_hat: predicted total_UPDRS")
        plt.title(f"{method} test predictions")
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename}_yhat_vs_y.png", dpi=150)


def main():
    plt.close("all")
    data = pd.read_csv(DATA_FILE)
    print(f"Original dataset shape: {data.shape}")
    print(f"Distinct patients: {data['subject#'].nunique()}")
    print("\nDataset information:")
    data.info()
    print("\nDescriptive statistics:")
    print(data.describe().T)

    # Dataset analysis is done before the train/test split, as specified in the PDF.
    correlation = ((data - data.mean()) / data.std()).cov()
    print("\nCorrelation with total_UPDRS:")
    print(correlation[TARGET].sort_values(ascending=False))

    experiment = "with_motor" if INCLUDE_MOTOR_UPDRS else "voice_only"
    experiment += "_shuffled" if SHUFFLE_DATA else "_not_shuffled"
    output_dir = OUTPUT_ROOT / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.matshow(np.abs(correlation.values), fignum=plt.gcf().number)
    plt.xticks(np.arange(len(data.columns)), data.columns, rotation=90)
    plt.yticks(np.arange(len(data.columns)), data.columns)
    plt.colorbar()
    plt.title("Absolute correlation coefficients", pad=18)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_coefficients.png", dpi=150)

    if SHUFFLE_DATA:
        data = data.sample(frac=1, random_state=RANDOM_SEED, ignore_index=True)

    n_train = len(data) // 2
    train_data = data.iloc[:n_train]
    mean = train_data.mean()
    std = train_data.std()
    normalized = (data - mean) / std  # Training statistics only: no data leakage.

    columns_to_drop = [TARGET, *DROP_ALWAYS]
    if not INCLUDE_MOTOR_UPDRS:
        columns_to_drop.append("motor_UPDRS")
    regressors = normalized.drop(columns=columns_to_drop).columns.tolist()
    x_all = normalized[regressors].to_numpy()
    y_all = normalized[TARGET].to_numpy()
    x_train, x_test = x_all[:n_train], x_all[n_train:]
    y_train_norm, y_test_norm = y_all[:n_train], y_all[n_train:]

    lls_weights = fit_lls(x_train, y_train_norm)
    sd_weights, iterations, learning_rate = fit_steepest_descent(x_train, y_train_norm)
    weights = {"LLS": lls_weights, "Steepest descent": sd_weights}
    print(f"\nExperiment: {experiment}")
    print(f"Regressors ({len(regressors)}): {regressors}")
    print(f"Steepest descent: {iterations} iterations; learning rate {learning_rate:.6g}")

    y_train = y_train_norm * std[TARGET] + mean[TARGET]
    y_test = y_test_norm * std[TARGET] + mean[TARGET]
    targets = {"train": y_train, "test": y_test}
    predictions = {}
    summary_rows = []
    for method, method_weights in weights.items():
        predictions[method] = {
            "train": (x_train @ method_weights) * std[TARGET] + mean[TARGET],
            "test": (x_test @ method_weights) * std[TARGET] + mean[TARGET],
        }
        for subset in ("train", "test"):
            summary_rows.append({
                "method": method,
                "subset": subset,
                **error_statistics(targets[subset], predictions[method][subset]),
            })

    results = pd.DataFrame(summary_rows).set_index(["method", "subset"])
    print("\nPerformance summary:")
    print(results)
    results.to_csv(output_dir / "performance_summary.csv")
    pd.DataFrame(weights, index=regressors).to_csv(output_dir / "weights.csv")
    save_figures(output_dir, regressors, weights, predictions, targets)
    print(f"\nSaved figures and CSV summaries to: {output_dir}")
    plt.show()


if __name__ == "__main__":
    main()
