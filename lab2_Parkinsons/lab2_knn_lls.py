# -*- coding: utf-8 -*-
"""Laboratory 2 - K-nearest-neighbour local LLS regression for Parkinson's data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_SEED = 358768  # matricola
K_MIN = 20
K_MAX = 350
K_STEP = 5
RIDGE_EPSILON = 1e-8

LAB_DIR = Path(__file__).parent
DATA_FILE = LAB_DIR.parent / "lab1_Parkinsons" / "parkinsons_updrs_av.csv"
OUTPUT_DIR = LAB_DIR / "lab2_results"
TARGET = "total_UPDRS"
REGRESSORS = [
    "sex", "age", "motor_UPDRS", "Jitter(%)", "Jitter(Abs)", "Jitter:RAP",
    "Jitter:PPQ5", "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "NHR", "HNR", "RPDE", "DFA", "PPE",
]


def error_statistics(y_true, y_pred):
    """Calculate the Lab 1/Lab 2 regression metrics."""
    error = y_true - y_pred
    mse = np.mean(error**2)
    return {
        "mean_error": error.mean(),
        "std_error": error.std(),
        "MSE": mse,
        "R^2": 1 - mse / np.var(y_true),
        "corr_coeff": np.corrcoef(y_true, y_pred)[0, 1],
    }


def local_lls_predict(x_query, x_reference, y_reference, k):
    """Predict queries with manually implemented KNN-LLS and ridge epsilon I."""
    n_features = x_reference.shape[1]
    identity = np.eye(n_features)
    predictions = np.empty(len(x_query))
    for index, point in enumerate(x_query):
        squared_distances = np.sum((x_reference - point) ** 2, axis=1)
        neighbour_indices = np.argsort(squared_distances)[:k]
        matrix_a = x_reference[neighbour_indices]
        vector_y = y_reference[neighbour_indices]
        weights = np.linalg.solve(
            matrix_a.T @ matrix_a + RIDGE_EPSILON * identity,
            matrix_a.T @ vector_y,
        )
        predictions[index] = point @ weights
    return predictions


def plot_error_histogram(y_true, y_pred, title, filename):
    error = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(error, bins=50, density=True)
    plt.xlabel("e = y - y_hat (UPDRS points)")
    plt.ylabel("density")
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)


def plot_regression_line(y_true, y_pred, title, filename):
    plt.figure(figsize=(5, 5))
    plt.plot(y_true, y_pred, ".")
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    plt.plot([low, high], [low, high], "r", linewidth=2)
    plt.xlabel("y: true total_UPDRS")
    plt.ylabel("y_hat: predicted total_UPDRS")
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_FILE).sample(
        frac=1, random_state=RANDOM_SEED, ignore_index=True
    )
    n_total = len(data)
    n_train = int(0.40 * n_total)
    n_validation = int(0.20 * n_total)
    n_development = n_train + n_validation
    print(f"Dataset: {data.shape}; seed: {RANDOM_SEED}")
    print(f"Split: train={n_train}, validation={n_validation}, test={n_total - n_development}")

    # Normalization statistics use true training data only, as required by the PDF.
    true_training = data.iloc[:n_train]
    mean = true_training.mean()
    std = true_training.std()
    normalized = (data - mean) / std
    x_all = normalized[REGRESSORS].to_numpy()
    y_all_normalized = normalized[TARGET].to_numpy()

    x_train = x_all[:n_train]
    y_train = y_all_normalized[:n_train]
    x_validation = x_all[n_train:n_development]
    y_validation = y_all_normalized[n_train:n_development]
    x_test = x_all[n_development:]
    y_test = y_all_normalized[n_development:]

    # Manual validation search for K, using only the true training subset.
    candidate_k = np.arange(K_MIN, K_MAX + 1, K_STEP)
    validation_mse = []
    for k in candidate_k:
        prediction = local_lls_predict(x_validation, x_train, y_train, k)
        validation_mse.append(np.mean((y_validation - prediction) ** 2))
    validation_mse = np.array(validation_mse)
    k_opt = int(candidate_k[np.argmin(validation_mse)])
    print(f"Kopt = {k_opt}; validation MSE (normalized) = {validation_mse.min():.6f}")

    plt.figure(figsize=(7, 4))
    plt.plot(candidate_k, validation_mse, "-o")
    plt.axvline(k_opt, color="r", linestyle="--", label=f"Kopt = {k_opt}")
    plt.xlabel("K nearest neighbours")
    plt.ylabel("validation MSE (normalized total_UPDRS)")
    plt.title("K optimization by validation error")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_mse_vs_k.png", dpi=150)
    pd.DataFrame({"K": candidate_k, "validation_MSE": validation_mse}).to_csv(
        OUTPUT_DIR / "validation_mse_vs_k.csv", index=False
    )

    # For the final test comparison, both methods use the same 60% development data.
    x_development = x_all[:n_development]
    y_development = y_all_normalized[:n_development]
    knn_test_normalized = local_lls_predict(x_test, x_development, y_development, k_opt)
    lls_weights = np.linalg.solve(
        x_development.T @ x_development, x_development.T @ y_development
    )
    lls_test_normalized = x_test @ lls_weights

    # KNN-LLS training metrics are measurable, but optimistic: each point is its own nearest neighbour.
    knn_train_normalized = local_lls_predict(
        x_development, x_development, y_development, k_opt
    )
    lls_train_normalized = x_development @ lls_weights

    scale = std[TARGET]
    offset = mean[TARGET]
    y_test_real = y_test * scale + offset
    y_development_real = y_development * scale + offset
    predictions = {
        "KNN-LLS": {
            "test": knn_test_normalized * scale + offset,
            "development": knn_train_normalized * scale + offset,
        },
        "Standard LLS": {
            "test": lls_test_normalized * scale + offset,
            "development": lls_train_normalized * scale + offset,
        },
    }
    truths = {"test": y_test_real, "development": y_development_real}
    rows = []
    for method, method_predictions in predictions.items():
        for subset in ("development", "test"):
            rows.append({
                "method": method,
                "subset": subset,
                **error_statistics(truths[subset], method_predictions[subset]),
            })
    results = pd.DataFrame(rows).set_index(["method", "subset"])
    print("\nPerformance summary (UPDRS points):")
    print(results)
    results.to_csv(OUTPUT_DIR / "performance_summary.csv")

    for method, method_predictions in predictions.items():
        filename_prefix = method.lower().replace(" ", "_").replace("-", "_")
        plot_error_histogram(
            y_test_real, method_predictions["test"], f"{method}: test error histogram",
            OUTPUT_DIR / f"{filename_prefix}_test_hist.png",
        )
        plot_regression_line(
            y_test_real, method_predictions["test"], f"{method}: test predictions",
            OUTPUT_DIR / f"{filename_prefix}_test_yhat_vs_y.png",
        )
    print(f"Saved results to {OUTPUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
