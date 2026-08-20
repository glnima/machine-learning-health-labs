# -*- coding: utf-8 -*-
"""Laboratory 5 - ROC analysis for two COVID-19 serological tests."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler


DATA_FILE = Path(__file__).with_name("covid_serological_results.csv")
OUTPUT_DIR = Path(__file__).with_name("lab5_results")
DBSCAN_EPS = 0.08
DBSCAN_MIN_SAMPLES = 3


def sensitivity_specificity(scores, truth):
    """Compute sensitivity and specificity for all relevant score thresholds."""
    thresholds = np.r_[0.0, np.unique(scores), np.nextafter(scores.max(), np.inf)]
    positive_scores = scores[truth == 1]
    negative_scores = scores[truth == 0]
    # Using >= at a score tie is the standard ROC convention used by scikit-learn.
    sensitivity = np.array([np.mean(positive_scores >= threshold) for threshold in thresholds])
    specificity = np.array([np.mean(negative_scores < threshold) for threshold in thresholds])
    return thresholds, sensitivity, specificity


def trapezoidal_area(x, y):
    """Manual area-under-curve calculation using the trapezoidal rule."""
    # Order vertical tied-score segments from lower to higher sensitivity.
    order = np.lexsort((y, x))
    return np.trapezoid(y[order], x[order])


def analyze_test(name, scores, truth):
    """Create the requested sensitivity/specificity, ROC, AUC, and Youden-J outputs."""
    thresholds, sensitivity, specificity = sensitivity_specificity(scores, truth)
    false_positive_rate = 1 - specificity
    manual_auc = trapezoidal_area(false_positive_rate, sensitivity)
    sklearn_auc = roc_auc_score(truth, scores)
    youden_j = sensitivity - false_positive_rate
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    result = {
        "test": name,
        "AUC_manual": manual_auc,
        "AUC_sklearn": sklearn_auc,
        "Youden_threshold": best_threshold,
        "Youden_J": youden_j[best_index],
        "sensitivity": sensitivity[best_index],
        "specificity": specificity[best_index],
        "false_positive_rate": false_positive_rate[best_index],
    }

    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, sensitivity, label="sensitivity")
    plt.plot(thresholds, specificity, label="specificity")
    plt.axvline(best_threshold, color="black", linestyle="--", label="Youden threshold")
    plt.xlabel("IgG threshold")
    plt.ylabel("probability")
    plt.title(f"{name}: sensitivity and specificity")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_sensitivity_specificity.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(false_positive_rate, sensitivity, label=f"manual AUC = {manual_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="random classifier")
    plt.scatter(false_positive_rate[best_index], sensitivity[best_index], color="red", zorder=3,
                label="Youden-J point")
    plt.xlabel("false positive rate = 1 - specificity")
    plt.ylabel("sensitivity")
    plt.title(f"{name}: ROC curve")
    plt.grid()
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_roc.png", dpi=150)
    plt.close()

    # Store the curve as a reproducible record of the manual calculation.
    pd.DataFrame({
        "threshold": thresholds, "sensitivity": sensitivity,
        "specificity": specificity, "false_positive_rate": false_positive_rate,
        "Youden_J": youden_j,
    }).to_csv(OUTPUT_DIR / f"{name}_roc_values.csv", index=False)
    return result


def main():
    plt.close("all")
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_FILE)
    data = data[data["COVID_swab_res"] != 1].copy()  # Remove uncertain swabs.
    data.loc[data["COVID_swab_res"] == 2, "COVID_swab_res"] = 1
    print(f"Rows after removing uncertain swabs: {len(data)}")
    print(data.describe())

    # Use min-max normalized marker values only for DBSCAN outlier detection.
    marker_columns = ["IgG_Test1_titre", "IgG_Test2_titre"]
    normalized_markers = MinMaxScaler().fit_transform(data[marker_columns])
    dbscan_labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(
        normalized_markers
    )
    outliers = dbscan_labels == -1
    removed = data.loc[outliers].copy()
    cleaned = data.loc[~outliers].copy()
    removed.to_csv(OUTPUT_DIR / "dbscan_removed_outliers.csv", index=False)
    print(f"DBSCAN removed {outliers.sum()} outliers; {len(cleaned)} rows remain.")

    plt.figure(figsize=(6, 5))
    plt.scatter(normalized_markers[~outliers, 0], normalized_markers[~outliers, 1],
                c=cleaned["COVID_swab_res"], cmap="coolwarm", s=14, label="retained")
    plt.scatter(normalized_markers[outliers, 0], normalized_markers[outliers, 1],
                facecolors="none", edgecolors="black", s=55, label="DBSCAN outlier")
    plt.xlabel("min-max normalized IgG Test1")
    plt.ylabel("min-max normalized IgG Test2")
    plt.title("DBSCAN outlier detection")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dbscan_outliers.png", dpi=150)
    plt.close()

    truth = cleaned["COVID_swab_res"].to_numpy()
    # Required illustrative result for Test2 at threshold 5.
    threshold_five = 5.0
    test2 = cleaned["IgG_Test2_titre"].to_numpy()
    sens_five = np.mean(test2[truth == 1] > threshold_five)
    spec_five = np.mean(test2[truth == 0] < threshold_five)
    print(f"Test2 at threshold 5: sensitivity={sens_five:.3f}, specificity={spec_five:.3f}")

    results = [
        analyze_test("Test1", cleaned["IgG_Test1_titre"].to_numpy(), truth),
        analyze_test("Test2", test2, truth),
    ]
    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUT_DIR / "roc_summary.csv", index=False)
    print("\nROC summary:")
    print(summary.to_string(index=False))

    # Independent Scikit-Learn check, requested by the slides.
    for name, scores in (("Test1", cleaned["IgG_Test1_titre"]), ("Test2", cleaned["IgG_Test2_titre"])):
        fpr, tpr, thresholds = roc_curve(truth, scores, pos_label=1)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
            OUTPUT_DIR / f"{name}_sklearn_roc_check.csv", index=False
        )
    print(f"\nSaved all figures and tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
