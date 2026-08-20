# -*- coding: utf-8 -*-
"""Laboratory 4 - chronic kidney disease data cleaning and classification."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


FEATURE_NAMES = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "htn", "dm", "cad", "appet", "pe",
    "ane", "classk",
]
FEATURE_TYPES = np.array([
    "num", "num", "cat", "cat", "cat", "cat", "cat", "cat", "cat", "num", "num",
    "num", "num", "num", "num", "num", "num", "num", "cat", "cat", "cat", "cat",
    "cat", "cat", "cat",
])
TARGET = "classk"
DATA_FILE = Path(__file__).with_name("chronic_kidney_disease_v2.arff")
OUTPUT_DIR = Path(__file__).parent / "lab4_results"
TARGET_NAMES = ["notckd", "ckd"]


def load_data():
    """Read the already corrected ARFF file and map categorical labels to 0/1."""
    data = pd.read_csv(
        DATA_FILE, sep=",", skiprows=29, names=FEATURE_NAMES, header=None,
        na_values=["?", "\t?"],
    )
    # Strip hidden spaces/tabs before mapping the categorical values.
    for column in data.select_dtypes(include=["object", "string"]):
        data[column] = data[column].str.strip()
    mapping = {
        "normal": 0, "abnormal": 1, "present": 1, "notpresent": 0,
        "yes": 1, "no": 0, "ckd": 1, "notckd": 0, "poor": 1, "good": 0,
    }
    return data.replace(mapping).apply(pd.to_numeric, errors="coerce")


def lls_impute(data):
    """Instructor-provided LLS imputation for rows with at least 19 observed values."""
    reduced = data.dropna(thresh=19).reset_index(drop=True)
    complete = reduced.dropna(thresh=len(FEATURE_NAMES)).reset_index(drop=True)
    training_array = complete.to_numpy(dtype=float)
    mean = training_array.mean(axis=0)
    std = training_array.std(axis=0)
    normalized_complete = (training_array - mean) / std
    normalized = (reduced.to_numpy(dtype=float) - mean) / std

    for row_index, row in enumerate(normalized):
        missing = np.isnan(row)
        if not missing.any():
            continue
        known_training = normalized_complete[:, ~missing]
        missing_training = normalized_complete[:, missing]
        # pinv is numerically safer than the explicit inverse in the original script.
        weights = np.linalg.pinv(known_training) @ missing_training
        row[missing] = row[~missing] @ weights
        normalized[row_index] = row

    imputed = normalized * std + mean
    # Map regressed categorical values back to their closest valid alphabet value.
    for index, kind in enumerate(FEATURE_TYPES):
        if kind != "cat":
            continue
        alphabet = np.sort(complete.iloc[:, index].dropna().unique())
        values = imputed[:, index, None]
        imputed[:, index] = alphabet[np.argmin((values - alphabet) ** 2, axis=1)]
    return pd.DataFrame(imputed, columns=FEATURE_NAMES), complete, reduced


def median_impute(data, complete):
    """Create y_new: all 400 rows, with each missing value replaced by Xtrain's median."""
    medians = complete.median(numeric_only=True)
    return data.fillna(medians)


def evaluate(model, dataset, label):
    """Fit a model on Xtrain and evaluate it on a fully imputed dataset."""
    features = dataset.drop(columns=TARGET)
    prediction = model.predict(features)
    matrix = confusion_matrix(dataset[TARGET], prediction, labels=[0, 1])
    return {
        "evaluation": label,
        "accuracy": accuracy_score(dataset[TARGET], prediction),
        "TN": matrix[0, 0], "FP": matrix[0, 1], "FN": matrix[1, 0], "TP": matrix[1, 1],
    }


def save_tree(model, filename, title):
    plt.figure(figsize=(18, 12))
    plot_tree(model, feature_names=FEATURE_NAMES[:-1], class_names=TARGET_NAMES,
              rounded=True, filled=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_importance(model, filename, title):
    importance = pd.Series(model.feature_importances_, index=FEATURE_NAMES[:-1]).sort_values()
    plt.figure(figsize=(8, 7))
    importance.plot.barh()
    plt.xlabel("feature importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return importance


def main():
    plt.close("all")
    OUTPUT_DIR.mkdir(exist_ok=True)
    original = load_data()
    x_new, x_train, reduced = lls_impute(original)
    y_new = median_impute(original, x_train)

    if y_new.isna().any().any() or x_new.isna().any().any():
        raise RuntimeError("Imputation failed: missing values remain.")
    print(f"Original data: {original.shape}; reduced data: {reduced.shape}; complete Xtrain: {x_train.shape}")
    print(f"x_new (LLS): {x_new.shape}; y_new (median): {y_new.shape}")
    x_new.to_csv(OUTPUT_DIR / "x_new_lls_imputed.csv", index=False)
    y_new.to_csv(OUTPUT_DIR / "y_new_median_imputed.csv", index=False)

    # Tree trained on complete Xtrain, then evaluated on both requested datasets.
    tree_complete = DecisionTreeClassifier(criterion="entropy")
    tree_complete.fit(x_train.drop(columns=TARGET), x_train[TARGET])
    save_tree(tree_complete, OUTPUT_DIR / "decision_tree_complete_training.png",
              "Decision tree trained on complete Xtrain")

    results = [
        {"model": "Decision tree (complete Xtrain)", **evaluate(tree_complete, x_new, "x_new (LLS)")},
        {"model": "Decision tree (complete Xtrain)", **evaluate(tree_complete, y_new, "y_new (median)")},
    ]

    # Random forests with the requested numbers of trees, on both imputed datasets.
    importance_table = pd.DataFrame(index=FEATURE_NAMES[:-1])
    for n_trees in (100, 1000):
        for dataset, dataset_name in ((x_new, "x_new (LLS)"), (y_new, "y_new (median)")):
            forest = RandomForestClassifier(n_estimators=n_trees, criterion="entropy")
            forest.fit(x_train.drop(columns=TARGET), x_train[TARGET])
            results.append({
                "model": f"Random forest ({n_trees} trees, complete Xtrain)",
                **evaluate(forest, dataset, dataset_name),
            })
            # One importance plot per forest size is enough: training is identical.
            if dataset_name == "x_new (LLS)":
                importance_table[f"RF_{n_trees}"] = save_importance(
                    forest, OUTPUT_DIR / f"random_forest_{n_trees}_importance.png",
                    f"Random forest feature importance ({n_trees} trees)",
                )

    # Required 50/50 shuffled evaluation: no random_state means a new split each run.
    train_set, test_set = train_test_split(y_new, test_size=0.5, shuffle=True)
    for model_name, model in (
        ("Random forest (1000 trees, 50/50 split)", RandomForestClassifier(n_estimators=1000, criterion="entropy")),
        ("Decision tree (50/50 split)", DecisionTreeClassifier(criterion="entropy")),
    ):
        model.fit(train_set.drop(columns=TARGET), train_set[TARGET])
        results.append({"model": model_name, **evaluate(model, test_set, "y_new test (50%)")})

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    importance_table.to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv")
    print("\nPerformance summary:")
    print(results_frame.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
