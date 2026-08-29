"""
ImageAssess - Model Evaluation

Evaluates the trained Random Forest using the same
26 engineered features and the same 80/20 stratified split
used during training.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "image_quality_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "random_forest.joblib"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "docs"
    / "evaluation"
)


# ============================================================
# BASE FEATURES
# ============================================================

BASE_FEATURES = [
    "brightness_mean",
    "brightness_std",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "contrast_std",
    "contrast_percentile_range",
    "laplacian_variance",
    "edge_density",
    "noise_estimate",
    "entropy",
    "saturation_mean",
    "saturation_std",
    "low_saturation_ratio",
    "blue_mean",
    "green_mean",
    "red_mean",
]


# ============================================================
# FEATURE ENGINEERING
# MUST MATCH TRAINING
# ============================================================

def create_robust_features(df):

    data = df.copy()

    data["brightness_normalized"] = (
        data["brightness_mean"] / 255.0
    )

    data["dark_bright_balance"] = (
        data["bright_pixel_ratio"]
        - data["dark_pixel_ratio"]
    )

    data["relative_contrast"] = (
        data["contrast_percentile_range"]
        / (data["brightness_mean"] + 1.0)
    )

    data["sharpness_relative"] = (
        data["laplacian_variance"]
        / (data["contrast_std"] + 1.0)
    )

    data["edge_texture_ratio"] = (
        data["edge_density"]
        / (data["entropy"] + 1.0)
    )

    data["relative_noise"] = (
        data["noise_estimate"]
        / (data["contrast_std"] + 1.0)
    )

    data["blue_red_difference"] = (
        data["blue_mean"]
        - data["red_mean"]
    )

    data["green_red_difference"] = (
        data["green_mean"]
        - data["red_mean"]
    )

    data["colour_spread"] = (
        data[
            [
                "blue_mean",
                "green_mean",
                "red_mean",
            ]
        ].max(axis=1)
        -
        data[
            [
                "blue_mean",
                "green_mean",
                "red_mean",
            ]
        ].min(axis=1)
    )

    data["saturation_normalized"] = (
        data["saturation_mean"] / 255.0
    )

    return data


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IMAGEASSESS MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{FEATURES_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading feature dataset...")

    df = pd.read_csv(FEATURES_PATH)

    print(f"Total rows: {len(df)}")

    if "quality_label" not in df.columns:
        raise ValueError(
            "quality_label column not found."
        )

    # --------------------------------------------------------
    # Create SAME 26 features used during training
    # --------------------------------------------------------

    print("\nCreating engineered features...")

    engineered_df = create_robust_features(df)

    feature_columns = [
        column
        for column in engineered_df.columns
        if column not in {
            "source_id",
            "source_category",
            "split",
            "condition",
            "severity",
            "quality_label",
            "filename",
            "filepath",
        }
        and pd.api.types.is_numeric_dtype(
            engineered_df[column]
        )
    ]

    print(
        f"Number of evaluation features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print("\nLoading trained Random Forest...")

    package = joblib.load(MODEL_PATH)

    if isinstance(package, dict):
        model = package["model"]
        trained_features = package["features"]
    else:
        model = package
        trained_features = feature_columns

    print(
        f"Model expects: "
        f"{len(trained_features)} features"
    )

    # --------------------------------------------------------
    # Verify feature compatibility
    # --------------------------------------------------------

    missing = [
        feature
        for feature in trained_features
        if feature not in engineered_df.columns
    ]

    if missing:
        raise ValueError(
            "Missing features required by trained model:\n"
            + "\n".join(missing)
        )

    # IMPORTANT:
    # Use EXACT feature order stored with the model.

    X = engineered_df[trained_features].copy()

    y = engineered_df["quality_label"].astype(str)

    # --------------------------------------------------------
    # Clean invalid values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    # --------------------------------------------------------
    # Same 80/20 split used during training
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nRunning predictions...")

    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy           : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy  : "
        f"{balanced_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    labels = model.classes_

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print("Confusion Matrix:")

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual: {label}"
            for label in labels
        ],
        columns=[
            f"Predicted: {label}"
            for label in labels
        ],
    )

    print(cm_df)

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame({
        "feature": trained_features,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print("\nTop Feature Importance:")

    print(
        importance.head(15)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save evaluation files
    # --------------------------------------------------------

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Feature importance CSV
    importance_path = (
        EVALUATION_DIR
        / "feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False
    )

    # Evaluation report
    report_path = (
        EVALUATION_DIR
        / "evaluation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "IMAGEASSESS MODEL EVALUATION\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Dataset rows: {len(df)}\n"
        )

        file.write(
            f"Features: {len(trained_features)}\n"
        )

        file.write(
            f"Test samples: {len(X_test)}\n\n"
        )

        file.write(
            f"Accuracy: "
            f"{accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Balanced Accuracy: "
            f"{balanced_accuracy * 100:.2f}%\n\n"
        )

        file.write(
            "Classification Report\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(report)

        file.write(
            "\n\nConfusion Matrix\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            cm_df.to_string()
        )

        file.write(
            "\n\nFeature Importance\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            importance.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Save confusion matrix as CSV too
    # --------------------------------------------------------

    confusion_path = (
        EVALUATION_DIR
        / "confusion_matrix.csv"
    )

    cm_df.to_csv(
        confusion_path
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"\nAccuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"\nSaved evaluation report:\n"
        f"{report_path}"
    )

    print(
        f"\nSaved feature importance:\n"
        f"{importance_path}"
    )

    print(
        f"\nSaved confusion matrix:\n"
        f"{confusion_path}"
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()