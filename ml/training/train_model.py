"""
ImageAssess - Robust Image Quality Model Training

Trains a Random Forest image-quality classifier using the
same features produced by extract_features.py.

Classes:
    ACCEPTABLE
    DEGRADED
    POTENTIALLY_DEFECTIVE

Important:
    Uses the dataset's existing split column when available.
    This prevents related versions of the same source image
    from leaking between training and testing.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


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
# ============================================================

def create_robust_features(df):
    """
    Create additional image-quality features.

    These features make the model less dependent on
    absolute brightness and colour values.
    """

    data = df.copy()

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    data["brightness_normalized"] = (
        data["brightness_mean"] / 255.0
    )

    data["dark_bright_balance"] = (
        data["bright_pixel_ratio"]
        - data["dark_pixel_ratio"]
    )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    data["relative_contrast"] = (
        data["contrast_percentile_range"]
        / (data["brightness_mean"] + 1.0)
    )

    # --------------------------------------------------------
    # Sharpness
    # --------------------------------------------------------

    data["sharpness_relative"] = (
        data["laplacian_variance"]
        / (data["contrast_std"] + 1.0)
    )

    # --------------------------------------------------------
    # Edge / texture
    # --------------------------------------------------------

    data["edge_texture_ratio"] = (
        data["edge_density"]
        / (data["entropy"] + 1.0)
    )

    # --------------------------------------------------------
    # Noise
    # --------------------------------------------------------

    data["relative_noise"] = (
        data["noise_estimate"]
        / (data["contrast_std"] + 1.0)
    )

    # --------------------------------------------------------
    # Colour balance
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Saturation
    # --------------------------------------------------------

    data["saturation_normalized"] = (
        data["saturation_mean"] / 255.0
    )

    return data


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("IMAGEASSESS - ROBUST MODEL TRAINING")
    print("=" * 70)

    print("\nLoading feature dataset:")
    print(FEATURES_PATH)

    if not FEATURES_PATH.exists():

        raise FileNotFoundError(
            f"\nFeature dataset not found:\n"
            f"{FEATURES_PATH}\n\n"
            "Run:\n"
            "python ml/features/extract_features.py"
        )

    df = pd.read_csv(FEATURES_PATH)

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    if "quality_label" not in df.columns:
        raise ValueError(
            "quality_label column is missing."
        )

    print(f"\nDataset rows: {len(df)}")
    print(f"Dataset columns: {len(df.columns)}")

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    # --------------------------------------------------------
    # Check base features
    # --------------------------------------------------------

    missing = [
        feature
        for feature in BASE_FEATURES
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required features:\n"
            + "\n".join(
                f" - {feature}"
                for feature in missing
            )
        )

    # --------------------------------------------------------
    # Engineer features
    # --------------------------------------------------------

    engineered_df = create_robust_features(df)

    # --------------------------------------------------------
    # Metadata columns
    # --------------------------------------------------------

    metadata_columns = {
        "source_id",
        "source_category",
        "filename",
        "filepath",
        "split",
        "condition",
        "severity",
        "quality_label",
    }

    feature_columns = [
        column
        for column in engineered_df.columns
        if column not in metadata_columns
        and pd.api.types.is_numeric_dtype(
            engineered_df[column]
        )
    ]

    if not feature_columns:

        raise ValueError(
            "No numeric feature columns available."
        )

    X = engineered_df[
        feature_columns
    ].copy()

    y = engineered_df[
        "quality_label"
    ].astype(str)

    # --------------------------------------------------------
    # Clean values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    # --------------------------------------------------------
    # Print classes
    # --------------------------------------------------------

    print("\nClass distribution:")

    print(
        y.value_counts()
        .to_string()
    )

    print(
        f"\nNumber of features: "
        f"{len(feature_columns)}"
    )

    return engineered_df, X, y, feature_columns


# ============================================================
# CREATE TRAIN / TEST SET
# ============================================================

def create_split(engineered_df, X, y):

    # --------------------------------------------------------
    # Prefer existing dataset split
    # --------------------------------------------------------

    if "split" in engineered_df.columns:

        split_values = (
            engineered_df["split"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        train_mask = split_values.isin(
            [
                "train",
                "training",
            ]
        )

        test_mask = split_values.isin(
            [
                "test",
                "testing",
            ]
        )

        if (
            train_mask.sum() > 0
            and test_mask.sum() > 0
        ):

            X_train = X.loc[
                train_mask
            ]

            X_test = X.loc[
                test_mask
            ]

            y_train = y.loc[
                train_mask
            ]

            y_test = y.loc[
                test_mask
            ]

            print(
                "\nUsing existing dataset split."
            )

            print(
                f"Training samples: "
                f"{len(X_train)}"
            )

            print(
                f"Testing samples : "
                f"{len(X_test)}"
            )

            return (
                X_train,
                X_test,
                y_train,
                y_test,
            )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    print(
        "\nWARNING: Dataset split not usable."
    )

    print(
        "Using stratified random split."
    )

    from sklearn.model_selection import train_test_split

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples : "
        f"{len(X_test)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(
    X_train,
    y_train,
):

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING RANDOM FOREST"
    )

    print(
        "=" * 70
    )

    model = RandomForestClassifier(

        # More trees = more stable predictions
        n_estimators=700,

        # Prevent individual trees becoming
        # excessively specialized
        max_depth=20,

        # Helps reduce overfitting
        min_samples_split=4,

        # Prevent tiny leaves
        min_samples_leaf=2,

        # Standard robust feature sampling
        max_features="sqrt",

        # Handle class imbalance
        class_weight="balanced",

        # Reproducibility
        random_state=42,

        # Use all CPU cores
        n_jobs=-1,
    )

    print(
        "\nTraining model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training completed."
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    feature_columns,
):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions,
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAccuracy           : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy  : "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    labels = model.classes_

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        pd.DataFrame(
            matrix,
            index=[
                f"Actual: {label}"
                for label in labels
            ],
            columns=[
                f"Predicted: {label}"
                for label in labels
            ],
        )
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame({

        "feature":
            feature_columns,

        "importance":
            model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print(
        "\nTop Feature Importance:"
    )

    print(
        importance.head(20)
        .to_string(index=False)
    )

    return (
        accuracy,
        balanced_accuracy,
        importance,
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    feature_columns,
    accuracy,
    balanced_accuracy,
):

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    package = {

        "model":
            model,

        "features":
            feature_columns,

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(balanced_accuracy),

        "base_features":
            BASE_FEATURES,

        "version":
            "3.0",
    }

    joblib.dump(
        package,
        MODEL_PATH,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL SAVED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nModel file:\n"
        f"{MODEL_PATH}"
    )

    print(
        f"\nFeatures saved: "
        f"{len(feature_columns)}"
    )

    print(
        f"Validation accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Load
    df = load_data()

    # Prepare
    (
        engineered_df,
        X,
        y,
        feature_columns,
    ) = prepare_features(df)

    # Split
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = create_split(
        engineered_df,
        X,
        y,
    )

    # Train
    model = train_random_forest(
        X_train,
        y_train,
    )

    # Evaluate
    (
        accuracy,
        balanced_accuracy,
        importance,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
        feature_columns,
    )

    # Save
    save_model(
        model,
        feature_columns,
        accuracy,
        balanced_accuracy,
    )

    print(
        "\nDONE."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()