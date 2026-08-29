"""
ImageAssess - Image Quality Prediction

Loads the trained Random Forest model and predicts the quality
of a new image using the same CV features used during training.

Quality rules are used to prevent obvious measurable defects
from being missed by the ML model.
"""

from pathlib import Path
import sys
import importlib.util

import cv2
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "random_forest.joblib"
)

FEATURES_DIR = PROJECT_ROOT / "ml" / "features"
TRAINING_DIR = PROJECT_ROOT / "ml" / "training"


# ============================================================
# LOAD FEATURE EXTRACTOR
# ============================================================

for folder in [
    FEATURES_DIR,
    TRAINING_DIR,
    PROJECT_ROOT,
]:
    folder_string = str(folder)

    if folder_string not in sys.path:
        sys.path.insert(0, folder_string)


EXTRACT_FEATURES_FILE = (
    FEATURES_DIR / "extract_features.py"
)

if not EXTRACT_FEATURES_FILE.exists():
    raise FileNotFoundError(
        f"Feature extractor not found:\n"
        f"{EXTRACT_FEATURES_FILE}"
    )


spec = importlib.util.spec_from_file_location(
    "imageassess_extract_features",
    EXTRACT_FEATURES_FILE,
)

if spec is None or spec.loader is None:
    raise ImportError(
        "Could not load extract_features.py"
    )


extractor_module = (
    importlib.util.module_from_spec(spec)
)

spec.loader.exec_module(
    extractor_module
)

extract_features = (
    extractor_module.extract_features
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
# ENGINEERED FEATURES
# ============================================================

def create_robust_features(features):

    data = pd.DataFrame(
        [features]
    ).copy()

    data["brightness_normalized"] = (
        data["brightness_mean"] / 255.0
    )

    data["dark_bright_balance"] = (
        data["bright_pixel_ratio"]
        -
        data["dark_pixel_ratio"]
    )

    data["relative_contrast"] = (
        data["contrast_percentile_range"]
        /
        (data["brightness_mean"] + 1.0)
    )

    data["sharpness_relative"] = (
        data["laplacian_variance"]
        /
        (data["contrast_std"] + 1.0)
    )

    data["edge_texture_ratio"] = (
        data["edge_density"]
        /
        (data["entropy"] + 1.0)
    )

    data["relative_noise"] = (
        data["noise_estimate"]
        /
        (data["contrast_std"] + 1.0)
    )

    data["blue_red_difference"] = (
        data["blue_mean"]
        -
        data["red_mean"]
    )

    data["green_red_difference"] = (
        data["green_mean"]
        -
        data["red_mean"]
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
# ISSUE DETECTION
# ============================================================

def detect_issues(features, prediction):

    issues = []

    brightness = float(
        features.get("brightness_mean", 0)
    )

    sharpness = float(
        features.get("laplacian_variance", 0)
    )

    noise = float(
        features.get("noise_estimate", 0)
    )

    bright_pixel_ratio = float(
        features.get("bright_pixel_ratio", 0)
    )

    dark_pixel_ratio = float(
        features.get("dark_pixel_ratio", 0)
    )

    entropy = float(
        features.get("entropy", 0)
    )

    # ========================================================
    # BLUR
    # ========================================================

    if sharpness < 100:

        issues.append({
            "type": "blur",
            "severity": "high",
            "confidence": 0.95,
        })

    elif sharpness < 250:

        issues.append({
            "type": "blur",
            "severity": "medium",
            "confidence": 0.80,
        })

    # ========================================================
    # UNDEREXPOSURE
    # ========================================================

    if brightness < 50:

        issues.append({
            "type": "underexposure",
            "severity": "high",
            "confidence": 0.95,
        })

    elif brightness < 65:

        issues.append({
            "type": "underexposure",
            "severity": "medium",
            "confidence": 0.80,
        })

    # ========================================================
    # OVEREXPOSURE
    # ========================================================

    if (
        brightness > 150
        and bright_pixel_ratio > 0.38
        and dark_pixel_ratio < 0.13
        and entropy > 6.2
    ):

        issues.append({
            "type": "overexposure",
            "severity": "high",
            "confidence": 0.90,
        })

    # ========================================================
    # NOISE
    # ========================================================

    if noise > 18:

        issues.append({
            "type": "noise",
            "severity": "high",
            "confidence": 0.90,
        })

    elif noise > 14:

        issues.append({
            "type": "noise",
            "severity": "medium",
            "confidence": 0.80,
        })

    # ========================================================
    # POTENTIAL VISUAL DEFECT
    # ========================================================

    if prediction == "POTENTIALLY_DEFECTIVE":

        issues.append({
            "type": "potential_visual_defect",
            "severity": "high",
            "confidence": 0.75,
        })

    return issues


# ============================================================
# QUALITY OVERRIDE
# ============================================================

def apply_quality_override(
    prediction,
    features,
):

    sharpness = float(
        features.get(
            "laplacian_variance",
            0,
        )
    )

    noise = float(
        features.get(
            "noise_estimate",
            0,
        )
    )

    brightness = float(
        features.get(
            "brightness_mean",
            0,
        )
    )

    edge_density = float(
        features.get(
            "edge_density",
            0,
        )
    )

    entropy = float(
        features.get(
            "entropy",
            0,
        )
    )

    bright_pixel_ratio = float(
        features.get(
            "bright_pixel_ratio",
            0,
        )
    )

    dark_pixel_ratio = float(
        features.get(
            "dark_pixel_ratio",
            0,
        )
    )

    # ========================================================
    # 1. BLUR
    # ========================================================

    if sharpness < 100:
        return "DEGRADED", "blur"

    # ========================================================
    # 2. SEVERE NOISE
    # ========================================================

    if noise > 18:

        # Severe degradation should remain
        # POTENTIALLY_DEFECTIVE when ML agrees.
        if prediction == "POTENTIALLY_DEFECTIVE":
            return (
                "POTENTIALLY_DEFECTIVE",
                "severe_degradation",
            )

        return "DEGRADED", "noise"

    # ========================================================
    # 3. UNDEREXPOSURE
    # ========================================================

    if brightness < 65:
        return "DEGRADED", "underexposure"

    # ========================================================
    # 4. OVEREXPOSURE
    # ========================================================

    if (
        brightness > 150
        and bright_pixel_ratio > 0.38
        and dark_pixel_ratio < 0.13
        and entropy > 6.2
    ):
        return "DEGRADED", "overexposure"

    # ========================================================
    # 5. GOOD QUALITY IMAGE
    # ========================================================

    good_brightness = (
        65 <= brightness <= 220
    )

    good_sharpness = (
        sharpness >= 250
    )

    good_noise = (
        noise < 15
    )

    usable_edges = (
        edge_density >= 0.04
    )

    usable_entropy = (
        entropy >= 4.5
    )

    if (
        good_brightness
        and good_sharpness
        and good_noise
        and usable_edges
        and usable_entropy
    ):
        return "ACCEPTABLE", "quality_check"

    # ========================================================
    # 6. OTHERWISE KEEP MODEL RESULT
    # ========================================================

    return prediction, None


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image_path):

    print("\n" + "=" * 70)

    print(
        "IMAGEASSESS - IMAGE QUALITY ASSESSMENT"
    )

    print("=" * 70)

    # ========================================================
    # MODEL CHECK
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run:\n"
            "python ml/training/train_model.py"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    package = joblib.load(
        MODEL_PATH
    )

    model = package["model"]

    feature_columns = package["features"]

    print(
        f"\nImage: {image_path}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    # ========================================================
    # READ IMAGE
    # ========================================================

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    # ========================================================
    # EXTRACT FEATURES
    # ========================================================

    print(
        "\nExtracting image quality features..."
    )

    base_features = extract_features(
        image
    )

    # ========================================================
    # CHECK BASE FEATURES
    # ========================================================

    missing_features = [
        feature
        for feature in BASE_FEATURES
        if feature not in base_features
    ]

    if missing_features:

        raise ValueError(
            "Missing base features:\n"
            +
            "\n".join(
                missing_features
            )
        )

    # ========================================================
    # ENGINEER FEATURES
    # ========================================================

    engineered = create_robust_features(
        base_features
    )

    # ========================================================
    # CHECK MODEL FEATURES
    # ========================================================

    missing_model_features = [
        feature
        for feature in feature_columns
        if feature not in engineered.columns
    ]

    if missing_model_features:

        raise ValueError(
            "Model expects missing features:\n"
            +
            "\n".join(
                missing_model_features
            )
        )

    # ========================================================
    # EXACT TRAINING ORDER
    # ========================================================

    X = engineered[
        feature_columns
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    model_prediction = str(
        model.predict(X)[0]
    )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    probabilities = model.predict_proba(
        X
    )[0]

    classes = model.classes_

    probability_dict = {
        str(class_name):
        float(probability)

        for class_name, probability
        in zip(
            classes,
            probabilities,
        )
    }

    # ========================================================
    # QUALITY OVERRIDE
    # ========================================================

    final_prediction, override_reason = (
        apply_quality_override(
            model_prediction,
            base_features,
        )
    )

    # ========================================================
    # DISPLAY PROBABILITY
    # ========================================================

    if override_reason is not None:

        if final_prediction == "DEGRADED":

            probability_dict = {
                "ACCEPTABLE": 0.01,
                "DEGRADED": 0.95,
                "POTENTIALLY_DEFECTIVE": 0.04,
            }

        elif final_prediction == "ACCEPTABLE":

            probability_dict = {
                "ACCEPTABLE": 0.95,
                "DEGRADED": 0.04,
                "POTENTIALLY_DEFECTIVE": 0.01,
            }

        # IMPORTANT:
        # If final prediction is POTENTIALLY_DEFECTIVE,
        # keep the actual ML probabilities.

    # ========================================================
    # DETECT ISSUES
    # ========================================================

    issues = detect_issues(
        base_features,
        final_prediction,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        f"MODEL PREDICTION : "
        f"{model_prediction}"
    )

    print(
        f"FINAL PREDICTION : "
        f"{final_prediction}"
    )

    if override_reason:

        print(
            f"QUALITY OVERRIDE : "
            f"{override_reason}"
        )

    print(
        "-" * 70
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    print(
        "\nConfidence:"
    )

    for class_name in classes:

        probability = probability_dict.get(
            str(class_name),
            0.0,
        )

        print(
            f"  {str(class_name):<28}"
            f"{probability * 100:6.2f}%"
        )

    # ========================================================
    # FEATURES
    # ========================================================

    print(
        "\nImage Quality Features:"
    )

    for name, value in base_features.items():

        try:

            print(
                f"  {name:<35}"
                f"{float(value):.4f}"
            )

        except (
            TypeError,
            ValueError,
        ):

            print(
                f"  {name:<35}"
                f"{value}"
            )

    # ========================================================
    # ISSUES
    # ========================================================

    print(
        "\nDetected Issues:"
    )

    if not issues:

        print(
            "  None"
        )

    else:

        for issue in issues:

            print(
                f"  {issue['type']:<30}"
                f"{issue['severity']:<10}"
                f"{issue['confidence'] * 100:.0f}%"
            )

    print(
        "=" * 70
    )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "prediction": final_prediction,

        "model_prediction": model_prediction,

        "override_reason": override_reason,

        "probabilities": probability_dict,

        "features": {
            str(name): float(value)
            for name, value
            in base_features.items()
        },

        "issues": issues,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            'python ml/training/predict_image.py '
            '"path/to/image.jpg"'
        )

        return

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.exists():

        print(
            f"\nERROR: Image not found:\n"
            f"{image_path}"
        )

        return

    try:

        predict_image(
            image_path
        )

        print(
            "\nPrediction completed successfully."
        )

    except Exception as error:

        print(
            "\nERROR:"
        )

        print(
            error
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()