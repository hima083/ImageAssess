"""
ImageAssess - Image Quality Feature Extraction

Extracts interpretable computer-vision features from images
for training the image-quality classification model.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "dataset_metadata.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "image_quality_features.csv"
)


# ============================================================
# IMPORT DATASET GENERATOR
# ============================================================

# Add the PROJECT ROOT, not the training folder.
#
# This allows:
#
# from ml.training.generate_dataset import apply_condition
#
# to work correctly.

PROJECT_ROOT_STRING = str(PROJECT_ROOT)

if PROJECT_ROOT_STRING not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STRING)


try:

    from ml.training.generate_dataset import apply_condition

except ImportError as error:

    raise ImportError(
        "\nCould not import apply_condition from "
        "ml.training.generate_dataset.\n\n"
        f"Project root:\n{PROJECT_ROOT}\n\n"
        "Make sure this file exists:\n"
        "ml/training/generate_dataset.py\n\n"
        f"Original error: {error}"
    ) from error


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def calculate_entropy(gray):
    """
    Calculate grayscale image entropy.
    """

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram.flatten()

    histogram = (
        histogram
        / (histogram.sum() + 1e-10)
    )

    non_zero = histogram[
        histogram > 0
    ]

    entropy = -np.sum(
        non_zero * np.log2(non_zero)
    )

    return float(entropy)


# ============================================================
# NOISE ESTIMATION
# ============================================================

def estimate_noise(gray):
    """
    Estimate high-frequency image noise.

    Uses the difference between the original image
    and a median-filtered image.
    """

    denoised = cv2.medianBlur(
        gray,
        3
    )

    residual = (
        gray.astype(np.float32)
        - denoised.astype(np.float32)
    )

    return float(
        np.std(residual)
    )


# ============================================================
# EXTRACT IMAGE FEATURES
# ============================================================

def extract_features(image):
    """
    Extract interpretable image-quality features.

    Features include:

    - brightness
    - exposure
    - contrast
    - sharpness
    - edge density
    - noise
    - entropy
    - saturation
    - colour-channel statistics
    """

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )


    if len(image.shape) != 3:

        raise ValueError(
            "Expected a colour image."
        )


    # --------------------------------------------------------
    # Convert colour spaces
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    # ========================================================
    # BRIGHTNESS / EXPOSURE
    # ========================================================

    brightness_mean = float(
        np.mean(gray)
    )

    brightness_std = float(
        np.std(gray)
    )

    dark_pixel_ratio = float(
        np.mean(gray < 40)
    )

    bright_pixel_ratio = float(
        np.mean(gray > 215)
    )


    # ========================================================
    # CONTRAST
    # ========================================================

    contrast_std = float(
        np.std(gray)
    )

    p05 = np.percentile(
        gray,
        5
    )

    p95 = np.percentile(
        gray,
        95
    )

    contrast_percentile_range = float(
        p95 - p05
    )


    # ========================================================
    # SHARPNESS
    # ========================================================

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    laplacian_variance = float(
        laplacian.var()
    )


    # ========================================================
    # EDGES / TEXTURE
    # ========================================================

    edges = cv2.Canny(
        gray,
        threshold1=100,
        threshold2=200
    )

    edge_density = float(
        np.mean(edges > 0)
    )


    # ========================================================
    # NOISE
    # ========================================================

    noise_estimate = estimate_noise(
        gray
    )


    # ========================================================
    # ENTROPY
    # ========================================================

    entropy = calculate_entropy(
        gray
    )


    # ========================================================
    # SATURATION
    # ========================================================

    saturation_mean = float(
        np.mean(hsv[:, :, 1])
    )

    saturation_std = float(
        np.std(hsv[:, :, 1])
    )

    low_saturation_ratio = float(
        np.mean(
            hsv[:, :, 1] < 30
        )
    )


    # ========================================================
    # COLOUR CHANNEL STATISTICS
    # ========================================================

    blue_mean = float(
        np.mean(image[:, :, 0])
    )

    green_mean = float(
        np.mean(image[:, :, 1])
    )

    red_mean = float(
        np.mean(image[:, :, 2])
    )


    # ========================================================
    # RETURN FEATURES
    # ========================================================

    return {

        "brightness_mean":
            brightness_mean,

        "brightness_std":
            brightness_std,

        "dark_pixel_ratio":
            dark_pixel_ratio,

        "bright_pixel_ratio":
            bright_pixel_ratio,

        "contrast_std":
            contrast_std,

        "contrast_percentile_range":
            contrast_percentile_range,

        "laplacian_variance":
            laplacian_variance,

        "edge_density":
            edge_density,

        "noise_estimate":
            noise_estimate,

        "entropy":
            entropy,

        "saturation_mean":
            saturation_mean,

        "saturation_std":
            saturation_std,

        "low_saturation_ratio":
            low_saturation_ratio,

        "blue_mean":
            blue_mean,

        "green_mean":
            green_mean,

        "red_mean":
            red_mean,
    }


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset():

    # --------------------------------------------------------
    # Check metadata
    # --------------------------------------------------------

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"\nMetadata file not found:\n"
            f"{METADATA_PATH}\n\n"
            "Run generate_dataset.py first."
        )


    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(
        METADATA_PATH
    )


    print("=" * 60)
    print("IMAGEASSESS FEATURE EXTRACTION")
    print("=" * 60)

    print(
        f"Metadata rows found: {len(metadata)}"
    )


    feature_rows = []

    total = len(metadata)


    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for index, row in metadata.iterrows():

        source_path = (
            PROJECT_ROOT
            / str(row["source_path"])
        )


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            str(source_path)
        )


        if image is None:

            print(
                f"WARNING: Could not read "
                f"{source_path}"
            )

            continue


        # ----------------------------------------------------
        # Apply controlled degradation
        # ----------------------------------------------------

        processed_image = apply_condition(
            image,
            row["condition"]
        )


        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features = extract_features(
            processed_image
        )


        # ----------------------------------------------------
        # Store metadata
        # ----------------------------------------------------

        feature_row = {

            "source_id":
                row["source_id"],

            "source_category":
                row["source_category"],

            "split":
                row["split"],

            "condition":
                row["condition"],

            "severity":
                row["severity"],

            "quality_label":
                row["quality_label"],
        }


        # ----------------------------------------------------
        # Add extracted features
        # ----------------------------------------------------

        feature_row.update(
            features
        )


        feature_rows.append(
            feature_row
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (index + 1) % 500 == 0:

            print(
                f"Processed "
                f"{index + 1}/{total} images..."
            )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    features_df = pd.DataFrame(
        feature_rows
    )


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if features_df.empty:

        raise RuntimeError(
            "No features were extracted. "
            "Check your dataset paths and metadata."
        )


    # ========================================================
    # SAVE FEATURES
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    features_df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "FEATURE EXTRACTION COMPLETED"
    )

    print(
        "=" * 60
    )


    print(
        f"Rows generated : "
        f"{len(features_df)}"
    )

    print(
        f"Features       : "
        f"{len(features_df.columns)} columns"
    )

    print(
        f"Output         : "
        f"{OUTPUT_PATH}"
    )


    # ========================================================
    # QUALITY LABELS
    # ========================================================

    print(
        "\nQuality labels:"
    )

    print(
        features_df[
            "quality_label"
        ]
        .value_counts()
        .to_string()
    )


    # ========================================================
    # CONDITIONS
    # ========================================================

    print(
        "\nConditions:"
    )

    print(
        features_df[
            "condition"
        ]
        .value_counts()
        .to_string()
    )


    # ========================================================
    # SPLITS
    # ========================================================

    print(
        "\nSplits:"
    )

    print(
        features_df[
            "split"
        ]
        .value_counts()
        .to_string()
    )


    # ========================================================
    # FEATURE COLUMNS
    # ========================================================

    print(
        "\nFeature columns:"
    )


    metadata_columns = {
        "source_id",
        "source_category",
        "split",
        "condition",
        "severity",
        "quality_label",
    }


    feature_columns = [
        column
        for column in features_df.columns
        if column not in metadata_columns
    ]


    for column in feature_columns:

        print(
            f"  - {column}"
        )


    print(
        "=" * 60
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_dataset()