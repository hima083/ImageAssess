from pathlib import Path
import csv
import random
import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "natural_images"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "data" / "processed"

SEED = 42
IMAGES_PER_CATEGORY = 200

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CATEGORIES = [
    "airplane",
    "car",
    "cat",
    "dog",
    "flower",
    "fruit",
    "motorbike",
    "person",
]

# These are the actual quality conditions used by our model.
CONDITIONS = [
    "clean",
    "blur",
    "underexposure",
    "overexposure",
    "noise",
    "severe_degradation",
    "visual_artifact",
]


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# Quality-label mapping
# ============================================================

def get_quality_label(condition):
    """
    Overall quality label used for the first ML task.

    ACCEPTABLE:
        No intentional degradation.

    DEGRADED:
        A quality problem is present, but the image remains usable.

    POTENTIALLY_DEFECTIVE:
        Severe degradation or a localized visual artifact is present.
    """

    if condition == "clean":
        return "ACCEPTABLE"

    if condition in {
        "blur",
        "underexposure",
        "overexposure",
        "noise",
    }:
        return "DEGRADED"

    return "POTENTIALLY_DEFECTIVE"


def get_severity(condition):
    if condition == "clean":
        return "none"

    if condition in {
        "blur",
        "underexposure",
        "overexposure",
        "noise",
    }:
        return "medium"

    return "high"


# ============================================================
# Controlled degradation functions
# ============================================================

def apply_blur(image):
    """Simulate insufficient sharpness."""
    return cv2.GaussianBlur(image, (11, 11), 0)


def apply_underexposure(image):
    """Reduce image brightness."""
    result = image.astype(np.float32) * 0.45
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_overexposure(image):
    """Increase image brightness and create clipped highlights."""
    result = image.astype(np.float32) * 1.65
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_noise(image):
    """Add Gaussian sensor-like noise."""
    image_float = image.astype(np.float32)

    noise = np.random.normal(
        loc=0,
        scale=30,
        size=image.shape
    )

    result = image_float + noise

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_severe_degradation(image):
    """
    Combine several strong quality degradations to represent
    severe image degradation.
    """
    result = apply_blur(image)
    result = apply_underexposure(result)
    result = apply_noise(result)

    return result


def apply_visual_artifact(image):
    """
    Introduce a localized block-like visual artifact.

    This is used as a controlled proxy for a potential
    visual defect. It is explicitly documented as a
    synthetic artifact rather than a real-world defect label.
    """
    result = image.copy()

    height, width = result.shape[:2]

    x1 = int(width * 0.30)
    y1 = int(height * 0.30)
    x2 = int(width * 0.70)
    y2 = int(height * 0.70)

    # Strong local corruption.
    block = result[y1:y2, x1:x2]

    if block.size > 0:
        block = cv2.resize(block, (20, 20))
        block = cv2.resize(
            block,
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_NEAREST
        )

        result[y1:y2, x1:x2] = block

    return result


def apply_condition(image, condition):
    if condition == "clean":
        return image.copy()

    if condition == "blur":
        return apply_blur(image)

    if condition == "underexposure":
        return apply_underexposure(image)

    if condition == "overexposure":
        return apply_overexposure(image)

    if condition == "noise":
        return apply_noise(image)

    if condition == "severe_degradation":
        return apply_severe_degradation(image)

    if condition == "visual_artifact":
        return apply_visual_artifact(image)

    raise ValueError(f"Unknown condition: {condition}")


# ============================================================
# Source-image collection
# ============================================================

def collect_source_images():
    all_sources = []

    for category in CATEGORIES:
        category_dir = SOURCE_DIR / category

        if not category_dir.exists():
            raise FileNotFoundError(
                f"Category directory not found: {category_dir}"
            )

        images = [
            path
            for path in category_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        images.sort()

        if len(images) < IMAGES_PER_CATEGORY:
            raise ValueError(
                f"{category} contains only {len(images)} images, "
                f"but {IMAGES_PER_CATEGORY} are required."
            )

        # Deterministic sampling.
        category_rng = random.Random(SEED + CATEGORIES.index(category))
        selected = category_rng.sample(
            images,
            IMAGES_PER_CATEGORY
        )

        for path in selected:
            all_sources.append({
                "source_path": path,
                "category": category
            })

    return all_sources


# ============================================================
# Source-level train/validation/test split
# ============================================================

def split_sources(sources):
    """
    IMPORTANT:
    Splitting happens BEFORE degradation generation.

    Therefore the same original source image can never
    appear in both training and testing.
    """

    split_data = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for category in CATEGORIES:

        category_sources = [
            item
            for item in sources
            if item["category"] == category
        ]

        rng = random.Random(
            SEED + 100 + CATEGORIES.index(category)
        )

        rng.shuffle(category_sources)

        total = len(category_sources)

        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        split_data["train"].extend(
            category_sources[:train_end]
        )

        split_data["validation"].extend(
            category_sources[train_end:val_end]
        )

        split_data["test"].extend(
            category_sources[val_end:]
        )

    return split_data


# ============================================================
# Metadata generation
# ============================================================

def create_metadata(split_data):
    rows = []

    source_counter = 0

    for split_name, sources in split_data.items():

        for item in sources:

            source_counter += 1

            source_path = item["source_path"]
            category = item["category"]

            source_id = f"S{source_counter:05d}"

            for condition in CONDITIONS:

                rows.append({
                    "source_id": source_id,
                    "source_category": category,
                    "source_path": str(
                        source_path.relative_to(PROJECT_ROOT)
                    ),
                    "split": split_name,
                    "condition": condition,
                    "severity": get_severity(condition),
                    "quality_label": get_quality_label(condition),
                })

    return rows


# ============================================================
# Save metadata
# ============================================================

def save_metadata(rows):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path = OUTPUT_DIR / "dataset_metadata.csv"

    fieldnames = [
        "source_id",
        "source_category",
        "source_path",
        "split",
        "condition",
        "severity",
        "quality_label",
    ]

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    return metadata_path


# ============================================================
# Create sample degraded images for inspection
# ============================================================

def create_sample_images(split_data):

    sample_dir = PROJECT_ROOT / "sample_images"

    sample_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Use one test image so these examples are based on
    # an unseen source image.
    test_source = split_data["test"][0]["source_path"]

    image = cv2.imread(
        str(test_source)
    )

    if image is None:
        raise ValueError(
            f"Could not read sample image: {test_source}"
        )

    for condition in CONDITIONS:

        degraded = apply_condition(
            image,
            condition
        )

        output_path = (
            sample_dir /
            f"sample_{condition}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            degraded
        )


# ============================================================
# Dataset summary
# ============================================================

def print_summary(split_data, rows):

    print("\n" + "=" * 60)
    print("IMAGEASSESS DATASET SUMMARY")
    print("=" * 60)

    print(f"Source images selected : {sum(len(v) for v in split_data.values())}")

    for split_name in ["train", "validation", "test"]:
        print(
            f"{split_name.capitalize():<22}: "
            f"{len(split_data[split_name])}"
        )

    print(
        f"\nQuality conditions      : {len(CONDITIONS)}"
    )

    print(
        f"Generated metadata rows : {len(rows)}"
    )

    print("\nConditions:")

    for condition in CONDITIONS:
        count = sum(
            1 for row in rows
            if row["condition"] == condition
        )

        print(
            f"  {condition:<22}: {count}"
        )

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    print("Starting ImageAssess dataset preparation...")

    print("\n1. Collecting source images...")
    sources = collect_source_images()

    print(
        f"   Selected {len(sources)} clean source images."
    )

    print("\n2. Splitting source images...")
    split_data = split_sources(sources)

    print("\n3. Creating degradation metadata...")
    rows = create_metadata(split_data)

    print("\n4. Saving metadata...")
    metadata_path = save_metadata(rows)

    print(
        f"   Metadata saved to: {metadata_path}"
    )

    print("\n5. Creating sample degraded images...")
    create_sample_images(split_data)

    print(
        "   Sample images saved in: sample_images/"
    )

    print_summary(
        split_data,
        rows
    )

    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    main()