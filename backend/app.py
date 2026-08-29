from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sqlite3
import shutil
import sys
from datetime import datetime
from pathlib import Path
import uuid


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make project root available for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.predict_image import predict_image


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Image Quality Assessment API",
    description="ML-powered image quality assessment system",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

UPLOAD_DIR = PROJECT_ROOT / "backend" / "uploads"
DB_PATH = PROJECT_ROOT / "backend" / "analysis.db"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATABASE
# ============================================================

def init_db():

    conn = sqlite3.connect(DB_PATH)

    try:

        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                quality_label TEXT NOT NULL,
                quality_score REAL NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()

    finally:

        conn.close()


init_db()


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality_score(
    prediction,
    probabilities
):
    """
    Convert model probabilities into a 0-100
    image quality score.

    ACCEPTABLE probability = quality score.

    Examples:

    ACCEPTABLE = 0.84
        -> 84.0

    ACCEPTABLE = 0.01
        -> 1.0

    ACCEPTABLE = 0.50
        -> 50.0
    """

    if not probabilities:
        return 0.0

    acceptable_probability = float(
        probabilities.get(
            "ACCEPTABLE",
            0.0
        )
    )

    score = acceptable_probability * 100.0

    # Keep score between 0 and 100
    score = max(
        0.0,
        min(100.0, score)
    )

    return round(
        score,
        1
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "Image Quality Assessment API"
    }


# ============================================================
# ANALYZE IMAGE
# ============================================================

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...)
):

    # ========================================================
    # VALIDATE FILE TYPE
    # ========================================================

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg"
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported."
        )


    # ========================================================
    # VALIDATE FILENAME
    # ========================================================

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )


    # ========================================================
    # SAFE ORIGINAL FILENAME
    # ========================================================

    original_filename = Path(
        file.filename
    ).name


    # ========================================================
    # CREATE UNIQUE FILE
    # ========================================================

    unique_filename = (
        f"{uuid.uuid4().hex}_"
        f"{original_filename}"
    )

    file_path = (
        UPLOAD_DIR
        / unique_filename
    )


    try:

        # ====================================================
        # SAVE UPLOADED IMAGE
        # ====================================================

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # ====================================================
        # CHECK FILE
        # ====================================================

        if not file_path.exists():

            raise HTTPException(
                status_code=400,
                detail="Uploaded file could not be saved."
            )


        if file_path.stat().st_size == 0:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )


        # ====================================================
        # RUN ML MODEL
        # ====================================================

        print()
        print("=" * 60)
        print("RUNNING IMAGE QUALITY ANALYSIS")
        print("=" * 60)

        result = predict_image(
            file_path
        )


        # ====================================================
        # GET PREDICTION
        # ====================================================

        quality_label = str(
            result.get(
                "prediction",
                "UNKNOWN"
            )
        )


        # ====================================================
        # GET PROBABILITIES
        # ====================================================

        probabilities = result.get(
            "probabilities",
            {}
        )

        probabilities = {
            str(key): float(value)
            for key, value
            in probabilities.items()
        }


        # ====================================================
        # QUALITY SCORE
        # ====================================================

        # Always calculate from ACCEPTABLE probability.
        #
        # This prevents the frontend from receiving 0.0
        # when predict_image.py does not return quality_score.

        quality_score = calculate_quality_score(
            quality_label,
            probabilities
        )


        # ====================================================
        # MODEL CONFIDENCE
        # ====================================================

        if probabilities:

            confidence = (
                max(probabilities.values())
                * 100.0
            )

        else:

            confidence = 0.0


        confidence = round(
            confidence,
            1
        )


        # ====================================================
        # GET ISSUES
        # ====================================================

        issues = result.get(
            "issues",
            []
        )


        # ====================================================
        # GET FEATURES
        # ====================================================

        features = result.get(
            "features",
            {}
        )


        # ====================================================
        # ORIGINAL MODEL PREDICTION
        # ====================================================

        model_prediction = str(
            result.get(
                "model_prediction",
                quality_label
            )
        )


        # ====================================================
        # PRINT RESULT
        # ====================================================

        print()
        print("-" * 60)
        print(
            f"MODEL PREDICTION : {model_prediction}"
        )
        print(
            f"FINAL PREDICTION : {quality_label}"
        )
        print(
            f"QUALITY SCORE    : {quality_score:.1f}/100"
        )
        print(
            f"CONFIDENCE       : {confidence:.1f}%"
        )
        print("-" * 60)


        # ====================================================
        # SAVE TO DATABASE
        # ====================================================

        conn = sqlite3.connect(
            DB_PATH
        )

        try:

            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO analyses
                (
                    filename,
                    quality_label,
                    quality_score,
                    confidence,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                original_filename,
                quality_label,
                quality_score,
                confidence,
                datetime.now().isoformat()
            ))

            analysis_id = cursor.lastrowid

            conn.commit()

        finally:

            conn.close()


        # ====================================================
        # RETURN TO FRONTEND
        # ====================================================

        return {

            "id": analysis_id,

            "filename": original_filename,

            "quality_label": quality_label,

            "quality_score": quality_score,

            "confidence": confidence,

            "probabilities": probabilities,

            "issues": issues,

            "features": features,

            "model_prediction": model_prediction
        }


    except HTTPException:

        raise


    except Exception as error:

        print()
        print("=" * 60)
        print("IMAGE ANALYSIS ERROR")
        print("=" * 60)
        print(error)
        print("=" * 60)

        raise HTTPException(
            status_code=500,
            detail=(
                "Image analysis failed: "
                f"{str(error)}"
            )
        )


    finally:

        try:

            await file.close()

        except Exception:

            pass


# ============================================================
# GET ANALYSIS HISTORY
# ============================================================

@app.get("/analyses")
def get_analyses():

    conn = None

    try:

        conn = sqlite3.connect(
            DB_PATH
        )

        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                id,
                filename,
                quality_label,
                quality_score,
                confidence,
                created_at
            FROM analyses
            ORDER BY id DESC
        """)

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not load analysis history: "
                f"{str(error)}"
            )
        )


    finally:

        if conn is not None:

            conn.close()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "message":
            "Image Quality Assessment API is running.",

        "docs":
            "/docs",

        "health":
            "/health",

        "analyze":
            "/analyze",

        "history":
            "/analyses"
    }