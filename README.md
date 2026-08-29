# ImageAssess – Image Quality Assessment System

ImageAssess is a machine-learning based image quality assessment application that analyzes an uploaded image and identifies common visual quality problems.

The system evaluates image features such as sharpness, brightness, contrast, noise, entropy, saturation and color information, then predicts the overall image quality.

## Project Objective

The main objective of ImageAssess is to automatically determine whether an image is:

- ACCEPTABLE
- DEGRADED
- POTENTIALLY_DEFECTIVE

The system also detects specific image quality issues such as:

- Blur
- Noise
- Underexposure
- Overexposure
- Potential visual defects
- Severe degradation

## Main Features

- Image quality feature extraction
- Machine-learning classification using Random Forest
- Quality override rules for important visual defects
- Quality score generation
- Confidence/probability reporting
- Image upload through a web interface
- Fast image analysis through a FastAPI backend
- Recent analysis/audit history
- Support for JPG, JPEG and PNG images
- Command-line image prediction for testing

## Technologies Used

### Backend
- Python
- FastAPI
- Uvicorn

### Machine Learning
- Scikit-learn
- Random Forest Classifier
- Joblib
- Pandas
- NumPy
- OpenCV

### Frontend
- HTML
- CSS
- JavaScript

### Storage
- SQLite

## Project Structure

```text
ImageAssess/
│
├── backend/
│   ├── app.py
│   ├── analysis.db
│   └── uploads/
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── ml/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   │
│   ├── features/
│   │   └── extract_features.py
│   │
│   ├── models/
│   │   └── random_forest.joblib
│   │
│   └── training/
│       ├── generate_dataset.py
│       ├── train_model.py
│       ├── evaluate_model.py
│       └── predict_image.py
│
├── sample_images/
│   ├── sample_clean.jpg
│   ├── sample_blur.jpg
│   ├── sample_noise.jpg
│   ├── sample_overexposure.jpg
│   ├── sample_underexposure.jpg
│   ├── sample_severe_degradation.jpg
│   └── sample_visual_artifact.jpg
│
├── docs/
│   ├── project_decisions.md
│   └── evaluation/
│       ├── confusion_matrix.png
│       ├── evaluation_report.txt
│       └── feature_importance.csv
│
└── README.md

How the System Works

The system follows these steps:

User uploads an image.
The backend receives the image.
Image quality features are extracted.
The trained Random Forest model predicts the image quality.
Quality override rules check important defects such as blur, noise and exposure problems.
The final quality classification is generated.
A quality score and confidence are displayed.
The analysis is stored in the audit history.
Image Quality Features

The feature extraction module evaluates features including:

Brightness mean
Brightness standard deviation
Dark pixel ratio
Bright pixel ratio
Contrast
Contrast percentile range
Laplacian variance
Edge density
Noise estimate
Entropy
Saturation statistics
Low saturation ratio
RGB channel means

These features are provided to the machine-learning model for classification.

Machine Learning Model

ImageAssess uses a Random Forest Classifier.

The trained model is stored at:

ml/models/random_forest.joblib

The model is trained using the processed image-quality feature dataset.

The training process uses a stratified train/test split so that the class distribution is maintained between training and testing data.

Model Evaluation

The project includes model evaluation outputs in:

docs/evaluation/

Available evaluation files include:

evaluation_report.txt
confusion_matrix.png
feature_importance.csv

The model training process also reports validation accuracy and balanced accuracy.

Training the Model

From the project root:

python ml/training/train_model.py

After successful training, the model is saved to:

ml/models/random_forest.joblib
Testing an Image from the Command Line

Run:

python ml/training/predict_image.py sample_images/sample_clean.jpg

For a file containing spaces, use quotes:

python ml/training/predict_image.py "sample_images/passport size - himaja.jpeg"

Example test images:

sample_clean.jpg
sample_blur.jpg
sample_noise.jpg
sample_overexposure.jpg
sample_underexposure.jpg
sample_severe_degradation.jpg
sample_visual_artifact.jpg
Running the Backend

Activate the virtual environment:

.venv\Scripts\Activate.ps1

Start the FastAPI application:

python -m uvicorn backend.app:app --reload

The API is then available locally.

API documentation:

/docs

Main analysis endpoint:

/analyze

Analysis history endpoint:

/analyses
Running the Frontend

Open:

frontend/index.html

The frontend provides:

Image upload
Image analysis
Quality result
Quality score
Detected issues
Recent analysis history
Example Results

The sample images were tested through the prediction system.

Clean Image
MODEL PREDICTION : ACCEPTABLE
FINAL PREDICTION : ACCEPTABLE
Blur Image
MODEL PREDICTION : DEGRADED
FINAL PREDICTION : DEGRADED
QUALITY OVERRIDE : blur
Noise Image
FINAL PREDICTION : DEGRADED
QUALITY OVERRIDE : noise
Underexposed Image
FINAL PREDICTION : DEGRADED
QUALITY OVERRIDE : underexposure
Overexposed Image
FINAL PREDICTION : DEGRADED
QUALITY OVERRIDE : overexposure
Severe Degradation
MODEL PREDICTION : POTENTIALLY_DEFECTIVE
FINAL PREDICTION : POTENTIALLY_DEFECTIVE
QUALITY OVERRIDE : severe_degradation
Web Application

The web application provides a simple interface where users can upload JPG, JPEG or PNG images and analyze their visual quality.

The application also maintains a Recent Analyses section that displays previous predictions.

Important Notes

The system is an image-quality assessment tool. Its predictions depend on the quality and distribution of the training data and the extracted image features.

The quality override rules provide additional deterministic checks for major image-quality problems.

Project Status

ImageAssess currently includes:

Machine-learning model
Feature extraction
Image prediction
Quality issue detection
FastAPI backend
Web frontend
Analysis history
Model evaluation outputs
Sample test images

The complete pipeline can be tested locally from image upload through final quality assessment.