FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    pandas \
    numpy \
    scikit-learn \
    joblib \
    opencv-python-headless

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]