FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict_service.py .
COPY best-indobert-sentiment-onnx ./best-indobert-sentiment-onnx

# Render injects PORT at runtime; predict_service.py reads it (falls back to 8000).
CMD ["python", "predict_service.py"]
