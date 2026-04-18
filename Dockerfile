FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src
ENV DATA_RAW=/app/data/raw
ENV DATA_PROCESSED=/app/data/processed
ENV MODEL_DIR=/app/models
ENV RANDOM_STATE=42
ENV TEST_SIZE=0.2
ENV FRAUD_CLASS=1
ENV TARGET_COLUMN=Class

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY scripts/ scripts/
RUN mkdir -p data/raw data/processed models

CMD ["python", "scripts/train.py", "--model", "rf"]
