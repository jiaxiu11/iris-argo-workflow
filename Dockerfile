FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir scikit-learn==1.4.2 pandas==2.2.2

COPY src/ .
