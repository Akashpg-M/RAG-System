FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/data/uploads
CMD ["python", "-m", "uvicorn", "src.api.app:create_api", "--factory", "--host", "0.0.0.0", "--port", "8000"]
