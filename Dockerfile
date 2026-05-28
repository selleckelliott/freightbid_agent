FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt psycopg2-binary>=2.9.10

COPY . .

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "adapters.inbound.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
