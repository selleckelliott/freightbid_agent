FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt psycopg2-binary>=2.9.10

COPY . .

ENV PYTHONPATH=/app
EXPOSE 8000

# Phase 7.4: drop privileges — run as a non-root user that owns the app dir.
RUN useradd --create-home --uid 10001 freight && chown -R freight:freight /app
USER freight

# Phase 7.4: container liveness probe hits the /health endpoint (no curl in the slim image).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "adapters.inbound.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
