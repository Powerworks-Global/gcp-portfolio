FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# BUILD-TIME index, not runtime. This pulls the embedding weights once and writes
# the finished vector index next to the code, so retrieval reads files on disk and
# the container never downloads a model or calls a hosted vector database.
RUN python ingest.py

# After the build, nothing is allowed to reach the model hub: if a weight goes
# missing the image fails to build instead of failing on a cold-start query.
ENV HF_HUB_OFFLINE=1

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
