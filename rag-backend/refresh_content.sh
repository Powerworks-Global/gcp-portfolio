#!/usr/bin/env bash
# Rebuild the local vector index from data/ inside a pinned container.
#
# You usually do not need this: the Dockerfile runs `python ingest.py` on every image
# build, so a committed change under data/ reaches Cloud Run through the normal
# pipeline. Use this to check a corpus edit, or to inspect the index, before pushing.
set -euo pipefail

cd "$(dirname "$0")"

echo "Building the local index from data/ ..."

# No credentials and no hosted vector database: embedding runs locally and the index
# lands in ./index_store, which is build output and gitignored on purpose.
docker run --rm \
  -v "$(pwd)":/app \
  -w /app \
  python:3.11-slim sh -c "pip install -q -r requirements.txt && python ingest.py"

echo "Index written to ./index_store — it is rebuilt into the image on the next pipeline run."
