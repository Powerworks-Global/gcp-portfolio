Gotchas from building and deploying a FastAPI + LlamaIndex + Qdrant RAG backend to Google Cloud Run, integrated with a static frontend.

1. Ingestion vs. execution share a dependency, not just a data contract

The query engine (app/engine.py) only reads from Qdrant, but it still has to embed the incoming prompt with the same model the ingestion script (ingest.py) used to build the index. Skip llama-index-embeddings-openai / llama-index-llms-openai in the runtime environment and LlamaIndex fails silently on startup — surfaces later as a generic 500.

2. Don't fight the host runtime

Developing on a host with a bleeding-edge/unsupported Python (e.g. pre-release 3.14 on Fedora) breaks C-extension deps like pandas/pydantic. Rather than patching headers, isolate ingestion in a pinned container:

bash
docker run -it --rm -v "$(pwd)":/app -w /app \
  -e QDRANT_URL=... -e QDRANT_API_KEY=... -e OPENAI_API_KEY=... \
  python:3.11-slim sh -c "pip install -r requirements.txt llama-index-readers-file llama-index-embeddings-openai && python ingest.py"
3. New GCP projects cripple the default service account by design

gcloud run deploy --source . fails at the build step on fresh projects until the default compute SA ({PROJECT_NUMBER}-compute@developer.gserviceaccount.com) has roles/cloudbuild.builds.builder and roles/storage.objectViewer bound explicitly.

4. Keep the raw corpus out of the deployed service

SimpleDirectoryReader("data", recursive=True) belongs to the ingestion phase only. Production reads exclusively via the Qdrant connection string — the data/ folder never ships in the container image.

5. Cloud Run is private by default

A 403 from curl is Google's IAM layer, not your app. Fix with roles/run.invoker for allUsers, and pass --allow-unauthenticated in CI/CD every deploy — otherwise it silently reverts. If org policy blocks allUsers (constraints/iam.allowedPolicyMemberDomains), you have to enable unauthenticated invocations manually in the console.

6. Diagnose by error shape, not just status code
Error	Body	Meaning
403	HTML	Cloud Run IAM blocked it before your container ran
404	{"detail": "Not Found"}	Reached FastAPI, route doesn't exist — check /docs
500	{"detail": "..."}	Reached your code, uncaught exception (missing key, DB failure)
7. Qdrant client hangs on Cloud Run without explicit config

The client's automatic version-compatibility check can time out in serverless environments:

python
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key,
                       check_compatibility=False, timeout=10.0)

Also assert required env vars (OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY) at startup so failures name the missing variable in logs instead of surfacing as a generic pipeline error.

8. Test in this order

/docs (routing/schema) → curl (raw HTTP) → frontend integration (hostname mismatches are a frequent last-mile bug).

---

## Follow-up: the hosted dependencies died, so they were deleted

Items 1-8 above describe the original build, which kept its vectors in a free-tier Qdrant Cloud cluster and called OpenAI directly. That build no longer exists. What replaced it, and what the failure taught.

9. A free cloud dependency dies of disuse, and your repository never mentions it

The free Qdrant Cloud cluster was reclaimed after inactivity — not suspended, gone. Every query started failing on a dropped connection while the deployed service, the image and the pipeline all stayed green. Nothing under version control had changed, so the outage was invisible from CI: a dependency that a stranger can delete without a commit is a timer, not a service.

10. Removing it meant removing the network from the request path, not moving it

Retrieval is now LlamaIndex's built-in SimpleVectorStore read from a directory inside the image, and the prompt is embedded on CPU by fastembed over ONNX Runtime (BAAI/bge-small-en-v1.5, 384 dimensions). Generation is the only call that leaves the container, aimed at OpenRouter through the OpenAI schema — one credential, one bill. Nothing left on the request path can be dismantled from a distance. (Items 1, 4 and 7 above were properties of the hosted store this replaced.)

11. A local embedding model is only local if its weights ship in the image

fastembed pulls weights from the Hugging Face Hub on first use. Left alone, that is a cold-start network call with no retry — the original failure with a different URL. The Dockerfile runs ingest.py as a build step, which pulls the weights into model_cache/ inside the image, and the container runs with HF_HUB_OFFLINE=1, so a missing weight fails the build instead of the first query after a scale-from-zero.

12. An index generated from files that are already in the repo is a build artifact

The index is ~130 KB of JSON derived from committed prose, so it is not committed: the Dockerfile builds it, `.gitignore` ignores it, and a fresh clone still builds. Committing it would put the pair permanently out of step the first time someone edits a project page and forgets to re-run ingestion.

13. Cloud Run keeps env vars you stopped reading

The dead credentials were plaintext variables set on the service by hand, and a deploy copies the previous revision's configuration, so they survived every redeploy and every dependency change. Generation now arrives through `--set-secrets=OPENROUTER_API_KEY=<secret>:latest`, and the obsolete variables still have to be removed once by hand — `gcloud run services update rag-backend --region=europe-west1 --remove-env-vars=QDRANT_URL,QDRANT_API_KEY,OPENAI_API_KEY`.