# Validation and deployment readiness

Validated locally on Windows with Python 3.11.16. This is an educational software demonstration, not a medically validated system. No clinical-performance benchmark has been established.

## Recovered project

Archive and GitHub baseline: `74d268a4e34ea02b18aa99b7f5989098743f2725`. The FAISS binary and document pickle match the archive byte-for-byte; the processed CSV matches after newline normalization. These artifacts are unchanged by preparation.

The old LangChain 0.2 environment failed to load the Pydantic 2 document pickle (`__fields_set__`). The archive-compatible LangChain 0.3/Pydantic 2 environment fixes this. MiniLM, FAISS top-four retrieval and the original LangChain stuff-documents/Gemini generation design are retained. Retrieval now finishes before generation so its evidence can survive provider errors.

Measured facts: 4,966 CSV records, 29,598 chunks, 384 vector dimensions. Three regenerated chunk vectors matched the saved vectors to maximum absolute errors below 1.2e-7. This is a compatibility check, not an answer-quality metric.

## Checks completed

- Four regression tests pass, including subcases for 503, quota/rate limiting, timeout/deadline and permission errors.
- Successful generation/recovery is tested deterministically with a fake LLM. The real retrieval chain formats the preserved documents correctly.
- Provider errors retain the question, four documents and distances. Retry uses the same evidence, performs no new retrieval and has no uncontrolled loop.
- Editing an unsubmitted question cannot silently change the evidence used by retry.
- Blank/oversized input is rejected. Missing Gemini settings still permit retrieval.
- Synthetic credential/error markers are absent from rendered UI output. Runtime logs include exception types only; provider retry logging is suppressed.
- Actual local Streamlit page loads; blank input warning, example retrieval, full source expansion and explicit retry were exercised in the browser.
- The browser encountered real Gemini 503 responses. Both initial failure and retry preserved the sources and displayed the concise fallback message.
- Earlier real Gemini 3.1 Flash-Lite smoke tests produced a supported answer with retrieved record IDs and an insufficient-sources answer for bicycle repair. These two questions are not a benchmark. Later provider requests were intermittently unavailable; successful real generation is not claimed for the final outage test.
- Unrelated input still has nearest neighbors. Distances are shown as squared L2 values, not confidence scores; the UI warns that retrieval alone does not establish relevance.
- Python compilation, dependency consistency and Linux x86-64/Python 3.11 dependency resolution pass. No local Linux runtime is installed, so a Linux execution test is reserved for the actual deployment.

Run regression tests with `python -m unittest discover -s tests -v`. They require no real API key or network access. The old `test.py` remains only an import smoke check.

## Resources

| Item | Observed size/use |
|---|---:|
| Tracked files including existing artifacts | 104,822,755 bytes (about 105 MB) |
| Existing Git pack | 41.05 MiB |
| FAISS index | 45,462,573 bytes |
| Document metadata pickle | 25,065,683 bytes |
| Processed CSV (not read at runtime) | 34,280,685 bytes |
| Downloaded embedding cache | 91,578,455 bytes |
| Fresh-process load using downloaded cache | 10.14 seconds |
| Retrieval process working set after load | 529.85 MiB |
| Retrieval process working set after query | 577.65 MiB |
| Actual Streamlit server after requests | 646.61 MiB; observed peak 646.68 MiB |

Measurements are Windows observations, not guaranteed cloud requirements. Startup UI loads before the encoder/index; first query loads them. A first-ever query downloads pinned MiniLM artifacts. Repeated queries reuse the cached store; a process restart reloads it and a new host may download again. Offline loading with the completed cache passed.

## Security and remaining cloud checks

History scan found no secrets. Scans of project source exclude private configuration and third-party environments; no confirmed exposed credential was found. `.env`, `.streamlit/secrets.toml`, virtual environments, model caches and `.audit/` are excluded from publication. No original archive file was edited. Pickle loading is restricted to the repository artifacts after checksum verification; user-uploaded indexes are not accepted.

Ready for owner review and a Streamlit Community Cloud deployment attempt. Remaining hosting checks are the actual Linux build, cloud memory, cold start/restart behavior, public source/retry rendering and the live URL. Intermittent Gemini availability is handled by the retrieval-only fallback and does not block source preparation. No cloud deployment is claimed yet.
