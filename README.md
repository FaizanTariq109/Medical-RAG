# Medical QA RAG

An educational Streamlit application that retrieves passages from sample medical transcriptions and uses Google Gemini to explain the retrieved material with record references.

**Author:** Faizan Tariq. Individual university project, developed for Generative AI in semester 7.

**Status:** local retrieval and real Gemini smoke tests verified; failure/retry regression checks pass. Deployed on Streamlit Community Cloud and tested by the project owner.

**Live Demo:** [Medical QA RAG](https://faizan-medical-rag.streamlit.app/)

## What it does

- Embeds questions locally with `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieves four chunks using the original FAISS index.
- Supplies retrieved excerpts to Gemini through the original LangChain stuff-documents generation chain.
- Requests record citations and an explicit insufficient-sources answer when the excerpts do not support a response.
- Displays full retrieved excerpts with titles, specialties and corpus record IDs.
- Supports downloadable answers, bounded questions and a shared generation cooldown.
- Preserves the submitted question, record IDs, full excerpts and retrieval distances when Gemini fails.
- Offers **Retry answer**, reusing that evidence without retrieving again or requiring re-entry.

The citation/abstention prompt is a preparation fix; it is not evidence that every generated answer is correct or properly grounded.

## Architecture

`Question → MiniLM embedding → FAISS retrieval → top-k clinical transcription chunks → Gemini → answer with source references`

The original LangChain/FAISS/Gemini design and recovered index are preserved. `app.py` provides the Streamlit UI; `rag_runtime.py` handles verified index loading and the retrieval chain. Streamlit caches the CPU encoder/index and answer service. A per-service lock serializes generation and a ten-second cooldown reduces accidental repeated calls. This is not a distributed abuse-prevention system.

The saved index contains **29,598 vectors with 384 dimensions**. The processed CSV contains **4,966 rows**. Three sampled index vectors match embeddings regenerated from their saved chunks to a maximum absolute difference below 1.2e-7. These checks verify artifact compatibility, not answer quality.

## Corpus and provenance

The original project attributes the corpus to [Tara Boyle's Medical Transcriptions dataset on Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions), described by its publisher as sample transcriptions scraped from MTSamples. Kaggle lists CC0 for that dataset. The repository's MIT license applies to its code; third-party corpus attribution is retained separately and the underlying text has not undergone an independent privacy or rights audit.

The archived and GitHub FAISS files match byte-for-byte. Their CSV contents match after newline normalization. The archive changes the Gemini model and dependency versions, while the existing working copy has useful path/configuration repairs. The prepared version combines those fixes; it does not rebuild or retrain the original artifacts.

No index-building notebook or preprocessing script was found in the project. The original preprocessing/chunking pipeline cannot currently be reproduced from source. Stored chunk lengths range from 14 to 1,000 characters; this observation does not establish the original splitting parameters. Multiple retrieved chunks may come from the same record.

## Run locally

Use Python 3.11:

```sh
python -m venv .venv-medical
# PowerShell:
.venv-medical\Scripts\Activate.ps1
# Linux/macOS:
# source .venv-medical/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Create a local `.env` using `.env.example`, then set:

```dotenv
GOOGLE_API_KEY=your-own-key
GEMINI_MODEL=gemini-3.1-flash-lite
```

The app loads `.env` from its own folder. On Streamlit Community Cloud, use the same names as root-level Secrets keys. Environment variables take precedence. Never commit the key or local secrets file.

The model ID is deliberately configurable. Use a text-generation model available to your API project and free-tier quota; availability can change. The app uses the archive-compatible LangChain generation integration. It does not silently switch models or enable paid billing.

The embedding model is pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; its first load requires network access. The two original FAISS artifacts already exist in Git history and are retained. Their hashes are verified against `artifact-manifest.json` before loading trusted pickle metadata. Never replace these with untrusted uploaded indexes.

## Provider availability and retry

Retrieval completes independently of generation. Gemini 503 errors, rate limits, timeouts and service/quota errors display a concise notice while preserving source evidence. An explicit **Retry answer** action reruns only generation for the stored question. Editing the input does not silently change the question associated with existing results; submit a new question to retrieve new sources.

There is no background retry loop. The pinned adapter can make at most two RPC attempts per action; each has a 45-second timeout and transport-level retries are disabled. A per-process lock and ten-second cooldown also apply. Missing Gemini configuration still permits retrieval. A genuine retrieval failure is reported separately and is never presented as a provider failure.

## Privacy and limitations

Questions and four retrieved excerpts are sent to Google Gemini. [Google's free-tier policy](https://ai.google.dev/gemini-api/docs/pricing) permits use of content to improve products. Use general, non-personal questions only; do not enter private health records or identifiers. The application does not write questions or answers to disk, but that does not describe the provider's retention policy.

Sample transcriptions describe individual cases, not clinical guidelines. Responses may contain unsupported claims, omit context or miscite records. The prompt requests abstention, but FAISS always returns nearest neighbors, including for unrelated questions. Retrieval distance is not a calibrated confidence score. No accuracy, diagnosis, treatment, clinical safety or benchmark performance claim is made.

Free API quotas may make the demo temporarily unavailable. Input is limited to 1,000 characters; generation is bounded to 768 output tokens with a 45-second per-attempt RPC timeout (the adapter may retry once). Slow embedding downloads and cloud memory limits need deployment testing.

No clinical-performance benchmark has been established. The two real-provider questions are smoke tests, not a benchmark.

## Validation and deployment

See [VALIDATION.md](VALIDATION.md) for completed checks and their limits, and [DEPLOYMENT.md](DEPLOYMENT.md) for account and hosting steps.

The earlier README described absent rebuilding/evaluation files and unsupported metrics. Those are not advertised as working features. `test.py` is an import smoke check, not a benchmark.

## Cloud resource preparation

Entrypoint: `app.py`; requirements: root `requirements.txt`; Python: **3.11**. No custom `.streamlit/config.toml` is required. Configure `GOOGLE_API_KEY` and `GEMINI_MODEL="gemini-3.1-flash-lite"` in Streamlit Secrets.

The tracked source/artifacts total approximately 104.8 MB; the Git pack is 41.05 MiB. The FAISS index is 45.46 MB, its document metadata is 25.07 MB, and the CSV is 34.28 MB. These files already exist in repository history and are unchanged. Runtime does not read the CSV. The downloaded embedding cache measured 91.58 MB. Environments, local caches, credentials and audit files are excluded.

A fresh Windows process using the downloaded embedding cache loaded the encoder/index in 10.14 seconds. Working set was about 530 MiB after loading and 578 MiB after retrieval. These are local measurements, not cloud guarantees. Linux x86-64/Python 3.11 dependency resolution passed; a Linux runtime was unavailable locally, so the actual Linux build and cloud resource behavior must be verified on Streamlit.

The first hosted query downloads the pinned public embedding model; later queries reuse Streamlit's resource cache. Restarting a process reloads the model/index; a fresh host/cache downloads the embedding artifacts again. Model download or retrieval failures remain distinct from generation outages.

## Tests

```sh
python -m unittest discover -s tests -v
```

Regression tests simulate provider failures and recovery without using a real key or making network requests. They check evidence preservation, retry reuse, blank input, missing configuration and sanitized errors. Real-provider smoke-test observations are summarized in VALIDATION.md; local audit files are not published.
