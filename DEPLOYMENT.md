# Medical QA RAG deployment

Target: Streamlit Community Cloud, after the owner approves the prepared source changes. No live URL exists yet.

1. Configure a free Gemini API key locally in the ignored `.env`. The locally tested model is `gemini-3.1-flash-lite`. Do not enter credentials in chat or Git.
2. Completed locally: recovered-index/embedding checks, successful real Gemini smoke tests, and deterministic provider-failure/retry tests. Intermittent Gemini 503 responses are handled by showing sources and an explicit retry action; they are not a reason to rewrite the pipeline.
3. Review the exact source changes, exclusions and commit message with the owner before pushing to `FaizanTariq109/Medical-RAG`.
4. The owner signs in to Streamlit Community Cloud. Create an app from repository `FaizanTariq109/Medical-RAG`, branch `main`, main file `app.py`, Python **3.11**.
5. In Advanced settings → Secrets, enter root-level `GOOGLE_API_KEY` and the tested `GEMINI_MODEL="gemini-3.1-flash-lite"`. A public demo consumes the owner's shared free quota. Keep billing disabled if free-only behavior is required; the application itself is not a billing cap.
6. Deploy and inspect installation logs. Requirements use CPU-only PyTorch. The index is already in GitHub; the embedding model downloads from a pinned Hugging Face revision on first use. No new model-storage account is required.
7. Verify the public UI and actual generation, full source excerpts, blank input and an unrelated question. Check handling of rate limits and temporary provider failure.
8. Test a reboot/cold start through the owner's Manage app controls. Observe memory and download behavior; do not infer cloud suitability solely from local results.
9. Record the public URL and completed checks before linking the demo from the portfolio.

The prompt encourages source citations and abstention but does not guarantee clinical correctness. The public UI must retain the educational scope and notice that questions/excerpts go to Google; do not solicit personal health information.

References: [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [Gemini models](https://ai.google.dev/gemini-api/docs/models), [Gemini quotas](https://ai.google.dev/gemini-api/docs/rate-limits), [Gemini pricing/data use](https://ai.google.dev/gemini-api/docs/pricing).


No custom Streamlit config file or operating-system package list is required by the tested dependency resolution. Linux runtime, hosted cold start, restart behavior and available memory are final deployment checks. The CSV and index are already tracked; do not upload local environments, .audit, embedding caches, .env or secrets.toml.
