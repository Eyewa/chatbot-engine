# Chatbot Engine

This project provides a lightweight FastAPI service powered by LangChain and LangServe. It exposes a `/chat` endpoint that proxies requests to a LangChain agent capable of answering customer queries.

Each request may optionally include a `conversationId` which is used to retrieve previous questions and answers from the database. When supplied, the service appends the retrieved history to the chat request and also persists the new interaction for future use. This history can later be leveraged as feedback/training data or for retrieval‑augmented generation (RAG).

## Features

- Schema-aware SQL validation prevents the agent from querying unknown columns.
- Responses merge data from multiple sources and return partial results when only one succeeds.

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your configuration values. The
   template includes placeholders for variables like `OPENAI_API_KEY`,
   `SQL_DATABASE_URI_LIVE` and `SQL_DATABASE_URI_COMMON`.
3. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
4. Open `http://localhost:8000/docs` for interactive API docs.

## Running Tests

Unit tests are written with `pytest`. Install dev requirements and run:

```bash
pytest
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
