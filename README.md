# Chatbot Engine

This project provides a lightweight FastAPI service powered by LangChain and LangServe. It exposes a `/chatbot/invoke` endpoint that proxies requests to a LangChain agent capable of answering customer queries.

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your configuration values.
3. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
4. Open `http://localhost:8000/docs` for interactive API docs.
