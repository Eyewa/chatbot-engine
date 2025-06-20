# Chatbot Engine

This project provides a lightweight FastAPI service powered by LangChain and LangServe. It exposes a `/chatbot/invoke` endpoint that proxies requests to a LangChain agent capable of answering customer queries.

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file with your configuration (see the provided template).
3. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
4. Open `http://localhost:8000/docs` for interactive API docs.
