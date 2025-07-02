# Chatbot Engine

A production-ready, modular FastAPI application for multi-agent chat, dynamic data routing, and comprehensive logging.

## Features
- Multi-agent orchestration
- Dynamic SQL and data routing
- FastAPI with automatic OpenAPI docs
- LangSmith tracing and token/cost monitoring
- Comprehensive chat logging (DB)
- Health and admin endpoints
- Modular, testable codebase

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit with your secrets
```

## Usage
```bash
python main_new.py
```
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/admin/health

## Testing
```bash
pytest -m "not integration"  # Unit tests only
pytest -m integration        # Integration tests
pytest --cov=.               # Coverage (after installing pytest-cov)
```

## Code Style
```bash
black .
isort .
ruff .
```

## Architecture
- `agent/` - Agent logic, core, config
- `app/` - FastAPI app, routes, services, utils
- `tools/` - SQL and ES toolkits
- `tests/` - Unit and integration tests

## Monitoring
- Health: `/admin/health`, `/admin/ping`
- Tracing: LangSmith (see .env)

---
**For more, see `CHAT_LOGGING.md` and code comments.** 