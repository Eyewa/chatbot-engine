# Eyewa Chatbot Engine

A modular, production-ready FastAPI-based multi-agent chatbot engine for dynamic, structured data queries and business logic orchestration.

## Features
- **Multi-agent orchestration**: Live and common (loyalty/wallet) data sources, dynamic routing.
- **Modular architecture**: Clean separation of API, agent logic, services, and utilities.
- **Schema-driven**: All table/field logic, joins, and business rules are config-driven.
- **Business rules**: Easily extendable via YAML for custom summary/response logic.
- **Comprehensive logging**: Tracks all LLM calls, conversation messages, and summaries.
- **LangSmith integration**: For tracing and token/cost tracking.
- **Robust testing**: Unit and integration tests, with clear separation and coverage.
- **OpenAPI docs**: All endpoints documented and type-safe.

## Directory Structure
```
chatbot-engine/
  agent/                # Agent logic, orchestration, prompt building, schema validation
    core/               # Core agent modules (router, classifier, orchestrator, etc.)
    intent_to_sql.py    # Intent-to-SQL logic
    reload_config.py    # Hot-reload config endpoint
  app/                  # FastAPI app, API routes, services, and utilities
    api/                # API endpoints and middleware
    core/               # App config and logging
    services/           # Chat logging, history, and business logic
    utils/              # Response formatting utilities
  config/               # YAML/JSON config for schema, routing, and templates
  tools/                # SQL toolkit and custom table info loaders
  tests/                # Unit and integration tests
  main_new.py           # Main FastAPI entrypoint
  requirements.txt      # Python dependencies
  README.md             # Project overview (this file)
  CHAT_LOGGING.md       # Chat logging schema and flow
```

## How It Works
- **User query** → `/chat/` endpoint → routed to agent(s) based on intent and keywords.
- **Agent** builds prompt using schema, allowed tables, and business rules.
- **LLM** generates SQL or structured response → executed against DB(s).
- **Business rules** (from YAML) may add summaries (e.g., customer_summary) based on query context.
- **Responses** are merged, formatted, and logged.

## Configuration & Extensibility
- **Schema**: `config/schema/schema.yaml` defines all tables, fields, joins, and field mappings.
- **Routing & Rules**: `config/query_routing.yaml` defines classification, response types, and business rules.
- **Response Types**: `config/templates/response_types.yaml` defines output schemas for summaries.
- **Adding new tables/rules**: See `EXTENDING_SCHEMA_AND_RULES.md` for a step-by-step guide.

## Logging
- All LLM calls and conversation messages are logged to MySQL tables (`chatbot_conversation_messages`, `chatbot_conversation_summary`).
- See `CHAT_LOGGING.md` for schema and flow details.

## Testing
- **Unit tests**: `pytest -m 'not integration'`
- **Integration tests**: `pytest -m integration` (requires running backend and DB)
- **Coverage**: `pytest --cov`

## Linting & Formatting
- **Ruff**: `ruff check .` (includes linting, formatting, and import sorting)
- **Black**: `black .`
- **Isort**: `isort .`

## OpenAPI & Monitoring
- Visit `/docs` for interactive API docs.
- Health endpoints: `/admin/health`, `/admin/ping`

## Contributing
- Fork, branch, and submit PRs. All code must pass linting and tests.
- See `EXTENDING_SCHEMA_AND_RULES.md` for how to add new tables, fields, and business rules.

## Getting Started: Running the Service

Follow these steps to start the chatbot engine service:

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd chatbot-engine
   ```
2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set environment variables (if needed):**
   - Copy `.env.example` to `.env` and edit as needed (if applicable).
5. **Run database migrations (if needed):**
   - Apply SQL scripts in `database_migrations/` to your database.
6. **Start the FastAPI server:**
   ```bash
   uvicorn main_new:app --reload
   ```
   - The API will be available at [http://localhost:8000](http://localhost:8000)
7. **(Optional) Run tests:**
   ```bash
   pytest
   ```

## Running with Docker

You can build and run the chatbot engine using Docker for easy deployment.

### Build the Docker Image

```bash
docker build -t chatbot-engine .
```

### Run the Docker Container

```bash
docker run -d \
  --name chatbot-engine \
  -p 8000:8000 \
  chatbot-engine
```
- The API will be available at [http://localhost:8000](http://localhost:8000)

### (Optional) Mount Local Config Files
If you want to use local config files instead of those baked into the image:
```bash
docker run -d \
  --name chatbot-engine \
  -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  chatbot-engine
```

### Stopping and Removing the Container
```bash
docker stop chatbot-engine
docker rm chatbot-engine
```

See the [Configuration](#configuration) section for details on customizing schema, routing, and business rules.

---

For detailed extension instructions, see `EXTENDING_SCHEMA_AND_RULES.md`. 