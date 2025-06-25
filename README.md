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

## Example: Structured Intent to SQL Flow

### User Query
> "Show me the last two orders and the customer name for customer 1338787"

### 1. Classification & Table Extraction
- Intent: `order`
- Sub-intents: `recent_orders`, `customer_profile.name`
- Tables: `sales_order`, `customer_entity`

### 2. Mini-Schema Prompt Injected to LLM
```
You are Winkly, a structured data assistant for customer and order data. Use only the schema and rules below.
Rules:
- Use only the tables and fields listed below. Never invent or guess columns, tables, or values.
- Only use joins that are listed below.
- The only valid database names are 'eyewa_live' and 'eyewa_common'. Never use 'magento_live' or any other database name.
- Always output a JSON object with: tables, fields, joins, filters, limit, order_by. Do NOT output SQL.

- sales_order: entity_id, increment_id, customer_id, grand_total, created_at
- customer_entity: entity_id, firstname, lastname

Known joins:
sales_order.customer_id → customer_entity.entity_id
```

### 3. LLM Output (Structured Intent)
```json
{
  "tables": ["sales_order", "customer_entity"],
  "fields": [
    {"table": "sales_order", "field": "increment_id"},
    {"table": "sales_order", "field": "grand_total"},
    {"table": "customer_entity", "field": "firstname"},
    {"table": "customer_entity", "field": "lastname"}
  ],
  "joins": [
    {
      "from_table": "sales_order",
      "from_field": "customer_id",
      "to_table": "customer_entity",
      "to_field": "entity_id"
    }
  ],
  "filters": {"customer_entity.entity_id": 1338787},
  "limit": 2,
  "order_by": "sales_order.created_at DESC"
}
```

### 4. Translate Intent to SQL
```python
from agent.intent_to_sql import intent_to_sql
sql = intent_to_sql(intent)  # intent is the dict above
print(sql)
```
**Output:**
```sql
SELECT sa.increment_id, sa.grand_total, cu.firstname, cu.lastname
FROM sales_order sa
JOIN customer_entity cu ON sa.customer_id = cu.entity_id
WHERE cu.entity_id = 1338787
ORDER BY sa.created_at DESC
LIMIT 2
```

### 5. Execute the SQL
```python
result = live_tool.run({"query": sql})
```

### 6. Error Handling Example
If the LLM outputs an invalid field or table, `intent_to_sql` will raise an error:
```python
try:
    sql = intent_to_sql(intent)
except ValueError as e:
    print(f"Error: {e}")
```

### 7. Aggregation Example
If the intent includes aggregation:
```json
{
  "tables": ["sales_order"],
  "fields": [
    {"table": "sales_order", "field": "customer_id"}
  ],
  "aggregation": {
    "function": "SUM",
    "field": "grand_total",
    "group_by": "customer_id"
  }
}
```
You can extend `intent_to_sql` to handle aggregation, or build the SQL as:
```python
agg = intent["aggregation"]
sql = f"SELECT {agg['group_by']}, {agg['function']}({agg['field']}) as total FROM sales_order GROUP BY {agg['group_by']}"
```

---

See `test_structured_intent_flow.py` for runnable examples and error handling.
