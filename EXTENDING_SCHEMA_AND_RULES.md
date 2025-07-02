# Extending Schema and Business Rules

This guide explains how to add new tables, fields, and business rules to the chatbot engine, and how these changes are used throughout the flow.

## 1. Add New Table(s) or Fields to the Schema

**File:** `config/schema/schema.yaml`

- Under the appropriate section (`live.tables` or `common.tables`), add your new table and its fields.
- Define `fields`, `joins`, and (optionally) `field_meta` and `customInfo`.

**Example:**
```yaml
live:
  tables:
    new_table:
      description: "Description of the new table."
      fields: [id, name, created_at]
      joins:
        - to_table: other_table
          from_field: id
          to_field: new_table_id
```

- If you want to add new field mappings (e.g., for summary fields), add them under `field_mappings` at the top of the schema file.

## 2. Update Routing, Response Types, and Business Rules

**File:** `config/query_routing.yaml`

- **Classification:** Add new keywords or types under `classification_rules.keywords` if your new table should be routed by certain keywords.
- **Response Types:** Add or update `response_type_detection.content_keywords` and `type_mapping` if your new table should trigger a new summary type.
- **Field Extraction:** Add extraction strategies for new summary types under `field_extraction`.
- **Business Rules:** Add new rules under `business_rules.always_include_summary` or other rule sections as needed.

**Example:**
```yaml
business_rules:
  always_include_summary:
    - type: new_summary_type
      condition: new_condition
```

## 3. Define Output Schema for New Summaries

**File:** `config/templates/response_types.yaml`

- Add a new entry for your summary type, listing the fields to include in the output.

**Example:**
```yaml
new_summary_type:
  fields:
    - id
    - name
    - created_at
```

## 4. How the Code Uses These Changes

- **Agent Construction:**
  - The agent loads the schema and allowed tables from `schema.yaml`.
  - Prompts are built using allowed tables, fields, and join info.
- **Business Rules:**
  - The orchestrator and `apply_business_rules` in `agent/core/agent_router.py` read business rules from `query_routing.yaml`.
  - When a rule is triggered (e.g., a certain keyword or condition), the code queries the relevant table and builds a summary using the output schema from `response_types.yaml`.
- **Response Formatting:**
  - The output is formatted according to the response type definition and returned to the user.
- **Logging:**
  - All queries, responses, and summaries are logged for traceability.

## 5. Step-by-Step Example: Adding a Table and Summary

1. **Add your table to `schema.yaml`:**
    - Define fields, joins, and (optionally) field_meta.
2. **Add field mappings if needed.**
3. **Update `query_routing.yaml`:**
    - Add keywords, response types, field extraction, and business rules as needed.
4. **Update `response_types.yaml`:**
    - Define the output fields for your new summary type.
5. **(Optional) Add tests:**
    - Add or update tests in `tests/` to cover your new table and summary logic.
6. **Restart the backend:**
    - The system will hot-reload config changes.

## 6. Relevant Files and Flow

- `config/schema/schema.yaml`: Table/field definitions, joins, field mappings
- `config/query_routing.yaml`: Routing, classification, business rules, field extraction
- `config/templates/response_types.yaml`: Output schemas for summaries
- `agent/core/agent_router.py`: Business rule application, summary building
- `agent/core/prompt_builder.py`: Prompt construction using schema
- `app/services/chat_service.py`: Orchestrates agent calls, logging, and response formatting

## 7. Tips
- Use clear, unique names for tables and fields.
- Keep business rules and field extraction strategies as simple as possible.
- Use debug logging to trace how your new table or rule is being used in the flow.

# Advanced Examples

## Example 1: Adding a Table with Custom Joins and Computed Fields

Suppose you want to add an `order_items` table that joins to both `orders` and `products`, and you want to compute a `total_price` field.

**In `config/schema/schema.yaml`:**
```yaml
live:
  tables:
    order_items:
      description: "Line items for each order."
      fields: [id, order_id, product_id, quantity, unit_price]
      joins:
        - to_table: orders
          from_field: order_id
          to_field: id
        - to_table: products
          from_field: product_id
          to_field: id
      field_meta:
        total_price:
          type: computed
          expression: quantity * unit_price
```

## Example 2: Adding a Business Rule with a Conditional Expression

Suppose you want to always include a high-value order summary if the total order value exceeds $1000.

**In `config/query_routing.yaml`:**
```yaml
business_rules:
  always_include_summary:
    - type: high_value_order_summary
      condition: order_total > 1000
```

**In `config/templates/response_types.yaml`:**
```yaml
high_value_order_summary:
  fields:
    - order_id
    - customer_id
    - order_total
    - item_count
    - created_at
```

## Example 3: Custom Field Extraction for a New Summary Type

Suppose you want to extract a list of product names for a `customer_favorites` summary.

**In `config/query_routing.yaml`:**
```yaml
field_extraction:
  customer_favorites:
    strategy: custom
    function: extract_customer_favorites
```

**In your code (e.g., `agent/core/agent_router.py`):**
```python
def extract_customer_favorites(customer_id, db):
    # Custom logic to find favorite products for a customer
    ...
```

## Example 4: Adding a Table with a CustomInfo Block

**In `config/schema/schema.yaml`:**
```yaml
live:
  tables:
    support_tickets:
      description: "Customer support tickets."
      fields: [ticket_id, customer_id, status, created_at, resolved_at]
      customInfo:
        escalation_policy: "auto-escalate after 48h if unresolved"
```

---

These advanced examples demonstrate how to:
- Add tables with multiple joins and computed fields
- Write business rules with conditions
- Use custom field extraction functions
- Add custom metadata to tables

For more, see code comments in `agent/core/agent_router.py` and `agent/core/prompt_builder.py`. 