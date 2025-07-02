# Chat Logging

This system logs all chatbot conversations and messages for analytics and debugging.

## Table Structure

### chatbot_conversation_messages
```sql
CREATE TABLE chatbot_conversation_messages (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    message_id VARCHAR(255) NOT NULL UNIQUE,
    conversation_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    sender VARCHAR(32) DEFAULT NULL,
    message_text TEXT,
    intent VARCHAR(255),
    context JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_conversation_id (conversation_id),
    INDEX idx_timestamp (timestamp)
);
```
- `id`: Auto-increment primary key (main identifier)
- `message_id`: Unique string/UUID for global reference (optional)
- `conversation_id`: Groups messages in a conversation
- `sender`: 'user' or 'bot'

### chatbot_conversation_summary
```sql
CREATE TABLE IF NOT EXISTS chatbot_conversation_summary (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    conversation_id VARCHAR(255) NOT NULL UNIQUE,
    first_message_at TIMESTAMP NOT NULL,
    last_message_at TIMESTAMP NOT NULL,
    total_messages INT NOT NULL DEFAULT 0,
    last_user_input TEXT,
    last_agent_output TEXT,
    metadata JSON,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_conversation_id (conversation_id),
    INDEX idx_last_message_at (last_message_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```
- Tracks summary info for each conversation

## How Logging Works
- Every message (user or bot) is logged with a unique `id` and `message_id`.
- All messages for a conversation share the same `conversation_id`.
- The summary table is updated for each conversation.

## Notes
- Token and cost tracking is handled by LangSmith.
- Only the above tables are actively written to in the current flow. 