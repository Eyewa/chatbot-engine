CREATE TABLE chatbot_conversation_messages (
    message_id VARCHAR(255) NOT NULL PRIMARY KEY,
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