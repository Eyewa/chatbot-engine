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