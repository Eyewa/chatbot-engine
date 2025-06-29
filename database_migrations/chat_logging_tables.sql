-- Chat Logging System - Database Migration
-- Senior AI Engineer Design: Multi-table architecture with proper indexing
-- Run this script to create the logging tables

-- =====================================================
-- Table 1: chatbot_conversation_messages
-- Main conversation tracking with detailed metrics
-- =====================================================

CREATE TABLE IF NOT EXISTS chatbot_conversation_messages (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    message_id VARCHAR(255) NOT NULL,
    conversation_id VARCHAR(255) NOT NULL,
    user_input TEXT NOT NULL,
    final_output TEXT NOT NULL,
    conversation_message TEXT,
    intent VARCHAR(100),
    classification VARCHAR(100),
    database_used VARCHAR(50),
    sql_queries JSON,
    sql_results_count INT,
    total_llm_calls INT NOT NULL DEFAULT 0,
    total_tokens_used INT NOT NULL DEFAULT 0,
    total_cost DECIMAL(10,6) NOT NULL DEFAULT 0,
    total_duration_ms DECIMAL(10,2) NOT NULL DEFAULT 0,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    debug_info JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_conversation_id (conversation_id),
    INDEX idx_message_id (message_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_intent (intent),
    INDEX idx_success (success),
    INDEX idx_database_used (database_used)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- Table 2: chatbot_llm_calls
-- Detailed LLM call tracking with token and cost metrics
-- =====================================================

CREATE TABLE IF NOT EXISTS chatbot_llm_calls (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    call_id VARCHAR(255) NOT NULL,
    conversation_id VARCHAR(255) NOT NULL,
    message_id VARCHAR(255) NOT NULL,
    model VARCHAR(100) NOT NULL,
    function_name VARCHAR(255) NOT NULL,
    input_tokens INT NOT NULL,
    output_tokens INT NOT NULL,
    total_tokens INT NOT NULL,
    cost_estimate DECIMAL(10,6) NOT NULL,
    duration_ms DECIMAL(10,2) NOT NULL,
    input_text TEXT NOT NULL,
    output_text TEXT NOT NULL,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    metadata JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_call_id (call_id),
    INDEX idx_conversation_id (conversation_id),
    INDEX idx_message_id (message_id),
    INDEX idx_function_name (function_name),
    INDEX idx_model (model),
    INDEX idx_timestamp (timestamp),
    INDEX idx_success (success),
    INDEX idx_cost_estimate (cost_estimate),
    INDEX idx_duration_ms (duration_ms)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- Table 3: chatbot_conversation_summary
-- Quick lookup summaries for monitoring and analytics
-- =====================================================

CREATE TABLE IF NOT EXISTS chatbot_conversation_summary (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    conversation_id VARCHAR(255) NOT NULL UNIQUE,
    first_message_at TIMESTAMP NOT NULL,
    last_message_at TIMESTAMP NOT NULL,
    total_messages INT NOT NULL DEFAULT 0,
    total_llm_calls INT NOT NULL DEFAULT 0,
    total_tokens_used INT NOT NULL DEFAULT 0,
    total_cost DECIMAL(10,6) NOT NULL DEFAULT 0,
    total_duration_ms DECIMAL(10,2) NOT NULL DEFAULT 0,
    success_rate DECIMAL(5,2) NOT NULL DEFAULT 100.00,
    last_user_input TEXT,
    last_agent_output TEXT,
    metadata JSON,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_conversation_id (conversation_id),
    INDEX idx_last_message_at (last_message_at),
    INDEX idx_success_rate (success_rate),
    INDEX idx_total_cost (total_cost),
    INDEX idx_total_tokens_used (total_tokens_used)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- Additional Performance Indexes
-- =====================================================

-- Composite indexes for common query patterns
CREATE INDEX idx_conv_msg_conversation_timestamp ON chatbot_conversation_messages (conversation_id, timestamp);
CREATE INDEX idx_conv_msg_success_timestamp ON chatbot_conversation_messages (success, timestamp);
CREATE INDEX idx_llm_calls_conversation_timestamp ON chatbot_llm_calls (conversation_id, timestamp);
CREATE INDEX idx_llm_calls_model_timestamp ON chatbot_llm_calls (model, timestamp);
CREATE INDEX idx_llm_calls_function_timestamp ON chatbot_llm_calls (function_name, timestamp);
CREATE INDEX idx_summary_last_message_cost ON chatbot_conversation_summary (last_message_at, total_cost);

-- =====================================================
-- Sample Data for Testing (Optional)
-- =====================================================

-- Uncomment the following section if you want to insert sample data for testing

/*
-- Sample conversation summary
INSERT INTO chatbot_conversation_summary (
    conversation_id, first_message_at, last_message_at, total_messages,
    total_llm_calls, total_tokens_used, total_cost, total_duration_ms,
    last_user_input, last_agent_output
) VALUES (
    'sample-conversation-123',
    NOW() - INTERVAL 1 HOUR,
    NOW(),
    2,
    3,
    1500,
    0.0035,
    2500.50,
    'Show me recent orders',
    'I found 5 recent orders for you...'
);

-- Sample conversation message
INSERT INTO chatbot_conversation_messages (
    message_id, conversation_id, user_input, final_output,
    conversation_message, intent, database_used, total_llm_calls,
    total_tokens_used, total_cost, total_duration_ms
) VALUES (
    'sample-message-456',
    'sample-conversation-123',
    'Show me recent orders',
    '{"type": "orders_summary", "orders": [{"id": 1, "amount": 100}]}',
    'I found 5 recent orders for you...',
    'order_query',
    'eyewa_live',
    3,
    1500,
    0.0035,
    2500.50
);

-- Sample LLM call
INSERT INTO chatbot_llm_calls (
    call_id, conversation_id, message_id, model, function_name,
    input_tokens, output_tokens, total_tokens, cost_estimate,
    duration_ms, input_text, output_text
) VALUES (
    'sample-call-789',
    'sample-conversation-123',
    'sample-message-456',
    'gpt-4o',
    'classify_query',
    500,
    200,
    700,
    0.00175,
    850.25,
    'Show me recent orders',
    '{"intent": "order_query", "database": "eyewa_live"}'
);
*/

-- =====================================================
-- Verification Queries
-- =====================================================

-- Verify tables were created successfully
SELECT 
    TABLE_NAME,
    TABLE_ROWS,
    DATA_LENGTH,
    INDEX_LENGTH
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = DATABASE() 
AND TABLE_NAME IN (
    'chatbot_conversation_messages',
    'chatbot_llm_calls', 
    'chatbot_conversation_summary'
);

-- Verify indexes were created
SELECT 
    TABLE_NAME,
    INDEX_NAME,
    COLUMN_NAME,
    SEQ_IN_INDEX
FROM information_schema.STATISTICS 
WHERE TABLE_SCHEMA = DATABASE() 
AND TABLE_NAME IN (
    'chatbot_conversation_messages',
    'chatbot_llm_calls', 
    'chatbot_conversation_summary'
)
ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX;

-- =====================================================
-- Migration Complete
-- =====================================================

-- The logging system is now ready to use!
-- 
-- Key features:
-- ✅ Multi-table architecture for efficient querying
-- ✅ Proper indexing for fast lookups
-- ✅ JSON storage for flexible metadata
-- ✅ Comprehensive tracking of conversations, LLM calls, and metrics
-- ✅ Minimal performance impact with optimized design
--
-- Next steps:
-- 1. Start your application with the logging enabled
-- 2. Use the /chat/ endpoint for comprehensive logging
-- 3. Monitor via /logging-dashboard/ endpoints
-- 4. Check LOGGING_README.md for detailed usage instructions 