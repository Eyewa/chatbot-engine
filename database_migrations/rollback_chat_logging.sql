-- Chat Logging System - Rollback Script
-- Use this script to remove the logging tables if needed
-- WARNING: This will permanently delete all logging data!

-- =====================================================
-- WARNING: DATA LOSS
-- =====================================================
-- This script will permanently delete all logging data
-- Make sure you have backups if you need to preserve any data
-- =====================================================

-- =====================================================
-- Drop Tables in Correct Order (due to foreign key constraints)
-- =====================================================

-- Drop the LLM calls table first (referenced by conversation messages)
DROP TABLE IF EXISTS chatbot_llm_calls;

-- Drop the conversation messages table
DROP TABLE IF EXISTS chatbot_conversation_messages;

-- Drop the conversation summary table last
DROP TABLE IF EXISTS chatbot_conversation_summary;

-- =====================================================
-- Verification
-- =====================================================

-- Verify tables were dropped successfully
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

-- Should return no rows if tables were successfully dropped

-- =====================================================
-- Rollback Complete
-- =====================================================

-- The logging tables have been removed
-- 
-- Note: If you want to re-enable logging:
-- 1. Run the logging_tables.sql script again
-- 2. Restart your application
-- 3. The tables will be recreated automatically 