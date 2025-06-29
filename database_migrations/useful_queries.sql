-- Enhanced Chat Logging System - Useful Queries
-- Collection of SQL queries for monitoring and analyzing logging data
-- Senior AI Engineer Design: Production-ready analytics queries

-- =====================================================
-- 1. OVERVIEW DASHBOARD QUERIES
-- =====================================================

-- Get overall statistics for the last 24 hours
SELECT 
    COUNT(DISTINCT conversation_id) as total_conversations,
    COUNT(*) as total_messages,
    SUM(total_llm_calls) as total_llm_calls,
    SUM(total_tokens_used) as total_tokens,
    SUM(total_cost) as total_cost,
    AVG(total_duration_ms) as avg_duration_ms,
    AVG(success_rate) as avg_success_rate
FROM chatbot_conversation_summary 
WHERE last_message_at >= NOW() - INTERVAL 24 HOUR;

-- Get hourly conversation count for the last 7 days
SELECT 
    DATE_FORMAT(last_message_at, '%Y-%m-%d %H:00:00') as hour,
    COUNT(*) as conversations,
    SUM(total_cost) as total_cost,
    AVG(total_duration_ms) as avg_duration_ms
FROM chatbot_conversation_summary 
WHERE last_message_at >= NOW() - INTERVAL 7 DAY
GROUP BY DATE_FORMAT(last_message_at, '%Y-%m-%d %H:00:00')
ORDER BY hour DESC;

-- =====================================================
-- 2. LLM CALLS ANALYSIS
-- =====================================================

-- LLM calls breakdown by model and function
SELECT 
    model,
    function_name,
    COUNT(*) as call_count,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    SUM(total_tokens) as total_tokens,
    SUM(cost_estimate) as total_cost,
    AVG(duration_ms) as avg_duration_ms,
    MIN(duration_ms) as min_duration_ms,
    MAX(duration_ms) as max_duration_ms,
    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_calls,
    ROUND(
        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 
        2
    ) as success_rate
FROM chatbot_llm_calls 
WHERE timestamp >= NOW() - INTERVAL 24 HOUR
GROUP BY model, function_name
ORDER BY call_count DESC;

-- Most expensive LLM calls
SELECT 
    call_id,
    conversation_id,
    model,
    function_name,
    input_tokens,
    output_tokens,
    total_tokens,
    cost_estimate,
    duration_ms,
    timestamp
FROM chatbot_llm_calls 
WHERE timestamp >= NOW() - INTERVAL 24 HOUR
ORDER BY cost_estimate DESC
LIMIT 20;

-- Slowest LLM calls
SELECT 
    call_id,
    conversation_id,
    model,
    function_name,
    duration_ms,
    total_tokens,
    cost_estimate,
    timestamp
FROM chatbot_llm_calls 
WHERE timestamp >= NOW() - INTERVAL 24 HOUR
ORDER BY duration_ms DESC
LIMIT 20;

-- =====================================================
-- 3. CONVERSATION ANALYSIS
-- =====================================================

-- Most expensive conversations
SELECT 
    conversation_id,
    total_messages,
    total_llm_calls,
    total_tokens_used,
    total_cost,
    total_duration_ms,
    success_rate,
    last_message_at
FROM chatbot_conversation_summary 
WHERE last_message_at >= NOW() - INTERVAL 24 HOUR
ORDER BY total_cost DESC
LIMIT 20;

-- Conversations with errors
SELECT 
    conversation_id,
    total_messages,
    success_rate,
    last_user_input,
    last_agent_output,
    last_message_at
FROM chatbot_conversation_summary 
WHERE success_rate < 100.0
AND last_message_at >= NOW() - INTERVAL 24 HOUR
ORDER BY success_rate ASC;

-- Longest conversations (by duration)
SELECT 
    conversation_id,
    total_messages,
    total_duration_ms,
    total_cost,
    success_rate,
    last_message_at
FROM chatbot_conversation_summary 
WHERE last_message_at >= NOW() - INTERVAL 24 HOUR
ORDER BY total_duration_ms DESC
LIMIT 20;

-- =====================================================
-- 4. SQL QUERIES ANALYSIS
-- =====================================================

-- Database usage breakdown
SELECT 
    database_used,
    COUNT(*) as query_count,
    SUM(sql_results_count) as total_results,
    AVG(sql_results_count) as avg_results,
    COUNT(DISTINCT conversation_id) as conversations_with_queries
FROM chatbot_conversation_messages 
WHERE timestamp >= NOW() - INTERVAL 24 HOUR
AND sql_queries IS NOT NULL 
AND sql_queries != 'null'
GROUP BY database_used
ORDER BY query_count DESC;

-- Recent SQL queries
SELECT 
    conversation_id,
    message_id,
    database_used,
    sql_queries,
    sql_results_count,
    timestamp
FROM chatbot_conversation_messages 
WHERE timestamp >= NOW() - INTERVAL 24 HOUR
AND sql_queries IS NOT NULL 
AND sql_queries != 'null'
ORDER BY timestamp DESC
LIMIT 20;

-- =====================================================
-- 5. ERROR ANALYSIS
-- =====================================================

-- Failed LLM calls with error messages
SELECT 
    call_id,
    conversation_id,
    model,
    function_name,
    error_message,
    input_tokens,
    output_tokens,
    duration_ms,
    timestamp
FROM chatbot_llm_calls 
WHERE success = 0
AND timestamp >= NOW() - INTERVAL 24 HOUR
ORDER BY timestamp DESC;

-- Failed conversation messages
SELECT 
    conversation_id,
    message_id,
    user_input,
    error_message,
    debug_info,
    timestamp
FROM chatbot_conversation_messages 
WHERE success = 0
AND timestamp >= NOW() - INTERVAL 24 HOUR
ORDER BY timestamp DESC;

-- Error patterns by function
SELECT 
    function_name,
    COUNT(*) as error_count,
    COUNT(DISTINCT conversation_id) as conversations_with_errors,
    AVG(duration_ms) as avg_duration_before_error
FROM chatbot_llm_calls 
WHERE success = 0
AND timestamp >= NOW() - INTERVAL 24 HOUR
GROUP BY function_name
ORDER BY error_count DESC;

-- =====================================================
-- 6. COST ANALYSIS
-- =====================================================

-- Daily cost breakdown
SELECT 
    DATE(timestamp) as date,
    SUM(cost_estimate) as daily_cost,
    COUNT(*) as total_calls,
    AVG(cost_estimate) as avg_cost_per_call,
    SUM(total_tokens) as total_tokens
FROM chatbot_llm_calls 
WHERE timestamp >= NOW() - INTERVAL 7 DAY
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Cost by model
SELECT 
    model,
    COUNT(*) as call_count,
    SUM(cost_estimate) as total_cost,
    AVG(cost_estimate) as avg_cost_per_call,
    SUM(total_tokens) as total_tokens,
    AVG(total_tokens) as avg_tokens_per_call
FROM chatbot_llm_calls 
WHERE timestamp >= NOW() - INTERVAL 24 HOUR
GROUP BY model
ORDER BY total_cost DESC;

-- =====================================================
-- 7. PERFORMANCE ANALYSIS
-- =====================================================

-- Response time percentiles
SELECT 
    model,
    COUNT(*) as call_count,
    AVG(duration_ms) as avg_duration,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) as median_duration,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_duration,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99_duration
FROM chatbot_llm_calls 
WHERE timestamp >= NOW() - INTERVAL 24 HOUR
GROUP BY model;

-- Token usage analysis
SELECT 
    model,
    COUNT(*) as call_count,
    AVG(input_tokens) as avg_input_tokens,
    AVG(output_tokens) as avg_output_tokens,
    AVG(total_tokens) as avg_total_tokens,
    MAX(total_tokens) as max_tokens,
    MIN(total_tokens) as min_tokens
FROM chatbot_llm_calls 
WHERE timestamp >= NOW() - INTERVAL 24 HOUR
GROUP BY model;

-- =====================================================
-- 8. REAL-TIME MONITORING QUERIES
-- =====================================================

-- Active conversations (last 5 minutes)
SELECT 
    conversation_id,
    total_messages,
    total_llm_calls,
    total_cost,
    total_duration_ms,
    last_message_at
FROM chatbot_conversation_summary 
WHERE last_message_at >= NOW() - INTERVAL 5 MINUTE
ORDER BY last_message_at DESC;

-- Recent LLM calls (last 10 minutes)
SELECT 
    call_id,
    conversation_id,
    model,
    function_name,
    total_tokens,
    cost_estimate,
    duration_ms,
    success,
    timestamp
FROM chatbot_llm_calls 
WHERE timestamp >= NOW() - INTERVAL 10 MINUTE
ORDER BY timestamp DESC;

-- =====================================================
-- 9. MAINTENANCE QUERIES
-- =====================================================

-- Table sizes and row counts
SELECT 
    TABLE_NAME,
    TABLE_ROWS,
    ROUND(DATA_LENGTH/1024/1024, 2) as data_size_mb,
    ROUND(INDEX_LENGTH/1024/1024, 2) as index_size_mb,
    ROUND((DATA_LENGTH + INDEX_LENGTH)/1024/1024, 2) as total_size_mb
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = DATABASE() 
AND TABLE_NAME IN (
    'chatbot_conversation_messages',
    'chatbot_llm_calls', 
    'chatbot_conversation_summary'
);

-- Old data cleanup (data older than 30 days)
-- WARNING: This will delete old data!
/*
DELETE FROM chatbot_llm_calls 
WHERE timestamp < NOW() - INTERVAL 30 DAY;

DELETE FROM chatbot_conversation_messages 
WHERE timestamp < NOW() - INTERVAL 30 DAY;

DELETE FROM chatbot_conversation_summary 
WHERE last_message_at < NOW() - INTERVAL 30 DAY;
*/

-- =====================================================
-- 10. CUSTOM ANALYTICS QUERIES
-- =====================================================

-- User interaction patterns (by conversation length)
SELECT 
    CASE 
        WHEN total_messages = 1 THEN 'Single Message'
        WHEN total_messages BETWEEN 2 AND 5 THEN 'Short (2-5)'
        WHEN total_messages BETWEEN 6 AND 10 THEN 'Medium (6-10)'
        ELSE 'Long (>10)'
    END as conversation_length,
    COUNT(*) as conversation_count,
    AVG(total_cost) as avg_cost,
    AVG(total_duration_ms) as avg_duration,
    AVG(success_rate) as avg_success_rate
FROM chatbot_conversation_summary 
WHERE last_message_at >= NOW() - INTERVAL 7 DAY
GROUP BY 
    CASE 
        WHEN total_messages = 1 THEN 'Single Message'
        WHEN total_messages BETWEEN 2 AND 5 THEN 'Short (2-5)'
        WHEN total_messages BETWEEN 6 AND 10 THEN 'Medium (6-10)'
        ELSE 'Long (>10)'
    END
ORDER BY conversation_count DESC;

-- Peak usage hours
SELECT 
    HOUR(last_message_at) as hour_of_day,
    COUNT(*) as conversation_count,
    SUM(total_cost) as total_cost,
    AVG(total_duration_ms) as avg_duration
FROM chatbot_conversation_summary 
WHERE last_message_at >= NOW() - INTERVAL 7 DAY
GROUP BY HOUR(last_message_at)
ORDER BY conversation_count DESC;

-- =====================================================
-- Query Usage Notes
-- =====================================================

/*
These queries are designed for production monitoring and analysis.
Key features:

1. Time-based filtering - Most queries filter by recent time periods
2. Performance optimized - Uses indexed columns for fast execution
3. Comprehensive metrics - Covers cost, performance, errors, and usage patterns
4. Real-time monitoring - Includes queries for active monitoring
5. Maintenance ready - Includes cleanup and maintenance queries

Usage tips:
- Adjust time intervals based on your needs (24 HOUR, 7 DAY, etc.)
- Monitor query performance on large datasets
- Consider creating views for frequently used queries
- Set up automated alerts based on these metrics
*/ 