"""
Logging Dashboard API routes
Senior AI Engineer Design: Comprehensive monitoring and analytics dashboard
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import text

from agent.chat_logger import get_chat_logger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/logging-dashboard", tags=["Logging Dashboard"])


def get_logger():
    """Dependency to get chat logger instance."""
    return get_chat_logger()


@router.get("/overview")
async def get_dashboard_overview(
    hours: int = Query(24, description="Hours to look back for overview"),
    logger_instance = Depends(get_logger)
):
    """
    Get dashboard overview with key metrics.
    
    Args:
        hours: Number of hours to look back
        logger_instance: Chat logger instance
        
    Returns:
        Dashboard overview with key metrics
    """
    try:
        since_time = datetime.now() - timedelta(hours=hours)
        
        with logger_instance.engine.connect() as conn:
            # Get conversation summary
            summary_result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_conversations,
                    SUM(total_messages) as total_messages,
                    SUM(total_llm_calls) as total_llm_calls,
                    SUM(total_tokens_used) as total_tokens,
                    SUM(total_cost) as total_cost,
                    SUM(total_duration_ms) as total_duration_ms,
                    AVG(success_rate) as avg_success_rate
                FROM chatbot_conversation_summary 
                WHERE last_message_at >= :since_time
            """), {"since_time": since_time}).fetchone()
            
            # Get LLM call breakdown
            llm_breakdown = conn.execute(text("""
                SELECT 
                    model,
                    function_name,
                    COUNT(*) as call_count,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(cost_estimate) as total_cost,
                    AVG(duration_ms) as avg_duration_ms,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_calls
                FROM chatbot_llm_calls 
                WHERE timestamp >= :since_time
                GROUP BY model, function_name
                ORDER BY call_count DESC
            """), {"since_time": since_time}).fetchall()
            
            # Get recent activity
            recent_activity = conn.execute(text("""
                SELECT 
                    conversation_id,
                    last_user_input,
                    last_agent_output,
                    total_messages,
                    total_llm_calls,
                    total_tokens_used,
                    total_cost,
                    last_message_at
                FROM chatbot_conversation_summary 
                WHERE last_message_at >= :since_time
                ORDER BY last_message_at DESC
                LIMIT 10
            """), {"since_time": since_time}).fetchall()
            
            # Get error summary
            error_summary = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_errors,
                    COUNT(DISTINCT conversation_id) as conversations_with_errors
                FROM chatbot_conversation_messages 
                WHERE success = 0 AND timestamp >= :since_time
            """), {"since_time": since_time}).fetchone()
        
        return {
            "status": "success",
            "overview": {
                "time_period": f"Last {hours} hours",
                "summary": dict(summary_result._mapping) if summary_result else {},
                "llm_breakdown": [dict(row._mapping) for row in llm_breakdown],
                "recent_activity": [dict(row._mapping) for row in recent_activity],
                "error_summary": dict(error_summary._mapping) if error_summary else {},
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard overview: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving dashboard overview: {str(e)}"
        )


@router.get("/conversation/{conversation_id}/detailed")
async def get_conversation_detailed(
    conversation_id: str,
    logger_instance = Depends(get_logger)
):
    """
    Get detailed breakdown of a specific conversation.
    
    Args:
        conversation_id: Conversation identifier
        logger_instance: Chat logger instance
        
    Returns:
        Detailed conversation breakdown
    """
    try:
        with logger_instance.engine.connect() as conn:
            # Get conversation summary
            summary = conn.execute(text("""
                SELECT * FROM chatbot_conversation_summary 
                WHERE conversation_id = :conversation_id
            """), {"conversation_id": conversation_id}).fetchone()
            
            if not summary:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Conversation {conversation_id} not found"
                )
            
            # Get all messages with details
            messages = conn.execute(text("""
                SELECT 
                    message_id,
                    user_input,
                    final_output,
                    conversation_message,
                    intent,
                    classification,
                    database_used,
                    sql_queries,
                    sql_results_count,
                    total_llm_calls,
                    total_tokens_used,
                    total_cost,
                    total_duration_ms,
                    success,
                    error_message,
                    debug_info,
                    timestamp
                FROM chatbot_conversation_messages 
                WHERE conversation_id = :conversation_id
                ORDER BY timestamp ASC
            """), {"conversation_id": conversation_id}).fetchall()
            
            # Get all LLM calls for this conversation
            llm_calls = conn.execute(text("""
                SELECT 
                    call_id,
                    message_id,
                    model,
                    function_name,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cost_estimate,
                    duration_ms,
                    input_text,
                    output_text,
                    success,
                    error_message,
                    metadata,
                    timestamp
                FROM chatbot_llm_calls 
                WHERE conversation_id = :conversation_id
                ORDER BY timestamp ASC
            """), {"conversation_id": conversation_id}).fetchall()
            
            # Calculate performance metrics
            total_duration = sum(msg.total_duration_ms for msg in messages)
            total_cost = sum(msg.total_cost for msg in messages)
            total_tokens = sum(msg.total_tokens_used for msg in messages)
            success_rate = (sum(1 for msg in messages if msg.success) / len(messages)) * 100 if messages else 0
            
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "summary": dict(summary._mapping),
                "messages": [dict(msg._mapping) for msg in messages],
                "llm_calls": [dict(call._mapping) for call in llm_calls],
                "performance_metrics": {
                    "total_duration_ms": total_duration,
                    "total_cost": total_cost,
                    "total_tokens": total_tokens,
                    "success_rate": success_rate,
                    "message_count": len(messages),
                    "llm_call_count": len(llm_calls)
                },
                "generated_at": datetime.now().isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation details: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving conversation details: {str(e)}"
        )


@router.get("/llm-calls/analysis")
async def get_llm_calls_analysis(
    hours: int = Query(24, description="Hours to look back for analysis"),
    model: Optional[str] = Query(None, description="Filter by model"),
    function_name: Optional[str] = Query(None, description="Filter by function name"),
    logger_instance = Depends(get_logger)
):
    """
    Get detailed analysis of LLM calls.
    
    Args:
        hours: Number of hours to look back
        model: Optional model filter
        function_name: Optional function name filter
        logger_instance: Chat logger instance
        
    Returns:
        LLM calls analysis
    """
    try:
        since_time = datetime.now() - timedelta(hours=hours)
        
        # Build query with optional filters
        where_clause = "WHERE timestamp >= :since_time"
        params: Dict[str, Any] = {"since_time": since_time}
        
        if model:
            where_clause += " AND model = :model"
            params["model"] = model
            
        if function_name:
            where_clause += " AND function_name = :function_name"
            params["function_name"] = function_name
        
        with logger_instance.engine.connect() as conn:
            # Get LLM calls analysis
            analysis = conn.execute(text(f"""
                SELECT 
                    model,
                    function_name,
                    COUNT(*) as total_calls,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(cost_estimate) as total_cost,
                    AVG(duration_ms) as avg_duration_ms,
                    MIN(duration_ms) as min_duration_ms,
                    MAX(duration_ms) as max_duration_ms,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_calls,
                    AVG(input_tokens) as avg_input_tokens,
                    AVG(output_tokens) as avg_output_tokens,
                    AVG(total_tokens) as avg_total_tokens,
                    AVG(cost_estimate) as avg_cost
                FROM chatbot_llm_calls 
                {where_clause}
                GROUP BY model, function_name
                ORDER BY total_calls DESC
            """), params).fetchall()
            
            # Get recent calls for timeline
            recent_calls = conn.execute(text(f"""
                SELECT 
                    call_id,
                    conversation_id,
                    message_id,
                    model,
                    function_name,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cost_estimate,
                    duration_ms,
                    success,
                    timestamp
                FROM chatbot_llm_calls 
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT 50
            """), params).fetchall()
            
            # Get error analysis
            error_analysis = conn.execute(text(f"""
                SELECT 
                    model,
                    function_name,
                    error_message,
                    COUNT(*) as error_count
                FROM chatbot_llm_calls 
                {where_clause} AND success = 0
                GROUP BY model, function_name, error_message
                ORDER BY error_count DESC
                LIMIT 20
            """), params).fetchall()
        
        return {
            "status": "success",
            "analysis": {
                "time_period": f"Last {hours} hours",
                "filters": {
                    "model": model,
                    "function_name": function_name
                },
                "summary": [dict(row._mapping) for row in analysis],
                "recent_calls": [dict(row._mapping) for row in recent_calls],
                "error_analysis": [dict(row._mapping) for row in error_analysis],
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting LLM calls analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving LLM calls analysis: {str(e)}"
        )


@router.get("/sql-queries/analysis")
async def get_sql_queries_analysis(
    hours: int = Query(24, description="Hours to look back for analysis"),
    logger_instance = Depends(get_logger)
):
    """
    Get analysis of SQL queries executed.
    
    Args:
        hours: Number of hours to look back
        logger_instance: Chat logger instance
        
    Returns:
        SQL queries analysis
    """
    try:
        since_time = datetime.now() - timedelta(hours=hours)
        
        with logger_instance.engine.connect() as conn:
            # Get SQL queries analysis
            sql_analysis = conn.execute(text("""
                SELECT 
                    database_used,
                    COUNT(*) as total_queries,
                    SUM(sql_results_count) as total_results,
                    AVG(sql_results_count) as avg_results,
                    COUNT(DISTINCT conversation_id) as conversations_with_queries
                FROM chatbot_conversation_messages 
                WHERE timestamp >= :since_time 
                AND sql_queries IS NOT NULL 
                AND sql_queries != 'null'
                GROUP BY database_used
                ORDER BY total_queries DESC
            """), {"since_time": since_time}).fetchall()
            
            # Get recent SQL queries
            recent_queries = conn.execute(text("""
                SELECT 
                    conversation_id,
                    message_id,
                    database_used,
                    sql_queries,
                    sql_results_count,
                    timestamp
                FROM chatbot_conversation_messages 
                WHERE timestamp >= :since_time 
                AND sql_queries IS NOT NULL 
                AND sql_queries != 'null'
                ORDER BY timestamp DESC
                LIMIT 20
            """), {"since_time": since_time}).fetchall()
        
        return {
            "status": "success",
            "sql_analysis": {
                "time_period": f"Last {hours} hours",
                "summary": [dict(row._mapping) for row in sql_analysis],
                "recent_queries": [dict(row._mapping) for row in recent_queries],
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting SQL queries analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving SQL queries analysis: {str(e)}"
        ) 