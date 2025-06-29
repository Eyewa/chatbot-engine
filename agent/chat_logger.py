"""
Chat Logger - Comprehensive conversation and LLM call tracking
Senior AI Engineer Design: Multi-table architecture with minimal performance impact
"""

import os
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Text, Integer, Float, DateTime, JSON, Index
from sqlalchemy.exc import SQLAlchemyError
import logging

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMCall:
    """Data structure for tracking individual LLM calls"""
    call_id: str
    conversation_id: str
    message_id: str
    model: str
    function_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_estimate: float
    duration_ms: float
    input_text: str
    output_text: str
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[str] = None  # JSON string for database storage
    timestamp: Optional[datetime] = None


@dataclass
class ConversationMessage:
    """Data structure for tracking conversation messages"""
    message_id: str
    conversation_id: str
    user_input: str
    final_output: str
    total_llm_calls: int
    total_tokens_used: int
    total_cost: float
    total_duration_ms: float
    success: bool
    conversation_message: Optional[str] = None
    intent: Optional[str] = None
    classification: Optional[str] = None
    database_used: Optional[str] = None
    sql_queries: Optional[List[str]] = None
    sql_results_count: Optional[int] = None
    error_message: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class ChatLogger:
    """
    Senior AI Engineer Design: Comprehensive logging with minimal performance impact
    
    Features:
    - Multi-table architecture for efficient querying
    - Async-friendly with connection pooling
    - Proper indexing for fast lookups
    - Structured data storage with JSON for flexibility
    - Minimal blocking operations
    - Comprehensive error handling
    """
    
    def __init__(self):
        self.uri = os.getenv("SQL_DATABASE_URI_LIVE_WRITE")
        if not self.uri:
            raise ValueError("SQL_DATABASE_URI_LIVE_WRITE is not set")
        
        # Engine with optimized settings for logging
        self.engine = create_engine(
            self.uri,
            pool_size=5,  # Smaller pool for logging
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            echo=False  # Disable SQL echo for performance
        )
        
        # Initialize tables
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Create tables if they don't exist with proper indexing"""
        try:
            logger.info("Ensuring chat logger tables exist...")
            with self.engine.begin() as conn:
                # Table 1: Conversation Messages (main conversation tracking)
                logger.debug("Creating chatbot_conversation_messages table...")
                conn.execute(text("""
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
                        INDEX idx_success (success)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """))
                
                # Table 2: LLM Calls (detailed LLM call tracking)
                logger.debug("Creating chatbot_llm_calls table...")
                conn.execute(text("""
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
                        INDEX idx_success (success)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """))
                
                # Table 3: Conversation Summary (for quick lookups)
                logger.debug("Creating chatbot_conversation_summary table...")
                conn.execute(text("""
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
                        INDEX idx_success_rate (success_rate)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """))
                
                logger.info("Chat logger tables initialized successfully")
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize tables: {e}")
            raise
    
    def start_conversation_message(self, conversation_id: str, user_input: str) -> str:
        """
        Start tracking a new conversation message
        Returns message_id for linking LLM calls
        """
        message_id = str(uuid.uuid4())
        logger.debug(f"Starting conversation message tracking: conversation_id={conversation_id}, message_id={message_id}")
        
        try:
            with self.engine.begin() as conn:
                logger.debug(f"Inserting into chatbot_conversation_messages...")
                conn.execute(text("""
                    INSERT INTO chatbot_conversation_messages (
                        message_id, conversation_id, user_input, final_output,
                        total_llm_calls, total_tokens_used, total_cost, total_duration_ms
                    ) VALUES (
                        :message_id, :conversation_id, :user_input, '',
                        0, 0, 0, 0
                    )
                """), {
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "user_input": user_input
                })
                
                # Update or create conversation summary
                logger.debug(f"Updating chatbot_conversation_summary...")
                conn.execute(text("""
                    INSERT INTO chatbot_conversation_summary (
                        conversation_id, first_message_at, last_message_at, total_messages,
                        last_user_input
                    ) VALUES (
                        :conversation_id, NOW(), NOW(), 1, :user_input
                    ) ON DUPLICATE KEY UPDATE
                        last_message_at = NOW(),
                        total_messages = total_messages + 1,
                        last_user_input = :user_input
                """), {
                    "conversation_id": conversation_id,
                    "user_input": user_input
                })
                
                logger.debug(f"Started tracking message {message_id} for conversation {conversation_id}")
                return message_id
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to start conversation message tracking: {e}")
            return message_id  # Return ID even if logging fails
    
    def log_llm_call(self, llm_call: LLMCall) -> None:
        """Log a single LLM call with detailed metrics"""
        try:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO chatbot_llm_calls (
                        call_id, conversation_id, message_id, model, function_name,
                        input_tokens, output_tokens, total_tokens, cost_estimate, duration_ms,
                        input_text, output_text, success, error_message, metadata
                    ) VALUES (
                        :call_id, :conversation_id, :message_id, :model, :function_name,
                        :input_tokens, :output_tokens, :total_tokens, :cost_estimate, :duration_ms,
                        :input_text, :output_text, :success, :error_message, :metadata
                    )
                """), {
                    "call_id": llm_call.call_id,
                    "conversation_id": llm_call.conversation_id,
                    "message_id": llm_call.message_id,
                    "model": llm_call.model,
                    "function_name": llm_call.function_name,
                    "input_tokens": llm_call.input_tokens,
                    "output_tokens": llm_call.output_tokens,
                    "total_tokens": llm_call.total_tokens,
                    "cost_estimate": llm_call.cost_estimate,
                    "duration_ms": llm_call.duration_ms,
                    "input_text": llm_call.input_text,
                    "output_text": llm_call.output_text,
                    "success": llm_call.success,
                    "error_message": llm_call.error_message,
                    "metadata": llm_call.metadata
                })
                
                # Update conversation message totals
                conn.execute(text("""
                    UPDATE chatbot_conversation_messages 
                    SET total_llm_calls = total_llm_calls + 1,
                        total_tokens_used = total_tokens_used + :total_tokens,
                        total_cost = total_cost + :cost_estimate,
                        total_duration_ms = total_duration_ms + :duration_ms
                    WHERE message_id = :message_id
                """), {
                    "message_id": llm_call.message_id,
                    "total_tokens": llm_call.total_tokens,
                    "cost_estimate": llm_call.cost_estimate,
                    "duration_ms": llm_call.duration_ms
                })
                
                # Update conversation summary
                conn.execute(text("""
                    UPDATE chatbot_conversation_summary 
                    SET total_llm_calls = total_llm_calls + 1,
                        total_tokens_used = total_tokens_used + :total_tokens,
                        total_cost = total_cost + :cost_estimate,
                        total_duration_ms = total_duration_ms + :duration_ms
                    WHERE conversation_id = :conversation_id
                """), {
                    "conversation_id": llm_call.conversation_id,
                    "total_tokens": llm_call.total_tokens,
                    "cost_estimate": llm_call.cost_estimate,
                    "duration_ms": llm_call.duration_ms
                })
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to log LLM call: {e}")
            raise
    
    def complete_conversation_message(
        self,
        message_id: str,
        final_output: str,
        conversation_message: Optional[str] = None,
        intent: Optional[str] = None,
        classification: Optional[str] = None,
        database_used: Optional[str] = None,
        sql_queries: Optional[List[str]] = None,
        sql_results_count: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        debug_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Complete a conversation message with final results"""
        try:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    UPDATE chatbot_conversation_messages 
                    SET final_output = :final_output,
                        conversation_message = :conversation_message,
                        intent = :intent,
                        classification = :classification,
                        database_used = :database_used,
                        sql_queries = :sql_queries,
                        sql_results_count = :sql_results_count,
                        success = :success,
                        error_message = :error_message,
                        debug_info = :debug_info
                    WHERE message_id = :message_id
                """), {
                    "message_id": message_id,
                    "final_output": final_output,
                    "conversation_message": conversation_message,
                    "intent": intent,
                    "classification": classification,
                    "database_used": database_used,
                    "sql_queries": json.dumps(sql_queries) if sql_queries else None,
                    "sql_results_count": sql_results_count,
                    "success": success,
                    "error_message": error_message,
                    "debug_info": json.dumps(debug_info) if debug_info else None
                })
                
                # Update conversation summary with last outputs
                if success:
                    conn.execute(text("""
                        UPDATE chatbot_conversation_summary 
                        SET last_agent_output = :final_output
                        WHERE conversation_id = (
                            SELECT conversation_id FROM chatbot_conversation_messages 
                            WHERE message_id = :message_id
                        )
                    """), {
                        "message_id": message_id,
                        "final_output": final_output
                    })
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to complete conversation message: {e}")
    
    def get_conversation_details(self, conversation_id: str) -> Dict[str, Any]:
        """Get comprehensive details for a conversation"""
        try:
            with self.engine.connect() as conn:
                # Get conversation summary
                summary_result = conn.execute(text("""
                    SELECT * FROM chatbot_conversation_summary 
                    WHERE conversation_id = :conversation_id
                """), {"conversation_id": conversation_id}).fetchone()
                
                # Get all messages
                messages_result = conn.execute(text("""
                    SELECT * FROM chatbot_conversation_messages 
                    WHERE conversation_id = :conversation_id
                    ORDER BY timestamp ASC
                """), {"conversation_id": conversation_id}).fetchall()
                
                # Get all LLM calls
                llm_calls_result = conn.execute(text("""
                    SELECT * FROM chatbot_llm_calls 
                    WHERE conversation_id = :conversation_id
                    ORDER BY timestamp ASC
                """), {"conversation_id": conversation_id}).fetchall()
                
                return {
                    "conversation_id": conversation_id,
                    "summary": dict(summary_result._mapping) if summary_result else None,
                    "messages": [dict(msg._mapping) for msg in messages_result],
                    "llm_calls": [dict(call._mapping) for call in llm_calls_result]
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get conversation details: {e}")
            return {"conversation_id": conversation_id, "error": str(e)}
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversations for monitoring"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT * FROM chatbot_conversation_summary 
                    ORDER BY last_message_at DESC 
                    LIMIT :limit
                """), {"limit": limit}).fetchall()
                
                return [dict(row._mapping) for row in result]
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get recent conversations: {e}")
            return []
    
    @contextmanager
    def track_llm_call(
        self,
        conversation_id: str,
        message_id: str,
        model: str,
        function_name: str,
        input_text: str
    ):
        """
        Context manager for tracking LLM calls with automatic timing and error handling
        Usage:
            with logger.track_llm_call(conv_id, msg_id, "gpt-4o", "classify", input) as call_tracker:
                result = llm.invoke(input_text)
                call_tracker.set_output(result.content, tokens_used, cost)
        """
        call_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create call tracker
        call_tracker = LLMCallTracker(call_id, conversation_id, message_id, model, function_name, input_text)
        
        try:
            yield call_tracker
        except Exception as e:
            # Log failed call
            call_tracker.set_error(str(e))
            self.log_llm_call(call_tracker.get_llm_call(start_time))
            raise
        else:
            # Log successful call
            self.log_llm_call(call_tracker.get_llm_call(start_time))


class LLMCallTracker:
    """Helper class for tracking LLM call details"""
    
    def __init__(self, call_id: str, conversation_id: str, message_id: str, model: str, function_name: str, input_text: str):
        self.call_id = call_id
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.model = model
        self.function_name = function_name
        self.input_text = input_text
        self.output_text = ""
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.cost_estimate = 0.0
        self.success = True
        self.error_message = None
        self.metadata = None
    
    def set_output(self, output_text: str, input_tokens: int, output_tokens: int, cost_estimate: float, metadata: Optional[dict] = None):
        self.output_text = output_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        self.cost_estimate = cost_estimate
        self.success = True
        if metadata is not None:
            self.metadata = json.dumps(metadata)
        else:
            self.metadata = None
    
    def set_error(self, error_message: str):
        self.success = False
        self.error_message = error_message
        self.output_text = f"Error: {error_message}"
        self.metadata = None
    
    def get_llm_call(self, start_time: float) -> LLMCall:
        duration_ms = (time.time() - start_time) * 1000
        return LLMCall(
            call_id=self.call_id,
            conversation_id=self.conversation_id,
            message_id=self.message_id,
            model=self.model,
            function_name=self.function_name,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            cost_estimate=self.cost_estimate,
            duration_ms=duration_ms,
            input_text=self.input_text,
            output_text=self.output_text,
            success=self.success,
            error_message=self.error_message,
            metadata=self.metadata,
            timestamp=datetime.now()
        )


# Global logger instance
_chat_logger = None

def get_chat_logger() -> ChatLogger:
    """Get singleton chat logger instance"""
    global _chat_logger
    if _chat_logger is None:
        _chat_logger = ChatLogger()
    return _chat_logger 