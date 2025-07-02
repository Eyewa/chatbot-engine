"""
Chat Logger - Comprehensive conversation and LLM call tracking
Senior AI Engineer Design: Multi-table architecture with minimal performance impact
"""

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from sqlalchemy import (create_engine, text)
from sqlalchemy.exc import SQLAlchemyError

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
            echo=False,  # Disable SQL echo for performance
        )

        # Initialize tables
        self._ensure_tables_exist()

    def _ensure_tables_exist(self):
        """Create tables if they don't exist with proper indexing"""
        try:
            logger.info("Ensuring chat logger tables exist...")
            with self.engine.begin() as conn:
                # Table: Conversation Messages (minimal chat history)
                logger.debug("Creating chatbot_conversation_messages table...")
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS chatbot_conversation_messages (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        message_id VARCHAR(255) NOT NULL,
                        conversation_id VARCHAR(255) NOT NULL,
                        user_id VARCHAR(255),
                        sender VARCHAR(10) NOT NULL,
                        message_text TEXT NOT NULL,
                        intent VARCHAR(100),
                        context JSON,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_conversation_id (conversation_id),
                        INDEX idx_message_id (message_id),
                        INDEX idx_timestamp (timestamp),
                        INDEX idx_intent (intent)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                    )
                )

                # Table: Conversation Summary (for quick lookups)
                logger.debug("Creating chatbot_conversation_summary table...")
                conn.execute(
                    text(
                        """
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
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                    )
                )

                logger.info("Chat logger tables initialized successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize tables: {e}")
            raise

    def start_conversation_message(
        self, conversation_id: str, user_id: str, message_text: str
    ) -> str:
        """
        Start tracking a new conversation message (user or bot)
        Returns message_id for linking
        """
        message_id = str(uuid.uuid4())
        sender = "user"
        timestamp = datetime.utcnow()
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                    INSERT INTO chatbot_conversation_messages (
                        message_id, conversation_id, user_id, sender, message_text, timestamp
                    ) VALUES (
                        :message_id, :conversation_id, :user_id, :sender, :message_text, :timestamp
                    )
                """
                    ),
                    {
                        "message_id": message_id,
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "sender": sender,
                        "message_text": message_text,
                        "timestamp": timestamp,
                    },
                )
                # Update or create conversation summary
                conn.execute(
                    text(
                        """
                    INSERT INTO chatbot_conversation_summary (
                        conversation_id, first_message_at, last_message_at, total_messages, last_user_input
                    ) VALUES (
                        :conversation_id, :first_message_at, :last_message_at, 1, :last_user_input
                    ) ON DUPLICATE KEY UPDATE
                        last_message_at = :last_message_at,
                        total_messages = total_messages + 1,
                        last_user_input = :last_user_input
                """
                    ),
                    {
                        "conversation_id": conversation_id,
                        "first_message_at": timestamp,
                        "last_message_at": timestamp,
                        "last_user_input": message_text,
                    },
                )
                return message_id
        except SQLAlchemyError as e:
            logger.error(f"Failed to start conversation message tracking: {e}")
            return message_id

    def complete_conversation_message(
        self,
        message_id: str,
        conversation_id: str,
        user_id: str,
        message_text: str,
        intent: Optional[str] = None,
        context: Optional[dict] = None,
        sender: str = "bot",
    ) -> None:
        """
        Log a bot response or update a message with intent/context
        """
        timestamp = datetime.utcnow()
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                    INSERT INTO chatbot_conversation_messages (
                        message_id, conversation_id, user_id, sender, message_text, intent, context, timestamp
                    ) VALUES (
                        :message_id, :conversation_id, :user_id, :sender, :message_text, :intent, :context, :timestamp
                    )
                """
                    ),
                    {
                        "message_id": message_id,
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "sender": sender,
                        "message_text": message_text,
                        "intent": intent,
                        "context": json.dumps(context) if context else None,
                        "timestamp": timestamp,
                    },
                )
                # Update conversation summary with last agent output
                if sender == "bot":
                    conn.execute(
                        text(
                            """
                        UPDATE chatbot_conversation_summary 
                        SET last_agent_output = :message_text, last_message_at = :timestamp
                        WHERE conversation_id = :conversation_id
                    """
                        ),
                        {
                            "conversation_id": conversation_id,
                            "message_text": message_text,
                            "timestamp": timestamp,
                        },
                    )
        except SQLAlchemyError as e:
            logger.error(f"Failed to complete conversation message: {e}")

    def get_conversation_details(self, conversation_id: str) -> Dict[str, Any]:
        """Get comprehensive details for a conversation"""
        try:
            with self.engine.connect() as conn:
                # Get conversation summary
                summary_result = conn.execute(
                    text(
                        """
                    SELECT * FROM chatbot_conversation_summary 
                    WHERE conversation_id = :conversation_id
                """
                    ),
                    {"conversation_id": conversation_id},
                ).fetchone()

                # Get all messages
                messages_result = conn.execute(
                    text(
                        """
                    SELECT * FROM chatbot_conversation_messages 
                    WHERE conversation_id = :conversation_id
                    ORDER BY timestamp ASC
                """
                    ),
                    {"conversation_id": conversation_id},
                ).fetchall()

                return {
                    "conversation_id": conversation_id,
                    "summary": (
                        dict(summary_result._mapping) if summary_result else None
                    ),
                    "messages": [dict(msg._mapping) for msg in messages_result],
                }

        except SQLAlchemyError as e:
            logger.error(f"Failed to get conversation details: {e}")
            return {"conversation_id": conversation_id, "error": str(e)}

    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversations for monitoring"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                    SELECT * FROM chatbot_conversation_summary 
                    ORDER BY last_message_at DESC 
                    LIMIT :limit
                """
                    ),
                    {"limit": limit},
                ).fetchall()

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
        input_text: str,
    ):
        """
        Context manager for tracking LLM calls with automatic timing and error handling
        Usage:
            with logger.track_llm_call(conv_id, msg_id, "gpt-4o", "classify", input) as call_tracker:
                result = llm.invoke(input_text)
                call_tracker.set_output(result.content, tokens_used, cost)
        """
        call_id = str(uuid.uuid4())

        # Create call tracker
        call_tracker = LLMCallTracker(
            call_id, conversation_id, message_id, model, function_name, input_text
        )

        try:
            yield call_tracker
        except Exception:
            # Log failed call
            raise


class LLMCallTracker:
    """Helper class for tracking LLM call details"""

    def __init__(
        self,
        call_id: str,
        conversation_id: str,
        message_id: str,
        model: str,
        function_name: str,
        input_text: str,
    ):
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
        self.start_time = time.time()

    def set_output(
        self,
        output_text: str,
        input_tokens: int,
        output_tokens: int,
        cost_estimate: float,
        metadata: Optional[dict] = None,
    ):
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

    def get_llm_call(self) -> LLMCall:
        duration_ms = (time.time() - self.start_time) * 1000
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
            timestamp=datetime.now(),
        )


# Global logger instance
_chat_logger = None


def get_chat_logger() -> ChatLogger:
    """Get singleton chat logger instance"""
    global _chat_logger
    if _chat_logger is None:
        _chat_logger = ChatLogger()
    return _chat_logger
