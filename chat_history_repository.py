# Repository for persisting chat history to MySQL
import os
import json
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

class ChatHistoryRepository:
    """Simple repository that saves and retrieves chat history."""

    def __init__(self):
        uri = os.getenv("SQL_DATABASE_URI_LIVE")
        if not uri:
            raise ValueError("SQL_DATABASE_URI_LIVE is not set")
        self.engine = create_engine(uri)

    def save_message(
        self,
        conversation_id: str,
        user_input: str,
        agent_output: str,
        *,
        intent: Optional[str] = None,
        debug_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a single interaction along with optional debug metadata."""
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO chatbot_engine_query_logs (
                        conversation_id,
                        user_input,
                        agent_output,
                        intent,
                        debug_info
                    ) VALUES (
                        :conversation_id,
                        :user_input,
                        :agent_output,
                        :intent,
                        :debug_info
                    )
                    """
                ),
                {
                    "conversation_id": conversation_id,
                    "user_input": user_input,
                    "agent_output": agent_output,
                    "intent": intent,
                    "debug_info": json.dumps(debug_info) if debug_info is not None else None,
                },
            )

    def fetch_history(self, conversation_id: str) -> List[str]:
        """Return chat history as a list alternating user and assistant messages."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT user_input, agent_output
                    FROM chatbot_engine_query_logs
                    WHERE conversation_id = :conversation_id
                    ORDER BY id ASC
                    """
                ),
                {"conversation_id": conversation_id},
            )
            history: List[str] = []
            for row in rows:
                history.append(row["user_input"])
                history.append(row["agent_output"])
            return history
