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
        uri = os.getenv("SQL_DATABASE_URI_LIVE_WRITE")
        if not uri:
            raise ValueError("SQL_DATABASE_URI_LIVE_WRITE is not set")
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

    def fetch_history(self, conversation_id: str, limit: Optional[int] = None) -> List[str]:
        """Return chat history as a list alternating user and assistant messages."""
        query = """
            SELECT user_input, agent_output
            FROM chatbot_engine_query_logs
            WHERE conversation_id = :conversation_id
            ORDER BY id ASC
        """
        params = {"conversation_id": conversation_id}
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        with self.engine.connect() as conn:
            result = conn.execute(
                text(query),
                params,
            ).mappings()  # 👈 This makes each row accessible by column name

            history: List[str] = []
            for row in result:
                history.append(row["user_input"])
                history.append(row["agent_output"])
            return history

