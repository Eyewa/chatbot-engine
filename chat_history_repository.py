import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

class ChatHistoryRepository:
    """Simple repository for persisting chat history."""

    def __init__(self):
        load_dotenv()
        uri = os.getenv("SQL_DATABASE_URI_LIVE_WRITE")
        if not uri:
            raise RuntimeError("SQL_DATABASE_URI_LIVE_WRITE is not set")
        self.engine = create_engine(uri)

    def save_message(self, message: str):
        # Placeholder implementation
        with self.engine.begin() as conn:
            conn.execute("INSERT INTO chat_history (message) VALUES (:message)", {"message": message})

