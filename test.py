from sqlalchemy import create_engine, text
import os

# DB URI is now read from the TEST_DB_URI environment variable
db_uri = os.getenv("TEST_DB_URI")

try:
    engine = create_engine(db_uri)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT NOW();"))  # Wrap query with text()
        for row in result:
            print("✅ DB Connected. Server Time:", row[0])
except Exception as e:
    print("❌ Error connecting to DB:", e)
