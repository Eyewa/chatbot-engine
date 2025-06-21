from sqlalchemy import create_engine, text

# Replace with your actual DB URI
db_uri = "mysql+pymysql://read_only:Aukdfduyje983idbj@db.eyewa.internal:3306/eyewa_live"

try:
    engine = create_engine(db_uri)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT NOW();"))  # Wrap query with text()
        for row in result:
            print("✅ DB Connected. Server Time:", row[0])
except Exception as e:
    print("❌ Error connecting to DB:", e)
