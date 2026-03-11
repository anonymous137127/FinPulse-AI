from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ---------------- MYSQL CONFIG ----------------

# Special characters in password must be URL encoded
DATABASE_URL = "mysql+pymysql://root:Malware%40127%23137@localhost:3306/finpulse"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)