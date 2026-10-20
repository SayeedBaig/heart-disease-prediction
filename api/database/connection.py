import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is required. Set it to a PostgreSQL connection URL "
        "(Neon in production); the API will not fall back to SQLite."
    )

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]

# SQL_ECHO=true enables per-statement SQL logging (dev only; never use in production)
_sql_echo = os.getenv("SQL_ECHO", "false").strip().lower() == "true"

engine = create_engine(
    DATABASE_URL,
    echo=_sql_echo,
    pool_pre_ping=True,
)
