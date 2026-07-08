import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# SQL_ECHO=true enables per-statement SQL logging (dev only; never use in production)
_sql_echo = os.getenv("SQL_ECHO", "false").strip().lower() == "true"

engine = create_engine(
    DATABASE_URL,
    echo=_sql_echo,
)