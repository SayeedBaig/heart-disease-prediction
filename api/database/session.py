from sqlalchemy.orm import sessionmaker

from api.database.connection import engine


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db():
    """
    FastAPI dependency that yields a SQLAlchemy session.
    Rolls back automatically on exception and always closes the session.
    """
    db = SessionLocal()

    try:
        yield db

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()