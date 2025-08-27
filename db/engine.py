# db/engine.py
from sqlmodel import SQLModel, create_engine

# 1) Point to your SQLite file (data/app.db).
engine = create_engine("sqlite:///data/app.db", echo=False)

# 2) Create tables once, and return the engine.
def init_db():
    # Import models here so the tables are registered before create_all
    from .models import Company, FinancialMetric, Beta  # noqa: F401
    SQLModel.metadata.create_all(engine)
    return engine
