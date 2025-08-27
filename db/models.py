from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import date

class Company(SQLModel, table=True):
    __tablename__ = "company"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    ticker: str
    name: str
    industry: Optional[str] = None

class FinancialMetric(SQLModel, table=True):
    __tablename__ = "financialmetric"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    company_id: int = Field(foreign_key="company.id")
    year: int
    revenue: float
    ebitda: float
    fcff: float
    wacc: float
    intrinsic_value: float

class Beta(SQLModel, table=True):
    __tablename__ = "beta"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    company_id: int = Field(foreign_key="company.id")
    model: str
    beta: float
    date: date
