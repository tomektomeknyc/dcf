# SQLite Backend – Production-Ready 2‑Week Plan

*Last updated: 10 Aug 2025 (Europe/Warsaw)*

## 1) Objectives

- Add a **persistent, production-grade** SQLite backend to your DCF app without slowing normal reruns.
- Use DB as a **smart cache + audit log** for FF5 factors, Damodaran betas, computed betas (CAPM/FF5/Damo), WACC, EV/EBITDA, FCFE, and Intrinsic Values.
- Keep Streamlit UI logic unchanged where possible; hydrate state from DB at start of run, persist results after computations.

---

## 2) Architecture Overview

- **Frontend/UI:** Streamlit (existing app.py and tabs)
- **Logic Layer:** existing modules (wacc.py, ev\_ebitda.py, fcfe.py, dcf\_valuation.py, damodaran.py, scrape\_ff5.py, etc.)
- **Data Layer:** SQLite via SQLAlchemy/SQLModel; Alembic migrations; WAL mode; indexes.
- **Services Layer:** thin wrappers that decide **read from DB or compute** and then **persist**.

```text
project/
  app.py
  db/
    __init__.py
    engine.py          # engine, session factory, PRAGMAs, helpers
    models.py          # SQLModel models & relationships
    repo.py            # read/write helpers (CRUD)
    queries.py         # heavier SELECTs/materializations
    migrations/
      env.py
      versions/
        <timestamp>_init.py
  services/
    ff5_service.py     # get_or_fetch_ff5(region)
    betas_service.py   # compute+persist CAPM/FF5/Damo betas
    wacc_service.py    # compute+persist WACC rows
    iv_service.py      # compute+persist intrinsic values
  scripts/
    refresh_ff5.py
    backup_db.py
  tests/
    test_repo.py
```

---

## 3) Database Setup & Tuning

**SQLite PRAGMAs (set once per connection):**

- `PRAGMA journal_mode = WAL;`
- `PRAGMA synchronous = NORMAL;`
- `PRAGMA foreign_keys = ON;`
- `PRAGMA temp_store = MEMORY;`

**Connection pattern:** one global engine, short-lived sessions per request.

---

## 4) Schema (Initial)

> Types assume SQLModel/SQLAlchemy. Timestamps in UTC. Monetary values as REAL (float) for speed; use DECIMAL if strict accounting is needed.

### 4.1 Core Reference

- **tickers** `(id PK, symbol TEXT UNIQUE, name TEXT, region TEXT, industry TEXT, created_at, updated_at)`
- **prices** `(id PK, ticker_id FK, date DATE, adj_close REAL, UNIQUE(ticker_id,date))`
- **ff5\_factors** `(id PK, region TEXT, date DATE, mktrf REAL, smb REAL, hml REAL, rmw REAL, cma REAL, rf REAL, UNIQUE(region,date))`
- **damodaran\_industry\_betas** `(id PK, region TEXT, industry TEXT, beta REAL, source TEXT, asof DATE, UNIQUE(region,industry,asof))`

### 4.2 Computed Results

- **betas** `(id PK, ticker_id FK, method TEXT CHECK(method IN ('CAPM','FF5','DAMO')), asof DATE, beta REAL, details_json TEXT, UNIQUE(ticker_id,method,asof))`
- **wacc\_runs** `(id PK, ticker_id FK, year INT, method TEXT, re REAL, rd_at REAL, w_e REAL, w_d REAL, wacc REAL, inputs_json TEXT, created_at)`
- **intrinsic\_values** `(id PK, ticker_id FK, method TEXT, asof DATE, value_ps REAL, equity_value REAL, shares_outstanding REAL, details_json TEXT)`

### 4.3 Utility / Cache

- **app\_settings** `(key TEXT PRIMARY KEY, value_json TEXT, updated_at)`
- **data\_cache** `(key TEXT PRIMARY KEY, value_json TEXT, updated_at)`

**Indexes:**

- `prices (ticker_id,date)`
- `ff5_factors (region,date)`
- `betas (ticker_id,method,asof)`
- `wacc_runs (ticker_id,year,method)`

---

## 5) Models (SQLModel hints)

- `Ticker`, `Price`, `FF5Factor`, `DamoIndustryBeta`, `Beta`, `WACCRun`, `IntrinsicValue`, `AppSetting`, `DataCache`.
- Keep relationships minimal (FKs), prefer explicit joins in `queries.py` for clarity.

---

## 6) Repository Layer (db/repo.py)

**Read/Write primitives:**

- `get_or_create_ticker(symbol, name=None, region=None, industry=None)`
- `bulk_upsert_ff5(region, rows)`
- `load_ff5_window(region, date_from, date_to)`
- `get_damodaran_beta(region, industry, asof=None)`
- `save_damodaran_beta(region, industry, beta, asof, source)`
- `save_beta(ticker_id, method, asof, beta, details_json)` / `load_latest_beta(ticker_id, method)`
- `save_wacc_run(ticker_id, year, method, re, rd_at, w_e, w_d, wacc, inputs_json)` / `load_wacc_rows(ticker_id, year=None)`
- `save_intrinsic_value(ticker_id, method, asof, value_ps, equity_value, shares_outstanding, details_json)` / `load_intrinsic_latest(ticker_id)`

**Rules:**

- Repository returns **pandas DataFrames** or simple dataclasses where convenient.
- All repo functions accept a `session` or create one internally via context manager.

---

## 7) Services Layer (Compute + Persist)

**ff5\_service.py**

- `get_or_fetch_ff5(region)` → repo check by date range; if missing, call existing `scrape_ff5.py`, normalize Monthly, persist.

**betas\_service.py**

- `compute_capm_beta(ticker)` → hydrate prices + FF5, resample monthly, run regression, `save_beta(..., method='CAPM')`.
- `compute_ff5_beta(ticker)` → multivariate regression; store market beta in `betas` and full vector in `details_json`.
- `compute_damo_beta(ticker)` → look up industry beta by region+industry; store in `betas (method='DAMO')`.

**wacc\_service.py**

- `build_wacc_rows(ticker, year, methods)` → compute Re per method, Rd\_after\_tax, weights, WACC; upsert into `wacc_runs`.

**iv\_service.py**

- `compute_intrinsic_values(ticker, methods, inputs)` → call your `calculate_intrinsic_value(...)`, persist into `intrinsic_values`.

**Pattern:** check DB freshness → compute if miss/stale → persist → return to UI.

---

## 8) Integration Map (from current code to DB-backed)

- **FF5 factors**: replace `st.session_state['ffs']` warm-up with `ff5_service.get_or_fetch_ff5(region)`; keep a DataFrame in memory for the session.
- **CAPM/FF5 betas**: after computing, call `save_beta(...)`; when redrawing charts, try `load_latest_beta(...)` first.
- **Damodaran**: on region change, `get_damodaran_beta(...)` for each industry; cache in state for plotting.
- **WACC**: build via `wacc_service.build_wacc_rows(...)`, persist; UI queries `wacc_runs` to display.
- **Intrinsic Value**: after LLM/fallback projections and discounting, `save_intrinsic_value(...)`.

**Important:** **Session remains the hot path** — DB hydrates at start; charts read from in-memory DataFrames during a run.

---

## 9) Performance & Consistency

- Align to **Monthly** frequency for FF5 & CAPM regressions.
- Use `@st.cache_data` on read-most service functions that only touch DB.
- Avoid N× tiny queries; issue **bulk** upserts/reads.
- Add `CHECK (method IN (...))` and `UNIQUE(...)` constraints to prevent duplicates.

---

## 10) Deployment Notes (Streamlit Cloud)

- Bundle `app.db` in a writable path (e.g., `./data/app.db`); ensure the folder exists on boot.
- Set `DATABASE_URL=sqlite:///data/app.db`.
- On first boot, run Alembic `upgrade head` (via `scripts/postdeploy.sh` or in-app guard).
- Backups: `scripts/backup_db.py` → copy + `VACUUM` weekly.

---

## 11) Testing & QA

- **Golden-data tests**: fix a tiny CSV of FF5 + prices; assert betas and WACC within tolerance.
- **Repo tests**: insert & read back rows; check constraints and indexes.
- **Service tests**: simulate compute → persist → reload → chart.

---

## 12) 2‑Week Timeline (pragmatic)

**Day 1–2**

- Engine + PRAGMAs; models; initial migration; seed FF5 (1 region) + sample tickers.

**Day 3–4**

- Repo functions (FF5, betas, WACC, IV). Wire FF5 service to app hydration.

**Day 5–6**

- CAPM/FF5 beta services; persist + read in UI; align monthly.

**Day 7–8**

- Damodaran beta load/save; WACC service; UI table reads persisted rows.

**Day 9–10**

- Intrinsic value service; persist latest per method/ticker; chart reads from in-memory hydrated frame.

**Day 11–12**

- Scripts (refresh\_ff5, backup\_db); indexes; smoke tests; cache tuning.

**Day 13–14**

- Polish: error handling, empty-state guards, docstrings; CI hooks; final QA.

---

## 13) Copy‑Paste Stubs

### 13.1 db/engine.py

```python
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel

DATABASE_URL = "sqlite:///data/app.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.execute("PRAGMA temp_store=MEMORY;")
    cursor.close()

def get_session():
    return Session(engine)

def init_db():
    SQLModel.metadata.create_all(engine)
```

### 13.2 db/models.py (excerpt)

```python
from sqlmodel import SQLModel, Field, Relationship
from typing import Optional
from datetime import date, datetime

class Ticker(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True, unique=True)
    name: Optional[str] = None
    region: Optional[str] = Field(index=True, default=None)
    industry: Optional[str] = Field(index=True, default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class FF5Factor(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    region: str = Field(index=True)
    date: date = Field(index=True)
    mktrf: float
    smb: float
    hml: float
    rmw: float
    cma: float
    rf: float
```

### 13.3 db/repo.py (excerpt)

```python
from .engine import get_session
from .models import Ticker, FF5Factor
from sqlmodel import select
from typing import List

def get_or_create_ticker(symbol: str, **kwargs) -> Ticker:
    with get_session() as s:
        t = s.exec(select(Ticker).where(Ticker.symbol==symbol)).first()
        if not t:
            t = Ticker(symbol=symbol, **kwargs)
            s.add(t); s.commit(); s.refresh(t)
        return t

def bulk_upsert_ff5(region: str, rows: List[dict]):
    with get_session() as s:
        for r in rows:
            rec = FF5Factor(region=region, **r)
            s.merge(rec)
        s.commit()
```

---

## 14) Risks & Mitigations

- **DB contention (multi-user):** use short sessions; WAL; avoid long transactions.
- **Data drift (daily vs monthly):** standardize on **monthly**; store frequency in settings.
- **Rerun latency:** hydrate once; keep heavy objects in memory for the session.
- **Migration pain:** start with a stable v1 schema; add Alembic from day one.

---

## 15) Next Steps

1. Confirm file tree & create empty modules.
2. I’ll generate full `models.py` and repo/service stubs tailored to your existing column names.
3. Wire the first service (`ff5_service.get_or_fetch_ff5`) and replace the current warm-up in `app.py`.
4. Iterate method-by-method (CAPM → FF5 → Damo → WACC → IV) with visible checkpoints.

---

If you want, I can now scaffold these files with runnable stubs so you can commit and iterate page-by-page in Streamlit.

