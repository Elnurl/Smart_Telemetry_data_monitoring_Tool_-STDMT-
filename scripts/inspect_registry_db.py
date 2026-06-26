import sqlite3
from pathlib import Path

path = Path("data/model_registry.sqlite")
print("exists:", path.exists(), path)
if not path.exists():
    raise SystemExit(0)

conn = sqlite3.connect(path)
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("tables:", cur.fetchall())
cur.execute("PRAGMA table_info(retrain_signals)")
print("retrain_signals:", cur.fetchall())
conn.close()
