import sqlite3
from pathlib import Path
from src.config import ExperimentResult


class SQLiteStorage:
    def __init__(self, db_path: str = "data/results.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id TEXT PRIMARY KEY,
                    experiment TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    temperature REAL,
                    cost_usd REAL NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    result_json TEXT NOT NULL
                )
            """)

    def save(self, result: ExperimentResult) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.id,
                    result.experiment,
                    result.timestamp.isoformat(),
                    result.model,
                    result.temperature,
                    result.cost_usd,
                    result.latency_ms,
                    result.model_dump_json(),
                ),
            )

    def get_by_experiment(self, experiment: str) -> list[ExperimentResult]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT result_json FROM results WHERE experiment = ? ORDER BY timestamp",
                (experiment,),
            ).fetchall()
        return [ExperimentResult.model_validate_json(row[0]) for row in rows]

    def get_all(self) -> list[ExperimentResult]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT result_json FROM results ORDER BY timestamp DESC"
            ).fetchall()
        return [ExperimentResult.model_validate_json(row[0]) for row in rows]
