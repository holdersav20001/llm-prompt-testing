import sqlite3
from pathlib import Path
from src.config import ExperimentResult, Variant

DEFAULT_SYSTEM_PROMPTS = [
    ("none", ""),
    ("short_role", "You are a helpful assistant."),
    ("detailed_role", "You are an expert technical writer who produces clear, well-structured, accurate responses."),
    ("format_constraints", "You are a helpful assistant. Always respond in bullet points. Keep responses under 100 words."),
    ("persona_constraints", "You are Alex, a senior software engineer with 10 years of experience. Respond in a friendly but direct tone. Structure your response with: 1) Direct answer 2) Brief explanation 3) One concrete example."),
]

DEFAULT_PROMPT_VARIANTS = [
    ("short", "What is 2+2?"),
    ("medium", "Explain the difference between supervised and unsupervised machine learning in 2-3 sentences."),
    ("long", "Write a detailed explanation of how neural networks work, covering: 1) perceptrons, 2) layers, 3) activation functions, 4) backpropagation, and 5) gradient descent."),
]


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
                    run_id TEXT,
                    experiment TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    temperature REAL,
                    cost_usd REAL NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    result_json TEXT NOT NULL
                )
            """)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(results)")]
            if "run_id" not in cols:
                conn.execute("ALTER TABLE results ADD COLUMN run_id TEXT")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS variants (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    paused INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            # Migrate existing variants table: add paused column if absent
            vcols = [r[1] for r in conn.execute("PRAGMA table_info(variants)")]
            if "paused" not in vcols:
                conn.execute("ALTER TABLE variants ADD COLUMN paused INTEGER NOT NULL DEFAULT 0")
            self._seed_variants(conn)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    review_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

    def _seed_variants(self, conn):
        sp_count = conn.execute("SELECT COUNT(*) FROM variants WHERE type = 'system_prompt'").fetchone()[0]
        if sp_count == 0:
            for name, content in DEFAULT_SYSTEM_PROMPTS:
                v = Variant(type="system_prompt", name=name, content=content)
                conn.execute(
                    "INSERT INTO variants (id, type, name, content, paused, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (v.id, v.type, v.name, v.content, int(v.paused), v.created_at.isoformat()),
                )

        pv_count = conn.execute("SELECT COUNT(*) FROM variants WHERE type = 'prompt_length'").fetchone()[0]
        if pv_count == 0:
            for name, content in DEFAULT_PROMPT_VARIANTS:
                v = Variant(type="prompt_length", name=name, content=content)
                conn.execute(
                    "INSERT INTO variants (id, type, name, content, paused, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (v.id, v.type, v.name, v.content, int(v.paused), v.created_at.isoformat()),
                )

    # ── Variants CRUD ─────────────────────────────────────────────────────────

    def get_variants(self, type: str) -> list[Variant]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, type, name, content, paused, created_at FROM variants WHERE type = ? ORDER BY created_at",
                (type,),
            ).fetchall()
        return [Variant(id=r[0], type=r[1], name=r[2], content=r[3], paused=bool(r[4]), created_at=r[5]) for r in rows]

    def save_variant(self, variant: Variant) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO variants (id, type, name, content, paused, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (variant.id, variant.type, variant.name, variant.content, int(variant.paused), variant.created_at.isoformat()),
            )

    def update_variant(self, id: str, name: str, content: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE variants SET name = ?, content = ? WHERE id = ?",
                (name, content, id),
            )

    def set_variant_paused(self, id: str, paused: bool) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE variants SET paused = ? WHERE id = ?", (int(paused), id))

    def delete_variant(self, id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM variants WHERE id = ?", (id,))

    # ── Results ───────────────────────────────────────────────────────────────

    def save(self, result: ExperimentResult) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO results
                   (id, run_id, experiment, timestamp, model, temperature, cost_usd, latency_ms, result_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.id,
                    result.run_id,
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

    def get_by_run_id(self, run_id: str) -> list[ExperimentResult]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT result_json FROM results WHERE run_id = ? ORDER BY timestamp",
                (run_id,),
            ).fetchall()
        return [ExperimentResult.model_validate_json(row[0]) for row in rows]

    def get_runs(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT run_id, experiment, MIN(timestamp) as started_at,
                       COUNT(*) as result_count, SUM(cost_usd) as total_cost,
                       GROUP_CONCAT(DISTINCT model) as models
                FROM results
                WHERE run_id IS NOT NULL
                GROUP BY run_id
                ORDER BY started_at DESC
            """).fetchall()
        return [
            {
                "run_id": r[0],
                "experiment": r[1],
                "started_at": r[2],
                "results": r[3],
                "total_cost": round(r[4], 6),
                "models": r[5],
            }
            for r in rows
        ]

    def get_all(self) -> list[ExperimentResult]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT result_json FROM results ORDER BY timestamp DESC"
            ).fetchall()
        return [ExperimentResult.model_validate_json(row[0]) for row in rows]

    # ── Comparisons ───────────────────────────────────────────────────────────

    def save_comparison(self, result_ids: list, model: str, comparison_text: str) -> None:
        import uuid, json
        from datetime import datetime
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS comparisons (id TEXT PRIMARY KEY, result_ids TEXT NOT NULL, model TEXT NOT NULL, comparison_text TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO comparisons (id, result_ids, model, comparison_text, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), json.dumps(result_ids), model, comparison_text, datetime.now().isoformat()),
            )

    def get_comparisons(self) -> list[dict]:
        import json
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS comparisons (id TEXT PRIMARY KEY, result_ids TEXT NOT NULL, model TEXT NOT NULL, comparison_text TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            rows = conn.execute(
                "SELECT result_ids, model, comparison_text, created_at FROM comparisons ORDER BY created_at DESC"
            ).fetchall()
        return [{"result_ids": json.loads(r[0]), "model": r[1], "comparison_text": r[2], "created_at": r[3]} for r in rows]

    # ── Reviews ───────────────────────────────────────────────────────────────

    def save_review(self, run_id: str, model: str, review_text: str) -> None:
        import uuid
        from datetime import datetime
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO reviews (id, run_id, model, review_text, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), run_id, model, review_text, datetime.now().isoformat()),
            )

    def get_reviews(self, run_id: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT model, review_text, created_at FROM reviews WHERE run_id = ? ORDER BY created_at DESC",
                (run_id,),
            ).fetchall()
        return [{"model": r[0], "review_text": r[1], "created_at": r[2]} for r in rows]
