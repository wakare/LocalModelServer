from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

RETENTION_SECONDS = 30 * 24 * 3600
MAX_ROWS = 200_000
_PRUNE_EVERY = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    caller TEXT NOT NULL,
    model TEXT NOT NULL,
    stream INTEGER NOT NULL,
    priority INTEGER NOT NULL,
    status INTEGER NOT NULL,
    error_code TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    queue_ms INTEGER,
    total_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS requests_started ON requests(started_at);
"""


class RequestHistory:
    """Request metadata only (never prompts or outputs), kept for a bounded period."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._inserts = 0
        self._db = sqlite3.connect(
            str(path) if path else ":memory:", check_same_thread=False, isolation_level=None
        )
        self._db.executescript(_SCHEMA)
        self.prune()

    def record(
        self,
        *,
        started_at: float,
        caller: str,
        model: str,
        stream: bool,
        priority: int,
        status: int,
        error_code: str = "",
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        queue_ms: int | None = None,
        total_ms: int,
    ) -> None:
        try:
            with self._lock:
                self._db.execute(
                    "INSERT INTO requests (started_at, caller, model, stream, priority, status,"
                    " error_code, prompt_tokens, completion_tokens, queue_ms, total_ms)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        started_at,
                        caller[:120],
                        model[:200],
                        int(stream),
                        priority,
                        status,
                        error_code[:60],
                        prompt_tokens,
                        completion_tokens,
                        queue_ms,
                        total_ms,
                    ),
                )
                self._inserts += 1
                due = self._inserts % _PRUNE_EVERY == 0
            if due:
                self.prune()
        except sqlite3.Error as error:
            logging.warning("Request history write failed: %s", error)

    def prune(self, now: float | None = None) -> None:
        cutoff = (now or time.time()) - RETENTION_SECONDS
        try:
            with self._lock:
                self._db.execute("DELETE FROM requests WHERE started_at < ?", (cutoff,))
                self._db.execute(
                    "DELETE FROM requests WHERE id <= (SELECT MAX(id) FROM requests) - ?",
                    (MAX_ROWS,),
                )
        except sqlite3.Error as error:
            logging.warning("Request history prune failed: %s", error)

    def summary(self, hours: float, limit: int = 200, now: float | None = None) -> dict:
        since = (now or time.time()) - hours * 3600
        with self._lock:
            totals = self._db.execute(
                "SELECT COUNT(*), SUM(status = 200), SUM(status != 200),"
                " SUM(prompt_tokens), SUM(completion_tokens), AVG(total_ms),"
                " AVG(CASE WHEN status = 200 THEN total_ms END)"
                " FROM requests WHERE started_at >= ?",
                (since,),
            ).fetchone()
            callers = self._db.execute(
                "SELECT caller, COUNT(*), SUM(status != 200),"
                " SUM(prompt_tokens), SUM(completion_tokens), AVG(total_ms)"
                " FROM requests WHERE started_at >= ? GROUP BY caller ORDER BY COUNT(*) DESC",
                (since,),
            ).fetchall()
            rows = self._db.execute(
                "SELECT started_at, caller, model, stream, priority, status, error_code,"
                " prompt_tokens, completion_tokens, queue_ms, total_ms"
                " FROM requests WHERE started_at >= ? ORDER BY id DESC LIMIT ?",
                (since, limit),
            ).fetchall()
        count, ok, failed, prompt, completion, average, average_ok = totals
        return {
            "window_hours": hours,
            "retention_days": RETENTION_SECONDS // 86400,
            "totals": {
                "requests": count,
                "succeeded": ok or 0,
                "failed": failed or 0,
                "prompt_tokens": prompt or 0,
                "completion_tokens": completion or 0,
                "average_ms": round(average) if average is not None else None,
                "average_success_ms": round(average_ok) if average_ok is not None else None,
            },
            "callers": [
                {
                    "caller": caller,
                    "requests": n,
                    "failed": bad or 0,
                    "prompt_tokens": p or 0,
                    "completion_tokens": c or 0,
                    "average_ms": round(avg) if avg is not None else None,
                }
                for caller, n, bad, p, c, avg in callers
            ],
            "recent": [
                {
                    "started_at": row[0],
                    "caller": row[1],
                    "model": row[2],
                    "stream": bool(row[3]),
                    "priority": row[4],
                    "status": row[5],
                    "error_code": row[6],
                    "prompt_tokens": row[7],
                    "completion_tokens": row[8],
                    "queue_ms": row[9],
                    "total_ms": row[10],
                }
                for row in rows
            ],
        }

    def close(self) -> None:
        with self._lock:
            self._db.close()
