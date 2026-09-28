"""Store-and-forward outbox. Plant networks drop out (basements, cable trays, Wi-Fi dead spots near switchgear);
results are written to local SQLite first and pushed when the link is up. Each result carries an idempotency
key so a retry after a lost ACK never creates a duplicate work order on the platform.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Callable, Iterable

# The platform (patrol-platform-spring) accepts a batch and returns the keys it has stored (new or already seen).
Sender = Callable[[list[dict]], set[str]]


def result_key(mission_id: str, checkpoint_id: str) -> str:
    """One result per checkpoint per mission; retries of the same capture reuse the key."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"patrol:{mission_id}:{checkpoint_id}"))


class Outbox:
    def __init__(self, path: str | Path = ":memory:"):
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS outbox (key TEXT PRIMARY KEY, body TEXT NOT NULL, "
                        "attempts INTEGER NOT NULL DEFAULT 0, sent INTEGER NOT NULL DEFAULT 0)")

    def put(self, result: dict) -> None:
        # INSERT OR REPLACE: a re-inspection of the same checkpoint in the same mission overwrites the pending one
        self.db.execute("INSERT OR REPLACE INTO outbox(key, body) VALUES (?, ?)", (result["idempotencyKey"], json.dumps(result)))
        self.db.commit()

    def pending(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute("SELECT body FROM outbox WHERE sent = 0 ORDER BY rowid LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def flush(self, send: Sender, batch: int = 50) -> dict:
        """Push pending results. Network errors leave rows pending; only keys the server confirms are marked sent."""
        sent = failed = 0
        while rows := self.pending(batch):
            keys = [r["idempotencyKey"] for r in rows]
            self.db.executemany("UPDATE outbox SET attempts = attempts + 1 WHERE key = ?", [(k,) for k in keys])
            try:
                acked = send(rows)
            except (ConnectionError, TimeoutError):
                failed += len(rows)
                self.db.commit()
                break
            self.db.executemany("UPDATE outbox SET sent = 1 WHERE key = ?", [(k,) for k in keys if k in acked])
            self.db.commit()
            sent += len(acked & set(keys))
            if not acked & set(keys):
                break                               # server rejected the whole batch; stop and surface it
        return {"sent": sent, "failed": failed, "pending": len(self.pending(10_000))}


def http_sender(base_url: str, timeout: float = 5.0) -> Sender:
    import httpx

    def send(rows: Iterable[dict]) -> set[str]:
        try:
            r = httpx.post(f"{base_url}/api/results", json=list(rows), timeout=timeout)
        except httpx.TransportError as e:
            raise ConnectionError(str(e)) from e
        r.raise_for_status()
        return set(r.json()["stored"])

    return send
