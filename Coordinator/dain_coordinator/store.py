"""SQLite-backed registry store (S2).

Persistence boundary only — no business logic. A single connection guarded by a
lock: registry operations are sub-millisecond and the MVP pool is ≤ 100 nodes,
so brief event-loop blocking is an accepted trade-off (async/Postgres is the
post-MVP HA path, spec §5). The schema is plain portable SQL; the only
SQLite-specific bits (WAL pragma) live here.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass

from dain_common.schemas import CapabilityManifest, LedgerEvent, MetricsReport, NodeState

log = logging.getLogger("dain.coordinator.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id           TEXT PRIMARY KEY,
    node_token        TEXT NOT NULL,
    agent_version     TEXT NOT NULL DEFAULT '',
    state             TEXT NOT NULL,
    manifest          TEXT NOT NULL,
    metrics           TEXT,
    score             REAL,
    score_components  TEXT,
    overload_strikes  INTEGER NOT NULL DEFAULT 0,
    uptime_ratio      REAL NOT NULL DEFAULT 1.0,
    failure_rate      REAL NOT NULL DEFAULT 0.0,
    last_seq          INTEGER,
    last_heartbeat    REAL,
    registered_at     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS state_history (
    id          INTEGER PRIMARY KEY,
    node_id     TEXT NOT NULL,
    from_state  TEXT,
    to_state    TEXT NOT NULL,
    reason      TEXT,
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_state_history_node ON state_history (node_id, ts);
CREATE TABLE IF NOT EXISTS ledger (
    id              INTEGER PRIMARY KEY,
    node_id         TEXT NOT NULL,
    job_id          TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    stage_idx       INTEGER NOT NULL,
    attempt         INTEGER NOT NULL,
    partition_id    TEXT NOT NULL,
    tokens_in       INTEGER NOT NULL,
    tokens_out      INTEGER NOT NULL,
    flops_est       REAL NOT NULL,
    compute_seconds REAL NOT NULL,
    gpu_util_avg    REAL,
    cpu_util_avg    REAL,
    energy_kwh_est  REAL,
    outcome         TEXT NOT NULL,
    verified        INTEGER NOT NULL DEFAULT 1,
    verification_note TEXT,
    ts              REAL NOT NULL,
    UNIQUE(job_id, stage_idx, attempt)
);
CREATE INDEX IF NOT EXISTS idx_ledger_node ON ledger (node_id, ts);
CREATE INDEX IF NOT EXISTS idx_ledger_job ON ledger (job_id);
"""


@dataclass
class NodeRow:
    node_id: str
    node_token: str
    agent_version: str
    state: NodeState
    manifest: CapabilityManifest
    metrics: MetricsReport | None
    score: float | None
    score_components: dict[str, float]
    overload_strikes: int
    uptime_ratio: float
    failure_rate: float
    last_seq: int | None
    last_heartbeat: float | None
    registered_at: float

    def heartbeat_ref(self) -> float:
        """Timestamp the timeout math runs against: last heartbeat, else registration."""
        return self.last_heartbeat if self.last_heartbeat is not None else self.registered_at


@dataclass(frozen=True)
class StateChange:
    node_id: str
    from_state: NodeState | None
    to_state: NodeState
    reason: str | None
    ts: float


class SQLiteRegistry:
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        log.info("registry_opened path=%s", self._path)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _require(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("registry is not open")
        return self._conn

    # -- nodes ------------------------------------------------------------------

    def get_node(self, node_id: str) -> NodeRow | None:
        with self._lock:
            cur = self._require().execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,))
            row = cur.fetchone()
        return self._row_from(row) if row is not None else None

    def list_nodes(self, state: NodeState | None = None) -> list[NodeRow]:
        with self._lock:
            if state is None:
                cur = self._require().execute("SELECT * FROM nodes ORDER BY node_id")
            else:
                cur = self._require().execute(
                    "SELECT * FROM nodes WHERE state = ? ORDER BY node_id", (state.value,)
                )
            rows = cur.fetchall()
        return [self._row_from(r) for r in rows]

    def save_node(self, row: NodeRow) -> None:
        with self._lock:
            self._require().execute(
                """
                INSERT OR REPLACE INTO nodes (
                    node_id, node_token, agent_version, state, manifest, metrics,
                    score, score_components, overload_strikes, uptime_ratio,
                    failure_rate, last_seq, last_heartbeat, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.node_id,
                    row.node_token,
                    row.agent_version,
                    row.state.value,
                    row.manifest.model_dump_json(),
                    row.metrics.model_dump_json() if row.metrics is not None else None,
                    row.score,
                    json.dumps(row.score_components),
                    row.overload_strikes,
                    row.uptime_ratio,
                    row.failure_rate,
                    row.last_seq,
                    row.last_heartbeat,
                    row.registered_at,
                ),
            )
            self._require().commit()

    # -- state history ------------------------------------------------------------

    def append_history(self, change: StateChange) -> None:
        with self._lock:
            self._require().execute(
                "INSERT INTO state_history (node_id, from_state, to_state, reason, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    change.node_id,
                    change.from_state.value if change.from_state is not None else None,
                    change.to_state.value,
                    change.reason,
                    change.ts,
                ),
            )
            self._require().commit()

    def history(self, node_id: str) -> list[StateChange]:
        with self._lock:
            cur = self._require().execute(
                "SELECT node_id, from_state, to_state, reason, ts "
                "FROM state_history WHERE node_id = ? ORDER BY ts, id",
                (node_id,),
            )
            rows = cur.fetchall()
        return [
            StateChange(
                node_id=r["node_id"],
                from_state=NodeState(r["from_state"]) if r["from_state"] is not None else None,
                to_state=NodeState(r["to_state"]),
                reason=r["reason"],
                ts=r["ts"],
            )
            for r in rows
        ]

    # -- ledger (S8, spec §15) ------------------------------------------------------

    def append_ledger(
        self, event: LedgerEvent, *, verified: bool = True, note: str | None = None
    ) -> bool:
        """Append one ledger event, deduped on `(job_id, stage_idx, attempt)`.

        Returns True when the row was inserted, False when a replay/retry
        re-delivered an existing key (idempotency contract, §15).
        """
        with self._lock:
            cur = self._require().execute(
                """
                INSERT OR IGNORE INTO ledger (
                    node_id, job_id, model_id, stage_idx, attempt, partition_id,
                    tokens_in, tokens_out, flops_est, compute_seconds,
                    gpu_util_avg, cpu_util_avg, energy_kwh_est, outcome,
                    verified, verification_note, ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.node_id,
                    event.job_id,
                    event.model_id,
                    event.stage_idx,
                    event.attempt,
                    event.partition_id,
                    event.tokens_in,
                    event.tokens_out,
                    event.flops_est,
                    event.compute_seconds,
                    event.gpu_util_avg,
                    event.cpu_util_avg,
                    event.energy_kwh_est,
                    event.outcome.value,
                    1 if verified else 0,
                    note,
                    event.ts,
                ),
            )
            inserted = cur.rowcount > 0
            self._require().commit()
        return inserted

    def ledger_rows(self, *, node_id: str | None = None, since: float | None = None) -> list[dict]:
        """Raw ledger rows as dicts (an id -> db row store; ordering by ts, id).

        `since` filters to events at or after the epoch timestamp.
        """
        clauses: list[str] = []
        params: list = []
        if node_id is not None:
            clauses.append("node_id = ?")
            params.append(node_id)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            cur = self._require().execute(f"SELECT * FROM ledger {where} ORDER BY ts, id", params)
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    # -- helpers --------------------------------------------------------------------

    @staticmethod
    def _row_from(r: sqlite3.Row) -> NodeRow:
        return NodeRow(
            node_id=r["node_id"],
            node_token=r["node_token"],
            agent_version=r["agent_version"],
            state=NodeState(r["state"]),
            manifest=CapabilityManifest.model_validate_json(r["manifest"]),
            metrics=MetricsReport.model_validate_json(r["metrics"]) if r["metrics"] else None,
            score=r["score"],
            score_components=json.loads(r["score_components"]) if r["score_components"] else {},
            overload_strikes=r["overload_strikes"],
            uptime_ratio=r["uptime_ratio"],
            failure_rate=r["failure_rate"],
            last_seq=r["last_seq"],
            last_heartbeat=r["last_heartbeat"],
            registered_at=r["registered_at"],
        )


def utc_now() -> float:
    return time.time()
