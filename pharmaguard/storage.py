"""
Persistent event store for PharmaGuard (SQLite, standard library only).

Tables
  events  one row per recorded alert (REVIEW / ALERT): identifiers, camera, track, incident, category,
          severity, zone, score, reasons, evidence clip, alarm acknowledgement, and staff review outcome.
  audit   every operator/alarm action (raised, acknowledged, reset, test, review updates).

Durability: the database lives on local disk. ``mirror()`` copies it to ``mirror_dir`` (e.g. Google Drive)
with SQLite's online backup API, which is safe while the database is in use.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json
import logging
import os
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uid        TEXT UNIQUE NOT NULL,
    created_at       TEXT NOT NULL,          -- local wall-clock ISO time
    created_epoch    REAL NOT NULL,
    camera_id        TEXT NOT NULL,
    tracker_id       INTEGER,
    incident_id      INTEGER,
    category         TEXT NOT NULL,          -- e.g. SUSPICION_ALERT, SUSPICION_REVIEW
    severity         TEXT NOT NULL,          -- WARNING | ALERT
    zone             TEXT,
    score            REAL,
    reasons          TEXT,                   -- JSON list
    title            TEXT,
    message          TEXT,
    clip_path        TEXT,
    clip_status      TEXT NOT NULL DEFAULT 'none',   -- none|recording|saved|saved_not_browser_playable|unavailable|failed
    clip_codec       TEXT,
    acknowledged_by  TEXT,
    acknowledged_at  TEXT,
    reset_at         TEXT,
    review_status    TEXT NOT NULL DEFAULT 'open',   -- open|confirmed|false_alarm|dismissed
    review_note      TEXT,
    reviewed_by      TEXT,
    reviewed_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_epoch);
CREATE TABLE IF NOT EXISTS audit (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL,
    at_epoch  REAL NOT NULL,
    action    TEXT NOT NULL,
    operator  TEXT,
    details   TEXT
);
"""

REVIEW_STATUSES = ("open", "confirmed", "false_alarm", "dismissed")


@dataclass
class EventRecord:
    event_uid: str
    created_at: str
    camera_id: str
    tracker_id: int
    incident_id: int
    category: str
    severity: str
    zone: str
    score: float
    reasons: List[str] = field(default_factory=list)
    title: str = ""
    message: str = ""
    clip_path: Optional[str] = None
    clip_status: str = "none"
    clip_codec: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    reset_at: Optional[str] = None
    review_status: str = "open"
    review_note: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EventRecord":
        d = dict(row)
        d["reasons"] = json.loads(d.get("reasons") or "[]")
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _now() -> tuple:
    t = time.time()
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"), t


class EventStore:
    """Thread-safe SQLite event store (one connection, serialised by a lock)."""

    def __init__(self, data_dir: str, camera_id: str = "webcam-01", mirror_dir: Optional[str] = None):
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "pharmaguard.db")
        self.camera_id = camera_id
        self.mirror_dir = mirror_dir
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                               (str(SCHEMA_VERSION),))

    # ------------------------------------------------------------------ writes
    def record_alert(self, alert: Any) -> bool:
        """Insert a SecurityAlert. Idempotent on ``alert_id`` (returns False if it already exists)."""
        iso, epoch = _now()
        with self._lock:
            cur = self._conn.execute(
                """INSERT OR IGNORE INTO events (event_uid, created_at, created_epoch, camera_id, tracker_id,
                       incident_id, category, severity, zone, score, reasons, title, message)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (alert.alert_id, alert.timestamp_iso or iso, alert.epoch_timestamp or epoch, self.camera_id,
                 alert.tracker_id, getattr(alert, "incident_id", 0), getattr(alert, "event_type", "") or "ALERT",
                 alert.severity, alert.zone_name, float(getattr(alert, "score", 0.0)),
                 json.dumps(list(getattr(alert, "reasons", []))), alert.title, alert.message))
            return cur.rowcount == 1

    def set_clip(self, event_uid: str, status: str, path: Optional[str] = None, codec: Optional[str] = None) -> None:
        with self._lock:
            self._conn.execute("UPDATE events SET clip_status=?, clip_path=COALESCE(?, clip_path), "
                               "clip_codec=COALESCE(?, clip_codec) WHERE event_uid=?",
                               (status, path, codec, event_uid))

    def mark_acknowledged(self, event_uids: List[str], operator: str) -> None:
        iso, _ = _now()
        with self._lock:
            self._conn.executemany(
                "UPDATE events SET acknowledged_by=?, acknowledged_at=? WHERE event_uid=? AND acknowledged_at IS NULL",
                [(operator, iso, uid) for uid in event_uids])

    def mark_reset(self, event_uids: List[str]) -> None:
        iso, _ = _now()
        with self._lock:
            self._conn.executemany("UPDATE events SET reset_at=? WHERE event_uid=? AND reset_at IS NULL",
                                   [(iso, uid) for uid in event_uids])

    def set_review(self, event_uid: str, status: str, operator: str, note: str = "") -> bool:
        """Staff outcome for an event: confirmed / false_alarm / dismissed / open."""
        if status not in REVIEW_STATUSES:
            raise ValueError(f"review status must be one of {REVIEW_STATUSES}")
        iso, _ = _now()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE events SET review_status=?, review_note=?, reviewed_by=?, reviewed_at=? WHERE event_uid=?",
                (status, note, operator, iso, event_uid))
        if cur.rowcount:
            self.audit("EVENT_REVIEWED", operator, {"event_uid": event_uid, "status": status, "note": note})
        return cur.rowcount == 1

    def audit(self, action: str, operator: Optional[str], details: Optional[Dict] = None) -> None:
        iso, epoch = _now()
        with self._lock:
            self._conn.execute("INSERT INTO audit (at, at_epoch, action, operator, details) VALUES (?,?,?,?,?)",
                               (iso, epoch, action, operator, json.dumps(details or {}, default=str)))

    # ------------------------------------------------------------------ reads
    def get_event(self, event_uid: str) -> Optional[EventRecord]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM events WHERE event_uid=?", (event_uid,)).fetchone()
        return EventRecord.from_row(row) if row else None

    def list_events(self, limit: int = 50, severity: Optional[str] = None,
                    review_status: Optional[str] = None) -> List[EventRecord]:
        q, args = "SELECT * FROM events WHERE 1=1", []
        if severity:
            q += " AND severity=?"
            args.append(severity)
        if review_status:
            q += " AND review_status=?"
            args.append(review_status)
        q += " ORDER BY created_epoch DESC, id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [EventRecord.from_row(r) for r in rows]

    def list_audit(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "details": json.loads(r["details"] or "{}")} for r in rows]

    def counts(self) -> Dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT severity, COUNT(*) AS n FROM events GROUP BY severity").fetchall()
        return {r["severity"]: r["n"] for r in rows}

    # ------------------------------------------------------------------ maintenance
    def purge_older_than(self, days: float) -> int:
        """Delete events (and their clip files) older than ``days``. Returns events deleted."""
        cutoff = time.time() - days * 86400
        with self._lock:
            rows = self._conn.execute("SELECT clip_path FROM events WHERE created_epoch < ?", (cutoff,)).fetchall()
            for r in rows:
                if r["clip_path"]:
                    try:
                        os.remove(r["clip_path"])
                    except OSError:
                        pass
            cur = self._conn.execute("DELETE FROM events WHERE created_epoch < ?", (cutoff,))
            self._conn.execute("DELETE FROM audit WHERE at_epoch < ?", (cutoff,))
        if cur.rowcount:
            self.audit("RETENTION_PURGE", "system", {"deleted_events": cur.rowcount, "days": days})
        return cur.rowcount

    def mirror(self) -> Optional[str]:
        """Copy the live database to ``mirror_dir`` using the SQLite backup API. Returns the copy path."""
        if not self.mirror_dir:
            return None
        try:
            os.makedirs(self.mirror_dir, exist_ok=True)
            target = os.path.join(self.mirror_dir, "pharmaguard.db")
            tmp = target + ".tmp"
            dst = sqlite3.connect(tmp)
            with self._lock:
                self._conn.backup(dst)
            dst.close()
            os.replace(tmp, target)
            return target
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Event database mirror to %s failed: %s", self.mirror_dir, exc)
            return None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
