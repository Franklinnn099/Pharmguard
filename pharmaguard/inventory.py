"""
Inventory and stock reconciliation for PharmaGuard.

Products are assigned to the same shelf zones the camera watches. Stock movements (opening stock,
restocks, sales, write-offs) give an EXPECTED quantity; a physical count gives the ACTUAL quantity.
The difference is the shortfall.

Reconciliation cross-checks each zone's shortfall against the REVIEW/ALERT events the vision system
recorded for that zone in the same period:

* ``CORROBORATED``      shortfall AND camera events: strongest case for staff review.
* ``UNEXPLAINED_LOSS``  shortfall, NO camera events: loss the camera did not see (blind spot, counting or
                        record-keeping error, or stock removed outside normal procedure). Needs an audit.
* ``NO_SHORTFALL``      camera events, NO shortfall: the flagged items were likely returned or paid for;
                        evidence that those events were false alarms.
* ``OK``                neither.

Wording stays neutral: these are reasons for staff review, never findings of theft.
The store is a separate SQLite file (``inventory.db``) next to the event database, using the same
one-connection-plus-lock pattern as ``EventStore``.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import os
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    sku         TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    zone        TEXT NOT NULL,           -- must match a ShelfZone.name to be cross-checked
    unit_price  REAL NOT NULL DEFAULT 0, -- GH¢
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS movements (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    sku       TEXT NOT NULL REFERENCES products(sku),
    kind      TEXT NOT NULL,             -- opening | restock | sale | writeoff
    qty       INTEGER NOT NULL,          -- always positive; sign comes from kind
    at        TEXT NOT NULL,
    at_epoch  REAL NOT NULL,
    operator  TEXT,
    note      TEXT
);
CREATE TABLE IF NOT EXISTS counts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sku          TEXT NOT NULL REFERENCES products(sku),
    counted_qty  INTEGER NOT NULL,
    expected_qty INTEGER NOT NULL,
    at           TEXT NOT NULL,
    at_epoch     REAL NOT NULL,
    operator     TEXT
);
CREATE INDEX IF NOT EXISTS idx_mov_sku ON movements(sku, at_epoch);
CREATE INDEX IF NOT EXISTS idx_cnt_sku ON counts(sku, at_epoch);
"""

MOVEMENT_SIGN = {"opening": +1, "restock": +1, "sale": -1, "writeoff": -1}
STATUSES = ("CORROBORATED", "UNEXPLAINED_LOSS", "NO_SHORTFALL", "OK")


def _now() -> tuple:
    t = time.time()
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"), t


@dataclass
class StockLine:
    sku: str
    name: str
    zone: str
    unit_price: float
    expected_qty: int
    last_counted_qty: Optional[int]
    last_count_at: Optional[str]


@dataclass
class CountResult:
    sku: str
    expected_qty: int
    counted_qty: int

    @property
    def shortfall(self) -> int:
        """Units missing (positive) or surplus (negative)."""
        return self.expected_qty - self.counted_qty


@dataclass
class ZoneReconciliation:
    zone: str
    status: str
    shortfall_units: int
    shortfall_value: float
    alerts: int
    reviews: int
    short_items: List[Dict[str, Any]] = field(default_factory=list)


def classify(shortfall_units: int, camera_events: int) -> str:
    """Pure decision rule for one zone (unit-tested)."""
    loss = shortfall_units > 0
    seen = camera_events > 0
    if loss and seen:
        return "CORROBORATED"
    if loss:
        return "UNEXPLAINED_LOSS"
    if seen:
        return "NO_SHORTFALL"
    return "OK"


class InventoryStore:
    """Thread-safe SQLite inventory (one connection, serialised by a lock)."""

    def __init__(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, "inventory.db")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ products
    def add_product(self, sku: str, name: str, zone: str, unit_price: float = 0.0,
                    opening_qty: int = 0, operator: Optional[str] = None) -> None:
        sku, name, zone = sku.strip(), name.strip(), zone.strip()
        if not sku or not name or not zone:
            raise ValueError("SKU, name and zone are required")
        if unit_price < 0 or opening_qty < 0:
            raise ValueError("Price and opening quantity cannot be negative")
        at, ep = _now()
        with self._lock:
            if self._conn.execute("SELECT 1 FROM products WHERE sku=?", (sku,)).fetchone():
                raise ValueError(f"SKU already exists: {sku}")
            self._conn.execute("INSERT INTO products(sku,name,zone,unit_price,created_at) VALUES (?,?,?,?,?)",
                               (sku, name, zone, float(unit_price), at))
        if opening_qty:
            self.record_movement(sku, "opening", opening_qty, operator=operator)

    def products_in_zone(self, zone: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT sku,name,unit_price FROM products WHERE zone=? ORDER BY name",
                                      (zone,)).fetchall()
        return [dict(r) for r in rows]

    def zones(self) -> List[str]:
        with self._lock:
            return [r[0] for r in self._conn.execute("SELECT DISTINCT zone FROM products ORDER BY zone")]

    # ------------------------------------------------------------------ movements & counts
    def record_movement(self, sku: str, kind: str, qty: int, operator: Optional[str] = None,
                        note: str = "", at_epoch: Optional[float] = None) -> None:
        if kind not in MOVEMENT_SIGN:
            raise ValueError(f"kind must be one of {sorted(MOVEMENT_SIGN)}")
        if int(qty) <= 0:
            raise ValueError("Quantity must be a positive whole number")
        at, ep = _now()
        if at_epoch is not None:
            ep, at = at_epoch, datetime.fromtimestamp(at_epoch).strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            if not self._conn.execute("SELECT 1 FROM products WHERE sku=?", (sku,)).fetchone():
                raise ValueError(f"Unknown SKU: {sku}")
            self._conn.execute("INSERT INTO movements(sku,kind,qty,at,at_epoch,operator,note) VALUES (?,?,?,?,?,?,?)",
                               (sku, kind, int(qty), at, ep, operator, note))

    def expected_qty(self, sku: str, as_of: Optional[float] = None) -> int:
        """Last counted quantity plus signed movements since that count (or all movements if never counted)."""
        as_of = time.time() if as_of is None else as_of
        with self._lock:
            last = self._conn.execute(
                "SELECT counted_qty, at_epoch FROM counts WHERE sku=? AND at_epoch<=? ORDER BY at_epoch DESC, id DESC LIMIT 1",
                (sku, as_of)).fetchone()
            base, since = (last["counted_qty"], last["at_epoch"]) if last else (0, float("-inf"))
            rows = self._conn.execute(
                "SELECT kind, qty FROM movements WHERE sku=? AND at_epoch>? AND at_epoch<=?",
                (sku, since, as_of)).fetchall()
        return base + sum(MOVEMENT_SIGN[r["kind"]] * r["qty"] for r in rows)

    def record_count(self, sku: str, counted_qty: int, operator: Optional[str] = None,
                     at_epoch: Optional[float] = None) -> CountResult:
        if int(counted_qty) < 0:
            raise ValueError("Counted quantity cannot be negative")
        at, ep = _now()
        if at_epoch is not None:
            ep, at = at_epoch, datetime.fromtimestamp(at_epoch).strftime("%Y-%m-%d %H:%M:%S")
        expected = self.expected_qty(sku, as_of=ep)
        with self._lock:
            if not self._conn.execute("SELECT 1 FROM products WHERE sku=?", (sku,)).fetchone():
                raise ValueError(f"Unknown SKU: {sku}")
            self._conn.execute("INSERT INTO counts(sku,counted_qty,expected_qty,at,at_epoch,operator) VALUES (?,?,?,?,?,?)",
                               (sku, int(counted_qty), expected, at, ep, operator))
        return CountResult(sku, expected, int(counted_qty))

    # ------------------------------------------------------------------ reads
    def stock_lines(self) -> List[StockLine]:
        with self._lock:
            products = self._conn.execute("SELECT * FROM products ORDER BY zone, name").fetchall()
        out = []
        for p in products:
            with self._lock:
                c = self._conn.execute("SELECT counted_qty, at FROM counts WHERE sku=? ORDER BY at_epoch DESC, id DESC LIMIT 1",
                                       (p["sku"],)).fetchone()
            out.append(StockLine(p["sku"], p["name"], p["zone"], p["unit_price"], self.expected_qty(p["sku"]),
                                 c["counted_qty"] if c else None, c["at"] if c else None))
        return out

    def shortfalls_between(self, start_epoch: float, end_epoch: float) -> List[Dict[str, Any]]:
        """Counts taken in the window that found fewer units than expected."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.sku, p.name, p.zone, p.unit_price, c.expected_qty - c.counted_qty AS shortfall, c.at "
                "FROM counts c JOIN products p ON p.sku=c.sku "
                "WHERE c.at_epoch>=? AND c.at_epoch<=? AND c.expected_qty > c.counted_qty ORDER BY c.at_epoch",
                (start_epoch, end_epoch)).fetchall()
        return [dict(r) for r in rows]

    def reconcile(self, zone_events: Dict[str, Dict[str, int]], start_epoch: float,
                  end_epoch: float) -> List[ZoneReconciliation]:
        """Cross-check shortfalls with camera events per zone.

        ``zone_events`` maps zone -> {"ALERT": n, "WARNING": n} for the same window
        (``EventStore.zone_event_counts``). Zones that appear in either source are reported."""
        short = self.shortfalls_between(start_epoch, end_epoch)
        zones = sorted(set(self.zones()) | set(zone_events))
        out = []
        for z in zones:
            items = [s for s in short if s["zone"] == z]
            units = sum(s["shortfall"] for s in items)
            value = round(sum(s["shortfall"] * s["unit_price"] for s in items), 2)
            ev = zone_events.get(z, {})
            alerts, reviews = int(ev.get("ALERT", 0)), int(ev.get("WARNING", 0))
            out.append(ZoneReconciliation(z, classify(units, alerts + reviews), units, value, alerts, reviews, items))
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()
