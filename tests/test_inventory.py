"""Inventory store, expected-quantity arithmetic, and stock vs camera reconciliation."""

import time

import pytest

from pharmaguard.inventory import InventoryStore, classify
from pharmaguard.storage import EventStore
from pharmaguard.ui_inventory import recon_rows, run_reconciliation, stock_rows


@pytest.fixture
def inv(tmp_path):
    s = InventoryStore(str(tmp_path))
    yield s
    s.close()


def test_classify_rule():
    assert classify(2, 1) == "CORROBORATED"
    assert classify(2, 0) == "UNEXPLAINED_LOSS"
    assert classify(0, 3) == "NO_SHORTFALL"
    assert classify(-1, 0) == "OK"      # surplus is not a loss


def test_expected_qty_from_movements_and_counts(inv):
    inv.add_product("PARA", "Paracetamol", "Shelf A", 12.5, opening_qty=20)
    inv.record_movement("PARA", "sale", 3)
    inv.record_movement("PARA", "restock", 10)
    inv.record_movement("PARA", "writeoff", 1)
    assert inv.expected_qty("PARA") == 26
    r = inv.record_count("PARA", 24)
    assert (r.expected_qty, r.counted_qty, r.shortfall) == (26, 24, 2)
    # After a count, expected restarts from the counted quantity.
    time.sleep(0.01)
    inv.record_movement("PARA", "sale", 4)
    assert inv.expected_qty("PARA") == 20


def test_validation(inv):
    inv.add_product("X", "Item", "Shelf A")
    with pytest.raises(ValueError):
        inv.add_product("X", "Dup", "Shelf A")
    with pytest.raises(ValueError):
        inv.add_product("", "No sku", "Shelf A")
    with pytest.raises(ValueError):
        inv.record_movement("X", "sale", 0)
    with pytest.raises(ValueError):
        inv.record_movement("X", "steal", 1)
    with pytest.raises(ValueError):
        inv.record_movement("NOPE", "sale", 1)
    with pytest.raises(ValueError):
        inv.record_count("X", -1)


def test_reconcile_all_four_outcomes(inv):
    now = time.time()
    for sku, zone in (("A1", "Shelf A"), ("B1", "Shelf B"), ("C1", "Shelf C"), ("D1", "Shelf D")):
        inv.add_product(sku, sku, zone, 10.0)
        inv.record_movement(sku, "opening", 10, at_epoch=now - 100)
    inv.record_count("A1", 8, at_epoch=now - 10)    # short 2, camera saw it
    inv.record_count("B1", 9, at_epoch=now - 10)    # short 1, camera saw nothing
    inv.record_count("C1", 10, at_epoch=now - 10)   # complete, but camera flagged
    inv.record_count("D1", 10, at_epoch=now - 10)   # complete, quiet
    events = {"Shelf A": {"ALERT": 1}, "Shelf C": {"WARNING": 2}}
    res = {r.zone: r for r in inv.reconcile(events, now - 3600, now)}
    assert res["Shelf A"].status == "CORROBORATED" and res["Shelf A"].shortfall_value == 20.0
    assert res["Shelf B"].status == "UNEXPLAINED_LOSS"
    assert res["Shelf C"].status == "NO_SHORTFALL" and res["Shelf C"].reviews == 2
    assert res["Shelf D"].status == "OK"
    rows = recon_rows(list(res.values()))
    assert any("A1 x2" in r[-1] for r in rows)


def test_counts_outside_window_are_ignored(inv):
    now = time.time()
    inv.add_product("A1", "A1", "Shelf A")
    inv.record_movement("A1", "opening", 5, at_epoch=now - 10 * 86400)
    inv.record_count("A1", 3, at_epoch=now - 9 * 86400)
    res = inv.reconcile({}, now - 86400, now)
    assert res[0].status == "OK"


class _Alert:
    def __init__(self, uid, sev, zone):
        self.alert_id, self.severity, self.zone_name = uid, sev, zone
        self.timestamp_iso, self.epoch_timestamp = "", time.time()
        self.tracker_id, self.incident_id, self.score = 1, 1, 0.9
        self.reasons, self.title, self.message, self.event_type = ["r"], "t", "m", "SUSPICION_ALERT"


def test_reconciliation_reads_real_event_store(tmp_path, inv):
    store = EventStore(str(tmp_path / "ev"))
    store.record_alert(_Alert("e1", "ALERT", "Shelf A"))
    store.record_alert(_Alert("e2", "WARNING", "Shelf A"))
    store.record_alert(_Alert("e3", "ALERT", "Shelf A"))
    store.set_review("e3", "false_alarm", "staff")
    now = time.time()
    assert store.zone_event_counts(now - 60, now + 60) == {"Shelf A": {"ALERT": 2, "WARNING": 1}}
    assert store.zone_event_counts(now - 60, now + 60, exclude_false_alarms=True) == {"Shelf A": {"ALERT": 1, "WARNING": 1}}
    inv.add_product("A1", "A1", "Shelf A", opening_qty=4)
    inv.record_count("A1", 3)
    res = run_reconciliation(inv, store, days=1, exclude_false_alarms=True, now=now + 60)
    assert res[0].status == "CORROBORATED" and res[0].alerts == 1
    assert stock_rows(inv)[0][:3] == ["A1", "A1", "Shelf A"]
    store.close()
