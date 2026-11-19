"""Shared pytest configuration."""

import os
import tempfile


def pytest_configure(config):
    # Tests must never append to the operational audit log (logs/pharmaguard_alerts.jsonl).
    # AlertConfig reads this env var at construction time.
    log_dir = tempfile.mkdtemp(prefix="pharmaguard_test_logs_")
    os.environ["PHARMAGUARD_ALERT_LOG"] = os.path.join(log_dir, "alerts.jsonl")
    # ...nor into the operational event database / evidence clips (data/).
    os.environ["PHARMAGUARD_DATA_DIR"] = os.path.join(log_dir, "data")
