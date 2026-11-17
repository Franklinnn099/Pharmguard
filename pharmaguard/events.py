"""
Event logging and audit trail module for PharmaGuard.
Maintains a structured, chronological sequence of system, detection, and security events.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import time


@dataclass
class SystemEvent:
    """Represents an atomic event in the PharmaGuard monitoring timeline."""
    event_id: int
    timestamp_iso: str
    category: str  # "SYSTEM", "DETECTION", "TRACKING", "SECURITY"
    message: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.event_id,
            "timestamp": self.timestamp_iso,
            "category": self.category,
            "message": self.message,
            "metadata": self.metadata
        }


class EventManager:
    """Ring buffer and persistent recorder for system events."""

    def __init__(self, max_buffer_size: int = 200):
        self.max_buffer_size = max_buffer_size
        self._events: List[SystemEvent] = []
        self._counter = 0

    def log(self, category: str, message: str, metadata: Optional[Dict] = None) -> SystemEvent:
        """Record a new event to the timeline."""
        self._counter += 1
        now_iso = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        event = SystemEvent(
            event_id=self._counter,
            timestamp_iso=now_iso,
            category=category,
            message=message,
            metadata=metadata or {}
        )
        self._events.insert(0, event)
        if len(self._events) > self.max_buffer_size:
            self._events.pop()
        return event

    def get_recent(self, limit: int = 20) -> List[SystemEvent]:
        """Get recent events sorted newest first."""
        return self._events[:limit]

    def clear(self) -> None:
        """Clear event log."""
        self._events.clear()
        self._counter = 0
