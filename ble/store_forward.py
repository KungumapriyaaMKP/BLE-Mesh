"""
Store & Forward system for BLE Mesh PoC.

When a BLE node goes offline, messages are persisted to disk.
When the node comes back online, undelivered messages are reloaded
and injected into the live feed — simulating offline-tolerant mesh delivery.

Flow (from research notes):
  Sender A → Sender B (stores in cache) → Recipient C offline
  → C comes online → Announces presence → B delivers cached message
"""

import json
import os
import time

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "message_store.json")


class StoreForward:

    def __init__(self):
        self._records = []
        self._load()
        pending = self.pending()
        if pending:
            print(f"[Store&Forward] {len(pending)} undelivered message(s) found from previous session.")

    def save(self, entry: dict):
        """Persist a classified message to disk."""
        record = {
            "message":      entry["message"],
            "prediction":   entry["prediction"],
            "confidence":   entry["confidence"],
            "reshare_count": entry.get("reshare_count", 1),
            "hop_count":    entry.get("hop_count", 1),
            "time":         entry["time"],
            "stored_at":    time.strftime("%H:%M:%S"),
            "forwarded":    False,
        }
        self._records.append(record)
        self._flush()

    def pending(self) -> list:
        """Return messages stored but not yet delivered to the UI."""
        return [r for r in self._records if not r.get("forwarded", False)]

    def mark_all_delivered(self):
        for r in self._records:
            r["forwarded"] = True
        self._flush()

    def clear(self):
        self._records = []
        self._flush()

    def _flush(self):
        try:
            with open(STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Store&Forward] Write error: {e}")

    def _load(self):
        if os.path.exists(STORE_PATH):
            try:
                with open(STORE_PATH, encoding="utf-8") as f:
                    self._records = json.load(f)
            except Exception:
                self._records = []
        else:
            self._records = []
