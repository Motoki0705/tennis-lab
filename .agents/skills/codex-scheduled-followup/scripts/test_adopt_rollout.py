"""Receipt migration tests; all state/logs are temporary and no queue/timer is used."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from adopt_rollout import adopt, load_helper


class AdoptRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.helper = load_helper(Path(os.environ.get(
            "FOLLOWUP_TEST_HELPER", Path(__file__).with_name("cli_heartbeat.py")
        )))
        self.parent = self.root / "parent.jsonl"
        self.child = self.root / "continuation.jsonl"
        self.marker = "[codex-heartbeat:test:unique-nonce]"
        self.meta: dict[str, Any] = {"id": "same-thread", "cwd": str(self.root)}
        self.parent.write_text(self.line({"type": "session_meta", "payload": self.meta})
                               + self.line({"type": "event_msg", "payload": {"type": "note", "text": "prior"}}))
        self.task: dict[str, Any] = {
            "version": getattr(self.helper, "STATE_VERSION", 1),
            "id": "test", "thread_id": "same-thread", "cwd": str(self.root),
            "rollout": str(self.parent), "interval_minutes": 120,
            "status": "active", "unit": "unchanged", "codex": "unchanged",
            "environment": {"CODEX_HOME": "unchanged"}, "deliveries": [],
        }
        _, boundary = self.helper.history(self.task, None)
        self.task.update(boundary=boundary, observed_boundary=copy.deepcopy(boundary))
        self.task["deliveries"].append({
            "marker": self.marker, "phase": "queued", "boundary": copy.deepcopy(boundary),
            "prepared_at": "2026-10-12T02:00:00+00:00", "queue_id": "same-queue-item",
        })
        self.header: dict[str, Any] = {"type": "session_meta", "payload": {
            **self.meta, "history_base": {
                "thread_id": "same-thread", "end_byte_offset": boundary["offset"],
                "end_ordinal_exclusive": 2,
            },
        }}
        self.events: list[dict[str, Any]] = [
            {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "receipt-turn"}},
            {"type": "response_item", "payload": {"type": "message", "role": "user",
              "content": [{"type": "input_text", "text": self.marker}]}},
            {"type": "event_msg", "timestamp": "2026-10-12T02:02:00Z",
             "payload": {"type": "task_complete", "turn_id": "receipt-turn"}},
        ]
        self.publish()

    @staticmethod
    def line(record: dict[str, Any]) -> str:
        return json.dumps(record) + "\n"

    def publish(self) -> None:
        (self.root / "task.json").write_text(json.dumps(self.task))
        self.child.write_text("".join(self.line(row) for row in [self.header, *self.events]))

    def test_dry_run_is_read_only(self) -> None:
        before = (self.root / "task.json").read_bytes()
        result = adopt(self.helper, self.root, self.child, apply=False)
        self.assertFalse(result["applied"])
        self.assertEqual(result["completed_at"], "2026-10-12T02:02:00Z")
        self.assertEqual(before, (self.root / "task.json").read_bytes())
        self.assertFalse(list(self.root.glob("before-rollout-adoption-*")))

    def test_apply_preserves_state_and_reobserves_receipt(self) -> None:
        adopt(self.helper, self.root, self.child, apply=True)
        after = json.loads((self.root / "task.json").read_text())
        for key in ("id", "thread_id", "cwd", "codex", "environment", "interval_minutes", "status", "unit"):
            self.assertEqual(after[key], self.task[key])
        self.assertEqual(after["rollout"], str(self.child))
        self.assertEqual(after["deliveries"][-1]["phase"], "completed")
        self.assertEqual(after["deliveries"][-1]["queue_id"], "same-queue-item")
        proof = after["rollout_adoptions"][0]
        self.assertEqual(proof["old_delivery_boundary"], self.task["deliveries"][-1]["boundary"])
        backup = list(self.root.glob("before-rollout-adoption-*"))
        self.assertEqual(len(backup), 1)
        self.assertEqual(json.loads(backup[0].read_text()), self.task)
        self.assertEqual(backup[0].stat().st_mode & 0o777, 0o600)
        # The original helper can still reconcile the adopted state normally.
        receipt = self.helper.reconcile(self.root, after)
        self.assertEqual(receipt["phase"], "completed")
        self.assertFalse(after.get("blocked_reason"))

    def test_rejects_wrong_identity_and_history_base(self) -> None:
        for field in ("id", "cwd"):
            with self.subTest(field=field):
                saved = self.header["payload"][field]
                self.header["payload"][field] = "wrong"
                self.publish()
                with self.assertRaises(ValueError):
                    adopt(self.helper, self.root, self.child, apply=True)
                self.header["payload"][field] = saved
        for field in ("thread_id", "end_byte_offset", "end_ordinal_exclusive"):
            with self.subTest(field=field):
                base = self.header["payload"]["history_base"]
                saved = base[field]
                base[field] = "wrong" if field == "thread_id" else saved + 1
                self.publish()
                with self.assertRaises(ValueError):
                    adopt(self.helper, self.root, self.child, apply=True)
                base[field] = saved
        self.assertFalse(list(self.root.glob("before-rollout-adoption-*")))

    def test_rejects_parent_rewrite(self) -> None:
        self.parent.write_text(self.parent.read_text().replace("prior", "other"))
        with self.assertRaisesRegex(ValueError, "prefix changed"):
            adopt(self.helper, self.root, self.child, apply=True)

    def test_assistant_echo_does_not_count(self) -> None:
        self.events[1]["payload"]["role"] = "assistant"
        self.publish()
        with self.assertRaisesRegex(ValueError, "does not prove"):
            adopt(self.helper, self.root, self.child, apply=True)

    def test_requires_successful_completion(self) -> None:
        for kind in ("task_aborted", "unrelated_event"):
            with self.subTest(kind=kind):
                self.events[-1]["payload"]["type"] = kind
                self.publish()
                with self.assertRaisesRegex(ValueError, "does not prove"):
                    adopt(self.helper, self.root, self.child, apply=True)

    def test_rejects_duplicate_turn_and_old_completion(self) -> None:
        originals = copy.deepcopy(self.events)
        self.events += [
            {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "other-turn"}},
            copy.deepcopy(self.events[1]),
        ]
        self.publish()
        with self.assertRaisesRegex(ValueError, "multiple turns"):
            adopt(self.helper, self.root, self.child, apply=True)
        self.events = originals
        self.events[-1]["timestamp"] = "2026-10-11T02:02:00Z"
        self.publish()
        with self.assertRaisesRegex(ValueError, "predates"):
            adopt(self.helper, self.root, self.child, apply=True)


if __name__ == "__main__":
    unittest.main()
