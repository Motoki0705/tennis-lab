"""CLI heartbeat tests: synthetic JSONL and mocked commands, never live queues."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import subprocess
import tempfile
import threading
import unittest
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cli_heartbeat as helper

THREAD = "11111111-2222-4333-8444-555555555555"
REAL_COMMAND = helper.command
# systemd 255 output from live-test/failure-inspection-1.json; Id is also
# requested by the helper. Keep both separately emitted TimersMonotonic lines.
SYSTEMD_255_TIMER = (
    "Id=fixture.timer\n"
    "Unit=fixture.service\n"
    "TimersMonotonic={ OnUnitActiveUSec=30min ; next_elapse=0 }\n"
    "TimersMonotonic={ OnActiveUSec=30min ; next_elapse=1d 3h 52min 49.642424s }\n"
    "NextElapseUSecRealtime=\n"
    "NextElapseUSecMonotonic=1d 3h 52min 49.642424s\n"
    "LastTriggerUSec=\nLoadState=loaded\nActiveState=active\nSubState=waiting\n"
)


class CliHeartbeatTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="cli-heartbeat-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cwd = self.root / "original cwd"
        self.cwd.mkdir()
        self.rollout = self.root / "rollout.jsonl"
        self.append("session_meta", {"id": THREAD, "cwd": str(self.cwd)})
        self.prompt = self.root / "prompt.txt"
        self.prompt.write_text(
            "監視 'quoted'\n$(touch SHOULD_NOT_EXIST); `false` $PATH", encoding="utf-8"
        )
        self.codex = self.root / "codex"
        self.codex.write_text("fake executable", encoding="utf-8")
        self.codex.chmod(0o700)
        self.environment = {
            "CODEX_HOME": str(self.root / "codex-home"),
            "CODEX_SQLITE_HOME": str(self.root / "wsl-db"),
            "PATH": "/custom/bin:/usr/bin",
            "SECRET_TEST_VALUE": "must-not-be-saved",
        }
        self.enterContext(patch.dict(os.environ, self.environment))
        self.enterContext(
            patch.object(helper.shutil, "which", return_value=str(self.codex))
        )
        self.command_mock = self.enterContext(
            patch.object(helper, "command", side_effect=self.external)
        )
        self.active = False
        self.interval = 30
        self.queue_returncode = 0
        self.queue_failure: Exception | None = None
        self.queue_hook: Callable[[], None] | None = None
        self.queue_calls: list[tuple[list[str], dict]] = []
        self.task_dir = self.root / "state" / "followup"

    def arguments(self, **overrides: object) -> argparse.Namespace:
        values = {
            "id": "followup",
            "thread_id": THREAD,
            "prompt_file": self.prompt,
            "rollout": self.rollout,
            "cwd": self.cwd,
            "interval_minutes": 30,
            "queue_timeout_seconds": 180,
            "state_dir": self.root / "state",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def external(
        self, argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        output = ""
        returncode = 0
        if argv[0] == "systemd-run":
            self.active = True
            self.interval = int(
                next(value for value in argv if value.startswith("--on-active="))
                .split("=")[1]
                .removesuffix("min")
            )
        elif argv[0] == "systemctl":
            if "stop" in argv:
                self.active = False
            elif "--property=Version" in argv:
                output = "255\n"
            elif "show" in argv:
                output = (
                    f"Id={argv[3]}\nLoadState=loaded\n"
                    f"ActiveState={'active' if self.active else 'inactive'}\n"
                    f"TimersMonotonic={{ OnActiveUSec={self.interval}min ; next_elapse=2h }} "
                    f"{{ OnUnitActiveUSec={self.interval}min ; next_elapse=2h }}\n"
                    f"NextElapseUSecMonotonic={'2h' if self.active else '0'}\n"
                    "NextElapseUSecRealtime=n/a\n"
                )
            elif "list-timers" in argv:
                output = "NEXT                         UNIT\nFri 2026-10-09 12:00:00 JST   test.timer\n"
        else:
            self.assertEqual(argv[0], str(self.codex))
            self.queue_calls.append((argv, kwargs))
            persisted = self.state()["deliveries"][-1]
            self.assertEqual(persisted["phase"], "uncertain")
            if self.queue_hook:
                self.queue_hook()
            if self.queue_failure:
                raise self.queue_failure
            returncode = self.queue_returncode
            output = (
                f"Queued message accepted-id for thread {THREAD}.\n"
                if not returncode
                else "Queue RPC failed\n"
            )
        return subprocess.CompletedProcess(argv, returncode, output, "")

    def state(self) -> dict:
        task: dict[str, Any] = json.loads(
            (self.task_dir / "task.json").read_text(encoding="utf-8")
        )
        return task

    def create(self) -> dict:
        return helper.create_task(self.arguments())

    def append(self, kind: str, payload: dict) -> None:
        with self.rollout.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "type": kind,
                        "payload": payload,
                        "timestamp": "2026-10-09T03:00:00Z",
                    }
                )
                + "\n"
            )

    def event(self, kind: str, **fields: object) -> None:
        self.append("event_msg", {"type": kind, **fields})

    def start_delivery(self, turn: str = "delivery-turn") -> str:
        marker: str = self.state()["deliveries"][-1]["marker"]
        self.event("task_started", turn_id=turn)
        self.append(
            "response_item",
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": marker + "\n" + self.prompt.read_text(),
                    }
                ],
            },
        )
        return marker

    def test_thirty_minutes_owner_context_and_actual_registration(self) -> None:
        result = self.create()
        self.assertEqual(result["interval_minutes"], 30)
        self.assertEqual(result["thread_id"], THREAD)
        self.assertEqual(result["cwd"], str(self.cwd))
        self.assertTrue(result["registration_verified"])
        self.assertFalse(result["execution_verified"])
        self.assertIn("2026-10-09", result["timer_listing"])
        state = self.state()
        self.assertEqual(set(state["environment"]), set(helper.ENV_KEYS))
        self.assertNotIn("must-not-be-saved", (self.task_dir / "task.json").read_text())
        register = next(
            call.args[0]
            for call in self.command_mock.call_args_list
            if call.args[0][0] == "systemd-run"
        )
        self.assertIn("--on-active=30min", register)
        self.assertIn("--on-unit-active=30min", register)
        self.assertIn("--working-directory=" + str(self.cwd), register)
        self.assertEqual(register[-3:], ["tick", "--task-dir", str(self.task_dir)])
        self.assertFalse(self.queue_calls)

    def test_cli_default_is_sixty_minutes_and_tick_is_single_shot(self) -> None:
        argv = [
            "cli_heartbeat.py",
            "create",
            "--id",
            "followup",
            "--thread-id",
            THREAD,
            "--prompt-file",
            str(self.prompt),
            "--rollout",
            str(self.rollout),
            "--cwd",
            str(self.cwd),
            "--state-dir",
            str(self.root / "state"),
        ]
        with (
            patch.object(helper.sys, "argv", argv),
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(helper.main(), 0)
        self.assertEqual(json.loads(output.getvalue())["interval_minutes"], 60)
        with (
            patch.object(
                helper.sys,
                "argv",
                ["cli_heartbeat.py", "tick", "--task-dir", str(self.task_dir)],
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(helper.main(), 0)
        self.assertEqual(len(self.queue_calls), 1)

    def test_wrong_thread_or_cwd_fails_before_registration(self) -> None:
        for overrides in (
            {"thread_id": "11111111-1111-4111-8111-111111111111"},
            {"cwd": self.root},
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, "session_meta"),
            ):
                helper.create_task(self.arguments(**overrides))
        self.assertFalse(self.task_dir.exists())
        self.assertFalse(self.command_mock.called)

    def test_existing_id_never_registers_duplicate_timer(self) -> None:
        self.create()
        self.command_mock.reset_mock()
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertFalse(self.command_mock.called)

    def test_queue_argv_and_environment_preserve_special_characters(self) -> None:
        self.create()
        result = helper.tick(self.task_dir)
        argv, kwargs = self.queue_calls[0]
        self.assertEqual(
            argv[:7],
            [
                str(self.codex),
                "--disable",
                "daemon_auto_start",
                "queue",
                "--thread",
                THREAD,
                "--message",
            ],
        )
        self.assertEqual(
            argv[7], result["delivery"]["marker"] + "\n" + self.prompt.read_text()
        )
        self.assertEqual(kwargs["cwd"], str(self.cwd))
        for key in helper.ENV_KEYS:
            self.assertEqual(kwargs["env"][key], self.environment[key])
        self.assertNotIn("shell", kwargs)
        self.assertEqual(
            result["delivery"]["queue_stdout"],
            f"Queued message accepted-id for thread {THREAD}.\n",
        )
        self.assertFalse(result["execution_verified"])

    def test_subprocess_wrapper_never_uses_a_shell(self) -> None:
        with patch.object(helper.subprocess, "run") as run:
            argv = ["codex", "queue", "--message", self.prompt.read_text()]
            REAL_COMMAND(argv, cwd=str(self.cwd))
        self.assertEqual(run.call_args.args, (argv,))
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["timeout"], 45)

    def test_queued_started_completed_are_distinct_and_no_duplicate_is_sent(
        self,
    ) -> None:
        self.create()
        self.event("task_started", turn_id="earlier-owner-turn")
        first = helper.tick(self.task_dir)
        self.assertEqual(first["delivery"]["phase"], "queued")
        self.event("task_complete", turn_id="earlier-owner-turn")
        self.assertFalse(helper.tick(self.task_dir)["enqueued"])
        self.start_delivery()
        started = helper.status(self.task_dir)
        self.assertEqual(started["delivery"]["phase"], "started")
        self.assertFalse(started["execution_verified"])
        self.event("task_complete", turn_id="different-turn")
        self.assertFalse(helper.status(self.task_dir)["execution_verified"])
        self.event("task_complete", turn_id="delivery-turn")
        finished = helper.status(self.task_dir)
        self.assertTrue(finished["execution_verified"])
        self.assertEqual(len(self.queue_calls), 1)
        self.assertTrue(helper.tick(self.task_dir)["enqueued"])
        self.assertEqual(len(self.queue_calls), 2)
        self.assertEqual(len(self.state()["deliveries"]), 2)

    def test_tool_output_nonce_and_assistant_echo_do_not_verify_execution(self) -> None:
        self.create()
        queued = helper.tick(self.task_dir)
        marker = queued["delivery"]["marker"]
        self.event("task_started", turn_id="unrelated")
        self.append("response_item", {"type": "function_call_output", "output": marker})
        for role in ("developer", "assistant", "system"):
            self.append(
                "response_item",
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": marker}],
                },
            )
        self.event("agent_message", message=marker)
        self.event("task_complete", turn_id="unrelated")
        result = helper.status(self.task_dir)
        self.assertEqual(result["delivery"]["phase"], "queued")
        self.assertFalse(result["execution_verified"])

    def test_partial_appends_preserve_complete_line_boundary(self) -> None:
        self.create()
        queued = helper.tick(self.task_dir)
        pending_lines = [
            {
                "type": "event_msg",
                "payload": {"type": "task_started", "turn_id": "partial-turn"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": queued["delivery"]["marker"] + "\n日本語",
                        }
                    ],
                },
            },
            {
                "type": "event_msg",
                "payload": {"type": "task_complete", "turn_id": "partial-turn"},
            },
        ]
        for row in pending_lines:
            before = self.rollout.stat().st_size
            encoded = json.dumps(row, ensure_ascii=False).encode() + b"\n"
            split = len(encoded) // 2
            with self.rollout.open("ab") as handle:
                handle.write(encoded[:split])
            partial = helper.status(self.task_dir)
            self.assertIsNone(partial["blocked_reason"])
            self.assertFalse(partial["execution_verified"])
            self.assertEqual(self.state()["observed_boundary"]["offset"], before)
            with self.rollout.open("ab") as handle:
                handle.write(encoded[split:])
            self.assertIsNone(helper.status(self.task_dir)["blocked_reason"])
        self.assertTrue(helper.status(self.task_dir)["execution_verified"])
        self.assertEqual(len(self.queue_calls), 1)

    def test_legacy_user_message_event_receipt_remains_supported(self) -> None:
        self.create()
        marker = helper.tick(self.task_dir)["delivery"]["marker"]
        self.event("task_started", turn_id="legacy")
        self.event("user_message", message=marker)
        self.event("task_complete", turn_id="legacy")
        self.assertTrue(helper.status(self.task_dir)["execution_verified"])

    def test_receipt_requires_a_start_after_its_byte_boundary(self) -> None:
        self.create()
        self.event("task_started", turn_id="old")
        marker = helper.tick(self.task_dir)["delivery"]["marker"]
        self.event("user_message", message=marker)
        self.event("task_complete", turn_id="old")
        result = helper.status(self.task_dir)
        self.assertIn("without a preceding", result["blocked_reason"])
        self.assertFalse(result["execution_verified"])
        helper.tick(self.task_dir)
        self.assertEqual(len(self.queue_calls), 1)

    def test_uncertain_timeout_or_nonzero_never_automatically_resends(self) -> None:
        self.create()
        self.queue_failure = subprocess.TimeoutExpired("codex", 45)
        result = helper.tick(self.task_dir)
        self.assertEqual(result["delivery"]["phase"], "uncertain")
        self.queue_failure = None
        helper.tick(self.task_dir)
        self.assertEqual(len(self.queue_calls), 1)
        self.start_delivery()
        self.event("task_complete", turn_id="delivery-turn")
        self.assertTrue(helper.status(self.task_dir)["execution_verified"])
        self.queue_returncode = 1
        self.assertEqual(helper.tick(self.task_dir)["delivery"]["phase"], "uncertain")
        helper.tick(self.task_dir)
        self.assertEqual(len(self.queue_calls), 2)

    def test_timeout_after_acknowledgement_retains_acceptance_and_diagnostics(
        self,
    ) -> None:
        self.create()
        self.queue_failure = subprocess.TimeoutExpired(
            "codex",
            180,
            output=f"Queued message accepted-id for thread {THREAD}.\n".encode(),
            stderr=b"teardown stalled",
        )
        result = helper.tick(self.task_dir)
        self.assertEqual(result["delivery"]["phase"], "queued")
        self.assertEqual(result["delivery"]["queue_id"], "accepted-id")
        self.assertEqual(result["delivery"]["queue_stderr"], "teardown stalled")
        self.assertTrue(result["delivery"]["queue_process_timed_out"])
        self.assertFalse(result["attention_required"])
        self.assertEqual(result["delivery_health"], "waiting_for_runtime")
        self.assertEqual(self.queue_calls[0][1]["timeout"], 180)
        self.queue_failure = None
        helper.tick(self.task_dir)
        self.assertEqual(len(self.queue_calls), 1)

    def test_uncertain_timeout_preserves_partial_output_and_reports_attention(
        self,
    ) -> None:
        self.create()
        self.queue_failure = subprocess.TimeoutExpired(
            "codex", 180, output=b"starting server", stderr=b"bad byte \xff"
        )
        result = helper.tick(self.task_dir)
        self.assertEqual(result["delivery"]["queue_stdout"], "starting server")
        self.assertIn("bad byte", result["delivery"]["queue_stderr"])
        self.assertTrue(result["attention_required"])
        self.assertEqual(result["delivery_health"], "needs_recovery")
        self.assertTrue(self.active)

    def test_acknowledgement_for_other_thread_or_multiple_receipts_is_not_trusted(
        self,
    ) -> None:
        for output in (
            "Queued message x for thread other.\n",
            f"Queued message x for thread {THREAD}.\nQueued message y for thread {THREAD}.\n",
        ):
            delivery = {"phase": "uncertain"}
            helper.acknowledge(delivery, THREAD, output, None)
            self.assertEqual(delivery["phase"], "uncertain")

    def test_recovery_is_explicit_preserves_old_attempt_and_does_not_send(self) -> None:
        self.create()
        self.queue_failure = subprocess.TimeoutExpired("codex", 45)
        first = helper.tick(self.task_dir)["delivery"]
        for marker, reason, allow in (
            (first["marker"], "authorized recovery", False),
            (first["marker"], "", True),
            ("wrong", "authorized", True),
        ):
            with self.assertRaises(ValueError):
                helper.recover(
                    self.task_dir, marker=marker, reason=reason, allow_duplicate=allow
                )
        result = helper.recover(
            self.task_dir,
            marker=first["marker"],
            reason="user authorized restoration",
            allow_duplicate=True,
        )
        self.assertTrue(result["recovered"])
        self.assertFalse(result["enqueued"])
        self.assertFalse(result["attention_required"])
        self.assertEqual(result["delivery"]["recovery"]["previous_phase"], "uncertain")
        self.assertIn("TimeoutExpired", result["delivery"]["error"])
        self.assertEqual(result["interval_minutes"], 30)
        self.assertEqual(len(self.queue_calls), 1)
        self.queue_failure = None
        second = helper.tick(self.task_dir)
        self.assertTrue(second["enqueued"])
        self.assertNotEqual(first["marker"], second["delivery"]["marker"])
        self.assertEqual(self.state()["deliveries"][0]["phase"], "superseded")
        helper.tick(self.task_dir)
        self.assertEqual(len(self.queue_calls), 2)

    def test_recovery_reconciles_a_late_receipt_before_allowing_retry(self) -> None:
        self.create()
        self.queue_failure = subprocess.TimeoutExpired("codex", 45)
        marker = helper.tick(self.task_dir)["delivery"]["marker"]
        self.start_delivery()
        with self.assertRaisesRegex(ValueError, "must wait"):
            helper.recover(
                self.task_dir,
                marker=marker,
                reason="operator requested",
                allow_duplicate=True,
            )
        self.assertEqual(self.state()["deliveries"][-1]["phase"], "started")
        self.assertEqual(len(self.queue_calls), 1)

    def test_recovery_cannot_bypass_a_changed_rollout(self) -> None:
        self.create()
        self.queue_failure = subprocess.TimeoutExpired("codex", 45)
        marker = helper.tick(self.task_dir)["delivery"]["marker"]
        self.rollout.unlink()
        with self.assertRaisesRegex(ValueError, "history"):
            helper.recover(
                self.task_dir,
                marker=marker,
                reason="operator requested",
                allow_duplicate=True,
            )
        self.assertEqual(len(self.queue_calls), 1)

    def test_crash_after_durable_preparation_suppresses_future_send(self) -> None:
        self.create()
        self.queue_hook = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            helper.tick(self.task_dir)
        self.assertEqual(self.state()["deliveries"][-1]["phase"], "uncertain")
        self.queue_hook = None
        self.assertFalse(helper.tick(self.task_dir)["enqueued"])
        self.assertEqual(len(self.queue_calls), 1)

    def test_process_lock_blocks_concurrent_send(self) -> None:
        self.create()
        entered, release = threading.Event(), threading.Event()
        failures: list[BaseException] = []

        def hold() -> None:
            entered.set()
            self.assertTrue(release.wait(timeout=5))

        def send() -> None:
            try:
                helper.tick(self.task_dir)
            except BaseException as error:
                failures.append(error)

        self.queue_hook = hold
        worker = threading.Thread(target=send)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=5))
            with self.assertRaises(BlockingIOError):
                helper.tick(self.task_dir)
        finally:
            release.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(failures)
        self.assertEqual(len(self.queue_calls), 1)

    def test_cli_lock_conflict_reports_busy_without_send(self) -> None:
        self.create()
        with (self.task_dir / "lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (
                patch.object(
                    helper.sys,
                    "argv",
                    ["cli_heartbeat.py", "tick", "--task-dir", str(self.task_dir)],
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(helper.main(), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "busy")
        self.assertFalse(self.queue_calls)

    def test_changed_or_missing_rollout_blocks_even_after_completion(self) -> None:
        self.create()
        helper.tick(self.task_dir)
        boundary_size = self.state()["deliveries"][-1]["boundary"]["offset"]
        self.start_delivery()
        self.event("task_complete", turn_id="delivery-turn")
        self.assertTrue(helper.status(self.task_dir)["execution_verified"])
        with self.rollout.open("r+b") as handle:
            handle.truncate(boundary_size)
        result = helper.tick(self.task_dir)
        self.assertIn("truncated", result["blocked_reason"])
        self.assertFalse(result["enqueued"])
        self.assertFalse(result["execution_verified"])
        self.assertEqual(len(self.queue_calls), 1)

    def test_rollout_replacement_is_rejected(self) -> None:
        self.create()
        helper.tick(self.task_dir)
        original = self.rollout.read_bytes()
        renamed = self.rollout.with_suffix(".old")
        self.rollout.rename(renamed)
        self.rollout.write_bytes(original)
        result = helper.status(self.task_dir)
        self.assertIn("replaced", result["blocked_reason"])
        self.assertFalse(helper.tick(self.task_dir)["enqueued"])

    def test_prefix_rewrite_without_size_or_inode_change_is_rejected(self) -> None:
        self.create()
        original = self.rollout.read_bytes()
        with self.rollout.open("r+b") as handle:
            handle.write(original.replace(b"03:00:00Z", b"04:00:00Z"))
        result = helper.tick(self.task_dir)
        self.assertIn("prefix changed", result["blocked_reason"])
        self.assertFalse(self.queue_calls)

    def pad_rollout(self, size: int) -> None:
        while self.rollout.stat().st_size < size:
            self.event("token_count", info="x" * 4000)

    def test_rewrite_inside_anchor_window_is_rejected_in_large_rollout(self) -> None:
        self.pad_rollout(3 * helper.WINDOW_BYTES)
        self.create()
        anchor = self.state()["boundary"]["offset"]
        with self.rollout.open("r+b") as handle:
            handle.seek(anchor - 100)
            handle.write(b"y")
        result = helper.tick(self.task_dir)
        self.assertIn("prefix changed", result["blocked_reason"])
        self.assertFalse(self.queue_calls)

    def test_rewrite_before_anchor_window_is_an_accepted_blind_spot(self) -> None:
        # Deliberate trade-off: only WINDOW_BYTES before an anchor are hashed, so
        # an in-place rewrite of older history (which Codex never does) is missed.
        self.pad_rollout(3 * helper.WINDOW_BYTES)
        self.create()
        with self.rollout.open("r+b") as handle:
            handle.seek(self.rollout.stat().st_size - 2 * helper.WINDOW_BYTES)
            handle.write(b"y")
        self.assertTrue(helper.tick(self.task_dir)["enqueued"])

    def downgrade_to_v1(self) -> None:
        task = self.state()
        data = self.rollout.read_bytes()
        anchors = [task["boundary"], task["observed_boundary"]]
        anchors += [delivery["boundary"] for delivery in task["deliveries"]]
        for anchor in anchors:
            del anchor["window_sha256"]
            anchor["sha256"] = hashlib.sha256(data[: anchor["offset"]]).hexdigest()
        task["version"] = 1
        helper.save(self.task_dir, task)

    def test_v1_state_is_verified_and_migrated_for_existing_timers(self) -> None:
        self.create()
        helper.tick(self.task_dir)
        self.start_delivery()
        self.event("task_complete", turn_id="delivery-turn")
        helper.status(self.task_dir)
        self.downgrade_to_v1()
        result = helper.status(self.task_dir)
        self.assertIsNone(result["blocked_reason"])
        self.assertTrue(result["execution_verified"])
        state = self.state()
        self.assertEqual(state["version"], helper.STATE_VERSION)
        self.assertNotIn("sha256", json.dumps(state).replace("window_sha256", ""))
        self.assertTrue(helper.tick(self.task_dir)["enqueued"])
        self.assertEqual(len(self.queue_calls), 2)

    def test_v1_state_with_changed_prefix_blocks_after_migration(self) -> None:
        self.create()
        helper.tick(self.task_dir)
        self.downgrade_to_v1()
        original = self.rollout.read_bytes()
        with self.rollout.open("r+b") as handle:
            handle.write(original.replace(b"03:00:00Z", b"04:00:00Z"))
        result = helper.tick(self.task_dir)
        self.assertIn("prefix changed", result["blocked_reason"])
        self.assertEqual(self.state()["version"], helper.STATE_VERSION)
        self.assertEqual(len(self.queue_calls), 1)

    def test_missing_rollout_explicitly_blocks_delivery(self) -> None:
        self.create()
        self.rollout.unlink()
        result = helper.tick(self.task_dir)
        self.assertTrue(result["blocked_reason"])
        self.assertFalse(self.queue_calls)

    def test_pause_retains_accepted_receipt_and_stops_future_sends(self) -> None:
        self.create()
        helper.tick(self.task_dir)
        paused = helper.pause(self.task_dir)
        self.assertTrue(paused["pause_verified"])
        self.assertEqual(paused["delivery"]["phase"], "queued")
        self.assertIn("may still run", paused["notice"])
        self.assertFalse(helper.tick(self.task_dir)["enqueued"])
        self.assertEqual(len(self.queue_calls), 1)

    def test_resume_changes_interval_and_prompt_and_keeps_receipts(self) -> None:
        self.create()
        helper.tick(self.task_dir)
        self.start_delivery()
        self.event("task_complete", turn_id="delivery-turn")
        helper.pause(self.task_dir)
        new_prompt = self.root / "new-prompt.txt"
        new_prompt.write_text("次の段階を確認", encoding="utf-8")
        self.command_mock.reset_mock()
        result = helper.resume(
            self.task_dir, interval_minutes=15, prompt_file=new_prompt
        )
        self.assertEqual(result["status"], "active")
        self.assertTrue(result["registration_verified"])
        self.assertEqual(result["interval_minutes"], 15)
        register = next(
            call.args[0]
            for call in self.command_mock.call_args_list
            if call.args[0][0] == "systemd-run"
        )
        self.assertIn("--on-unit-active=15min", register)
        self.assertIn("--unit=" + self.state()["unit"], register)
        self.assertEqual(self.state()["deliveries"][0]["phase"], "completed")
        second = helper.tick(self.task_dir)
        self.assertTrue(second["enqueued"])
        self.assertTrue(self.queue_calls[-1][0][7].endswith("\n次の段階を確認"))

    def test_resume_omitted_options_keep_interval_and_prompt(self) -> None:
        self.create()
        helper.pause(self.task_dir)
        result = helper.resume(self.task_dir, interval_minutes=None, prompt_file=None)
        self.assertEqual(result["interval_minutes"], 30)
        self.assertEqual(self.state()["prompt"], self.prompt.read_text())

    def test_resume_waits_for_a_queued_delivery_instead_of_resending(self) -> None:
        self.create()
        helper.tick(self.task_dir)
        helper.pause(self.task_dir)
        result = helper.resume(self.task_dir, interval_minutes=None, prompt_file=None)
        self.assertEqual(result["delivery_health"], "waiting_for_runtime")
        self.assertFalse(helper.tick(self.task_dir)["enqueued"])
        self.assertEqual(len(self.queue_calls), 1)

    def test_resume_refuses_active_unresolved_or_blocked_tasks(self) -> None:
        self.create()
        with self.assertRaisesRegex(ValueError, "Only a paused"):
            helper.resume(self.task_dir, interval_minutes=None, prompt_file=None)
        self.queue_failure = subprocess.TimeoutExpired("codex", 45)
        helper.tick(self.task_dir)
        helper.pause(self.task_dir)
        with self.assertRaisesRegex(ValueError, "Recover"):
            helper.resume(self.task_dir, interval_minutes=None, prompt_file=None)
        self.rollout.unlink()
        with self.assertRaisesRegex(ValueError, "history"):
            helper.resume(self.task_dir, interval_minutes=None, prompt_file=None)
        with self.assertRaisesRegex(ValueError, "positive"):
            helper.resume(self.task_dir, interval_minutes=0, prompt_file=None)
        with self.assertRaises(FileNotFoundError):
            helper.resume(
                self.task_dir,
                interval_minutes=None,
                prompt_file=self.root / "missing.txt",
            )
        self.assertEqual(self.state()["status"], "paused")
        self.assertFalse(self.active)

    def test_failed_resume_registration_keeps_previous_paused_settings(self) -> None:
        self.create()
        helper.pause(self.task_dir)

        def unverified(
            argv: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            completed = self.external(argv, **kwargs)
            if argv[0] == "systemd-run":
                self.interval = 999
            return completed

        self.command_mock.side_effect = unverified
        with self.assertRaisesRegex(RuntimeError, "did not verify"):
            helper.resume(self.task_dir, interval_minutes=15, prompt_file=None)
        state = self.state()
        self.assertEqual(state["status"], "paused")
        self.assertEqual(state["interval_minutes"], 30)

    def test_cli_resume_reports_registration(self) -> None:
        self.create()
        helper.pause(self.task_dir)
        argv = [
            "cli_heartbeat.py",
            "resume",
            "--task-dir",
            str(self.task_dir),
            "--interval-minutes",
            "45",
        ]
        with (
            patch.object(helper.sys, "argv", argv),
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(helper.main(), 0)
        receipt = json.loads(output.getvalue())
        self.assertTrue(receipt["registration_verified"])
        self.assertEqual(receipt["interval_minutes"], 45)

    def test_unavailable_systemd_fails_without_fallback_or_state(self) -> None:
        self.command_mock.side_effect = FileNotFoundError("systemctl unavailable")
        with self.assertRaises(FileNotFoundError):
            self.create()
        self.assertFalse(self.task_dir.exists())
        self.assertFalse(self.queue_calls)

    def test_invalid_schedule_and_windows_state_directory_are_rejected(self) -> None:
        for overrides in (
            {"interval_minutes": 0},
            {"interval_minutes": -1},
            {"queue_timeout_seconds": 0},
            {"state_dir": Path("/mnt/c/unsafe")},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                helper.create_task(self.arguments(**overrides))
        self.assertFalse(self.command_mock.called)

    def test_systemd_duration_parser_accepts_canonical_hour_format(self) -> None:
        self.assertEqual(helper.duration_seconds("1h 30min"), 5400)
        self.assertEqual(helper.duration_seconds("1min 500ms"), 60.5)
        with self.assertRaises(ValueError):
            helper.duration_seconds("unexpected")

    def test_systemd_255_repeated_timer_keys_are_retained_and_verified(self) -> None:
        self.command_mock.side_effect = [
            subprocess.CompletedProcess([], 0, SYSTEMD_255_TIMER, ""),
            subprocess.CompletedProcess([], 0, "fixture.timer next run\n", ""),
        ]
        observed = helper.timer_status({"unit": "fixture", "interval_minutes": 30})
        self.assertTrue(observed["registration_verified"])
        self.assertEqual(
            observed["actual_timer"]["TimersMonotonic"],
            "{ OnUnitActiveUSec=30min ; next_elapse=0 }\n"
            "{ OnActiveUSec=30min ; next_elapse=1d 3h 52min 49.642424s }",
        )

    def test_timer_requires_both_distinct_flags_at_the_requested_interval(self) -> None:
        for incorrect in (
            SYSTEMD_255_TIMER.replace("OnUnitActiveUSec", "OnActiveUSec"),
            SYSTEMD_255_TIMER.replace("OnActiveUSec", "OnUnitActiveUSec"),
            SYSTEMD_255_TIMER.replace(
                "OnUnitActiveUSec=30min", "OnUnitActiveUSec=60min"
            ),
        ):
            with self.subTest(raw=incorrect):
                self.command_mock.side_effect = [
                    subprocess.CompletedProcess([], 0, incorrect, ""),
                    subprocess.CompletedProcess([], 0, "fixture.timer next run\n", ""),
                ]
                self.assertFalse(
                    helper.timer_status({"unit": "fixture", "interval_minutes": 30})[
                        "registration_verified"
                    ]
                )

    def test_pause_exit_zero_preserves_past_failure_when_collected_timer_is_stopped(
        self,
    ) -> None:
        self.create()
        reason = "Timer registration/interval/next run did not verify; inspect status or pause"
        task = self.state()
        task.update(status="registration_failed", blocked_reason=reason)
        helper.save(self.task_dir, task)

        def stopped(
            argv: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if argv[0] == "systemctl" and "show" in argv:
                # Matches the collected timer in live-test/cleanup-0.json.
                output = (
                    f"Id={task['unit']}.timer\nTriggers=\nLoadState=not-found\n"
                    "ActiveState=inactive\nNextElapseUSecRealtime=\n"
                    "NextElapseUSecMonotonic=infinity\n"
                )
                return subprocess.CompletedProcess(argv, 4, output, "")
            return self.external(argv, **kwargs)

        self.command_mock.side_effect = stopped
        argv = ["cli_heartbeat.py", "pause", "--task-dir", str(self.task_dir)]
        with (
            patch.object(helper.sys, "argv", argv),
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(helper.main(), 0)
        receipt = json.loads(output.getvalue())
        self.assertTrue(receipt["pause_verified"])
        self.assertEqual(receipt["blocked_reason"], reason)
        self.assertEqual(self.state()["blocked_reason"], reason)
        self.assertEqual(self.state()["status"], "paused")
        self.assertFalse(self.queue_calls)
        # A status query still reports the retained historical failure.
        argv[1] = "status"
        with patch.object(helper.sys, "argv", argv), redirect_stdout(io.StringIO()):
            self.assertEqual(helper.main(), 1)

    def test_pause_exit_remains_nonzero_when_stop_did_not_verify(self) -> None:
        self.create()

        def failed_stop(
            argv: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "stop" in argv:
                return subprocess.CompletedProcess(argv, 0, "", "")
            return self.external(argv, **kwargs)

        self.command_mock.side_effect = failed_stop
        with (
            patch.object(
                helper.sys,
                "argv",
                ["cli_heartbeat.py", "pause", "--task-dir", str(self.task_dir)],
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(helper.main(), 1)
        self.assertIn("did not verify", json.loads(output.getvalue())["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
