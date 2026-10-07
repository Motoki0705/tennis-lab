"""Behavioral tests using isolated Codex homes and synthetic app databases."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import heartbeat


class HeartbeatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="heartbeat-test-")
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.task_id = "test-followup"
        self.prompt = '監視対象 "job A"\n終了後だけ処理し、完了時に停止。'
        self.thread = "verified-current-thread"
        heartbeat.create_task(
            self.home, self.task_id, "毎時監視", self.prompt, self.thread, 1
        )
        self.config_path = heartbeat.config_path(self.home, self.task_id)
        self.database = self.home / "sqlite/codex-dev.db"
        self.database.parent.mkdir()
        self.db = sqlite3.connect(self.database)
        self.addCleanup(self.db.close)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA wal_autocheckpoint=0")
        self.db.execute("""CREATE TABLE automations (
            id TEXT PRIMARY KEY, name TEXT, prompt TEXT, kind TEXT, status TEXT,
            rrule TEXT, target_thread_id TEXT, next_run_at INTEGER,
            next_run_nominal_at INTEGER, last_run_at INTEGER, legacy_automation_id TEXT
        )""")
        self.db.commit()
        # Leave the row in WAL, so verifying only the base file would fail.
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        config, _ = heartbeat.load_config(self.config_path)
        keys = ("id", "name", "prompt", "kind", "status", "rrule", "target_thread_id")
        self.db.execute(
            "INSERT INTO automations VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (*[config[key] for key in keys], 1791338717393, 1791338437393, None, None),
        )
        self.db.commit()

    def verify(self) -> dict:
        return heartbeat.verify_task(
            self.home, self.task_id, self.database, "Asia/Tokyo"
        )

    def change_native(self, field: str, value: object) -> None:
        allowed = {"target_thread_id", "prompt", "rrule", "status", "next_run_at"}
        if field not in allowed:
            raise ValueError(field)
        self.db.execute(f"UPDATE automations SET {field}=?", (value,))
        self.db.commit()

    def test_unicode_and_newline_prompt_round_trip(self) -> None:
        config, _ = heartbeat.load_config(self.config_path)
        self.assertEqual(config["prompt"], self.prompt)
        self.assertEqual(config["target_thread_id"], self.thread)
        self.assertIsInstance(config["created_at"], int)
        self.assertEqual(config["rrule"], "FREQ=HOURLY;INTERVAL=1")

    def test_identical_creation_is_noop(self) -> None:
        before = self.config_path.read_bytes()
        result = heartbeat.create_task(
            self.home, self.task_id, "毎時監視", self.prompt, self.thread, 1
        )
        self.assertFalse(result["changed"])
        self.assertFalse(result["registration_verified"])
        self.assertEqual(self.config_path.read_bytes(), before)

    def test_different_task_does_not_overwrite(self) -> None:
        before = self.config_path.read_bytes()
        with self.assertRaises(FileExistsError):
            heartbeat.create_task(
                self.home, self.task_id, "毎時監視", self.prompt, "other-thread", 1
            )
        self.assertEqual(self.config_path.read_bytes(), before)

    def test_invalid_id_and_missing_thread_create_nothing(self) -> None:
        for task_id, thread in (("../outside", self.thread), ("missing-thread", "")):
            with self.subTest(task_id=task_id), self.assertRaises(ValueError):
                heartbeat.create_task(self.home, task_id, "name", "prompt", thread, 1)
        self.assertFalse((self.home / "automations/missing-thread").exists())

    def test_invalid_interval_is_rejected(self) -> None:
        for interval in (0, -1, 1.5, True):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                heartbeat.create_task(
                    self.home, "invalid-hours", "name", "prompt", self.thread, interval
                )

    def test_cli_honors_explicit_schedule_and_defaults_only_when_omitted(self) -> None:
        custom_rule = "DTSTART;TZID=Asia/Tokyo:20261007T090000\nRRULE:FREQ=WEEKLY;BYDAY=MO,FR;BYHOUR=9;BYMINUTE=0"
        cases = (
            ([], "FREQ=HOURLY;INTERVAL=1"),
            (["--interval-minutes", "30"], "FREQ=MINUTELY;INTERVAL=30"),
            (["--interval-hours", "2"], "FREQ=HOURLY;INTERVAL=2"),
            (["--rrule", custom_rule], custom_rule),
        )
        prompt = self.home / "prompt.txt"
        prompt.write_text(self.prompt, encoding="utf-8")
        for index, (options, expected) in enumerate(cases):
            with self.subTest(options=options):
                task_id = f"cadence-{index}"
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(heartbeat.__file__)),
                        "create",
                        "--codex-home",
                        str(self.home),
                        "--id",
                        task_id,
                        "--name",
                        "cadence",
                        "--prompt-file",
                        str(prompt),
                        "--thread-id",
                        self.thread,
                        *options,
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                config, _ = heartbeat.load_config(
                    heartbeat.config_path(self.home, task_id)
                )
                self.assertEqual(config["rrule"], expected)
                self.assertEqual(json.loads(result.stdout)["rrule"], expected)

    def test_cli_invalid_or_conflicting_schedule_does_not_fall_back(self) -> None:
        prompt = self.home / "prompt.txt"
        prompt.write_text(self.prompt, encoding="utf-8")
        for options in (
            ["--interval-minutes", "0"],
            ["--interval-hours", "-1"],
            ["--rrule", ""],
            ["--interval-hours", "1", "--interval-minutes", "30"],
            ["--interval-hours", "1", "--rrule", "FREQ=DAILY"],
        ):
            with self.subTest(options=options):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(heartbeat.__file__)),
                        "create",
                        "--codex-home",
                        str(self.home),
                        "--id",
                        "invalid-cadence",
                        "--name",
                        "cadence",
                        "--prompt-file",
                        str(prompt),
                        "--thread-id",
                        self.thread,
                        *options,
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.home / "automations/invalid-cadence").exists())

    def test_minute_intervals_and_conflicting_function_inputs_are_rejected(
        self,
    ) -> None:
        for interval in (0, -1, 1.5, True):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                heartbeat.schedule_rrule(interval_minutes=interval)
        with self.assertRaises(ValueError):
            heartbeat.schedule_rrule(interval_hours=1, interval_minutes=30)
        with self.assertRaises(ValueError):
            heartbeat.schedule_rrule(rrule="   ")

    def test_symlinked_task_is_rejected(self) -> None:
        (self.home / "automations/alias").symlink_to(
            self.config_path.parent, target_is_directory=True
        )
        with self.assertRaises(ValueError):
            heartbeat.set_status(self.home, "alias", "PAUSED")

    def test_pause_preserves_metadata_and_create_does_not_resume(self) -> None:
        extra = '\n# Preserve custom metadata\nmodel = "explicit-model"\n'
        with self.config_path.open("a") as handle:
            handle.write(extra)
        before, _ = heartbeat.load_config(self.config_path)
        heartbeat.set_status(self.home, self.task_id, "PAUSED")
        after, _ = heartbeat.load_config(self.config_path)
        self.assertEqual(after["status"], "PAUSED")
        self.assertGreater(after["updated_at"], before["updated_at"])
        self.assertEqual(after["model"], "explicit-model")
        self.assertIn(extra, self.config_path.read_text())
        result = heartbeat.create_task(
            self.home, self.task_id, "毎時監視", self.prompt, self.thread, 1
        )
        self.assertFalse(result["changed"])
        self.assertEqual(result["status"], "PAUSED")

    def test_update_cadence_and_resume_preserves_unrelated_fields(self) -> None:
        extra = '\n# Operator note\nmodel = "chosen-model"\nreasoning_effort = "high"\n'
        with self.config_path.open("a") as handle:
            handle.write(extra)
        heartbeat.set_status(self.home, self.task_id, "PAUSED")
        before, _ = heartbeat.load_config(self.config_path)
        result = heartbeat.update_task(
            self.home, self.task_id, interval_minutes=15, status="ACTIVE"
        )
        after, raw = heartbeat.load_config(self.config_path)
        self.assertTrue(result["changed"])
        self.assertFalse(result["registration_verified"])
        self.assertEqual(after["rrule"], "FREQ=MINUTELY;INTERVAL=15")
        self.assertEqual(after["status"], "ACTIVE")
        self.assertGreater(after["updated_at"], before["updated_at"])
        for key in before.keys() - {"rrule", "status", "updated_at"}:
            self.assertEqual(before[key], after[key], key)
        self.assertIn(extra, raw.decode())
        repeated = heartbeat.update_task(
            self.home, self.task_id, interval_minutes=15, status="ACTIVE"
        )
        self.assertFalse(repeated["changed"])
        self.assertEqual(self.config_path.read_bytes(), raw)
        self.assertEqual(repeated["config_sha256"], result["config_sha256"])

    def test_update_preserves_omitted_status_and_cadence(self) -> None:
        heartbeat.set_status(self.home, self.task_id, "PAUSED")
        heartbeat.update_task(self.home, self.task_id, interval_minutes=15)
        changed, _ = heartbeat.load_config(self.config_path)
        self.assertEqual(changed["status"], "PAUSED")
        heartbeat.update_task(self.home, self.task_id, status="ACTIVE")
        resumed, _ = heartbeat.load_config(self.config_path)
        self.assertEqual(resumed["rrule"], "FREQ=MINUTELY;INTERVAL=15")

    def test_update_preserves_multiline_recurrence(self) -> None:
        rule = "DTSTART;TZID=Asia/Tokyo:20261008T090000\nRRULE:FREQ=WEEKLY;BYDAY=MO,FR"
        heartbeat.update_task(self.home, self.task_id, rrule=rule)
        config, _ = heartbeat.load_config(self.config_path)
        self.assertEqual(config["rrule"], rule)
        heartbeat.set_status(self.home, self.task_id, "PAUSED")
        config, _ = heartbeat.load_config(self.config_path)
        self.assertEqual(config["rrule"], rule)

    def test_invalid_update_cannot_reset_or_modify_existing_task(self) -> None:
        original = self.config_path.read_bytes()
        for options in (
            {},
            {"rrule": ""},
            {"interval_minutes": 0},
            {"interval_hours": 2, "interval_minutes": 15},
            {"status": "DELETED"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                heartbeat.update_task(self.home, self.task_id, **options)
            self.assertEqual(self.config_path.read_bytes(), original)

    def test_update_missing_task_does_not_create_it(self) -> None:
        with self.assertRaises(FileNotFoundError):
            heartbeat.update_task(self.home, "missing-task", status="ACTIVE")
        self.assertFalse((self.home / "automations/missing-task").exists())

    def test_cli_update_from_outside_source_and_repeat(self) -> None:
        command = [
            sys.executable,
            str(Path(heartbeat.__file__)),
            "update",
            "--codex-home",
            str(self.home),
            "--id",
            self.task_id,
            "--interval-minutes",
            "15",
            "--status",
            "ACTIVE",
        ]
        first = subprocess.run(
            command, cwd=self.home, text=True, capture_output=True, check=False
        )
        self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
        original = self.config_path.read_bytes()
        second = subprocess.run(
            command, cwd=self.home, text=True, capture_output=True, check=False
        )
        self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
        self.assertTrue(json.loads(first.stdout)["changed"])
        self.assertFalse(json.loads(second.stdout)["changed"])
        self.assertEqual(
            json.loads(first.stdout)["config_sha256"],
            json.loads(second.stdout)["config_sha256"],
        )
        self.assertEqual(self.config_path.read_bytes(), original)

    def test_verify_reads_wal_without_changing_source(self) -> None:
        files = (self.database, Path(str(self.database) + "-wal"))
        before = [file.read_bytes() for file in files]
        result = self.verify()
        self.assertTrue(result["registration_verified"])
        self.assertFalse(result["execution_verified"])
        self.assertEqual(result["next_run_at_iso"], "2026-10-07T11:05:17.393000+09:00")
        self.assertEqual([file.read_bytes() for file in files], before)

    def test_stale_native_configuration_is_rejected(self) -> None:
        original, _ = heartbeat.load_config(self.config_path)
        for field, wrong in (
            ("target_thread_id", "another-thread"),
            ("prompt", "old-prompt"),
            ("rrule", "FREQ=HOURLY;INTERVAL=2"),
            ("status", "PAUSED"),
        ):
            with self.subTest(field=field):
                self.change_native(field, wrong)
                with self.assertRaisesRegex(ValueError, field):
                    self.verify()
                self.change_native(field, original[field])

    def test_active_without_next_run_is_not_registered(self) -> None:
        self.change_native("next_run_at", None)
        with self.assertRaisesRegex(ValueError, "no scheduled next run"):
            self.verify()

    def test_pause_needs_native_acknowledgement(self) -> None:
        heartbeat.set_status(self.home, self.task_id, "PAUSED")
        with self.assertRaisesRegex(ValueError, "status"):
            self.verify()
        self.change_native("status", "PAUSED")
        with self.assertRaisesRegex(ValueError, "cleared"):
            self.verify()
        self.change_native("next_run_at", None)
        self.assertTrue(self.verify()["registration_verified"])

    def test_unimported_task_fails(self) -> None:
        self.db.execute("DELETE FROM automations")
        self.db.commit()
        with self.assertRaisesRegex(ValueError, "found 0"):
            self.verify()

    def test_missing_database_is_not_created(self) -> None:
        path = self.home / "missing.db"
        with self.assertRaises(FileNotFoundError):
            heartbeat.native_rows(path, self.task_id)
        self.assertFalse(path.exists())

    def test_changed_snapshot_is_inconclusive(self) -> None:
        original = heartbeat.file_signature(self.database)
        with (
            patch.object(
                heartbeat, "file_signature", side_effect=[original, None, (0,), None]
            ),
            self.assertRaisesRegex(RuntimeError, "changed during snapshot"),
        ):
            self.verify()

    def test_migrated_native_id_is_reported(self) -> None:
        self.db.execute(
            "UPDATE automations SET id=?,legacy_automation_id=?",
            ("native-assigned-id", self.task_id),
        )
        self.db.commit()
        result = self.verify()
        self.assertEqual(result["id"], "native-assigned-id")
        self.assertEqual(result["configuration_id"], self.task_id)

    def test_cli_reports_unverified_without_thread_id(self) -> None:
        prompt = self.home / "prompt.txt"
        prompt.write_text("監視してください")
        environment = {
            key: value for key, value in os.environ.items() if key != "CODEX_THREAD_ID"
        }
        result = subprocess.run(
            [
                sys.executable,
                str(Path(heartbeat.__file__)),
                "create",
                "--codex-home",
                str(self.home),
                "--id",
                "cli-test",
                "--name",
                "test",
                "--prompt-file",
                str(prompt),
            ],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)["registration_verified"])
        self.assertFalse((self.home / "automations/cli-test").exists())

    def test_receipt_cannot_overwrite_native_database_or_config(self) -> None:
        for destination in (self.database, self.config_path):
            with self.subTest(destination=destination):
                before = destination.read_bytes()
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(heartbeat.__file__)),
                        "verify",
                        "--codex-home",
                        str(self.home),
                        "--id",
                        self.task_id,
                        "--receipt",
                        str(destination),
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("must not overwrite", json.loads(result.stdout)["error"])
                self.assertEqual(destination.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
