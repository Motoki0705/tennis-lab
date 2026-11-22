"""Report a systemd user unit failure through a Windows toast."""

from __future__ import annotations

import argparse
import shlex
import sys

from .windows_toast import ToastError, ToastNotSent, send_windows_toast


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unit", help="failed systemd user unit")
    unit = parser.parse_args(argv).unit
    journal = shlex.join(("journalctl", "--user", "-u", unit))
    try:
        send_windows_toast("tennis-lab systemd failure", f"{unit} failed.\n{journal}")
    except ToastError as exc:
        outcome = (
            "was not sent (confirmed)"
            if isinstance(exc, ToastNotSent)
            else "submission outcome is unknown (resending may produce a duplicate)"
        )
        print(f"Windows toast for {unit} {outcome}. {exc}", file=sys.stderr)
        return 1
    print(f"Windows toast submitted for {unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
