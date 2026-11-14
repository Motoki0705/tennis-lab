"""PYTHONPATH=.agents/ops .venv/bin/python -m ci_watch."""

from lib.checkout_lock import guard_interactive_cli

guard_interactive_cli()

from .cli import main  # noqa: E402

raise SystemExit(main())
