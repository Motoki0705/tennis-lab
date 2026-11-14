"""Parent CLI bootstrap; lock finite expiry before importing project code."""

import argparse

# Mirror only root options to identify the subcommand. Values such as a spec
# filename named improvement-expire must not change another command's policy.
bootstrap = argparse.ArgumentParser(add_help=False)
for option in ("--repo", "--state-dir", "--dispatcher-state-dir", "--parent"):
    bootstrap.add_argument(option)
_, remaining = bootstrap.parse_known_args()
if remaining and remaining[0] == "improvement-expire":
    from lib.checkout_lock import require_checkout_job

    require_checkout_job()

from .cli import main  # noqa: E402

raise SystemExit(main())
