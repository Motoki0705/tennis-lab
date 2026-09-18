"""Compare matched SLCS conditions using explicit video-prefix domain rules."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.tasks.slcs.evaluation.comparison import (
    CONDITIONS,
    compare_conditions,
    save_comparison,
)
from src.utils.configuration.paths import PathResolver, PathRole, RuntimePathRoots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    for mode in CONDITIONS:
        parser.add_argument(f"--{mode.replace('_', '-')}", required=True, help="OUTPUT-relative evaluation bundle")
    parser.add_argument("--domain-prefix", action="append", required=True, help="Explicit PREFIX=DOMAIN, e.g. video_=meiji")
    parser.add_argument("--default-domain", required=True, help="Explicit domain for unmatched prefixes, e.g. broadcast")
    parser.add_argument("--output", required=True, help="slcs/analyze/<experiment>/<run-id>")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output_root = args.output_root.resolve()
    resolver = PathResolver(RuntimePathRoots(
        project_root=root, data_root=(root / "data").resolve(), checkpoint_root=(root / "ckpt").resolve(),
        artifact_root=output_root, output_root=output_root, cache_root=(root / ".cache").resolve(), external_asset_root=(root / "third_party").resolve(),
    ))
    fragment = Path(args.output)
    if len(fragment.parts) != 4 or fragment.parts[:2] != ("slcs", "analyze"):
        parser.error("--output must be slcs/analyze/<experiment>/<run-id>")
    prefixes = []
    for rule in args.domain_prefix:
        prefix, separator, domain = rule.partition("=")
        if not prefix or not separator or not domain:
            parser.error("--domain-prefix must be a nonempty PREFIX=DOMAIN")
        prefixes.append((prefix, domain))
    bundles = {mode: resolver.resolve(PathRole.OUTPUT, getattr(args, mode)) for mode in CONDITIONS}
    with np.load(bundles["full"] / "eval_arrays.npz", allow_pickle=False) as archive:
        videos = archive["video_ids"].tolist()
    domains = {}
    for video in videos:
        matches = [domain for prefix, domain in prefixes if video.startswith(prefix)]
        if len(matches) > 1:
            parser.error(f"Overlapping domain rules for {video}")
        domains[video] = matches[0] if matches else args.default_domain
    report = compare_conditions(bundles, domains)
    for path in save_comparison(report, resolver.resolve(PathRole.OUTPUT, args.output)):
        print(path)


if __name__ == "__main__":
    main()
