#!/usr/bin/env python
"""Generate train/val convergence curves for knowledge-graph run nodes.

Each run node records its final ``test/*`` metrics in frontmatter. Those metrics
are a *fingerprint*: the TensorBoard run whose final ``test/*`` scalars match a
node's ``metrics`` is that node's run — no matter which ``outputs/.../version_N``
directory it landed in. This script uses that fingerprint to locate the event
directory, then plots the train/val curves (loss + headline errors) into
``knowledge/runs/<id>/curves.png`` and records ``artifacts.curves`` /
``artifacts.tb_logdir`` in the node so the webui can show it.

As a fast path, a node that already records ``artifacts.tb_logdir`` or
``artifacts.output_dir`` (kg_register fills the latter from the run's
``output_dir.txt``) resolves its event dir directly — the repo-wide fingerprint
scan only runs for older nodes that have neither. ``--refresh`` forces the scan.

The goal is qualitative: *see how a run converged* when clicking a node.

If no event directory matches a node's metrics within tolerance, the node is
skipped by design — some runs predate TensorBoard logging or their event files
are gone. That is expected and not an error.

Usage:
    PY=.venv/bin/python
    SKILL=.agents/skills/knowledge-control/scripts
    $PY $SKILL/kg_curves.py --all                 # every node, auto-discover
    $PY $SKILL/kg_curves.py run-i520-canon-both    # one node
    $PY $SKILL/kg_curves.py --all --dry-run        # report matches, write nothing
    $PY $SKILL/kg_curves.py --all \
        --scan-roots /abs/path/to/main-checkout/outputs   # extra search roots

Run from the repo root. When run inside a linked git worktree the main
checkout's ``outputs/`` is added to the search roots automatically (the event
files live there, not in the worktree).
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

from kg_lib import dump_frontmatter, load_nodes, nodes_dir, parse_node, repo_root

# Metrics to draw, in priority order. Only bases that actually exist as a
# ``train/<base>`` or ``val/<base>`` series are plotted; we keep at most MAX_PANELS.
CURVE_PRIORITY = [
    "loss",
    "pos_error_m",
    "position_error_m",
    "ang_error_deg",
    "angular_error_deg",
    "miou",
    "mean_iou",
    "iou",
    "dice",
    "best_val_miou",
    "accuracy",
    "acc",
    "mAP",
    "psnr",
]
MAX_PANELS = 4
# Tags that are never interesting as convergence panels.
SKIP_BASES = {"epoch", "hp_metric", "step"}

# Fingerprint matching thresholds.
REL_TOL = 0.02       # a metric "matches" within 2% relative error
MIN_SHARED = 3       # need at least this many shared metric keys to trust a match
MIN_MATCH_FRAC = 0.8  # and at least this fraction of shared keys must match


def git_common_checkout() -> Path | None:
    """Return the main checkout dir of the current repo (for worktree usage)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=repo_root(), capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    common = Path(out)
    if not common.is_absolute():
        common = (repo_root() / common).resolve()
    # common is <main-checkout>/.git ; its parent is the main checkout
    return common.parent if common.name == ".git" else None


def bundle_scan_roots() -> list[Path]:
    """Output dirs recorded by run bundles (run.json ``cwd`` / ``repo_root``).

    Runs done in sibling worktrees (e.g. ``/home/.../wt/i525-asym``) keep their
    TensorBoard files outside this checkout; the bundle records where they ran.
    """
    roots: list[Path] = []
    runs = nodes_dir().parent / "runs"
    if not runs.exists():
        return roots
    for rj in runs.glob("*/run.json"):
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        for key in ("cwd", "repo_root"):
            base = data.get(key)
            if base and (Path(base) / "outputs").exists():
                roots.append((Path(base) / "outputs").resolve())
    return roots


def default_scan_roots() -> list[Path]:
    roots: list[Path] = []
    for base in (repo_root(), git_common_checkout()):
        if base is None:
            continue
        for sub in ("outputs", ".claude/worktrees"):
            p = base / sub
            if p.exists():
                roots.append(p.resolve())
    roots += bundle_scan_roots()
    # de-dup preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def events_under(path: Path) -> Path | None:
    """Resolve a path hint (a dir or file path) to an event dir, if any."""
    if not path.exists():
        return None
    cand = path if path.is_dir() else path.parent
    if list(cand.glob("events.out.tfevents*")):
        return cand
    for v in sorted(cand.glob("logs/version_*"), reverse=True):
        if list(v.glob("events.out.tfevents*")):
            return v
    hits = [
        Path(p).parent
        for p in glob.glob(str(cand / "**" / "events.out.tfevents*"), recursive=True)
        if "Zone.Identifier" not in p
    ]
    return sorted(hits)[-1] if hits else None


def checkout_bases() -> list[Path]:
    """Checkout roots a relative ``output_dir`` could be anchored to.

    ``output_dir`` is recorded relative to the checkout that ran the job, which
    is usually the main checkout — not this worktree, whose gitignored
    ``outputs/`` is empty."""
    bases: list[Path] = [repo_root()]
    common = git_common_checkout()
    if common is not None:
        bases.append(common)
    runs = nodes_dir().parent / "runs"
    if runs.exists():
        for rj in runs.glob("*/run.json"):
            try:
                data = json.loads(rj.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            for key in ("cwd", "repo_root"):
                if data.get(key):
                    bases.append(Path(data[key]))
    seen: set[Path] = set()
    uniq: list[Path] = []
    for b in bases:
        rb = b.resolve()
        if rb not in seen and rb.exists():
            seen.add(rb)
            uniq.append(rb)
    return uniq


def path_hint_event_dir(artifacts: dict, bases: list[Path]) -> Path | None:
    """Fallback for runs without ``test/*`` scalars (e.g. court seg): use the
    ``output_dir`` / ``checkpoint`` path recorded in the node's artifacts."""
    for key in ("output_dir", "checkpoint", "log"):
        val = artifacts.get(key)
        if not val:
            continue
        p = Path(str(val))
        candidates = [p] if p.is_absolute() else [b / p for b in bases]
        for cand in candidates:
            d = events_under(cand)
            if d:
                return d
    return None


def find_event_dirs(roots: list[Path]) -> list[Path]:
    dirs: set[Path] = set()
    for root in roots:
        for ev in glob.glob(str(root / "**" / "events.out.tfevents*"), recursive=True):
            if "Zone.Identifier" in ev:
                continue
            dirs.add(Path(ev).parent)
    return sorted(dirs)


def _accumulator(event_dir: Path):
    from tensorboard.backend.event_processing import event_accumulator

    acc = event_accumulator.EventAccumulator(
        str(event_dir), size_guidance={event_accumulator.SCALARS: 0}
    )
    acc.Reload()
    return acc


def final_test_metrics(event_dir: Path, _cache: dict[Path, dict] = {}) -> dict[str, float]:
    """Final value of every ``test/<key>`` scalar in an event dir (cached)."""
    if event_dir in _cache:
        return _cache[event_dir]
    out: dict[str, float] = {}
    try:
        acc = _accumulator(event_dir)
        for tag in acc.Tags().get("scalars", []):
            if tag.startswith("test/"):
                try:
                    out[tag[len("test/"):]] = float(acc.Scalars(tag)[-1].value)
                except (ValueError, IndexError):
                    pass
    except Exception as exc:  # noqa: BLE001 - one bad event file shouldn't abort
        print(f"  warn: could not read {event_dir}: {exc}", file=sys.stderr)
    _cache[event_dir] = out
    return out


def match_event_dir(
    metrics: dict[str, float], event_dirs: list[Path]
) -> tuple[Path | None, float, int]:
    """Best-matching event dir for a node's test metrics (fingerprint match)."""
    best: tuple[Path | None, float, int] = (None, 1e9, 0)
    for d in event_dirs:
        tm = final_test_metrics(d)
        shared = [k for k in metrics if k in tm and isinstance(metrics[k], (int, float))]
        if len(shared) < MIN_SHARED:
            continue
        rel = [abs(float(metrics[k]) - tm[k]) / (abs(tm[k]) + 1e-9) for k in shared]
        matched = sum(1 for r in rel if r < REL_TOL)
        if matched / len(shared) < MIN_MATCH_FRAC:
            continue
        mean_err = sum(rel) / len(rel)
        # a near-exact match on every shared key is the run itself; stop scanning
        # (avoids reading the remaining event dirs).
        if matched == len(shared) and mean_err < REL_TOL / 10:
            return (d, mean_err, matched)
        # otherwise prefer more matched keys, then lower error
        if (matched, -mean_err) > (best[2], -best[1]):
            best = (d, mean_err, matched)
    return best


def _series(acc, base: str) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {}
    tags = set(acc.Tags().get("scalars", []))
    for split in ("train", "val"):
        tag = f"{split}/{base}"
        if tag in tags:
            pts = [(s.step, s.value) for s in acc.Scalars(tag)]
            if len(pts) >= 2:
                out[split] = pts
    return out


def select_bases(acc) -> list[str]:
    tags = acc.Tags().get("scalars", [])
    bases_available: set[str] = set()
    for t in tags:
        for split in ("train/", "val/"):
            if t.startswith(split):
                bases_available.add(t[len(split):])
    chosen: list[str] = []
    for base in CURVE_PRIORITY:
        if base in bases_available and base not in chosen:
            chosen.append(base)
        if len(chosen) >= MAX_PANELS:
            return chosen
    # fall back to any remaining base that has both train & val, alphabetical
    for base in sorted(bases_available):
        if len(chosen) >= MAX_PANELS:
            break
        if base in SKIP_BASES or base.startswith("lr") or base in chosen:
            continue
        if f"train/{base}" in tags and f"val/{base}" in tags:
            chosen.append(base)
    return chosen


def plot_curves(acc, node_id: str, out_png: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bases = select_bases(acc)
    if not bases:
        return False
    n = len(bases)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax, base in zip(flat, bases):
        series = _series(acc, base)
        for split, color in (("train", "#d97757"), ("val", "#4285f4")):
            if split in series:
                xs = [p[0] for p in series[split]]
                ys = [p[1] for p in series[split]]
                ax.plot(xs, ys, label=split, color=color, linewidth=1.4)
        ax.set_title(base, fontsize=11)
        ax.set_xlabel("step", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
        if base == "loss":
            vals = [v for s in series.values() for _, v in s]
            if vals and min(vals) > 0:
                ax.set_yscale("log")
    for ax in flat[n:]:
        ax.axis("off")
    # node id only: titles may be Japanese and matplotlib has no CJK font here;
    # the webui shows the title next to the image anyway.
    fig.suptitle(node_id, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return True


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def rel_to_checkout(path: Path) -> str | None:
    """Path relative to whichever checkout owns it (worktree or main checkout).

    Event dirs usually live in the main checkout's gitignored ``outputs/`` (i.e.
    outside this worktree), so we strip the checkout prefix to keep the stored
    ``tb_logdir`` portable (``outputs/...``) rather than an absolute machine path.
    Returns None for dirs under no known checkout (e.g. sibling worktrees), in
    which case the caller omits ``tb_logdir`` and re-fingerprints next time."""
    p = path.resolve()
    bases = [repo_root().resolve()]
    common = git_common_checkout()
    if common is not None:
        bases.append(common.resolve())
    for b in bases:
        try:
            return str(p.relative_to(b))
        except ValueError:
            continue
    return None


def update_artifacts(node_path: Path, curves_rel: str, tb_rel: str | None) -> None:
    node = parse_node(node_path)
    artifacts = dict(node.meta.get("artifacts") or {})
    artifacts["curves"] = curves_rel
    if tb_rel:
        artifacts["tb_logdir"] = tb_rel
    else:
        artifacts.pop("tb_logdir", None)
    node.meta["artifacts"] = artifacts
    text = f"---\n{dump_frontmatter(node.meta)}---\n\n{node.body}\n"
    node_path.write_text(text, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("nodes", nargs="*", help="node id(s); omit with --all")
    p.add_argument("--all", action="store_true", help="process every run node")
    p.add_argument("--scan-roots", nargs="*", type=Path, default=[], help="extra dirs to search for event files")
    p.add_argument("--refresh", action="store_true", help="re-fingerprint even if artifacts.tb_logdir is set")
    p.add_argument("--dry-run", action="store_true", help="report matches; write no files")
    args = p.parse_args()

    extra_roots = [r.resolve() for r in args.scan_roots if r.exists()]

    # Scanning every event file under the repo + worktrees is the slow part, so
    # build the fingerprint index lazily — only once a node has no fast path hint
    # (artifacts.tb_logdir / artifacts.output_dir) to resolve directly. Nodes
    # registered with an output_dir never trigger it.
    _index: dict[str, list[Path]] = {}

    def event_dirs_index() -> list[Path]:
        if "dirs" not in _index:
            roots = default_scan_roots() + extra_roots
            print(f"scan roots: {', '.join(rel(r) for r in roots)}")
            _index["dirs"] = find_event_dirs(roots)
            print(f"found {len(_index['dirs'])} event dirs")
        return _index["dirs"]

    all_nodes = {n.id: n for n in load_nodes()}
    if args.all:
        targets = [n for n in all_nodes.values() if n.type == "run"]
    else:
        if not args.nodes:
            p.error("pass node id(s) or --all")
        targets = []
        for nid in args.nodes:
            if nid not in all_nodes:
                print(f"  warn: node '{nid}' not found", file=sys.stderr)
                continue
            targets.append(all_nodes[nid])

    runs_dir = nodes_dir().parent / "runs"
    bases = checkout_bases()
    written = skipped = 0
    for node in sorted(targets, key=lambda n: n.id):
        metrics = {
            k: float(v)
            for k, v in (node.meta.get("metrics") or {}).items()
            if isinstance(v, (int, float))
        }
        artifacts = node.meta.get("artifacts") or {}
        event_dir: Path | None = None
        # Fast paths first (no global scan): the cached tb_logdir, then the
        # output_dir/checkpoint path recorded at registration. --refresh forces
        # the fingerprint scan to re-derive from scratch.
        if not args.refresh:
            if artifacts.get("tb_logdir"):
                cand = repo_root() / str(artifacts["tb_logdir"])
                if cand.exists():
                    event_dir = cand
            if event_dir is None:
                event_dir = path_hint_event_dir(artifacts, bases)
                if event_dir is not None:
                    print(f"  hint  {node.id}: {rel(event_dir)} (from artifacts path)")
        # Fallback: fingerprint the run by its test/* metrics (builds the event
        # index lazily). Needed for older nodes with no output_dir recorded.
        if event_dir is None and metrics:
            event_dir, err, matched = match_event_dir(metrics, event_dirs_index())
            if event_dir is not None:
                print(f"  match {node.id}: {rel(event_dir)} ({matched} keys, {err*100:.2f}% mean err)")
        if event_dir is None:
            print(f"  skip {node.id}: no event dir matched its metrics or path hints")
            skipped += 1
            continue

        if args.dry_run:
            continue

        out_png = runs_dir / node.id / "curves.png"
        try:
            acc = _accumulator(event_dir)
            ok = plot_curves(acc, node.id, out_png)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {node.id}: plotting failed: {exc}", file=sys.stderr)
            skipped += 1
            continue
        if not ok:
            print(f"  skip {node.id}: no train/val curves in {rel(event_dir)}")
            skipped += 1
            continue
        update_artifacts(node.path, rel(out_png), rel_to_checkout(event_dir))
        print(f"  wrote {rel(out_png)}")
        written += 1

    print(f"done: {written} written, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
