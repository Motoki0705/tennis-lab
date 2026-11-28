"""Generate, list and verify the person-model comparison assets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import DecimalException
from pathlib import Path
from subprocess import TimeoutExpired
from tempfile import TemporaryDirectory
from urllib.error import URLError
from xml.etree.ElementTree import ParseError

from filelock import FileLock

from .markdown import snippet
from .pages import PAGES, Asset, asset_data, project
from .publication import prepare_pair
from .sources import DEFAULT_REVISION, ROOT, Sources, require
from .table_renderer import render_table
from .table_spec import Table
from .typst_runner import VERSION, resolve
from .verify import (
    VerifiedBundle,
    asset_paths,
    check_asset_directory,
    describe,
    file_digests,
    manifest_path,
    verify_bundle,
    verify_wiki_page,
)


@contextmanager
def _errors_for(page: str) -> Iterator[None]:
    try:
        yield
    except KeyError as error:
        raise ValueError(
            f"{page}: unknown source key or missing field: {error.args[0]}"
        ) from error


def render(
    output: Path, page: str, sources: Sources, assets: tuple[Asset, ...], typst: Path
) -> None:
    from matplotlib import __version__ as matplotlib_version
    from PIL import __version__ as pillow_version

    from .chart_renderer import render_chart

    directory = output / "assets" / page
    directory.mkdir(parents=True)
    for asset in assets:
        target = directory / asset_data(asset).slug
        if isinstance(asset, Table):
            render_table(asset, typst, target)
        else:
            render_chart(asset, target)
        prepare_pair(target.with_suffix(".svg"))
    (output / f"{page}.md").write_text(snippet(page, sources, assets), encoding="utf-8")
    manifest = describe(page, sources, assets)
    manifest["tools"] = {
        "typst": VERSION,
        "matplotlib": matplotlib_version,
        "pillow": pillow_version,
    }
    manifest["files"] = file_digests(output, page, assets)
    path = manifest_path(output, page)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _source(page: str, revision: str) -> tuple[Sources, tuple[Asset, ...]]:
    sources = Sources.from_git(ROOT, revision)
    return sources, project(page, sources)


def _generate(output: Path, pages: list[str], args: argparse.Namespace) -> None:
    typst = resolve(args.typst_bin, offline=args.offline)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".wiki-assets-", dir=output) as temporary:
        staging = Path(temporary)
        for page in pages:
            with _errors_for(page):
                sources, assets = _source(page, args.data_revision)
                check_asset_directory(output, page, asset_paths(page, assets))
                render(staging, page, sources, assets, typst)
                verify_bundle(staging, page, sources, assets)
        with FileLock(output / ".generate.lock", timeout=10):
            for page in pages:
                source_files = sorted((staging / "assets" / page).iterdir())
                source_files.extend(
                    (staging / f"{page}.md", manifest_path(staging, page))
                )
                targets = [output / path.relative_to(staging) for path in source_files]
                check_asset_directory(
                    output,
                    page,
                    {path.relative_to(staging) for path in source_files[:-2]},
                )
                for parent in {target.parent for target in targets}:
                    require(
                        parent.resolve().is_relative_to(output),
                        f"{page}: output directory escapes the bundle",
                    )
                    parent.mkdir(parents=True, exist_ok=True)
                # Replace complete files. The manifest is the last record of the new set.
                for source, target in zip(source_files, targets, strict=True):
                    source.replace(target)
                print(f"{page}: generated {len(source_files) - 2} images and Markdown")


def _verify(output: Path | None, pages: list[str], args: argparse.Namespace) -> None:
    bundles: dict[str, VerifiedBundle] | None = None
    if output is not None:
        bundles = {}
        typst = resolve(args.typst_bin, offline=args.offline) if args.rerender else None
        for page in pages:
            with _errors_for(page):
                sources, assets = _source(page, args.data_revision)
                bundle = verify_bundle(output, page, sources, assets)
                bundles[page] = bundle
                if typst is not None:
                    with TemporaryDirectory(prefix="wiki-assets-verify-") as temporary:
                        rerendered = Path(temporary)
                        render(rerendered, page, sources, assets, typst)
                        require(
                            bundle.files == file_digests(rerendered, page, assets),
                            f"{page}: re-rendered files differ from the bundle",
                        )
            print(
                f"{page}: verified {len(assets)} SVG/PNG pairs and source-aligned Markdown"
            )
    for path in args.wiki_page:
        verify_wiki_page(path, bundles)
        result = (
            " and matched copied blocks and images to the bundle"
            if bundles is not None
            else ""
        )
        print(
            f"{path.name}: verified Wiki Markdown and relative SVG/PNG assets{result}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-pages", help="List supported Wiki page slugs")
    for name, help_text in (
        ("generate", "Build a staging bundle"),
        ("verify", "Check a bundle or Wiki Markdown"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--page",
            action="append",
            choices=sorted(PAGES),
            help="Select one or more pages; default: all",
        )
        command.add_argument(
            "--output",
            type=Path,
            required=name == "generate",
            help="Root of the staging bundle",
        )
        command.add_argument(
            "--data-revision",
            default=DEFAULT_REVISION,
            help="Git revision of the published source files",
        )
        command.add_argument(
            "--typst-bin",
            type=Path,
            help=f"Path to an existing Typst {VERSION} executable",
        )
        command.add_argument(
            "--offline",
            action="store_true",
            help="Require local Typst and skip any download",
        )
        if name == "verify":
            command.add_argument(
                "--wiki-page",
                action="append",
                type=Path,
                default=[],
                help="Check a Wiki Markdown file",
            )
            command.add_argument(
                "--rerender",
                action="store_true",
                help="Regenerate outputs and compare the exact bytes",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "list-pages":
        print("\n".join(sorted(PAGES)))
        return 0
    output: Path | None = args.output.resolve() if args.output is not None else None
    pages = list(dict.fromkeys(args.page or PAGES))
    if (
        args.command == "verify"
        and output is None
        and (not args.wiki_page or args.page or args.rerender)
    ):
        parser.error(
            "verify needs --output for a generated bundle, or --wiki-page for standalone Markdown"
        )
    try:
        if output is not None:
            args.data_revision = Sources.from_git(ROOT, args.data_revision).revision
        if args.command == "generate":
            if output is None:
                parser.error("generate needs --output")
            _generate(output, pages, args)
        else:
            _verify(output, pages, args)
    except (
        OSError,
        ValueError,
        DecimalException,
        csv.Error,
        ParseError,
        tarfile.TarError,
        TimeoutExpired,
        URLError,
        UserWarning,
    ) as error:
        print(f"wiki-assets: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
