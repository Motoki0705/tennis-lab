# Development CLI

このディレクトリは、プロジェクト固有の開発コマンドを
[Scientific Python `spin`](https://github.com/scientific-python/spin) で提供する。
長い `ruff` / `mypy` / `pytest` コマンドと CI の実行条件をここへ集約し、
ローカルと GitHub Actions が同じ入口を使うことを目的とする。

## セットアップ

```bash
uv sync --locked
uv run spin setup
uv run spin
```

2 回目以降は仮想環境を有効化して `spin ...` と直接実行してもよい。
`spin setup` は lock 済み依存関係を同期し、pre-commit hook をインストールする。

synthetic scene pipeline を利用する場合は、NHT submodule の更新後または初回に
次も実行する。

```bash
spin setup-nht
```

このコマンドは親repositoryが固定した `third_party/nht` commitをcheckoutし、
NHTとそのgsplat runtimeをtennis-labの`.venv`とは独立した`uv tool`環境へ
editable installする。production SfMの任意retry backendも用意する場合は
`spin setup-nht --with-sfm-learned`を使用する。`uv tool`のbin directoryが
`PATH`にない場合は、コマンドが具体的な追加先を表示して失敗する。

## コマンド

| コマンド | 用途 |
|---|---|
| `spin doctor [--strict]` | Python、`.venv`、lockfile、主要 CLI、submodule を診断する。通常は任意機能の不足を警告し、`--strict` では警告も失敗にする |
| `spin setup-nht [--with-sfm-learned]` | NHT submoduleを固定commitへ更新し、`nht-reconstruct`と`nht-render`を独立した`uv tool`環境へインストールする |
| `spin lint [--fix] [PATHS...]` | `src/`, `tests/`, `.spin/` を Ruff で検査する。`--changed --base <ref>` で差分だけを検査できる |
| `spin typecheck [PATHS...]` | 既定では `origin/main` との差分だけを mypy で検査する。`--all` で全体を検査する |
| `spin test [PYTEST_ARGS...]` | `local_data` / `cuda` marker を除外して pytest を実行する。`--all`, `--coverage`, `--serial` を利用できる |
| `spin ci [--lane LANE]` | 全体 Ruff と環境非依存テストを実行する。GitHub Actions は `--lane` で分割実行する |
| `spin setup [--no-hooks]` | `uv sync --locked` と pre-commit hook の導入を行う |

例:

```bash
spin lint --changed
spin lint --fix src/utils
spin typecheck src/utils/geometry
spin test tests/unit/utils -q
spin test --all -m cuda
spin ci
spin ci --lane remainder
```

`spin ci` はローカルで全テストを実行する。GitHub Actions は同じ入口に
`--lane remainder|long-tail|scene-pipeline` を渡し、全テストファイルを重複・欠落なく
並列レーンへ分割する。lane ごとの worker 数や pytest 条件もこのコマンドが所有する。

`typecheck` が差分を既定とするのは、リポジトリ全体には段階的に解消中の
既存 mypy error があるためである。対象 ref が存在しない場合は暗黙に別の ref へ
切り替えず失敗するので、`git fetch origin main` または `--base <ref>` で明示する。

## PyTorch の `.spin` から採用しなかったもの

[PyTorch の `.spin/cmds.py`](https://github.com/pytorch/pytorch/blob/main/.spin/cmds.py)
を参考にしたが、次はこのリポジトリでは提供しない。

- `clean`: gitignore 対象に dataset、checkpoint、学習出力が含まれ、機械的な削除が危険。
- regenerate 系: 現時点では type stub や CI template などの正規の生成フローがない。
- docs build: 現時点では Sphinx 等の単一ビルド入口がない。
- tennis-lab本体の独自 build/install: 本体のPython環境とeditable installは
  `uv sync`を正規の入口とする。独立CLIであるNHTだけは`spin setup-nht`が所有する。

新しい横断的な開発コマンドは `.spin/cmds.py` に追加する。学習や推論のように
Hydra config が正規のインターフェースである処理は、各 task の `scripts/` と README
に残し、ここで設定を二重管理しない。
