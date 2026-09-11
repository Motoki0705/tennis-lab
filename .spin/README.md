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
| `spin ci [--shard N --shards TOTAL]` | 通常CI対象のテストを実行し、JUnit・実行時間・割当表を保存する。全体実行または先頭shardでRuffも実行する |
| `spin ci-update-durations REPORTS_DIR` | 成功した全shardの計測結果から、git管理する分割用の時間データを更新する |
| `spin setup [--no-hooks]` | `uv sync --locked` と pre-commit hook の導入を行う |

例:

```bash
spin lint --changed
spin lint --fix src/utils
spin typecheck src/utils/geometry
spin test tests/unit/utils -q
spin test --all -m cuda
spin ci
spin ci --shard 1 --shards 4
```

## CIの実行・拡張

`spin ci` は `local_data` / `cuda` を除外した通常CI対象を実行する。
長時間のscene pipeline統合テストとコートデータセット生成の統合テストは
CI対象外とし、必要時に次のコマンドで実行する。

```bash
spin test --serial tests/integration/synthetic_data_generation/test_scene_pipeline_cpu.py
spin test --serial tests/integration/synthetic_data_generation/test_court_dataset.py
```

通常CIはCPU版PyTorchを使用する。GPU版が既定の開発・学習環境と同じ`uv.lock`で
管理し、排他的なdependency groupで選ぶ。CPU環境を再現する場合は次を使う。

```bash
uv sync --locked --no-group gpu --group cpu
.venv/bin/python -m spin ci --shard 1 --shards 4
```

CPU環境を同期した後に追加のgroup指定なしで`uv run`を使うと、既定のGPU環境へ
同期し直される。そのためActionsでは同期後に`.venv/bin/python`を直接使用する。
GPU環境へ戻す場合は`uv sync --locked`を実行する。

分割は `src/automation/ci/` が所有する。`.spin/ci-durations.json` のファイル別
実測時間を使い、長いファイルから合計時間が最小のshardへ割り当てる。同じrevision・
時間データ・shard数からは同じ割当になる。新しいテストは自動検出し、未計測ファイルは
明示的に1秒と見積もる。未計測ファイル一覧とCI除外理由は割当表に記録する。
存在しない除外ファイル、壊れた時間データ、不正なshard指定はエラーにする。
`--list-tests`で実行前に対象を確認できる。

GitHub Actionsの`matrix.shard`が並列数の唯一の設定場所である。例えば
`[1, 2, 3, 4, 5, 6]`へ変えると6分割になり、各jobは`strategy.job-total`を参照する。
各shard内ではpytest-xdistの`worksteal`を使用し、Actionsでは2 worker、
数値計算ライブラリは各1 threadで実行する。ファイル単位の分割のため、
単一ファイルが長くなった場合は独立したテストファイルへの分割を検討する。

計測結果は`artifacts/ci/shard-N/`へ保存する。`junit.xml`、`metrics.json`、
`plan.json`、`summary.md`を失敗時もartifactとして保存し、GitHubのjob summaryに
pytestの経過時間と遅いファイルを表示する。依存関係・OSパッケージの準備時間は
独立したActions stepで確認する。ファイル別の時間はsetup/call/teardownの合計であり、
並列実行の経過時間とは異なる。

成功したrunの`ci-shard-*` artifactsをすべて同じディレクトリへダウンロードし、
次で時間データを更新して変更をレビュー・コミットする。

```bash
gh run download <RUN_ID> --pattern 'ci-shard-*' --dir /tmp/ci-timings
spin ci-update-durations /tmp/ci-timings
```

更新処理は同じrevisionの全shardが成功し、現在のCI対象と重複・欠落なく一致する場合
だけ受け付ける。新規ファイル追加やテストの構成変更後に更新することで偏りを修正できる。
PRごとに時間データを書き換える処理は行わず、割当の再現性を保つ。

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
