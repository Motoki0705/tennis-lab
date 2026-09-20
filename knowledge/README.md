# Knowledge Control — 研究ライブラリ

実験の結果・考察・関連研究をgitで共有する。**1 run = 1ノード**。実験群はgroupで束ね、前提・比較・反証の関係はIDで繋ぐ。
現在の判断は [summary.md](summary.md)、閲覧は [Web UI](webui/README.md)、登録・更新の手順は [knowledge-control skill](../.agents/skills/knowledge-control/SKILL.md) を起点にする。このREADMEが保存形式の正本。

## 保存構造

```text
knowledge/
  README.md
  summary.md                         # 横断的な研究判断。詳細の正本は各ノード
  nodes/
    plcs/.sequence                  # 過去に予約した最大番号（削除後も保持）
    plcs/000001-run-<slug>.md
    blcs/000001-run-<slug>.md
    synthetic_data_generation/...
    <task>/000002-group-<slug>.md
  Papers/
    paper-2024-gvhmr/
      paper.md                       # 書誌情報・読解・プロジェクトとの関係
      paper.pdf                      # 出典の版を固定したPDF
  runs/<run-id>/                     # 再現性bundle・予測・曲線（既存パスを維持）
  webui/
```

タスクは小文字snake_caseの**拡張可能なトピック**。`plcs` / `blcs` / `court_detection` / `ball_detection` に限定しない。合成データ生成、統合、独立した研究クラスタも追加できる。ノードは主目的となる1タスクに置き、タスク横断の関係はparents / relations / membersで表す。同じノード・論文を複製しない。issueは分類階層ではなく検索用メタデータ。

### ID・連番・ファイル名

- 不変ID: runは `run-<slug>`、groupは `group-<slug>`。slugは小文字英数字をハイフンで繋ぐ。新規は `run-i<issue>-<短い施策名>-s<seed>` など、差分がわかる名前にする。issueやseedが無い場合は省略し、架空の値を付けない。
- ファイル名: **`<6桁連番>-<id>.md`**。タスク内でrun/group共通の連番を1から付ける。IDと連番は別物で、関係参照は連番やファイル名ではなくIDを使う。
- `sequence` は登録順。新規登録は `.sequence` に記録された予約済み最大値+1、既存の番号は変更・再利用しない。`--force`も番号とタスクを維持する。削除や登録中断で欠番が生じても詰めない。`.sequence` はノード削除後・タスク廃止後もgitに残し、削除・縮小しない。欠落や現存ノードより小さい値は検証エラーにする。
- `date` は実験日（判明している場合のみ）、`recorded_at` は登録日。過去実験の後日登録は末尾に追加し、UIの時系列表示は `date` 優先で並べる。
- 既存200ノードの初回移行では実験日→ID順で採番。日付不明のノードはGit追加日を使用し `date_source: git_added` を記録した。判明しているものは `experiment_date`。同日内の実際の実行順は復元できないためIDで決定した。初回の `recorded_at` はこの移行時のソート基準日。
- 同じcheckout内の同時登録はディレクトリロックで直列化する。独立したworktree/branchはそれぞれ採番するため、マージ時の重複番号はvalidatorがエラーにする。**未マージ側の新規ノードだけ**をマージ先の最大値以降に採番し直し、`.sequence` は両branchの予約済み最大値と新規番号以上に合わせ、リンクとsummaryを更新する。ロックが別branchまで一意性を保証するとは扱わない。

## ノードのfrontmatter

```yaml
---
id: run-i900-geometry-s42
type: run
task: plcs
sequence: 103
recorded_at: '2026-09-20'
title: 幾何特徴を用いた位置推定
issue: 900                         # 整数または整数配列
provider: codex                    # claude / codex / gemini / human / other
date: '2026-09-20'                 # 実験日が不明なら省略
status: done                      # done / failed / running / planned
config: {model: multiview_axial_base, run.seed: 42}
metrics: {position_error_m: 0.38}   # 実測値のみ。split・単位を本文に記す
repro: {commit: '<sha>', command: '<exact command>'}
artifacts:
  run_dir: knowledge/runs/run-i900-geometry-s42
parents: []                        # baseline / 前提 → このノード
relations: []                      # [{to: run-..., rel: compares}]
papers: [paper-2024-gvhmr]         # 論文ID。本文で関係を説明
tags: []
---
```

`id` / `type` / `task` / `sequence` / `recorded_at` / `title` が必須。
`session`、`repro.branch` / `remote`、`artifacts.predictions` / `curves` / `log` / `output_dir` / `tb_logdir` は利用可能な根拠を記す。
`parents`, `members`, `tags`, `papers` は文字列配列。`relations` は `to` と `rel` を持つmappingの配列。

- **group**: `type: group`、`id: group-...` とし、`members` に既存run/groupのIDを列挙する。本文は群の結論を書く。
- **parents**: baseline / 前提から子への有向関係。
- **relations**: 非階層の有向関係。`compares` / `confirms` / `contradicts` / `supersedes` など。
- **papers**: 関連研究の出典。引用だけで再現・実証したとはみなさない。背景、実装採用、比較対象、仮説のどれかを本文に説明する。

保存時のスキーマ、ID・連番重複、関係先、bundle実在、論文参照・PDF hashを `kg_validate.py` で検証する。
`runs/` の再現コマンド・patch・予測は歴史的証拠なので、ノード移動に伴って書き換えない。実体は従来通り `runs/<id>` に保ち、巨大なcheckpointを追加しない。

## Papersの仕様

論文はタスクをまたぐため、`Papers/` に一元化する。UIは `tasks` によりタスク別に絞り込む。
ディレクトリIDは `paper-<発表年4桁>-<短いkebab-case名>`。同名論文は著者名などで区別し、版を追加する場合は別IDに `-v2` などを付ける。ファイル名は `paper.md` と `paper.pdf` に統一する。

`paper.md` の必須frontmatter:

| キー | 内容 |
|---|---|
| `id` / `type` | ディレクトリID / `paper` |
| `title` / `year` / `authors` | 正式題名 / 発表年 / 著者名の配列 |
| `tasks` | 関係するタスクの配列（未実験のタスクも可） |
| `source` | 保存した版の一次資料URL |
| `license` | PDFの利用条件を確認できるURL |
| `pdf` | `paper.pdf` |
| `sha256` | 保存PDFのSHA-256。登録スクリプトで計算 |

本文には研究の要点、プロジェクトとの関係、検証仮説と適用限界を記す。参照元実験の一覧はノードの `papers` からUIが生成するので手で二重管理しない。
公開PRにPDFを含める際は転載条件を確認し、著者・原論文・ライセンス・改変の有無を併記する。PDF固有のライセンスはrepo本体のMITとは別に維持する。

## summaryの継続更新

`summary.md` は自動生成したノード一覧ではなく、現在の判断・根拠・未解決課題・次の実験を要約する。正確な更新手順はskillに集約する。
`kg_summary.py` はノード本文・metadataと論文metadataの指紋で未レビューの変更を検出する。`--mark-reviewed` は**内容を見直したという記録**であり、考察を自動生成したり、その正しさを証明したりしない。

## 移行・検証

既存のflat形式は `kg_migrate.py`（dry-run、適用は `--write`）で移行した。不明な分類は `--task-map <JSON>` で明示する。混在状態はエラー、移行済みの再実行は無変更。保守対象Markdownの相対リンクを移動先に合わせて更新する。初回移行後もID、metrics、config、graph関係、再現bundleの内容を保持する。

```bash
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_validate.py --check-summary
```
