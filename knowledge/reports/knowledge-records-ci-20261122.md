# ナレッジ記録専用PRの軽量CI・専用マージ導入レポート（2026-11-22）

## 概要

親Issue #2390（子Issue #2383・#2386）で、実験・文献の記録だけを変更するPR（記録専用PR）に対する軽量CI経路と、`parent-session` での正式レビュー除外・専用自動マージ経路を導入した。本レポートはその実測値と導入契約を記録する。

## 導入前後の実測比較

| 項目 | 導入前（PR #2380 `65eeda06`） | 軽量CI実測（検証用ドラフト #2396 `96a35b7a`） |
| --- | --- | --- |
| `CI` ワークフロー | 15ジョブ・22分20秒（[run 45016518349](https://github.com/Motoki0705/tennis-lab/actions/runs/45016518349)） | `scope` ジョブのみ5秒で完了し、CPUプラン・CPUマトリクス・システム依存・Lint・`Python` 集約を省略（[run 45024770259](https://github.com/Motoki0705/tennis-lab/actions/runs/45024770259)） |
| `Knowledge library` | 3分22秒（[run 45016518347](https://github.com/Motoki0705/tennis-lab/actions/runs/45016518347)） | 42秒で完了（`kg_validate` 8秒、`kg_repro_paths` 2秒。Web UIと開発用pytestを省略、[run 45024770222](https://github.com/Motoki0705/tennis-lab/actions/runs/45024770222)） |
| 結果の照合 | 通常 `Python` チェック名のみ | `knowledge-records-<run>-<attempt>` artifact 内の `knowledge-records.json` に PR番号・HEAD・比較元・テスト用マージSHA・run/attempt・各検査outcomeを記録 |

## 導入した構成

1. **共通判定と軽量CI（#2383 / PR #2395, merge commit `785db4ac7`）**:
   - `.agents/ops/lib/knowledge_ci.py` の `classify_pr` が全履歴から PR 差分とテスト用マージ差分の両方を検査し、記録専用範囲（`knowledge/nodes/`、`knowledge/runs/`、`knowledge/Papers/`、`knowledge/citation_map/`、`knowledge/reports/`、`knowledge/summary.md` の通常ファイル）のみに閉じる場合だけ軽量CIを選択する。
   - Web UI・検査器・スキーマ・レジストリ・保存規則・CI設定・実装混在・範囲外rename・比較不能時は理由付きで通常CIへ送る。`main` push では通常のCPUテストを維持する。
   - `verify_result` が Actions REST API の run 情報と artifact JSON の PR番号・HEAD・比較元・テスト用マージSHA・run ID・attempt・両検査成功を照合する。
2. **正式レビュー除外と専用自動マージ（#2386 / PR #2407, merge commit `b5a6204b0`）**:
   - `review-inspect` 候補と `review-request` 受付から記録専用PRを除外し、混在変化時は未送信・停止確認後に正式レビューへ移管できる。
   - `merge-knowledge` が構造化受入証拠とPR本文 `## 実行証拠`・GitHub閉鎖対象・最新 `main`・最新軽量CI run/attempt・`needs-human`/停止指定を照合して自動マージし、マージ後の merge commit と `main` の通常CPU CI（`Python`）を `status.completion.knowledge_merges` へ反映する。

## 実イベント通貫検証

本レポート自体のPRを、導入後の `main` 上で「軽量CI → 正式レビュー除外 → `merge-knowledge` 専用自動マージ → マージ後の `main` 通常CPU CI」の実イベント検証として実行する。
