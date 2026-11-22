# #2383 記録専用PRへの軽量CI — 実装指示

担当Issue: #2383（親Issue #2390 の正式な子Issue）。Issue本文の目的・スコープ・対象外・Acceptance checklist 5項目が受入条件の正本。全条件を満たすPRを1本作る（`Closes #2383`、本文は日本語）。

## スコープの境界（重要）
- **本タスク（#2383）の範囲**:
  1. PRの全Git差分（rename・削除含む）と比較元から「記録専用」かを判定する共通実装（CIと下流 #2386 の `parent-session` から同じ実装を呼べる形）。
  2. 記録専用PRに適用する軽量CI経路（`kg_validate.py --check-summary --base-ref <base>` と `kg_repro_paths.py`）。
  3. 軽量CI結果に検査したPRのHEAD・比較元（base）・判定結果・成否を機械可読に残す仕組みと、後続の専用マージ処理（#2386）がHEAD・比較元を照合して受入/拒否するための再利用可能な照合ヘルパー。
  4. `main` への push 時には通常のCPUテストと変更に応じたナレッジ検査を維持すること。
- **対象外（#2386 で実装する）**:
  - `parent-session` の正式レビュー候補選定・`review-request` 受付からの記録専用PR除外、専用マージ・マージ後確認、および `parent-session` の運用文書更新。
  - branch protection 等リポジトリ設定の変更（調査は `gh api` で読むだけ。変更が必要と判明したら親へ `needs_parent` で返す）。
- **切替え時の互換性**:
  - 既存の集約チェック名や通常PRでの `Python` チェックなど、現行の親CLI（`.agents/ops/parent/review_handoff.py` 等）や required checks が参照する契約を壊さないこと。

## 前提として調べること
- `.github/workflows/ci.yml`（`scope` / `plan` / `lint` / `tests` / `system-deps` / `python` 最終集約ジョブ）、**先例となる `.agents/ops/shared_memory/ci_scope.py`**（`ci.yml` の `scope` ジョブが Git diff から `shared_memory_only` を判定し後続ジョブを省略している仕組み）、`.github/workflows/knowledge.yml`（`knowledge` と `knowledge-webui` ジョブ）、`scripts/ci_metrics/`（`plan.py`・`lanes.py`・`lanes.json` 等）とそのREADME。
- `.agents/ops/parent/review_handoff.py` および `.agents/ops/ops-report/README.md` の専用メモリPR経路（下流 #2386 がどのように軽量CI結果・HEAD・baseを照合するかを踏まえて、過不足ない出力・照合APIを設計するため）。
- branch protection / required checks（`gh api` で読むだけ。設定変更はしない）。
- `knowledge/README.md` の保存規則とレジストリ契約（`deployments.yaml` 等）、`kg_validate.py` / `kg_repro_paths.py` の現行仕様と終了コード・出力。

## 設計上の要求
- **単一実装の共通判定**: 記録専用の判定は1か所に実装し、CI（workflow / `scripts/ci_metrics`）と `parent-session`（`.agents/ops/`）の両方から同じ実装をインポートまたは実行できるようにする（Python import パス・依存関係が `.agents/ops` 側からも最小依存で壊れない配置にすること）。
- **判定入力と安全なフォールバック禁止**: 入力は PR の全 Git 差分（rename の旧パス・新パス両方、削除含む）と比較元。比較元や差分が確定できなければ通常CIを選択し、静かなフォールバック禁止の規則に従って「なぜ通常CIになったか（選定理由・該当パス等）」を明示的な出力に残す。
- **記録専用の範囲**:
  - 対象: `knowledge/nodes/`、`knowledge/runs/`（run同梱の再現専用Python・シェル含む）、`knowledge/Papers/`、`knowledge/citation_map/`、`knowledge/reports/`、`knowledge/summary.md`、ノードの `.sequence`。
  - 対象外（通常CI）: Web UI、検査器、スキーマ、レジストリ（`deployments.yaml` 等）、保存規則（`knowledge/README.md` 等）、CI設定（`.github/workflows/` 等）、実装コードとの混在、記録専用範囲から対象外への rename（および対象外から範囲内への rename）、比較元や差分が確定できないイベント。
- **軽量経路のイベント限定**: 軽量経路は PR イベント（`pull_request`）のみ。`main` への `push` では通常のCPUテストと変更に応じたナレッジ検査を実行する。
- **軽量検査と照合可能な結果**:
  - `kg_validate.py --check-summary --base-ref <base>` と `kg_repro_paths.py` を実行し、構造・連番（マージ先との連番衝突、#2380 事例）・履歴・保存基準・参照を検証する。
  - 検査に使った PR の HEAD SHA・比較元（base SHA / base ref）・判定種別・成否を、GitHub Actions の check run 出力（summary/output/annotations や artifact 等、#2386 の親CLIから `gh api` 等で決定的に取得できる形）に機械可読な固定フォーマットで残し、それをパース・照合するヘルパー関数を提供する。HEAD または比較元のいずれかが一致しない場合は明確な理由付きで拒否する。
- **下流 #2386 への引き継ぎ情報**:
  - 実装した共通判定API・軽量CI結果の記録形式・照合APIのファイルパス・関数シグネチャ・契約をPR本文および最終報告に明記する。

## テスト・証拠（Acceptance checklist 5項目に対応させてPR本文に記載）
1. **判定のテスト**:
   - 記録専用（`knowledge/nodes/`, `runs/` 同梱の再現専用 `.py`[local path]`.sh`, `Papers/`, `citation_map/`, `reports/`, `summary.md`, `.sequence`）→軽量CI。
   - ソース混在、Web UI、レジストリ（`deployments.yaml`）、検査器、スキーマ、保存規則、CI設定、対象外へ移す rename（および逆方向 rename）、比較元不明・差分確定不能 → 通常CI（理由付き）。
   - 同一差分の PR イベントと `main` push イベントでジョブ選定を比較し、`main` push では通常CPUテストが選ばれること。
2. **専用チェックのテスト**:
   - 正常例、既存連番との衝突、不正な削除記録、再現スクリプトの参照切れで期待どおり成否を返すこと。
3. **HEAD・比較元照合のテスト**:
   - 固定した比較元と HEAD を使う正常例（受入）と、HEAD 違い・比較元違い・未成功・出力欠損の各例（拒否）。
4. **実イベントの証拠**:
   - 記録専用の代表PR（このPRとは別の小さな検証用draft PR、または同等の実GitHub Actionsイベント）で、通常のCPUマトリクス・システム依存・Lint・Web UIのジョブが省略されることと、実行したチェック・所要時間・記録されたHEAD/比較元を示す。検証用PRを使った場合は確認後にcloseし、その実行URLと結果を本PR本文に残す。
   - ワークフロー変更はPRブランチ上の workflow ファイルが `pull_request` イベントで評価されるため、検証用draft PRを本ブランチベースで作る等、実際に変更後ワークフローが動く形で検証すること。
- テストは `tests/README.md` の方針で設計し、`pytest` および `pre-commit`（ruff / mypy）を通す。

## 受け渡し
実装・ローカル検証・commit・push・PR作成まで行い、PR番号と完全なHEADで CI 待機を宣言して終了する（CI成功後に実イベント検証や証拠追記を行う場合は `resume_on_success: true` で後続作業を明記する）。マージ・正式レビュー依頼は親が行う。範囲や受入条件の解釈に迷う点、branch protection の変更が必要な点は `needs_parent` で返す。
