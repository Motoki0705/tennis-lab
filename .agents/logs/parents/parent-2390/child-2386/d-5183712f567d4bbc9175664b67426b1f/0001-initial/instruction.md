# #2386 parent-sessionでナレッジ記録専用PRを正式レビューから外して自動マージする — 実装指示

担当Issue: #2386（親Issue #2390 の正式な子Issue）。Issue本文の目的・変更内容・対象外・依存関係・Acceptance checklist 4項目が受入条件の正本。全条件を満たすPRを1本作る（`Closes #2386`、本文は日本語）。

## 先行依存とスタック構成（重要）
- 本タスクは #2383（PR #2395、branch `codex/parent-2390-issue-2383-primary`、HEAD `43381eadc7b6ac3d09b74249722a88d1ba270a0a`）を土台にスタックして実装する。
- PR作成時の **base branch は `codex/parent-2390-issue-2383-primary`** とする（#2383 のマージ後に `main` へ載せ替える）。
- #2383 の公開契約（このworktreeに同梱済み）をそのまま利用すること（判定ロジックやCI結果検証を二重実装しない）:
  - モジュール: `.agents/ops/lib/knowledge_ci.py`（`PYTHONPATH=.agents/ops` で `lib.knowledge_ci`）
  - 仕様: `.agents/ops/lib/README.md` の「ナレッジ記録専用PRのCI」節
  - `classify_pr(root: Path, *, head_sha: str, base_sha: str, merge_sha: str) -> Scope`
  - `artifact_name(run_id: int, run_attempt: int) -> str`
  - `verify_result(data: bytes | None, run: Mapping[str, object], *, pr_number: int, head_sha: str, base_sha: str) -> None`

## 前提として調べること
- `.agents/ops/parent/`（`cli_impl.py`, `cli_help.py`, `review_handoff.py`, `session.py`, `completion.py`, `policy.py`, `worker_monitor.py`, `worker_prompt.md`, `snapshot.py`, `work_targets.py`）
- `.agents/ops/review/`（`merge_gate_impl.py`, `merge_gate_policy.py`, `progression.py`, `progression_backend.py`, `post_merge.py`, `review_inspection.py`, `review_policy.py`）
- `.agents/ops/ops-report/README.md`（L586-655、専用メモリPRのマージ条件）およびその実装（`ops-report` のメモリマージ検証）
- `.agents/skills/parent-session/`（`SKILL.md`, `references/workers.md`, `references/review.md`, `references/finish.md`, `references/recovery.md`）
- `.agents/ops/lib/knowledge_ci.py` および `.agents/ops/lib/README.md`

## 設計上の要求
1. **作業者による事前のPR分離の文書化**:
   - `parent-session` の手順・ワーカー指示の正本（ドキュメント二重管理を避けて適切な正本ファイルに配置）に以下を明記する:
     - 独立したナレッジ記録は作業者が初めから記録専用PRとして作成すること（run同梱の再現専用Python・シェルを含む）。
     - スクリプト実行結果が受入条件の場合は、実行コマンドと結果の証拠をPR本文に添えること。
     - 記録だけが成果のIssueでは記録専用PRに `Closes #<issue>` を書き、実装に付随する記録PRでは対象Issue・実装PRへの参照を残し、Issueの完了は受入条件を満たすPRの反映に合わせること。
     - 混在PR、Web UI・検査器・レジストリ（`deployments.yaml` 等）・保存規則・CI設定の変更、本番チェックポイントと一体の採用根拠は記録専用にせず通常PR・通常レビュー経路とすること。
2. **正式レビュー候補の選定と `review-request` 受付からの除外**:
   - #2383 の `lib.knowledge_ci.classify_pr` を、正式レビュー候補の選定（`review-inspect` / 候補表示等）・`review-request` の受付・専用自動マージで共通利用する。
   - 記録専用PR（再現スクリプト含む）は正式レビュー候補から外し、`review-request` で直接指定された場合も正式レビューとして受け付けない（明確な理由と専用経路への案内を返す）。
   - 実装混在、Web UI・検査器・スキーマ・レジストリ変更、本番チェックポイント一体の変更、比較元や差分が確定できないPRは通常どおり正式レビュー対象とする。
3. **記録専用PRの専用待機・自動マージ経路**:
   - 専用メモリPRのマージ条件を参考にしつつ、`parent-session` の既存コマンド・progression 基盤（`merge-status` / `wait-merge` / `merge-continue` / `merge-confirm` 等）と一貫した形で、記録専用PRの待機・自動マージ・結果照合・再開を提供する。
   - 切替え時（既存進行中PRや通常PR）の親CLIのCI確認・レビュー進行を壊さないこと。正常完了レビュー1回固定方針など既存の正式レビュー契約を緩和しないこと。
   - 専用マージ実行前に以下をすべて厳密に照合する（静かなフォールバック禁止。いずれか不成立なら理由を残して保留または拒否）:
     - PRの最新差分が `lib.knowledge_ci.classify_pr` で記録専用と確定すること（途中で混在等に変わっていれば自動マージせず保留・拒否）。
     - PRの現在HEADと、軽量CIの検査対象HEADが一致すること（マージ実行時も `--match-head-commit` / `sha` 固定）。
     - マージ先 `main` の最新 tip SHA と、軽量CIの `base_sha` が一致すること。`main` が先行した場合は古い結果でマージせず、新しい比較元でのPRイベント検証が完了するまで保留する。
     - `lib.knowledge_ci.verify_result` で `Knowledge library` の `pull_request` run（最新attempt）と artifact `knowledge-records-<run>-<attempt>` 内の `knowledge-records.json` の明示的成功を確認すること（未完了・失敗・欠損・破損は保留/拒否）。
     - 親セッションの実行根拠（`authorization`、`AGENT_AUTO_MERGE` 等）および実行証拠（受入・実行結果の記録）。
     - PR・対象Issue・親セッションのいずれかに `needs-human` ラベルまたは停止指定がある場合はマージを保留し、保留理由が観測・照合できること。
4. **記録だけで閉じるIssueと、実装に付随する記録PRの扱い**:
   - 記録だけが成果のIssueでは、記録専用PR（`Closes #<issue>`）でIssueを閉じられるようにする。
   - 実装に付随する記録PR（`Closes` を持たず対象Issue・実装PRを参照する記録専用PR）も専用マージ経路で扱え、Issueの完了は受入条件を満たすPRの反映に合わせる（未閉鎖のIssueを誤って完了扱いにしたりエラー終了したりしない）。
5. **専用マージ後の反映と受入確認**:
   - マージ後は、取り込まれた `merge_commit` と PR の merged 状態を確認する。
   - 該当Issueの完了状態（記録だけで閉じるIssueなら `closed/completed`、実装付随PRなら関連Issueの現在状態を区別して記録）と、`main` の通常CPU CI（`Python`）を追跡し、親セッションの受入（`status` / `completion`）へ反映する。
   - 自動マージ実行の応答喪失（GitHub側では既にマージ済み）からの再確認・再開、および `main` CI 待機（pending → 完了確認）に対応する。
   - 親の `worker_monitor` 等が記録専用PRの軽量CI結果を確認する必要がある場合の整合も確認する。

## テスト・証拠（Acceptance checklist 4項目に対応させてPR本文に記載）
1. **正式レビュー除外（候補表示と `review-request` 直接指定）**:
   - 記録専用、再現スクリプトを含む記録専用 → 候補から除外・`review-request` 拒否。
   - 実装混在、Web UI変更（およびレジストリ・検査器・判定不能） → 候補に含まれ、`review-request` で正式レビューへ渡せる。
2. **専用マージの照合**:
   - 正常例（現HEAD・最新main・軽量CI成功・記録専用差分） → 専用経路でマージ成功。
   - CI未完了、CI失敗、PR HEADの変更、mainの先行、途中で差分が混在へ変わった場合 → マージせず理由付きで保留/拒否。
3. **`needs-human` と停止制御**:
   - PR、対象Issue、親セッションのそれぞれに `needs-human` または停止指定を置き、専用経路でマージが保留され保留理由が確認できること。
4. **専用マージ後の反映・受入確認**:
   - 記録だけで閉じるIssue（`closed/completed` + `main` 通常CPU CI 確認 → 受入反映）。
   - 実装に付随する記録PR（Issue未閉鎖を正しく区別しつつ、マージ反映・`main` CI 確認）。
   - マージ実行の応答喪失からの再確認・再開。
   - `main` CI 待機（pending → success）の追跡。
- テストは `tests/README.md` の方針で設計し、`pytest`（既存 `tests/unit/agents_ops` / `tests/e2e/agents_ops` 回帰含む）および `pre-commit`（ruff / mypy）を通す。ドキュメントの相対リンクテスト（`tests/e2e/agents_ops/test_agents_docs.py`）も通す。

## 受け渡し
実装・ローカル検証・commit・push・PR作成（base は `codex/parent-2390-issue-2383-primary`）まで行い、PR番号と完全なHEADで CI 待機を宣言して終了する（`resume_on_success: false`）。マージ・正式レビュー依頼は親が行う。範囲や受入条件の解釈に迷う点があれば `needs_parent` で返す。
