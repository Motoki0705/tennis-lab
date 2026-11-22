親Issue #2384 配下の子Issue #2358 の実装担当として、専用worktreeで実装・検証・PR作成まで行ってください。

## 対象Issueと目的
- 担当子Issue: #2358（親 #2384）
- 目的:
  1. (#2358) 一時的な接続障害（Codex backend 503 等）や起動直後の中断で有効なレビュー結果（`result.json`）が残らなかった正式レビューを、前回の終了・後始末・対象PR・HEAD・実行権限を照合したうえで親が CLI から明示的に再試行できるようにする（正常完了レビューの1回固定方針 #2255 は維持し、障害復旧時の例外条件・上限を運用文書に明記する）。
  2. (#2112 現行 v4 残余検証) 現行 v4 で `execution.json` あり・`result.json` なし・`agent-runtime/result.json` 完了済みの中断状態から `merge-continue` が AI 再起動 0 回で結果を採用・保存することを回帰テストで保証し、`execution.json` 作成直後〜`agent-runtime/result.json` 未作成で中断した場合も安全に再試行判定できるよう境界を整える。
  3. (`I-2011-review-cross-finding-outside-run` / #2145) 統合レビュー AI が対象外 PR に触れる横断指摘を出したときに `parse_output()` で run 全体を失敗させず、対象 PR 側の指摘として保存・公開し対象外 PR を参照として残せるようにする。

## 事前調査で特定した対象箇所（最新コードを再確認すること）
1. `.agents/ops/review/parent_review_impl.py` (`completed_result`, `run_review`):
   - L69-L95: `execution.json` があると `completed_result()` を呼ぶだけで、503 等の `runtime_error` / `usage_limit` / `startup_failed` や `agent-runtime/result.json` 未生成の終了から再試行する経路がない。
   - L99-L104: 他ラウンド `review-*` の `execution.json` があると無条件で拒否している。
   - 設計方針:
     - 通常の `review-request` 再送は従来どおり既存 run を返し、二重起動しない。
     - 通常の `merge-continue` も従来どおり、完了済みの `agent-runtime/result.json` があれば AI 再起動 0 回で回収し、有効な結果が既にある場合や終了未確認の場合は再起動しない。
     - 親の**明示的な再試行指示**（例: `merge-continue --retry-review` または専用フラグ）があり、かつ以下をすべて満たす場合のみ新しい review round（例: `review-2`、再試行上限は例として障害復旧再試行 2 回＝合計最大 3 試行までを定数・文書で明記）を起動する：
       - 有効な `result.json` が生成されていない（正常完了レビューは一切再起動不可）
       - 前回の reviewer の終了と後始末が照合確認済み（`runner.ended(...)` と cleanup 確認、または `execution.json` 作成直後にプロセス非存在・runtime未作成の確証がある場合。生存不明・未終了は厳格に拒否）
       - 対象 PR・現在 HEAD・実行権限（`authorization` / owner / generation）が照合済み
       - 再試行上限未満であること
       - 同時実行は `controller_lock` と durable state により排他され二重起動ゼロ
     - 旧試行の失敗理由（`turn_end_reason` / `error` / `round` 等）と再試行履歴を `state.json` / `merge-status` / `review_usage` から追跡できること。
     - 再試行も失敗した場合や現 HEAD の Python CI 未通過時に自動マージされない既存契約を厳格に維持すること。
2. `.agents/ops/review/parent_review_policy.py` (`parse_output`)・`result_schema.json`・`prompt_template.md`:
   - L36: `any(type(n) is not int or n not in numbers for n in prs)` により、対象外 PR 番号を 1 つでも含む major 指摘が出ると `ValueError` で run 全体が失敗する。
   - 修正方針: `prs` に含まれる番号のうち対象 PR（`numbers`）を `prs` として保持し、対象外 PR 番号があれば重複を除いて参照フィールド（例: `related_prs` 等、既存表示・`merge-continue` 判定と整合する形）へ分離・保存・PR コメント表示する。対象 PR が 1 件も含まれない指摘や型不正は引き続き拒否すること。`result_schema.json` / `prompt_template.md` との整合も確認すること。
3. `.agents/ops/parent/cli_impl.py`・`cli_help.py`・`.agents/ops/review/README.md`・`.agents/skills/parent-session/references/review.md`・`recovery.md`:
   - 明示再試行オプションの CLI 接続、利用条件・再試行上限の運用文書追記。

## 受入条件（#2358 の Acceptance checklist をすべて満たすこと）
1. 503 相当の失敗（および `execution.json` 直後の中断）を模擬したテストで、前回終了・後始末・対象 PR/HEAD・実行権限の照合後に明示再試行で reviewer が起動し、有効結果で通常マージ判定に進むことを確認する。
2. 通常 `review-request` 再送、明示再試行の同時実行、前回終了未確認、有効結果既存在、再試行上限超過で意図しない起動がゼロであることをテストする。
3. 旧試行の失敗理由と再試行履歴を照会でき、再試行失敗時や現 HEAD の CI 未通過時に自動マージされないことを確認する。
4. `execution.json` あり・`result.json` なし・`agent-runtime/result.json` 完了済みから `merge-continue` が AI 再起動 0 回で結果を採用・保存する回帰テストを追加する。
5. 対象 PR と対象外 PR が混在する横断指摘で `parse_output()`・公開・`merge-continue` が成功し、対象 PR を 1 件も含まない不正指摘は拒否されるテストを追加する。
6. 運用文書（`.agents/ops/review/README.md`・`review.md`・`recovery.md`）へ利用条件・上限を明記し、関連テスト・ruff・mypy が通る。

## 検証・PR作成・終了ルール
- 関連テスト（`tests/unit/agents_ops/`, `tests/e2e/agents_ops/` の review / merge_gate / progression 関連）＋新規テストを全通過させる。
- pre-commit（ruff / mypy）を通してコミット・push し、PR 本文に `Closes #2358`、変更概要、受入条件ごとの検証証拠、テスト結果を日本語で記載する。
- push・PR作成が終わったら、このターン内で CI 待ちループを行わず、最終メッセージ末尾に以下の形式で 1 行だけ出力して終了する（`<PR番号>` と `<40桁HEAD SHA>` は実際の値に置換）：
`PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":<PR番号>,"head":"<40桁HEAD SHA>","resume_on_success":false}}`
