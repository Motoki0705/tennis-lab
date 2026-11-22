# PR #2407 正式レビュー major 指摘（4件）の修正指示

正式レビュー（run `parent-2390-bc4a9a572caa46a29f0a6768512a14ae`）で以下の major 指摘 4 件があり、親として **全4件を採用** します。同じブランチ（`codex/parent-2390-issue-2386-primary`）で全4件を修正・回帰テスト追加・文書[local path] まで行い、新HEADで CI 待機を宣言して終了してください（`resume_on_success: false`）。

## 採用した major 指摘（4件）と修正要求

### 1. 現HEADの受入・実行証拠を確認できないまま専用マージへ進める
- **指摘**:
  - `knowledge_progression.py:25` の初回登録は証拠に空でない文字列を要求するだけで、PR番号や固定HEADとの対応を確認していない。
  - `:168` の続行処理では、新HEADと「実行は未着手。結果は未確認」と書いた指示でも `evidence_head` が更新されてしまう。
  - 実行結果が受入条件のPRでも、構造と参照だけを検査する軽量CIが成功すればそのHEADをマージできてしまう。
- **修正要求**:
  - 初回登録（`merge-knowledge`）および続行（`merge-continue`）の両方で、対象PRごとに **PR番号・固定HEAD・受入/実行確認の対応** を厳密に照合すること（単なる非空文字列や、未確認・未着手・未完了の申告、HEAD未一致の証拠で登録・`evidence_head` 更新を許さないこと）。
  - 特に run 同梱の再現スクリプト（`knowledge/runs/` 配下の `.py`[local path]`.sh` 等）を含むPRや実行結果が受入条件のPRについて、PR本文に実行コマンドと実行結果の証拠が含まれること（および親側の証拠が固定HEADに対して確認済みであること）を初回登録・続行の両方で確認し、不足・未確認なら明確な理由付きで拒否・保留すること。
  - 必要なCLI契約・PR本文フォーマット・親手順（`parent-session` の `review.md` / `worker_prompt.md` / `cli_help.py` 等）を明確に整合させ、回帰テスト（PR番号[local path]、未着手・未確認の証拠、実行証拠欠落、正常な確認済み証拠）を追加すること。

### 2. 記録専用で登録したPRが混在差分に変わると正式レビューへ移れない
- **指摘**:
  - 混在へ変わったPRは `review-inspect` の候補に入るが、`review_handoff.py:82` は現差分を判定する前に既存登録を照合し、`merge_gate_impl.py:82` が経路の違いを理由に正式レビュー登録を拒否する。
  - 専用runの続行も混在差分を拒否し、保存済みのPR予約を解除・移管する操作がないためデッドロックする。
- **修正要求**:
  - 記録専用runで管理中にPRが混在差分（通常レビュー対象）へ変わり保留された場合、**マージ未送信（および未マージ・controller未稼働）を確認したうえで**、その専用run（または対象PR予約）を安全に終結・移行し、同じPRを通常の正式レビュー（`review-request`）へ移せるようにすること。
  - すでにマージ送信済み・マージ済み・稼働中の専用runは誤って解除・上書きされないことを保証すること。
  - 回帰テスト（専用登録 → 混在差分へ変化して保留 → `review-inspect` 候補化 → 専用run終結/移管して `review-request` 成功、およびマージ送信済みは移管拒否）を追加すること。

### 3. マージ送信前に軽量CIの最新attemptが変わっても旧成功で送信する
- **指摘**:
  - `progression_backend.py:529` で成功attemptを再取得した後、対象Issue・親セッションの停止状態や実行根拠を照合する間に次のattemptが `queued` 等へ変わるケースで、PRとmainが同じままだと `:550` のマージ要求が送信されてしまう。
- **修正要求**:
  - 実行根拠・停止状態・閉鎖対象などの照合をすべて完了した後の **マージ送信直前の最終確認** に、最新の `Knowledge library` run ID・`run_attempt`・状態（およびHEAD・base）の再照合を含め、run/attemptの変更や未完了化（`queued` / `in_progress` / 失敗等）を観測した場合はマージ要求を送信せず待機（`ci_wait` 等）へ戻すこと。
  - 回帰テスト（初回確認は成功attempt 1 → 停止状態・根拠照合後に最新attempt 2が `queued` へ変化 → マージ送信0件で待機へ戻る）を追加すること。

### 4. GitHubが実際に閉じるIssueの一部が停止照合から漏れる
- **指摘**:
  - `knowledge_progression.py:53` と `progression_backend.py:185` はPR本文を `closing_issues` で解析して対象を固定しているが、`Closes #11` と `Closes owner/repo#10` を併記すると `#11` しか返らない等、GitHubが認識する修飾付きの閉鎖指定が漏れる。
  - そのため対象外Issueや `needs-human` 付きIssueが照合されないままマージでき、付随PRでも意図しないIssue閉鎖が起き得る。
- **修正要求**:
  - 登録時（`merge-knowledge`）およびマージ送信前の両方で、GitHubが実際に認識する閉鎖対象（GraphQL `closingIssuesReferences` の取得と、`owner/repo#N`・完全URL・全キーワードを含む本文解析の整合確認／他repo閉鎖や未解決参照の拒否を含む）を確実に照合すること。
  - 対象外の閉鎖指定、修飾付き閉鎖指定先の `needs-human` / 停止状態、付随PRにおける意図しない閉鎖指定（`Closes owner/repo#N` 等）を確実に検出・保留／拒否すること。
  - 回帰テスト（`Closes #11` + `Closes owner/repo#10` 併記で対象外・`needs-human` 捕捉、付随PRでの修飾付き閉鎖指定拒否、GraphQL `closingIssuesReferences` にのみ現れる閉鎖対象の捕捉）を追加すること。

## 検証・受け渡し
- 上記4件すべての回帰テストを追加し、対象テスト一式・`tests/e2e/agents_ops/test_agents_docs.py`・`pre-commit`（ruff / mypy / diff check）を通過させること。
- PR #2407 本文の検証・契約記述を最新仕様に同期し、commit・push 後、新HEADの完全SHAと各指摘の修正内容・回帰テスト結果を報告して CI 待機（`resume_on_success: false`）を宣言すること。
