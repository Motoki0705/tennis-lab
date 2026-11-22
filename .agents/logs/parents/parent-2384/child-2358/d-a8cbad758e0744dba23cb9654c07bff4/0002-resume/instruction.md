親Issue #2384 配下の子Issue #2358（PR #2397、run `parent-2384-b8d6be8c136445c68df2474548aace78`）の正式レビューで検出された major 指摘 2 件をすべて修正し、検証・commit・push まで行ってください。

## 修正対象の major 指摘 2 件（すべて採用・修正必須）

### Major 1: 再試行の権限・HEAD照合後に起動までの競合が残る
- 指摘詳細:
  `parent_review_impl.py:410-429` で全 PR の HEAD と実行権限を確認した後、起動意図の保存と全 PR の worktree 作成を経てランナーを起動する。この間に `basis-revoke` は別の操作ロックで完了でき、対象 PR への push も可能。`controller_lock` はこれらの操作を排他しないため、根拠が取り消された後や HEAD が変わった後にも再試行が古い照合結果で起動する。
- 修正要件:
  - worktree 作成後の起動境界（ランナー起動直前）で対象 PR の HEAD を再照合し、かつ親の権限変更（`basis-revoke` 等が取る `store.operation(...)` / registry 照合）と起動意図確定・起動境界を同じ排他で直列化して、権限取消しや HEAD 変更後に再試行が起動しないようにする。
  - worktree 作成中・precheck 後〜起動直前のタイミングで権限変更（`basis-revoke`）や PR HEAD 変更が起きた場合に再試行が確実に拒否され、ランナーが起動しない回帰テストを追加する。

### Major 2: runtime作成から最初の結果保存までの中断を復旧できない
- 指摘詳細:
  共通ランナー（`agent_runtime_impl.py:469-587`）は `agent-runtime/` を作成し、初期ファイルを書いてから最初の `result.json` を永続化し、その後に AI を起動する。この区間でランナーが停止すると、起動ロックが解放され worktree と Git 登録を回収しても、`retry_failure()`（`parent_review_impl.py:137-171`）は `agent-runtime/` ディレクトリの存在だけで再試行を拒否する。通常の `merge-continue` も `result.json` を要求するため、その run は親待ちから復帰できない。
- 修正要件:
  - 起動ロックの解放・worktree と Git 登録の回収・プロセス非存在に加えて、`agent-runtime/` の初期化記録（例: `result.json` 未作成かつ AI 起動記録[local path]`events.jsonl`[local path]`pid` 等が存在せず AI 未起動を確実に立証できる状態）を検証することで、その区間での中断を安全に再試行可能にする（逆に AI 起動済み・生存不明・不完全な runtime は従来どおり厳格に拒否する）。
  - `agent-runtime/` 作成後かつ最初の `result.json` 保存前の中断を再現する回帰テスト（明示再試行で復旧できるケースと、AI 起動痕跡ありで拒否されるケースの対）を追加する。

## 検証・push・終了ルール
- 関連テスト（新規回帰テスト＋既存の review / merge_gate / progression / parent_session 関連テスト）を実行し全通過を確認する。
- pre-commit（ruff・mypy）を通して commit・push し、PR #2397 本文（`Closes #2358` を維持）に major 2件ごとの修正 commit と検証結果を追記する。
- push が終わったら、このターン内で CI 待ちループを行わず、最終メッセージ末尾に以下の形式で 1 行だけ出力して終了する（`<40桁HEAD SHA>` は push した実際の値に置換）：
`PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2397,"head":"<40桁HEAD SHA>","resume_on_success":false}}`
