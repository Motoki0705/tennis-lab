# #2176 レビュー〜mergeの自動進行への切り替え

担当Issue #2176 の本文（目的・変更内容・対象外・Acceptance checklist）を正本として実装してください。親は #2157 です。要件の出典は次の3つです。

- #2157 本文の「全体のスコープ」「全体方針・合意事項」(a)〜(h)・「全体の完了条件」1〜11（12は親が実PRで確認する）。
- #2157 の 2026-11-16 の決定コメント: https://github.com/Motoki0705/tennis-lab/issues/2157#issuecomment-6903903409
  （自動修正はPRごとに合算3回、WSL再起動後は親AIの明示操作で再開、CI待機は60秒間隔・上限6時間、worker起動前の needs-human 確認、確定できない副作用は `recovery_required`）
- 設計調査の報告（コード変更前の現状調査。行番号は調査時点の値）: `.agents/logs/parents/parent-2157/child-2157/d-6ea773efec904ab7a83c6b1d73b441ca/0001-initial/final.md`

## 既にmainへ入った土台（利用するだけ。作り直さない）

- #2174（PR #2177）: `GitHub.get_conditional()`（ETagによる条件付きGET、304再利用、rate limitの `resume_at`）、`GitHub.job_log(job_id)`（理由付きのログ取得）、GET数・200/304のmetrics。利用契約は `.agents/ops/issues/README.md`。ETagキャッシュはプロセス内メモリだけなので、再開後は200から始まる前提で設計する。
- #2175（PR #2178）: 作成workerの同じsession・モデルを修正ターンだけ effort=high で再開する仕組み。接続口は PR #2178 本文「#2176からの接続」と `.agents/ops/parent/README.md`:
  - 手動: `run-worker --resume --repair-id ID --repair-pr N --prompt-file FILE` / `manual_workers.run_worker(..., resume=True, repair=lib.repair_turn.request(ID, N))`
  - dispatcher: `dispatcher.repairs.register_progression(...)` と `Supervisor.start_repair(...)`
  - 報告の検証: `lib.repair_turn.parse_report(text, repair_id=ID, pr=N)`（outcome repaired / needs_parent）
  - 起動直前に親・子・PRの needs-human と登録の世代を再確認する（`parent/repair_admission.py`）。

## 実装上の要点

- 削除する厳密照合（diff SHA-256の固定、postcheck、公開順序とdigest、GitHubコメントとローカル記録の照合、resolveの4項目証拠と変更パス完全一致、complete-evidence、base取込みのファイル単位継承、merge前後の全体取り直し）は、旧操作・旧判定経路として残さない。旧記録は履歴表示のために読めるようにする（既存registry・過去記録を書き換えない）。
- 自動進行の状態はprivate stateに保存し、待機中にregistry lockを持たない。中断後は親AIの明示操作で保存段階から再開し、二重merge・二重worker起動をしない。確定できない副作用（merge・worker起動の応答喪失など）は推測で成功扱いにせず `recovery_required` で親AIへ返す。
- 軽いgateは「手元確認 → CI → 競合・draft・head → needs-human → 依存 → `gh pr merge --match-head-commit`」の順で、最初の不合格で止まる。依存未mergeはCLIが待つ。needs-human と AGENT_AUTO_MERGE=0 は親AIへ返す。
- CI失敗は毎回作成workerを修正ターンで起こす（自動rerunはしない）。gate停止もworkerへ通知し、自分で直すか親AIを呼ぶかはworkerが決める。CI修正・競合解消では再レビューしない。再レビューは親AIの続行操作で `--rereview` を指定したときだけ。
- 自動進行の経路からrepo全体のIssue一覧（`EvidenceIssues.find` など）を外し、GET回数を対象家族・PR数に比例させる。
- 親AIへの通知: 手動の親は `wait-worker` と同じ背景待機の形、dispatcher起動の親は既存の待機・再開へ載せる。
- ドキュメントは新しい流れだけを説明するよう更新し、同じ状態表を複数箇所に書かない（AGENTS.mdの二重管理禁止）。
- 静かなフォールバックを作らない。テストは tests/README.md の方針で設計してから書き、fake GitHub・fake agent・fake clock のe2eで完了条件ごとにテスト名を対応付ける。旧契約のテストは新契約のテストへ置き換え、削除した旧テストと対応する新テストの一覧を報告する。

## 参考: この親で観測した現行gateの負担

PR #2178 では、レビュー指摘の修正後に新headでmerge gateが `pr_unverified_head` / `unresolved_findings` / `review_missing` で止まり、指摘ごとに4項目の証拠と変更パス完全一致の `resolve` を親が手で記録する必要がありました。新しい流れではこの手順が不要になることを、テストで確認してください。

## 完了

規模が大きいので、途中で設計上の判断が要件から一意に決まらない点が出たら、推測で進めずに親へ質問してください。完成したらPRを作成（本文は日本語、`Closes #2176`、`References #2157`）し、完了条件と証拠の対応表を本文に載せてください。CIの完了は待たずに報告してよい。
