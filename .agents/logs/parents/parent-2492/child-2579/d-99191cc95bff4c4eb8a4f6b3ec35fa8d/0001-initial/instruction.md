# 子 #2579: 人物モデル速度の記録2件の queue ログを保全する

あなたは親 #2492 の子 #2579 の担当。issue 本文の受入条件が正。小さな記録の修正なので、手早く記録専用 PR にする。

## 親が確認済みの事実
- main の `knowledge/runs/run-person-model-speed-baseline/queue-receipts.json`（prepare・detection・body の3件）と `knowledge/runs/run-person-model-speed-low-precision/queue-receipts.json`（detection・body の2件）が、`.training_queue/logs/<job>.log` を参照している。
- 5件とも、現時点で元のログ（repo root の `.training_queue/logs/`）と親の複製（`.worktree-outputs/parent-2492-queue-log-archive/logs/`）の両方が、記録の `queue_log_sha256` と一致する。`.training_queue/repro/<job>[local path]` のディレクトリはあるが、`queue.log` はまだない。
- `.training_queue/` は repo root（メイン worktree）のものを共有する。worktree の中に作らない。

## やること
1. 専用の worktree と branch で、最新の main から作業する。
2. 5件のログを `.training_queue/repro/<job>[local path]` に複製し、SHA-256 を照合する（既存のファイルは上書きしない）。
3. 2つの `queue-receipts.json` の `queue_log` を新しい場所に付け替える。同じ記録の README などに旧パスがあれば合わせて直す。値と判定は変えない。
4. training queue の実装で、`completion_record`（`.training_queue/control/completions/`）や `run_record`（`repro/`）など、記録が参照するほかのファイルに削除規則があるか確かめる。削除されるものは同じ方式で保全する。
5. knowledge 検査と pre-commit を通す。記録専用 PR を main に向けて作る。本文は日本語で、先頭に `Closes #2579`、`Refs #2492, #2496` を書き、`## 実行証拠` の直下に照合コマンド・終了値・結果を `json` のコードブロック1つで書く（`head_sha` は PR HEAD の完全 SHA）。draft にしない。

## 終わり方
最終回答に、保全したファイル、照合結果、ほかの参照の削除規則の調査結果、PR 番号と HEAD を書き、CI を待つ `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":<PR番号>,"head":"<HEAD>","resume_on_success":false}}` を返す。
