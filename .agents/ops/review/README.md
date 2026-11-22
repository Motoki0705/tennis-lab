# 正式レビューとマージの自動進行

親が使う操作は [レビューとマージ](../../skills/parent-session/references/review.md)、中断後の照合は [復旧](../../skills/parent-session/references/recovery.md) を参照。

## 実行と保存

- [merge_gate_impl.py](merge_gate_impl.py) はrepoとPR番号で予約を照合する。親の明示要求でcontrollerが起動し、親stateの `progressions/<run ID>/state.json` に対象HEAD・指摘・修正・通知を永続化する。
- [parent_review_impl.py](parent_review_impl.py) は起動意図を保存してread-only reviewerを呼ぶ。同じrunに保存された出力は、対象・指示・runnerの終了・回収を照合して採用する。PRコメントには保存結果の表示用コピーを投稿する。横断指摘の `prs` はレビュー対象内のPRを示す。対象外の番号は重複を除いて `related_prs` に保存し、対象PRへのコメントで参照を表示する。
- [progression.py](progression.py) がCI待ち、依存待ち、修正、マージ後確認を進める。[progression_backend.py](progression_backend.py) が権限、現在のHEAD、GitHub上の状態を照合する。完了判定はmerge commitのCIを使う。
- workerログ・判断ログの公開はそれぞれの専用経路を使う。[worker_logs](../worker_logs/README.md)・[decision_logs](../decision_logs/README.ja.md)

レビュー資料の取得範囲は [review_tools](../lib/review_tools/README.md)、過去事例での検証は [比較harness](evaluation/README.ja.md) を参照。

## 正式レビューの障害復旧

有効な `approve` または `major` を得たrunの正式レビューは1回で確定する。`review-request` を再送すると既存runを返す。通常の `merge-continue` は元のラウンドを継続し、完了済みのrunner出力から保存結果を回収する。

障害時は `merge-continue --run-id RUN --retry-review ROUND --instruction-file recovery.md` で、`merge-status` の `review.round` にある失敗ラウンドを指定する。run単位で障害復旧の再試行は最大2回、初回を含めて最大3試行。全対象PRと前回から同じHEAD、現在の担当・世代・実行根拠を照合して、次のラウンドを保存する。同じラウンドへの指示の再送は拒否する。

再試行はworktree作成後に権限とGitHub上のHEADを再照合する。起動意図の保存からランナー起動まで、実行根拠の取消しと同じ親操作ロックで直列化する。

ランナーの終了理由が `runtime_error`、`usage_limit`、`startup_failed` のいずれかで、プロセス終了・回収とレビュー用worktreeの回収を確認できた場合に再試行できる。ランナーが継承する起動ロックを取得でき、runtimeディレクトリが未作成でworktreeの残存もない中断も対象となる。終了を立証できる記録が不足している場合は親待ちを維持する。保存済み結果または完了済みrunner出力があれば通常の `merge-continue` から継続する。CI・依存・停止・マージ後確認は通常どおり照合する。

`merge-status` の `review_history` に次のラウンドを予約した時点の失敗理由が残り、`review_usage` では起動意図、起動を確認できた回数と各ラウンドの終了理由を確認できる。
