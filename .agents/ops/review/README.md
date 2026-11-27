# 正式レビューとマージの自動進行

親が使う操作は [レビューとマージ](../../skills/parent-session/references/review.md)、中断後の照合は [復旧](../../skills/parent-session/references/recovery.md) を参照。

## 実行と保存

- [merge_gate_impl.py](merge_gate_impl.py) はrepoとPR番号で予約を照合する。親の明示要求でcontrollerが起動し、親stateの `progressions/<run ID>/state.json` に対象HEAD・指摘・修正・通知を永続化する。
- [parent_review_impl.py](parent_review_impl.py) は起動意図を保存してread-only reviewerを呼ぶ。同じrunに保存された出力は、対象・指示・runnerの終了・回収を照合して採用する。PRコメントには保存結果の表示用コピーを投稿する。横断指摘の `prs` はレビュー対象内のPRを示す。対象外の番号は重複を除いて `related_prs` に保存し、対象PRへのコメントで参照を表示する。
- [progression.py](progression.py) がCI待ち、依存待ち、修正、マージ後確認を進める。[progression_backend.py](progression_backend.py) が権限、現在のHEAD、GitHub上の状態を照合する。完了判定はmerge commitのCIを使う。
- [knowledge_records.py](knowledge_records.py) は [記録専用CIの共通判定](../lib/README.md#ナレッジ記録専用prのci) から差分と結果を照合し、GitHubが認識する閉鎖対象を確認する。[knowledge_evidence.py](knowledge_evidence.py) が現HEADの受入確認とPR本文を照合し、[knowledge_progression.py](knowledge_progression.py) が専用マージ要求を保存する。
- workerログ・判断ログの公開はそれぞれの専用経路を使う。[worker_logs](../worker_logs/README.md)・[decision_logs](../decision_logs/README.ja.md)

観測間隔の間にcheckoutの版が変わった場合、controllerは保存済みの進行段階と中断理由を `parent_wait` に記録し、
終了コード75を返す。中断の通知は新しく起動したCLIから通常の通知経路で配送する。
`wait-merge`・`merge-status` と `controller.log` で理由を確認し、`merge-continue --run-id RUN --instruction-file FILE` で同じ進行を再開できる。
マージ後のCI待ちから再開する場合は、保存したmerge commitを照合する。

`review_capacity_deferred` で中断した場合は、容量回復後に `merge-continue` で同じレビュー回を再試行できる。
レビューCLI未起動の証拠とランナーの終了を照合したときに起動枠が戻る。各試行の記録は
`review_usage` で確認できる。`repair_capacity_deferred` も、同じ修正IDで再試行するための中断になる。

レビュー資料の取得範囲は [review_tools](../lib/review_tools/README.md)、過去事例での検証は [比較harness](evaluation/README.ja.md) を参照。

記録専用マージは、同じ親レジストリを使うcontrollerのマージ送信をリポジトリ単位で排他し、送信直前にmainのtipを照合する。GitHubの同期マージAPIではPRのHEADを固定できる。同時に外部操作でmainが進んだ場合は、観測されたマージコミットの両親を照合し、不一致を受入保留にする。 [GitHubの同期マージAPI](https://docs.github.com/en/rest/pulls/pulls#merge-a-pull-request)

## GitHub の再観測とマージ後 CI

- 通常 PR と記録専用 PR は、`mergeability_unknown` を 5、10、15、30、30、60 秒の間隔で最大6回再観測する。記録専用 PR の最終照合にも同じ上限を使う。再観測後も判定不能なら `mergeability_unknown` で親待ちに入り、`ci_wait` から再開できる。
- GitHub の HTTP 500・502・503・504 と接続障害は、CI・依存・マージ後の読み取りとレビュー起動前の照合で 10、20、40、60、60 秒の間隔で最大5回再観測する。正常な観測で回数をリセットし、再観測後も障害が続けばエラーを記録して親へ返す。`merge_pending` と `repair_pending` の過渡エラーは `recovery_required` で親へ返し、保存済みの実行意図を照合する。
- merge commit の最新の `.github/workflows/ci.yml` の push run がジョブ開始前にキャンセルされた場合は、run ID・試行番号・時刻を保存してから最大2回自動で再実行する。同じ試行のキャンセルが2分後も続く場合、API が失敗した場合、または再実行枠を使い切った場合は `main_ci_cancelled_before_start` で親へ返す。ジョブが作成済みで当該試行の `Python` が成功していなければ、`main_ci_failed` で親へ返す。 [GitHub の workflow 再実行 API](https://docs.github.com/en/rest/actions/workflow-runs#re-run-a-workflow)

## 正式レビューの障害復旧

有効な `approve` または `major` を得たrunの正式レビューは1回で確定する。`review-request` を再送すると既存runを返す。通常の `merge-continue` は元のラウンドを継続し、完了済みのrunner出力から保存結果を回収する。

障害時は `merge-continue --run-id RUN --retry-review ROUND --instruction-file recovery.md` で、`merge-status` の `review.round` にある失敗ラウンドを指定する。run単位で明示再試行は最大2回、初回を含めて最大3ラウンド。全対象PRと前回から同じHEAD、現在の担当・世代・実行根拠を照合して、次のラウンドを保存する。同じラウンドへの指示の再送は拒否する。

再試行はworktree作成後に権限とGitHub上のHEADを再照合する。起動意図の保存からランナー起動まで、実行根拠の取消しと同じ親操作ロックで直列化する。

ランナーの終了理由が `runtime_error`、`usage_limit`、`startup_failed` のいずれかで、プロセス終了・回収とレビュー用worktreeの回収を確認できた場合に再試行できる。最初のruntime結果を保存する前の中断も対象となる。ランナーが継承する起動ロックの解放、worktreeとGit登録の回収、AI起動前の初期化で生成されるファイルだけが残っていることを確認する。一時結果が残る場合はrunと指示への一致、ランナー終了とAI未起動も照合する。起動痕跡がある場合や照合できない場合は親待ちを維持する。保存済み結果または完了済みrunner出力があれば通常の `merge-continue` から継続する。CI・依存・停止・マージ後確認は通常どおり照合する。

`merge-status` の `review_history` に次のラウンドを予約した時点の失敗理由が残る。`review_usage` では同じラウンド内の容量不足も個別の起動試行として表示し、試行ごとの終了理由とAI起動を確認できた回数を確認できる。
