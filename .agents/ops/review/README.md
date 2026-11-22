# 正式レビューとマージの自動進行

親が使う操作は [レビューとマージ](../../skills/parent-session/references/review.md)、中断後の照合は [復旧](../../skills/parent-session/references/recovery.md) を参照。

## 実行と保存

- [merge_gate_impl.py](merge_gate_impl.py) はrepoとPR番号で予約を照合する。親の明示要求でcontrollerが起動し、親stateの `progressions/<run ID>/state.json` に対象HEAD・指摘・修正・通知を永続化する。
- [parent_review_impl.py](parent_review_impl.py) は起動意図を保存してread-only reviewerを呼ぶ。同じrunに保存された出力は、対象・指示・runnerの終了・回収を照合して採用する。PRコメントは表示用のコピー。
- [progression.py](progression.py) がCI待ち、依存待ち、修正、マージ後確認を進める。[progression_backend.py](progression_backend.py) が権限、現在のHEAD、GitHub上の状態を照合する。完了判定はmerge commitのCIを使う。
- [knowledge_records.py](knowledge_records.py) は [記録専用CIの共通判定](../lib/README.md#ナレッジ記録専用prのci) から差分と結果を照合し、GitHubが認識する閉鎖対象を確認する。[knowledge_evidence.py](knowledge_evidence.py) が現HEADの受入確認とPR本文を照合し、[knowledge_progression.py](knowledge_progression.py) が専用マージ要求を保存する。
- workerログ・判断ログの公開はそれぞれの専用経路を使う。[worker_logs](../worker_logs/README.md)・[decision_logs](../decision_logs/README.ja.md)

`review_capacity_deferred` で中断した場合は、容量回復後に `merge-continue` で同じレビュー回を再試行できる。
レビューCLI未起動の証拠とランナーの終了を照合したときに起動枠が戻る。各試行の記録は
`review_usage` で確認できる。`repair_capacity_deferred` も、同じ修正IDで再試行するための中断になる。

レビュー資料の取得範囲は [review_tools](../lib/review_tools/README.md)、過去事例での検証は [比較harness](evaluation/README.ja.md) を参照。

記録専用マージは、同じ親レジストリを使うcontrollerのマージ送信をリポジトリ単位で排他し、送信直前にmainのtipを照合する。GitHubの同期マージAPIではPRのHEADを固定できる。同時に外部操作でmainが進んだ場合は、観測されたマージコミットの両親を照合し、不一致を受入保留にする。 [GitHubの同期マージAPI](https://docs.github.com/en/rest/pulls/pulls#merge-a-pull-request)
