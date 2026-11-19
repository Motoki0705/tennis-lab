担当は #2142（親 #1893）。main を取り込んだPRで `parent_review.py resolve` が GitHub のコメント上限超過で失敗する問題を直す。issue本文の受入条件をすべて満たす。

## 実例（読み取りに使ってよい）

- PR #2085（#2081）の正式レビュー run `parent-1893-269d4c5ae7964e529b16ef1a83efcd0f`（review head cd62f0aa8724ed48ea60ea81676202fcb770ac93）。指摘 pr-2085-1・pr-2085-2 を worker が修正し、main 0ff897fc5 を merge した fixed head 2ced17f646c380942db6f2996b58f02698a9e1fd。
- 親が `parent_review.py resolve --run <上のrun> --pr 2085 --finding pr-2085-1 --fixed-head 2ced17f6... --evidence <4キーJSON>` を実行し、`reviewed_paths` に review head→fixed head の全変更パス（1,532件）を入れたところ、`comment_body` が「review result exceeds the GitHub comment limit; 対象の縮約／親判断待ち」で拒否した。証拠JSONは `[local path]`、出力は `resolve-pr-2085-1.out`。review run の状態は `[local path]`。

## 設計の要点（判断はworkerに任せるが、PR本文で理由を説明する）

- 親の確認対象は「PR自身の変更の差」に限る。考え方の例: review head 時点のPR差分（merge-base→review head）と fixed head 時点のPR差分（merge-base→fixed head）をファイル単位で比べ、差が出たファイル（衝突解消・修正・renameを含む）を確認対象とし、PRの差分に現れない base 側だけの変更は除く。`review_workspace.verify_rebased_snapshot` の考え方を再利用できるか検討する。base とPRの両方が変えたファイルは確認対象に残す。
- resolve と merge gate（`merge_gate_policy` の changed_paths 照合、`parent_review_policy`）が同じ定義を使うこと。既存の公開済み resolution（旧定義）の扱いを決めて書く。
- 公開コメントにはパス一覧全体を載せず、件数・digest・確認対象パス（上限内）を載せ、全量はローカル記録に残す。上限を超える場合は静かに切り詰めず明示的に失敗する。
- 静かなフォールバック禁止。

## 制約

- 実際の PR #2085 への resolve の公開、merge gate の実行、systemd unit 操作、git-sync のONはしない（merge後に親が行う）。テストは一時repo・fake GitHub で行う。読み取りだけなら実run の state を参照してよい（書き換えない）。
- メインcheckout（`[local path]`）と他のworktreeを変更しない。PR #2085 も `.agents/ops/review/` を大きく変える（入口を薄いbootstrapにし本体を `*_impl.py` へ移す）。#2085 が先に main に入ったら追従する。
- GitHub API は共有上限（5000回/時）が逼迫している。CI の確認は数分おきの必要最小限にする。
- pytest の並列は `-n 4` 以下。修正を外すと新しいテストが失敗することを確かめ、関連テストと pre-commit を通し、PR（本文は日本語、「Closes #2142」「親: #1893」）を作って対象HEADのCIを確認する。force push しない。
- 作業時間の目安は最大5時間。止めてよいのは、受入条件を満たしPRのCIが成功した場合か、親の判断なしには進めない矛盾を見つけた場合だけ。
