# #2487 edgecrafter: 親が診断 job 2件を入れ直した（中身は同じ）

ユーザー判断（2026-11-25 21:50 JST）で、提出用の Blender レンダリング4件に GPU の順番を譲った。そのため親があなたを一時停止し、未開始だった診断 job 2件を取り消した。同じコマンド・同じ作業ディレクトリ・`resource all` で、queue の後ろに入れ直した。障害ではない。

| 元の job | 入れ直した job |
|---|---|
| `i2487-ecdet-l-o365-trt16-diagnosis-v2` | `0-8204388989148236245_2026-11-25T12-50-10-851763754Z_3908487_i2487-ecdet-l-o365-trt16-diagnosis-v2-requeued.job` |
| `i2487-ecdet-l-coco-trt16-intermediate-probe` | `0-8204388988887258277_2026-11-25T12-50-11-112741722Z_3908559_i2487-ecdet-l-coco-trt16-intermediate-probe-requeued.job` |

取り消した元の2件は実行されていない（出力なし）。記録に job ID を書くときは、入れ直した方を使う。

このターンでは作業をしない。コードと worktree を変えずに、次の1行だけを返して待機する。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204388989148236245_2026-11-25T12-50-10-851763754Z_3908487_i2487-ecdet-l-o365-trt16-diagnosis-v2-requeued.job","0-8204388988887258277_2026-11-25T12-50-11-112741722Z_3908559_i2487-ecdet-l-coco-trt16-intermediate-probe-requeued.job"]}}
