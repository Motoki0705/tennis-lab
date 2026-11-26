# #2491 再開: 2つの queue job の完了と、#2497 最新 head の取り込み

## 何が起きたか

- 待っていた2つの job は、どちらも queue 上は `done` で終わった。
  - `i2491-fasthmr-trt16-export`: ログ `.training_queue/logs/0-8204434740687790252_2026-11-25T00-07-39-312209747Z_2393387_i2491-fasthmr-trt16-export.log`。末尾に `torch_export` と `torchscript_export` の phase が `failed` と出ている（cuDNN の TF32 フラグ不一致、CameraHMR が Trainer に未接続）。そのあと `camera_backbone_export` の phase がある。どの経路で engine ができたか、できなかったかをログと出力で確かめる。
  - `i2491-fasthmr-pt16-visual-6clips`: ログ `.training_queue/logs/0-8204421172867557512_2026-11-25T03-53-47-132442487Z_31336_i2491-fasthmr-pt16-visual-6clips.log`。clip ごとの `status ok` 行と npz の sha256 が出ている。
- 自動再開は、管理ツールの検査（担当 branch が #2497 の現在の head を含むこと）で拒否された。#2497 の head が `13e508cb7`（「Remeasure PT16 and TRT16 with parity explicitly unevaluated」）に進んでいたため。
- 親が `13e508cb7` をあなたの branch にマージした（`3d8af2bad`、コンフリクトなし、push はしていない）。中身を確かめ、テストが通ることを確認してから、あなたが push する。

## 方針（前回の指示 pt32-note のまま）

- PT32 の計測はしない。PT16・TRT16 との parity 判定もしない。FastHMR の PT16・TRT16 は速度だけの参考値で、比較図から外す。8フレーム試行の関節差 P95 45.1 mm は記録として残す。
- 失敗を隠さない。TRT16 の engine ができなかった経路は「実行不能」として理由と一緒に記録する。黙ってほかの経路に切り替えない。

## 次にやること

1. TRT16 export の結果を判定して記録する。engine ができていれば、どの経路でできたか、入力 profile と sha256 を残す。できていなければ「実行不能」と理由を残し、直せる見込みがあれば根拠付きで書く（再変換を勝手に何度も回さない）。
2. PT16 の6 clip 結果と保存済み GVHMR の出力で、ユーザーが目視確認する同期比較動画を作る（GVHMR を再実行しない）。必要なら CPU で作ってよい。
3. 速度計測（PT16、TRT16 は engine があれば）を、取り込んだ #2497 の harness（`13e508cb7` 時点）で queue に入れる。parity は「未判定（PT32計測はスコープ外、ユーザー判断 2026-11-25）」の明示モードを使う。記録に harness の commit を残す。#2497 がマージ前に計測コードを変えたら、親が測り直しを頼む。
4. テストと pre-commit を通して push し、PR #2501 を更新する。
5. 最終回答に、TRT16 の判定、同期比較動画の場所、入れた job、push した HEAD を書き、待つものに合わせた `PARENT_RESULT` を1行で返す（queue を待つなら `waiting` / `queue`、CI を待つなら `waiting` / `ci`）。
