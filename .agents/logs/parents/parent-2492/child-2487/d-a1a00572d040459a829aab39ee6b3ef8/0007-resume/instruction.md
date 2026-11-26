# #2487 edgecrafter 再開: 精度ジョブ6件の完了と #2497 最新 head の取り込み

## 何が起きたか

- 待っていた精度ジョブ6件（ECPose L/X、ECDet L/X × COCO/O365、いずれも `-cohort-accuracy-v3`）は、queue 上すべて `done`。ログは `.training_queue/logs/` の各 job 名の `.log`。
- 自動再開は、管理ツールの検査（担当 branch が #2497 の現在の head を含むこと）で拒否された。#2497 の head が `13e508cb7`（「Remeasure PT16 and TRT16 with parity explicitly unevaluated」）に進んでいたため。
- 親が `13e508cb7` をあなたの branch にマージした（`80619ead8`、コンフリクトなし、push はしていない）。中身を確かめ、テストが通ることを確認してから、あなたが push する。

## 方針（前回の指示のまま）

PT32 は計測しない。新しい条件の parity は「PT32未計測」と記録する。ECDet v1 の PT32 の記録は残す。

## 次にやること

1. 6件の結果を、条件（モデル・サイズ・学習データ・PT16/TRT16）ごとに確かめて記録する。
   - `edgecrafter_run.sh` の `accuracy ... || true` は、失敗しても job を `done` で終わらせる。queue の `done` は成功を意味しないので、**条件ごとに出力（metrics・predictions）があるか、失敗したかを実ファイルで判定する**。失敗した条件は理由（OOM・変換失敗など）付きで「実行不能」として記録し、黙って抜かない。
   - 失敗を記録に残さず次へ進む作りになっているなら、失敗を明示的に記録するよう script を直す（次の条件へ進むこと自体はよい）。
2. 速度計測を、取り込んだ #2497 の harness（`13e508cb7` 時点）で queue に入れる。対象は ECPose L/X と ECDet 4重みの PT16・TRT16（TRT16 は engine ができた条件だけ）。PT32 は測らない。parity は harness の明示的な「未判定」モードを使い、記録に harness の commit を残す。#2497 がマージ前に計測コードを変えたら、親が測り直しを頼む。
3. テストと pre-commit を通して push する。
4. 最終回答に、条件ごとの精度（疑似GTでの recall など）と失敗の有無、入れた速度 job、push した HEAD を書き、`PARENT_RESULT` を1行で返す（queue を待つなら `waiting` / `queue`）。
