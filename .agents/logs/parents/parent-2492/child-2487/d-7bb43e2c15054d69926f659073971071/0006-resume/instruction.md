# #2487 rtdetrv4 再開: 精度 suite 2件の完了と #2497 のマージ

## 親からの共通連絡: #2497（速度harnessの修正版）が main にマージされた

- PR #2497 は main にマージされた（merge commit `59faa32a2`）。共通部品は main の `src/utils/inference_benchmark.py`・`src/utils/inference_parity.py`・`src/utils/inference_benchmark_inputs.py`。使い方は `src/utils/inference_benchmark.md` と `knowledge/runs/run-person-model-speed-baseline/README.md`。
- 修正前との主な違い:
  - 1反復の終わりを、adapter が返った後のデバイス同期までにした。計時区間が変わったので、修正前の harness で取った速度は使わない。
  - 条件の状態は `ok`・`parity_failed`・`infeasible`。parity を測らないときは `parity_not_evaluated_reason`（理由必須、parity callback と同時指定は不可）で明示する。
  - 固定入力は、期待 SHA を検証する共通の入口から読む（manifest `64c74d75…1a62`、batch 別の計時 ID 順は `timing-protocol.json`）。
  - 既存出力の再利用は拒否される。新しい出力先を使う。
- 自動再開が「git merge-base failed」で拒否されていたのは、管理ツールが #2497 の head を含むことを求めていたため。マージで、この検査は外れた。

### やること（共通）
1. `git fetch origin main` をして、main を自分の branch に取り込む。コンフリクトは両方の意図を保って解消し、影響範囲のテストと pre-commit を通す。
2. 速度計測は main の harness で行う。GPU は training queue 経由。
   - PT32 は測らない（ユーザー判断 2026-11-25）。PT16・TRT16 だけを測る（TRT16 は engine ができた条件だけ）。
   - parity は `parity_not_evaluated_reason` に「PT32計測はスコープ外（ユーザー判断 2026-11-25）」と書いて、明示的に未判定にする。表では「PT32未計測」。
   - 記録には harness の commit を残す（main を取り込んだ自分の HEAD と、それが含む `59faa32a2`）。
3. 13e508cb7 で固定して既に queue にある速度 job は有効として扱う。main の harness と計測コードが同じで、30f3fcddf は記録の追加だけだったため。取り消さない。
4. 測定済みの PT32 の値はそのまま残す。
5. 失敗を隠さない。queue の `done` は成功を意味しない。条件ごとに出力の有無を実ファイルで判定し、失敗した条件は理由を付けて `infeasible` として記録する。
6. 最後の `PARENT_RESULT` は1行の正しい JSON にする。queue を待つなら、入れた job の ID を `job_ids` に入れる。

## この担当でやること
- 待っていた2件（`i2487-rtdetrv4-l-suite-fp16-v3`・`i2487-rtdetrv4-x-suite-fp16-v3`）は、queue 上は `done` で終わった。ログの末尾には TRT16 B1/B2 の `ok` が出ている。ログは `.training_queue/logs/` の各 job 名の `.log`。
- 条件ごとに出力を実ファイルで確かめ、疑似GTでの精度を記録する（Recall@0.50/0.75 など、EdgeCrafter と同じ採点器）。
- この suite に修正前 harness の速度が入っていれば、それは修正前の値として扱う。main の harness で PT16・TRT16 の速度を queue に入れ直す。
- TRT16 の精度が PT16 から大きく落ちていないか比べる。参考: EdgeCrafter の ECDet は、TRT16 の Recall@0.50 が PT16 の約97%から63〜70%に落ちた。ONNX Runtime CPU は PT16 と一致したので、TensorRT の段階が疑われている（同じ担当が GPU で診断中）。RT-DETRv4 で同じような低下があれば、同じ前処理済み入力テンソルを PT16 と TRT16 に入れる小さい診断で切り分ける（GPU は queue 経由）。原因が分からなければ「原因未特定」と明記し、黙って比較表に載せない。
- テストと pre-commit を通して push する。最終回答には、条件ごとの精度、失敗の有無、入れた job、push した HEAD を書く。
