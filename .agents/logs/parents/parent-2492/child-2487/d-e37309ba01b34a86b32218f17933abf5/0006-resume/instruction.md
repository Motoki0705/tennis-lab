# #2487 rfdetr 再開: #2497 のマージを受けて速度を計測する

## 親からの指示: PT32 の計測をスコープから外す（ユーザー判断 2026-11-25 13:00 JST）

親が自動再開を止め、待っていたqueue jobのうち PT32 を含むものを取り消した。障害ではない。新しい方針で、このターンから進める。

### ユーザーの判断
- **PT32 の計測は、全体で今後行わない。** PyTorch FP32 の精度計測・速度計測が対象。測定済みの PT32 の結果（精度・暫定速度・parity）は、そのまま残す。消したり測り直したりしない。
- **PT16・TRT16 と PT32 の parity 判定もやめる。**
  - 新しく測る PT16・TRT16 は parity を判定しない。
  - そのために PT32 を推論することもしない。基準を得るためだけの1回の推論も含む。
- **検出・姿勢で新しく測る条件は、疑似GTで実測した精度と速度でトレードオフ図に載せる。** parity は「PT32未計測」と記録する。
- **身体復元で新しく測る PT16・TRT16（FastHMR）は、速度だけの参考値として扱い、比較図から外す。** 品質を確かめる手段がないため。
- **TRT16 engine を作るための ONNX 書き出しは続けてよい。** PT32 の重みからの書き出しは変換であって、計測ではないため。
  - 書き出し直後に少数のサンプルで eager の出力と比べる確認は、変換の診断として記録するだけにする。parity の判定には使わない。
- **投入済みjobに含まれる PT32 も外す。** 親が取り消したjobは下に書いてある。
- **Sapiens2 は 1B・5B とも比較から外した**（ユーザー判断、GPU時間の制約）。

### 共通の注意
- PT32 を外すときは、明示的な設定として外す。「PT32 の基準が見つからないので判定を飛ばす」のような、黙ったfallbackにしない。parity を要求した条件で基準が欠けていたら、従来どおりエラーにする。
- このターンの開始時点で、親が main（#2498 を含む）と #2497 の最新headをこのbranchへ取り込んでいる。影響範囲のテストと pre-commit を通してから、commit・pushする。
- 最後の `PARENT_RESULT` は1行の正しいJSONにする。queue待ちなら、入れ直したjobのIDを `job_ids` に入れる。

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
- 前回取り消した速度計測（修正前 harness の job）を、main の harness で queue に入れ直す。対象は精度を測った条件の PT16・TRT16。
- 上の pt32 指示のうち「親が main と #2497 の最新headを取り込んでいる」は、今回は当てはまらない。共通連絡の手順1のとおり、自分で main を取り込む。
- テストと pre-commit を通して push する。最終回答には、入れた job と push した HEAD を書き、queue を待つ `PARENT_RESULT` を返す。
