# #2487 統合担当 再開: DINO PT16 の採点完了、#2497 のマージ、取込と親の判断

> 下の「PT32 の計測をスコープから外す」は、前回の再開が起動しなかったため届いていない指示の再送。そのうち「このターンの終わり方」と「取り込みについての訂正」の項は、最後の「この担当でやること」で置き換える。

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

### #2487 統合担当でやること
- **取り消したjob:** `i2487-current-dino-vitpose-pt32-cohort`（実行中だったため途中で止めた。出力は使わない）。`i2487-current-dino-vitpose-pt16-cohort` は残してあり、今実行中。
- **DINO の PT32 精度は計測しない。** 結果の表には「対象外（ユーザー判断）」と記録する。
- **補助担当の結果を統合するとき**は、上の方針で表と図を作る。
  - 新しい条件の parity は「PT32未計測」にする。
  - RF-DETR L と ECDet の測定済み PT32 は、そのまま残す。
  - 補助担当（EdgeCrafter・RT-DETRv4）も、PT32 を抜いて入れ直す指示を受けている。
- **このターンの終わり方:** 新しいjobは投入しない。`i2487-current-dino-vitpose-pt16-cohort` の完了を待つ（queue待ち）。

### 取り込みについての訂正
上の「親が main と #2497 の最新headを取り込んでいる」は、このbranchには当てはまらない。親の取り込みがコンフリクトで失敗したため、親が中止して元の状態に戻した。
- 担当が自分で `git fetch origin main codex/parent-2492-issue-2496-primary` を実行し、`origin/main` と #2497 の最新headを取り込む。
- コンフリクトは、両方の意図を保つように解消する。main 側には #2475 の単眼PLCS（PR #2535・#2536）などが入っている。
- 解消した後、影響範囲のテストと pre-commit を通してから進める。

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

### 取り込み
- PR #2499 の base は #2501 の branch（共通 recipe を再利用するため）のままにする。取り込む先は次の2つ。
  - `origin/main`（`59faa32a2`、#2497 と #2536 を含む）
  - #2501 の最新 head（`origin/codex/parent-2492-issue-2491-primary`、`8a25fe12b`）
- 親が試したところ、`src/tennis_scene/pipeline/definition.py`（FastHMR と #2536 の SkeletonOnlyPlacement）と `tests/integration/tennis_scene/test_fp32_cache_compatibility.py` でコンフリクトする。両方の意図を保って解消し、影響範囲のテストと pre-commit を通す。

### 結果の記録
- `i2487-current-dino-vitpose-pt16-cohort` は完了した。ログ末尾では、dino-swin-l の Recall@0.50/0.75 は 0.9669/0.4699 で、DINO＋ViTPose-H も同じ値。条件ごとに出力を実ファイルで確かめて記録する。
- `i2487-current-dino-vitpose-pt32-cohort` は親が取り消した。DINO の PT32 精度は「対象外（ユーザー判断）」と記録する。
- 現行モデル（DINO・DINO＋ViTPose-H）の PT16・TRT16 の速度は、#2497 で main に入った再計測記録（`knowledge/runs/run-person-model-speed-low-precision/`、parity 未判定）を使う。修正前の速度や、PT32 との一致判定は比較に使わない。

### 前回の質問への親の判断: ECPose と #1739 の下流
- #1739 の入力検査は、このPRでは拡張しない。#1739 側で先行して直すことも頼まない。
- 理由: #2492 の目的は比較と、速度・品質のトレードオフの報告。既定モデルの切替は、別にユーザーの承認が要る。ECPose を下流に流す必要が出るのは、切替が承認されてから。
- #2499 の ECPose 経路は、比較と単独の recipe として使える範囲にとどめる。#1739 の経路と組み合わせたときは、いまの入力検査が明示的に拒否する。この挙動はそのままでよく、黙って DINO に戻す fallback を作らない。PR 本文の「残る制約」に、この拒否と、切替時に #1739 の入力検査の拡張が要ることを書く。

### 補助担当の結果の統合
- EdgeCrafter の TRT16 精度低下（ECDet の Recall@0.50 が約97%から63〜70%、ECPose-X が40%から10%）は、補助担当が TensorRT の段階を診断中。RT-DETRv4・RF-DETR の速度も、main の harness で測り直し中。
- 揃うまでは比較表と図を確定しない。いまのターンは、取込・記録・push までで区切る。

### 終わり方
テストと pre-commit を通して push し、PR #2499 を更新する。最終回答に、取込の結果、記録した値、push した HEAD を書く。補助担当の結果を待つ間は、`PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}` で返す。
