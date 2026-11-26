# 担当: ネット接触を含む合成データの生成と、ball_refiner_3d イベント出力の3クラス学習

担当Issueの本文が仕様の正本。以下は親からの具体指示。

## 共通の前提（親 #1722、2026-11-26 夜間作業）

- 人間の指示: イベント検出モデルを meiji_4cam / multiview-v2 に適用した結果を今夜中に出す。モデルをネット接触にも対応させるため、データ生成経路を拡張してデータ生成・学習まで行う。規則による検出も並行して作り、モデルと比較する。代表8クリップでワーカーが目視でイベント時刻を付け、両方の精度を測る。
- 人間の判断（2026-11-26）: 旧 traj3d_transformer の重みはローカル[local path]、現行 `ball_refiner_3d` のイベントheadをショット・バウンド・ネット接触の3クラスにして代替する。PRは正式レビュー・CI・マージ条件を満たせば親がマージする。
- ネット接触 = ボールとネットの物理的な接触（白帯に触れて越える球、ネットに当たって落ちる球を含む）。通過だけのイベントは扱わない。
- ベンチマーク: `data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v2`（読み取り専用、manifest.json sha256 `0d11010fa2a10d73ba4b03ba01f107c90641868fbb8dd86978503cc761d4ac44`）。69クリップ、全クリップ 59.94005994005994 fps、ダブルス（片側2人）。各クリップの `scene/scene.json` が SceneResult v2。ボール3Dはclip_064（270 frame欠測）以外ほぼ全frameで有効。clip_012・015・026 はボール再注釈版。
- 予測イベントの共通JSON（モデル・規則とも同じ形で保存し、#2542 が照合する）:
  `{"schema":"ball_event_predictions.v1","clip_id":"video_000/clip_012","fps":59.94005994005994,"frames":N,"method":"refiner3|rule","method_version":"<commit or checkpoint sha256>","inputs":{"scene_sha256":"..."},"status":"ok|failed","reason":null,"events":[{"kind":"shot|bounce|net","frame":int,"time_s":float,"score":float,"features":{...},"flags":[...]}]}`。
  frameはクリップ動画の0始まりのframe番号（SceneResultのframe番号と同じ）。
- 目視ラベル（#2542）は、モデル・規則の出力を一切見ずに作る。ラベル作成者以外のワーカーは、ラベルを閾値の調整に使わない（調整用クリップとして明示的に割り当てられた場合を除く）。


## このターンでやること

1. 現状の確認: `data/ball_refiner/single_object`（E:側 `[local path]` にある。D:には無い）は1280ラリーで、shot 4524・bounce 4849・toss 366 に対し net・fence が0件。BLCSシミュレータには `check_net_collision`[local path]`apply_net_collision`（白帯の形状あり）と `record("net", ...)` があるが、ネット接触でラリーが終わる・生成の採否で落ちる等の理由で、datasetに残っていない。原因をコードで特定する。
2. 生成経路の拡張: ネット接触を含むラリーを、設定で明示した割合で生成する。少なくとも (a) 白帯に触れて相手側へ越え、プレーが続く球、(b) ネットに当たって打者側に落ち、ラリーが終わる球、を含める。サーブのレット（白帯に触れて入る）も (a) に含めてよい。接触時刻・位置は物理記録の `net` イベントとして保存し、生成時のbit再現検証を維持する。割合と内訳はmanifestに記録する。既存の生成結果を変える既定値の変更は避け、新しい設定（例: generation の別プリセット）として切り替える。
3. 新しいdataset版を生成する（既存 `single_object` は上書きしない。配置は `src/tasks/OUTPUTS.md` と storage tier の規約に従う）。ラリー数は既存と同程度（train 1024 / val 128 / test 128）を目安に、netイベントがクラスとして学習できる件数になるよう割合を決め、理由を記録する。
4. イベント出力を3チャネル（shot, bounce, net）にする。教師（種別ごとのGaussian）、loss、評価（クラス別のピーク照合 P/R/F1・時刻誤差、照合許容は既存の考え方に合わせて明示）、推論出力、checkpoint契約を更新する。旧1チャネルcheckpointは契約で区別し、静かに読み替えない。toss・fenceは出力しない（fence教師は保持してよい）。
5. テスト（生成・教師・loss・契約）を設計して追加し、pre-commitを通す。
6. training queue（親が共有queueのworker稼働を確認済み）に学習を `add` する。`--provider codex --session <自分のsession> --issue <担当Issue番号>`、`--resource half`。既存の最良設定（knowledge `ball_refiner_3d/000011` のイベントhead付き回帰、L1＋soft CE、GANなし）を基準にし、ステップ数はGPU 1時間以内に収める。queueの使い方は `.agents/skills/training-queue/SKILL.md` に従う。worktreeのコードで学習するため、`add` には `TRAINING_QUEUE_DIR=[local path]` を指定し、worktreeをcwdにしてenqueueする（jobはenqueue時のcwdへcdして実行される）。dataset・output_dirは絶対パスで渡す。Hydraに新しい設定キーを渡すと、古いmainのコードで動いた場合に構造化設定のエラーで気づける。
7. PRを作り（日本語本文、`Closes` は担当Issue）、job IDを宣言して終了する。

## 結果を受け取ったターンでやること

- 学習結果を確認し、合成testでクラス別 P/R/F1・時刻誤差を出す。knowledge記録（knowledge-control skill）をナレッジ記録PRとして作る。
- multiview-v2 の1クリップ（clip_000）で推論する。SceneResult v2 の `ball_3d`・`ball_3d_valid` から入力（座標m・欠損mask・fps）を作る部品を用意する。fpsは 59.94 と学習時 60 が異なる。静かに通さず、許容の仕方（明示の公称fps指定、許容誤差の設定など）を契約として決め、記録する。
- 学習済みcheckpointの絶対パスとsha256、推論APIの呼び方、clip_000での出力の要約を最終報告に書く。#2542 がこの情報で全69クリップに適用する。
