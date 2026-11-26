# 担当: clip_056 のイベント時刻の目視ラベル（#2542 の補助担当 label-056）

担当Issue #2542 の「予測を参照せずに4視点映像でラベル付けする」部分のうち、1クリップ `video_000/clip_056`（役割: scoring）だけを行う。

## 共通の前提（親 #1722、2026-11-26 夜間作業）

- 人間の指示: イベント検出モデルを meiji_4cam / multiview-v2 に適用した結果を今夜中に出す。モデルをネット接触にも対応させるため、データ生成経路を拡張してデータ生成・学習まで行う。規則による検出も並行して作り、モデルと比較する。代表8クリップでワーカーが目視でイベント時刻を付け、両方の精度を測る。
- 人間の判断（2026-11-26）: 旧 traj3d_transformer の重みはローカル[local path]、現行 `ball_refiner_3d` のイベントheadをショット・バウンド・ネット接触の3クラスにして代替する。PRは正式レビュー・CI・マージ条件を満たせば親がマージする。
- ネット接触 = ボールとネットの物理的な接触（白帯に触れて越える球、ネットに当たって落ちる球を含む）。通過だけのイベントは扱わない。
- ベンチマーク: `data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v2`（読み取り専用、manifest.json sha256 `0d11010fa2a10d73ba4b03ba01f107c90641868fbb8dd86978503cc761d4ac44`）。69クリップ、全クリップ 59.94005994005994 fps、ダブルス（片側2人）。各クリップの `scene/scene.json` が SceneResult v2。ボール3Dはclip_064（270 frame欠測）以外ほぼ全frameで有効。clip_012・015・026 はボール再注釈版。
- 予測イベントの共通JSON（モデル・規則とも同じ形で保存し、#2542 が照合する）:
  `{"schema":"ball_event_predictions.v1","clip_id":"video_000/clip_012","fps":59.94005994005994,"frames":N,"method":"refiner3|rule","method_version":"<commit or checkpoint sha256>","inputs":{"scene_sha256":"..."},"status":"ok|failed","reason":null,"events":[{"kind":"shot|bounce|net","frame":int,"time_s":float,"score":float,"features":{...},"flags":[...]}]}`。
  frameはクリップ動画の0始まりのframe番号（SceneResultのframe番号と同じ）。
- 目視ラベル（#2542）は、モデル・規則の出力を一切見ずに作る。ラベル作成者以外のワーカーは、ラベルを閾値の調整に使わない（調整用クリップとして明示的に割り当てられた場合を除く）。


## 守ること（独立性）

- 見てよいもの: 4視点の元動画 `videos/video_000/clips/clip_056/media/cam{0..3}.mp4`、manifestのframe数・fps・注記。
- 見ないもの: SceneResultのボール3D・速度・軌跡、モデル・規則の出力、他のラベル担当の結果、`src/utils/rendering/trajectory_analysis.py` 等の自動候補。
- ベンチマークのボール2D注釈（`ball_revision` 等）は、画像内でボールを探す目印としてだけ使ってよい。イベントの時刻は、画像に写った接触（ラケット・地面・ネット）から決め、注釈座標の極値などで決めない。使った場合はラベルに記録する。

## ラベルの付け方

- クリップ全体（0〜最終frame）で、ボールの次のイベントをすべて付ける。ポイント外のものも含め、`in_point` で区別する。
  - `shot`: ラケット（または身体）とボールの接触。サーブ・ボレー・ポイント外の打球を含む。
  - `bounce`: ボールと地面の接触。コート外を含む。
  - `net`: ボールとネット（白帯・ネット本体・ポスト）の物理的な接触。上を通過しただけは付けない。
- frameは0始まり（動画のframe番号＝SceneResultのframe番号）。接触の瞬間に最も近いframeを選ぶ。接触がframe間に起きたと見える場合は近い方を選び、`frame_uncertainty`（±frame）に幅を書く。
- 4視点のうち最も見やすいカメラで決め、他視点で矛盾がないか確かめる。根拠のcamera・観察を `evidence` に1行で書く。
- `status`: `certain`（接触が見える）[local path]`probable`（前後の軌道から確実だが接触の瞬間は隠れている）[local path]`uncertain`（有無または時刻に疑いがある）。
- 判断できない区間（ボールが全視点で見えない、画面外など）は `unlabelable` に区間と理由を書く。その区間のイベントは推測で作らない。
- 主要なイベントごとに、接触frameと前後のcropを並べた根拠画像を保存する。

## 出力

`{"schema":"meiji_event_labels.v1","benchmark":"multiview-v2","clip_id":"video_000/clip_056","fps":59.94005994005994,"frames":N,"role":"scoring","labeler":"<model名>","independence":"モデル・規則・3D軌跡を参照していない","used_ball_annotation_as_locator":true|false,"events":[{"id":"e001","kind":"shot|bounce|net","frame":int,"frame_uncertainty":int,"status":"certain|probable|uncertain","in_point":true|false,"subtype":"serve|groundstroke|volley|smash|let_cord|net_fault|other|null","player_hint":"near|far|null","cameras":["cam0"],"evidence":"..."}],"unlabelable":[{"start":int,"end":int,"reason":"..."}],"notes":"..."}`

保存先は worktree の `outputs/tennis_scene/analyze/event-eval-labels/clip_056/labels.json`（根拠画像は同じdirの `evidence/`）。絶対パスを報告する。コード変更・PRは不要。

## 報告

- イベント数（種別・status別）、unlabelable区間、迷った箇所、labels.json の絶対パス。
- `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` で終える。
