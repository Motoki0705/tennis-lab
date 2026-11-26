# 担当: 目視評価用の代表8クリップの選定（#2542 の統合担当 primary の最初のターン）

担当Issue #2542 の「代表例と遮蔽・低いボレー・ネット接触・ポイント外の打球などを含む評価用クリップを選び」の部分だけを行う。ラベル付けは別の補助担当が1クリップずつ行う。あなた（primary）は後のターンで、3クラスモデル（#2567）と規則（#2568）の出力をラベルと照合する統合を担当する。このターンでは選定だけを行い、選定後もラベル付けはしない。

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

1. multiview-v2 の69クリップを、4視点の元動画（`videos/video_000/clips/clip_XXX/media/cam{0..3}.mp4`）から粗く見て、各クリップで見える現象を記録する。ffmpegで間引いたframeのcontact sheetを作るなど、効率の良い方法を選んでよい。ボール3D・SceneResultの軌跡・モデルや規則の出力は見ない（まだ存在しないが、他のworktreeに現れても見ない）。manifestのframe数・注記は使ってよい。
2. 次を満たすように8クリップを選ぶ。1クリップが複数を兼ねてよい。
   - ボールとネットの接触が見えるクリップを2つ以上（ネットに掛かって終わるポイント、白帯に触れて越える球、サーブのレットなど）。見つからなければ、探した範囲と結果を報告する。
   - ネット際の低いボレーを含むクリップ。
   - ボールが選手に隠れる場面が多いクリップ。
   - ポイント外の打球・バウンド（ポイント前後の球出し、球を返す、サーブ前に地面につくなど）を含むクリップ。
   - 長いラリー1つと、短いポイント（サービスエース・フォルト等）1つ。
   - clip_064（ボール3D 270 frame欠測）を候補として検討する。
3. 8クリップのうち2つを「調整用」（モデルのピーク閾値や時刻のずれの確認に使ってよい）、6つを「採点用」に固定する。採点用にネット接触のクリップを最低1つ入れる。
4. 選定結果を保存する: `{"schema":"meiji_event_eval_selection.v1","benchmark":"multiview-v2","manifest_sha256":"0d11010f...","clips":[{"clip_id":...,"role":"tuning|scoring","phenomena":[...],"reason":"...","approx_events":{"shot":n,"bounce":n,"net":n}}],"survey":[69クリップ分の短い観察]}`。保存先は `src/tasks/OUTPUTS.md` の規約に沿ったworktreeの `outputs/tennis_scene/analyze/event-eval-selection/<run-id>[local path]`。絶対パスを報告する。
5. 時間の目安は60分以内。コード変更・PRは不要。

## 報告

- 8クリップ（役割・理由・見えた現象・おおよそのイベント数）と、ネット接触を確認したframe付近、selection.json の絶対パス。
- `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` で終える。
