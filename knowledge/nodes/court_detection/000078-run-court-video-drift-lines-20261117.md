---
id: run-court-video-drift-lines-20261117
type: run
task: court_detection
sequence: 78
recorded_at: '2026-11-17'
title: 共通Hを固定した線offsetによる複数frame drift診断
issue: 2191
provider: codex
status: failed
summary_impact: updated
date: '2026-11-17'
config:
  mode: video_shared
  profile: signed_line_similarity_v1
  common_h_refit: false
  frame_numerators: [1, 3, 7, 8, 9]
  frame_denominator: 10
  temporal_pooling: false
  reference_frames_used: false
  manual_points_or_net_mask_used: false
  min_lines: 4
  min_transverse: 1
  min_longitudinal: 2
  min_direction_degrees: 12
  line_anchor_sigma_px: 1
  max_projection_std_px: 5
  cpu_processes: 4
  gpu_used: false
  development_seed: 219178
metrics:
  acceptance_met: false
  common_h_accepted_cameras: 4
  common_h_artifact_bytes_unchanged: true
  real_camera_clips: 276
  real_diagnostic_frames: 1380
  real_usable_camera_clips: 121
  real_warning_camera_clips: 4
  real_unobservable_camera_clips: 155
  real_unobservable_frames: 437
  real_tolerance_exceeded_frames: 0
  real_uncertain_projection_frames: 0
  complete_four_camera_clips: 0
  development_cases: 56
  development_matches: 56
  development_missing_direction_unobservable: 8
  development_moving_safe: 0
  development_seconds: 4.883881654008292
  independent_synthetic_cases: 24
  independent_synthetic_matches: 24
  independent_synthetic_unobservable: 0
  independent_synthetic_moving_safe: 0
  independent_synthetic_seconds: 2.3737522330193315
  real_evaluation_seconds: 550.7369017939782
  posthoc_conditional_observable_frames: 384
  posthoc_conditional_unobservable_frames: 53
  posthoc_used_for_acceptance: false
  existing_geometry_and_estimator_bytes_unchanged: true
  default_replay_reused_images: 3545
  default_replay_reused_regions: 8530
  default_replay_mismatch_images: 0
  human_reviewed: false
  frame0_overlays: 276
  diagnostic_overlays: 1380
  native_crop_pairs: 6624
  render_seconds: 237.7925272640132
  rgb_correction_numeric_mismatches: 0
  actual_component_shared_outputs: 121
  actual_component_explicit_failures: 155
  added_test_cases: 22
  validator_requested: 0
repro:
  commit: edf33290d5e22383dd3d66e4e277ec2538c81f62
  command: See knowledge/runs/run-court-video-drift-lines-20261117/reproduce.md
artifacts:
  run_dir: knowledge/runs/run-court-video-drift-lines-20261117
  output_dir: .worktree-outputs/court-video-shared/tennis_scene/evaluate/court-video-drift-lines-20261117
  review: .worktree-outputs/court-video-shared/tennis_scene/evaluate/court-video-drift-lines-20261117/review/index.html
parents:
- run-court-video-shared-acceptance-20261117
- run-court-white-line-reference-20261116
relations: []
papers: []
tags: [video-shared, drift, CPU-evaluation, independent-synthetic]
---

**全69clip×4cameraの受入は未達。** 共通Hはrun 77から再fitせず、bundleと4camera文書の内容hash・ファイルbytesを維持した。
人工8視点の静止/5px/11pxは新診断で24/24一致し、検査不能・移動の安全誤判定は0だった。
実動画は1,380 frameを測り、155/276 camera-clipが検査不能、4cameraそろうclipは0/69だった。
旧診断の投影不確かさ超過から、抽出した一部の線と共通変位の矛盾へ停止理由が変わった。
閾値・採否を結果に合わせて変更せず、矛盾した線を黙って除外しない。次の設計判断は親へ返す。
summary_impactはupdated。既定clip・checkpointの採用は変えず、現在の未解決要因を更新する。

## 固定した入力・方式

共通Hのbundleは `4ac8e386763687fcb6c81f46abf0774e38fc118af4f79b31cdb6e3c2283e3f75`。
共通fitの方式・設定・使用frame・推定値・全camera文書は不変である。
推定値の参照mean/maxはcam0 2.00/3.36px、cam1 0.38/0.70px、cam2 0.49/0.82px、cam3 0.59/0.70pxのまま。
これはcamera内で同じHの比較を69回使う値であり、69回の独立精度測定ではない。
参照と推定器が白線抽出部品を共有する独立性の限界も変わらない。

診断は `floor(k*(N-1)/10), k=[1,3,7,8,9]` の5frame/clipを、結果を見る前に固定した。
旧heldoutの4/5を含む。候補frame 0、fit 1/5・2/5・3/5、参照15frame/cameraと、
clip内index・元source frameの両方で重複しない。先頭frameの画像は目視用で、数値判定の対象ではない。
[frame protocol](../../runs/run-court-video-drift-lines-20261117/frame-protocol.json)の `settings_frozen:false` は
frame選定時の記録で、数値設定の凍結は後の[frozen profile](../../runs/run-court-video-drift-lines-20261117/frozen-profile.json)が正本。
参照H・手指定点・ネット除外帯を診断器へ渡していない。CPUのみ。

共有fitの `supported_lines` をそのまま物理線のIDとして使い、各frameで符号付きの垂直offsetを測る。
画像上の平行移動・回転・一様scaleの4自由度を推定し、可視KPの変位と不確かさへ伝播する。
2本の交差線だけでは交点回りのscaleが不定なので、初期解には3線、最終解には4線以上を要求する。
1物理線につき10/90 percentileの2空間アンカーだけを数え、密な画素を独立支持として加算しない。
最低横1線・縦2線、2方向12度以上、アンカーsigma下限1px、最大投影std 5px、線残差3pxを使う。
1920×1080の対角長へ正規化し、3px warning・mean10/max25px失敗・検査不能は失敗を維持した。
warningにはmean変位から2stdを引いても3px超という条件を使う。

frame間の平均や共分散共有は一切行わず、各frameを独立に判定しclipはworst frameで決める。
同じ姿勢のframe群という仮定はない。疎なサンプル間の急変は検証していない。
線方向が不足する、異なる変位が同等に支持される、観測線が同じ4自由度変位で説明できない場合は失敗。
4自由度で表せないprojectiveなcamera変化も矛盾として失敗しうるため、失敗だけで移動有無を断定しない。

## 開発集合と独立人工評価

開発seed 219178の別8視点について、静止・5px・11px×2frameの48件と、横方向の地面線を除いた8件を測定。
56/56が期待どおり、移動の安全誤判定0。最低支持・不確かさはこの結果を根拠に2026-11-17 09:28:27 UTCに凍結した。
その後、run 77の8視点×静止/5px/11pxを再評価し、24/24一致、検査不能0、移動の安全誤判定0だった。
run 77では20/24一致・4検査不能だった。真値Hから独立に5cm白線、浮いたnet、影、人物、遮蔽物を描く。
既知Hの代わりに抽出器の出力をGTとして使っていない。

run 77は人工共通H行列を保存していなかったため、**人工fixtureだけ**を旧ソースで再構成した。
5frame×8視点のRGB hashとrun 77のprojection誤差mean/maxが1e-9以内で一致した。
実動画の共通Hを再fitしたという意味ではない。復元matrix・検証値・乱数seedはbundleへ保存する。
人工の広い画角・均一paintは、実映像の狭い投影、隣接線、弱い線、遮蔽をすべて代表しない。
今回の人工合格は実動画の全clip採用を代替しない。

## 実動画の結果

|camera|使用可 / 69|うちwarning|失敗＝検査不能 / 69|frame使用可 / 345|frame矛盾|観測可能frameの最大投影std中央値(px)|
|---|---:|---:|---:|---:|---:|---:|
|cam0|21|0|48|217|128|2.47|
|cam1|61|4|8|334|11|2.20|
|cam2|39|0|30|294|51|2.79|
|cam3|0|0|69|98|247|2.79|

数値として信頼できる10/25px超過、投影std上限超過、方向不足による失敗は0件。
失敗437 frameはすべて `non_similarity_or_conflicting_line`。
観測可能943 frameのうち30frameがwarningで、cam1の4clipに集約される。
共通Hは全cameraで参照目標内のまま。155件を安全として通していない。
旧run 77は1frame/clip、新runは5frameのworst判定なので、107→121使用可という数だけを方式の同条件比較と解釈しない。
同じ4/5 frameだけの比較は `result-integrity.json` に別に残す。

![frame別の変位と検査不能](../../runs/run-court-video-drift-lines-20261117/drift-timeline.svg)

全frame表はcamera別の `frames-cam0.csv`〜`frames-cam3.csv`、clip表は `clips.csv`。
旧clip推定との差・採否・選んだモデル候補の来歴はrun 77の値を保持してclip表にも引き継いだ。
HTMLでは全5測定frameと先頭frameを選べ、frameごとの変位・std・判定・矛盾線を参照できる。
先頭276枚・診断1,380枚のoverlayと、手前baseline・net側・左右sidelineの原寸raw/overlay crop 6,624組を保持する（237.79秒）。
CPU browserでframe切替・5行の表・原寸canvas・拡大・warning・連続切替・cropと画像の対応を確認した。
人間による品質承認は未完了。縮小図だけで判断せず、outputsのreviewで原寸を確認する。

## 本当のcameraの動きと測定限界の切り分け

|camera|矛盾した物理線の件数（1frameに複数あり）|事後の部分集合でも検査不能|
|---|---|---:|
|cam0|左doubles 123、左singles 4、centre service 1|0/128|
|cam1|右singles 11|0/11|
|cam2|左singles 27、near service 22、near baseline 10、centre service 1|10/51|
|cam3|右doubles 149、near service 54、near baseline 43、右singles 42、centre service 1|43/247|

![緑が共通H、赤が矛盾した線の抽出点、青が他の抽出点](../../runs/run-court-video-drift-lines-20261117/conflicting-lines.png)

代表cropは元画像も確認した。cam0 clip000/frame106は、左doubles用の探索で近傍の別の線を拾い、
符号付きoffsetが約+16.69pxになる一方、他の線はほぼ0px。cam2 clip004/frame409も左singlesが約−23.03px、
他の線は約1px。cam3 clip000/frame35は右doublesが約−10.26px、他は約+2〜4px。
これらはcamera全体の整合した10px級変位より、線の候補抽出・対応の限界を示唆する。
cam1の代表frameは暗いsidelineと部分的な支持で、他の線が示す3〜4pxの変位と線の傾きが整合しない。
影・人物・隣接線の寄与を、今回の目視だけで全437frameの確定原因へ一般化はしない。

事後分析として、最初の診断が記録した整合subsetだけで同じ4自由度を解き直した。
これは矛盾線を除くと何が残るかの説明であり、**成果物・採否・受入には使っていない**。
437件中384件で条件付きの変位が観測可能だった。mean範囲はcam0 0.09〜0.87px、cam1 3.58〜3.98px、
cam2 0.15〜1.21px、cam3 0.83〜4.66px。いずれも10/25px失敗を裏付ける大変位ではない。
残るcam2 10件・cam3 43件はsubsetでも検査不能で、真のprojective motionと測定の限界を分離できない。
条件付きで求めた値を根拠に安全へ昇格させず、全155 camera-clipを失敗のまま親へ返す。

次案は、drift専用の抽出で正しい弱い線の候補を早期に落とさず、共有した線配置と複数線の変位から
候補の対応を検証する設計。現在の3/10/25pxや不確かさの閾値を緩める案ではない。
狭い画角・隣接線・部分遮蔽を含む独立人工開発集合も必要。追加実装の判断は親が行う。

## 成果物と互換性

新診断は元Hのstoreを変更せず、別の内容アドレスstoreに保存する。
`build_video_drift` は明示した全camera/clipのframe計画を受け取り、動画hash・時間軸を前後で照合する。
`court_kp.video_shared.drift_directory` と `drift_digest` を元Hの明示loadに添えて使う。
H digestと診断camera digestを別々にclip cacheとdataset markerへ入れ、receiptには全frameの結果を残す。
指定した診断の破損・別parent H・不足はエラーで、旧診断へ切り替えない。
両設定を省略した場合だけ元のv1埋込み診断を使うので、過去のartifact/replayを再現できる。

既存geometry・共有fit・estimatorとclip推論の数値ソースはbyte一致。
YAMLの既存値も一致し、追加は診断load用のnull項目2つだけ。
親の許可どおり、run 77の全3,545画像/8,530領域の完全一致replay（mismatch 0）を再利用した。
今回GPUモデルforward・3D再構成・未測定frameの安全・人間の承認を実施したとは扱わない。
一括CLIへの接続は#1752が担当し、この子でCLI全体を受入済みとは書かない。

## 検証・実装中の修正

テストは、変位回復・covariance・画素数で支持を水増ししないこと・方向不足・矛盾・競合する帯・非similarity・
1frameの急変を平均で隠さないことをunitで検証した。integrationは既存のfake動画/実store/計画を再利用し、
診断変更時の依存node再計算、元H不変、marker/receipt、明示失敗、破損・別parent拒否、
単一/複数clipの一致、明示load設定、正確なframe計画とRGB来歴を確認する。
追加22ケース（unit12、integration10）。既存の8自由度診断のテストも歴史的artifact契約として保持する。

関連143ケースは20.35秒、設定catalogを含む464ケースは205.68秒で成功した（suite間に重複あり）。
既存cold importは最大64.77秒、source auditは24.92秒。新境界のcaseは同suiteで最大2.60秒だった。
追加のRGB reader・motion 13ケースは9.04秒、最大0.15秒。
診断あり/なしの単一/複数clip比較2ケースは22.16秒、各6.77/7.07秒で5秒目安を超えた。
実storeと複数計画を使い、実動画CPU評価との同時実行時の実測である。runtime明示loadは0.11秒。

初回起動はdataset pathの正規化契約で測定前に停止したため、絶対pathへ修正した。
初回全件実行ではdecoded BGRを `rgb_sha256` として記録していたので、BGR→RGB変換をreaderへ補った。
channel最小値を使う測定式は不変で、全1,380frameを再実行し、来歴hash以外の全診断・判定が完全一致した。
最終drift bundleは `7a4be8bf20397552a7b287131c3cb0e5fa57ba0dedc16e4a05fa23287c81a7f6`。
実court componentも全276件で確認し、121件は同camera内で同一KP、155件は明示失敗となった（96.37秒）。
修正前の文書とログもoutputsへ保持し、最終成果物としてはRGB版だけを指定する。
数値profile凍結後のmotion sourceの変更はmypy用コメント1行だけで、AST完全一致を記録した。
元の凍結sourceと固定設定を残し、再現時もその差だけを許す。実データ結果による設定調整は0回。
実装commitのpre-commitはruff/mypyとも成功した。最初のhookは追加testのimport整形で停止し、整形後に通過した。
validator指定は0回のため起動していない。

PR #2222はDraftのまま。merge・issue close・統合レビュー・追加設計の判断は親が行う。
