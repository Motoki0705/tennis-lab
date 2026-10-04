# person_tracking

#964のcamera内追跡。検出rowごとの特徴を保存し、同じ検出・pose・外観を複数の追跡方式で使う。
標準pipelineの人物sourceとコート選別は[pipeline README](../../tennis_scene/pipeline/README.md)を参照。
[保存tracking・ID区間のデータレビューUI](review/README.md)は、元画像/crop・欠測・保存GSIを読取専用で点検する。
[選手poseのGPTレビュー](../../tennis_scene/chat_annotation/player_pose/README.md)も、
`features.py` / `sequence.py` の共通処理からコート選別前のraw追跡を利用する。
既定はユーザーが採用した **StrongSORT++＋pose/CLIP**。run 10の固定設定を使用する。
AFLinkは論文再実装と公開重みを当面使うが、**重みの独立した利用条件は未確認**。
継続利用か自前再学習かは後日判断する。出自・hash・再配布しない方針は[NOTICE](strongsort_NOTICE.md)を参照。

| モジュール | 責務 |
|---|---|
| `contracts.py` | 1 frameの検出row、source画素box、score、COCO17 pose、外観と明示mask、追跡の実観測→検出row対応 |
| `features.py` | decoded BGR frameの全検出へViTPoseを1回適用し、encoder用cropと同じ座標のpose promptを作る。encoderはtracking方式と独立 |
| `archive.py` | 連続frame・一意rowを検証して特徴をNPZへ保存/読込。元検出artifact・重みhashなどの出自は呼び出し側が渡す |
| `botsort_pose.py` | XYWH Kalman、high/lowの2段対応、外観EMAとpose距離を使う固定camera向けBoT-SORT派生 |
| `deep_ocsort_pose.py` | 公式Deep OC-SORTのobservation-centric Kalman再更新・方向速度・adaptive appearanceに共通poseコストを加えたadapter。出自・差分は`deep_ocsort_vendor/NOTICE.md` |
| `strongsort.py` / `strongsort_offline.py` | 論文からのStrongSORT・AFLink・GSI推論再実装。明示的なpose重み（既定0）で共通pose距離を両照合段へ加算できる。GSI補間は別maskで保持。[出自と重みの制約](strongsort_NOTICE.md) |
| `part_archive.py` | 検証済みKPR native archiveのreader。Deep OC-SORT / StrongSORTへ共通可視partのEuclidean距離を渡す |
| `feature_tracks.py` / `evaluation.py` | 元検出rowを維持するscatter・共通外観samplingと、部分参照ラベル上のcamera内IDF1/switch/fragment |
| `methods.py` / `sequence.py` | 比較用tracker factoryと、採用profile専用のproduction/文脈共通`track_sequence`。元row・pose・CLIP、AFLink source ID、GSI syntheticを保存 |
| `duplicate_boxes.py` | 検出直後の任意greedy統合（IoU>=.8）。score降順・同点元row順でkeep/dropを記録。既定off |
| `court_candidates.py` | CPU開発診断用。全人物を追跡した後、既存プレー領域内の実観測滞在時間で候補を選び、最後に上限6を適用。scoreは使わない |
| `court_linking.py` | 標準pipelineと開発比較で共有する固定選別。足元連続性とCLIPで断片を連結して滞在を集約する。領域はmembership判定にだけ使い、選択済み断片の全実観測を保持する。定義・限界はmodule docstring |
| `selection_diagnosis.py` | 選択されたtrackの人物unit構成と足元座標から、人物混在と領域の誤採用を分離する事後診断。ラベルを選別へ渡さない |
| `linked_timeline.py` | 選別で保持した全実観測から既存v3対応用の連結group軸を作る。handoff重複の1box化と元rowを明示 |
| `appearance_cache.py` | 全raw trackで遮蔽検査し、coreに入るtrackのCLIP cropをCPU計算。完全一致する入力/hashだけ再利用 |
| `selection_burden.py` | 同じcamera×近遠層でraw/選別trackの観測数・累計ID数・frame負荷を集計 |
| `selection_metrics.py` | #933の人物/frame単位とidentity単位による選別評価。非検出と、追跡後の非選手除外を分ける |
| `court_consistency.py` | camera間対応後、同じ予測identityの足元がz=0上でどれだけ一致するかを確認。新しい閾値やidentity補完は加えない |

`DetectionFeatures.rows`はclip・cameraの元検出row。並べ替えや欠落でrowの意味を変えない。
track出力は実観測だけを持ち、Kalman予測boxを実検出とは扱わない。raw ID数は無制限で、IDを再利用しない。
1 cameraごとにtrackerを構築し、空frameも含め0から順に渡す。
上限6は共通コート選別後のgroupにだけ適用する。

`TrackingConfig`は採用済み`strongsort_pp_pose`＋CLIPだけを受け付ける。
データセット生成用の`DatasetTrackingConfig`は`strongsort_pp_appearance`＋CLIPを固定し、
pose重み0でpose距離・更新を計算しない。`FeatureExtractor(None, UnpromptedEncoder(...))`は
poseモデルやpromptを使わず、`DetectionFeatures.poses`と`TrackEvidence.poses`を`None`に保つ。
pose必須の消費側は`require_poses()`で明示検証する。通常profileへposeなし入力を渡すと失敗する。
他のtrackerは`methods.build_tracker`と`tests/benchmarks/person_tracking_*.py`の比較入口で使用する。
標準pipelineに比較方式を指定すると停止する。
名前の誤り・欠損重み・不正な特徴や状態は停止する。profileの値は`TrackingConfig.identity()`と
[run 10定義](../../../knowledge/runs/run-i964-tracker-hybrids-r10-20260930/protocol-addendum.md)で確認できる。
既定のpose重みは.15、低信頼poseの扱い・AFLink/GSIも固定定義どおり。Kalman潜在状態は実boxとは区別する。

#935の文脈生成は、検出直後に`merge_person_boxes(..., enabled=設定値)`を呼び、保持した元rowを
`FeatureExtractor`へ渡し、全frameを`track_sequence(..., config=TrackingConfig(), aflink=AFLink(path))`
へ渡す。同じ入口を標準`PersonTrackingModule`も使う。学習用JPEGの復号は呼び出し側が所有する。
`TrackingSequence.evidence`から実観測のpose/外観を得る。`reconstruction.interpolated`はsyntheticであり、
文脈の観測maskには使わない。camera全体を処理してからAFLinkするため、chunkごとにIDを再初期化しない。

外観はCLIP-ReID既定。低いcrop等の外観不足はzero embeddingとmaskで明示する。poseの第3channelは
ViTPoseの回帰heatmapの最大値（確率ではなく有限の実数）を加工せず保持する。1を超える値や負の値を
clip/sigmoidで変換しない。特徴archiveはこの契約を明示したv2のみを読み、v1を暗黙変換しない。
非有限値はframe・検出row・関節・channel・値を付けて停止する。poseはjoint confidenceを持ち、
双方の信頼できる4関節以上のbox内正規化距離を照合へ加える。
BoT-SORT+poseでは外観不一致をhigh/low両段でIoUによって打ち消さない。
特徴抽出のprompt契約はKPRの入力にも使える。KPR推論portはnative parts/visibilityを保持し、
共通の単一embeddingへ暗黙変換しない。距離・EMA・区間平均は`player_association/appearance/parts.py`、
比較の明示的な尺度転用は[run 9 addendum](../../../knowledge/runs/run-i964-tracker-linking-r9-20260930/protocol-addendum.md)を参照。
SOLIDERの出自と前処理は[notice](../player_association/appearance/solider_vendor/NOTICE.md)を参照。
`encode_appearance`は保存済みposeを再利用し、encoder追加でViTPoseを再推論しない。
CLIP用adapterはpromptを使わないことを明示する。重み不足やモデル出力不正は停止する。

[BoT-SORT原論文](https://arxiv.org/abs/2206.14651)と
[Ultralyticsの実装](https://docs.ultralytics.com/reference/trackers/bot_sort/)を参照した派生方式。
固定cameraなのでGMCを使わず、pose距離・外観vetoを追加する。high scoreの新規trackを直ちに出し、
IoUだけで重複trackを削除しない。原論文の追試とは呼ばない。閾値は初期値でMeijiによる選択・調整は未完了。

`Pose2DFrameSequenceRequest`と共通crop処理は#935の`74ac16d6`から取り込んだ。raw動画とJPEG文脈の双方から同じ
特徴処理を呼べる。#935 branch/worktreeは変更しておらず、積み直し時に同じ修正を重複させない。

単独の実データ実行は`tests/benchmarks/person_tracking_features.py`。GPUでの特徴生成は共有training queueを使う。
`features` phaseが元検出artifactのhash、ViTPose/CLIP重みのSHA-256、feature config、source動画、出力hashを保存し、
`track` phaseは同じNPZをCPUで使う。BoT-SORT-style derivativeの結果には必ず
`--method ultralytics_botsort`の旧実装（同じ#937検出、sparse optical flowあり）を並べる。
baselineは旧wrapperの観測box/IDを保存し、Kalman更新boxを検出row対応とは称さない。
`--max-frames`を指定したrunはsmokeであり全clipの評価ではない。
