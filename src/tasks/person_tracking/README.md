# person_tracking

#964のcamera内追跡。検出rowごとの特徴を保存し、同じ検出・pose・外観を複数の追跡方式で使う。
標準pipelineの人物sourceとコート選別は[pipeline README](../../tennis_scene/pipeline/README.md)を参照。
外観＋pose追跡は最初の方式と単独benchmarkの段階で、複数方式の実測は後続。

| モジュール | 責務 |
|---|---|
| `contracts.py` | 1 frameの検出row、source画素box、score、COCO17 pose、外観と明示mask、追跡の実観測→検出row対応 |
| `features.py` | decoded BGR frameの全検出へViTPoseを1回適用し、encoder用cropと同じ座標のpose promptを作る。encoderはtracking方式と独立 |
| `archive.py` | 連続frame・一意rowを検証して特徴をNPZへ保存/読込。元検出artifact・重みhashなどの出自は呼び出し側が渡す |
| `botsort_pose.py` | XYWH Kalman、high/lowの2段対応、外観EMAとpose距離を使う固定camera向けBoT-SORT派生 |
| `methods.py` | 方式の明示選択。未実装名は停止し、別方式へ戻さない。Deep OC-SORT/StrongSORT++のadapterも同じ入出力を使う |
| `court_candidates.py` | CPU開発診断用。全人物を追跡した後、既存プレー領域内の実観測滞在時間で候補を選び、最後に上限6を適用。scoreは使わない |
| `court_linking.py` | 標準pipelineと開発比較で共有する固定選別。足元連続性とCLIPで断片を連結して滞在を集約する。領域はmembership判定にだけ使い、選択済み断片の全実観測を保持する。定義・限界はmodule docstring |
| `selection_diagnosis.py` | 選択されたtrackの人物unit構成と足元座標から、人物混在と領域の誤採用を分離する事後診断。ラベルを選別へ渡さない |
| `linked_timeline.py` | 選別で保持した全実観測から既存v3対応用の連結group軸を作る。handoff重複の1box化と元rowを明示 |
| `appearance_cache.py` | 全raw trackで遮蔽検査し、coreに入るtrackのCLIP cropをCPU計算。完全一致する入力/hashだけ再利用 |
| `selection_burden.py` | 同じcamera×近遠層でraw/選別trackの観測数・累計ID数・frame負荷を集計 |
| `selection_metrics.py` | #933の人物/frame単位とidentity単位による選別評価。非検出と、追跡後の非選手除外を分ける |
| `court_consistency.py` | camera間対応後、同じ予測identityの足元がz=0上でどれだけ一致するかを確認。新しい閾値やidentity補完は加えない |

`DetectionFeatures.rows`はclip・cameraの元検出row。並べ替えや欠落でrowの意味を変えない。
track出力は実観測だけを持ち、Kalman予測boxを実検出とは扱わない。累計6 IDを超えれば停止し、IDを再利用しない。
1 cameraごとにtrackerを構築し、空frameも含め0から順に渡す。
この累計上限は未接続の派生trackerの現実装であり、2026-09-29の新方針への移行は未完了。
新方針の候補上限は`court_candidates.py`と[CPU診断](../../../tests/benchmarks/README.md)で検証する。

外観はCLIP-ReID既定。低いcrop等の外観不足はzero embeddingとmaskで明示する。poseの第3channelは
ViTPoseの回帰heatmapの最大値（確率ではなく有限の実数）を加工せず保持する。1を超える値や負の値を
clip/sigmoidで変換しない。特徴archiveはこの契約を明示したv2のみを読み、v1を暗黙変換しない。
非有限値はframe・検出row・関節・channel・値を付けて停止する。poseはjoint confidenceを持ち、
双方の信頼できる4関節以上のbox内正規化距離を照合へ加える。外観不一致はhigh/low両段でIoUによって打ち消さない。
特徴抽出のprompt契約は将来のKPRに対応するが、**SOLIDER/KPR推論はまだ実装していない**。
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
