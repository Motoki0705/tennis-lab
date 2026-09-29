# person_tracking

#964のcamera内追跡。検出rowごとの特徴を保存し、同じ検出・pose・外観を複数の追跡方式で使う。
現在は最初の方式と単独benchmarkの段階。標準pipelineのcomponent接続と複数方式の実測は後続。

| モジュール | 責務 |
|---|---|
| `contracts.py` | 1 frameの検出row、source画素box、score、COCO17 pose、外観と明示mask、追跡の実観測→検出row対応 |
| `features.py` | decoded BGR frameの全検出へViTPoseを1回適用し、encoder用cropと同じ座標のpose promptを作る。encoderはtracking方式と独立 |
| `archive.py` | 連続frame・一意rowを検証して特徴をNPZへ保存/読込。元検出artifact・重みhashなどの出自は呼び出し側が渡す |
| `botsort_pose.py` | XYWH Kalman、high/lowの2段対応、外観EMAとpose距離を使う固定camera向けBoT-SORT派生 |
| `methods.py` | 方式の明示選択。未実装名は停止し、別方式へ戻さない。Deep OC-SORT/StrongSORT++のadapterも同じ入出力を使う |

`DetectionFeatures.rows`はclip・cameraの元検出row。並べ替えや欠落でrowの意味を変えない。
track出力は実観測だけを持ち、Kalman予測boxを実検出とは扱わない。累計6 IDを超えれば停止し、IDを再利用しない。
1 cameraごとにtrackerを構築し、空frameも含め0から順に渡す。

外観はCLIP-ReID既定。低いcrop等の外観不足はzero embeddingとmaskで明示する。poseはjoint confidenceを持ち、
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
`track` phaseは同じNPZをCPUで使う。`--max-frames`を指定したrunはsmokeであり全clipの評価ではない。
