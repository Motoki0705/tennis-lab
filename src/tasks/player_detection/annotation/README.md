# Player 注釈と学習循環

共通の[注釈基盤](../../base/annotation/README.md)にplayer bbox固有の処理を接続する。
`PlayerPredictionAdapter` は、ballと同じ `Predictions` 台帳から新規clipだけを
推論する。別モデルになったことを理由に既存教師を再生成・上書きしない。

- `DinoCheckpointSpec` はexport形式、DINO 4-scale Swin-L/class 1の構成、DINO
  repository commit、学習epoch/stepを固定する。既存 `DinoPlayerDetector` の
  出自検証・strict load・BGR前処理を再利用し、exportの元checkpoint hashが
  planの学習用重みと一致することを確認する。COCO重みへの読み替えはしない。
- `DinoInferenceConfig` はscore閾値、short/max-long side、CUDA、native BGR復号と
  bbox座標系を固定する。model内の前処理後、出力は元画像の `xyxy` edge座標へ
  戻る。画像端は `(width,height)`。ballの点座標 `(width-1,height-1)` とは異なる。
- 全実frameへ一度ずつ推論し、source frame/PTSとclip PTSを全行に残す。各候補の
  観測IDは推論request、frame、元候補順位から決まる。scoreと元bboxを保持し、
  追跡ID・選手ID・poseはこの工程で生成しない。
- modelが0件を返したframeも空の候補行を残す。すべての行は
  `bbox_proposals_unreviewed` であり、検出0件を「画像確認済みの対象なし」という
  負例へ変換しない。漏れの追加・bbox修正・非選手/重複の除去は後続レビューが行う。
- 1 adapterで1モデル/設定を共有し、RGBは1frameずつ復号する。`close()` でモデルを
  解放できる。GPUジョブは共通queueとadmissionから起動し、レビューworker数だけ
  DINOをロードしない。

保存先は `predictions/player/<clip-id>/<request-digest>/boxes.json` と共通receipt。
再開・hash・新規限定・競合の契約は共通 `src.tasks.base.annotation.predictions` が
所有する。raw候補は不変で、将来の修正bboxと追加観測は別のrevisionに保存する。

単体試験は実VFR MP4と合成検出出力でframe/PTS・BGR・native box・score閾値の
float32境界・空候補・観測ID・export対応・cacheを検証する。実DINOモデルのCUDA
推論と画像上の品質確認はpilotで実施する。
