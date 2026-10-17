# Player 注釈と学習循環

共通の[注釈基盤](../../base/annotation/README.md)にplayer bbox固有の処理を接続する。
差分編集・画像確認・教師採用の実装は [review](review/README.md) を参照する。
production への全工程接続と実運用のレビュー方針値は未完了である。
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

## 保存済み予測の確認用動画

`scripts.preview_predictions` は保存済みの player receipt・推論設定・予測・入力 clip の
hash、観測 ID、全 frame/PTS を照合し、bbox と元 confidence を重ねた無音 MP4 を CPU で生成する。
モデルはロードせず、固定したモデル参照の hash を来歴へ残す。重みファイルの再検査・推論は行わない。
native 解像度・全 frame・可変フレーム間隔を保ち、画像は1 frameずつ処理する。
動画には未確認予測であることと target/context を表示し、空検出もそのまま残す。

```bash
.venv/bin/python -m src.tasks.player_detection.scripts.preview_predictions \
  --prediction /absolute/main-repo/outputs/chat_annotation/predictions/player/CLIP/REQUEST/receipt.json \
  --sha256 RECEIPT_SHA256 \
  --output "$PWD/tmp/player-preview" \
  --reason 'player予測の誤検出・見逃し・bbox修正負担を確認'
```

出力先は実行コードの worktree 内 `tmp/` の未作成ディレクトリに限定する。
`preview.mp4` と、入力/モデル/設定の参照、frame/PTS対応、動画hash、描画版・提示理由を持つ
`provenance.json` を保存する。campaign 台帳・予測・採用状態は変更せず、生成は画像確認や
方針承認の証拠にはならない。既存の静止画 preview は[統合CLI](../../annotation/README.md#画像確認公開学習)を使う。

方針の確定には代表例で誤り・漏れ・修正負担と未確認条件を記録する。
対象選手・bbox の意味、レビュー範囲、教師条件、数値の採否閾値をこのスクリプトから決めない。
固定評価は[snapshot契約](../data/snapshots/README.md)に従い source 動画ごとに分離する。
確認済み空 target のない旧 store では負例の誤検出を評価できず、partial label 評価も
全画面 AP の代わりにはならない（[評価実装](../evaluation/partial_labels.py)）。
