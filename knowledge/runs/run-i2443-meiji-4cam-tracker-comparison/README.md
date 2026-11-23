# meiji_4camでのrun 8〜10追跡比較

[#2443](https://github.com/Motoki0705/tennis-lab/issues/2443) の比較条件。対象は [#2442の抽出計画](../run-i2442-meiji-pseudo-association-pilot/README.md) の69 clip、46,228 source frame。全1,205標本frameの採点は全体注釈の公開後に行う。追跡、選別、対応付けは標本だけに間引かず全時間で行う。

## 比較条件

[run 8](../run-i964-tracker-matrix-r8-20260930/protocol.md)、[run 9](../run-i964-tracker-linking-r9-20260930/protocol-addendum.md)、[run 10](../run-i964-tracker-hybrids-r10-20260930/protocol-addendum.md) のアルゴリズムと閾値を使う。全候補で #2363 の同期RGB、COCO DINO全画面検出（.30、短辺800、長辺上限1333、重複box統合off）、ViTPoseを共用する。検出と姿勢は#2363で記録したTensorRT FP16の成果物で、歴史的な3cam比較ではこの入力の精度がFP32だった。照合には保存scene indexのSHA-256、descriptorと配列のSHA-256、元rowとframe offsetを使う。

比較用のCLIPとSOLIDERは、run 7の重み・FP32・最小crop高1 px（1920×1080換算）・batch 8で、同じ同期RGBと元boxから生成する。KPRはMarket/SOLIDERの重みとbatch 2を使い、同一frameの他検出poseをnegative promptとする。#2363の保存CLIPはTensorRT FP16・最小crop高16 pxで、現行基準の再利用と試行のCPU疎通に使う。 [run 7の特徴計画](../run-i964-coco-person-features-r7-20260930/plan.json)

| run | 追跡条件 | camera間外観 |
| --- | --- | --- |
| 8・9 | Deep OC-SORT＋pose / CLIP、SOLIDER | CLIP、SOLIDER、KPR |
| 9 | Deep OC-SORT＋pose / CLIP＋Lab | 同上 |
| 9 | StrongSORT++ / CLIP | 同上 |
| 9 | StrongSORT++ / native KPR（当時の事前規則で選ばれた方式） | 同上 |
| 10 | StrongSORT++＋pose / CLIP、Deep OC-SORT＋pose / CLIP＋AFLink＋GSI | CLIP |
| 現行の基準 | #2363 の保存済み StrongSORT++＋pose/CLIP、選別・対応・3D | #2363の保存値 |

run 9の規則による4camでの再選択は、全体注釈でCLIPのDeep、Deep＋Lab、StrongSORT++を採点して行う。raw IDF1が最大、差が1e-6以下ならswitch、fragment、名称の順。採点後に、選ばれた方式のnative KPRを確認する。run 10の比較には当時のStrongSORT++ / KPRを残す。

run 8の `old_old` は3camの観測artifactを再利用した条件で、4camの対応artifactがない。COCO ROIと旧BoT-SORTの再実行経路も削除済み。`new_old`、`botsort_pose__CLIP`、`botsort_pose__SOLIDER` は現行の [削除コミット](https://github.com/Motoki0705/tennis-lab/commit/b772768ad8c7e7ae2988ea0e303dfde64693cdca) で旧アダプタとUltralytics依存がなくなったため除外する。

新しく比較する候補は、run 8の固定コート選別、候補上限6、CLIPによる断片連結、appearance尺度 `62.7 * (similarity - .847)` を共用する。カメラ間対応は既定v3設定を使い、ダブルスのため `players_per_side=2` とする。確定sideは#2363の保存値。未確定のclipはcamera内まで測り、対応・3Dは停止として記録する。新しい比較候補の選手3Dは同じ保存poseとside適用済みの共通校正で、reprojection 40 px（1920×1080換算）、関節confidence .3を使う。三角測量はframe独立のため計画上の標本frameだけを処理する。現行基準は採用済みの産物を読み、再推論した候補と独立した行に載せる。

## 実装前の検証設計

- contractは明示された層・連続区間、IoU 0.5での重複人物unit、既知非選手の誤採用、停止時の分母、カメラ間IDと3D有効mask。既存の `tests/unit/tasks/person_tracking/test_evaluation.py`、`tests/unit/tasks/player_association/test_evaluation.py` と部分注釈の `test_ingest.py` を回帰実行する。
- 新規のCPU unitでは非対称な校正済み2視点で射影位置と元box、奥行き・マスク、同一boxへの重複割当、IDが混線した場合のpairと3D coverageを確認する。層の欠落や注釈範囲外を負例に入れる変更も検出する。
- 実データは#2442試行の16 frame、3 clipで、保存済みの実観測と停止clipを含めた読み込みからCSV出力までを実行する。新規テストはCPU・小配列で1ケース5秒以内、setup 1秒以内を目安に計測する。

## 採点と再実行

[共通指標と3D IDの射影](../../../src/tasks/person_tracking/README.md#部分注釈での新旧経路の比較)を使う。各疑義層の集計には、その層に割り当てられた確定ラベルの全frameを含める。複数の疑義があるframeは該当する各層に入り、疑義全体と全標本の行では一度だけ数える。確信層のみの無作為標本は比較用の対照で、全46,228 frameの無条件無作為標本とは分布が異なる。未検出の人物にはbox参照がないため、選別recallと3D coverageはラベル付き選手に対する値。

下の変数で固定入力と今回の出力先を指定する。元のscene indexと一致する#2363のdatasetは [保存の記録](../run-i2363-trt-player-video/README.md) を参照する。`FULL_LABELS` は #2442 が全標本ラベルを公開したdataset。`REPORT` は全体用と試行用で別ディレクトリを指定する。

```bash
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENCV_FFMPEG_CAPTURE_OPTIONS='threads;2'
compare=(.venv/bin/python tests/benchmarks/person_tracking_meiji.py
  --snapshot "$SNAPSHOT" --selection "$SELECTION"
  --adopted-dataset "$ADOPTED_DATASET" --report "$REPORT")
"${compare[@]}" --phase preflight

# Run each features command as its own job in the shared training queue.
"${compare[@]}" --phase features --encoder clipreid_vitb16_market1501
"${compare[@]}" --phase features --encoder solider_swin_base_msmt17
"${compare[@]}" --phase features --encoder kpr_market_solider_parts

"${compare[@]}" --phase track
"${compare[@]}" --phase score --scope full --labels-dataset "$FULL_LABELS"
```

`--phase score --scope full` は抽出計画の全clip・全frameが公開済みであることを要求する。試行には `--scope pilot --variant saved_default --labels-dataset "$SNAPSHOT/dataset"` を指定する。保存CLIPで候補の疎通を確認するには試行用のディレクトリで `--saved-clip` を明示し、追跡・採点する。clipとvariantを限定するには `--clip` と `--variant` を必要な分だけ指定する。

`--annotation-errors` には、人の確認が集計された場合に、`schema_version: 1`、`input_snapshot_sha256`、`selection_sha256`、`strata` を持つJSONを渡す。`strata` の各項目は層名をキーとし、`checked_boxes` と `error_boxes` を記す。人でも判定できないboxは `checked_boxes` から省く。入力のない層の誤り率はJSONでnull、CSVで空欄になる。4camの人手誤り率には、同じ4camの各層の監査値を用いる。

## 試行の結果

#2442公開済みの3 clip・16 frame・486 boxを現行経路の保存出力で採点した。raw/group IDF1は0.8484/0.8527、選別precision/recallは0.9854/0.8080、欠測を含むpair F1は0.5333、確定・照合済みpair F1は0.6998。3D coverageは34/64（53.1%）、停止は1/3 clip。注釈したframeは連続区間が短く、各層のswitchとfragmentは0。4camの人手誤り率は計測待ち。 [6層と合計の暫定値](pilot-saved-default.csv)

clip_025の4 frameでは#2363の保存CLIPを入力に、Deep、Deep＋Lab、StrongSORT++、run 10の2条件で追跡から採点まで通した。StrongSORT++＋poseの4カメラでは、保存済みのID・box・実観測・元rowと再実行が配列一致した。Deep＋Labのcam3は複数の連結候補が残って停止した。採点ではそのカメラの予測を空とし、固定コート選別とカメラ間対応を続けた。 [疎通時の暫定値](pilot-clip-025.csv)

全体のpreflightでは、69 clip・276カメラclip、動画と同期した46,228 source frame、1,332,834検出rowを照合した。4camの全1,205 frameでの採点、FP32外観による全条件の出力、層別の人手誤り率は後続工程。

## 採否基準の暫定案

#2446の主比較では同じ4camラベルに対する現行経路を対照にする。疑義全体でpair F1か3D coverageのいずれかが0.05以上改善し、もう一方は0.02以内の低下に収まることを候補にする。確信層の無作為標本ではpair F1、3D coverage、選別precisionとrecallの低下を各0.02以内とし、69 clip全体の停止は現行の14件以下を候補にする。全体集計の比較表と、clip単位のばらつき、層別の人手誤り率を見て担当親が閾値を確定する。
