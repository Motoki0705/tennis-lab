# meiji_4camでのrun 8〜10追跡比較

[#2443](https://github.com/Motoki0705/tennis-lab/issues/2443) の比較条件。対象は [#2442の抽出計画](../run-i2442-meiji-pseudo-association-pilot/README.md) の69 clip、46,228 source frame。全1,205標本frameの採点は全体注釈の公開後に行う。追跡、選別、対応付けは標本だけに間引かず全時間で行う。

## 比較条件

[run 8](../run-i964-tracker-matrix-r8-20260930/protocol.md)、[run 9](../run-i964-tracker-linking-r9-20260930/protocol-addendum.md)、[run 10](../run-i964-tracker-hybrids-r10-20260930/protocol-addendum.md) のアルゴリズムと閾値を使う。全候補で #2363 の同期RGB、COCO DINO全画面検出（.30、短辺800、長辺上限1333、重複box統合off）、ViTPose、CLIPの成果物を共用する。記録時のTensorRT FP16を入力とし、3cam比較のFP32入力とは精度・撮影条件が異なる。追加特徴は同じbox・pose・元rowでSOLIDER、KPR native partsを生成する。特徴を流用する前に保存scene index、依存fingerprint、row・offsetを照合する。

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
