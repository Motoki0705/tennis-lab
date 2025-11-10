# チケット: 検証指標とロギング（HOTA/IDF1 の記録）

## 背景/目的
DanceTrack Val に対する HOTA/IDF1 の算出・ログ出力の仕組みを追加し、早期収束の可視化を行う。

## 対象
- `src/training/scene_model/lightning.py`（validation フック）
- `src/training/scene_model/callbacks.py`（必要なら）

## タスク
- [ ] 検証ループで推論結果（bbox/ID）を収集し、エポック終了時に集計する構造を追加。
- [ ] 依存や実装コストを鑑み、最初は簡易指標（mAP/ID 精度）で代替し、後続で HOTA/IDF1 を導入。
- [ ] `self.log("metrics/idf1", value)` 等で TensorBoard に出力。
- [ ] モック評価器での単体テスト（形だけの指標でもOK）。

## 受け入れ基準（DoD）
- [ ] 検証フェーズで少なくとも1つの数値指標が TensorBoard に出力される。
- [ ] 本指標（HOTA/IDF1）導入方針・TODO をコード/コメントに明記。
