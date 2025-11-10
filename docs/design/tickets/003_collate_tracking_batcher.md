# チケット: トラッキング用 Collate 実装（可変長→パディング）

## 背景/目的
可変長 window のバッチ化に対応するため、トラッキング専用の collate を用意して `SceneBatch(frames, targets, masks)` を返却する。

## 対象
- `src/datasets/collate_tracking.py`

## タスク
- [ ] `collate_tracking(samples: list[Sample]) -> SceneBatch` を実装（フレームを最大長でゼロパディング、ターゲットはマスク付）。
- [ ] 画像テンソルは `[B,T,3,H,W]`、マスクは `[B,T]` で `False` が実フレーム、`True` がパディングを表すよう統一。
- [ ] ターゲット（bbox/ID）はシーケンス長に合わせてパディング or 空配列を保持。
- [ ] 型・ドキュメント・最小スモークテストを追加。

## 受け入れ基準（DoD）
- [ ] ばらつく長さの2サンプルを与えたときに形状が揃った `frames/targets/masks` が返る。
- [ ] `DataLoader(..., collate_fn=collate_tracking)` でエラーなく動作。
