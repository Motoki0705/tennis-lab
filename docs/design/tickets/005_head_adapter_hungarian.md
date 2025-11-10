# チケット: HeadAdapter 実装（HungarianMatcher + bbox/ID ロス）

## 背景/目的
`SceneModel` デコーダ出力を bbox/ID ロス計算へ接続するアダプタを実装する。マッチングは `third_party/DINO` の `HungarianMatcher` を利用する。

## 対象
- `src/training/scene_model/head_adapter.py`

## タスク
- [ ] `HungarianMatcher` を `third_party.DINO.models.dino.matcher` から利用（source=local import）。
- [ ] `compute_loss(outputs, targets, dn_state=None)` を実装（分類CE/IoU/存在スコア/IDロスの合成）。
- [ ] `exist_conf` による gating、`bbox_delta`（オフセット）学習の取り扱いを定義。
- [ ] DINO のマッチング返り値（インデックス組）をそのままロスに適用。
- [ ] ロス内訳を辞書で返し、Lightning の `self.log_dict` で記録可能にする。
- [ ] 型・ドキュメント・最小スモークテスト用のモック出力を用意。

## 受け入れ基準（DoD）
- [ ] モック `outputs/targets` で `compute_loss` が辞書（`total`含む）を返す。
- [ ] Denoiser 併用時もエラーにならない（引数 `dn_state`）。
