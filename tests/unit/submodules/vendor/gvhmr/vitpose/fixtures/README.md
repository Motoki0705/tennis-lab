# ViTPose 数値リグレッションfixture

`heatmap_reference.npz` は置き換え前の実装で計算した合成データのみを含む。
旧実装のコード、画像データ、学習済み重みは含まない。常設テストは外部checkpointや
git履歴、ネットワークを使わず、このfixtureを読み込む。

生成元は tennis-lab commit `9772783ca5179435f969d3be64d67a5523a14e30` の
`src/submodules/vendor/gvhmr/vitpose/{__init__,heatmap_head}.py`。
生成環境は PyTorch `2.13.0+cu130`、CPU・float32・eval mode・1 thread。
乱数は NumPy `default_rng(20261019)` で生成した。

| case | 対象 |
| --- | --- |
| `released_layout` | 標準と同じ2段kernel 4 deconvとkernel 1最終層（チャンネル数は縮小） |
| `alternate_kernels` | kernel 2・3のdeconvとkernel 3最終層 |
| `extra_convs` | kernel 3・1の追加Conv/BN/ReLUとkernel 3最終層 |
| `direct_projection` | deconvなし、最終Convのみ |
| `identity_projection` | deconv後の最終層がIdentity |
| `identity_head` | deconvと最終層が両方Identity |

全caseでbackboneを小さい `Conv2d(3, config.in_channels, 1)` とし、旧 `VitPoseModel` と
旧headを使用した。state dictはBNの統計量を含む全項目を乱数で埋めた。
重み・running meanは正規分布（標準偏差0.25）、BN weight・running varianceは
一様分布 `[0.7, 1.3)`、biasは `[0.05, 0.15)`、整数counterは0。
入力は標準正規分布の `(1, 3, 3, 2)` とその左右反転の2サンプル。

各caseの配列は `case/` prefixで保存する。

- `config`: 設定値をJSON文字列として保存。pickle不要。
- `state/*`: 旧modelの全state dict（`backbone.*`, `keypoint_head.*`）。
- `images`: 入力2サンプル。
- `heatmaps`, `flip_average`: 旧modelの出力と左右反転を戻した平均。
- `*_coordinates`, `*_confidence`: 上記2種類を `keypoints_from_heatmaps` で
  UDP復号した結果。center=`[100, 200]`, scale=`[0.9, 1.2]`, kernel=3。

期待値を再確認する場合は、上記commitの2ファイルを独立した一時packageに取り出し、
保存済みの `config`・`state/*`・`images` を旧modelにstrict loadして再計算する。
入力とstate自体を保存しているため、乱数生成器やtorch初期化の将来の変更に依存しない。
変更後の実装から期待値を上書きしないこと。

実checkpointによる標準ViTPose-Hの新旧比較はPRの検証欄に記録する。
