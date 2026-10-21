# ViTPose 数値リグレッションfixture

`heatmap_reference.npz` と `udp_reference.npz` は置き換え前の実装で計算した合成データのみを含む。
旧実装のコード、画像データ、学習済み重みは含まない。常設テストは外部checkpointや
git履歴、ネットワークを使わず、このfixtureを読み込む。

## ランダム重みによるheatmap比較

`heatmap_reference.npz` は、符号の異なる乱数重みでheatmapの一致を検査する。
生成元は tennis-lab commit `3ad2bb3806e8e266d51878c7f27950512f93744f` の
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
  ランダムheatmapは平坦でUDPの局所Hessianが特異に近くなるため、これらの
  座標・confidenceは保存時の記録として保持し、常設テストの座標比較には使わない。

期待値を再確認する場合は、上記commitの2ファイルを独立した一時packageに取り出し、
保存済みの `config`・`state/*`・`images` を旧modelにstrict loadして再計算する。
入力とstate自体を保存しているため、乱数生成器やtorch初期化の将来の変更に依存しない。
変更後の実装から期待値を上書きしないこと。

## 明瞭なピークによるUDP座標比較

`udp_reference.npz` は同じ旧commitのmodel・head・flip・UDP復号を使って生成した。
6構成すべてのstate dict・入力・UDP座標・confidenceを含む。
入力は13×11のGaussian（中心 `[4.3, 5.1]`、sigma=1.2）を3チャンネルに
`[1, 0.8, 0.6]` 倍で配置したものと、その左右反転。
Convの重みは正の乱数を入力チャンネル数とkernel面積で正規化する。
stride=2のdeconvは4倍し、BNはweight=1、bias/mean=0、variance=1に設定する。
これにより入力のピークを維持する。
乱数は NumPy `default_rng(20261020)`、生成環境は上記と同じ。

ランダム重みfixtureのheatmap比較（rtol=1e-5、atol=1e-7）と、
このfixtureの座標比較（rtol=1e-5、atol=1e-4）を分ける。
CPUの演算順によるheatmapの丸め差を、平坦な分布のUDP復号で増幅させない。
座標の許容誤差は変更していない。
テストではheatmapの比較許容範囲内の誤差でargmaxが変わらないピーク差も検査し、
同範囲内の固定乱数摂動を加えてもUDP座標が許容誤差内に収まることを確認する。

再生成はrepo rootで次を実行する。スクリプトは `git show` で旧ソースを
メモリに読み込み、置き換え後のhead・builderを参照しない。
git履歴を必要とするのは再生成時だけである。

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python tests/unit/submodules/vendor/gvhmr/vitpose/fixtures/generate_udp_reference.py
```

実checkpointによる標準ViTPose-Hの新旧比較はPRの検証欄に記録する。
