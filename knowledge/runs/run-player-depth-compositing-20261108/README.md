# NHTと不透明前景の奥行き合成の検証

親 #1723 / C4 #1728。C5 #1729へ渡す方式の研究記録。
現在はCPUの既知層を検証済みで、Colab L4で固定NHT runtimeを構築中。
背景再学習・NHT本体の変更・本番統合は実施していない。

現行RGB/alpha/EDだけでは前景を途中へ挿入できない。RGB/alpha/EDが完全に同じでも
遮蔽結果が異なる反例を[CPU出力](layers.json)に記録した。赤(α=.5,z=2)と
赤(α=1,z=8)の背景と、赤(α=1,z=5)の背景はいずれもRGB=(1,0,0)、alpha=1、
ED=5になる。z=4の緑前景を挿入すると、それぞれ(.5,.5,0)と(0,1,0)になる。

必要な合成情報は前景までのpremultiplied linear RGB寄与 `C_<` と透過率 `T_<`。
不透明前景の色を `C_fg` とすると `C = C_< + T_< C_fg`。
ただし現行NHTは特徴量を合成後、非線形MLPを適用する。途中までの特徴量をdecode
したRGBがこの `C_<` になるとは限らず、そのalphaで再度重み付けしても保証はない。

## 再現する入口

```bash
.venv/bin/python knowledge/runs/run-player-depth-compositing-20261108/layers.py \
  --output tmp/player-depth-layers.json
.venv/bin/python -m pytest -n 2 \
  knowledge/runs/run-player-depth-compositing-20261108/test_layers.py \
  tests/unit/synthetic_data_generation/rendering/nht/test_depth.py::test_nht_depth_to_metric_preserves_zero_and_converts_scale \
  tests/unit/synthetic_data_generation/rendering/nht/test_contracts.py::test_render_request_writes_strict_public_payload \
  --durations=0 --durations-min=0
```

`prepare_assets.py`は採用済みDrive manifestとB00の6ファイルだけを取得・検証する。
`gpu_probe.py`はNHTの隔離tool Pythonで実行する。3つの256×144 court-eye cameraと
4つの中央rayで手計算できるGaussian sheetを使う。内部hookは研究専用で、公開
file境界へ追加済みの機能ではない。現行公開APIの意味は
[NHT consumer README](../../../src/synthetic_data_generation/rendering/nht/README.ja.md)を参照。

GPU probeの完了後、終了状態とDrive保存receiptを照合して実測・契約・再開条件を記録する。
