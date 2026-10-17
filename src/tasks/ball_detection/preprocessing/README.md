# 固定の画像前処理

`mdd.py`は共通の正負輝度差・sigmoid変換の正本。
`RGBToMDD`はRGB順uint8 `B,T,3,H,W`から、入力と同じdeviceでFP32の2ch MDDを作る固定layer。
座標モデルのforwardに含めてcompileする。学習parameterはなく、係数・色順序・normalizationを
checkpointのinput contractに保存する。FPS間引きはreaderが先に行い、窓先頭のMDDは0。
ConvNeXtの既存model I/Oも同じ変換関数を参照する。

`rgb_mdd.py` は5chのversion付き入力契約・変換・契約照合を担当する。
仕様は [MDDモデル](../models/mdd/README.md) を参照。
