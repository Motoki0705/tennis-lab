# Models

`generators/regression.py` は座標+maskから直接予測する。
`generators/flow.py` は同じ条件にstate・timeを加え、x0を予測する。
共通trunkは `components/temporal_transformer.py`、座標・イベントheadは `components/heads.py`。
`src/utils/models/components` のRoPE・RMSNorm・SwiGLUを使用する。
学習済み重みとの互換性のためparameter名と計算順序を保持する。

イベントheadは各frameの2logit（イベントなし／あり）を出力する。
確率はクラス軸softmaxのイベント成分。時間軸では正規化せず、複数イベントを表現する。
欠損座標はゼロ化し、欠損frameもattentionのqueryとして残す。

`discriminators/trajectory.py` は生成された3D軌道だけを評価する。
G/Dの構成値の正本は [model/_base.yaml](../configs/model/_base.yaml) と
[training/_gan.yaml](../configs/training/_gan.yaml)。

モデルはforwardだけを担当する。Flow matching lossは `training/losses/flow_matching.py`、
明示seedによるFlow samplingは `inference/flow_sampler.py`、
モデル選択と入力検証は `model_io/` が担当する。
