# 公式 MSDA の実行境界

`OfficialMSDeformAttn` は公式 Deformable DETR の Python module と CUDA autograd
function を使う。出典と import のみの変更は [origin.json](vendor/origin.json) と
[NOTICE](vendor/NOTICE.md) を参照する。CUDA は既存の DINO 拡張を使い、公式ソースとの
照合と互換パッチを実験の build manifest に残す。

ビルドと選択は [導入手順](../../../../../../docs/installation/README.ja.md#dino-cuda-extension)
に従う。CPU での構築・保存重みの読込は可能。実行には明示した
`TENNIS_LAB_DINO_EXTENSION`、CUDA、head 幅64、正しい shapes/starts と
batch/im2col の整除条件が必要。CPU・reference への代替実行はない。

FP32/BF16 入力を contiguous FP32 に変換し、公式の Value 射影・sampling・出力射影は
autocast を無効にして実行する。出力は FP32 で、cast を通して勾配を保持する。
BF16 native ではない。境界では FP16/double を拒否するが、低水準の公式関数は
数値検証の double gradcheck に使える。

コンパイラ境界を明示的に無効化するので、呼出側は graph break を許可して記録する。
vendor 内の PyTorch reference は検証専用である。
