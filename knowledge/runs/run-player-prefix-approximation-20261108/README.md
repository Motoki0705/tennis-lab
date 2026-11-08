# 再学習しないNHT prefix近似の比較

[親の人間判断](https://github.com/Motoki0705/tennis-lab/issues/1723#issuecomment-6722754922)を
受けたC4追加run。初回の正確なlocal RGB契約の不成立は
[前のrun](../run-player-depth-compositing-20261108/README.md)に保持する。
今回は再学習・蒸留を行わず、近似を1方式へ絞る。現在GPU比較前で、選定結果は未記録。

比較するのはalpha掛け直し、特徴正規化、全背景alphaを基準に正規化する3候補。
色はNHTのdisplay encodingをsRGBと仮定してlinearizeし、同じsampleで混ぜて戻す。
これは物理的なlocal RGBを復元する契約ではない。正解色の誤差は測定しない。

`approximation.py`がCPU reference、`test_approximation.py`が独立した極限・色・mask・
既知球面のテスト。`probe.py`はB00の3 cameraを256×144から開始し、うち2 cameraだけを
512×288の共通SSAA2 gridでも確認する。人物輪郭は既知の平面silhouette、球は半径.0335 mの
解析球面であり、C3の本番SMPL/球のrenderではない。NHT本体のtracked fileは変更しない。

prefix取得は前景の各sampleに必要なscalar depthでnative far-plane renderを実行し、
各pixelの特徴・alphaを取り出して組み立てる。depthのbinningやEDのsurface扱いはしない。
この研究用の方法はmean-center-zによるgateを再現し、Gaussianのsurface積分ではない。
C5のper-pixel kernelの性能測定としても扱わない。

`evaluate.py`に数値基準を事前記録した。元RGB・空/全prefix、opacity scaling、mask外不変、
前景色の透過感度、背景alpha、ネット前後、SSAA2の細い球とlarge Gaussianの既知の問題を
比較する。基本極限を満たす候補がなければ成功扱いにしない。

CPUの再現入口:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -n 2 \
  knowledge/runs/run-player-prefix-approximation-20261108/test_approximation.py \
  --durations=0 --durations-min=0
```

GPUの入口は、前runと同じNHT tool環境で `probe.py --asset-root <B00-root> --output <new-output>`。
出力のmetrics.jsonへ `evaluate.py <metrics.json> --output <selection.json>`を適用する。
raw配列はDriveへ保存し、Gitには小さな数値・画像・再現手順だけを残す。
