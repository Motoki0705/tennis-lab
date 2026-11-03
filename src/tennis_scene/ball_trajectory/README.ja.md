# 区間物理軌道とcamera相対時刻

[English](README.md) · [简体中文](README.zh-CN.md)

CPUの任意方式。courtのmetre座標で `dX/dt=V`、`dV/dt=(0,0,-g)-k|V|V` をfitする。
区間ごとの初期位置・速度6変数と、clip共通の非負drag係数・camera相対時刻を同時推定する。校正は固定。
`ball_reconstruction.method=segment_physics` と
`ball_reconstruction.segment_physics.observation=candidates` / `heatmap` で選ぶ。
既定は `triangulation`。設定の正本は `configs/pipeline.yaml`。
任意componentは採用済み `ball_points` とgate前の `ball_detections` v2に依存し、
model evidenceを要求する。注釈importは拒否し、初期値・side・alignmentは標準pipelineを使う。


Pythonの設定には既定値を持たせず、評価adapterの初期化方式も含め、呼び出し側がHydra compositionの値を明示する。
初期3Dを固定した場合、solver内部の残差行と自由clock列はcamera ID順に統一し、camera依存の
出力・共分散・診断は呼び出し側のview順へ戻す。終了閾値を変えず、浮動小数の順序依存を除く。

`segments.py` はbaseline 3Dの各点の左右で独立な2次式をfitし、速度変化の閾値とNMSで
イベント候補を選ぶ。低い位置で下向きから上向きになればbounce、他はhit候補とする。
正解イベントとは保証しない。長い欠測は分割し、最大区間長による切断はイベントに数えない。
短い欠測を内部fitが跨いでも、出力のvalid・使用viewはbaselineの実観測支持を超えない。

`fit.py` は疎な有界least squaresとsoft-L1で全変数を同時推定する。候補点形式は
距離²/sigma²−2log(score)が最小の有効候補を選び、XY残差とscore罰則をfitする。
heatmap形式はnative mapのbilinear尤度をpeakで正規化し、−2log尤度の平方根と
画像外への連続的罰則をfitする。各観測4cellだけを読み、大きなmapを複製しない。
RK4で積分し、検出器・GPU・動画・正解を使わない。

offsetは秒で、cameraのframe fは `X(f/fps+offset)` を観測する。
**保持camera** の辞書順先頭を0に固定し、他は絶対値1frame未満とする。
区間両端各1frameをfitだけから除き、offsetがイベントの不連続を跨がないようにする。
出力はframe時刻。区間間の位置連続性やbounce法則を強制せず、位置差と鉛直反発率を診断へ保存する。

任意componentは通常の `ball` fieldに加えてfitと診断を保存する。
[評価adapter](../evaluation/ball3d/segment_physics.py)も同じsolverを呼び、offsetを基盤の結果形式へ渡す。
LOVOは保持cameraだけで再fitし、基盤の採点では除外cameraのoffsetは0のままである。
foldごとに相対時刻の基準が異なり得る。絶対同期は識別できない。

診断は区間・イベント・画像残差・物理加速度・状態・drag・offset・収束状態・情報行列のrankを含む。
full rankの場合だけ局所曲率に基づくoffset共分散と標準誤差を保存する。
rank不足は利用不可の理由、境界到達はflagを保存する。分割と候補選択を条件とした近似で、較正済み信頼区間ではない。
未収束は明示的状態と無効出力を返し、baselineへ黙って戻らない。
平坦なmap、短い飛行、校正誤差は失敗要因。Magnus・rolling shutter・校正変更は対象外。

clip固有の実験とテスト設計・実測は
[`knowledge/runs/run-i1565-segment-physics-offset-20261103`](../../../knowledge/runs/run-i1565-segment-physics-offset-20261103/) を参照。
