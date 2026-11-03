# 候補GMMによるball再構成

[English](README.md) · [简体中文](README.zh-CN.md)

標準pipelineで `ball_reconstruction.method=candidate_gmm` を指定します。
既定は `triangulation` のままです。ball再構成だけが `ball_detections` v2の
保存済み証拠を読み、side/alignmentは従来のball pointsを使います。CPU処理です。
設定の正本は `CandidateGMMConfig` と `configs/pipeline.yaml`。
既定値はHydraのYAML compositionだけに置き、Pythonの設定と評価adapterは
replay scriptも含め、設定の明示的な受け渡しを要求します。
`ball_reconstruction.candidate_gmm` の `estimator=map|mean` と
`temporal=none|kalman|rts` を選べます。較正済み `covariance_scale`・
`presence_slope`・`presence_bias` は明示指定し、productionは正解を読みません。
clip実験と前半だけの較正は
[`knowledge/runs/run-ball3d-candidate-gmm-20261103`](../../../knowledge/runs/run-ball3d-candidate-gmm-20261103/) にあります。

## 分布とmask

camera/frame別にscore順で候補を調べ、3×3 log確率の二次fitがfull rankかつ
正のprecisionとなる最初のK=2〜3候補を使います。交差項を保持し、subpixel最適点を
native cell内に制限します。共分散の固有値は `min_variance_cells2` / `max_variance_cells2`
で明示的に制限し、制限・棄却件数を保存します。端の支持不足・平坦・鞍点は棄却し、
等方共分散へ戻しません。平均と共分散を `(W-1)/(w-1)`・`(H-1)/(h-1)` で
元画素へ変換します。dense heatmap全体を複製しません。

成分重みはpeak scoreの正規化です。存在の代理値は
`sigmoid(presence_slope*logit(max retained score)+presence_bias)`、score端点は
[1e-7,1-1e-7]に制限します。有効peakなしは0です。利用可能な候補支持の代理で、
物理的な球の存在や遮蔽を識別した確率ではありません。

既存の[方式A](../../utils/geometry/probabilistic_triangulation/README.md)を使います。
コートm単位のpriorは `prior_mean_m`（既定[0,0,2]）・`prior_std_m`（[8,16,6]、
対角共分散）で必須です。列挙は最悪 `(K+1)^V`、予算を超えれば計算前に停止します。
最適化失敗は理由付きでframe全体を欠測にし、失敗成分だけの破棄・別solverへの切替はしません。
契約・予算エラーは例外です。

2camera以上がactiveな事後分布へ条件付け、その元の確率質量が
`min_multiview_mass` 以上のときだけ三角測量点を出します。質量を保存し、
priorだけ・単眼だけの項を観測に数えません。`map` は最大重み成分のMAPで、
混合全体のmodeではありません。`mean` は成分間分散を含む混合momentです。
MAPの共分散は選択成分内の局所値です。使用viewはMAPではその成分のsubset、
meanでは正の重みを持つsubsetの和集合です。

共通の等速度Kalman/RTSは秒・m・加速度/初速scaleを使い、欠測でstateをresetします。
valid/使用view maskを変えず、欠測の予測点やイベント注釈を作りません。
バウンド・打球を丸める可能性は残ります。

## 出力と評価

`CandidateGMMOutput` は従来の `ball` と、MAP/mean・選択共分散・支持確率・
候補index/平均/全共分散/重み・曲率診断・solver失敗を保持します。
2D診断点は出力の投影に最も近い採用候補平均で、距離を追加gateには使いません。
scene exportは `ball`、詳細な共分散はcomponent artifactと評価結果に残します。

評価adapterも同じproduction関数を呼び、正解・正解mask・除外viewを入力しません。
較正はfoldの保持viewだけの[0,505)で行い、主評価前に固定します。
共分散scaleは20px以内の局在候補、score関数は採点可能な前半点からfitします。
共通採点は再投影誤差・可用性coverageを出します。`score_uncertainty` はその採点点と
出自検証を再利用し、50/90/95%楕円の被覆率を区間/fold/観測・推定別に出します。
投影はpinhole Jacobianによるposterior共分散だけで、参照ノイズ・校正誤差・
非線形tail・mode選択誤りを含みません。非ゼロ時刻offsetは明示拒否します。
被覆率は採点できた予測に条件付きで、予測可用率とは異なります。
共通rendererは共分散があるとcamera/top/sideへ95%楕円を描き、HTMLは追加図を埋め込めます。
合成unit/integrationで曲率・camera順・欠測・較正の隔離・投影・平滑化・設定/codec・
main固定配列との既定回帰を検証します。
