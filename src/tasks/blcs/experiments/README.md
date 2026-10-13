# Issue #1044 比較プロトコル（学習前固定）

2026-10-13のユーザー指定: 各4,000 optimizer updates、計10 run。採用閾値は設けず、結果を観測してから判断する。方式採用・本番checkpoint置換は行わない。

評価軸は全有効frameの3D軌道RMSE[m]を主軸とし、平均/p95距離、clip別RMSE、速度head RMSE[m/s]、位置の同じ差分演算による速度/加速度誤差、共有積分器の残差[m,m/s]、clean観測の再投影[px]と評価可能率、地面下率を併記する。欠損/観測、イベント±2frame/飛行、view数、surface、clip長、バウンド数、欠損区間長を層別する。支持数0は未定義（保存JSONではnull）。batch平均は使わない。

固定snapshotは `data/blcs/single_object-physics-v3-cc863cacd-20261007`。既存runのmanifestは `knowledge/runs/run-i1032-blcs-physics-v3-e200-s42/dataset_files.jsonl.gz`。train/validation/testは800/100/100 scene。trainは128frame以上の715 sceneを従来どおりランダム128frame cropで学習。validationは固定中心128frame（短いsceneは全長）で全100 scene、カメラ抽選の評価seedは42固定。最終保存評価は全100 sceneの全frameを固定128frame窓で推論し、連結後に計測する。従来の88 scene中心窓の値とは直接比較しない。

全条件でaxial base、xyz/velocity head、3–6 view、batch2×accum8、AdamW LR1e-4・WD0.01、200 update warmup→4,000 update cosine、bf16、GAN/compileなし。checkpoint選定は同じ `val/trajectory_rmse_m` 最小。全runのtestは選定後のみ。

| arm | 前段への追加 |
|---|---|
| position | 位置教師のみ |
| reprojection | ＋clean再投影（既存v3学習戦略の主対照） |
| velocity | ＋sim速度教師 |
| rollout_position | ＋GT場/spinによる短積分位置整合 |
| rollout_velocity | ＋同じ積分の速度整合（候補A） |
| parameters | ＋最小drag/wind/有効Magnus/event headと教師・再構成・整合（候補B） |

最初に6条件×seed42、その後reprojectionとvalidation位置RMSE最良の新方式をseed43/44で追加する（計10）。これは追加seed対象の選択であり採用判定ではない。候補Bはアーキテクチャ全体の比較で、B内部の各auxiliary lossの独立効果はこの予算では切り分けない。全条件に同じ速度headを置き、その有効/無効によるモデル容量の交絡を避ける。旧9,000更新checkpointは更新数・head構成の異なる参考対照とする。

位置損失は物理mに戻し、scale=beta=11.885 mで無次元化するため旧標準Smooth L1と同値。再投影は各軸を画像幅/高さで正規化したSmooth L1（beta=1画像幅/高さ）、重み0.1。速度はscale30 m/s・beta10 m/s、重み0.1。rolloutはhorizon1/3frame、記録dt×strideを使用（このsnapshotでは約0.0333/0.1秒）。両端に勾配を流し、200更新後から400更新かけて物理項を線形増加する。重みと尺度の正本は `configs/loss/default.yaml` と `issue1044.py`。

GT自己評価、位置ジッター・時間ずれ・イベント越境の摂動、同じ観測からの三角測量＋線形補間を対照にする。重い共有ODE fittingは保存評価で固定支持集合を使う。Bの積分出力はGT区間（oracle）と予測イベント区間を区別する。実観測はMeiji固定clipに対して再投影・物理診断・軌道図を作るが、独立3D GTがないため実3D改善とは解釈しない。

一次資料: [TT3D](https://arxiv.org/html/2504.10035v2)は空力とバウンド状態の最適化による再構成、[APHYNITY](https://arxiv.org/html/2010.04456v6)は微分可能な積分軌道による物理と学習補正の学習を示す。短horizonのBLCS損失はそれらから着想した本実験の仮説。[PINNsの収束分析](https://arxiv.org/abs/2007.14527)を踏まえ項ごとの尺度・warmup・勾配到達を検証する。各論文の別問題での成果をBLCSの改善保証とはしない。
