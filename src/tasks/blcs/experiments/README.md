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

GT自己評価、位置ジッター・時間ずれ・イベント越境の摂動、同じ観測からの三角測量＋線形補間を対照にする。幾何対照は同じ2D CourtKP14から共有planar fitterでカメラを推定し、保存カメラGTを入力しない。校正・三角測量の支持不足clipは除外理由と母数を保存し、成功clipだけのスコアを全sceneのモデル結果と同一視しない。重い共有ODE fittingは保存評価で固定支持集合を使う。Bの積分出力はGT区間（oracle）と予測イベント区間を区別する。イベントは±2frame以内の一対一対応数を最大化し、その中で時刻差を最小化する。実観測はMeiji固定clipに対して再投影・物理診断・軌道図を作るが、独立3D GTがないため実3D改善とは解釈しない。

GPU実行は共有training queue経由。最初のposition runは`all`、計測したVRAM使用量に基づき残り9 runは`half`（最大2並列）とする。更新数・batch・学習条件は同一だが、所要時間は競合の影響を受けるため速度性能の比較には使わない。6条件の終了後に`repeat_selected --select-only`を`all`で実行してvalidationによる選択を保存する。10 run後の`final_evaluation`も`all`とし、checkpointのSHA・更新数を確認してからtestを初めて評価する。

実行時の補足: positionとreprojectionの初回runはCUDA unknown errorで途中停止した。失敗ログと途中checkpointを別の出力先に保持し、同じseed・条件で最初から4,000更新を`half`予約でやり直す。途中のrunは比較対象10件へ混ぜない。

全長評価・保存値の再計算・実観測診断・図表作成は `python -m src.tasks.blcs.experiments.scripts.final_evaluation --data-root <data> --historical-checkpoint <ckpt> --real-store <Meiji store>`。各runの`selection.json`と`additional-seed-selection.json`を入力とし、出力は`outputs/blcs/i1044/{evaluations,real,report}`。`pred_test.npz`はモデルなしで`evaluation.recompute(path)`により指標を再計算できる。実観測は同じ保存2D注釈を2frameごとに使用し、コート点は保存校正から生成する。このため生検出器入力のend-to-end評価ではなく、条件を固定した観測診断である。

Bの積分位置・速度をともに保存し、`evaluation.recompute(path, include_diagnostics=True)`でGT区間／予測区間別の指標とイベント診断も再計算する。保存済みの積分位置・速度から指標を再計算するもので、積分器自体は再実行しない。実観測は投影行列も保存し、`real_observations.recompute_real(directory)`により元のstoreやcheckpointなしで診断を再計算できる。一括評価では、これらの値・支持数・未定義状態が保存JSONと一致することを確認する。

10runと最終評価の結果・退行・事後診断は[最終比較](../../../../knowledge/nodes/blcs/000068-group-i1044-progressive-losses-u4000.md)を参照。ここに定めた評価窓・支持集合・方式選択は結果を見て変更していない。

一次資料: [TT3D](https://arxiv.org/html/2504.10035v2)は空力とバウンド状態の最適化による再構成、[APHYNITY](https://arxiv.org/html/2010.04456v6)は微分可能な積分軌道による物理と学習補正の学習を示す。短horizonのBLCS損失はそれらから着想した本実験の仮説。[PINNsの収束分析](https://arxiv.org/abs/2007.14527)を踏まえ項ごとの尺度・warmup・勾配到達を検証する。各論文の別問題での成果をBLCSの改善保証とはしない。

実行入口は `experiments.scripts.{issue1044,evaluation,repeat_selected,final_evaluation}` に置き、入力は絶対パス、学習outputはcheckoutのoutputs下の相対パスとして検証する。旧入口は初期runの再現commitにのみ残る。学習の演算・設定は移設時にも変更していない。最終評価直前にsnapshotの全ファイルと既存manifestのSHA-256を再照合し、変更があれば停止する。
