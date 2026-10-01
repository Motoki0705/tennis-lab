# Ball Refiner

単眼の2D分布推定（#935）と、その分布を使う3D軌道推定（#936）のタスク。
出力先は[タスク出力規約](../OUTPUTS.md)に従う。

## 実装の境界

- `refiner_2d/`: cameraごとに独立して、検出証拠・COCO17の肘/手首・静的court・時間窓から
  1球の位置分布と存在確率を推定する。GMMの各成分は同じ球の位置仮説で、複数球ではない。
- データ生成・評価はこのtask直下へ置き、#936と共有する。モデル入力には他camera、
  camera ID、正解座標、三角測量を渡さない。検出器の時間的な参照範囲は
  [証拠cache](#検出器の局所証拠)へ記録し、refinerのattention窓長と区別する。
- 文脈なしの基準学習は[学習pilot](#文脈なし学習pilot)から実行する。
  [専用pipeline recipe](../../tennis_scene/pipeline/README.md#2d-ball-refinerの専用recipe)は
  未較正の文脈なしpilotを明示的に実行・保存する。文脈あり学習・最終holdout評価は後続PRで実装する。
  標準pipelineの点consumerは同じconfidence規則を通したrefiner点へ接続済み。最終的な#936の入力はrefinerの全分布のみとし、
  detectorの点推定へ戻す経路は設けない。court_sideの幾何的な仮説検定は別の利用者である。

## 2DモデルのAPI

| ファイル | 責務 |
|---|---|
| `configs/model/refiner_2d.yaml` | モデル・分散範囲・dropout・ablation設定の既定値の正本 |
| `refiner_2d/config.py` | 完全な設定を要求するtyped契約と意味検証 |
| `refiner_2d/contracts.py` | 入力と教師のtyped契約 |
| `refiner_2d/model_io.py` | float32入力の検証、mask処理、MDNの復号、model/adapterの構築 |
| `refiner_2d/model.py` | 計算だけのforward。候補集合→時間→文脈→時間→MDN |
| `refiner_2d/distribution.py` | GMM検証、条件付き密度、source画素への平均/共分散変換 |
| `refiner_2d/loss.py` | 既知frameの重み付きjoint NLLとepoch集計用の和・分母 |
| `data/inputs.py` | 教師なし入力の切出し、人物軸だけのcollate、device転送 |
| `data/temporal.py` | 実frameだけの窓と中心距離による採用規則 |
| `inference.py` | camera全frameのGMM推論と、各frameを採用した窓の出自 |

`configs/model/refiner_2d.yaml`を合成して全fieldを`Refiner2DConfig(**values)`へ渡す。
省略値をPython側で補完しない。既定のcourt軸はpipelineのcamera-local KP14に合わせる。
`build_ball_refiner_2d(config)`は共通の`BoundModelIO`を返す。
`pair.run(Refiner2DInput(...))`で検証→forward→復号し、
`refiner_2d_nll(prediction, Refiner2DTarget(...)).loss.backward()`で学習できる。
GPUへ移す場合はmodelと全入力tensorを同じdeviceへ明示的に配置する。
入力は全て実frameで、時間paddingは受け付けない（教師maskで時間attentionのpaddingは代用できない）。
候補は`BallCandidates`、poseのjoint軸はCOCO17の`[7,8,9,10]`、
courtはframe 0の`(B,C,2)`、時刻はclip内の実秒`(B,T)`。
pose/courtの画像外座標は有限値なら保持する。無効なcontext座標はmaskで除外する。
検出器証拠が未生成な場合と、生成済みだが候補0件（全valid=false）は呼び出し側で区別する。

`BallGMM2D`のconsumer向け契約は以下。Bの各行は独立した1camera-window、
Tは元frame、Kは同じ球の代替位置仮説である。

| property | shape | 意味 |
|---|---|---|
| `means` | B,T,K,2 | x/(W−1), y/(H−1)、[0,1] |
| `covariance` | B,T,K,2,2 | 正規化uv²の対称正定値行列（相関あり） |
| `weights` | B,T,K | 非負、成分軸の和が1、存在を条件とする重み |
| `presence_probability` | B,T | [0,1]、画面内amodal存在 |

学習用に同じobjectが`scale_tril`・`mixture_logits`・`presence_logits`を保持し、
確率に丸めてからlogを取らない。`log_prob(uv)`は存在項を含まない条件付き密度。
`pixel_moments(source_size_wh)`はcameraごとの`(B,2)`サイズを受け、
平均をD倍、共分散をDΣDᵀへ変換する（D=diag(W−1,H−1)）。
pixel log densityはuv log densityからlog((W−1)(H−1))を引く。
モデル出力はAMP中もfloat32へ復号する。GMMのGaussian tailは画面外にも残る。
極端なlogitから確率0/1が得られても学習ではlogitを維持する。

`predict_sequence(pair, inputs, window_length=..., stride=..., batch_size=..., device=...)`は
1cameraの全CPU入力を受け取り、教師・store・学習runを参照せず推論する。
modelを呼び出し側でdeviceへ配置し、入力batchだけを順に転送する。pose/courtを使う場合は
生成済みの入力を明示的に渡す。`detector_only_input`は文脈無効の設定だけを受け付ける。
短いclipの時間paddingや検出有無によるframe選別はせず、欠損frameにも全GMMを返す。
重複窓は中心に最も近いもの、同点なら早い開始位置から全成分と存在logitをまとめて採用する。
結果の`SequencePrediction.distribution`はCPUの`BallGMM2D`、`window_start/time_index/window_length`は
各frameの採用窓を示す。混合成分を平均せず、推論前のmodelのtrain/eval状態を終了・例外時に復元する。
学習時のvalidationもこの共通経路を使う。pipeline登録・永続保存は上記の専用recipeを参照。

### 候補残差平均と絶対平均の明示的な設定

既存の全fieldだけを持つ`Refiner2DConfig`は絶対uv回帰のまま保持する。
`parse_model_config`はこれに`mean_parameterization: candidate_residual_v1`、
`anchored_components`、`max_offset_uv`を**全て明示した別schema**も受け付ける。
省略補完せず、未知の方式・一部だけの追加設定・候補軸不足はエラー。
既存checkpoint/bundleのconfigや既定YAMLは変更しない。

この方式はscore上位候補に、`max_offset_uv * tanh(raw)`という学習可能な残差を加える。
内部の同score成分割当だけはy→xの昇順で一意化し、候補集合の並べ替えに依存させない。
detectorのtop-1/recall集計とcheckpoint選択の既存同率規則には影響しない。
平均は[0,1]内へclipし、既存decoderへ渡すlogitの有限性に限り1e-6の端点余裕を取る。
該当候補が無効なframeは、その成分の学習した絶対uvを使う明示的な欠損分岐とする。
全gapでも全成分を返し、少なくとも1成分は全frameで自由な絶対uvとする。
候補基準成分の平均headのみweightをゼロ初期化し、初期残差を0にする。
共分散・weight・presenceの学習、BallGMM2Dの保存契約、joint NLLは共通。
実験数値はknowledgeを正本とする。絶対平均headは旧checkpointの復元と対照実験で
使用しているため、明示的な選択肢として保持する。

### 採用設計: 候補残差head（anchored_12k）

**[2026-09-30のユーザー判断A–D](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5908081470)により採用。**
上記の方式をscore上位3候補＋有界残差（`max_offset_uv=0.02`）と1自由成分、
K=4、文脈なし、12,000更新の学習予算でball refinerの基準設計とする。
検出器は既に候補recallで固定したepoch9を使い、33frame/stride16、
source平衡・joint NLL・Meiji選択側のobserved/gap等重みNLLによるcheckpoint選択を維持する。
12,000更新の最終checkpointを無条件採用する意味ではない。
欠損時の絶対平均分岐・自由成分・full covariance・presenceとBallGMM2D契約も上記のまま。
seed42に加えて43・44で再現性を確認し、再現しない条件も報告する。

根拠・反証・較正の限界は[knowledge 000018](../../../knowledge/nodes/ball_refiner/000018-run-i935-precision-variants-s42-r21-20260930.md)、
採用判断の正本は上記issueコメントを参照。
共分散だけを倍率で補正し、平均・混合重み・存在確率は固定する。clip単位の交差検証で
HDR50/90/95のcoverage、位置NLL、面積を併記し、明示したhash付き較正artifactを保存する。
#936へは補正後の全GMM残差bankを渡し、旧bank/合成dataは対照として保持する。
pipelineの既定値は[専用recipeのB判定](../../tennis_scene/pipeline/README.md#ボール経路の既定と明示option)に従う。
[2026-10-01の判断](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5921216642)で
e9の既定化とrefiner後のconfidence選別を採用する方針へ更新した（標準sceneへの既定切替は実装済み）。
#964完了前に文脈生成へ進める。現在の生成予算・未完事項は
[knowledge 000027](../../../knowledge/nodes/ball_refiner/000027-run-i935-context-budget-r28-20261001.md)を参照。

`refiner_2d/calibration.py`はcheckpoint SHA256とartifact SHA256を必須とする明示的な読込API。
`CovarianceCalibration.apply()`はΣをs倍（Choleskyを√s倍）し、全成分の平均とlogitを保持する。
artifactの自動探索・倍率1への省略補完はしない。pipelineの明示的な較正optionと保存schemaは
[専用recipe](../../tennis_scene/pipeline/README.md#2d-ball-refinerの専用recipe)を参照。
`evaluation/calibration_fit.py`が位置NLLのfit、`evaluation/covariance_calibration.py`が
保存済みval出力のhash/教師/PTS照合・clip交差検証・層別比較を担当する。
calibration halfでは全cameraをまとめたleave-one-clip-out、その他のvalでは
calibration half全体でfitした倍率を使う。checkpoint選択済みhalfを独立評価と呼ばない。
出力の`bank_input/`は配布用倍率をfitしたframeにも適用する明示的なin-sample bank材料で、
OOF評価NPZとは別に保存する。実測と採用判断はknowledgeに記録する。

```bash
# CPU専用。inputはrun_cached_comparisonのcomplete出力。
# bounds/gridは実験前に宣言し、artifactは元のcheckpointと同じdirectoryへ新規保存する。
.venv/bin/python -m src.tasks.ball_refiner.scripts.calibrate_covariance \
  --predictions <絶対保存済みval-directory> --output <絶対新規output-directory> \
  --calibration-artifact <絶対checkpoint-directory>/covariance-calibration.json \
  --bounds 0.25 64 --grid-points 129 --cpu-threads 2
```

教師の`weight`はjoint項に共通のframe重み。lossは位置NLLの和と存在BCEの和を足し、
存在既知frameの重みの和で割る。位置の条件付きNLLを報告するときは
`position_nll_sum / position_weight`を使い、位置教師0件ならN/Aとする。
位置なしのframeに仮のuvで密度を評価せず、全教師なしbatchはエラーにする。
検証例は[unit](../../../tests/unit/tasks/ball_refiner/refiner_2d)と
[integration](../../../tests/integration/tasks/ball_refiner/test_refiner_2d.py)を参照。

## 点consumerの信頼度規則

`refiner_2d/confidence.py` は最大weight成分の平均点と、全GMMの2次モーメントから
90%以上を条件付きで含む保守的な楕円面積をsource px²で計算する（HDRではない）。
存在確率と面積による規則は [固定設定](configs/confidence/meiji_val_r29.yaml)、
Meiji valの選定方法・camera別保持率/誤差・多視点frame数・限界は
[knowledge 000028](../../../knowledge/nodes/ball_refiner/000028-run-i935-confidence-r29-20261001.md) が正本。
clip_000を選定対象に含めず、実行時の再fitや教師参照はしない。

## 学習戦略（#935）

文脈なし基準と較正・追加seedの方針は上記の採用設計に従う。
以下の文脈・RGB遮蔽・amodal追加実験は未完了の計画であり、採用済み性能とは扱わない。
決定・変更履歴は[#935](https://github.com/Motoki0705/tennis-lab/issues/935)の
【要判断】コメントを正本とする。以下は実験前の計画で、性能結果ではない。

### 検証する仮説

[SLCSのcourt文脈追加](../../../knowledge/nodes/slcs/000071-run-slcs-full-real-rgb-missing-ball-court-val-v1.md)
では欠損境界と平均位置誤差が悪化した。文脈を足すだけでは改善しない。
[#934のholdout](../../../knowledge/nodes/ball_detection/000022-run-i934-meiji-holdout-e0-r7.md)
ではtop-Kに正解が残る余地と、大誤検出が残る問題を確認した。
検証するのは「時間的な候補選択を基準に、選択的なpose/court融合が欠損時の尤度と
カバー率を改善し、観測時の裾誤差を悪化させないか」である。
ゼロ初期化gateは初期の文脈なしモデルとの一致を保証するだけで、学習後の退行を防ぐ保証ではない。

### 教師と存在の意味

存在は「このcameraの画面内にプレー中の球の中心がある（遮蔽中も含む）」とする。
検出器のscoreやvisibleとは別の確率である。球があるか不明なframeを負例にしない。
位置の教師maskと存在の教師maskを独立に持ち、単一球と確定できないframeは教師から除外する。

| 出どころ・状態 | 位置教師 | 存在教師 | 役割 |
|---|---|---|---|
| Meiji / chat / TrackNetのobserved | 正規化uv | 1 | 主学習・主評価 |
| 明示的なout_of_frame | なし | 0 | 存在学習。画面内位置NLLなし |
| 未レビュー / unresolved / instanceなし | なし | 原則なし | 不可視と不存在を同一視しない |
| interpolated / occlusion_estimated | 主学習ではなし | 主学習ではなし | 独立した参考評価。observedに混ぜない |
| observedへ人工遮蔽を適用 | 元のuv | 1 | 位置既知の遮蔽対照 |
| 他viewから品質検査済み再投影 | 疑似uv、品質に応じたweight | 1 | amodal追加実験、主評価と区別 |
| BLCS物理軌道の2D投影 | 合成uv | 投影内外で1/0 | 前学習の別ablation |

「注釈済みinstanceなし」はstoreでは可視球なしを意味し、amodal不存在を保証しない。
sourceで明示的な不在と証明できたものだけを後の生成処理で負例へ昇格させる。
負例の少なさは件数とともに報告し、Meijiの正例だけから存在較正の良否を結論しない。
元のpoint_kind・event・segment_breakは
[ball frame store](../ball_detection/README.md)の意味を保持する。

### splitと生成の漏洩防止

`ball-mix-v1`のvideo/group splitを変更しない。
Meijiはtrain video_002 / val video_000 / test video_001で、#934・#936と共通。
同じ収録の別camera・overlap窓を別splitに入れない。testは最終設定固定後の一回比較とし、
既に#934で見たtestの再利用であることも明示する。

detectorは候補recallによるvalidation選択で混合FT epoch9に固定済み。
ft-e13は歴史的対照として明示optionに残す。checkpoint、前処理、動画/注釈hash、frame/PTS、sourceサイズ、
窓集約、候補設定をcache manifestへ保存する。検出証拠は
[#934の契約](../ball_detection/README.md#検出証拠の出力契約)を使い、score閾値やtrajectory gateで捨てない。

poseはperson_detection → tracking → ViTPoseのcamera内COCO17から左右肘・左右手首を使う。
track IDを特徴にせず、可変人数の集合として扱う。欠落はmask、失敗・未実行はmanifestの状態として
区別し、ゼロ座標を観測とみなさない。courtはそのcameraのframe 0から推定したKPだけを固定する。
既存成果物の再利用にはmedia hash・frame/PTS・座標系の照合を必須にし、未整合なcacheは停止する。
RGB遮蔽対照ではpose/courtにも同じ入力映像条件を適用し、非遮蔽映像からの情報混入を防ぐ。

疑似amodal GTは、対象cameraを除く**2台以上**の同期したobserved点と校正済みcameraから作る。
1台だけから深さを捏造しない。正のdepth、ray angle、再投影残差、同期誤差を検査し、
閾値はtrain/valで固定する。教師の生成にrefinerの予測を使わず、教師用3Dを入力へ戻さない。
対象viewでobservedのframeを隠したleave-one-view-out検査で教師誤差と採用率を先に測る。
この擬似教師の評価は独立した人手amodal GTの精度とは呼ばない。

### モデルと損失

候補ごとのuv・score・局所probability patch・patch境界maskを埋め込み、
候補集合へのattentionから各frameのball tokenを作る。候補なしも学習可能なnull tokenで表す。
実時刻の差を入力し、camera内の時間attention、pose/courtへのcross-attention、
時間attentionの順に処理する。poseとcourtの残差gateは個別にゼロ初期化し、
pose dropoutはcamera-window単位で全poseを落とす。人数・trackの列順に依存させない。
offlineの双方向窓なので因果推論・real-time性能は主張しない。

MDNはK成分の平均・2×2正定値共分散・混合重みと、独立した存在logitを出す。
平均は画面の正規化uv、GaussianはR²上の密度とし、境界で切断・再正規化しない。
Cholesky因子の対角を正にし、数値下限を設定で固定する。
NLLはlogsumexpで混合し、存在はBCEWithLogitsで学習する。
存在が既知の全frameでBernoulli項、位置既知の正例だけで条件付き位置項を計算する。
未知frame・paddingは損失から除く。疑似教師のweightは別集計し、存在確率で位置lossを弱めない。
全教師なしbatchは明示的に拒否する。

### 実験の順序と予算

初期pilotのseed・実窓長・source・AdamW・batch・epoch/更新数の上限は
[Hydra設定](configs/train.yaml)を正本とする。epochは指定数のsource等比率再標本化を意味し、
全窓を一度通過する単位ではない。epoch上限と最大更新数の先に達した方で停止する。
checkpointはMeiji validationの観測/人工遮蔽を等重みとした位置NLLで選び、
存在BCEと両層のp95/coverageも確認する。較正用val clipは選択用とclip単位で分け、
testでcheckpoint・温度・分散scaleを選び直さない。

1. CPUの契約・gradient・欠損・camera独立性を確認し、固定cacheの小規模GPU pilotをqueueで実行。
2. detector単体、文脈なしの時間MDN、fullを同じデータ・初期値seed・更新数で比較。
3. 同予算でposeなし、文脈のみ（全detector証拠を無効化）、pose摂動を比較。
   pose摂動はsource画素で20px相当の座標noiseと、窓内で8frameずらす診断を別々に報告する。
   full学習済み重みの入力除去と、再学習ablationを混同しない。
4. artificial gapは1/4/8/16frameを固定manifestで共有する。証拠dropoutは安価なpilotだが、
   [TOTNet](https://arxiv.org/html/2508.09650v1#S3.S2)型のRGB遮蔽とは呼ばない。
   RGB上の球領域maskを時間区間へ適用し、凍結detectorの証拠を再生成する実験を別に行う。
   同じRGB増強・教師・予算のRGB-only detector再学習対照も用意し、文脈だけの効果と切り分ける。
5. amodal疑似教師とBLCS投影前学習はそれぞれ追加ablation。
   合成の完全軌道からheatmapを逆生成しただけの入力を実検出と扱わない。
   合成pretrainを使う場合はreal trainで測った誤検出・欠損分布に合わせ、real holdoutで採否を決める。

GPU学習・証拠生成は共有training queueを使い、jobと実験設定をissueに残す。
pilot後の追加seed（43,44）や予算拡大は、validationで候補を固定してから別runとして記録する。

### 評価・較正と採否

全source frameを一度だけ採点する。cameraごとに元時刻順の窓を作り、
重複frameは窓中心への距離最小、同点なら先の開始frameを採用する計画とする。
出自を保存し、窓をまたいでGMM成分を平均したりGTを見て候補を選んだりしない。

- observed: 条件付きGMMの最高weight成分の平均を固定の点要約とし、全frameのsource px誤差
  mean/median/p95、20px recall、存在閾値による欠損数を出す。最大weight成分の選択はMAPの近似である。
  受理frameだけのp95も出すが、母数の違う比較と明記する。
- 遮蔽: 人工遮蔽・実occlusion_estimated・品質検査済みpseudo-amodalを分離して、
  条件付き位置NLL、50/90/95%の混合分布の最高密度領域のcoverage、領域面積、
  mixture全体の分散（成分間の分散も含む）を測る。欠損が長いほど分散が広がるか検査する。
  coverageは混合からの固定seed Monte Carloで密度閾値を推定し、
  1成分の楕円の和をGMMの厳密な信頼領域とは扱わない。
- 存在: 既知ラベルに対するBCE/Brier/reliabilityと母数・正負比を報告する。
  Meiji単独には確定負例が乏しく、不存在の較正は他sourceの固定holdoutも必要。
- detector単体: 点誤差には元のargmaxを使う。密度比較にはnative heatmapに明示した微小一様成分を
  混ぜた正規化分布を使い、空/平坦mapでも定義する。温度・一様成分率はvalidationで固定する。
  これは未較正scoreを存在確率とみなす操作とは別で、検出器の元scoreも併記する。
- NLLはuv²単位とsource px²単位を明記する。camera、gap長、point_kind、pose有無で層別し、
  clipを再標本化するpaired bootstrap 95%区間を出す。同じ収録内の相関は残るため、
  frameを独立とした信頼区間や単一test videoから他会場への汎化を主張しない。

fullが文脈なしより遮蔽NLL/coverageを改善し、observed p95のpaired区間が退行を示さず、
存在が崩壊しない場合に次段への候補とする。満たさなければ負の結果とともに基準を維持し、
testへ適応した再選択はしない。較正によるcoverage改善と位置精度改善は別々に報告する。
実験結果はknowledge-controlで`ball_refiner`へ登録し、ここに指標表を重複保存しない。

## 教師と既存文脈の準備

`data/targets.py`はball frame storeの全frameを単一球の教師へ写す。
observedだけが位置教師、明示的out_of_frameだけが存在の負例で、理由コード別の件数を返す。
推定座標は参考評価用に保持するが位置maskを立てず、複数instanceのframeでは球を選ばない。
storeの縮小率を戻してsourceの`(W-1,H-1)`で正規化し、範囲外の教師を丸めない。
元のframe index・PTS・event・segment_breakも保持し、モデルの時刻は整数PTS差から作る。

`data/context.py`は明示されたpipelineの`scene.json`から、採用中のpose/courtだけを読む。
media hash・元解像度・全frame数・FPS・成果物checksum・現在の依存関係を照合する。
poseは同一mediaのdense frame indexによってstoreのPTSに束縛し、時刻の近傍照合はしない。
未生成の成果物は`None`と状態を返す。文脈必須の利用者は`require_complete()`で未生成を拒否する。
実行済みだが検出なしのmaskとは別である。入力不整合・破損はエラーになる。

poseにはCOCO17の肘・手首を使い、補間boxを観測としない。
ViTPoseのscoreは非負のheatmap peakで1を超えうるため、refinerの有界特徴へ
`min(score,1)`で写す。変換名・上限に達したslot数・元の最大値をprovenanceに記録する。
これは確率較正ではない。有限な画像外の関節位置は保持する。
courtは`camera_view_v2`のKP14、frame 0のみを受け付ける。

CPUの`data/audit.py`と次の入口で全source/splitの教師数とMeiji全cameraの文脈coverageを確認する。
出力先は新規ディレクトリを要求する。元動画/JPEGの再decode・再hashや推論は実施しない。

```bash
.venv/bin/python -m src.tasks.ball_refiner.scripts.audit_data \
  --store <絶対data-root>/ball_detection/ball-mix-v1 \
  --meiji-context-root <絶対artifact-root>/<生成run>/stores \
  --output <絶対output-root>/ball_refiner/analyze/data_audit/<run-id>
```

実データ監査の結果と生成不足の判断は[knowledge](../../../knowledge/nodes/ball_refiner/000001-run-i935-data-audit-r2.md)を参照。
文脈なしDataLoaderは下記のpilotへ接続する。文脈あり入力の未生成は補完しない。

### Meijiの凍結人物経路による文脈cache

`data/meiji_context_inference.py` は #964 の凍結人物設定・重み・共有実装を検証し、
同じJPEGの全画面DINO → `FeatureExtractor`（全17関節＋CLIP）→ `track_sequence` →
frame 0のcourt校正と共通court選別を呼ぶ。選別したgroupの実観測だけを保存し、
GSI補間をpose観測にしない。各frameの未検出・選別なし・court校正失敗を区別する。
負の値を含む有限な生heatmap peakを保持し、モデル入力への変換時だけ[0,1]にclipして
負値と飽和の件数を記録する。非有限値やmodel/runtime例外は失敗として停止する。

`scripts/meiji_context.py plan|generate` は既存cacheと別の入口。
Meiji train/video_002・val/video_000の108 camera-clip / 52,866 frameを固定する。
TrackNet/chatは計画と完成manifestで `absent_by_policy` とし、未生成を観測へ変換しない。
`--store --evidence --scene-config --freeze --output` は全て絶対path。
`plan` はCPUだけで新しいoutput内に `plan.json` を作り、hashを出力する。
`generate --plan-sha256 <hash>` は同じ引数を指定して共有queueで実行する。
CUDA allocatorは7GiB、モデルstageを分離し、外側の監視jobで全GPU使用量も制限する。

`data/meiji_context.py` がclip単位の完了receiptを保存する。再開時は計画・入力JPEG・
code・重み・完成NPZのhashとframe/PTSを照合する。失敗/中断clipは記録を保持し、
明示的な次回実行で新しいattemptへ全clipを再計算する。途中結果はreaderへ公開しない。
全clip完了後は同じ `ContextCache` 契約へ統合し、coverageと出自を付ける。
このcache作成は文脈モデルの学習・clip_000 scene qualificationを実行しない。
2系列の繰り返しcross-attentionと同予算の旧融合比較は
[ユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5913365577)に従う後続作業。

### 旧全sourceのJPEG文脈cache（BoT-SORT比較用）


`scripts/generate_context.py`はdetector cacheと**同一のJPEG shard**を使い、
TrackNet・Meiji・chat_annotationの各clipを独立に処理する。camera_id=Noneもそのまま保存し、
架空のcamera IDや動画を作らない。各clipのframe/PTS・stored/source寸法・元media/注釈hashを保持する。
元media/注釈hashはstoreの記録の継承であり、元動画を再decodeした証拠ではない。

| ファイル | 責務 |
|---|---|
| `data/cache_identity.py` | detector/context共通のstore hash・clip identity・source/split選択 |
| `data/context_inference.py` | 全画面DINO→BoT-SORT→ViTPose、frame 0のcourt推論 |
| `data/context_arrays.py` | 全frame・生COCO17 peak・観測mask・KP14とモデル座標への変換 |
| `data/context_cache.py` | detectorのJPEG hashへの束縛、immutableな生成・検証・読込 |

人物検出は全frameで全画面を処理する明示的な実験方針であり、court ROIがない場合の代替処理ではない。
標準sceneのROI方針は変更しない。観客・他court人物も含みうるため、pilotで人数と品質を監査する。
BoT-SORTのraw IDを保持し、scene用のtracklet結合・player選別は適用しない。既存のbox補間・平滑化を
cropに使うが、ViTPoseは実観測frameだけで実行する。累計track上限超は停止し、切捨て・ID再利用をしない。
`--max-tracks`は生成時のCPU配列の容量制限で、モデルの同時人数制限ではない。
`data/inputs.py`の窓切出しは、その窓で肘/手首がすべて無効な人物列だけを省く。
有効な関節が一つでもある人物は全て保持し、人数で順位付け・切捨てしない。
ViTPoseへのJPEG列入力は[submodulesのフレーム列API](../../submodules/README.md)を使う。
モデル入力はIDを含まない人物集合。生COCO17全17点を保存し、読込後に肘/手首4点を選ぶ。

courtは標準componentと同じ画像推論・領域探索をframe 0で実行する。支持される領域がない場合は
`CourtRegionUnavailable`の候補診断とall-false maskを記録する（実行済みの失敗）。
読込・画像・model/runtime例外は捕捉せず、cacheを完成扱いにしない。
全clipの人物検出/追跡frame数・pose crop数・court実行frameを保存し、未実行を検出0件に読み替えない。

cacheは`ball_refiner_context.v1`。既存出力への追記/上書き・自動resumeは拒否する。
clip完了ごとのNPZ checksum/進捗と、全clip終了後のJPEG/store/model/code再hashを保存する。
`status=complete`だけが読込可能。検出器の既存cacheとはmanifest hash・JPEG hash・frame/PTSを照合する。
この版は無加工JPEGだけを扱い、RGB遮蔽実験にはdetectorと文脈双方の別cacheが必要である。
`ContextCache(...).require_clips(...)`で学習対象の全clip被覆を要求し、pilot subsetをfull比較へ流用しない。
`load(...).arrays.model_context(...)`がstored画素/scaleをsourceのW−1/H−1で正規化する。
poseの画像外座標を保持し、raw peakの1超は上記と同じ明示的な変換・件数記録を使う。

モデル設定はsceneのYAML・strict adapterを共有する。`--scene-config`は
`src.tennis_scene.scripts.run_pipeline --cfg job --resolve`で出力した完全な設定を指定し、
checkpoint/external_asset等のrootは実環境の絶対pathへoverrideしておく。
生成はcourt/peopleだけを実行し、sceneの動画入力・三角測量・GVHMRは実行しない。
`--dry-run`はCPUで設定・資産hash・選択clipを検査し、モデルを構築せず結果を標準出力へ出す。
生成開始時と完了時も含め、DINO拡張は[submodulesのCPU事前検査](../../submodules/README.md#上流コードの扱い)を必須とし、
実際にimportしたバイナリのpath/hashを保存する。GPU実行の成否はこの事前検査だけでは確定しない。

```bash
# CUDAは共有queue経由。--clip-idを繰り返すと明示的なpilot subset、
# 省略するとdetector cacheの全clip（教師の有無では選ばない）。
.venv/bin/python -m src.tasks.ball_refiner.scripts.generate_context \
  --scene-config <絶対artifact-root>/scene-context.yaml \
  --store <絶対data-root>/ball_detection/ball-mix-v1 \
  --evidence <絶対cache-root>/ball_refiner/<detector-cache-id> \
  --output <絶対cache-root>/ball_refiner/<new-context-cache-id> \
  --max-tracks 64
```

文脈あり学習runner・bundle接続、全sourceの生成完了・精度/ablationは別途検証する。

#### clip単位の分割生成と統合

`data/context_shards.py`と`scripts/context_shards.py`で、同じ生成契約をclipごとのjobへ分割する。
`plan`はdetector cacheの**全clip**を元の順序で列挙し、store・detector manifest・JPEG identityと
モデル設定/重み/DINO拡張/生成codeのhashを`plan.json`へ固定する。実JPEGの再hashは各clip生成時と
全体統合時に行う。trackingはclipをまたがず、clip途中の分割・再開はしない。
planはモデルを構築しないCPU処理だが、実際のDINO拡張のCPU事前検査は実行する。

```bash
# 各modeとも--store/--evidence/--outputは必須で、全pathは絶対path。
# scene_configとDINO拡張は全job共通。生成中はcheckoutと資産を変更しない。
.venv/bin/python -m src.tasks.ball_refiner.scripts.context_shards plan \
  --store <store> --evidence <detector-cache> --output <new-plan-directory> \
  --scene-config <scene-context.yaml> --max-tracks <explicit-generation-cap>

# このmodeは共有queueで実行。indexはplanの0始まりの番号。
.venv/bin/python -m src.tasks.ball_refiner.scripts.context_shards generate \
  --store <store> --evidence <detector-cache> --output <new-shard-directory> \
  --plan <plan-directory>/plan.json --shard-index <index> \
  --scene-config <scene-context.yaml> --max-tracks <same-generation-cap>

# 全clipのjob成功後にCPUで実行。--shardを全clip分明示し、入力順は任意。
.venv/bin/python -m src.tasks.ball_refiner.scripts.context_shards merge \
  --store <store> --evidence <detector-cache> --output <new-complete-cache> \
  --plan <plan-directory>/plan.json --shard <shard-0> --shard <shard-1>
```

失敗した出力を保持し、新しい出力先へそのclipだけ再投入できる。統合には採用する成功pathを
明示し、最新runの自動選択や失敗clipの除外は行わない。旧pilotや別code/設定の混在、重複、
欠落、途中cache、checksum/frame/PTS/実行記録の不一致は停止する。
全NPZを再圧縮せず新規directoryへコピーし、入力cacheとinodeを共有しない。
終了時にJPEG・出力NPZ・入力manifest・計画・storeを再検査してから`complete`を公開する。
生成条件、計画hash、各成功shardのmanifest hashを保持した通常の`ContextCache`となり、
学習側のreaderを追加しない。母数は固定したdetector cacheで決まり、train/valの計画にtestは追加しない。

## 検出器の局所証拠

### validationによる検出器選定

`scripts/compare_detectors.py`は、[候補recallによる選定判断](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5886001394)
に従いft-e13・混合e0・混合e11を比較する。Meiji video_000のobserved frameに対する
recall@8（距離≤20 source px）を主指標とし、TrackNet game9・chat valとcamera別の値を報告する。
testを指定する入口はない。正解が単一のobserved球であるframeだけを分母とし、unknown・推定・複数球を混ぜない。
recall@1、候補外率、正解候補があるのにtop-1が誤りである率を同じ分母で出す。
同scoreはdecoder順のまま扱い、strict scoreで上回る率と同率による順位誤りの件数もJSONへ残す。
同率首位ではft-e13→e0→e11の順に現行/早いcheckpointを優先する。

比較は下記cacheと同じ`infer_clip_evidence`を呼ぶ。K=8/NMS=5/patch=5/subpixelと
窓規則・stride・画像サイズを`--cache-manifest`で固定し、checkpointのsha256を明示して照合する。
全val clipを1frame一度だけ採点し、各checkpointのsource座標候補/教師/mask/採用窓をNPZへ保存する。
checkpoint・store・JPEGの不変性を検証し、途中成果は`status=running`、全件成功後だけ`complete`とする。
これは比較成果物であり、新しい学習cacheを生成しない。pose/person/courtの入力も読まない。

```bash
# CUDAは共有queue経由。すべて絶対path、sha256はこの順に明示する。
.venv/bin/python -m src.tasks.ball_refiner.scripts.compare_detectors \
  --store <ball-store> --cache-manifest <existing-evidence>/manifest.json \
  --ft-e13 <ft-e13.ckpt> --mixed-e0 <epoch00.ckpt> --mixed-e11 <last.ckpt> \
  --expected-sha256 <ft-e13-sha256> <e0-sha256> <e11-sha256> \
  --output <new-comparison-directory> --device cuda --batch-size 2 \
  --cuda-allocator-limit-gib 6 --cpu-threads 4
```

`--dry-run`はモデルを作らずCPUでsplit/identityを検証する。allocator上限はPyTorchだけの制限であり、
CUDA context等のためにVRAM予算の余裕を別に確保する。OOM時のbatch/精度変更やCPU切替はしない。
`comparison.md`と`manifest.json`に表・hash・選定結果・e11>e0の判定を残す。
e11がe0を上回った場合は結果を報告し、全epoch保存による再学習は別判断とする。
今後のdetector学習は[毎epochの候補recallとcheckpoint保存](../ball_detection/README.md#毎epochの候補recallとcheckpoint)に従う。

### cache生成契約

`data/evidence_inference.py`はJPEGを逐次decodeし、各実frameのtop-Kとnative patchを返す。
重複窓は中心への距離が最小のもの、同点なら早い開始位置を採用し、
argmax・候補・patch・境界maskを同じ窓からまとめて保持する。
`data/evidence.py`はsource正規化uv、元frame/PTS、採用窓、native格子の整合を検証する。
短いclipをRGB反復で延長せず、明示的にエラーにする。

`data/evidence_cache.py`はcheckpointをstrict復元し、入力正規化・画像サイズ・候補設定を固定して
`ball_refiner_detector_evidence.v1`を生成する。全source/splitを明示的に指定し、
教師の有無ではframe/clipを選別しない。元storeは変更しない。
checkpoint、storeのmetadata/index、実際にdecodeしたJPEG shardを推論の前後でhashし、
元media/annotationのhashはstoreに記録された値として引き継ぐ。
生成codeのhash、frame/PTS・sourceサイズ・採用窓も保存する。
元動画/注釈ファイルを再hashした証拠とは扱わない。

clipの完了ごとにNPZのchecksumと進捗manifestを公開する。
全clipが完了して入力不変を再確認した後だけ`status=complete`になる。
既存出力への追記・上書きは拒否し、中断cacheは調査用に残す。
`EvidenceCache(directory, store).load(clip_id)`は、全clipの被覆、storeのhash、
各NPZのchecksum・座標単位・frame/PTS・実秒・patch格子を照合する。
未生成clip、破損、未完了cacheを空証拠へ置き換えない。

ft-e13の検出器は8frameを参照するため、33frameのrefiner入力が参照するRGBは
33frameを超えうる。`ClipEvidence.rgb_support(start, stop)`は採用された検出窓の和集合を
含む元RGBの半開区間を返す。このcacheはcamera-clip内だけで生成し、
group/split境界をまたがない。全比較条件で同じcache・RGB参照範囲を使う。
RGB遮蔽の再生成時にもこの出自を用い、遮蔽範囲と検出器への入力条件を明示する。

cacheは局所probability patchだけを保存し、dense heatmapを保存しない。
検出器単体の画素密度評価には別途dense heatmapの生成が必要になる。
pose/courtは`not_generated`と記録する。このcacheだけで文脈あり学習を行わず、
文脈なし基準では`use_pose=false, use_court=false`を明示する。

以下をworktreeから共有training queueへ投入する（パスは実環境の絶対pathを指定）。
結果を使う学習入口は次節を参照。

```bash
.venv/bin/python -m src.tasks.ball_refiner.scripts.generate_evidence \
  --store <絶対data-root>/ball_detection/ball-mix-v1 \
  --checkpoint <絶対checkpoint-root>/ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt \
  --output <絶対data-root>/ball_refiner/detector-ft-e13-v1 \
  --sources tracknet meiji chat_annotation --splits train val \
  --device cuda --stride 4 --batch-size 4 \
  --max-candidates 8 --nms-kernel 5 --patch-size 5 --subpixel-refine
```

## 文脈なし学習pilot

`scripts/train.py`はHydraの[train.yaml](configs/train.yaml)を厳密に検証する。
`use_detector=true, use_pose=false, use_court=false`だけを受け付け、未生成の文脈を
fullモデルの欠損観測に読み替えない。設定の省略・未知key・不正値は停止する。
role rootは絶対pathで指定し、`data.store`/`data.evidence`/`run.output_dir`は各root内の相対pathにする。

| ファイル | 責務 |
|---|---|
| `data/windows.py` | 教師の付与、学習窓の除外監査、source等比率sampling |
| `data/gaps.py` | 教師に依存しない証拠欠損とMeiji validationのcamera一括分割 |
| `training/configuration.py` | 型・意味・pathの検証と完全な実行設定 |
| `training/evaluation.py` | source frameごとのGMM復元、NLL/位置/存在/全混合分散の集計 |
| `training/runner.py` | cache/教師接続、学習、固定validation選択、epoch成果物保存 |

短clipは時間paddingせず除外し、全教師なしのtrain窓も除外して理由・母数を
`data_manifest.json`へ保存する。観測がないframeは窓内の入力として残る。
sourceごとの抽出数はepoch内で差1以内、各sourceでは教師のある窓を等確率で再標本化する。
train時は指定確率で実窓内の連続した候補・score・cell・patch・maskをまとめて消す。
教師とcache原本は変更しない。この増強はRGB遮蔽を再現せず、別frameのdetector証拠には
元RGBの球情報が残りうる。正式なRGB遮蔽比較は別生成・別実験で行う。

Meiji valは`meiji/video/clip`をseed付きhashで並べ、交互に選択用と較正用へ分割する。
同時刻の全cameraを同じ側へ固定し、入力順やラベルには依存しない。
選択用clipの全frameを中心距離規則で一度だけ採点し、GMM成分を窓間で平均しない。
人工gapは元clipの時間軸に固定して全重複窓へ共通に適用し、intervalもmanifestへ保存する。
選択指標は全observedとgap内observedの位置NLLを等重みにする。
NLL/存在値の集計はframeの和・分母を使い、batch平均の平均にしない。
較正用・他sourceのval・testはこのpilotの選択には使わない。

`epoch-NNN.pt`はmodel設定・厳密なstate dict・optimizer・epoch/step・data manifest hashを保持する。
更新したbest epochのvalidation分布をNPZへ残し、`best.json`にcheckpoint hashと参照先を記録する。
`learning_curve.jsonl`はepochごとの結果、`run_state.json`は完了/途中状態を示す。
既存runへの上書き・自動resumeは行わない。各epochの成果物は不変であり、中断時にも残す。
標準のcompile契約を使い、失敗時のeager切替は行わない。

```bash
.venv/bin/python -m src.tasks.ball_refiner.scripts.train \
  paths.data_root=<絶対repo-root>/data paths.cache_root=<絶対repo-root>/data \
  paths.output_root=<絶対repo-root>/outputs \
  run.output_dir=ball_refiner/train/detector_only/<run-id>
```

GPU実行は共有queueへ投入する。`run.dry_run=true`はCPUでcache/教師/分割/除外母数を検査し、
モデル・optimizerは作らない。新規の出力先を使う。
このpilotは学習接続の確認であり、実遮蔽GT・RGB対照・文脈ablation・最終holdoutの採否を
完了したとは扱わない。保存checkpointの分布診断は次節から別runで行う。Meijiの確定負例不足も残る。

## 固定checkpointのvalidation分布診断

`scripts/evaluate_pilot.py`は完了したpilotの`best.json`を読み、checkpoint・モデル設定・
data manifest・入力cache/storeのhashとvalidation分割を照合してから推論する。
設定は[評価YAML](configs/evaluate_pilot.yaml)が正本。
`evaluate.partition`で選択側または較正側を明示し、testの指定は拒否する。
較正側の**診断**であり、この入口では分散scaleや温度をfitしない。
入力runはARTIFACT、結果は別のOUTPUT配下へ保存する。入力データrootは学習runの保存設定を使う。

| ファイル | 責務 |
|---|---|
| `evaluation/hdr.py` | 全GMMの密度閾値・coverage・領域面積とMonte Carlo標準誤差 |
| `evaluation/bootstrap.py` | 同時刻camera群を保ったframe加重のpercentile bootstrap |
| `evaluation/configuration.py` | 評価設定・役割別path・許可するvalidation用途の検証 |
| `evaluation/runner.py` | 固定checkpoint復元、元frameの分布保存、観測/人工gap長ごとの診断 |

HDRはGaussianの画面外tailを含むR²上の条件付き領域で、存在確率で縮めない。
混合からsampleした点のlog密度の分位点を閾値とし、別の独立sampleで
`E_p[1(p(X) >= threshold) / p(X)]`から面積を求める。
成分ごとの楕円の和ではない。面積はsource画素のJacobianを掛けてpx²で報告する。
計算はfloat64、固定数のCPU一様乱数をframe順に割り当て、chunkサイズで乱数列を変えない。
保存するframeごとの面積のMonte Carlo標準誤差は**推定した閾値に条件付けた値**で、閾値推定の誤差を含まない。
集計欄はそのframeごとの標準誤差の平均であり、平均面積の標準誤差ではない。

bootstrapは`meiji/video/clip`を復元抽出し、各群の全camera・全frameを一緒に含める。
frame母数で加重し、clip平均の平均にしない。少なくとも2群を要求し、frame単位の独立標本へ
読み替えない。bootstrapでは推定済みHDRを固定し、Monte Carloの乱数を引き直さない。
群数が少ないintervalは探索的であり、母集団でのcoverageを保証しない。
観測条件だけで元argmaxとの点誤差・paired平均誤差差を比べ、gap条件に未遮蔽detectorを
同じ入力条件の基準として置かない。dense heatmapの密度比較は未実装。

全観測と人工gap内の観測を分け、gap長ごとには**同じ元frame**の無欠損/欠損予測を集計する。
clipごとの全GMM・frame/PTS・gap mask、採点frame、誤差・NLL・HDR閾値/coverage/面積をNPZへ保存し、
checksumとMonte Carlo seedをmanifestへ残す。clip完了ごとに進捗を公開し、最終入力hash確認後にのみ
`status=complete`を公開する。中断/既存runへの上書きや自動resumeは行わない。

```bash
# GPU実行は共有training queueへ投入する。
.venv/bin/python -m src.tasks.ball_refiner.scripts.evaluate_pilot \
  paths.artifact_root=<絶対repo-root>/outputs paths.output_root=<絶対repo-root>/outputs \
  evaluate.training_run=ball_refiner/train/detector_only/<training-run-id> \
  evaluate.partition=calibration \
  run.output_dir=ball_refiner/evaluate/detector_only/<evaluation-run-id>
```

## 新旧pilotの同一validation比較

`evaluation/pilot_comparison.py::run_comparison`は完了した2つのpilotを復元し、
全sourceのval（Meiji video_000、TrackNet game9、chat val）で同一frameを比較する。
学習recipe・split・教師/frame/PTSの一致を要求し、旧pilotは旧cache、新pilotは新cacheを使う。
source/camera別、Meiji選択側/較正側別に、観測、人工証拠gap、occlusion_estimated参考値、
interpolated参考値、確定不存在、unknownを分ける。推定ラベルをobserved GTへ昇格させない。
各指標の母数とN/Aを明示し、frameごとの全GMM・指標・教師理由・入力hashを保存する。
不存在はrefinerの存在Bernoulli NLLだけが定義され、条件付き位置NLL/coverageはN/A。
unknownは存在の負例にしない。detector scoreからamodal存在確率を捏造しない。

`evaluation/paired_metrics.py`でGMMのNLLとHDRを既存診断と同じ方法で算出する。
detectorはnative heatmapを同じJPEG/窓/精度で再推論し、cacheの採用窓・argmaxと照合する。
`infer_clip_evidence`のheatmap sinkは中心距離が改善するたびに上書きし、最終値を候補と同じ窓に揃える。
密度はsource UVに写したnative格子の中点を境界とする区分定数分布で、画像端まで積分し、
明示した一定の一様成分を混合する。全ゼロheatmapは一様分布と定義する。
HDRは同密度cellを全て含むので平坦分布のcoverageは保守的に100%となる。
人工gapではdetectorの空間証拠が無く、一様分布・点推定なしと明記する。
これはRGB遮蔽再推論ではなく、未較正の有限画像密度とR²のGMMとの比較である。
温度・一様成分率・分散scaleをこの評価でfitしない。設定・実行計画・結果はknowledgeへ記録する。

`visualization/overlay.py`は保存済みGMMの各成分のfull covarianceを表示画素へ変換し、
2σ楕円を成分weightのalphaで描く。混合平均は点markerとして描き、成分間分散を各楕円へ混ぜない。
各成分の2σ楕円はGMMの95% HDRではない。storeのsource→JPEG変換率も座標変換に含める。
具体的なval動画の再現script・入力hash・frame/PTS・読み方は対応するknowledge runへ保存する。

## 推論bundleの書き出し

`deployment.py`の`export_pilot_bundle`は、完了したpilotのbest checkpointとconfig・data manifest・
選択結果のhashを照合して、`manifest.json`と`weights.pt`のimmutableなdirectoryを作る。
推論に必要なmodel設定、検出器checkpointのhash・画像サイズ・正規化・候補設定・窓規則、
refinerの窓長・strideを束ねる。既存directoryへの上書きと未完了runのexportは拒否する。
学習data/cache/注釈は書き出しにもruntimeにも不要で、由来のpath/hashは記録だけに使う。

`load_inference_bundle`はchecksumと全設定fieldを検証し、モデル構築は`load_model()`まで行わない。
weightは`weights_only=True`で読み、有限値とstrictなstate dict復元を要求する。
現在のbundle schemaは`use_detector=true, use_pose=false, use_court=false`の未較正pilot専用であり、
文脈ありcheckpointを空pose/courtで実行しない。存在確率・共分散に補正を加えない。

学習のJPEG storeとruntimeの元動画直接decodeは画素値が一致するとは限らない。
bundleに両方のRGB経路を記録する。窓・座標・モデル設定の整合は保証するが、この媒体差の
精度影響とdeploy採否は別評価で扱う。

```bash
.venv/bin/python -m src.tasks.ball_refiner.scripts.export_pilot \
  --training-run <絶対output-root>/ball_refiner/train/detector_only/<training-run-id> \
  --output <絶対checkpoint-root>/ball_refiner/<bundle-id>
```

pipelineの実行方法・保存schema・load-onlyは上記の専用recipeの文書が正本。
