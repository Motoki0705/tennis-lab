# Ball Refiner

単眼の2D分布推定（#935）と、その分布を使う3D軌道推定（#936）のタスク。
出力先は[タスク出力規約](../OUTPUTS.md)に従う。

## 実装の境界

- `refiner_2d/`: cameraごとに独立して、検出証拠・COCO17の肘/手首・静的court・時間窓から
  1球の位置分布と存在確率を推定する。GMMの各成分は同じ球の位置仮説で、複数球ではない。
- データ生成・評価は今後このtask直下へ置き、#936と共有する。モデル入力には他camera、
  camera ID、正解座標、三角測量、未来のwindow外情報を渡さない。
- pipeline component・学習runner・実データ評価は後続PRで実装する。
  既存pipelineの三角測量はまだ切り替わっていない。最終的な#936の入力はrefinerの全分布のみとし、
  detectorの点推定へ戻す経路は設けない。court_sideの幾何的な仮説検定は別の利用者である。

## 学習戦略（#935、暫定設計）

無人campaignの指示に従った暫定案であり、ユーザーの合意済みとは扱わない。
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

detectorはft-e13を凍結した基準から開始する。混合FT epoch 0は比較候補として別cacheに固定し、
選択はvalidationで行う。checkpoint、前処理、動画/注釈hash、frame/PTS、sourceサイズ、
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

初期pilotはseed 42、実時刻付き33frame窓、実3 sourceを等比率、AdamW lr 3e-4、
batch 32、12 epochかつ最大3,000更新（先に達した方）を上限とする。
実行前に後続PRのHydra configへ移して、この節にはconfigのリンクだけを残す。
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
