# 全フレーム教師での長系列比較プロトコル改訂案

**規則案はユーザー確認待ち。** 比較目的・固定軸・実験行列は
[親issueの実験設計](https://github.com/Motoki0705/tennis-lab/issues/1367#実験設計)と
[評価条件](https://github.com/Motoki0705/tennis-lab/issues/1367#目標指標評価条件)を参照する。
親で決定済みのclip内文脈、入力全フレーム教師、T=32/64/128、不在評価の追加を前提に、
ここでは集合・窓・採点・予算を具体化する。初回runの中央16案は履歴として残す。

## 入力と共通区間

- `inputs.json` の固定manifest・snapshot・API版を使用する。splitは既存の
  `(source, group_id)` 単位を維持し、元動画をsplit間で共有しない。
  別snapshot・schema・原注釈へ自動的に差し替えない。
- `hybrid.eligible_spans(states, config)` の**最大連続区間のうち長さ128以上の全区間**を採る。
  frame_step=1、区間は半開区間 `[a,b)`。reference-only frameを含めず、
  `timestamp_gap_factor × clip全体のmedian(ΔPTS秒)` より大きいgapを跨がない。
  隣り合うframe indexでもPTS境界を挟んだ区間を結合しない。`segment_break`を一律には切らない。
- 旧play区間や旧32窓への重なりは追加条件にしない。これは「clip内条件(b)で128枚を置ける区間」
  という指定をそのまま区間へ適用する案で、play外の位置教師・不在も含む。
  全Tの入力をこの区間内に閉じ、32/64だけが使える短い区間は学習・採点から除く。
- 共通frame集合Cは区間内の全frame。位置採点Pはそのobserved部分、共通不在Nはその確認済み不在部分。
  C内でも教師不明のframeはmaskのまま入力する。区間・frameのidentityをT・モデル間で一致させる。
- coverageの分母Rは旧manifestのframe_step=1・32枚窓が覆う**一意frame教師**。
  `|P∩R_observed|/|R_observed|` をcoverageとし、新集合の総数をそのまま分子にしない。
  旧32窓の外から増えた教師 `|P\R_observed|` も別に報告する。不在・maskも同じ規則で比較する。

## 教師の契約

分類は初回runの`classify`を再利用し、`annotation_states.supervision`の単一observedを位置教師とする。
`hybrid.reviewed_empty_evidence`・`negative_mask`による`reviewed_empty`と単一`out_of_frame`を
確認済み不在として全面0のheatmap教師にする。Chat空ラベルは原注釈の内容・clip/frame対応・notesを照合し、
notes付き空frameは除外する。元注釈が欠ける・変わる場合は停止する。
TrackNet/Meijiは統一storeの既存契約を使い、不要な元CSVの存在を新しい条件に加えない。

未レビュー・複数球・unresolved・推定位置・不在未確定の空frame・referenceはmask。
プレイ区間の外という理由で不在を捏造せず、欠損位置を補完しない。
`position_valid=observed`、`heatmap_valid=observed OR confirmed_negative`を**全T枚**に適用する。
中央専用maskは設けない。これは固定v1 snapshotについての調査であり、明示teacher maskを持つ別schemaは対象外。

## 学習窓と採用条件の推奨案

Tごとに窓を独立抽出せず、先に**共通の128枚の組**を固定する。

1. 各共通区間`[a,b)`で、128枚の開始位置を`a, a+16, ... ≤ b-128`とし、
   末尾`b-128`を未登録なら1回追加する。pad、反復、区間外contextは使わない。
2. 組内の既知教師（observedまたは確認済み不在）が1枚以上ある組を採用する。
   最小observed数は0。判定はこの共通128枚に1回だけ行い、Tに依存させない。
   全maskの組を除いても、C内の既知教師はすべてどこかの採用組に残る。
3. 同じ開始pから、T=32は`p,p+32,p+64,p+96`、T=64は`p,p+64`、T=128は`p`の入力を作る。
   各窓の全frameで教師を使い、子窓ごとの採用判定は追加しない。
   窓の通常部分の開始間隔はT、128枚の組の開始strideは16であり、両者を混同しない。
   T=32の4窓も同じclipの連続128枚から取る。重なる組が同じ子窓を含んでも、その組ごとの出現重みを保つ。
4. trainの採用組を一様に復元抽出する。同seedの組列を両モデル・全Tで共有する。
   これは窓出現への一様抽出で、source/FPS/group/frameの等確率抽出ではない。
   長い区間や128枚の組の重なりによる重みは全Tで一致する。source等の再重み付けは加えない。
5. 子窓が全maskでも128枚の組から捨てず、forwardには含め、損失の分子・分母への寄与を0にする。
   optimizer更新全体の分母は必ず正。負例のみの組も自然な出現頻度で残る。
   不在集合からの追加抽出・不在のoversampling比率は今回決めない。

例えば同じ128枚にobserved 80、不在20、mask28があれば、32×4・64×2・128×1のどれも
1更新で80/20/28となる。この規則は期待値だけでなく、同じ組列の実教師数も一致させる。
各T独立の「教師1枚以上」や「observed率25%以上」の案は`budget.csv`で比較したが、
Tによる採用とframe出現重みの差が残るため推奨しない。
子窓の全maskを許容するための学習loop対応は #1373 が必要。

拡張を使う場合は、共通128枚の組・出現番号・seedに空間変換と人工遮蔽を結び付け、
分割前の同一frameへ同じ変換を与える。cropによるGT除外等がある場合は変換後の教師数も照合する。
MDDは**各入力窓**の拡張後RGBから再計算する。32枚窓のMDDを128枚で計算したものから切り出さない。
Tが短いほどMDD先頭0と時間処理の窓端が増える効果は残るので、文脈長の差と併せて解釈する。

## 評価窓・重複予測・区間端

評価は学習の採用組を流用せず、Cの各区間でラベルによらず全frameを覆う。

- 各Tの開始位置は`a, a+T/2, ... ≤ b-T`、末尾`b-T`を未登録なら1回追加する。
  strideは32→16、64→32、128→64。十分な重なりを持たせ、区間の最初と最後のframeも残す。
- frame fを含む窓について、`(abs(2*f-(2*start+T-1)), start)` が辞書順最小の窓をownerとする。
  すなわち窓中心に最も近い予測、同距離なら早い開始位置。
  ownerはTごとに変わるが、採点するframeと教師は全Tで同じ。
- 重複排除keyは`(clip_id, frame_index, frame_step=1, exact_native_fps)`。
  ownerの座標とscoreを一緒に採り、正解への近さ・高いscoreで選び直さず、窓間平均もしない。
  評価manifestの採点集合と実予測の集合を照合し、欠落・余分・非finite出力では失敗させる。
- frameの区間端距離は`d_edge=min(f-a,b-1-f)`。`0 / 1–7 / 8–15 / 16–31 / 32–63 / 64+`枚別に
  位置誤差・不在FPR・observed FNR・各教師数を報告する。source/FPS別にも支持数を付ける。
  端のframeは全Tで片側文脈が0になる。そこを主指標から除かず、内部frameと分けた副集計を併記する。
  `edge-support.csv`はTによらない支持数。実PTSから左右の秒数も記録できるようにする。
- owner窓内のframe offsetと前後の実context枚数、MDD先頭に当たるかも評価記録へ残す。
  区間端距離と窓端距離は別の量である。

## 確認済み不在の評価専用scope

snapshot全体の確認済み不在から、条件(b)で128枚を置ける連続区間に属するものをAとする。
そのframeが入る窓で採点し、**窓全128枚が不在である必要はない**。
位置教師・未知が混じる窓も許し、各Tの評価ownerは上と同じ規則を使う。
区間長128未満の不在は、短いTだけで足さない。

今回のCを全適格区間と定義すると、集合の定義から**A=N**となる。
不在の専用artifact・専用レポートは追加できるが、新しいframeは増えない。
`A∩C`と`A\C`を明示し、後者の支持数0を隠さない。
共通scopeと不在scopeの成績は別に示してよいが、両者を足して分母を倍にしたり、
独立した2組の証拠と解釈したりしない。統合するときはframe keyの和集合で一度だけ数える。
ユーザーが共通集合外の新しい不在frameを必要とするなら、Cにplay等の追加条件を課すなど
共通区間の範囲を変える再判断が必要。今回はその変更を勝手に採らない。

splitは保持する。trainの不在一覧は支持数監査・診断用で、閾値較正やholdout成績には使わない。
専用scopeを新しい学習samplerへ追加しない。共通trainに元からある不在は上の自然頻度で教師に使う。
「評価専用scope」は、commonと教師が重ならない独立holdoutという意味ではない。

## 指標・不在閾値

位置はnative heatmapのglobal argmaxと既存log-parabolic subpixel補正で復号する。
元画像への座標変換は既存契約に従い、正規化予測に`(source_width-1,source_height-1)`を掛け、
保存GTは`clip.scale`で割る。復号・decoder・precision・subpixel設定は全条件で揃える。

主指標は共通observedの元画像pixel誤差`sum(||pred_i-GT_i||₂)/n`。
存在判定前の全observedを使い、予測不在でも分母・誤差を残す。
source/FPS/group平均への置き換えはしない。副指標は中央値、P95（linear quantile）、
20 source px以下の割合、source/group/正確なnative FPS/自然遮蔽/区間端距離別の成績と支持数。
支持数0は`null`。自然遮蔽フラグがないframeの遮蔽を推測して補わない。
人工遮蔽を使う場合は同じframe・変換・既知GTを保ったstressとしてcleanとは別に評価する。

`score=max(sigmoid(native_heatmap_logits))`、`present=(score ≥ τ)`。
scoreを存在確率とは呼ばず、hybridだけ存在headやgateを追加しない。
共通Nと専用Aについて、それぞれ確認済み不在に対するpresent数/支持数をFPRとし、
`reviewed_empty` / `out_of_frame` / 合算を出す。observedのabsent数/共通observed数をFNRとする。
不在だけのAにはobservedがないため、A単独のFNRと位置指標は`null`。
共通observedのFNRを添えて、不在scopeのFPRと対応づける。

各固定予算checkpoint・モデル・T・seedのτはvalidationの共通observed Pと
`A∪N`の一意負例（今回はA=N）から、`(FPR+FNR)/2`を最小化して決定する。
score変化点を全探索し、同scoreを分割せず、all-present/all-absentも端点modeとして評価する。
同値ならFPRが小さい点、さらに同値なら高いτ。source別閾値を後付けしない。
閾値・端点mode・混同行列・支持数を保存し、そのままtestへ適用する。
validationで正例か負例が0なら較正不能として停止する。testやtrainで穴埋めしない。

## checkpoint・seed・予算

主評価はUpre/Upostの固定更新時点のcheckpointとする。事前学習は同じ共通train集合のT=32で行う。
各Tへの分岐は固定Upreの全重みから独立に行い、optimizer/scheduleは新規開始する。
resumeではT・集合・recipeを維持し、optimizer・schedule・RNG・組列の位置を復元する。
validation-bestは併記できるが主評価を置き換えず、testでcheckpoint・LR・閾値を選ばない。
本調査のtest支持数は集合のcoverage監査で、test精度による選択ではない。

seed42で全条件。主要対比較は両モデルの32対128を推奨し、seed43/44の範囲と事前学習の再実行は
#1376で合意する。同じ事前学習を使う追加seedはfine-tuning変動だけの評価と明記する。
改善量は`error(32)-error(T)`、主仮説の効果量は`改善量_hybrid-改善量_DPT`。
同じframe・元動画group・seedの対応を保つ。source内のgroupを対応付きで復元抽出する
10,000回のbootstrapを推奨し、乱数seed・各学習seedの値を残す。1 groupしかないsourceの
会場間変動を多数frameから推定したかのように扱わない。

1更新の入力予算Fは128枚（または共通の128の整数倍）を推奨する。
`B_eff(T)=F/T`窓、固定microbatchなら`B_eff=b×A×W`（b=窓/device、A=蓄積、W=device数）。
変則microbatchは実窓数を合計する。F=128なら4/2/1窓で、同じ128枚の組を1つ処理する。
不足した更新を黙って小さくせず、全条件でU回の完全な更新を完了する。
総入力は`U×F`、総窓数は`U×F/T`。各更新で入力frame、observed、各不在kind、maskの実数を記録し照合する。

損失は更新全体の`Σ(valid_frame_loss)/Σ(valid_frame_count)`。
空間平均Focal BCEをframe単位で集め、全maskの子窓の寄与は0にする。
microbatch平均を単に蓄積回数で割ると教師数の違いで重みがずれるため使わない。
複数deviceでも更新全体の分子・分母で正規化する。

F・Upre/Upost・LR schedule・追加seed・GPU時間上限の最終値は、#1376の実測とユーザー判断で固定する。
見積もりは各runの`U×更新時間`に、評価・保存照合・初回準備・VM再作成時の再取得/復元を加える。
Fと更新数を揃えてもFLOPs・wall time・VRAMは等しくならない。
未収束診断の延長は主比較とは別予算・別結果で残す。

## 後続への受け渡し

`common-intervals-*.jsonl`、`train-packs-*.jsonl`、`absence-targets-*.jsonl`と
`clip-catalog.csv`が小さな候補入力。機械向けの形式・規則は`assignment-rules.json`にある。
#1372で正式manifestへ変換し、区間・教師・採点集合のidentityと支持数を本runに照合する。
可搬storeへのclip index変更は`clip_id`で明示的に対応付け、FPS・教師・実PTSを維持する。
hash形式が異なる場合は値を直接比べず、復号したframe key・class集合で照合する。
#1374はChat原注釈から検証した不在mask・来歴も配布版へ凍結する。
本案の合意前にsrcの常設規則として採用しない。
