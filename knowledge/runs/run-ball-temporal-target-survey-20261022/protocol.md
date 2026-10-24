# 長系列比較の評価規則案

**未合意の案。** 比較の目的・固定軸・実験行列は
[親issueの実験設計](https://github.com/Motoki0705/tennis-lab/issues/1367#実験設計)と
[目標指標・評価条件](https://github.com/Motoki0705/tennis-lab/issues/1367#目標指標評価条件)を正本とする。
ここにはCPU調査から具体化した規則を置く。文脈条件・中央領域長・上限候補・共通集合は
人間の判断後に親へ記録し、本案の状態も更新する。学習予算の数値は #1376 で決める。

## 対象のidentityと窓

1. 入力版は `inputs.json` のmanifest・snapshotに固定する。split単位は `(source, group_id)`。
   clip数・camera数を元動画数として数えない。train/validation/testのgroup重複を拒否する。
2. 候補anchorは元manifestの **frame_step=1の32枚窓だけ**。開始を `s` とすると中央教師ブロックは
   `[c,c+16) = [s+8,s+24)`、T枚の入力は `[c-(T-16)/2,c+16+(T-16)/2)`。
   既存strideと末尾backfillを保つ。clip全体の新しい開始gridや、不在専用窓を追加した集合ではない。
3. 文脈案(a)は入力全体が同一play区間内、案(b)は `hybrid.eligible_spans` 内であることを求める。
   (b)もreference-only frameとnative PTSの大きいgapを跨がない。gapの定義は元manifestの
   `timestamp_gap_factor × clip全体のmedian(ΔPTS秒)` より大きい差。`segment_break`は一律に切らない。
   play外の文脈・教師のない文脈を、不在ラベルには変えない。
4. 合意した文脈条件と最大長Nで成立する中央ブロック集合をsplitごとに固定し、全Tへ同じ集合を渡す。
   中央16枚の既知教師が0枚のブロックは、全条件から除く。元窓の「8 observed以上」を
   中央16枚へ再適用する追加の足切りはしない。位置教師を持たず確認済み不在だけを持つブロックは残す。
5. `centers-<context>-<N>-<split>.jsonl` は、この全maskブロック除外後の候補集合。
   各行は `[snapshot内clip_index, [[中央start先頭,中央start末尾inclusive,step],...]]`。
   `step` はanchor開始間隔で、入力frame_stepは常に1。末尾backfillの非整列startも保持する。
   128と256は**代替する集合**であり、T=128だけ128集合、T=256だけ256集合と混在させない。
6. `summary.json` の `target_identity_sha256` は全maskブロックを除いた中央frameの一意集合、
   `teacher_identity_sha256` は同じidentityへ教師classを付加したもの。生成順・正確なFPS表記は `inputs.json` にある。
   #1372 の常設manifestのhash形式はその実装が宣言する。形式が違うhashを直接比較せず、
   この候補を復号した `(clip_id, frame_index, frame_step, native_fps)` と教師maskの集合を照合する。
   欠落・余分なtarget、教師maskの差は採点前に停止する。

共通集合のcoverageの分母は、元manifestの全frame_step=1・32枚窓の**一意frame教師**。
中央16枚への変更だけの損失を区別するため、元窓の中央16枚の一意教師を分母とした率も残す。
前者を `*_coverage_full32`、後者を `*_coverage_center16` とする。snapshot全体との比較は
`*_coverage_snapshot`。窓出現回数を分母にしない。source/group/FPS構成も同じ母集団と比較する。

## 教師と重複予測

- 位置教師は `annotation_states.supervision` の単一observed。
  確認済み不在は `hybrid.reviewed_empty_evidence` と `negative_mask` で判定し、
  `reviewed_empty` と単一 `out_of_frame` の両方を採る案とする。Chat空ラベルは元注釈の
  hash・clip/frame対応・notesを検証したmaskだけを使う。配布版への凍結は #1374 の責任。
- 未レビュー・reference・複数球・unresolved・推定位置・notesで不在未確定の空frameはmask。
  文脈にも中央にも位置を補完せず、位置が分からないことを不在へ読み替えない。
  この調査は固定v1 snapshotに限定した。別schemaの明示teacher maskを黙ってlegacy扱いしない。
- 重複のkeyは `(clip_id, frame_index, frame_step=1, native_fps)`。
  所属ブロックの中央 `c+7.5` から対象frameまでの距離が最小の出力を1回使い、同点は早い `c` を使う。
  このowner順序はTによらない。座標と存在scoreは**同じowner**から採る。
  窓間で平均したり、正解に近い予測・scoreの高い予測へ選び直したりしない。
- `masked` は幾何候補の中央領域の一意frame数。全maskブロック除外後のmask数は `usable_masked`。
  `*_occurrences` は重複込みの教師量で、一意frame指標と分ける。学習ログには両方の量を残す。

## 採点の定義

モデル出力をnative heatmap上のglobal argmax＋既存log-parabolic subpixel補正で復号する。
source pixelへの変換は既存datasetの端点・width-ratio契約に合わせ、
予測は正規化座標に `(source_width-1, source_height-1)` を掛け、GTは保存座標を `clip.scale` で割る。
stored画像のpixelと混ぜない。復号・JPEG decoder・precision・subpixel設定を条件間で揃えて記録する。

一意observed frame iの誤差を `d_i = ||prediction_source_px - GT_source_px||₂` とする。
主指標は全observedの `sum(d_i)/n`。**native FPS/source/groupを等重み平均して置き換えない。**
存在判定前に全observedを採点し、予測不在でも誤差を残す。非finite出力・欠落出力は評価失敗であり、
難しいframeを分母から落とす理由にしない。

副指標は同じ `d_i` の中央値、P95（linear quantile）、`d_i ≤ 20 source px` の割合。
20 pxは既存candidate評価と距離の単位を合わせるための規則案で、testを見て動かさない。
source別・**native FPSの正確な有理数別**・自然遮蔽フラグ別に、平均/中央値/P95/20px内率を出す。
各行へobserved/negative/unknownの一意frame数、clip数、`(source,group_id)` 数を添える。
支持数0の指標は `null`。少数group・同一元動画の複数cameraを独立な動画多数と扱わない。

不在判定は `score = max(sigmoid(native_heatmap_logits))`、`present = score ≥ τ`。
scoreは未較正であり存在確率とは呼ばない。確認済み不在に対するpresent数からFPRを、
observedに対するabsent数からFNRを求める。FPRは `reviewed_empty` / `out_of_frame` 別と合算、
FNRは全observedを分母にする。unknownはどちらにも入れない。

閾値τは各固定予算checkpoint・モデル・T・seedについて、**共通validationだけ**で
`(FPR + FNR)/2` を最小化するoperating pointを選ぶ。validationのscoreの変化点を全探索し、
同scoreを分割しない。all-present/all-absentも明示した端点として評価する。
同値ならFPRが小さい点、なお同値なら高い閾値の点を採る。
score定義・比較演算・閾値または端点mode・混同行列・支持数を凍結し、同じものをtestへ適用する。
source別閾値を後付けしない。validationで正例または負例が0なら較正不能として止め、
testや別のscopeの閾値で埋めない。

今回の全候補でvalidationの中央 `reviewed_empty` は0件、自然遮蔽付きobservedも全splitで0件。
それらの副指標は `null` となる。閾値の負例根拠は主にChatの `out_of_frame` に限られる。
clip全体の不在・非プレイでの性能を主張するには、独立した不在target集合と比較規則の追加合意が必要。
人工遮蔽を評価する場合は既存のstress設定・変換の来歴を別に固定し、同じcentral frameに
同じ遮蔽を与え、観測GTを保持する。自然遮蔽の代用とは呼ばず、clean主評価から分ける。

## checkpoint・seed・不確実性

- 主評価は **Upre / Upostの固定更新数** のcheckpoint。Upreの全重みから各Tへ独立に分岐し、
  optimizer/scheduleは新規。resumeは同じT・recipe・target集合・更新位置の継続とする。
  validation-bestを併記する場合も主評価を置き換えず、testでcheckpointを選ばない。
  LR/上限/拡張などの選択と閾値較正を凍結した後にtestの予測・精度を開く。
  本CPU調査のtest教師数・構成はcoverage監査で、test精度による設定選択ではない。
- seed42で親の全条件。seed43/44は主要対比較へ追加する。推奨する主要対は
  両アーキテクチャのT=32対採用最大T。対象と事前学習の再実行範囲は #1376 で合意する。
  事前学習もseedごとに再実行すれば工程全体のseed変動、同じ事前学習を使えば
  固定初期checkpointに条件付けたfine-tuning変動と明記する。
- 各モデルで `改善量(T)=error(32)-error(T)`、主仮説の効果量は
  `改善量_hybrid(T)-改善量_DPT(T)`。絶対位置誤差差と混同しない。
  同一target・同一seed・同一元動画groupの対応を保つ。
- 不確実性はsource内の元動画groupを復元抽出する対応付きbootstrap（10,000回案）で示す。
  各反復では条件間で同じgroup multiplicityを使い、全observedの平均とその差を再計算する。
  seed間の値も個別に示す。元動画1 groupのsourceで会場間の不確実性は推定できず、
  多数frameを独立sampleとする狭い区間で補わない。bootstrap自体の乱数seedも記録する。

## 更新数・実効batchの算出規則

事前学習もfine-tuningも、合意した共通trainブロック・中央maskを使う案とする。
モデル対・T対で同seedの窓列を共有し、サンプル位置をresume状態へ保存する。
窓の並び・拡張乱数はcentral block identityとseedへ結び付け、Tの長さで中央画像やmaskを変えない。
広い32集合を事前学習に使う変更は別の明示recipeとし、片側だけへ適用しない。

`b` = deviceあたりmicrobatch窓数、`A` = 蓄積回数、`W` = device数とし、
`B_eff = b × A × W` 窓/optimizer更新。分割が一定でない場合は実microbatch窓数を合計する。
最後の不足batchを黙って小さな更新にせず、ちょうどB_eff窓の更新をU回完了する。
処理窓出現数は `U × B_eff`、入力frame数は `U × B_eff × T`。
教師量はサンプル列の各central maskのobserved/各negative数を累積し、
maskと列が同じならTによらず完全一致することをログで照合する。

損失は更新全体の `Σ(valid_frame_loss) / Σ(valid_frame_count)`。
各microbatchの有効教師数が異なるとき、microbatch平均を単にAで割る方法ではこの値にならない。
蓄積・複数deviceでも分子と分母を同じ更新内で正規化する。B_eff窓の全maskはデータ誤りとして停止する。

#1376 は両モデル・各Tの同じB_effを成立させるb/A、更新時間、評価・checkpoint保存時間を測る。
共通Upre/Upostはその実測とGPU時間上限から決め、速い条件だけ更新を増やさない。
総時間見積もりは各runの `U × 実測更新時間` に、評価回数×評価時間、保存回数×照合付き保存時間、
初回転送・環境構築・VM再作成回数×再取得/復元時間を加える。
同更新数は同FLOPs・同wall timeを意味しない。未収束診断の延長は別予算・別結果とする。

Upre/Upost、B_eff、LR schedule、追加seedの実行範囲、GPU時間上限の数値はここでは決めない。
