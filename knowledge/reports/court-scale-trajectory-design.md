# Courtデータ規模比較の軌道・画像選択案

**提案・ユーザー合意前。Refs #1350 / 親 #1343。** 2026-10-22。
既存の生成器・presetの仕様を変更する文書ではない。採用後の実装仕様は
Court dataset READMEへ移し、この文書には判断根拠への参照を残す。
CPUで観測した結果の正本は
[診断run](../nodes/synthetic_data_generation/000024-run-court-nested-plan-cpu-20261022.md)。
ここではアルゴリズム案・契約・未決事項を扱う。

## 提案の要点

推奨は「分布を固定した軌道列、固定弧長表からの2倍細分、採択後の入れ子選択」。
固定評価のfamilyと近傍を先に予約し、生成条件と採択結果を不変のデータ版へ結び付ける。
**候補集合の入れ子は成立するが、指定枚数の採択集合は単なる先頭n本では表せない。**
後者には採択後のID列が必要である。

| 方式 | 利点 | 限界・今回の扱い |
|---|---|---|
| 現行の格子・予算選択を増量 | 実装差が小さい | 密度の入れ子、乱数、再split、選択順依存のview、有限候補、whole-group shardが未解決。設定値だけの変更は推奨しない |
| 固定層別のcounter乱数列＋2倍細分（推奨） | 拡張で既存点を変えない。途中再開・再描画・採択理由の追跡が容易 | 層別化しても品質棄却後の分布は変わる。分布診断と不足時の停止が必要 |
| 層ごとのscrambled Sobol列＋2倍細分 | 少数候補でもパラメータ空間を埋めやすい | 2の冪以外・間引き・棄却後には元のbalance保証を持ち込めない。guard/QA棄却が多い今回には保証を主張しにくい。代替案 |
| 撮影カメラ近傍の局所軌道 | 画質リスクを小さくできる可能性 | 閉軌道の全周分布と異なる。既存撮影範囲の密度に依存する。hull内候補が試走で破綻したときに比較して合意し直す |

Sobolの制約は[SciPy公式資料](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)
のWarningに基づく。乱数は
[NumPyのIDとroot seedを組み合わせる方法](https://numpy.org/doc/stable/reference/random/parallel.html)
を参考に、状態を共有せず各IDから構築する。どちらも描画品質や棄却後の均一性の保証ではない。

## 1. 現行方式の確認と補足

参照版は `7e307f0499be3a2511afaba51966ae3362ec81b1`。行番号はこの版に対するもの。
以下は提案の推測ではなくコードの挙動。

| 確認点 | 根拠 | 設計への影響 |
|---|---|---|
| 4096区間の折線からLを近似し、`max(minimum, ceil(L/step))`をmultipleへ丸め、最大隣接距離が収まるまで加算する | [sampling.py:21,37–74](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/sampling.py#L21) | 半分のstepでも点数がちょうど2倍にならず、共通点の包含を保証しない。`ceil`だけの問題ではない |
| minimum countはminimum group数から導き、count multipleにshard数を使う | [contracts.py:798–820](../../src/synthetic_data_generation/dataset/court/contracts.py#L798) | group下限・処理shard数だけの変更でもgeometryが変わり得る。新計画のN0を処理分割から独立させる |
| v2/v3のjitterは`sample_index=len(samples)`を渡す | [selection.py:1019–1071](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/selection.py#L1019)、[targeting.py:122–138](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/targeting.py#L122) | 既存点より前への追加、前の軌道の高密度化、再選択で姿勢が変わる。**末尾への単純な追加だけなら以前のindexは変わらない**。issueの「軌道を足すと」は条件付き |
| 選択されたgroupの位置からcoverage・HFOV・注視高さ・view IDを決める | [selection.py:858–889](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/selection.py#L858) | jitterを直すだけでは不十分。viewも軌道の固定IDから決める |
| 全groupをsort→shuffleし、件数依存のfraction丸めでsplitする | [selection.py:589–624](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/selection.py#L589) | seedが同じでもgroup追加で評価が再分割される |
| 半径scaleはlinspaceの3値、その他は有限の形状・向き・高さ・振幅格子。typed sort後に番号を振る | [trajectory.py:148–245](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/trajectory.py#L148) | 数値範囲を広げることと本数だけを増やすことは別。circleの向き違い等はtyped keyが違っても物理的に同じ曲線になり得る |
| 全候補を距離サンプリングしてからcoverageを予算内で選ぶ | [selection.py:365–457,476–567](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/selection.py#L365) | 候補格子の単純拡大は計画メモリ・計算量も増やす。既存greedyの結果を条件ごとに作り直してprefixになる保証もない |
| proposal 4800、8 shard、各600frame、1800秒、最大8 NHT呼出し | [base.yaml:39–56](../../src/synthetic_data_generation/configs/dataset/court/base.yaml#L39) | 1シーン24000枚以上には固定上限が合わない。600はNHTのGPU microbatch数とは別 |
| groupを分割せずshardへ入れ、合計容量超過・単独group超過で失敗 | [selection.py:627–690](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/selection.py#L627) | 密な1軌道だけでも600を超える。split familyと処理shardを別にする |
| pre-renderでは全コート合計4点を条件にする | [court/rendering/nht.py:251–267](../../src/synthetic_data_generation/dataset/court/rendering/nht.py#L251) | 対象コートに4点ある保証ではない。学習がtarget_court専用なので、対象単位の診断が必要 |
| α/depth由来の可視点が全体で0なら棄却。JPEG化前の配列で検査し、配列は保存しない | [assembler.py:456–491](../../src/synthetic_data_generation/dataset/court/assembler.py#L456) | 既存acceptedは新しい画質基準の合格を意味しない。元のalpha/depthは再描画なしでは検査できない |
| visibilityは近傍にα≥0.01かつdepth>0があるかを見る | [labels.py:885–1018](../../src/synthetic_data_generation/dataset/court/components/labels.py#L885) | 投影KPの期待depthとの差や白線との一致を判定していない。alphaは画質の十分条件でない |
| release時は最低枚数・率、group非空、full/near_full/partialと全semantic classの出現を検査 | [assembler.py:496–539](../../src/synthetic_data_generation/dataset/court/assembler.py#L496) | 目標train枚数・各splitの品質を保証する別契約が必要 |
| shard再利用はattempt tokenと完全一致する計画に束縛 | [court/rendering/nht.py:92–146](../../src/synthetic_data_generation/dataset/court/rendering/nht.py#L92) | VM消失後の永続再開をそのまま既存実装済みとは扱わない |
| JPEG quality95＋疎教師を保存し、descriptorは最後に公開 | [sample_store.py:24–71](../../src/synthetic_data_generation/dataset/court/sample_store.py#L24) | 保存基盤を再利用し、条件参照と描画cacheの責務をその上に追加する |

入力はメインtreeのB00〜B03のpublic export・accepted alignmentを読み取りだけで使用した。
B01/B02のhuman-confirmed形式も`AlignmentResult.from_dict`で読めた。
B01の保存済み設定だけは`sfm_complex_center_on_hull`が欠け、現行の厳格な設定parserが拒否する。
prototypeでは**B01に限りfalseを明示追加したコピー**で旧候補数を診断した（入力は変更なし）。
これは資産可搬化の解決ではない。#1351が扱う変換・検証と混同しない。

## 2. シーン領域と軌道分布の案（論点1・2・7）

位置はaccepted courtのmetric座標で計画し、公開カメラを`MetricSceneAdapter`で同じ座標へ移す。
NHTの正規化座標をmと誤認しない。全コート・座標・schema・renderer資産版を先にpinする。
新キャンペーンでは全シーン共通の分布と、そこから求めるシーン別の許容領域を別々に記録する。

推奨候補分布は以下。数値は**試走用の推奨候補**であり、本実験の確定値ではない。

| 変数 | 推奨候補 | 代替・追加で判断する材料 |
|---|---|---|
| 形状 | circle / ellipse / rectangle / superellipseを等重み | 既存の採用頻度を重みにする案。形状による画質・可視KP・棄却率を比較 |
| 中心 | complex:1/2、court全体:1/2。後者は各courtへ等分 | 全centerを等重みにするとコート数の多いB01だけcourt寄りになる |
| complex中心 | 全シーンでcaptured-camera hullの頂点平均 | accepted court中心平均。B02だけ例外にする案は方針差を残すので非推奨 |
| 水平領域 | hullを0%拡張し0.5m内側へoffset、shape supportで全曲線を収める | 既存保存設定は5%拡張。0%でも視線・高さ・空隙の品質は保証しない |
| scale / axis ratio / 向き | scale U[.65,.95]、非円ratio U[.35,.8]、向きU[0,π) | 旧ratio .65/.8等を固定離散で均衡化する案。円の向きは0に正規化 |
| 高さ / 垂直変動 | base U[1.5,3]m、planarとsinusoidalを等分。後者amplitude U[.05,.5]m、1周期、phase U[0,2π) | 撮影高さの支持範囲と照合して狭める案。範囲を変更したら全条件を新版で計画し直す |
| HFOV / 注視高さ | 75/92.5/110度の3層、0/1.5mの2層を固定IDで均衡化 | B01の旧45–90度も含めた共通分布。広角の全体コート率・歪みと比較 |
| 注視jitter | 半径1mの一様面積disk、点IDごとの乱数 | 0m、または軌道に沿う連続jitter。後者は周辺分布が変わるので別合意が必要 |

同じ形状の比例縮小を継承するanalytic hull fittingは既存`sfm_bounds.py:26–100`を再利用できる。
円/楕円/矩形/superellipseの支持関数を使い、サンプル間の角を含めて境界を守る。
ただしhullは未観測の空隙を橋渡しし、高さと視線の外挿を抑えない。
撮影カメラへの位置距離、視線角差、対象courtごとの支持カメラ数、撮影高さの分位点からの逸脱を併記する。
新分布に対して全centerが許容hull内に入らない場合は、共通方針を満たせないシーンとして停止する。
中心を黙ってずらす・範囲を拡大する・他シーンで埋める処理は入れない。

候補生成は固定の層別blockを繰り返し、各層内の連続値を
`RNG(hash(policy_version, scene, generation_seed, stratum, counter, field))`から得る。
blockの重み・順序とcounterは最大画像数に依存させない。
小prefixでは完全なjoint均衡が取れないこと、丸め誤差、guard/QAによる歪みを記録する。
棄却後の採択列も安定にできるが、元の分布の保持とは別である。
全シーン共通の分布からの**条件付き分布**になるため、全4シーンのparameter・姿勢・画角・KP支持分布を比較する。
CPUの位置だけのguardではB00/B01のcircle、B03のcomplex中心が大きく残り、
固定順に棄却するだけでは出力分布を維持できなかった。
実装案では層ごとの独立counterと採択reservoirを持ち、guard後にも所定の層順で追加する。
各層の探索上限を計画に含め、満たせない層は別の層へ付け替えず停止する。
この方法も層内の条件付き分布差までは消せない。位置だけのguardは保守的なstress probeとして扱い、
pose/画角を併用するguardと残存支持・動画を比較してD4で判断する。

重複は2段で扱う。exactは正規化した物理曲線＋view関数で同一性を判定する
（円の向き、閉曲線のstart phase、進行方向、単一courtで一致するcomplex/court経路を考慮）。
形状名やcandidate IDだけで数えない。派生する半径・view・jitter違いは同じfamilyへ入れる。
nearは「曲線の位置・視線・画角」距離と「共通物理KPの画像対角長で正規化した投影距離」の2診断を使う。
単に異なるseedなら別軌道、とは扱わない。閉曲線の循環shift・逆向きにも不変な比較を使う。
全ペアO(G²P²)を避け、bounding box/空間cellで候補を絞ってから精査する。
同一軌道内でA2が増やす近接点は実験操作そのものなので、nearを理由に間引かず相関として記録する。
近重複率は軌道内・異なる軌道・split間を分け、最近傍が閾値内にある点の割合を推奨指標とする。
train内の軌道重複を棄却する場合も、固定した先行候補に対する判定として固定し、
後から高密度点が増えたことを理由に以前の点を棄却し直さない。

## 3. 弧長の2倍細分とID/seed（論点3）

軌道tの固定3D弧長表を一度作り、長さをL_t、基準点数をN_tとする。
`N_t = round_up(max(N_min, ceil(L_t/d)), multiple)`を**基準時にだけ**確定する。
その後は点数を`N_t * 2^k`にする。必要な精度検査で不足したら基準表/点数を再固定し、
公開済み版へ途中から修正しない。全kで別々にceilやwhile補正をしない。
ここでdは計画間隔の上限で、実際の基準弧長間隔は軌道ごとのL_t/N_tになる。
phaseを付ける場合も軌道IDから一度だけ決め、全kで同じ弧長offsetを加える。
prototypeはphase=0で、courtのmid-plane上へ点が乗る棄却例も含む。

```python
arc_map = freeze_arc_map(trajectory, tolerance, algorithm_version)
N0 = freeze_base_count(arc_map, base_step=d)
for j in range(N0 * (2 ** k)):
    q = Fraction(j, N0 * (2 ** k))  # 既約分数、0を含み1は含まない
    point_id = hash(scene_asset_id, family_id, trajectory_id, view_id, q)
    position = arc_map.at(q)
    target_court = fixed_target_rule(position, accepted_layout)
    jitter = rng(hash(generation_seed, point_id, "look_at"))
    K, pose, look_at = make_camera(fixed_view, target_court, position, jitter)
```

粗点jは細点2jと同じ既約分数になるので、包含が成立する。
**N0が2の冪である必要はない。** 分数をfloat化してからhashしてはいけない。
共通点は同じ固定表へ同じ値で問い合わせる。高密度版全体を再補間して粗点を取り出す方法は避ける。
実装では新規の奇数点だけを足し、既存点のgeometryを再利用すると数値差を防ぎやすい。

`family_id`はsplitの単位、`trajectory_id`は具体的な曲線、`view_id`は固定view仕様、
`point_id`は撮影点、通し`sample_index`は保存順だけの属性にする。
IDの入力はversion付きcanonical JSON等で定義し、hash全体を保持、衝突を検出したら停止する。
frame/source index、shard番号、列挙位置、最大n・k、worker数、処理順をseedへ混ぜない。
PCG64等のアルゴリズム・NumPy版・分数表現・float精度をpinする。
fieldごとに独立のseed名前空間を持ち、パラメータを1個追加しても他の乱数消費がずれないようにする。
prototypeはtrajectoryごとのPCG64を使用した簡略版で、field独立化は後続実装要件。

nearest courtは既存v3のcourt経路固定／complex経路の3D最短距離＋tie規則を維持する。
target courtが切り替わる箇所、独立jitter、矩形の角では動画の姿勢が滑らかになる保証はない。
これはFPSの動画撮影モデルではなく距離サンプリングのデータ生成である。

保証の範囲はK・pose・注視点・target・投影KP/physical ID/canonical poseの一致。
renderer可視性は同じrenderer/checkpoint・設定でもバッチ差やGPUで変わり得るため、
**同一pointの採択・RGB・renderer教師を一度生成したcacheから共有**する。
再描画のbyte一致はCPU計画の証拠から主張しない。Colab/ローカル差は#1351で照合する。

## 4. 固定評価と漏れ防止（論点6）

評価はtrain量を決める前に、別の`evaluation_seed`・policyと、全B00〜B03について固定する。
validation/testのfamily・軌道・全点・採択ID・順序・hashをregistryへ書き、train seedから独立にする。
比率で全候補を再分割しない。Bでtrainに未使用のシーンも評価から外さない。
画像の選択seed・学習seedを変えてもregistryを変更しない。
trainの生成seedを変えるとそのtrain点自体は変わる。固定されるのは同じseed内の共通点と評価である。

familyは同じ基本経路からの密度・phase・view・jitter・局所変形をまとめる。
独立生成familyの近接もあり得るので、family分離だけでは十分でない。
評価の連続軌道（jitterを含むview許容域）周囲を予約し、train family全体をそのguardから除外する。
validationとtestの間にも同じ検査をかける。将来の高密度化で新たな衝突が見つかって
既存trainを消す設計を避け、連続曲線の保守的な距離boundか、固定した最大kまでの検査を先に行う。
後者の場合は最大k拡張を別データ版にする。

nearの推奨比較値は位置0.5m/視線2度/HFOV1度と、緩い位置1m/視線5度/HFOV2度の感度比較。
すべての姿勢条件を満たす場合をnearとする案と、位置だけで除外する保守案を比べる。
数値の採用は重複動画と残存候補・分布を見て判断する。prototypeは後者の位置0.5mを試した。
1024点の弧長格子と両曲線の`(L_a+L_b)/(2*1024)`を距離余裕に足したが、
弧長表自体の数値誤差boundは未認証であり、productionの漏れなし証明ではない。
原画像のcrop違い・違う対象courtでも同じRGB視点になり得るので、guardをtarget IDだけで分離しない。

評価候補の不足、val/test近接、評価の品質不足は本実験前に停止する。
評価点だけ都合よく補充せず、原因・候補変更を人間へ示し、registry全体を新版で固定する。
評価画像数・シーン別支持数の最終確定は#1356。prototypeの各split1軌道はその推奨枚数ではない。

## 5. 入れ子仮説・計数・採択後の選択（論点4・7）

固定した列t_1,t_2,…に対し候補集合を
`C_s(n,k) = union_{i<=n} points(t_i,k)`と書けば、nまたはkを増やすと包含が成立する。
前提は資産、軌道/view仕様、N0、弧長表、点ごとの乱数、guard、品質規則の固定。
点単位の固定predicate Qなら`E=C∩Q`も包含される。
条件全体のaccepted fractionやcoverageで点の採否を変えるとこの前提が崩れる。
release gateは点を再判定するのではなく、条件全体を公開可/不可にする。

一方、採択枚数は`sum_t sum_q 1[Q(t,q)]`であり、長さと採択率に依存する。
全軌道が同じ長さでない限り、nを2倍にしても候補数は正確に2倍にならない。
密度なら候補数は2倍にできるが、採択数の倍増は保証できない。
したがって、**「全条件を先頭n本×kだけで指定する」強い仮説は不成立**。
「それを候補上限とし、採択後の入れ子ID選択を加える」弱い仮説を推奨する。

| 系列 | 変えるもの | 固定するもの |
|---|---|---|
| A1 | 新しい軌道blockを追加 | d、候補分布・範囲、評価、共通点、既に選択した点 |
| A2 | 固定軌道のkを増やし、新生点だけ追加 | 軌道集合、parameter、view、N0、評価、既に選択した点 |
| B | scene集合とsceneごとの選択枚数 | d、生成方針、評価、共有pointの画像・教師 |

各sceneの基準は1500枚。A1/A2は同一の基準選択ID列を共有する。
A1は新しい軌道blockから追加枚数だけ、A2は新しい密度birth層から追加枚数だけ選ぶ。
候補数・目標数・幾何pass数・描画数・QA採択数・最終選択数を別々に数える。

```python
selected = freeze_shared_baseline(accepted_candidates, target=1500)
for rung in (1, 2, 3, 4):
    candidates = fixed_new_trajectory_block(rung) if A1 else fixed_new_density_layer(rung)
    accepted = check_or_load_QA(candidates, immutable_policy)
    extra = target[rung] - len(selected)
    # 全軌道を使う契約なら1点以上/軌道を先に予約し、その後は均衡順で選ぶ。
    added = balanced_stable_order(accepted, per_trajectory_floor, fixed_strata)[:extra]
    if len(added) != extra or missing_required_groups_or_coverage(added):
        fail_with_counts_and_reasons()  # 別step、重複、別sceneへ切り替えない
    selected += added
    freeze(selected_ids, targets, actual_groups, actual_spacing_distribution)
```

均衡順は軌道内の点を固定hash順にし、軌道/層を固定順でround-robinする案。
連続する弧長の先頭だけを採る偏りを避ける。QA後にquotaを達成できない層は明示して失敗させる。
Gは計画上の軌道数と実際に1点以上採った軌道数を区別する。
「全G本使用」を満たすならfloor=1以上、target≥Gが必要。単にG本の候補から選ぶ場合とは異なる。
prototypeはA1について後者で、全group使用・最終coverage gateまでは実装していない。
A2は基準で1点以上採れる軌道だけを先に固定し、全段で実使用family集合の一致も検査する。
この基準のpreflightは仮想QAを用いた実験上のものなので、productionでは実QAで行う必要がある。

補充には選択肢がある。推奨は、試走で共通Gと候補予算を十分に確保し、各blockに
固定の予備候補を先に計画する方式。A1の本数倍率は目安として、実際に採用した本数を記録する。
**A2で軌道追加による補充は不可**。A2のいずれかのbirth層が不足なら、固定Gや基準dの見直しを
全系列に適用して新版を作るか、その最大規模を断念する。条件ごとの勝手な変更はしない。
代替は全候補をそのまま使い、枚数を実測値に変更する方式。親の等枚数比較と異なるため合意が要る。
採択率が悪い軌道を外すと位置・画角が偏り、長い軌道は画像重みが大きくなる。
候補/採択/選択の三段で分布とgroup別支持数を報告する。
採択後のquota選択は距離格子を間引くため、最終画像の隣接間隔までd/2^kになる保証はない。
「同じd」は生成候補側の統制と明記し、選択後の弧長gap分布も報告する。
A2で軌道ごとの選択枚数も厳密に倍増させるなら、基準quotaを各軌道に固定して
各birth層でその追加quotaを要求し、不足した軌道を他で補わない方式へ強める。
prototypeのround-robinはその追加制約まで検証していない。exact枚数・全軌道使用・
選択後の等間隔を同時に保証できるとは扱わず、D3でどの条件を優先するか合意する。

Bの24000枚はsceneごとのA1選択ID列のprefixから24000/12000/8000/6000枚を取る。
8000枚では軌道blockの一部になるので「軌道先頭n本」だけでは表せない。
Bの4sceneはA1の24000枚版と**ID列まで共用**する。
B各条件はA1最大集合の部分集合になれるが、B同士は総数固定でscene配分が変わるので包含関係はない。
固定scene追加順の効果とsceneの個性は分離できず、組合せ変更の補足確認は#1356で判断する。

96kを維持し、各sceneにA1最大24000枚があり、上記の不変性を守るなら、
`|A1_max ∪ A2_max| = 96000 + 96000 - 6000 = 186000`枚の**選択済みtrain**で全12条件を表せる。
これは描画枚数の上限ではない。QA棄却、選択外候補、固定評価、試走、再試行は加算される。
概算描画数は`sum_s ceil(required_s / measured_acceptance_s)`にquota/guard余裕を加える。
主実験の最大Aを48k以下へ縮めるとA1のscene容量は12000以下になり、B00単独24000枚は収まらない。
その場合はB用の同じdの候補を別途延長するかB総数を見直す判断が必要で、「19万で済む」は使えない。
QA・分布・資産・evalを条件ごとに変更した場合も再利用の主張は成立しない。

## 6. 品質診断と閾値を決める試走（論点4・5）

| 段階 | 記録・検査 | 閾値の扱い |
|---|---|---|
| 計画前 | accepted layout、変換の相互逆、単位、finiteなproper rotation、K/解像度/座標規約、asset hash | 既存契約を維持。欠損・異版はhard error |
| pre-render | full-curve境界、高さ・撮影支持、near/far曖昧、各courtのin-front/in-frame KP、target KP、pose再投影・canonical往復 | 曖昧・不正幾何は既存どおり停止/棄却。対象4KP以上を追加候補として比較 |
| post-render | RGB/alpha/depth shape・dtype・finite・範囲、正depth率、target ROI内α分位点/穴率、可視KP、破綻画像、投影白線とのずれ | 既存α≥.01＋正depthは基準。ROI穴率/ぼけ/白線のhard gate値は未確定 |
| 条件公開 | 各scene/splitの目標数、group非空、coverage/class支持、近重複、品質レポート、全manifest/hash | 不足はfail。最終的な支持数quota・全体/部分の割合は合意する |

KP/pose/denseの教師は同じgeometryを共有する。denseを別に保存する提案ではない。
α/depthの取得は描画時にしかできないので、再開用のQA要約（検査版・値・理由）を疎recordとともに残す。
depthはrendererが返す座標系を明記し、metricへの変換を検証する。
depthの期待値とKPのzの差を新たなgateにする前に、renderer depthの定義・遮蔽・ネットの影響を調べる。
白線のpixel差はalignment誤差とrenderer誤差が混ざるので、そのまま真の3D誤差とは呼ばない。
既存の[SfM地面高さ不整合診断](../nodes/synthetic_data_generation/000023-run-court-sfm-all-scenes-drift-20260920.md)
も独立の絶対GTではなく、教師の不確実性として扱う。

診断表には以下をscene/shape/center/split/密度birthごとに残す。
位置XYZ・方位仰角・HFOV/fx/fy・注視offset・target切替数・軌道長・隣接距離分位点・
target/全courtの可視KP・full/near_full/partial・採択率/棄却理由・exact/near重複率。
nominal HFOVの層名を実測coverageと同一視しない。candidate/accepted/selected間の分布差を並べる。

後続の小規模描画は4scene各24–48枚程度を上限候補に、支持の厚い/薄い高さ・hull境界・
target切替・全体/部分・QA不良候補を層別に選ぶ。RGB単体とKP/線/target/depthのoverlay、
同じ軌道の3D表示と短い動画で良例・失敗例を人間が見る。
閾値候補を厳しく/緩くしたときの棄却率・残る分布・目視失敗の取り逃しを比較してから固定する。
CPU runのsaved RGB動画は既存視点の参考であり、この新候補の画質合格証拠ではない。
対象courtが0点でも画面外の負例として有用な可能性はあり、それだけで不良とは断定しない。
負例を許容するか、対象4点以上へ限定するかもD5で選び、全条件で同じ規則を適用する。

## 7. manifest、出力、Colab分割と再開（論点8）

入力/出力は明示的なschemaを持つ。以下は**責務の案**で、名前・JSON keyは#1352/#1354と合意後に確定する。

| 成果物 | 内容・不変性 | 所有者 |
|---|---|---|
| asset manifest | scene/checkpoint/renderer/runtime config、public export、alignment、metric adapterとhash。取得時にも照合 | #1351 |
| evaluation registry | 全sceneのval/test family・撮影点・採択ID・順序・seed・guard・geometry/QA版とhash | #1352計画、#1356最終確認 |
| candidate plan | asset ID、policy版・全数値/分布、seed、軌道/view/弧長表/N0/点ID、split、候補上限、guard不採択 | #1352 |
| condition request | sceneごとの目標**train**数、A1のblock/A2のk、共有baseline/eval ID、計画hash。描画前に固定 | #1352 |
| QA ledger / render cache | point ID→geometry/renderer key、実測QA、採択/不採択理由、JPEG/疎教師とhash、完了record | #1352の契約を#1355が実行 |
| selection manifest | 目標数と実際の候補・描画・採択・選択数、sceneごとのordered selected ID、実G/d分布、store参照とhash | #1352選択、#1355公開 |
| immutable dataset version | 条件request、selection/evalへのhash参照と検証済みstores。一意のrootから全sampleを解決 | #1355生成、#1354読込 |

データ版は既存workspaceの`scenes/<id>/datasets/court`の外に置く。
例は`<explicit-output-root>/<dataset-version>/...`で、暗黙の固定owner探索をしない。
#1354はデータ版manifestを1つ受け取り、train選択と固定eval storesを解決し、欠損/異版を拒否する。
既存`sample_store`のJPEG shard＋sparse index・v3幾何を基本とする。
同じJPEGを各条件に複製する必要はないが、現行readerは条件ID参照を解決しないため
**選択adapterは新実装が必要**。保存順indexとstable sample IDを混ぜない。
幾何schemaを変える場合は既存v3 parserへ新fieldを黙って混ぜず、明示した計画schema/metadata境界で扱う。
条件・ledger・storage indexは責務が異なり、教師のコピーを複数のauthorityにしない。

split familyは評価漏れを防ぐ単位で、処理shardはGPU/VM/保存の単位。
高密度の1familyを複数shardへ分けてもsplitは同じままにする。
逆に1shardに複数familyが入っても各pointのsplitは保存する。
shard IDは固定sample ID集合/順序とplan/renderer fingerprintから求め、worker数や総shard数で振り直さない。
固定した最大件数/byte予算で分け、解像度別にも分ける。GPU microbatchはさらに小さく分ける。
NHTは呼出しごとにcheckpointをloadし、内部batchでcameraを処理するので、
VRAMを増やさずload回数を減らすには呼出しcamera shardとGPU batchを別々に調整する。

```text
verify pinned inputs -> plan/request frozen -> each shard:
  inspect durable completion + all hashes
  complete exact shard: reuse
  incomplete or conflicting shard: quarantine/fail; explicit retry in fresh staging
  render bounded microbatches -> QA -> pack JPEG+sparse -> strict reread
  upload data+QA+hashes -> verify remote -> publish shard completion last
all required shards verified -> select exact scene quotas -> validate disjoint/fixed eval
  -> publish dataset-version completion last -> release renderer before training
```

plan、asset、renderer、QA版の不一致は別shard/別データ版。古い完了recordで成功扱いにしない。
途中JPEGだけ残る・dataだけupload済み・completionだけ残る・retry重複・hash不一致をテストする。
worker完了順やretryで採択順を変えず、計画順で選択する。
現在のattempt-local shard再利用を、Driveを含む永続transactionと同一視しない。
Drive操作・Colab導入手順は#1351/#1355の責務で、この設計へ複写しない。

## 8. 時間・メモリ・保存量の見積もり

計画はcandidateをstream生成し、1軌道の固定弧長表と必要なID列だけ持つ。
parameter生成はO(G)、弧長表はO(GD)、点計画はO(P)、投影はO(P×court数)。
guardは空間index＋局所比較、pair全探索をしない。ID/QA ledgerはchunkで永続化する。
prototypeは検査を容易にするためdictを保持する実験コードで、productionのstream実装ではない。

float32 RGB+alpha+depthだけで`W*H*5*4` byte/画。
既存959×539では約9.86MiB/画、600画なら約5.78GiBで、preview/コピー・checkpoint/GSは別。
全24000画なら約231GiBとなるため、一括保持しない。GPU batchは最初1–4cameraで測り、
host raw配列は1cameraずつQA/packして解放する案。
CPU/Colabの上限は実測peakと余裕から定める。600をGPU batchへ流用しない。

時間は`Σ(load_count_s * load_time_s + rendered_s * seconds_per_frame_s) + QA + pack + transfer`、
保存量は`Σ(selected_or_cached_s * measured_JPEG_sparse_bytes_s) + indices + QA + manifests`。
一時領域は「未packの処理chunk数×frame raw bytes＋モデル＋転送中コピー」のpeakで見積もる。
GPU時間は描画と学習を別計上、評価も別。rateは全体平均だけでなくscene/層ごとの下限で予算化する。
186000は選択済み集合の理論値で、候補の棄却や予備枠を含まない。
本runはGPU時間/JPEG平均byte/Drive帯域を測っていないため、時間や保存容量を実測値として捏造しない。
具体値・最大規模・更新数は#1356のColab試走で固定する。

## 9. 検証計画と完了条件の対応

| 原要件 | 今回の証拠・提案 | 残る確認 |
|---|---|---|
| 論点1/2、完了条件1（方式比較） | §1/2、実sceneの領域・旧候補数、比較表 | ユーザーの方式/分布合意 |
| 論点3/6、完了条件3（不変性・漏れ） | §3/4、CPU共通点のK/pose/注視点/投影一致、固定eval、guard実測 | 連続曲線の数値誤差bound、renderer教師、全eval規模の最終診断 |
| 論点4/7、完了条件2（枚数・配分） | §5、CPUのA1/A2全5段/B計数・不足fail・長さと仮想QAのずれ | 全group floor/quota、最終gate、実QA採択率による容量確認 |
| 論点5、完了条件4（品質根拠） | §6、3D表示・既存RGB overlay・CPU動画へのrun参照 | 新候補の小規模描画、良例/破綻例からの閾値合意 |
| 論点8、完了条件5（実行境界） | §7/8、manifest/処理単位/容量式、後続担当の接続点 | #1352/#1354/#1355の実装、Drive中断再開と実測 |

後続CPUテストでは、非整数L/d・min count・multiple・矩形角・nonplanar・k拡張、
seed/列挙順/shard/worker数を変えた共通点、val/test/trainの派生family/近接衝突、
rate=0・特定層全棄却・quota超過/不足・B3のblock途中選択・最大A縮小時B容量不足を検査する。
不足で条件や点を変更せず明示的に失敗することを確認する。
renderer結合では共通pointのRGB・可視性・疎教師のcache再利用、JPEG後の厳格再読込、
既存v1/v2/v3/sfm_boundedの回帰を確認する。
精度スケーリング・Colab・本学習・資産可搬化は今回の検証対象外。

## 10. ユーザーへ戻す判断

実装へ進むには下記を決める。提案を既定値として本番へ入れない。

| 判断 | 推奨と理由 | 代替と影響 / 次の材料 |
|---|---|---|
| D1 方式・分布 | 層別counter列＋固定弧長表の2倍細分。§2の共通分布候補から試走 | Sobol列、旧格子延長、撮影近傍。棄却後の分布保証は別途必要 |
| D2 領域・画角 | 共通hull中心/0%拡張/.5m内側、75–110度の3層を比較の出発点にする | 5%拡張・45–90度・scene別設定の維持。B01とB02を含む動画・分布で判断 |
| D3 枚数と軌道数 | exact train枚数を優先し、固定block内の予備と採択後選択。A2は軌道を追加しない | 本数の厳密倍率/全点使用/選択後の等間隔を優先すると枚数を変更する必要がある。全G本使用ならfloor、軌道別の倍増ならquotaも合意 |
| D4 重複・固定評価 | family予約＋独立familyへのnear guard、全scene共通の固定registry | 位置だけの保守guardかpose/KP併用か、0.5/1m等の感度動画。eval数は#1356で最終確認 |
| D5 品質gate | 対象courtのKP支持とROI診断を追加し、数値は4scene小規模描画後に固定 | 既存gateのみではtarget支持・白線品質の保証が弱い。追加gateで不足/偏りも起こる |
| D6 jitterと動画 | point IDごとの独立disk jitterを比較基準にする | 滑らかな連続jitter/0m。連続動画の見え方と分布が変わるので合意する |

D1–D6の選択と新規視点の画質根拠が未確定のため、#1350はcloseせず親へ提案を返す。
評価枚数・最大A・共通更新数・追加seed/長期学習対象・Bの組合せ補足は#1356へ引き継ぐ。
最終合意は親が#1343へ記録する。本作業からGitHubへ書き込まない。
