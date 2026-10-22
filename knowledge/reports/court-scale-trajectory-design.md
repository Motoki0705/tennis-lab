# Courtデータ規模比較の軌道・サンプリング設計

**#1350 の設計成果物 / 親 #1343。2026-10-22更新。**
D1/D3/D4と評価の扱いは同日の[親issueの合意](https://github.com/Motoki0705/tennis-lab/issues/1343)に従う。
比較する5方式と共通契約を提示し、採用方式およびD2/D5/D6の最終選択は
#1352のCourt Review Web UIでユーザーが行い、親が#1343へ記録する。
既存生成器・presetの仕様は変更していない。採用後に#1352が実装仕様を
[Court dataset README](../../src/synthetic_data_generation/dataset/court/README.md)へ移し、
この文書は判断根拠と実験記録への参照を残す。実装後の仕様を二重管理しない。

CPUでの観測の正本は[計画診断run](../nodes/synthetic_data_generation/000024-run-court-nested-plan-cpu-20261022.md)、
新規視点2080枚のQA・比較動画・良例/失敗例の正本は
[描画診断run](../nodes/synthetic_data_generation/000025-run-court-novel-view-quality-20261022.md)。
後者の既存成果を今回再点検した。再描画・新方式の本実装・学習は行っていない。

## 0. 合意済みの変更と設計工程の完了範囲

| 論点 | 2026-10-22のユーザー判断 | この設計・後続への反映 |
|---|---|---|
| D1 軌道生成 | 複数候補を実装し、既存Web UIでユーザーが判断する | §2の5方式を#1352で比較。単一方式を採用扱いにしない |
| D3 枚数 | 「候補を全部使い、枚数は実測値」 | §4。品質検査を通った全候補をtrainへ入れる。枚数合わせのquota・選択・補充は設けない |
| D4 評価 | 合成はtrain、既存実画像は評価専用のval/test | 合成val/test・評価registry・評価guard・familyによる漏れ防止を設計から削除。実画像の動画単位分割は#1354 |
| pose | 学習するが主評価から外す | 合成のpose教師を維持。主比較は実画像のKP / SEG / LINE / semantic LINE |
| D2 / D5 / D6 | 新規視点の描画・動画を見て判断 | §5の推奨と診断材料を提示。#1352で方式と併せて最終確認し、本番生成前に固定 |

旧CPU prototypeに残る固定評価・近傍guard・仮想QA・採択後選択は、旧案を検査した歴史的証拠である。
その結果やコードを今回の採用仕様へ書き換えず、不変性の確認だけを共通基盤の参考にする。
更新済み#1350の受入条件は方式候補・仕様・計数・診断・接続点が揃うことで満たせる。
D2/D5/D6の最終判断は#1352の到達点であり、#1350を閉じる妨げとはしない。
本番生成を許可する合意とは区別する。

## 1. 現行方式の確認と補足

参照版は `7e307f0499be3a2511afaba51966ae3362ec81b1`。行番号はこの版に対するもの。
以下は提案の推測ではなくコードの挙動。

| 確認点 | 根拠 | 設計への影響 |
|---|---|---|
| 4096区間の折線からLを近似し、`max(minimum, ceil(L/step))`をmultipleへ丸め、最大隣接距離が収まるまで加算する | [sampling.py:21,37–74](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/sampling.py#L21) | 半分のstepでも点数がちょうど2倍にならず、共通点の包含を保証しない。`ceil`だけの問題ではない |
| minimum countはminimum group数から導き、count multipleにshard数を使う | [contracts.py:798–820](../../src/synthetic_data_generation/dataset/court/contracts.py#L798) | group下限・処理shard数だけの変更でもgeometryが変わり得る。新計画のN0を処理分割から独立させる |
| v2/v3のjitterは`sample_index=len(samples)`を渡す | [selection.py:1019–1071](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/selection.py#L1019)、[targeting.py:122–138](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/targeting.py#L122) | 既存点より前への追加、前の軌道の高密度化、再選択で姿勢が変わる。**末尾への単純な追加だけなら以前のindexは変わらない**。issueの「軌道を足すと」は条件付き |
| 選択されたgroupの位置からcoverage・HFOV・注視高さ・view IDを決める | [selection.py:858–889](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/selection.py#L858) | jitterを直すだけでは不十分。viewも軌道の固定IDから決める |
| 全groupをsort→shuffleし、件数依存のfraction丸めでsplitする | [selection.py:589–624](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/selection.py#L589) | 旧契約の制約。新profileは全てtrainとし、この分割を呼ばない |
| 半径scaleはlinspaceの3値、その他は有限の形状・向き・高さ・振幅格子。typed sort後に番号を振る | [trajectory.py:148–245](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/trajectory.py#L148) | 数値範囲を広げることと本数だけを増やすことは別。circleの向き違い等はtyped keyが違っても物理的に同じ曲線になり得る |
| 全候補を距離サンプリングしてからcoverageを予算内で選ぶ | [selection.py:365–457,476–567](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/selection.py#L365) | 候補格子の単純拡大は計画メモリ・計算量も増やす。既存greedyの結果を条件ごとに作り直してprefixになる保証もない |
| proposal 4800、8 shard、各600frame、1800秒、最大8 NHT呼出し | [base.yaml:39–56](../../src/synthetic_data_generation/configs/dataset/court/base.yaml#L39) | 1シーン24000枚以上には固定上限が合わない。600はNHTのGPU microbatch数とは別 |
| groupを分割せずshardへ入れ、合計容量超過・単独group超過で失敗 | [selection.py:627–690](../../src/synthetic_data_generation/dataset/court/components/camera_sampling/selection.py#L627) | 密な1軌道だけでも600を超える。軌道groupと処理shardを別にする |
| pre-renderでは全コート合計4点を条件にする | [court/rendering/nht.py:251–267](../../src/synthetic_data_generation/dataset/court/rendering/nht.py#L251) | 対象コートに4点ある保証ではない。学習がtarget_court専用なので、対象単位の診断が必要 |
| α/depth由来の可視点が全体で0なら棄却。JPEG化前の配列で検査し、配列は保存しない | [assembler.py:456–491](../../src/synthetic_data_generation/dataset/court/assembler.py#L456) | 既存acceptedは新しい画質基準の合格を意味しない。元のalpha/depthは再描画なしでは検査できない |
| visibilityは近傍にα≥0.01かつdepth>0があるかを見る | [labels.py:885–1018](../../src/synthetic_data_generation/dataset/court/components/labels.py#L885) | 投影KPの期待depthとの差や白線との一致を判定していない。alphaは画質の十分条件でない |
| release時は最低枚数・率、group非空、full/near_full/partialと全semantic classの出現を検査 | [assembler.py:496–539](../../src/synthetic_data_generation/dataset/court/assembler.py#L496) | 新profileのtrainのみ・実測枚数に対応した公開判定が必要 |
| shard再利用はattempt tokenと完全一致する計画に束縛 | [court/rendering/nht.py:92–146](../../src/synthetic_data_generation/dataset/court/rendering/nht.py#L92) | VM消失後の永続再開をそのまま既存実装済みとは扱わない |
| JPEG quality95＋疎教師を保存し、descriptorは最後に公開 | [sample_store.py:24–71](../../src/synthetic_data_generation/dataset/court/sample_store.py#L24) | 保存基盤を再利用し、条件参照と描画cacheの責務をその上に追加する |

入力はメインtreeのB00〜B03のpublic export・accepted alignmentを読み取りだけで使用した。
B01/B02のhuman-confirmed形式も`AlignmentResult.from_dict`で読めた。
B01の保存済み設定だけは`sfm_complex_center_on_hull`が欠け、現行の厳格な設定parserが拒否する。
prototypeでは**B01に限りfalseを明示追加したコピー**で旧候補数を診断した（入力は変更なし）。
これは資産可搬化の解決ではない。#1351が扱う変換・検証と混同しない。

## 2. Court Reviewで比べる5方式（論点1・2・7、D1）

以下は#1352の**比較用recipeの仕様案**。数値は本番採用値ではない。
方式・recipe版・seedを明示して選べるようにし、暗黙の方式切替をしない。
最初の4方式は要求された比較対象、5番目は既存の角のある閉曲線を補う候補である。
共通の描画・教師・計数・閲覧基盤を使い、方式間の画像数を揃えるための間引きはしない。

| method（機能名） | 生成するもの・比較する意義 | 長所の仮説 / 短所 | UIで見る点 |
|---|---|---|---|
| `grid_budget_replay` | 保存presetの有限格子＋既存coverage予算選択を再現する基準 | 既存経路との差が分かる / 有限候補、予算選択による偏り、拡張限界 | 元の軌道が再現されるか、形状・中心・画角の偏り、候補枯渇 |
| `stratified_counter` | 固定層の順にID由来の独立乱数で閉曲線を追加 | 任意長のprefixと追跡が容易 / 小prefix・品質棄却後の均衡は保証しない | 本数追加で観測領域が広がるか、特定層の棄却集中、重なる曲線 |
| `sobol` | 同じ層・同じ周辺分布で連続パラメータをscrambled Sobol列から得る | 少数での空間被覆を期待 / 2の冪以外や棄却後の均衡保証は使えない | counterと同じ本数で半径・向き・高さの穴が減るか、採択後にも残るか |
| `capture_local` | 公開撮影カメラの近くを短い開線分で移動 | 撮影支持に近く画質を保てる可能性 / 視点多様性が小さく近重複が多い | ネット・地面・線の鮮明さ、近接画像の反復、カメラ姿勢も撮影支持に近いか |
| `open_bezier` | 撮影点近くを結ぶ3次Bézierの滑らかな開曲線 | 矩形の角を避け、局所線分より広く観測空間を使える可能性 / hull内の空隙やtarget切替は残る | 経路の折返し・隙間、弧長点間隔、target切替、全体/部分コートの変化 |

### 共通の座標・領域・view recipe

accepted courtのmetric座標を使い、公開カメラも`MetricSceneAdapter`で同じ単位へ移す。
3DGSの正規化座標をmと扱わない。alignment・public export・renderer/checkpoint・コード版をpinする。
共通試走案はhull拡張0%、内側margin 0.5m、HFOV層75/92.5/110°、注視高さ0/1.5m、
点ID由来の半径1m・面積一様disk jitter。D2/D6の代替は別recipeとして明示する。
局所/開曲線でも同じHFOV・注視高さ・jitter recipeを使い、撮影K/姿勢との差を診断する。
`grid_budget_replay`だけは比較元の保存view分布を使い、その違いを画面へ表示する。

境界は基準court plane内のcaptured-camera XY hullを用いる。
全curveに境界を課し、sampleだけ内側でも合格にしない。既存の閉形状は
`sfm_bounds.py`の支持関数を再利用できる。hullが橋渡しする未観測空隙、高さや視線の外挿は保証外。
許容領域・支持camera数・撮影高さの分位点・最近の撮影カメラとの距離/視線角差を併記する。
実行不能な中心・空の候補層・非finiteな幾何を、中心移動や領域拡張で黙って修正しない。

court中心の閉軌道はそのcourtをtargetとして固定する。
complex中心、`capture_local`、`open_bezier`は各点の3D最近courtをtargetとし、
既存v3と同じ等距離tie規則で決める。視線はtarget courtのlocal `(0,0,look_height)`＋jitterへ向ける。
撮影cameraの元姿勢をそのまま教師へ流用しない。near/far曖昧は明示棄却する。
独立jitterとnearest target切替を含むので、位置が滑らかでも姿勢が滑らかとは限らない。

### `grid_budget_replay`：既存方式の再現範囲

各sceneの保存configを入力hash付きでpinし、同じtyped格子・coverage選択・proposal予算から
選択済み軌道とviewを一度生成して固定する。shape/center/radius/height/HFOV等の重みは
保存configと選択結果そのもので、連続一様分布へ置き換えない。
B01の45–90°等のscene差も残し、方式差と共通分布化の差が混ざる基準であると表示する。

凍結した選択順のprefixを比較に用い、要求本数ごとにgreedyをやり直さない。
候補格子や予算を拡げた選択は別recipe。既存凍結列を超えた本数は不足として明示する。
新しいstable ID/seedと§3の2倍細分は共通基盤として適用する。
再現対象は**有限格子と予算選択による経路/view**であり、旧通し番号jitter・旧split・旧サンプリングを
使った過去RGBのbyte再現ではない。元の公開画像も参考として同じsceneの別データ版から閲覧する。
この基準だけで16Gの全系列を作れるとは前提にしない。

### `stratified_counter`：固定層と分布

層は形状4種×center kind 2種×垂直profile 2種×HFOV 3層×注視高さ2層の96組。
形状はcircle/ellipse/rectangle/superellipseを等重み、center kindはcomplex/court各1/2、
court側はID順の各courtへ層内counterで巡回して等分する。
complex中心は全scene共通でcaptured hullの頂点平均。court中心は実際のaccepted位置。
96組の順序はrecipe・scene・seedから一度固定した順列を繰り返し、要求本数で並べ替えない。
小prefixで全joint層が埋まらないことも記録する。

連続値はscale U[0.65,0.95]、非円axis ratio U[0.35,0.8]、向きU[0,π)、
base height U[1.5,3]m。planarは振幅0、sinusoidalは振幅U[0.05,0.5]m・1周期・phase U[0,2π)。
circleはratio=1・向き=0へ正規化する。scaleはanalytic fitting後の最大許容半径に掛ける。
乱数は`hash(recipe, scene_geometry_id, generation_seed, stratum, counter, field)`から独立生成する。

棄却は元のordinalと理由を残し、その層で採れるまで引き直さない。
全候補のprefixを固定してから品質判定するため、低採択層は枚数減少・分布差として見える。
各層の採択数を均すreservoirや補充はD3に従って置かない。

### `sobol`：同一分布で列を交換する比較

層・離散順序・中心・viewと周辺分布はcounter方式と同じ。
層ごとに`scipy.stats.qmc.Sobol(d=6, scramble=True, optimization=None)`を1つ置き、
座標をscale / ratio / orientation / height / amplitude / vertical phaseの固定順に対応させる。
使わない座標も詰め直さず、circle/planarの正規化後に無視する。
scramble seedはscene・generation seed・層・recipeから決め、各層counter番の点を直接参照する。
拡張時にscrambleや最適化をやり直さない。利用したSciPy版・bits・方向数実装を固定する。

[SciPy公式資料](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)は、
2の冪から外れる件数・先頭のskip・間引きではbalanceを失うと説明する。
必要数を含む2の冪の列を生成してprefixを取れても、そのprefixや棄却後集合の保証は主張しない。
Sobolは軌道パラメータの列であり、軌道上の弧長サンプリングを置換しない。

### `capture_local`：撮影カメラ近傍の局所軌道

公開camera IDとgeometryをpinし、`hash(recipe, seed, camera_id)`順にanchorを巡回する。
全anchorを1周する前に同じanchorを重複選択しない。2周目以降はcycle番号をfield乱数へ含める。
camera XYがinset hull内にあるanchorだけを有効とし、無効anchorも理由と元IDを記録する。
anchorのmetric高さをそのまま使うため、高さ分布は撮影支持に従う。

各anchor `p`でXY方向`u`をU[0,2π)から決め、`C(t)=p+(t-1/2)ell*u`の開線分を作る。
`ell_max`は両端がinset hull内に収まる対称線分の全長。
長さは`U[0.5, min(2,ell_max)]m`、上限が0.5m未満ならその候補を棄却する。
端点がconvex hull内なら全線分も内側。zはanchor高さ一定、垂直振動は入れない。
経路中心はanchorであり、court/complex中心のorbitではない。
HFOV・注視点は共通recipe、弧長・IDは§3を使う。短軌道の近接点過多は採択後の間引きで隠さない。

### `open_bezier`：撮影支持を結ぶ滑らかな開曲線

inset hull内にある撮影anchorからXY距離2–8mの異なるpairを列挙し、
両camera IDから作る固定hash順を候補列とする。順序は全体の要求本数に依存させない。
`P0/P3`はpairのmetric位置、`P1=(2P0+P3)/3+delta1`、`P2=(P0+2P3)/3+delta2`とする。
各deltaのXYは半径0.5mの面積一様disk、zはU[-0.25,0.25]mからfield別のID乱数で決める。
全control pointがinset hull内、正の高さ、正則な接線を持つことを検査する。
不適合pairを別pairへ自動交換しない。候補列が尽きたら不足を表示する。

`C(t)=(1-t)^3 P0+3(1-t)^2 t P1+3(1-t)t^2 P2+t^3 P3`。
係数が非負で合計1なので、全control pointがconvex inset内なら曲線全体も内側である。
中心の表示値は4control pointの平均、高さはcontrol点から決まる曲線であり、独立のorbit高さ分布ではない。
短い滑らかなspline候補として単一segmentから始め、複数segmentの接続規則は今回含めない。
距離2–8mでも未観測空隙を跨ぐ可能性は残り、全周の被覆は閉曲線より小さくなり得る。
現行`OrbitShape`は4閉形状だけなので、#1352で明示的な開曲線descriptor/validatorが必要。
円等の既存shapeへ偽装して保存しない。

高所・遠方の放送風軌道は最初の5方式に含めない。
CPU診断のcaptured-camera高さ・hull支持から離れる視点は今回の描画で検証しておらず、
局所方式と閉曲線の比較に加えて未観測高所まで同時に変える根拠が不足するためである。
UIで必要性が見えた場合は、撮影支持と追加描画を調べ、D2の新recipeとして親へ提案する。

## 3. 弧長の2倍細分とstable ID/seed（論点3）

方式ごとのcurveを固定3D弧長表へ変換し、長さL、基準距離dから
`N0=round_up(max(N_min,ceil(L/d)), multiple)`を基準時に一度だけ決める。
N_min・multiple・積分精度はrecipeに記録し、処理shard数から独立させる。
閉曲線も開曲線も**半開区間q∈[0,1)**の`N0*2^k`点とし、開曲線の終点q=1は撮影点に含めない。
UIの計画曲線は終点を表示できるが、始点へ閉じない。終点までの未撮影距離も記録する。
こうすれば全方式で候補点数を正確に2倍にできる。両端を含む別仕様を暗黙に混ぜない。

```python
arc_map = freeze_arc_map(curve, tolerance, algorithm_version)
N0 = freeze_base_count(arc_map, base_step=d)  # n, k, shard数に依存させない
for j in range(N0 * 2**k):
    q = Fraction(j, N0 * 2**k)  # hash前に既約分数へ。floatをhashしない
    point_id = hash(scene_geometry_id, recipe_id, trajectory_id, view_id, q)
    position = arc_map.at(q)
    target = resolve_target(position, pinned_layout, target_policy)
    jitter = field_rng(generation_seed, point_id, "look_at")
    K, pose, look_at = make_camera(view, target, position, jitter)
```

粗点jと細点2jは同じ分数・同じ固定表・同じIDへ問い合わせる。
N0が2の冪である必要はない。全kごとにceilや4096分割の精度補正をやり直さず、
新しい奇数点だけを計算し、共通geometryを再利用する。
弧長積分の収束・最大隣接距離・矩形角・nonplanarの検査が不十分なら公開前に停止し、
表/N0を新版で固定する。公開済み版へ後から補間精度を変えない。

trajectory IDはgeometry recipe・scene・generation seed・元の層/counterまたはanchor ID・curveパラメータに結び付ける。
view IDはHFOV・注視高さ・target規則・jitter規則に結び付ける。
point IDは上記と既約qから作り、birth levelは別属性とする。
point IDのrecipeはcurve/view/弧長のgeometry recipeのみで、QA版・閾値・採否は含めない。
`sample_index`・保存順・frame番号・最大n/k・worker数・shard番号・採択数をID/seedへ混ぜない。
version付きcanonical JSON、float64のシリアライズ、SHA-256全長、分数の分子/分母、
PCG64/SeedSequenceとNumPy版を固定し、衝突は停止する。
[NumPy公式資料](https://numpy.org/doc/stable/reference/random/parallel.html)のID＋root seedの組合せを参考に、
fieldごとに独立のseed namespaceを使い、field追加で他の乱数をずらさない。
方式・recipeが違う点を同一point IDとは扱わず、D2/D6比較の対応は別のpair IDへ記録する。

現行jitterのvalidatorも`sample_index`から注視点を再計算する。
#1352は生成側と読込/検証側を同じ明示的なRNG契約へ接続し、旧v3/presetの分岐は維持する。
旧fieldへ新意味を流し込まず、新plan/seed schemaとgeometry readerの対応を宣言する。
v3のKP14/canonical poseの意味を変更する提案ではない。
familyによるsplit単位は設けず、派生元IDを残すなら重複診断・来歴だけに使う。

CPU prototypeの共通672点はID・K・pose・注視点・target・投影教師が一致した。
5方式すべてやrenderer教師の不変性を検証済みという意味ではない。
共通pointのRGB・renderer visibility・疎教師・採否は同じrenderer keyのcacheから共有する。
GPU/バッチ差の再描画byte一致をCPU計画から主張しない。Colab差の照合は#1351。

重複はID一致と物理的同一性を分ける。円の向き、閉曲線の循環shift/逆向き、同じ位置の複数viewを
正規化したcurve/view fingerprintでexact重複として計画時に記録する。
同じ画像を別IDで水増ししない。near重複は位置・視線・HFOV・共通KPの投影距離の診断とし、
A2が意図して増やす近接点を間引かない。先行候補への重複判定を固定し、後のn/kで既存点を消さない。
評価guardのためのnear閾値やfamily予約は置かない。

## 4. A1/A2/Bの統制と全採択候補の計数（論点4・7、D3/D4）

固定recipeの候補を`C_s(n,k)=union_{i<n} points(trajectory_i,k)`とすると、n/kを増やして包含できる。
固定した点単位の幾何・品質predicate Qと同一cacheを使えば、採択集合`E_s=C_s∩Q`も包含する。
全E_sをtrainへ入れる。条件全体のrelease判定は公開可否だけを決め、quotaのため点の採否を変えない。
0件の条件、壊れた入力、未確定の品質recipe、未完了shardは明示的に失敗する。
個別軌道や層が空になったことは0件として診断し、他の軌道/層で補わない。

記録は`logical_records`（重複view参照を含む場合）、`candidate_unique`、`geometry_pass/rejected`、
`rendered_unique`、`render_cache_reused`、`qa_accepted/rejected`、`pending/failed`を区別する。
正常完了時は`candidate_unique=geometry_rejected+geometry_pass`、
`geometry_pass=qa_accepted+qa_rejected`、`pending=failed=0`が基本。
描画前棄却も描くstress probeは例外フラグを付け、通常の計数式と混ぜない。
重複ID除去・cache共有を含むので、論理record数・GPUで描いた枚数・学習枚数は異なる。

```python
freeze_request(method, recipe, scenes, trajectory_prefixes, density_level, seed, asset_ids)
for point in planned_unique_ids_in_stable_order():
    if not valid_geometry(point):
        record(point, "geometry_rejected", reason)
        continue
    result = verified_cache_or_render(point, pinned_renderer)
    record(point, QA(result, fixed_policy))
require_no_pending_or_failed_points()
train_ids = ordered_all_accepted_ids()  # 目標枚数でsliceしない。補充もしない
if not train_ids:
    fail_with_counts_and_reasons()
publish_actual_counts_and_store_references(train_ids)
```

### 実測の小規模計数例

新規描画runの推奨候補からの例。D5の「対象4KP」案に合格する**全画像**を数えた場合であり、
学習用データ版を公開した結果ではない。全候補を描く診断なので幾何落ちも描画数に含む。
詳細・他3条件は[残存成果監査](../runs/run-court-novel-view-quality-20261022/resume-audit.json)を参照。

| scene | logical record | unique候補 | 既存geometry pass | 実描画 | 既存gate採択 | 対象4KP案の全採択 |
|---|---:|---:|---:|---:|---:|---:|
| B00 | 132 | 130 | 113 | 130 | 113 | 106 |
| B01 | 132 | 130 | 113 | 130 | 113 | 111 |
| B02 | 132 | 130 | 120 | 130 | 120 | 120 |
| B03 | 132 | 130 | 113 | 130 | 113 | 113 |
| 合計 | 528 | 520 | 459 | 520 | 459 | 450 |

この例で1500枚/sceneを要求して増補する操作はない。横軸は450枚、scene配分は106/111/120/113となる。
軌道長が違うA1は候補も正確な倍数にならず、A2は候補を2倍にできても採択の2倍は保証しない。
QA後のgap・0件軌道・採択した軌道本数も実測する。curve/viewを固定しても、品質棄却後の分布は変わる。

### 主実験の目安表との対応

`g_s`をsceneごとの基準候補軌道本数、`G=sum_s g_s`を全4sceneの基準とする。
基準dと各g_sは#1356の試走で固定し、採択枚数から実行中に本数を逆算し直さない。

| 系列 | 変えるもの | 固定するもの | 目安表との関係 |
|---|---|---|---|
| A1 r=0..4 | 各sceneの候補本数`2^r*g_s` | d、分布/recipe、資産、seed、共通点とQA | 6k/12k/24k/48k/96kは目安。実測枚数をx軸にする |
| A2 r=0..4 | 同じg_s本の密度段k=r | curve/view、N0、弧長表、seed、共通点とQA | 候補間隔は`L/(N0*2^k)`。目安枚数へ間引かない |
| B | scene集合、各sceneの事前計画prefix長 | k=0/d、同じ生成recipeとQA、共有point | 24kと24k/12k/8k/6kのscene配分は目安。全採択数と配分を併記 |

A1/A2のr=0は同じ候補・同じ全採択IDを共有し、主Aは引き続き9条件。
Bの4scene版はA1 r=2の**データ版まで共有**する。
Bの1/2/3sceneは1scene当たり約16g_s/8g_s/(16g_s/3)本を候補とする計画案で、
整数化を含む実際のprefix長は#1356で本実験前に固定する。
8kを得るために軌道の途中を切ったり、棄却分を補充したりしない。
B各条件はA1最大のscene prefix内に収まる場合に再利用できるが、B相互の包含は要求しない。

**D3によりBの総枚数固定は保証しない。** 親本文の旧見出し・初期表や残る子issueの旧表現より、
2026-10-22の全採択・実測の合意を優先する。採択差が大きければシーン数と枚数の影響が混ざる。
同枚数の厳密比較や既存点の間引きを追加したことにせず、実測枚数・分布と解釈の限界を報告し、
本実験前のprefix計画は#1356で確認する。固定順のscene追加とsceneの個性も分離できない。

全sceneでA1最大/A2最大が同じ基準を含めば、必要なunique採択画像は
`|E_A1max|+|E_A2max|-|E_baseline|`から求める。
旧案の186,000枚は96k/96k/6kを仮定した値で、今回の描画上限や必要画像数ではない。
実際の候補和・QA棄却・試走・再試行は別に数える。
A最大を縮小してBのprefixが収まらなければ、事前にB用延長を別requestで計画するか規模を見直す。
黙って候補を足さず、#1356で計画と費用を確定する。

学習は合成trainのみ、実画像は動画単位で固定したval/testのみ。
旧実画像train/valの動画重複434件は親の合意記録を根拠とし、分割比率・seed・一覧・漏れ検査は#1354が扱う。
pose教師は学習に残し、主評価のKP誤差・SEG/semantic LINE mIoU・LINE Diceにposeを混ぜない。
合成val/testを作る旧split関数・空split拒否を新profileで呼ばないよう、#1352/#1354双方で明示する。

## 5. 品質診断、描画結果とD2/D5/D6の判断材料（論点4・5）

| 段階 | 診断・判定 | 後続の扱い |
|---|---|---|
| 計画前 | accepted layout、metric変換の相互逆、finite/proper rotation、K/解像度、asset hash | 欠損・異版はhard error |
| pre-render | 全curveの境界、高さ/撮影支持、near/far、target/全courtのin-frame/in-front KP、canonical往復 | 不正幾何を明示棄却。対象4KP案等は固定recipeを選んで比較 |
| post-render | RGB/alpha/depthのshape/dtype/finite/範囲、正depth率、target ROIの穴・alpha分位点、可視KP、ぼけ/破綻/線一致 | renderer失敗をQA棄却で隠さない。画質のhard gate値は未確定 |
| 公開 | 全候補の採否/理由、実測採択枚数、0件軌道、coverage/class支持、store/hash、pending=0 | 空の条件は失敗。支持不足は報告し、枚数合わせ・coverage補充をしない |

KP・pose・dense教師は同じgeometryを共有する。denseマスクの別保存は提案しない。
α/depthはJPEG化前に検査し、再開時に使うQA要約・規則版・理由を疎recordと結び付ける。
depthはpublic NHT座標とmetric変換を区別し、期待court depthとの誤差を未検証のgateにしない。
白線のpixel差にはalignmentとrendererの不整合が混ざり、絶対3D誤差とは呼べない。
[SfM地面高さ診断](../nodes/synthetic_data_generation/000023-run-court-sfm-all-scenes-drift-20260920.md)も独立GTではない。

scene/method/shape/center/HFOV/target/密度birthごとに、XYZ・方位仰角・fx/fy・注視offset・
target切替・長さ・隣接距離/gap・geometry/renderer KP・full/near_full/partial・採択率と理由・
exact/near重複を記録する。候補と採択後の分布を並べ、nominal HFOV層を実測coverageと混同しない。

### 既存の新規視点描画から推奨する比較の出発点

全4scene×4条件、各130 unique cameraのstress probeを実描画済み。
連続96点と幾何で選んだ36点のうち2点が重複し、2080枚と容量probe5枚になった。
標本の選び方・閾値感度・動画・良例/失敗例の正本は描画runとそのbundle。
今回、入力32ファイルと再計画36ファイルのbyte一致、34 mediaのhash、動画16本の全frame decodeを再確認した。
この標本は本番分布の採択率の推定でも、5方式の比較実装でもない。

| 判断 | 描画結果に基づく推奨 | 限界とユーザーが見る点 |
|---|---|---|
| D2 領域・画角 | hull 0%拡張/.5m内側、75/92.5/110°を比較の出発点にする | 狭角で全sceneの対象KP支持が減った。5%に画質上の優位は未確認だが、今回の5%点も元hull内で外側は未試験。全体/部分の見え方と線の鮮明さで判断 |
| D5 品質gate | 対象renderer KP≥4を正例支持の候補、ROI穴≤1%を最低限の生成不良検査候補にし、画質は動画と失敗例を併用 | 穴0でもぼけを通す。厳しいα<.9面積閾値はB01/B03のpartialを偏って失うため感度表示に留める。対象0の負例を許すかも未決 |
| D6 注視jitter | 静止画の視線分布を重視するならstable ID独立1mを第一比較候補にし、0mも常に表示 | 0mは特に単一courtのB02/B03で振れを減らしたが、複数courtのtarget切替やぼけは残る。滑らかな動画を求めるなら0mを検討。連続jitterは未描画の別案 |

ROIはtarget physical cornerのpolygonをnear planeと画面でclipし、ROIなしはnullとして扱う。
全16条件でROIの低alpha/非正depth穴率は0だったが、画質の合格を意味しない。
B02/B03は対象4を満たしてもほぼpartialで、全sceneへ対象10以上を一律強制すると大半を失う。
明るいridge距離も暗い細線を捉えられず、白線誤差の自動hard gateには用いない。
D5で採用する規則は方式選択後の全本番条件で固定し、品質不足を余剰候補で補わない。

### ユーザーに見せる動画・画像

このworktreeでの入口（静的な診断mediaの一覧。Court Review本体とは別）:

```text
/home/kamimura/projects/tennis-lab/.claude/worktrees/court-scale-trajectory-design/tmp/court-render-quality/run-01/review/index.html
```

同じdirectoryに、`<scene>`をB00/B01/B02/B03へ置き換えた次の成果物がすべて存在する。

| 材料 | ファイル名 | 見方 |
|---|---|---|
| 推奨候補の連続動画 | `<scene>-recommended-rgb-overlay.mp4` | 左RGB、右教師overlay。破綻・ぼけ・線一致、対象切替 |
| D2-a 拡張比較 | `<scene>-compare-hull5.mp4` | 上0%、下5%。同じ弧長割合で、位置/長さが変わる |
| D2-b 画角比較 | `<scene>-compare-hfov45_90.mp4` | 上92.5°、下67.5°。位置/pose固定、Kだけ変更 |
| D6 jitter比較 | `<scene>-compare-jitter0.mp4` | 上1m、下0m。位置/K固定、注視点変更 |
| 各条件のcontact sheet | `<scene>-{recommended,hull5,hfov45_90,jitter0}-contact.jpg` | 連続動画以外の軌道・端のHFOV層も確認 |
| 良例・失敗例一覧 | `representative-cases.jpg` | 可視KP・ROI穴率と実際のRGB品質を併読 |

全動画はH.264/yuv420p、96frame・12fps・8秒。弧長順の再生で実際のカメラ速度ではない。
codec・寸法・全ファイルhashは[media manifest](../runs/run-court-novel-view-quality-20261022/media-manifest.json)。
撮影支持との位置関係は[CPU計画の3D軌道図](../runs/run-court-nested-plan-cpu-20261022/geometry.png)も参照する。
同図の予約評価軌道は旧案の表示で、今回の合成trainのみの設計には引き継がない。
小さい[支持の多い例](../runs/run-court-novel-view-quality-20261022/cases-clearer.png)と
[失敗・不確実な例](../runs/run-court-novel-view-quality-20261022/cases-failure.png)はGitに残す。
B01 continuous 0、B00 continuous 48は対象4・穴0でもネット/地面がぼける例。
B00 sparse 18は対象1点でも旧gateを通す例、B02 sparse 12は対象0点の負例候補である。
独立したRGB品質正解はなく、「良例」はこの標本内で比較しやすい例という意味に留める。

## 6. manifest・データ版と担当間の接続（論点8）

以下は#1352/#1354が合わせる境界案。実装と同時に正本をCourt dataset READMEへ移す。
既存sceneの`scenes/<id>/datasets/court`を変更せず、明示した別のoutput root以下へ置く。
候補計画の版と、QA完了後のimmutableデータ版を分ける。

| 成果物 | 含むもの・責務 | 担当 |
|---|---|---|
| asset manifest | checkpoint/runtime、public scene/cameras、alignment・metric変換とhash、取得後照合 | #1351 |
| candidate plan | method/recipe、全分布/数値、seed、元ordinal、curve/view/弧長表/N0/point ID、geometry棄却理由 | #1352 |
| condition request | scene集合・各prefix長・k、plan/recipe/資産hash、trainのみ。描画前に固定 | #1352 |
| QA ledger / render cache | pointとrenderer key→採否・実測QA・JPEG/疎教師hash。共通点の一意な描画authority | #1352の契約を#1355が永続実行 |
| dataset version manifest | request/plan/asset参照、実測計数、scene/store参照、全accepted IDのmembershipとhash | #1352契約・小規模出力、#1355本番公開、#1354読込 |

#1354へ渡すのは**データ版manifestの明示path**で、scene固定ownerや「最新」を自動探索しない。
最小の論理契約は`schema=court_scale_dataset_version_v1`、`dataset_version`、`request_sha256`、
`plan_sha256`、`method`、`recipe_sha256`、`generation_seed`、`split=train`、`scenes[]`。
各scene entryは`scene_id`、`asset_manifest_sha256`、`stores[]`、`accepted_ids_path/sha256`、`counts`を持つ。
store entryは明示root相対のpath、dataset descriptorとindexのSHA-256を持つ。
`counts`は§4の候補/幾何/描画/再利用/採択/不採択/未完了を区別する。
`dataset_version`は自己fieldを除くcanonical完了manifestのSHA-256、request hashは描画前の固定入力を識別する。
JSON keyやCLI設定名の変更が必要なら#1352/#1354で合わせてREADMEに確定し、同名で意味を変えない。

membershipはstable sample IDの順序付き参照で、カメラ/教師の別コピーではない。
原則として既存sample storeのJPEG shard＋sparse indexとv3幾何を使い、全条件へJPEGを複製しない。
各条件のaccepted IDが共有storeのsampleへ一意に解決するadapterが必要で、現行readerだけでは未対応。
0件/重複/不明ID、hash不一致、異版、train以外、countsとの不一致、未完了は学習開始前に拒否する。
必要な新metadata/curve/RNG schemaは#1352が宣言し、既存v3へ未定義fieldを黙って混ぜない。
学習側も対応schemaを明示し、互換性がないときは停止する。

実画像val/test manifestは#1354の別契約。学習runは合成dataset versionとその実画像評価版の両方をpinする。
合成の評価storeや評価familyへの参照はない。分割比率や評価listをこの設計に複製しない。

軌道groupは生成・診断の単位、処理shardはメモリ・時間・保存の単位。
1軌道を複数shardへ分けてよく、全sampleのsplitはtrainのまま。
固定件数/byte上限と解像度で処理shardを分け、GPU microbatchはさらに小さくする。
shard IDはsample IDの固定集合/順序＋plan/renderer fingerprintに結び付け、worker数で振り直さない。

```text
verify pinned assets -> freeze plan/request -> for each shard:
  exact durable completion + hashes: reuse
  incomplete/conflicting: fail or explicit retry in new staging
  bounded rendering -> QA -> pack JPEG+sparse -> strict reread
  upload data+QA+hashes -> verify remote -> publish completion last
all required shards verified -> count all accepted and validate memberships
  -> publish dataset-version completion last -> release renderer -> training
```

#1355はDriveのdataだけ/完了だけ/途中JPEG/retry重複/VM消失/hash不一致を検査する。
採択ID順をworkerの完了順で変えない。asset/renderer/QA版の不一致は別データ版であり、
旧attempt-localの再利用を永続再開済みの証拠とはしない。導入・Drive手順は#1351/#1355が所有する。

## 7. Court Reviewの現状と#1352で必要な拡張

調査対象は[Court Review README](../../src/synthetic_data_generation/dataset/court/review/README.md)と次の実装。
ここではUIを変更していない。

| 現状の根拠 | 今できること・制約 | #1352の接続点 |
|---|---|---|
| [review_court_dataset.py:44](../../src/synthetic_data_generation/scripts/review_court_dataset.py#L44) | CLIはdata-root / scenes-root / portのみ。manifest指定はない | 明示的な候補データcatalog/データ版manifestを入力できる入口 |
| [service.py:66](../../src/synthetic_data_generation/dataset/court/review/service.py#L66)、[95](../../src/synthetic_data_generation/dataset/court/review/service.py#L95) | `scenes_root/<scene>/datasets/court/dataset.json`だけを列挙。同sceneの複数方式を発見しない | scene・method・recipe・dataset versionを区別したcatalog。元scene IDを保持 |
| [service.py:106](../../src/synthetic_data_generation/dataset/court/review/service.py#L106)、[194](../../src/synthetic_data_generation/dataset/court/review/service.py#L194) | completed ownerとscene identityを要求し、alignmentも同scene配下から読む | 別outputのstoreとpinしたalignmentを明示参照。main ownerの書換えや偽scene名による回避をしない |
| [service.py:216](../../src/synthetic_data_generation/dataset/court/review/service.py#L216) | 採択sampleをframe/view順に結ぶ線。完全な計画曲線ではなく、採択0軌道は読込error | candidate planとacceptedを区別。棄却位置/0件軌道も診断できる表示。閉/開の別と元の弧長順 |
| [web.py:46](../../src/synthetic_data_generation/dataset/court/review/web.py#L46)、[static/app.js:24](../../src/synthetic_data_generation/dataset/court/review/static/app.js#L24) | APIと選択状態はscene中心、filterはsplit。方式切替はない | 選択key/API/cacheをversion・method・sceneへ結び付け、切替後の古い応答を破棄 |
| [service.py:58](../../src/synthetic_data_generation/dataset/court/review/service.py#L58)、[345](../../src/synthetic_data_generation/dataset/court/review/service.py#L345) | 遅延JPEG/overlay、並行2・画像cache128・scene cache2。RGB/教師/target詳細を既に表示できる | 既存sample_store/overlayと上限を再利用し、全方式を一括decodeしない |

現状でも、別rootに正しいscene layoutを持つ独立publicationを作れば別port/サーバーで閲覧は可能。
ただし同一UIでの方式切替はなく、tmpの診断PNG/NPYや不完全なdataset.jsonをそのまま読み込めない。
今回の静的media入口を、Court Reviewへの5方式の実装完了とは称さない。

#1352の最小UI成果は、4scene×5方式の小規模データをmethod/scene/versionで選べ、
同じsceneの3Dコート・curve・撮影点とRGB/教師を見ながら方式を切り替えられること。
横並びの動画合成や推論機能を新設する必要はない。
候補/採択数・採否理由・recipe/seed・k・対象KP・QA感度・採択0層を確認できるようにする。
パラメータの異なる方式間で画像indexが対応すると仮定せず、対照recipeにのみpair IDを使う。
accepted線が棄却区間を直線で跨ぐ場合は、計画曲線と誤認しない表示が必要。
rejectのRGBは現行publicationにない。保存されていなければ「画像なし」とし、他画像を代入しない。

検証はcatalogの同scene複数版、path境界、hash/版不一致、trainのみ、0件軌道、開曲線、
切替中の古い応答、target-only教師とsampleの対応をCPU/browserで確認する。
小規模描画はtraining queueから小batchで実行し、本番presetへ反映する前にユーザーがUIで判断する。
表示上の採用記録を学習データへ自動反映する仕組みは要求しない。最終判断の正本は親#1343。

## 8. 時間・メモリ・保存量の見積もり

candidateをstream生成し、1軌道の固定弧長表と必要なID列だけ保持する。
parameter生成O(G)、弧長表O(GD)、撮影点O(P)、投影O(P×court数)。重複診断は空間indexで絞り、
全curve/全pointの二乗比較を避ける。`open_bezier`のanchor pairも距離indexから列挙する。
CPU prototypeは検査用dictを保持する実験コードで、productionのstream実装ではない。

float32 RGB+alpha+depthだけで`W*H*5*4` byte/画。
959×539なら約9.86MiB/画、600画は約5.78GiB、24000画は約231GiBで、モデル・コピーは別。
GPU batchは1から測り、host配列は1cameraずつQA/packして解放する。
既存600frame/shardをGPU microbatchへ流用しない。
新規描画runの容量probe1→4cameraとpeak/時間は同runの記録を参照し、
ローカルのrateをColabの予算値へ流用しない。

時間は`sum_s(load_count_s*load_time_s + new_rendered_s*seconds_per_frame_s) + QA + pack + transfer`、
保存量は`sum_s(unique_cached_accepted_s*JPEG_sparse_bytes_s) + indices + QA + manifests`。
一時領域は未pack chunk・転送中コピー・checkpoint/GSを含むpeakで見積もる。
描画・学習・評価のGPU時間を別計上し、scene/方式ごとの採択率・rate・保存bytesを実測する。
最大画像数・共通更新数・追加seed/長期学習対象・Bのprefix計画は#1356でユーザーと固定する。

## 9. 受入条件と後続の検証

| 更新済み#1350の受入条件 | この成果物と証拠 | 後続の責務 |
|---|---|---|
| 現行制約・複数方式・各仕様 | §1/2、code参照、5方式の分布/中心/高さ/view/ID・UI判断点 | #1352で実装・小規模4sceneデータ・UI比較・ユーザーの採用判断 |
| A1/A2/Bと全採択の計数 | §4の疑似コード・実描画計数表・目安表の再解釈 | #1356でprefix/規模を固定。等枚数比較の断定をしない |
| stable ID/seed・2倍細分 | §3、共通672点のCPU証拠、paired変数の4追加テスト | 5方式のn/k/seed/順序/worker/shard変更、K/pose/投影/cache教師の不変性を検証 |
| B00〜B03の画質・幾何・分布診断 | §5、2080枚のQA、16動画・16sheet、3D図と良例/失敗例、残存監査 | D2/D5/D6をUIで最終判断。全候補分布の採択率/支持は#1356 |
| Colab処理単位・manifest・再開境界 | §6/7/8、schema責務・学習側参照・UI制約・費用式 | #1351資産、#1352生成/UI、#1354学習/実画像分割、#1355永続再開 |

通常検証のコマンドと結果は[validation.txt](../runs/run-court-novel-view-quality-20261022/validation.txt)。
既存8＋描画用4のCPUテストとCourt Reviewの20テストを同じlock下で実行した。
新方式の実装テストに代わる証拠ではなく、今回保存したprototype/QAと読込制約の検査である。

#1352のテストでは非整数L/d、min count/multiple、矩形角、nonplanar、開曲線の終点除外、
k拡張、格子の候補枯渇、全棄却の条件、採択0軌道、exact重複、pending/render失敗を扱う。
共通点はn/k・列挙順・shard/workerを変えてID/K/pose/教師/採否を照合する。
全採択を使い、目安数へ一致しなくても補充・間引きをしないことを検査する。
renderer結合では共通cache、JPEG後の厳格再読込、既存v1/v2/v3/sfm_boundedの回帰を確認する。

#1350は仕様候補と診断・判断材料を揃えて完了できる。D1の実装比較とD2/D5/D6の最終判断は#1352へ渡す。
本番生成は採用方式・品質recipeの合意記録後に#1355/#1356で接続する。
この作業からGitHubへの投稿・pushは行わず、親が合意記録とPRを更新する。
