# C5へ渡す合成契約とNHT拡張案

## 結論の位置付け

以下のlinear RGB prefix合成は **`radiance_prefix_linear_v1` の目標契約**である。現行NHT checkpointと
公開RGB/alpha/EDからこの契約を満たす出力を得られる、という採用宣言ではない。
既存shaderの非線形性・部分特徴量の色の不定性が残り、C5はv1出力だけで
この合成器を実装した扱いにしてはいけない。
[2026-11-08の人間判断](https://github.com/Motoki0705/tennis-lab/issues/1723#issuecomment-6722754922)
により、背景再学習をせず現行shaderを使う近似へ進む。別名の
[`nht_prefix_alpha_closed_srgb_v1`](#c5-approximation)を後述する。
局所radianceの学習・蒸留、ネット遮蔽の省略は採らない。

## 合成式

同じcamera rayの上で、背景の局所linear RGBを `c_i`、opacityを `a_i`、
その寄与のcamera zを `z_i` とし、手前から順に並べる。

```text
T_0 = 1
w_i = T_(i-1) a_i
T_i = T_(i-1) (1-a_i)
ED = sum(w_i z_i) / sum(w_i)

I_< = {i : z_i < z_fg}
C_< = sum(i in I_<, w_i c_i)
T_< = product(i in I_<, 1-a_i)
C_out = C_< + T_< C_fg       # covered opaque foreground sample
```

RGBはlinear Rec.709のpremultiplied寄与、透過率は無次元。空prefixは `C_<=0, T_<=1`。
空背景なら前景色そのもの、手前の不透明面ならその面の色、奥だけに不透明面があれば
前景色、α=.25の赤い層だけが手前にあれば赤.25＋緑.75になる。
前景より奥の背景色が不透明前景へ漏れる項はない。同一depthは前景を優先し、
無断のdepth biasを入れない。

前景がないsampleは、元の全背景RGBと指定したenvironmentで描く。欠測人物・球は
前景validをfalseにし、zero depthを有効面として使わない。複数の人物と球はBlenderで
最も近い有効な不透明面を選んでからNHTへ渡す。

coverage alphaがfractionalな最終画素へ、単一のZをそのまま付けると境界で誤る。
上式は共通のsampleごとに適用し、その後linear RGBで平均する。prototypeで
binaryなpixel-center coverageだけを扱うなら明示する。AA済みRGBAと非AAのZを
同じsampleとみなさない。motion blur・DOFを使う場合も共通sampleと時刻が必要。

## 座標・深度・単位

| 項目 | 契約 |
|---|---|
| コート | m、右手系、+Xは右sideline、+Yはfar baseline、+Zは上。ネット中心が原点 |
| metric scene | `MultiCourtLayout.courts[explicit court_instance_id].scene_from_court` のSE(3)で配置 |
| NHT scene | `MetricSceneAdapter.nht_scene_from_metric_scene` のSim(3)。`s=nht_scene_units_per_metre>0` |
| カメラ | OpenCV: +X右、+Y下、+Z前。`camera_to_scene` はSE(3)、回転へscaleを残さない |
| Blender | camera local -Z前、+Y上。`R_blender=R_opencv diag(1,-1,-1)`。worldは選択したmetric/court座標と一致 |
| z-depth | `z_fg=e_z^T R_cw^T (X_fg-C)`、比較前に `z_fg_nht=s*z_fg_m` |
| ray距離 | unit ray `r=(r_x,r_y,r_z)` に沿う距離 `d` なら `z=d*r_z`。`r_z`はcamera座標の成分 |
| near/far | camera zに対する正のクリップ値。mとNHT単位でそれぞれ記録し、相互にs倍 |
| image | H×W、左上からのrow-major、RGB。背景・前景・mask・元frame/timeを一対一対応 |

コートからNHTへの点の変換は `X_nht=A_nht_from_metric S_metric_from_court X_court`。
`A`の3×3部分を`s R_A`とすると、cameraのrotationは `R_A R_S R_camera`、
centerは点と同じSim(3)で変換する。translationだけを単純にs倍する方式は、
回転・並進を持つ一般のadapterでは不正。既存`MetricSceneAdapter.nht_from_metric_camera`
を使うのが正本で、probeの行列計算は研究用の独立照合である。

B00の保存adapterは `s=0.06983810646574976`。現行runtimeのnear=.01、far=1e10は
NHT単位であり、mではnear約.143188、far約1.431883e11になる。
near=.01 mと読み替えない。現行public requestはnear/farのoverrideを受理しない。
C5で実用的な共通far（例100 m）を選ぶなら、新APIで明示するかexport runtimeの
権限ある変更として扱い、元の設定と出力を黙って差し替えない。

NHTのworld-evaluation kernelは画素 `(j+.5,i+.5)` からrayを作る。今回のcameraは
`cx=W/2, cy=H/2, fx=fy`で、既存Blender exporterもこの中央・square pixel条件に対応する。
ファイルの抽象的な「zero-based pixel centres」という表現から `.5` の補正を
追加しない。既存Kをそのまま両rendererへ渡し、rayの実座標を比較する。
integer-center由来の別Kを導入する場合は、その座標原点からの変換を入力契約で明示する。
off-center・非square pixelは現行Blender exporterが拒否するため、C5で必要になれば
shift/pixel-aspectまで実装・投影検証し、中央cameraへ置換しない。

今回のBlender 4.5.3 LTS / Cycles CPUでは、z=4 mの既知平面上の273 sampleすべてで
Z passはcamera zと一致した（最大差0 m）。ray距離との最大差は0.70114 mだった。
この組合せのZを使うならmをNHT単位へ変換するだけであり、さらにr_zを掛けない。
pixel-center rayの差は最大0.0005301 pxだった。C3/C5が別のversion/engineを選ぶ場合は
このprobeを再実行するか、Position passのcamera座標変換からzを明示的に作る。
この結果を未検証のEEVEEや別versionのZへ一般化しない。

## RGB・alpha

Blenderの取得対象は、view transformを焼き込まないfloatのscene-linear Rec.709
Combined RGBA。premultipliedをstraightへ直す必要がある操作では、valid alphaで
一度だけ除算し、色変換後に再度premultiplyする。α=0のRGBは契約上0とする。
不透明SMPL/球の材質opacityは1であり、境界のcoverage alphaとは区別する。
AgX/Filmic、exposure、contrast、gammaを片側だけに適用しない。
Blender probeでは132の部分coverage画素を含め、CombinedのRGBが
alpha×既知linear emission色に最大差0で一致した。

NHTのtrainerは画像byteを255で割った値を教師にし、public rendererはsigmoid
出力をfloatで保存する。これは撮影画像と同じdisplay-referredな符号化であり、
linear radianceではない。入力動画のtransfer functionまで校正済みという証拠もない。
現行decoded RGBへalphaを掛ければ正しいpremultiplied色になる、とは言えない。

目標APIは `rgb_encoding=linear_rec709` と `alpha_association=premultiplied` を宣言し、
最後のvideo出力で一度だけ選択したdisplay transform/OETFを適用する。
現行NHTの最終RGBをsRGB逆変換するだけでは、層ごとのlinear寄与は復元されない。
以下の近似ではsRGBと仮定して逆変換した値を混合するが、物理的な局所radianceの
復元とは呼ばない。人間が第2フェーズの動画で見た目を判断する。

## 現行shaderと整合しない理由

NHTはGaussian内のtetrahedral補間とharmonic encodingを行い、
`F=sum(w_i h_i)`を蓄積し、view rayを加えて `RGB=D(F,ray)` とdecodeする。
`D`はReLUのMLPとSigmoidであり、一般に次は成り立たない。

```text
D(sum(w_i h_i)) = sum(w_i D(h_i))
D(alpha F) = alpha D(F)
D(0) = 0
```

したがって `D(F_<)`、`alpha_< D(F_<)`、`alpha_< D(F_</alpha_<)` は、
いずれも `C_<` と同一とは保証されない。最後の正規化も学習時の入力分布と違う。
`D(F_<)-T_< D(0)`というbias補正も、非負・alpha以下のRGB寄与を保証しない。
ネット奥の特徴量を除いた色の教師は元の画像教師には含まれず、depth/alpha出力を
足すだけのAPI変更ではこの色の問題は解決しない。

今回のB00 shaderでは空特徴のRGBが0.2319〜0.7026になり、affineな混合との差は
最大0.1901だった（0〜1のencoded RGB上）。これは単なる丸めの差ではない。
`D(F_<)`が正しいpremultiplied色という候補は棄却する。alphaで掛け直す候補と
特徴を正規化する候補は最大0.1263異なったが、その差をground truthに対する誤差と
解釈しない。この初回probeでは採用せず、後述の継続比較へ引き継いだ。

現行EDは別のeval3d passの `sum(w_i z_mean_i)` をNHT alphaで割る。
`z_mean_i`はGaussian中心のcamera zである。NHTのnear/farもprojectionで中心を
選別し、per-pixel surfaceで切っていない。probeのfar-plane prefixはこの意味での
**mean-zによる選別**であり、任意の前景surfaceまでの物理的な積分の実装ではない。
大きいGaussianが前景を跨ぐ場合、中心の前後判定だけでは不十分になる。

## 目標契約の公開file境界（現行NHTでは未達）

既存の`nht_render_request_v1/result_v1`の意味を変更しない。以下はC5がNHT側で
実装するなら必要となる独立したversion付き境界の案であり、このPRには実装しない。

| request | 意味・拒否条件 |
|---|---|
| scene / model identity | 既存の公開scene.json経由。consumerはcheckpoint内部を読まない |
| camera / frame / samples | 既存PINHOLE K・SE(3)・W/Hに加え、元frame、時刻、sample patternを記録 |
| `foreground_z_nht.npy` | float32 H×W×1、validで有限かつnear<z<far。ray距離のまま渡さない |
| `foreground_valid.npy` | bool H×W×1。無効depthは0に固定し、maskのfalseと組にする |
| common clipping | 正のnear/farと単位。foregroundとbackgroundの適用方針を明示 |
| requested color model | `radiance_prefix_linear_v1`のような意味を固定した識別子。未対応なら拒否 |

| result | 意味・検証 |
|---|---|
| `background_rgb_linear.npy` | 前景のないsampleに使う全背景。linear Rec.709 float32 H×W×3 |
| `prefix_rgb_premult_linear.npy` | 前景までの実RGB寄与。空prefixは0、alphaをもう一度掛けない |
| `prefix_transmittance.npy` | float32 H×W×1、有限[0,1]。背景全体alphaから作らない |
| metadata | encoding/alpha association/depth type/units/ordering/clipping/sample patternを必須にする |
| diagnostics | 全背景alpha/EDは診断として別名保存し、prefixやsurfaceへ読み替えない |

NHT内部で必要な変更は、同じrasterization中のper-pixel foreground depthの照合と
`T_<`の保存、prefixの局所linear色の蓄積、version付きschema・transactionalな公開出力。
GPU batchは最初1に固定してOOM時の自動縮小を入れず、frameごとにstreamする。
foreground深度のarray単位・dtype・camera ID・解像度・sample順の不一致は公開前に拒否する。

研究用に `F_<, T_<` を追加公開するだけなら、現行重みを変更せず実装可能である。
しかしそれを上表のlinear RGB prefixと名付けてはいけない。
実RGB寄与を保証するには、compositing可能な局所radiance decoderの学習・蒸留等を
検討する必要がある。その場合は前景なしの元画像再現とprefixの色を別々に検証する。
sample-depth順への変更も既存tileのmean-z順と同一とは限らず、既存描画との一致を
未検証のまま宣言しない。

## 初回の目標契約に対する検証設計

これは正確なlinear prefix RGBを検討した初回の設計記録である。採用近似の必須数値検査は
後述の別節に置く。既知層は遮蔽の正のcontrolとして再利用し、NHTの局所radiance学習を
C5の前提に追加しない。

1. CPUの独立した既知層とGPUの薄いGaussian sheetで、手前opaque・奥opaque・
   手前net/奥opaque・奥netの期待色と透過率を照合する。GPUの.999 opacity上限と
   `alpha<1/255`の無視、`T<=1e-4`の早期終了を理想α=1と区別する。
2. 既知の非対称点を両rendererで投影し、Z/ray距離・scale・camera回転・row上下の
   どれでも誤りを見逃さない。中心画素だけを検証しない。
3. 全背景alphaが.99超でもprefix alphaが低いpixelを調べ、ネット以外の奥の面を
   全体alphaが含むことを確認する。画像でネット領域を確認し、数値ROIをsemantic maskと呼ばない。
4. 近似候補を採るなら、ネットの前後、細い球、人物輪郭、低alphaと空prefix、
   大きいGaussianが前景を跨ぐ位置で、元RGB保存・色の変化・遮蔽漏れを測る。
   現状、局所色の正解を持たないB00だけで物理的に正しい色を実証したとはしない。

近似で進める判断は上記の人間回答で確定した。C3の人間受入は独立したC5開始条件であり、
この調査のPR/CI成功では解除されない。

<a id="c5-approximation"></a>

## C5近似契約: full-alpha closure

識別子は **`nht_prefix_alpha_closed_srgb_v1`**。B00の3視点＋うち2視点のSSAA2、
54配置の比較で下記の数値検査を全て通し、C5向けの近似として採用した。
[`run-player-prefix-approximation-20261108`](../run-player-prefix-approximation-20261108/README.md)
が比較・実測の正本。この節は上の正確なlinear prefix契約とは異なる近似である。

### 量と比較候補

`F_< = sum(z_mean_i < z_fg, w_i h_i)`、`T_<`は同じprefixのnative透過率、
`a=1-T_<`。`F_all`、`A`は同じcamera/sample・near/farの全背景特徴とalpha、
`B=D(F_all,ray)`は現行public背景RGB。`D`は固定済み学習shaderであり、
ray encodingを正規化・alpha除算しない。`F`はharmonic特徴のchannelだけを指す。

| 候補 | straight RGB proxy `Q` | 背景proxyの混合重み |
|---|---|---|
| W: alpha掛け直し | `D(F_<,ray)` | `a` |
| N: 特徴正規化 | `D(F_</a,ray)` | `a` |
| R: full-alpha closure | `D(A F_</a,ray)` | `rho=a/A` |

Rでは、`tau=1-rho=(T_<-(1-A))/A`を前景の**近似混合重み**とする。
`tau`をnativeな物理透過率`T_<`と同一と呼ばない。既存decoderがほぼopaqueな
全背景特徴を教師にした点を保ち、同じ特徴平均のopacityだけを変えてもproxy色を変えず、
全prefixで元の入力`F_all`へ戻すための補正である。

```text
valid == false: out = B                         # bitwise original RGB
a <= 1e-6:      out = E_fg                      # skip biased D(0)
otherwise:
    Q = D((A/a) * F_<, ray)                     # straight, encoded proxy
    rho = a/A; tau = 1-rho
    L_out = rho * srgb_eotf(Q) + tau * srgb_eotf(E_fg)
    out = srgb_oetf(L_out)
```

`E_fg`は前景のstraightなdisplay RGB。sRGB EOTFは`e<=.04045`で`e/12.92`、
それ以外`((e+.055)/1.055)^2.4`。OETFはその逆であり、EOTF後の値を
scene-linear radianceと称さない。撮影動画のtransferがsRGBだった証拠はなく、
この仮定は見た目の制約として残す。式は`approximation.py`のCPU参照実装を正本とする。

空prefixでは前景色そのもの。全prefixでは`a=A,F_<=F_all`なので`rho=1,Q=B`、
全背景alphaが1未満でも元画像へ戻る。不透明前景にsuffixのRGBを加える項はない。
ただし補正に全背景alpha `A`を使うので、suffixのopacityまで変えれば結果は変わり得る。
`A>=.98`のサポート条件では`|tau-T_<|<=1-A<=.02`。これは重みの上限であり、
局所色の正解に対する誤差上限ではない。

NHTのharmonic sin/cos特徴は各channelの絶対値が1以下なので`|F_<|<=a`。
Rのdecoder入力は`|A F_</a|<=A`となるが、相関や意味が学習分布内という保証はない。
`a<=1e-6`を捨てた場合、同じ式とのlinearized RGB差の上限は`1e-6/.98`、
encoded差は最大約`1.32e-5`。これは量子化方針の数値上限である。

### 深度選別と共通sample

既存near/farとtileのmean-z順を保ち、各sampleで**Gaussian中心のcamera zが
前景zより小さい**寄与だけを蓄積する。同値は前景を優先。ED/median depth/ray-peakへ
置換しない。現在のfar選別を画素ごとの前景深度へ拡張する案であり、Gaussian体積を
不透明surfaceで物理的に切る機能ではない。大きいGaussianの跨ぎ・順序の誤りは残る。

初期profileは出力に対しSSAA2: NHTとBlenderを`2W x 2H`、Kの第1・2行を2倍で描き、
同じpixel-center rayの**binaryな不透明coverage**と最近面camera-zを使う。
各working sampleで上式を適用してから、2x2の`L_out`を平均しOETFを一度かける。
Blenderの最近面Z/Positionから作るbinary maskを使い、filtered Combined alphaの
しきい値だけで幾何maskを作らない。C5ではPositionから求めたrayとpixel-center rayの
最大差<=1e-3 px、既知面のcamera-z差<=1e-5 mを確認し、Combinedのstraight色が
有効幾何sampleに供給できない場合はcoverage不整合として拒否する。
背景だけの場合の基準も同じworking-gridのpublic背景を同じ方法で縮小した画像とする。
native低解像度画像との完全一致は要求しない。AA済みRGBAと別sampleのZを混ぜず、
DOF・motion blur・TAAはこのprofileに含めない。

Blenderのpremultiplied scene-linear Combinedは有効sampleのalphaで一度だけstraight化し、
明示したdisplay transformで`E_fg`へ変換する。初期契約はStandard/exposure0/gamma1/sRGB。
AgXを二重にかけない。HDRのclipping数を記録し、色transformをmetadataへ残す。
C3の承認済み球・材質・時刻を尊重し、probeの球radius .0335 mを本番既定にしない。

球のsample消失は遮蔽と区別する。保守的なサポート条件は
`q * min(fx,fy) * radius_m / (z_center_m + radius_m) >= 1`（`q=2`）。
満たさない有効球は`undersampling`として検出し、承認済みの解像度または明示した固定qを
上げる。欠測や全遮蔽として黙って落とさない。この条件でも物理的な可視色は保証しない。

### C5が追加する公開file境界

既存v1を変更せず、version付きrequest/resultの新modeとして追加する。
foreground入力のshape・dtype・単位・camera/sample対応は上記の目標APIと共通だが、
`requested_color_model=nht_prefix_alpha_closed_srgb_v1`を明示する。非対応なら拒否する。
NHTコマンド内で固定shaderを適用し、consumerはNHT internalsやcheckpointをimportしない。
背景のnear/farは保存runtimeのNHT単位をそのまま保つ。前景のBlender clipは別にmで記録し、
有効zがその範囲内かつ背景nearより先・背景farより手前であることを検査する。
前景farだけが100 m等の場合も、元背景のfarを暗黙に100 mへ変えない。

| result | 近似modeの意味 |
|---|---|
| `background_rgb_encoded.npy` | 同じworking gridの元public RGB `B`、float32 HxWx3 |
| `background_alpha.npy` | native全背景alpha `A`、float32 HxWx1 |
| `prefix_features.npy` | native蓄積harmonic特徴 `F_<`、float32 HxWxE。view-rayを含めない |
| `prefix_transmittance.npy` | native `T_<`、float32 HxWx1。全背景alphaから代用しない |
| `prefix_rgb_proxy.npy` | `D(A F_</a,ray)`、**straight encoded RGB proxy**。物理premultiplied RGBではない |
| metadata | model/scene・adapter・camera・frame/time・q・pixel origin・near/far・mean-z strict gate・encoded RGB/assumed-sRGB・前景display transform・mode/schema version |
| diagnostics | support違反・undersampling・HDR clipping・非有限・範囲/shape不一致をframe/objectに対応させる |

入力はvalidで有限・near<z<far、invalidならz=0。`A`は有限[0,1]、foreground valid上で
`A>=.98`をサポート条件とする。mask外の`A=0`は許可し、除算をせず元RGBを保存する。
`a<=A+2e-6`を検査し、差が丸めの範囲の場合だけaをAへclampする。
それ以上の不整合、単位やshapeの誤りは失敗とし、ED合成や最前面貼りへフォールバックしない。
特徴・Tの出力は全背景特徴と同じraster passで取得し、prefix skipはsuffixのRGBを読まない。
native早期終了とopacity閾値は明示し、全prefixを元rendererと照合する。

変更箇所の案はNHTの`nht_pipeline/schemas/{render-request,render-result}.schema.json`に対応する
新version、`nht_pipeline/render.py::ResidentSceneRenderer`、gsplatのNHT world-raster CUDA境界。
per-pixel zを読み、現行tile順の最初の`z_mean>=z_fg`で蓄積特徴とTをsnapshotし、
全背景の蓄積は続ける。harmonic特徴だけを上式でscaleし、同じview-rayを使ってdecodeする。
consumerの`NHTRenderClient`は公開fileだけを読み、schema/shape/encoding/unitsを検査する。
本issueでこのNHT変更を実装したとはしない。

### 描画時間とメモリの見積り方

実測時間・PyTorch peak・研究用depth別renderの回数は継続runのREADME/数値を参照する。
本番のper-pixel kernelは未実装なので、そのframe時間を実測済みとは呼ばない。
初期見積りは同じworking-gridのnative raster 1回とshader 1回をprefix追加分とする。
full RGB/alphaのrasterとdecoder、Blender、file I/O、resizeは別に加算する。
研究probeがsurface上の各異なるzごとにscalar renderする時間を、本番時間へ外挿しない。

`F_<`と`T_<`だけでもFP32で`4(E+1)q^2WH` byte必要。B00のE=24なら
256x144/q1で3.52 MiB、同じ出力のq2で14.06 MiB。例として1920x1080/q2では
791.02 MiBであり、全背景特徴・RGB・depth/mask・tile/Gaussian workspace・TCNNを含まない。
GPU batch1でframeごとに処理し、配列を全動画分GPU/RAMへ蓄積しない。
Fpを公開fileへ保存するI/O・保持容量も同じ式で見積もり、C5が固定入力で実測する。
今回のPyTorch peakはTCNN等の外部allocationを含む総VRAM上限ではない。

### 自動で検査する数値条件

以下は0〜1 floatのworking-grid値。公開API実装時の固定入力受入にも再利用する。
物理的な正解色への精度基準ではなく、再現・境界条件・計算の整合を測る。
比較候補の選定器は継続runの`evaluate.py`であり、閾値はGPU実行前に固定した。

| 検査する量 | 合格条件 |
|---|---|
| raw全特徴をdecodeしたRGB vs同gridのpublic RGB | 最大絶対差 <=2e-5 |
| 前景なし・mask外RGB vs元背景 | bitwise同一、最大差0 |
| 空prefix vs前景色 | 最大差 <=2e-6 |
| 全prefix vs元背景RGB | 最大差 <=2e-5 |
| 同一特徴平均をalpha .005/.02/.1/.5/.99で減衰したstraight proxy色 | 全alpha対の最大差 <=.005 |
| 前景色変更時のlinearized応答 vs `tau * delta(L_fg)` | 最大差 <=2e-6 |
| 同じprefix/T/AでsuffixのRGBだけ変更した有効面 | bitwise不変（CPU参照テスト） |
| `a>A` / `abs(F_<)>a`の超過 | 各最大 <=2e-6 |
| foreground valid上の全背景alpha | 最小 >=.98 |
| 全出力 | finite、RGB/Tは[0,1] |
| 固定3cameraのnet前後の球 | 後のprefix alpha平均が前より >=.02大きい |
| SSAA2・2camera・各8位置の細い球 | 各位置の有効working sample数 >0、coverage面積を併記 |
| 大きいGaussianの既知破綻fixture | 中心を跨ぐprefix alpha jump >.05、ray-peakとmean-zの前後不一致 >0画素を**検出・記録** |

最後の行は問題を解消したという合格ではない。mean-z近似の失敗条件を再現する検査である。
薄い不透明sheetの期待式と前後関係は初回runの既知層CPU/GPU検査で独立に守る。
実モデルには局所色のground truthがなく、net ROIの数値をsemantic segmentationとは呼ばない。

### 人間が第2フェーズ動画で確認する点

- ネット・テープ・支柱の手前／奥で、人物と球の見え方が切り替わるか。
- 細い球の移動中の消失・ちらつきと、人物輪郭のhalo・背景の色のにじみ。
- 低alpha領域の灰色化・彩度変化、背景と人物の露出・照明・ぼけの違い。
- 大きいGaussianを跨ぐときの突然の遮蔽変化と、ネット本体以外の誤った遮蔽。
- support/undersampling等の診断を元frame/object IDで追え、欠測・遮蔽と区別できるか。

不具合をopacity調整・ネット除去・無記録のdepth biasで隠さない。見た目が不合格なら
そのframeと診断を親へ返し、人間の再判断を得る。C3の動画受入と、この近似による第2フェーズの
動画受入は別の条件として残る。B00の小規模比較の合格をB01〜B03の受入に一般化しない。

## 参照する実装・一次資料

- 固定submoduleの`nht_pipeline/render.py::ResidentSceneRenderer.render_batch`。
- 固定gsplatの`gsplat/rendering.py`のED定義、`nht_depth_mode`と最終のalpha除算。
- 同じgsplatの`gsplat/nht/deferred_shader.py::DeferredShaderModule.forward`。
- 同じgsplatの`RasterizeToPixelsFromWorldNHT3DGSFwd.cu`と`ProjectionUT3DGSFused.cu`。
- [gsplat公式のdepth定義](https://docs.gsplat.studio/main/apis/rasterization.html)。
  upstreamの追加modeを、この固定forkが受理すると仮定しない。
- [Blenderの色管理](https://docs.blender.org/manual/en/4.0/render/color_management.html)、
  [render passes](https://docs.blender.org/manual/en/4.0/render/layers/passes.html)、
  [OpenEXRのalpha慣習](https://openexr.com/en/latest/TechnicalIntroduction.html)。
  Blenderの数値上の意味は別途小規模probeで確認する。
