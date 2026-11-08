# C5へ渡す合成契約とNHT拡張案

## 結論の位置付け

以下のlinear RGB prefix合成を**必要な目標契約**とする。現行NHT checkpointと
公開RGB/alpha/EDからこの契約を満たす出力を得られる、という採用宣言ではない。
既存shaderの非線形性・部分特徴量の色の不定性が残り、C5はv1出力だけで
この合成器を実装した扱いにしてはいけない。背景再学習・受入条件変更の必要性は
親へ返す判断であり、この調査で暗黙に許可・実施しない。

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

## RGB・alpha

Blenderの取得対象は、view transformを焼き込まないfloatのscene-linear Rec.709
Combined RGBA。premultipliedをstraightへ直す必要がある操作では、valid alphaで
一度だけ除算し、色変換後に再度premultiplyする。α=0のRGBは契約上0とする。
不透明SMPL/球の材質opacityは1であり、境界のcoverage alphaとは区別する。
AgX/Filmic、exposure、contrast、gammaを片側だけに適用しない。

NHTのtrainerは画像byteを255で割った値を教師にし、public rendererはsigmoid
出力をfloatで保存する。これは撮影画像と同じdisplay-referredな符号化であり、
linear radianceではない。入力動画のtransfer functionまで校正済みという証拠もない。
現行decoded RGBへalphaを掛ければ正しいpremultiplied色になる、とは言えない。

目標APIは `rgb_encoding=linear_rec709` と `alpha_association=premultiplied` を宣言し、
最後のvideo出力で一度だけ選択したdisplay transform/OETFを適用する。
現行NHTの最終RGBをsRGB逆変換するだけでは、層ごとのlinear寄与は復元されない。
encoded RGB上でのalpha合成を採るなら別の近似契約と人間の判断が必要である。

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

現行EDは別のeval3d passの `sum(w_i z_mean_i)` をNHT alphaで割る。
`z_mean_i`はGaussian中心のcamera zである。NHTのnear/farもprojectionで中心を
選別し、per-pixel surfaceで切っていない。probeのfar-plane prefixはこの意味での
**mean-zによる選別**であり、任意の前景surfaceまでの物理的な積分の実装ではない。
大きいGaussianが前景を跨ぐ場合、中心の前後判定だけでは不十分になる。

## 提案する公開file境界

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

## C5の実装前に満たす検証

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

現行重みを保つ近似を採用するか、局所radianceを学習する範囲を追加するかは親と
人間が決める必要がある。C3の人間受入も独立したC5開始条件であり、この調査の
PR/CI成功では解除されない。

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
