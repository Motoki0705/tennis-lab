推奨は **`wide_equidistant_d159_16x9_v1`：1280×720、対角159°の等距離魚眼、DOFなし**です。MISSION 1 Wideに着想を得た再現可能な近似とし、実機レンズの校正再現とは呼びません。

**BlenderとNHTを同一の広角PINHOLEで描き、その共通サンプル上でC5合成してから、合成済みRGBへ一度だけ魚眼warpを適用する設計は成立します。** C5内部の光線幾何を共有できます。ただし離散画像の補間には誤差があり、直接魚眼描画との画素単位の完全一致は保証しません。

2026-11-09時点の一次資料確認結果です。

| 項目 | 確認結果・根拠 |
|---|---|
| 最新シリーズと発売 | 固定レンズのMISSION 1／MISSION 1 PROは2026-05-28発売。交換レンズ式MISSION 1 PRO ILSは2026-09-02発売。HERO13を最新機種とは扱いません。[固定モデル発売発表](https://investor.gopro.com/press-releases/press-release-details/2026/GoPros-New-MISSION-1-Series-Cameras-Mounts-and-Accessories-Now-Available-on-Retail-Shelves-Globally-and-at-GoPro-com/default.aspx)、[ILS発売発表・Availability](https://investor.gopro.com/press-releases/press-release-details/2026/GoPro-Launches-MISSION-1-PRO-ILS-the-Worlds-Smallest-Most-Rugged-Compact-Cinema-Camera-with-Interchangeable-Lenses/default.aspx) |
| 固定レンズのnative FOV | 公式表記は “159° native field of view”。**確認した公式本文は水平・垂直・対角のどれかを明記していません。** 159°を実測対角値として断定できません。[MISSION 1「159° Native FOV」節](https://gopro.com/en/us/shop/learn-cameras/mission-1) |
| センサー・aspect | 50MP、1.0型、写真8192×6144、4:3。動画は16:9／4:3／9:16。Open Gateはフルセンサー4:3です。[公式Tech Specs](https://gopro.com/en/us/shop/buy-cameras/mission-1-series)、[シリーズ発表・Open Gate節](https://investor.gopro.com/press-releases/press-release-details/2026/GoPro-Announces-New-MISSION-1-Line-of-Professional-8K-and-4K-Open-Gate-Compact-Cinema-Cameras-for-Filmmakers-Creators-and-Aspiring-Enthusiasts/default.aspx) |
| センサー実寸 | 同じセンサーを使うILSの公式資料は約13×10mm、対角16.4mm。固定モデルへの対応は、同一センサーという発表に基づく推定です。[ILS Lens Compatibility「Sensor Size」](https://gopro.my.site.com/GoProSupportHUB/s/article/MISSION-1-PRO-ILS-Lens-Compatibility-And-Information?language=en_US) |
| 絞り・換算焦点距離 | 固定モデルの仕様は**F2.8**、35mm換算 **15–27mm**。一方、商品紹介は「standard 14mm lens」と記載し、数値に差があります。モード別対応は確認できず、勝手に統合しません。[公式Tech Specs「Imaging」](https://gopro.com/en/us/shop/buy-cameras/mission-1-series) |
| ILSの扱い | MFTマウントですがセンサーは1.0型。焦点距離・絞り・合焦は装着レンズ次第です。固定モデルの159°／F2.8をILS共通仕様へ移せません。[公式ILS Field Guide](https://gopro.com/en/gb/news/mission-1-pro-ils-field-guide) |

固定モデルの焦点範囲について、正式な光学仕様としての近端・遠端、最適合焦距離、許容錯乱円は確認できませんでした。GoPro Labsのdnewman-gpswによる一次発言には “Details at 60cm and infinite should look about the same” がありますが、本人が光学チーム外の個人的推測と明示しています。したがって、**約0.6m〜∞という目安の補助証拠に留めます**。[2026-06-13の発言](https://github.com/gopro/labs/discussions/1700#discussioncomment-17292032)

近接用の別アクセサリには正式な範囲があり、+0.5は50–150cm、+1は45–90cm、+2は30–50cmです。これは標準レンズの焦点範囲ではありません。[GoProの2026-10-22発表](https://investor.gopro.com/press-releases/press-release-details/2026/GoPro-Launches-Close-Up-Filters-That-Add-Cinematic-Depth-of-Field-to-MISSION-1-and-MISSION-1-PRO-Cameras/default.aspx)

デジタルレンズの比較は次のとおりです。

| モード | 投影・加工の意味 | MISSION 1系 | 確認できた換算焦点距離 |
|---|---|---|---|
| Wide | 魚眼の湾曲を残す | 掲載あり | MISSIONのモード別値は未確認。参考：HERO13説明書は16–34mm |
| SuperView | 4:3画像を16:9へ伸長するソフトウェア処理 | 掲載あり、16:9のみ | MISSIONの個別値は未確認。参考：HERO13は16mm |
| HyperView | 8:7画像を16:9へ伸長するソフトウェア処理 | 確認したMISSION仕様・説明書には掲載なし。HERO13には存在 | HERO13は16mm |
| Linear | 魚眼歪みを補正するrectilinearな投影 | 掲載あり | MISSIONの個別値は未確認。参考：HERO13は19–39mm |

MISSIONの説明は[公式Mode Settings「Image Settings」](https://gopro.my.site.com/GoProSupportHUB/s/article/MISSION-1-And-MISSION-1-PRO-Mode-Settings-And-Information?language=en_US)と[公式説明書pp.30–31](https://static.gopro.com/assets/blta2b8522e5372af40/blt58994dc5c9a8e580/6a144dcaa924ef7755edc65e/M1Series_UM_en-US_REVB.pdf)、HERO13の値は[GoPro著の説明書pp.131–132・スペイン語配布版](https://gopro.com.ar/manuales-gopro/camaras/Hero_13.pdf)によります。HERO13の値をMISSIONへ転用してはいけません。

MISSION各モードの**H/V/D画角の組は一次資料で確認できませんでした**。また、[HERO13の公式FOVページ](https://community.gopro.com/s/article/HERO13-Black-Digital-Lenses-FOV-Information?language=en_US)は取得時にCSS Errorとなり、数表を検証できませんでした。販売店転載表の値は採用していません。

Linearは単純な中央cropだけではありません。GoPro Labsの説明は “it is a lens distortion projection” と明示しています。[2026-08-06の説明](https://github.com/gopro/labs/discussions/1769)。また、**35mm換算焦点距離だけでは魚眼投影関数を特定できません**。14mm／15mmをBlenderの通常カメラへ指定するだけではWideの再現になりません。

推奨近似の数式を固定すると、以下になります。角度は三角関数へ渡す際にradへ変換します。

```text
W = 1280, H = 720
c = (640, 360)
R = sqrt((W/2)^2 + (H/2)^2) = 734.302390027 px
θd = 159° / 2 = 1.387536755335 rad
fe = R / θd = 529.212928741 px

出力上の連続座標 p = (u,v)
Δ = p - c
ρ = ||Δ||
θ = ρ / fe

r(p) = (sinθ Δx/ρ, sinθ Δy/ρ, cosθ)
ρ=0 では r=(0,0,1)
```

ここで159°を**設計上の対角画角**と定義します。この近似の水平中央画角は138.580510°、垂直中央画角は77.951537°です。これらは計算値であり、GoPro公称値ではありません。実機の4:3像を16:9へcropした結果とも異なります。仮に4:3の対角159°等距離像を同じ横幅で16:9へ中央cropすると、対角は約145.943°になります。

魚眼矩形を収めるPINHOLEでは、四隅が最大の投影座標になります。

```text
tx = (W/2R) tanθd = 4.702600779272
ty = (H/2R) tanθd = 2.645212938340

PINHOLE水平FOV = 2 atan(tx) = 155.989943602°
PINHOLE垂直FOV = 2 atan(ty) = 138.582664896°
PINHOLE対角FOV             = 159°
```

最大の光軸からの角度は79.5°、最小`r_z`は0.182235525なので全光線が前向きです。**一つのPINHOLEで包含できます。**

縦横4倍の5120×2880を中間画像とした場合：

```text
fp = 4R / tanθd = 544.379614635 px

Kp = [[544.379614635,   0,             2560],
      [  0,           544.379614635, 1440],
      [  0,             0,              1]]

inverse map:
Φ(p) = (2560,1440) + fp tanθ · Δ/ρ
```

このKは、中央principal point・square pixel・zero skewという現在の
`src/tennis_scene/visualization/blender/export/camera.py::to_blender_camera`
の条件を満たします。36mm horizontal sensor設定なら仮想PINHOLEのlens値は約3.827669mmです。これは中間投影用の値で、GoProの物理焦点距離や換算焦点距離ではありません。

中心で拡大補間を避けるための最小倍率は、

```text
kmin = tanθd / θd = 3.888558017344
```

です。したがって4倍が妥当です。**この4倍とC5のSSAA2は別の倍率です。**

| 量 | 推奨159°近似 | 明示的な軽量候補150°近似 |
|---|---:|---:|
| 魚眼の水平／垂直画角 | 138.581°／77.952° | 130.736°／73.539° |
| 包含PINHOLEの水平／垂直画角 | 155.990°／138.583° | 145.822°／122.683° |
| 中心密度を保つ最小倍率 | 3.88856 | 2.85108 |
| 中間画像 | 5120×2880、4倍 | 3840×2160、3倍 |
| C5 SSAA2の実working画像 | 10240×5760 | 7680×4320 |
| `F_<`24ch＋`T_<`、FP32だけの容量 | 約5.493GiB | 約3.090GiB |

容量には全背景特徴・RGB・深度・Blender・raster workspaceなどを含みません。159°の本番性能は未測定です。150°は負荷を下げる別プロファイルとして有用ですが、159°から自動変更してはいけません。

C5との接続は、読了した
`knowledge/runs/run-player-depth-compositing-20261108/contract.md`、
`knowledge/runs/run-player-prefix-approximation-20261108/README.md`、
`src/synthetic_data_generation/rendering/nht/contracts.py`
に対し、次の条件で整合します。

1. **共通PINHOLEのworking gridを正本にする。** 同じK・pose・時刻・pixel-center規約を両rendererへ渡します。NHT公開契約は`PINHOLE`／`distortion_model=NONE`であり、任意魚眼対応を仮定しません。
2. **深度はwarp前のcamera-zのまま比較する。** `z=e_zᵀRᵀ(X−C)`と既存のmetric→NHT単位変換を維持します。魚眼へ深度を補間してからprefix選別する方式は不適合です。
3. **共通サンプルごとにC5合成し、その後に平均する。** binary coverage・最近面z・色の対応を保ちます。採用済みのmean-z近似と`nht_prefix_alpha_closed_srgb_v1`の意味は変わりません。
4. **レンズ変換は合成済みRGBだけへ適用する。** 背景・前景・alpha・深度を別々にwarpして再合成しません。

再現用の初期warp規則として、C5の共通SSAA2合成と2×2平均後の**linearized RGB**を保持し、出力各画素の固定4点`(j+.25/.75, i+.25/.75)`を上記Φで写し、bilinear補間した4値を平均して最後にOETFを一度適用する方式を提案します。補間座標は画素中心`.5`を明示し、マップ・kernel・サンプル配置・境界処理を全経路で共有します。背景のみの比較基準にも同じwarpを通します。

この規則は再現可能ですが、強い周辺縮小に対する十分なAAを保証しません。159°の四隅では、4倍中間画像への写像が半径方向に約30.975 source-pixel／output-pixelとなります。細い球やネットについて、出力側の消失・ちらつきを確認する必要があります。面積/EWAフィルタへ進める場合も、合成後RGBへ適用し、kernelと重みを版付きで固定してください。

直接Cycles魚眼はBlender単独の参考描画には使えますが、現行NHTと共通の魚眼サンプルを作る経路にはなりません。Blenderの等距離魚眼はsensor寸法を使わない独自の正規化を持つため、`fisheye_fov=159°`だけで今回の矩形対角159°と一致すると仮定できません。[Blender 4.5公式「Fisheye」](https://docs.blender.org/manual/en/4.5/render/cycles/object_settings/cameras.html)。C5用途ではPINHOLE＋共通warpを推奨します。

未確定なのは、MISSION固定レンズの投影係数・校正K・モード別cropと伸長マップ・実機DOF/PSFです。公開校正係数は今回確認できませんでした。F2.8は参考仕様として記録し、今回の近似ではDOFを無効にします。頭部カメラ位置は現在の`face_offset_head_m=(0, 0.08, 0.12)`を維持します。GoPro筐体寸法から額マウントの人体オフセットを導く根拠はありません。

小規模CPU計算では、四隅・軸・内部を含む9,417点で魚眼ray→包含PINHOLE→rayの最大成分差は約`4.16e-16`でした。実Blender/NHTでの超広角投影、C5の幾何許容差、背景alphaサポート条件、動画品質は未検証です。ファイル・branch・issue・他セッションへの変更、重い描画、Colab実行は行っていません。