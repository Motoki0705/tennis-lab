# NHT公開描画データ

[English](README.md) · [简体中文](README.zh-CN.md)

`NHTRenderClient`は`nht-render`の公開ファイルだけを消費します。request/result
schemaとCUDA runtimeは[NHT](../../../../third_party/nht/README.md#standard-renderer)が所有します。
以下はview encodingが有効な、採用済みB00〜B03 exportについての意味です。

`rgb.npy`は学習画像の符号化で得るfloat32のdecoded RGBです。現行rendererは
調和特徴量を蓄積してから非線形の学習済みshaderを適用するため、linear-light、
straight alpha、premultiplied alphaのRGBを保証しません。`alpha.npy`は別途得る
Gaussianの累積opacityであり、最前面の物体やネットを表すmatteではありません。

`depth.npy`はcanonical NHT scene単位のfloat32の**期待camera z-depth**です。
定義は`sum(w_i * z_i) / sum(w_i)`であり、最前面surfaceでも画素rayに沿う距離でも
ありません。蓄積が空なら0です。`nht_depth_to_metric`は明示した
`nht_scene_units_per_metre`で割り、単位だけを変換します。depthの統計的な意味は
変わりません。near/farはexportのruntime configに従い、同じNHT単位です。

この3配列だけでは、半透明の背景の途中へ不透明前景を挿入する方法は定まりません。
前景depthと期待depthの比較や、背景全体alphaのネットopacityへの流用は、対応済みの
遮蔽契約ではありません。現行公開requestには前景depth入力がなく、resultには
depthで条件付けたRGB寄与・透過率の出力がありません。

提案する契約と実測上の制約は[合成方式の研究記録](../../../../knowledge/runs/run-player-depth-compositing-20261108/README.md)
に置いています。公開schemaに実装済みの拡張ではありません。
[C5向けの別名の近似契約](../../../../knowledge/runs/run-player-depth-compositing-20261108/contract.md#c5-approximation)
も実装案であり、上記の現行配列の意味は変更しません。
