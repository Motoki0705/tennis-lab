---
id: group-geometric-residual-v2
type: group
task: plcs
sequence: 111
recorded_at: '2026-09-21'
title: Court14校正・持続誤検出・損失と入力conditioningの比較
members:
- run-plcs-residual-v2-balanced-s42
- run-plcs-residual-v2-legacy-s42
- run-blcs-residual-v2-balanced-s42
- run-blcs-residual-v2-legacy-s42
- run-residual-v2-conditioning-cpu
parents:
- group-geometric-residual-v1
papers: []
tags:
- triangulation-residual-v2
- court-calibration
- persistent-errors
- loss-ablation
- conditioning
---

## まとめ（比較実行中）

独立camera摂動をCourt14再推定に置き換え、四隅と両端中央フェンス付近の6候補、秒単位の持続誤検出を使う。camera側とfalse-track側の案、損失案を3つのscoutで独立に検討して実装した。既存ACCAD/物理worldとsplitを再利用し、v1の再現profileを残す。

同一v2データ・モデル・seed42でlegacy lossとbalanced regretを比較する。PLCSは30epoch完走し、legacyのval0.111298/test0.119929 mがbalancedのval0.122009/test0.131759 mより良かった。新損失bundleの優位は支持されない。Meijiでは両方とも極端な骨長異常を減らしたが、再投影誤差は増え、独立3D精度は不明。長い前腕も残る。

BLCS legacyはtest平均0.844004→0.826194 m、中央値0.079263→0.079876 m。多数例の改善は未確認。balancedはepoch23後にnative workerが中断し、保存checkpointから別runで復旧中。同じepoch24全8000件の単一processでは失敗せず、原因は未確定。復旧はworker0、次の比較はspawn/OpenCV1threadで行い、アルゴリズム変更とruntime対策を区別する。

CPU embedding診断では、微小な再投影差分の影響がBF16で丸めに埋もれる例を確認した。そこでlegacy lossを両taskで固定し、asinh(residual_uv/0.01)だけを変えた比較を追加する。この選択はvalidation入力の診断に基づき、test結果によるscale調整は行わない。変換はconfig・checkpoint schema2へ明示し、既知schema1をSwiGLU/rawへ警告付き移行した出力が元checkpointとバイト一致することを確認した。

全20,000 sceneのepoch0入力走査は失敗0。各runの指標・再現性・曲線は個別nodeを正本とする。最終集計は同一test scene/target/initial/maskを照合し、平均・中央値・p95・改善率・event/初期誤差binを併記する。単一seed、合成誤差仮定、実clipに独立3D正解がない制限は残る。
