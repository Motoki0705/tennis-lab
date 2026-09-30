---
id: run-i936-roughness-r14-s936
type: run
task: ball_refiner_3d
sequence: 22
recorded_at: '2026-10-01'
date: '2026-10-01'
title: T128の粗さは継ぎ目と窓内の共通jitterでありflow乱数だけでは説明できない
issue: 936
provider: codex
status: done
config: {checkpoint_update: 20000, cpu_threads: 1, local_error_window: 5}
metrics:
  flow_inside_acceleration_p95: 1604.998902400709
  regression_inside_acceleration_p95: 1876.5522617680494
  flow_acceleration_seam_energy_fraction: 0.42124640220325695
  regression_acceleration_seam_energy_fraction: 0.4089318779749216
artifacts:
  run_dir: knowledge/runs/run-i936-roughness-r14-s936
parents: [run-i936-anchored-t128-flow-regression-r13-s936]
relations: []
papers: []
tags: []
---

## 診断対象と再現

[run13回収](000021-run-i936-anchored-t128-flow-regression-r13-s936.md)の固定20k、全16val・6,383frameをCPU1threadで診断した。
全保存予測はgitに保持し、[diagnose.py](../../runs/run-i936-roughness-r14-s936/diagnose.py)はそれだけから
[全数値・hash](../../runs/run-i936-roughness-r14-s936/diagnosis.json)を再現する。train/test配列は読まない。
損失だけは回収時にhashを固定したtrain JSONLを読み、評価値と混ぜない。
予測の平滑化・seed/ラリー選別は行わず、差分は元の連続時系列で計算する。

## 継ぎ目と窓内

保存された各frameの窓ownershipが差分stencil内で変わる箇所をseamとした。
自由飛行は全stencilがGTイベント除外mask内であることを要求する。
全自由飛行加速度3,661件中seam48件（1.31%）、jerk3,534件中seam70件（1.98%）。

| 手法 | accel free p95 全体 / 窓内 / seam m/s² | jerk free p95 全体 / 窓内 / seam m/s³ | seamの二乗和比 accel / jerk |
|---|---|---|---|
| RTS | 158.1 / 158.1 / 130.5 | 7,133 / 6,939 / 43,460 | 0.23% / 0.36% |
| flow平均 | 1,847.9 / 1,605.0 / 16,219.1 | 199,756 / 164,263 / 1,048,286 | 42.12% / 41.21% |
| 回帰 | 2,203.6 / 1,876.6 / 17,653.7 | 241,101 / 196,372 / 1,261,659 | 40.89% / 38.25% |

継ぎ目の不連続は少数の大きなspikeを作る。ただし窓内だけでも加速度はRTSの10.1/11.9倍、
jerkは23.7/28.3倍なので、seamを直すだけで問題が解決するという仮説は支持されない。
trainの物理lossは各窓内だけを見ており、validationは継いだ元系列の差分を採点する。
これはモデルの窓端での不一致を増幅する構造的な要因だが、実際のseam対策の効果量はまだ未測定。

## 時間変動・camera数・sample乱数

各ラリーの一定XYZ誤差はflow1.816m/回帰1.057m RMSあるが、定数biasの2階差分はゼロで粗さの説明にはならない。
同じ窓・自由飛行内の5frame対称平均で誤差をtrendと残差に分解した（診断だけ。出力軌道は変更しない）。
フレーム単位の残差位置RMSはflow0.113m/回帰0.129m。
7frame全supportが自由飛行の区間で、誤差加速度RMS933/1,162m/s²に対し、
trend側207/210、残差側917/1,162。二乗和は相関項を持つためそのまま足せないが、
小さな速い揺れが59.94Hzの2/3階差分で大きくなる説明と整合する。

窓内free加速度p95をcamera0/1/2/3で分けると、flowは1,577/6,318/4,816/1,275、
回帰は2,048/6,910/6,230/1,454m/s²。少数camera層が悪いが、最多camera3層でも粗い。
cameraの層別は存在確率でなくvisibility、差分を取った後に中央frameで割り当てた。
flowのsample平均との差の位置RMSは0.0956m、自由飛行加速度二乗和の**0.1179%**、jerkの**0.1219%**のみ。
全sampleと平均のp95も近い。乱数を平均すれば大半が消えるという仮説は棄却し、
条件と学習済みmappingに共通した時間jitterとseam不連続を主要因として支持する。
これは入力条件のjitterとネットワーク自身の時間応答を分離した因果同定ではない。

## 損失の大きさと次の一因子試験

更新18,001〜20,000の2,000stepを均等に集計した。

| arm | 重み付きx0 | 再投影 | physics | event | physics / total |
|---|---:|---:|---:|---:|---:|
| flow | 0.003087 | 0.093826 | 0.001905 | 0.004095 | 1.851% |
| 回帰 | 0.002105 | 0.094357 | 0.002057 | 0.003916 | 2.008% |

physicsのraw値19.05/20.57に対してweightは1e-4。再投影項が損失値の約91〜92%を占める。
一方、loss値の比は勾配の比ではなく、秒差分とSmoothL1の勾配は大きくなり得る。
全項のclip前gradient normは平均8.72/8.86、clip閾値1。項別parameter勾配は未測定なので、
「physics勾配が足りない」とは断定しない。

窓内の広いjitterを直接対象にする次の一因子は、元64train/16valとrun13設定を保ち
**physics weightだけ1e-4→1e-3（10倍）**とする。窓/stride・reprojection・学習更新数・seed・samplerは変更しない。
重みの効果は未観測で、同時に640条件へ変更しない。seam改善は副次診断として計測し、
window構成を変える追加GPU jobは今回は投入しない。正式な比較規則・費用はenqueue前のissue投稿を正本とする。

通常検証はseam step・定数bias・交互noise・イベント/窓跨ぎを含む7tests成功（-n1）。
新しい統計は予測の変更やパレート優位を意味しない。元run13の不合格・H/default・#959 controlは維持する。


## 事前規則で固定した3D動画

[選択規則の事前投稿](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5914941598)に従い、
全16valの辞書順最初`val-00000`を全178frameで描画した。
[動画MP4](../../runs/run-i936-roughness-r14-s936/val-00000-3d-comparison.mp4)は
**H.264・1800×1000・15fps・11.866667秒・328,728 bytes**。
元59.94fpsの約1/4速度で、GT・混合平均・RTS・flow20k平均・回帰20kの上面XY・側面YZ・斜め3Dを共通範囲で表示する。
全raw trace・直近24frameのtrailと現在位置を示し、位置/時間の平滑化・outlier/frame除外を行わない。
表示範囲は全手法の全点とコートを含む。frame128の窓切替を明示した。

[poster](../../runs/run-i936-roughness-r14-s936/video-poster.png)、
[render.py](../../runs/run-i936-roughness-r14-s936/render.py)、
[入力・動画hash/選択規則](../../runs/run-i936-roughness-r14-s936/video.json)、
[ffprobeによる178frame確認](../../runs/run-i936-roughness-r14-s936/video-probe.json)を保存した。
CPUのみ、ffmpeg1thread。posterを目視確認し、全動画をffprobeでデコードしてframe数を照合した。
これは固定した一例の可視化で、全16valの数値に基づく結論を変更しない。summaryの結論も確認し維持した。
