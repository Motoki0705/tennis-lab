---
id: group-i1044-qualification
type: group
task: blcs
sequence: 53
recorded_at: '2026-10-13'
title: BLCS物理損失の実装検証・GT自己評価・中断記録
issue: 1044
members:
- run-i1044-smoke-parameters-v1
- run-i1044-smoke-parameters-v2
- run-i1044-smoke-parameters-v3
- run-i1044-smoke-chunked-v1
- run-i1044-smoke-chunked-v2
- run-i1044-smoke-gan-v1
- run-i1044-smoke-evaluation-v1
- run-i1044-gt-val-preflight
- run-i1044-position-s42-cuda-interrupted
- run-i1044-reprojection-s42-cuda-interrupted
parents:
- run-i1032-blcs-physics-v3-e200-s42
papers: []
tags:
- qualification
- physics-loss
- not-adoption-evidence
provider: codex
date: '2026-10-13'
status: done
artifacts:
  run_dir: knowledge/runs/group-i1044-qualification
---

## 確認できた範囲

物理記録からの教師、イベント区間内の短い共有積分、速度教師、新指標の集約と保存を接続した。通常・chunk切替・任意GANのGPUスモークは各4 optimizer stepで終了し、最終validationとcheckpoint保存を確認した。これは実装経路の確認であり、方式A/Bや損失の優劣を示す結果ではない。

GT自己評価は固定validationの全100 scene・37,287 frameで行った。3D位置、速度head、同じ差分演算による誤差は0。rollout位置RMSEは3.83948e-7 m、速度RMSEは4.16577e-7 m/s、clean再投影RMSEは2.79269e-5 px（評価可能率1）だった。先頭2 clipの有限反復ODE fittingには約0.000713 mと0.009661 mの残差が残った。直接積分の丸めと、有限予算の非凸fitの最適化誤差を同一視しない。

評価処理の追加確認では、既存9,000更新checkpointと幾何対照をvalidation先頭2 clipで全長評価し、保存NPZのみの指標再計算が一致した。幾何対照は保存カメラGTを入力せず、同じ2Dコート点で校正する。Meijiの固定505 frameについて、4更新のスモーク重みで直接軌道・予測区間による積分・再投影・fit診断の保存が完了した。短いモデルの数値を候補比較には使わない。実観測の2D注釈はモデル補助であり、コート入力も校正から生成するため、独立した実3D精度や生検出器のend-to-end性能を測っていない。

## 検出した問題と対応

| 記録 | 観測と対応 |
|---|---|
| parameters-v1 | 4更新よりwarmupが長く、構築時に停止。スモークだけ短いwarmupへ修正。 |
| parameters-v2 | 最終部分epochでvalidation未実施のため選定値がNone。終了時の明示的validationを追加。 |
| chunked-v1 | chunk切替後に古いloaderが破棄済みsceneを参照。BLCS設定で毎epoch loaderを再作成。 |
| reprojection正式比較初回 | 同じCUDA unknown errorで中断。並行velocity runは継続。同条件で最初から再試行。 |
| position正式比較初回 | CUDA unknown errorで途中停止。原因は未特定。同じseed・条件で最初からやり直し、途中runは正式比較へ混ぜない。 |

初期parametersスモーク3件のqueue captureは、新規untrackedファイルをpatchへ保存していなかった。捕捉原本は各bundleのoriginal_capture/へ内容を変えず保管し、厳密な再現が不可能であることを各ノードに明記した。ログ・設定・保存結果に限定した証拠である。正式4,000更新runは必要な実装をcommitした版から実施する。

## 通常検証と比較実験の境界

本体実装時のBLCS全体pytestは171 passed / 4 skipped、追加の選択・対照・保存再計算テストは9 passed。ruffとmypyも通過した。batch分割/padding/分散samplerの重複、欠損、event support、勾配到達、入力GT漏洩の回帰テストを含む。指定回数がないため独立validatorは起動していない。

旧APIを置換した[比較プロトコル](../../runs/group-i1044-progressive-losses-u4000/protocol.md)を学習前に固定した。主軸は全frameの3D軌道RMSEで、運動・rollout・fit・再投影・支持数を併記する。追加seedはvalidationのみで選び、testは選定完了後に評価する。ユーザー指定により採用閾値はなく、方式採用や本番checkpointの置換は行わない。

本群は事前検証と中断の記録である。4,000更新×10 runの比較結果と因果考察は、完了した各runと比較groupに記録する。

候補BのGT区間／予測区間ごとの積分位置・速度・イベント・失敗支持数と、Meijiの再投影・fit診断を保存NPZから再計算した。commit `434f0660bfc0208e31b07fe1968cc6d96e41b3a3` のCPU資格確認では元のJSONと一致し、元の重みやstoreなしでも検証できる。追加ログと2つのroundtrip bundleを本群に添付した。CIで検出した統合fixtureの速度と物理記録の不一致は、fixtureを同じシミュレーションの位置・速度・記録に揃えて修正した。

CI追加検証では構造・設定・loss・CLIの82件中81件が先に通過し、残った共有root参照を修正後、そのauditとCLI 5件が通過した。元の学習版6b2c24acfb3a1f1f4b58b52b2abee08a57b9b225に対して6条件の解決済み設定が完全一致し、モデル・loss・教師のソースも一致する。CLI移設は演算変更ではない。保存レポータはスキーマ確認用fixtureで表と7枚の図を生成できたが、このfixtureの複製数値は比較結果ではなくgitへ保存していない。

CIが検出した既存のレビュー処理の秒単位保存先衝突もローカルで再現したため、実行ごとの一意な保存先へ修正し、関連51テストが通過した。この修正はBLCSの数値実験とは別のCI修正である。データ再照合も59,006ファイルすべて一致した。
