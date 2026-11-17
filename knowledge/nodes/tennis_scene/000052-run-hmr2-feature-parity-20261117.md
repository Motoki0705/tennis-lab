---
id: run-hmr2-feature-parity-20261117
type: run
task: tennis_scene
sequence: 52
recorded_at: '2026-11-17'
title: HMR2特徴分離のclip_035身体推論境界比較
issue: 1754
provider: codex
session: 01a1e957-e495-7ef2-81cd-bf26c4ae465a
date: '2026-11-17'
repro:
  commit: a9c684ed67adaf24088d1289e64206bb34e11e29
  command: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 .venv/bin/python knowledge/runs/run-hmr2-feature-parity-20261117/compare.py
status: failed
summary_impact: none
config: {}
metrics:
  cpu_regression_passed: 272
  gpu_body_comparison_completed: false
  upstream_wall_seconds: 23.872405783
artifacts:
  run_dir: knowledge/runs/run-hmr2-feature-parity-20261117
  output_dir: outputs/hmr2_feature_parity
parents: []
relations: []
papers: []
tags: []
---

C6の実GPU比較は未達。C1固定入力 `video_000/clip_035` の1clipを使い、CPUでcourt sideを再現した後、実CLIPによる人物対応が `player_association_ambiguous_players` で停止した。identity 3とidentity 0が共に2,106frameの候補となり、人物の採用が一意に決まらなかった。HMR2・GVHMR・body placementへは到達しておらず、実GPUでの身体出力一致や高速化は主張しない。現在の標準pipelineの判断は変えず、summaryへの影響はnoneとする。

C1保存済みの2D観測を別出力へコピーしてchecksum/schemaを検証し、CPUでcourt sideまで再現した。C1は1,055frameの30fps格子が既定上限1,024を超えて停止しているため、この身体境界比較の両経路に同じ `frame_sampling.max_frames=2048` を明示する。格子自体、crop、重み、精度、閾値は変えない。この条件での比較はC1全段基準値の達成とは区別する。

`unsplit_gvhmr.py` は分離前commit `7ed91523b` のcomponentをそのまま保存したもの。C1 recorderは `origin/codex/video-batch-baseline` の `src/utils/inference_recording.py` をコピーし、runtimeのload/unload hookへ接続する。製品のsrcへ旧経路やrecorderの複製は追加しない。

共有training queueのall枠で、人物対応・CPU幾何/三角測量・view選択を1回だけ準備し、分離前HMR2+GVHMRと分離後HMR2保存/再読込+GVHMRを同じ入力で実行する。両者に実body placementを適用し、全parameter・位置・mesh・mask・拒否理由を比較する。seedは各身体経路前に42、float32、HMR2の既定batch上限を維持する。

事前許容差はrad/betasがatol=1e-5・rtol=1e-5、位置/translation/mesh(m)がatol=1e-4・rtol=1e-5、再投影(px)がatol=1e-3・rtol=1e-5。ID・frame・mask・拒否コードは厳密一致とし、1frame以上の有効な配置を要求する。allocator上限10GiBを設定する。準備時のRTX 5060 Tiは16,311MiB中1,423MiB使用、RAM available 30GiB、出力filesystem空き356GiB。これは起動時点の保証ではなく事前観測である。

再現入口:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python knowledge/runs/run-hmr2-feature-parity-20261117/prepare.py
# 次は共有training queueへresource=allとして登録する。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 .venv/bin/python knowledge/runs/run-hmr2-feature-parity-20261117/compare.py
```

初回queue job（attempt-1）は受入スクリプトでPlayerAssociationModuleの必須enabled引数を渡し忘れ、モデル起動前にTypeErrorで終了した。実推論の成功には数えない。queueのrepro bundleと出力のattempt-1-failed/に原ログ/事前条件を保持し、引数だけを修正して同じ1clipを再試行する。比較条件は変更しない。


2回目はqueueのall枠で終了し、exit=1/failedを確認した。人物対応までのrecorder区間は23.872秒、appearanceモデルはload/unload各1回、PyTorch allocated peak 820.475MiB / reserved 1,032MiB、sampled RSS peak 2,065.742MiB。人物対応の推論自体と解放は実行され、HMR2関連のloadは0だった。これらは停止prefixの資源値であり、身体段や全pipelineへ外挿しない。実測の正本は[upstream recorder](../../runs/run-hmr2-feature-parity-20261117/upstream_recorder.json)、[結果](../../runs/run-hmr2-feature-parity-20261117/outcome.json)、各attemptのqueue原ログ・reproである。

C6の実装についてはCPU回帰272caseが30.69秒で成功した。中心3ファイルは32case（既存4caseを更新、28caseを追加）、最大parametrize展開7件、最長case合計0.804秒、setupは全件表示0.00秒だった。非対称な人物/segment/frameの具体値、保存特徴の再読込、別clip/source拒否、旧schema load拒否、全cache hit/load時のモデル生成0回を検証した。1clip/3clipのfake計画ではHMR2・GVHMR・body geometryのload/unloadが各1回。fake GVHMR parameterは分離前の手計算参照と厳密一致し、配置後の位置・betas・有効maskも既知値を維持した。[実測](../../runs/run-hmr2-feature-parity-20261117/cpu_validation.json)を参照。

source frame metadataを保ったままHMR2特徴rowだけを反転する故障注入では、3行中2行の特徴値が異なり最大差4でunitが失敗した。注入を戻した同じnodeは成功した。pre-commit ruff/mypyも成功。独立validatorは未指定のため指定0・試行0・完了0。

次の受入には、身体view選択まで一意に確定する固定入力または保存済みBodyViewSelectionが必要。今回は親の1clip制限を守り、別clipへのGPU実行、人物IDの手動統合、閾値・cameraの変更は行っていない。親が入力/対応方針を判断した後、同じ比較手順を使って身体出力と実モデル読込記録を取得する。C4の一括CLIをこの最適化の未達によって止める判断はworkerでは行わない。

CIでHMR2 mean params欠落時の取得先URLが新nodeの汎用identity検査により失われた退行を検出した（[4a0bcca09 CPU shard 2](https://github.com/Motoki0705/tennis-lab/actions/runs/44081276599/job/134032912772)）。identity作成前に既存require_smpl_assetを通すよう修正し、既存の取得案内契約テストを含む279caseが29.74秒で成功した。正しい資産を使う推論の入力/数値処理は変わらず、GPUの上流停止は未解消である。

続くCIは[4056ef5b7 CPU shard 1](https://github.com/Motoki0705/tennis-lab/actions/runs/44083406887/job/134040209733)でgalleryの新node登録漏れを検出した。HMR2特徴のsource frame別normとsegment/人物/cameraの表示を登録し、GVHMRのv2も明示対応した。既存の全component描画・store不変契約とsource frame/gap契約を拡張して確認し、関連297caseが31.57秒で成功。さらにtennis_sceneのunit/integration全体1,170caseが56.34秒で成功した。既存の全artifactを実描画するgallery 1caseはcall 6.67秒で5秒目安を超えた。新nodeは図1枚だけを追加しており、schema登録と実描画/store不変を一緒に検証する既存の固有契約を維持するため、この重いcaseを残した。
