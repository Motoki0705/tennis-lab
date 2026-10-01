---
id: run-i936-head-context-r19-s936
type: run
task: ball_refiner_3d
sequence: 33
recorded_at: '2026-10-01'
date: '2026-10-01'
title: 全成分pool tokenを絶対位置headへ連結する単因子比較（queue待ち）
issue: 936
provider: codex
status: planned
config:
  factor: position_head_input
  control: (c)512train/physics1e-3/reprojection.01
  candidate: concat final temporal token and full-mixture pooled token at absolute
    position head
  seed: 936
  updates: 20000
  primary_update: 20000
  validation: same16val/6383frames
  added_parameters: 384
metrics:
  cpu_initial_max_output_difference: 0.0
  initial_parameter_tensors_equal: 61
  cpu_initialization_seconds: 1.177090626093559
artifacts:
  run_dir: knowledge/runs/run-i936-head-context-r19-s936
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/h-dev/r19-head-context-512-physics10-s936-t128-20k
  queue_job: 1790819790903430816_1781754_i936-head-context-r19-s936-20261001
parents:
- run-i936-condition-readout-r19-s936
- run-i936-combined512-physics10-r16-s936
relations:
- to: run-i936-repro3-512-physics10-r17-s936
  rel: compares
papers: []
tags:
- single-factor
- head-conditioning
- preregistered
repro:
  command: timeout --signal=TERM --kill-after=15s 5340s env CUDA_VISIBLE_DEVICES=0
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/.venv/bin/python
    -u -m src.tasks.ball_refiner.scripts.experiment_dev_3d --plan /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/knowledge/runs/run-i936-head-context-r19-s936/plan.json
    --device cuda
  branch: campaign930/i936-2-synthetic-diffusion
  commit: 7c4521f9acb031881c4dc83cfb93f8134fbc0e8b
---

[実行前規則・暫定判断](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5923087105)に従う
単因子試験。CPUで現encoderの位置情報保持を確認したため、pool表現の拡張ではなく
**絶対位置headへの既存pool tokenの直接連結**を調べる。GPU結果はまだ無い。

## 何を変え、何を固定するか

position headに渡す特徴だけを、時間Transformer/最終LayerNorm後の128次元から、
そのtokenと既存の全成分pool tokenを連結した256次元へ変える。
1つのLinear headは絶対x0を予測し、三角測量座標・read-out予測への残差加算をしない。
共分散・camera subsetを含む全125成分の非線形encoder、重み付きpool自体は同じ。
state/time/Transformer/LayerNorm/event head、loss、データ、optimizer、窓順、seedは固定する。

(c)の512train/physics1e-3/reprojection.01を対照に、両arm各20k更新。
同16val・6,383frame、T128/stride128、flow4sample/8step、
0/2k/5k/10k/20k保存と主20k判定を維持する。追加48val/test配列を開かない。
H・anchored bank・#959 control・本番経路は変更しない。

[plan.json](../../runs/run-i936-head-context-r19-s936/plan.json)を設定・規則・hash・コマンドの正本にする。
既定temporal-onlyは維持し、候補はtemporal_and_conditionを明示する。
単因子guardは他のモデル/loss/seed/optimizer/データ量/窓設定の差を拒否する。
過去の設定で未指定のhead-inputは、当時存在したtemporal-onlyを明示的に正規化して比較し、
不正な値のfallbackはしない。

## 仮説と反証

現tokenから線形に平均位置を読めるにもかかわらず、出力では観測へのoffsetが大きい。
時間処理後だけを読む経路が位置の忠実度の学習を妨げるなら、この直接経路で再投影p50/meanが減るはず。
一方、局所の条件jitterや外れ値が直接headへ流れ、粗さや裾誤差を悪化させる可能性もある。
この試験でTransformerとLayerNormを個別に原因同定したとはしない。
read-outのSVD係数やGT依存oracle translationは候補へ移植しない。

## 判定規則

同arm(c)比で再投影p50≤80%、mean≤90%、他の15数値軸≤105%、behind非増加。
15軸はRMSE全体/gap/no-evidence/event±5/可視camera0〜3の8軸、
全/free/窓内freeの加速度・jerk p95の6軸、再投影p95。
閾値はinclusiveで絶対1e-6＋1e-6×対照値の数値許容。
flowは平均と全sampleの両方を要求し、回帰も別判定する。
両armが通らなければ共通改善とはしない。
実装はrun17の17軸diagnoseを再使用し、hashをplanに固定した。

正式15軸非悪化＋1軸改善＋behind=0も別に適用する。
比較対象は混合平均・RTS・同backbone回帰で、診断合格を正式優位にしない。
元比較のGT/分母、全160予測、全40,000更新の窓順を回収時に検査する。
初期stateはhead形状が異なるためfile SHAの一致ではなく、
共通tensor/時間側列/biasのbit一致＋追加列ゼロを検査する。
失敗時のretry・checkpoint/seed/sample選択・係数再調整なし。

## 投入前検証

178tests（89.01秒、-n4）、ruff/mypy4filesが成功。
headの絶対出力・初期値/RNG保持・新しい列への勾配/optimizer更新、
単因子の追加変更拒否、既存dev trainerのval隔離・両arm・失敗処理を含む。

[実train windowのCPU検査](../../runs/run-i936-head-context-r19-s936/initialization.json)は
(c)保存初期stateの61tensorとのbit一致、乱数状態一致、
初期位置出力の差0とevent出力bit一致を確認した。
B1/T128/全125成分、train-00000だけで全4lossが有限、
追加384係数の勾配と1step更新が非ゼロ。CPU1step後のモデルは破棄し、GPU学習へ渡さない。
元817,285→817,669 parameter。1.177秒/RSS0.967GB。

[CPU preflight](../../runs/run-i936-head-context-r19-s936/cpu-preflight.json)は13.183秒で完了。
75の固定file hash、全640件のmetadata/file hash、
元528train+valの保存hash、同16valの6,383frameと単因子設定が一致。
testと追加valはhash/metadata監査のみで配列を開かない。

## 資源・状態

唯一のresource=all jobとして共有queueに登録する計画。
見積55〜80分/driver peak2〜3GB、上限は外側5340秒＋kill15秒=89分15秒、
trainer5100秒、driver10GB、allocator6GiB、空きRAM6GiB。
新出力150MB予約でrun19のread-out約21.2MBと合わせ5GB以内。
実VRAM・学習時間・性能は未測定で、実行開始を成功として扱わない。
唯一のjob **1790819790903430816_1781754_i936-head-context-r19-s936-20261001** をresource=allで登録済み。
登録時commitは7c4521f9a/clean、worker2083655稼働中。
[queue記録](../../runs/run-i936-head-context-r19-s936/queue.json)と原jobを保存。
学習完了を待たずrunを終了する。
再学習curve/TensorBoard等の証拠はまだ無い。

実Meiji評価・pipeline統合は未実施。項目1の完了、項目2の最終独立較正未完、
項目3の正式優位未達、項目4未着手を維持する。
