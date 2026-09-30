---
id: run-i936-combined512-physics10-r16-s936
type: run
task: ball_refiner_3d
sequence: 26
recorded_at: '2026-10-01'
title: 512trainとphysics10を併用する固定20k候補
issue: 936
provider: codex
date: '2026-10-01'
status: done
config:
  source: /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/src/tasks/ball_refiner/refiner_3d/training_pilot512_physics10_t128.yaml
  primary_update: 20000
  factor: training_rallies_and_physics_weight
  validation_rallies: 16
metrics:
  flow_val_rmse_m: 1.9531850263102082
  regression_val_rmse_m: 2.054918208718
artifacts:
  run_dir: knowledge/runs/run-i936-combined512-physics10-r16-s936
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/h-dev/r16-combined512-physics10-s936-t128-20k
parents:
- run-i936-pilot512-t128-r15-s936
- run-i936-physics10-t128-r14-s936
relations: []
papers: []
tags: []
session: 01a0f3ed-4151-7303-886f-52e91fdac10e
---

[事前登録の原文](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5918885662)が比較規則の正本。
[plan.json](../../runs/run-i936-combined512-physics10-r16-s936/plan.json)に全code/config/input hash、command、資源上限を固定した。
別々に測った512trainのRMSE改善とphysics10の粗さ低下を併用する**候補モデル**であり、一因子の原因診断とはしない。
同じ16val/6,383frame、seed936、初期化、T128/stride128/B8、fp32、各20k、0/2k/5k/10k/20k、flow4samples×8steps。
主20kを守り、checkpoint/seed/component選択をしない。H/default、全125成分、#959 controlを維持する。
追加48val/testは配列を開かず、実Meiji/pipelineには進まない。

候補診断は(a)の誤差/再投影と(b)の粗さの両効果保持を要求し、失敗軸も全て残す。
正式な15軸＋behindゼロ規則は変更しない。5%を許容する候補診断と正式優位を区別する。
新guardは宣言した二因子以外を拒否し、固定val/preflight/完了manifestの要件は既存と同じ。
TensorBoardはなく全更新JSONL・曲線PNG・各時点予測を保存する。結果は未観測。

GPU見積60〜85分、peak VRAM2〜3GB、disk150MB、resource=allを1件。
soft5100秒、外側timeout5340秒+kill15秒、allocator6GiB/driver10GB、空きRAM6GiB下限、native1。
自動retryなし。worker停止時は起動し、学習完了を待たずWAITING_QUEUEで終了する。
(d)は既存(a)重みのCPU overlap比較なので追加GPU枠は使わない。

通常検証は27tests（-n4、12.28秒）成功。guard単位テストで二因子を一因子として扱うこと、第三の設定変更、
宣言外のcounts/physics倍率を拒否した。実行経路の固定val/初期化/全窓/失敗記録のintegrationも通過。
ruff/mypy成功。今回の型検査はfollow-imports=skipで既存returnのAnyを検出したため、返却dictの型を明示した。

[cpu-preflight.json](../../runs/run-i936-combined512-physics10-r16-s936/cpu-preflight.json)は4.232秒で成功。
全640保存hash/JSON/plan、元80train+val、同16val/6,383frame、66固定入力hashを確認した。
学習出力・本番preflight先は未作成。今回変更しない対照(a)/(b)のmanifest hashもpinした。


## Run 16: 共有queue登録

2026-10-01 **05:29:54 JST** に **1790800194473949395_4059977_i936-combined512-physics10-r16-s936-20261001** をresource=allで1件だけ登録した。
[queue.json](../../runs/run-i936-combined512-physics10-r16-s936/queue.json)と[queued.job](../../runs/run-i936-combined512-physics10-r16-s936/queued.job)が登録時原本。
登録HEAD **a7ba039b1ddd9df80c3b76a9c8a7da3eec7d7b71**、worktree clean、plan SHA **e8ca185a9f9cfbe05514653001129181d86b2185ddf12acbbd0650f0e2c8b19a**、66固定hashを再確認。
登録時はworker停止、commit後に起動してPID/状態をissueへ記録する。`worker_start_pending`は登録瞬間のsnapshot。
(d)はCPU完了・GPU0件で、追加GPU jobやretryは行わない。CPU overlapの結果を(c)設定へ反映せず、元stride128を維持。
log/reproは共有queue内の本job ID、出力は上記output_dirと同pathの`-preflight.json`。
GPU完了を待たずWAITING_QUEUE。次runでpreflight/repro/全5評価時点/全予測/hash/資源を回収し、事前登録の候補診断と正式規則を別々に適用する。

## Run 17: 全評価点の回収と固定20k判定

queueはdone、実行commit **50098f0e6923b11ce0f856ef473e0d8bc4d4c8a0**、reproのstatus/patchは空。
[全時点・全軸の比較表](../../runs/run-i936-combined512-physics10-r16-s936/collected/comparison.md)、
[回収監査・正式規則](../../runs/run-i936-combined512-physics10-r16-s936/collected/collection.json)、
[両効果保持診断](../../runs/run-i936-combined512-physics10-r16-s936/collected/diagnostic-rule.json)を保存した。
全160予測の平均/全sample・可視camera層を再集計し、元metric/分母/Markdownと一致。
GT/mask/camera、同16val/6,383frame、run12 baseline、初期重み/初期平均metric、
(a)と両armの全40,000更新窓・実frame数も一致。preflightの再計算・66固定hashを確認した。
追加48val/test配列は開いていない。directiveの27数値は指定桁で全て一致した。

| 20k | RMSE/gap m | accel all/free p95 | jerk all/free p95 | repro mean/p50/p95 px | behind |
|---|---|---|---|---|---|
| RTS | 4.408/3.020 | 669/158 | 87998/7133 | 16.436/2.472/74.812 | 1/17528 |
| flow平均 | 1.953/1.944 | 412/183 | 42809/14299 | 20.731/12.096/64.352 | 4/17528 |
| flow全sample | 1.956/1.946 | 416/185 | 43143/14446 | 20.736/12.104/64.216 | 16/70112 |
| 回帰 | 2.055/2.170 | 753/196 | 91523/13783 | 18.581/9.624/60.007 | 4/17528 |

**両効果保持はflow平均/全sampleが不合格、回帰は合格**。
flowの失敗軸は1camera可視RMSEだけで、(a)比6.151/5.800m=1.0605、
全sampleは6.163/5.812m=1.0602。105%上限を超えるため両arm共通の保持とはしない。
他の誤差軸、(b)比のfull/窓内free accel/jerk、behind非増加は通過した。
**正式15軸＋behind=0は両arm未達**。flowはRTSにfree accel/jerkとrepro mean/p50が悪く、
混合平均にもrepro mean/p50が悪い。回帰に対してもevent/camera2/3・free jerk・repro3軸が悪い。
回帰もRTS比の粗さ4軸とrepro mean/p50が悪い。全失敗軸は上のJSONに残した。

全5評価点を残す。flowのRMSEは0/2k/5k/10k/20kで14.265/3.539/2.180/2.077/1.953m、
回帰は16.333/3.268/1.965/2.612/2.055m。回帰5kなどへのcheckpoint再選択はしない。
二因子併用であり、これだけでdata量/physicsの相互作用の因果を断定しない。

trainer3141.439秒（52.357分）、queue wall2973秒（05:31:02→06:20:35 JST）。
168.439秒の時計差の原因は未確認で、短い方だけを費用として採用しない。
allocated422,902,272 bytes、reserved471,859,200 bytes、driver最大観測1,772,683,264 bytes
（連続peakではない）、最小空きRAM24,100,556,800 bytes、元出力46,011,370 bytes。
再計算scriptはcollect.py→diagnostic.pyの順に、当bundleが追加されたcommitをcheckoutしてCPU/native1で実行する。
学習再現commitと、後から追加した収集scriptを含むcommitを区別する。TensorBoardなし、全更新JSONL.gzと曲線PNGを保存。

次は指示された同一固定blendを(c)20kへCPUで適用し、RTSへの残差を保存予測で分解する。
H/default・bank・#959 control・正式規則は不変。実Meiji/pipelineは未評価。
