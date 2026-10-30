# CPU 成本 baseline 与测试契约

[English](baseline.md) · [日本語](baseline.ja.md)

这里记录 #1401 及runner配置追试 #1523，用于 #1402–#1405。**当前源码的完整CPU观测已完成但有失败，不是成功基线。** 本文不
声称已达到 CI 目标或证明性能提升。机器可读的环境 metadata、inventory、JUnit 和 phase
事件是精确源码和测试集合的依据；失败及取消不会被替换为成功。

## 已完成的本地观测

完整CPU请求执行了 **12,764例：12,551通过、88跳过、125失败（含8个setup错误）**，
命令耗时 **5,790.692秒**。inventory、phase和JUnit均无缺失。这是**失败观测**，
不是成功的性能基线；pytest的117 failures和8 errors均计入summary的failed。

环境为WSL2、Python 3.11.15、Ryzen 7 9700X的12个可见逻辑CPU、23.47 GiB RAM、
2 worker、原生线程各1、隐藏CUDA。其他本地验证同时运行，缓存状态未知。
收集完成耗时78.673秒，末尾报告和退出25.273秒。单进程最大RSS为1,531,464 KiB，
并非并发进程树峰值。墙钟发生−183.694秒校正；耗时使用单调时钟。

| File | 用例 | setup秒 | call秒 |
|---|---:|---:|---:|
| merge gate | 161 | 1630.339 | 766.868 |
| board messages | 539 | 88.313 | 877.462 |
| parent session | 97 | 194.544 | 380.229 |
| ball refiner coordinate training | 9 | 4.012 | 559.781 |
| annotation screening | 86 | 168.074 | 254.512 |

优先减少merge fixture准备并按实测分片。dense/RGB契约已单独测量构造次数下降；
SLCS/MDD smoke成本较低，保留原验证。详情见子任务文档。现有API/Fake失败由
#1504、#1509、#1511修复。screening一例在负载下超过15秒期限，单独重跑通过；
保留此不稳定性记录，不抹去原失败。

相同12例交替测量3组：普通命令中位数 **5.992秒**，带计测 **6.243秒**（+0.251秒）。
启动成本占主导且有其他工作，不能外推整个suite的overhead百分比。全量事件写入4.269秒
仅是开销下限。尚无成功全量运行的中位数或p95；保留失败n=1及其5,790.692秒成本。
[机器证据](baseline-evidence.json)记录来源、环境、inventory标识、phase/fixture成本和
3组overhead。整合后进行同source的Actions比较。


## Actions 参考观测

[ci_attempts.json](../../tests/support/development/ci_attempts.json) 保存真实重试 fixture，
包含各 attempt 的 REST 时间、状态、run/head 标识和来源 endpoint。

| Run | Attempt | 结果 | Apt | Test | 全部 job runner 时间 | Python 观测完成时间 |
|---|---:|---|---:|---:|---:|---:|
| [39767861734](https://github.com/Motoki0705/tennis-lab/actions/runs/39767861734) | 1 | Cancelled | 1480秒 | Skip（0秒） | 1518秒 | 初次创建后1581秒 |
| 同一 run | 2 | Success | 25秒 | 2966秒 | 3027秒 | 重试开始代理时刻后3030秒 |

两次 attempt 共消耗 **4545 runner 秒**。只计算成功重试会丢弃1518秒。未保存准确的重试请求
时刻和 cache 状态，重试起点使用明确标记的 API 开始代理值。成功样本 n=1，中位数及
nearest-rank p95 为3030秒，p95 仅供参考。这不是前后改善比较。

父任务保存的最近23次 main push run 来自不同 revision：成功1、失败1、取消21。
最近保存的成功 [40047681315](https://github.com/Motoki0705/tennis-lab/actions/runs/40047681315)
中 Test 为3338秒，Python runner 为3396秒。
[40063537455](https://github.com/Motoki0705/tennis-lab/actions/runs/40063537455) 在 Test 中
取消。23次 run 的 job/step 数据并不齐全，因此不声称该混合样本的全部 attempt runner
总量、warm-cache 中位数或可信的 p95。

#1401 记载的历史本地部分执行在14分28秒后以退出值2中断，3786 passed / 1 skipped。
其中 merge-gate testcase 总时间641.41秒仅用于发现候选，不是完整 baseline，也不是当前
源码或 Actions runner 的时间估计。

## 候选契约与决策规则

下表的精确 file/function 固定候选组。测量 summary 提供数量及 setup/call/teardown 贡献。
fixture 时间已包含在 setup 内，测试中的 model/CLI 构建属于 call。worker 时间可能重叠，
不能直接作为 job 墙钟节省量。

| 候选/后续 | 当前测试 | 重复契约/成本 | 必须保留的边界 | 决策与不确定性 |
|---|---|---|---|---|
| merge-gate 准备 / #1402 | `tests/e2e/agents_ops/test_merge_gate.py`（`gate_env`）；`test_parent_session.py`（`parent_env`、`make_review_env`） | 每 case 的临时 git 历史、3个PR worktree、review/CLI 准备；历史部分总计641.41秒 | 人类指令及一次 review 消费，最新 head/base/diff coverage，独立/依赖 merge 顺序，共享基础权限，过期/merge失败恢复，dry-run 与 auto-merge 授权；CI 仅为观测 | 当前测量确认后优先复用 fixture、最小化场景。保留 case 私有可变状态和真实 git/CLI 边界，不仅因耗时而共享可变 session 状态。 |
| dense-mask 直积 / #1403 | `test_block.py::test_dense_boundary_rejects_noncanonical_full_masks`（30组） | 5类异常×MHA/GQA×module/direct/update | bool dtype，精确 rank/shape，拒绝 batch/query broadcast，device 不一致；每个 public 入口接入 validator，attention 特有路径及 mask/state 排斥 | 候选方案为公共 validator 异常分类加代表入口连接，需缺陷注入证据，不接受无依据30→5。 |
| direct/module 等价 / #1403 | `test_block.py::test_direct_forward_matches_module_call_for_valid_arguments`（3）；`...values_gradients_and_state`（3） | 重复值比较 | 精确输出、全部输入/参数梯度及 state；MHA/GQA；CSWA all-valid 与部分 invalid 分支 | 明确不同 CSWA 分支后再合并，相似 assert 不足以证明包含。 |
| RGB predictor 直积 / #1403 | `tests/unit/tasks/ball_detection/model_io/test_predict.py`（24组） | 2种 channel×2种实际模型×6类错误 RGB；forward patch 前分配模型 | frame/channel/layout/dtype/spatial/empty 拒绝；DPT/DeepQuery 与 channels2/5 连接；forward 不可到达；保留实际类型 dispatch | 候选为公共 validator 保留全部异常，实际入口保留代表连接。不能用删除 `isinstance` 行为的通用 Mock。 |
| SLCS smoke / #1404 | `tests/integration/tasks/slcs/test_training_smoke.py::test_dataset_model_loss_backward_smoke`（4组） | frames6/18×context 开/关，重复训练剩余 window 及4种 mode 评估 | padded/short/full window；adapter→model→loss/backward；context 梯度连接；optimizer 更新；checkpoint 重载；full/no_rgb/detector_gap/rgb_only 保存 target/mask/ID/frame | 耗时显著时再分离契约并最小化重复 step。unit 梯度检验不能替代 integration context 连接。 |
| 正常 fit/callback / #1404 | `...test_real_rgb_profile_normal_fit_saves_monitored_checkpoint` | 生产 trainer/logger/callback 初始化 | 真实监控 metric，正常 fit，checkpoint callback，真实 TensorBoard event | 保留真实正常 fit smoke；`fast_dev_run` 或手写 optimizer 不能覆盖 callback 契约。 |
| MDD module 结构 / #1404 | `tests/unit/tasks/ball_detection/models/mdd/test_cnn.py` 的生产 config 结构检验 | 为 Conv2d/Conv3d 数量及 SwiGLU 宽度分配全部权重 | 真实组装 module tree 及宽度/数量不变量 | 仅在耗时显著且所有初始化支持时考虑免分配构建。不可替换为 config 值 assert，拒绝不支持的 meta 初始化。 |
| repo 静态扫描 / 后续候选 | `tests/e2e/development/test_architecture_boundaries.py` | 重复文件发现、AST、引用解析 | 新增文件、已删 module、动态/CLI 入口、transitive forward helper、负例 policy 检验 | 保留覆盖；若实测支持，可复用每 worker 的不变扫描/index，不能仅因数量大而删除。 |

上述 `test_block.py` 为 `tests/unit/utils/models/components/test_block.py`。这些是契约组，
不是已批准删除清单。未测量的 overlay/installer/DSH 直积暂缓。#1401 不修改测试、marker、
生产 model 或 validator。

## 固定比较流程

1. 比较 job 组成时固定默认 CPU 命令、源码、完整 inventory、依赖版本、runner OS/image/
   CPU/RAM、worker/thread、cache 和竞争负载，保留取消及准备失败在内的全部 attempt。
2. 合并测试导致 inventory 改变时，比较保留契约及代表缺陷检测，另记 case、fixture/CLI/
   model 初始化次数及 phase 时间。移动测试层级不等于减少 case。
3. 同时记录 n、成功中位数/nearest-rank p95、失败/取消数量及全部 job runner 累计。
   小样本 p95 仅供参考；cache 或准备/后处理缺失时不得声称 warm-cache。
4. 在固定源码上运行 [README.zh-CN.md](README.zh-CN.md) 的完整 CPU 入口。部分候选测量
   只用于排序，不能证明全套等价或达到50%/约15分钟目标。

完整观测及限制已记录于上文；整合后记录成功的同source Actions比较。

加入job日志转发后，重新交替测量相同12例3组：普通中位数 **4.213秒**，计测中位数 **4.467秒**（+0.254秒）。六次均成功且inventory相同。机器证据的 `overhead_live_log` 保留独立来源；启动成本占主导，不能将与旧样本的差异视为版本间提速。

## runner配置比较

最初四个branch配置使用相同source、更新后的profile和12,795例CPU inventory，其中包括13个新增CI回归用例。10 shard追试新增3个CI回归用例（共12,798例），在机器证据中单独标明source，产品侧测试保持相同。各配置n=1。旧4 × 2参照使用旧profile和12,782例，只是背景，不是同源码对照。总时间从Actions的 `created_at`（含queue）到 `Python` 汇总及后处理完成；runner时间累计lint和汇总在内的全部job。包括失败在内的所有attempt均保存在[机器证据](scheduling-evidence.json)中。

| 配置 / run | Python完成 分钟 | shard最短–最长 分钟（差 秒） | 全部job runner 分钟 | test cache hit | 最大单进程RSS GiB | 失败用例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [4 × 2 / 40986313121](https://github.com/Motoki0705/tennis-lab/actions/runs/40986313121) | 26.22 | 14.42–25.80 (683) | 82.60 | 4/4 | 1.70 | 1 |
| [4 × 4 / 40992996490](https://github.com/Motoki0705/tennis-lab/actions/runs/40992996490) | 20.53 | 12.62–20.13 (451) | 66.83 | 4/4 | 1.65 | 2 |
| [6 × 4 / 40995509440](https://github.com/Motoki0705/tennis-lab/actions/runs/40995509440) | 16.22 | 11.73–15.77 (242) | 87.98 | 6/6 | 1.64 | 2 |
| [6 × 2 / 40997143981](https://github.com/Motoki0705/tennis-lab/actions/runs/40997143981) | 18.30 | 9.93–17.88 (477) | 90.33 | 6/6 | 1.60 | 1 |
| [8 × 2 / 40999125159](https://github.com/Motoki0705/tennis-lab/actions/runs/40999125159) | 15.02 | 10.20–14.62 (265) | 104.20 | 8/8 | 1.51 | 1 |
| [10 × 2 / 41001256601](https://github.com/Motoki0705/tennis-lab/actions/runs/41001256601) | 14.33 | 9.57–13.88 (259) | 118.45 | 10/10 | 1.50 | 1 |

profile重新生成与fixture边界以[分片文档](sharding.zh-CN.md#重新生成实测profile)为准。单进程RSS不包含测试子进程，也不是同时process tree峰值。实测runner均有4 vCPU、约16 GB RAM，但CPU代际不同。完整shard观测保留全部worker结束记录，未观察到OOM或worker丢失。native library环境变量保持1，但部分既有测试将PyTorch intraop改为2/4并泄漏到后续用例，因此4 worker仍可能争用CPU。测试本体修复交由父会话。

所有run均保留已知README.zh-CN失败。两次4 worker观测还出现 `test_timeout_reaps_group_when_output_is_open_or_closed[True]` 失败：timeout cleanup后立即检查时观察到仍在运行的child。没有忽略失败或转为成功。这些失败观测不能建立成功warm-cache的中位数或p95。source/profile/inventory审计保留；完整执行但失败时记录 `observations_complete=true`、`success=false`。

可选的本地全量collect-only在竞争负载下取消，不作为性能或成功证据。完整执行证据来自Actions manifest和coverage成果物。父会话合并fixture优化后应重新生成profile并重测所选配置。hardware差异、样本量与thread状态泄漏限制结果外推。

默认选择10 × 2，这是实测2 worker配置中，仅有已知README失败且Python在900秒内完成的最小配置；8 shard耗时901秒。观测runner成本从104.20升至118.45分钟（+13.7%），比旧参照82.60分钟增加43.4%。CPU代际与source不同，不能将全部成本差归因于增加job。新增5个attempt共消耗467.80 runner分钟，包含所有失败，无取消。父会话合并fixture优化后，应重新生成profile并重新评估减少job数量。
