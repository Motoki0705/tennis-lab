# CPU 成本 baseline 与测试契约

[English](baseline.md) · [日本語](baseline.ja.md)

这是 #1401 的测量记录，用于 #1402–#1405。**当前源码的完整CPU观测已完成但有失败，不是成功基线。** 本文不
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
