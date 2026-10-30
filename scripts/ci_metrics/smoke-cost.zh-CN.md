# 模型 smoke 测试成本决策

[English](smoke-cost.md) · [日本語](smoke-cost.ja.md)

## 决策

针对 [#1404](https://github.com/Motoki0705/tennis-lab/issues/1404)，保留现有 SLCS
训练/评估与 MDD 结构测试。模型处理的测量成本较小，没有依据缩减集成覆盖或增加 meta
device 初始化支持。优先优化 merge fixture 准备和 CPU 分配。

## 2026-10-25 的观测

复现命令见[英文版](smoke-cost.md#observation-2026-10-25)。本地 Linux 上使用两个
worker、原生线程上限为 1，并隐藏 CUDA，12 个测试全部通过。这只是一次部分观测，
不是全 CPU baseline，也不是 GitHub-hosted runner 的性能。源码/环境、inventory、
phase event、JUnit 和完整性结果一并保存，没有复用之前取消的测量。

| 工作 | 用例数 | call 合计秒 |
|---|---:|---:|
| SLCS 训练 smoke（含普通 fit/callback） | 5 | 4.582 |
| MDD CNN 测试 | 7 | 1.054 |
| 正式 MDD 配置结构检查（已包含在上行） | 1 | 0.099 |

命令的单调时钟时间为 24.853 秒，worker call 合计 5.636 秒。全部 worker 收集完成耗时
14.347 秒，最后一次 phase 报告到命令退出为 7.248 秒。worker 时间可能重叠，不应与实际
时间相加。单个 pytest 进程峰值 RSS 为 958,972 KiB，不代表进程树总内存或单个模型内存。

## 保留的契约与重新评估条件

没有删除、移动或合并测试。保留 frames 6/18 × context off/on 的全部组合、真实
adapter/model/loss backward 接线和 context 梯度、optimizer 更新、checkpoint 重载，
以及四种评估模式的 target/mask/frame/ID 保存。继续验证普通 fit、监控指标、checkpoint
callback 和真实 TensorBoard event。MDD 测试仍构造正式模块树并验证卷积数量及 SwiGLU 宽度。

由于没有删除用例，无需删除映射或前后变异比较。若未来全量测量显示这些测试成为主要
成本，或正式配置扩大会改变初始化成本，再重新评估。一般测量与比较契约见
[README](README.zh-CN.md)。
