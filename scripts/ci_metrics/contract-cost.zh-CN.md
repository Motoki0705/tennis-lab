# 输入契约测试整合

[English](contract-cost.md) · [日本語](contract-cost.ja.md)

[#1403](https://github.com/Motoki0705/tennis-lab/issues/1403) 将异常分类放在共享 validator，
并单独验证各公开入口、实际模型配置在计算前调用它。不改变正式代码、数值容差或已有 state/互斥检查。

| 原有用例 | 保留的检测和交互 | 决策 |
|---|---|---|
| dense mask 5 类错误×MHA/GQA×module/direct/update＝30 | 共享 validator 的5类检查和全部6种入口/方式；保留 dtype、严格 rank/shape、广播拒绝及 device | 11例，block 构造30→6 |
| 三种方式的 direct/module 正常值相等 | 已有值、输入/参数梯度及 state 比较覆盖 dense；另保留 CSWA all-valid 分支 | 合并两个较弱 dense 用例 |
| RGB 6类错误×DPT/DeepQuery×channels 2/5＝24 | 共享 validator 保留原6类并增加 rank、width、最小有效输入；4种实际类型/channel 接线验证和 forward 未到达 | 13例，detector 构造24→4 |

完整文件/函数映射见[英文表格](contract-cost.md)。原114例和新82例均通过，净减少32例，
不是单纯移动测试层级。候选用例的 block/detector 构造57→11，没有共享可变模型。

## 故障检测与测量

| 临时注入的故障 | 原版失败数 | 新版失败数 |
|---|---:|---:|
| 允许 float dense mask | 6 | 7 |
| 允许 float decoded RGB | 4 | 5 |
| 绕过 DPT 公开输入 validator | 12 | 2 |

6次执行均以普通测试失败、退出码1结束，没有收集/setup错误。spy 包装实际绑定的函数；
替换为已知正确的 validator 会隐藏接线缺陷。两个 channel 配置都能检测 DPT 绕过。

[机器证据](contract-evidence.json)记录版本、数量、phase 时间和变异结果。每版只进行一次
本地部分测量，禁用 xdist worker、原生线程限制为1并隐藏 CUDA。其他验证同时运行，
没有控制启动、缓存或主机负载；不能据此证明 CI 实际时间改善或父任务的 warm-cache
中位数/p95目标。复现方式见[英文版](contract-cost.md)，通用观测契约见[README](README.zh-CN.md)。
