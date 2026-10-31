# 完整 CPU 测试分片

[English](sharding.md) · [日本語](sharding.ja.md)

CI 独立执行 lint，并默认将请求的 CPU 目标分给10个测试 job，每个使用2个 worker。手动dispatch可选 `shards=1/2/4/6/8/10` 和 `workers=2/4`，比较相同源码的调度。明确指定worker数量，不使用依赖主机的 `auto`。手动单job基准上限120分钟，普通shard仍为60分钟。固定名称
`Python` 使用 `always()`，仅在plan、lint、所有测试job和覆盖审计成功后通过，保持merge必需check。
PR选择与main/手动完整请求的正本为[通道注册表与计划](README.zh-CN.md#通道注册表与计划)。仅PR取消之前的run。

## 分配与 fixture

每个 worker 用既有 CPU marker 和 court dataset 排除规则收集当前源码，先记录通道选择前的完整集合，再分配请求的
nodeid，包括未注册的新测试。`cpu-file-times.json` 保存测得的 phase 时间；按成本从大到小分配
给当前最轻的 shard，同成本按名称/index稳定排序。默认按文件保留共享准备。
仅将 function fixture 独立的重文件明确列入 `split_files`，使用各用例 `test_seconds` 分配。
添加前须检查 fixture 共享范围。若后来加入module/class fixture，收集会拒绝该分割文件。

未测文件用已测文件中位数，新分割用例用该文件平均成本估计；manifest 明确列出估计值，
不会因此省略测试。不合法 profile 明确失败。命令见[英文版](sharding.md#assignment-and-fixture-boundaries)，
不分片的完整执行见[README](README.zh-CN.md)。单个 shard 的 `full_cpu_baseline=false`；
只有汇总验证完整覆盖。通道选择在分片前执行；时间或slow marker不决定成员。

## 证据与失败语义

schema-2的 `shard-gw*.json` 保存 `universe_nodeids`（U）、`requested_nodeids`（R）、
`deferred`（D: nodeid到suite id）、plan/policy/scope、分组、选中nodeid（S_i）、profile和未知估计。
独立的 `cpu_universe` event在CPU marker之后、通道选择之前记录U；`lane-plan.json` 保存固定请求。
汇总从实际 phase 重新生成报告，并要求：

- 所有 shard index 和 worker manifest 恰好出现一次。
- 干净源码commit、run、profile、固定lane plan（`plan_id`）、policy、U一致。
- attempt必须为正整数字符串，且lane plan attempt ≤ 各shard attempt ≤ 汇总job attempt。
- 注册表和源码文件清单与固定plan一致，独立event的U与manifest的U一致。
- U = R ⊎ D，每个延后case均有注册表与plan依据。
- 请求scope的观测完整且成功，实际执行等于计划。
- R = S_0 ⊎ ... ⊎ S_n，无缺失、重复或空shard。
- plan、lint与全部matrix job成功，不能把skip/cancel视为成功。

从所有manifest删除同一个case也会被独立的选择前event检出。D不计入skip或成功。
汇总报告 `scoped_cpu_success`，而 `full_cpu_success` 还要求D为空。
U是既有marker与明确排除后的CPU集合，不代表所有环境的全部测试。
历史schema-1 artifact保留 `full_nodeids` 的含义，可供profile读取；混用schema或与lane plan一起使用会失败。

准备或分配失败、缺失 artifact、失败/取消或其他 HEAD 均使 `Python` 失败。
原有合法 CPU 排除和 setup/module skip 保持可见。matrix 禁用 fail-fast。apt 的网络重试
和 step 时长有上限，下载失败不会转换为省略测试。

部分重试仅在同一run、同一HEAD（`source_sha`）和同一lane plan（`plan_id`）内，
复用固定plan与每个shard的最新结果。未重试的成功shard可保留之前的attempt。
attempt可在上述范围内混用；其他run、HEAD或plan会被拒绝。

plan artifact名为 `cpu-lane-plan-RUN`，测量名为 `cpu-test-measurements-RUN-shard-N`。
两者上传均设置 `overwrite: true`，仅替换重试的plan/shard。
失败attempt的测量可能被覆盖删除，job日志仍保留。重试全部job时，GitHub会删除之前的artifact。
下载明确指定 `run-id` 与 `github-token`，使用覆盖整个run的REST API。
汇总先取得REST清单的所有分页，要求未过期的plan与每个预期shard各恰好一个artifact。
缺失、重复或额外shard均明确失败。下载使用检查过的artifact ID，并保持shard独立目录，防止文件静默混合。

`cpu-test-coverage-RUN-ATTEMPT` 名称保留汇总job的attempt。
`coverage.json` 保存 `attempt`（汇总）、`plan_attempt`、`shard_attempts`（index → attempt）、
scope、suite结果与失败原因。同一artifact内的 `artifacts.json` 保存检查过的下载集合的
ID、name、`created_at`、`size_in_bytes` 以及清单错误。
仅重试Python时，也能带着这些证据重新验证较早attempt的shard。

## 测量限制

phase 合计不是 runner 实际时间；增加 job 会重复环境准备和收集。负载与内存须结合实际
观测，证据及比较限制见[baseline](baseline.zh-CN.md)。fixture/suite 大幅变化后，从完整
测量刷新 profile，不能只按文件数量选择分片数。

依赖准备检查已有ffmpeg/rclone版本，仅安装缺失binary，再验证两个命令。apt各限180秒，网络重试2次、超时30秒，Actions step上限8分钟。7个subprocess场景覆盖已有/缺失、update/install失败、timeout、安装后仍缺失与版本检查失败。

## 重新生成实测profile

下载同一run、HEAD和lane plan的最新 `cpu-test-measurements-RUN-shard-*`，每个shard恰好放在一个独立目录中。

```bash
.venv/bin/python -m scripts.ci_metrics profile --root /absolute/artifacts \
  --output scripts/ci_metrics/cpu-file-times.json
```

命令重建phase/JUnit报告，核对source/run/plan、attempt范围、全部worker manifest、分配与完整inventory多重集合。混用attempt时，以shard的最大attempt作为汇总attempt，并记录 `source.plan_attempt` 与 `source.shard_attempts`。拒绝phase缺失、重复、dirty/mixed source、取消和收集失败。完整记录的测试失败保留在权重及 `source.counts` / `unsuccessful_cases` 中；生成权重不会令CI通过。`Python` 仍要求job和观测成功。审计另用 `observations_complete` 表示完整性。

只要存在延后case，就拒绝生成profile。部分成功不能覆盖完整CPU权重或宣称完整baseline。

默认保留观测中的 `split_files`。检查fixture后，重复传入 `--split-file tests/path.py` 可明确替换列表。收集guard会根据当前fixture graph验证声明。merge gate、board messages、parent session继续按function fixture拆分。坐标训练使用module scope的 `shared`，所以9个用例的579.907秒保留在一个shard内；不修改测试本体与fixture scope。

新profile来自[run 41043837101](https://github.com/Motoki0705/tennis-lab/actions/runs/41043837101)（#1527/#1529 fixture优化之后）：832个文件、12,883例（12,787 passed / 96 skipped / 0 failed）、共6,681.075 phase秒。4 / 6 / 8 / 10 shard的估计最大负荷为1,670.269 / 1,113.513 / 835.135 / 668.108秒。这是并行worker phase的累计值，不是wall时间预测。旧profile下10 shard的实测负荷为156–916秒。合并改变测试成本的修改后需重新生成。

## 默认配置选择

默认使用10 shard、2 worker，这是实测2 worker配置中，在15分钟内完成且没有已知README之外失败的最小配置。4 worker重复出现process cleanup后立即检查的额外assert失败，保留为手动比较选项。时间、runner成本、cache、RSS的完整表与样本限制以[baseline](baseline.zh-CN.md#runner配置比较)为准。父会话合并fixture优化后，应重新生成profile并重测6/8 shard是否足够。

也检查了 `test_parent_development.py`、`test_execution_basis.py` 和 `test_parent_completion_resume.py` 的function fixture。其文件成本在更新后的每shard预算内，因此实测默认配置保留既有3个文件的拆分声明，待计划中的fixture优化后用新测量重新判断。
