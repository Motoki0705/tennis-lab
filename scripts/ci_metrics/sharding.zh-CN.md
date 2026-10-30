# 完整 CPU 测试分片

[English](sharding.md) · [日本語](sharding.ja.md)

CI 独立执行 lint，并默认将完整 CPU 目标分给4个测试 job，每个使用2个 worker。手动dispatch可选 `shards=1/2/4/6/8` 和 `workers=2/4`，比较相同源码的调度。明确指定worker数量，不使用依赖主机的 `auto`。手动单job基准上限120分钟，普通shard仍为60分钟。固定名称
`Python` 使用 `always()`，仅在 lint、所有测试 job 和覆盖审计成功后通过。CI 不恢复为
merge 前条件。本次保留现有 trigger；每日维护迁移属于
[#1349](https://github.com/Motoki0705/tennis-lab/issues/1349)。

## 分配与 fixture

每个 worker 用既有 CPU marker 和 court dataset 排除规则收集当前源码，再分配全部
nodeid，包括新测试。`cpu-file-times.json` 保存测得的 phase 时间；按成本从大到小分配
给当前最轻的 shard，同成本按名称/index稳定排序。默认按文件保留共享准备。
仅将 function fixture 独立的重文件明确列入 `split_files`，使用各用例 `test_seconds` 分配。
添加前须检查 fixture 共享范围。若后来加入module/class fixture，收集会拒绝该分割文件。

未测文件用已测文件中位数，新分割用例用该文件平均成本估计；manifest 明确列出估计值，
不会因此省略测试。不合法 profile 明确失败。命令见[英文版](sharding.md#assignment-and-fixture-boundaries)，
不分片的完整执行见[README](README.zh-CN.md)。单个 shard 的 `full_cpu_baseline=false`；
只有汇总验证完整覆盖。不按修改路径或 slow marker 省略。

## 证据与失败语义

`shard-gw*.json` 保存完整 CPU inventory、分组、选中 nodeid、profile身份和未知估计。
汇总从实际 phase 重新生成报告，并要求：

- 所有 shard index 和 worker manifest 恰好出现一次。
- 干净源码 commit、run、attempt、profile、完整 inventory 一致。
- 分片前请求完整 CPU，观测完整且成功，实际执行等于计划。
- 所有执行 nodeid 的多重集合等于完整目标，不缺失也不重复。
- lint 与全部 matrix job 成功，不能把 skip/cancel 视为成功。

准备或分配失败、缺失 artifact、失败/取消或其他 HEAD 均使 `Python` 失败。
原有合法 CPU 排除和 setup/module skip 保持可见。matrix 禁用 fail-fast。apt 的网络重试
和 step 时长有上限，下载失败不会转换为省略测试。

artifact 名包含 run/attempt/shard。`cpu-test-coverage-*` 中的 `coverage.json` 保存汇总
结果和失败原因，每日 CI 可用这些标识关联 JUnit/manifest。重试须重新执行全部 job；
不会将以前 attempt 的成功结果混入部分重试。

## 测量限制

phase 合计不是 runner 实际时间；增加 job 会重复环境准备和收集。负载与内存须结合实际
观测，证据及比较限制见[baseline](baseline.zh-CN.md)。fixture/suite 大幅变化后，从完整
测量刷新 profile，不能只按文件数量选择分片数。

依赖准备检查已有ffmpeg/rclone版本，仅安装缺失binary，再验证两个命令。apt各限180秒，网络重试2次、超时30秒，Actions step上限8分钟。7个subprocess场景覆盖已有/缺失、update/install失败、timeout、安装后仍缺失与版本检查失败。

## 重新生成实测profile

下载同一attempt的所有 `cpu-test-measurements-RUN-ATTEMPT-shard-*`，每个shard放在单独目录中。

```bash
.venv/bin/python -m scripts.ci_metrics profile --root /absolute/artifacts \
  --output scripts/ci_metrics/cpu-file-times.json
```

命令重建phase/JUnit报告，核对source/run/attempt、全部worker manifest、分配与完整inventory多重集合。拒绝phase缺失、重复、dirty/mixed source、取消和收集失败。完整记录的测试失败保留在权重及 `source.counts` / `unsuccessful_cases` 中；生成权重不会令CI通过。`Python` 仍要求job和观测成功。审计另用 `observations_complete` 表示完整性。

默认保留观测中的 `split_files`。检查fixture后，重复传入 `--split-file tests/path.py` 可明确替换列表。收集guard会根据当前fixture graph验证声明。merge gate、board messages、parent session继续按function fixture拆分。坐标训练使用module scope的 `shared`，所以9个用例的579.907秒保留在一个shard内；不修改测试本体与fixture scope。

新profile来自[run 40986313121](https://github.com/Motoki0705/tennis-lab/actions/runs/40986313121)：830个文件、12,782例（12,685 passed / 96 skipped / 1 failed）、共8,527.317 phase秒，保留已知README.zh-CN失败。4 / 6 / 8 shard的估计最大负荷为2,131.830 / 1,421.220 / 1,065.916秒。这是并行worker phase的累计值，不是wall时间预测。旧分配的实测负荷为2,781 / 1,435 / 1,691 / 2,621秒。父会话合并fixture优化后需重新生成。
