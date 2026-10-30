# CPU CI 测量

[English](README.md) · [日本語](README.ja.md)

这些工具规划并观测 CPU 测试集。CI 使用[分片与覆盖审计](sharding.zh-CN.md)，默认本地命令仍完整执行且不分片。
CPU marker 表达式仍以 `.spin/cmds.py` 为准，court dataset
集成测试保留原有的明确排除。CI 与本地测量均使用 `spin test -q --no-cov --dist=worksteal`、
明确worker数量（本地默认2，CI参见[分片设置](sharding.zh-CN.md)）、各原生库线程数为 1、隐藏 CUDA、`--durations=50 --durations-min=1` 和 JUnit。

pytest的stdout/stderr实时转发到job日志，并保存到`pytest.log`，保持现有`ci_watch`对失败nodeid和setup异常的识别。输出pipe关闭时launcher失败并停止测试process group。


## 通道注册表与计划

[`lanes.json`](lanes.json) **只注册按文件指定的例外**，其余 CPU 测试始终执行。
初始 `suites` 为空，保持原有目标。每个suite必须包含 `id`（唯一标识）、`files`（存在的明确测试路径）、
`paths`（依赖项的仓库相对shell pattern）、`reason` 和 `owner`（非空字符串）。文件重复、不存在、
未知字段、缺少负责人/理由或非法pattern都会使plan失败。文件发现遵循 `pyproject.toml` 的pytest设置。
未注册文件始终执行；已注册文件的全部收集用例，包括新增用例，都属于该例外。
suite自身的测试文件修改也会要求执行，不受 `paths` 限制。注册时须包括局部helper、配置与依赖路径。

PR比较event中的 `base.sha..head.sha`，即使之前的PR run被取消，也包含所有commit的改动。
仅PR按路径选择。main push和手动dispatch始终要求完整CPU，不设置定期通道。
共享基础设施（`src/utils`、`src/tasks/base`、`tests/support`、所有 `conftest.py`、依赖配置/lock、
`.spin`、workflow及本CI工具/注册表）修改、rename、删除或无法取得diff时，记录理由并执行全部。
非法注册表或plan直接失败，不转成部分执行。

只使用标准库的 `plan` job生成一个固定source/run/attempt的 `plan.json`，供所有shard与 `Python` 汇总使用：

```bash
python3 -m scripts.ci_metrics.plan --output .ci-plan/plan.json
```

命令读取 `GITHUB_EVENT_PATH` 与Actions context，并将注册表、源码测试文件清单与各checkout核对。
普通shard命令添加 `--lane-plan .ci-plan/plan.json`，不能同时传任意pytest参数。
本地分片测量未指定plan时明确要求完整CPU。

plan与汇总summary列出延后suite、负责人、理由以及main最近一次完整成功的commit。
只读Actions API最多检查最近30次成功的main push，核对祖先关系、policy、run/attempt与
`cpu-test-coverage-*` 证据。绿色job、部分run、全部skip的suite不能证明成功。
缺失/过期/不兼容证据与API失败均保留为附理由的 **unknown**。coverage artifact保留90天；
不向仓库commit成功index。

## 执行与中断恢复

在待测 checkout 中安装锁定环境后运行：

```bash
.venv/bin/python -m scripts.ci_metrics run --output /absolute/new/output --workers 2
```

输出目录必须为空。遵循仓库的 pytest 并行运行规则，默认不取得主机共享锁，
也不使用 `PYTEST_LOCK`。显式协调的性能比较可用 `--lock /absolute/path`，
仅串行化指定相同路径的测量。普通 pytest 不受该锁约束；须记录主机竞争负载并在相同条件下比较。

从启动起（包括锁等待）须固定源码。等待期间发生变化时，应重新启动进程，避免已导入的
测量代码仍是旧版本。

部分测试可放在 `--` 后，例如 `-- tests/unit/utils/geometry/test_angles.py`。任何额外参数都会使
`full_cpu_target` 为 false，部分执行不能建立完整 CPU baseline。不要覆盖 worker、marker、
JUnit、输出目录或 plugin 参数。

runner 仅从测试子进程移除真实 agent 的 Herdr/session 标识及外层 pytest 的 worker 标识和
选项，不修改调用者环境。setup 失败或 skip 时没有 call report 属于正常情况，仍分别记录
失败或 skip，与 phase 缺失区分。信号会转发到整个 pytest 进程组并记录为 cancelled。
强制 kill 或 runner 丢失可能没有结束记录；可从已有证据重新汇总，而不重跑测试：

```bash
.venv/bin/python -m scripts.ci_metrics report --output /absolute/partial/output
```

`run` 保留 pytest 的非零退出值；pytest 成功但测量不完整时返回 2。`report` 无法确认成功
观测时也返回 2，不用另一次执行填补缺失证据。
若取消或启动失败导致测试命令尚未开始，`run.json` 的测试退出值保持 null，
与 CLI 自身的信号或错误退出值分开。

## 产物与时间定义

| 文件 | 内容 |
|---|---|
| `environment.json` | revision、本地改动、命令、依赖版本、OS/image、CPU/affinity、RAM、worker/thread、run/attempt、cache、起始主机负载 |
| `run.json` | 请求/开始/结束时间、独立锁等待、退出值/信号、执行期间源码一致性 |
| `events-*.jsonl` | 每进程及时 flush 的收集/排除 nodeid、收集失败/skip、worker 收集、fixture setup、测试 phase、结束记录 |
| `inventory.json` | 被选中的 nodeid，逐 worker 检查顺序与内容一致 |
| `lane-plan.json`, `shard-*.json`, `cpu_universe` event | 固定请求、U/R/D和通道选择前的独立收集证据，见[分片](sharding.zh-CN.md#证据与失败语义) |
| `results.json` | 每个 node 的 passed/skipped/failed/xfail/xpass/incomplete 及 phase 时间 |
| `junit.xml`, `pytest.log` | 独立核对的 JUnit 数量、标识与时间，以及标准 durations 输出 |
| `summary.json`, `summary.md` | 完整性检查、进程启动/收集、file/fixture 时间贡献、结果数量 |

`full_cpu_baseline` 同时要求默认完整 CPU 集合及成功观测。runner/session/JUnit 缺失、worker
收集不一致、phase 缺失/重复、JUnit 数量/标识/时间不一致或运行中源码变化，都会使其无效。
失败和取消不会变成成功。收集时的 module skip/error 可能没有对应的选中 nodeid，因此另存
事件。

UTC 时间戳用于关联 Actions。命令、锁等待、启动、收集和后处理区间使用同一主机的
`time.perf_counter`，使经过时间不受主机时钟校正影响。后处理从最后一个 worker phase
报告开始。另存 UTC 差值与单调时钟经过时间的差异。缺少单调时钟观测时，相应区间保持
未知且不能构成成功 baseline，不用 UTC 相减作为回退。

phase 时间是**各 worker 的经过时间之和**，不是 CPU 时间或 job 墙钟时间。启动区间为
Python/spin/pytest 到 plugin import；worker 收集为 session 开始到收集结束，同时保留原始
时间戳及 controller 接收最后一个 worker 清单的时刻。后处理为最后一个 phase 到命令退出。
fixture 统计依赖解析后实际执行的 setup 次数、时间和错误；finalizer 仍在测试 teardown 中。
fixture setup 包含在测试 setup 内，不可相加。共享 fixture 归属其 scope node，不重复分摊
给每个 consumer。

RSS 为 pytest controller/worker 中**单进程的最大 high-water RSS**，单位 Linux KiB；不含
测试子进程，也不是进程树的同时峰值或总内存需求。记录事件序列化/flush 和报告生成时间，
其中写入时间仅是观测开销的下限。整体开销应以相同锁和环境交替运行小测试集的开/关观测
版本，保存全部命令、样本及失败。不要从启动主导的小样本推断全套测试的开销。

## Actions attempt

CI 在依赖准备前记录 runner。summary、REST snapshot 和 artifact 步骤使用 `always()`，在
准备或测试失败后仍尝试保存。强制取消或 runner 丢失可能阻止上传；REST 侧仍记录取消或
不完整的 attempt。artifact 名称包含 run 和 attempt。cache hit 来自 `setup-uv` 输出；
无法获得时明确标记 unknown，不假定 warm cache。

job 内 REST snapshot 是暂定值，其自身 job、上传和 post-cache 尚未结束。**job 完成后**
重新获取，才能确定所有 runner 和后处理时间：

```bash
.venv/bin/python -m scripts.ci_metrics fetch-actions --repo Motoki0705/tennis-lab \
  --run-id 39767861734 --output /absolute/new/actions
.venv/bin/python -m scripts.ci_metrics actions --input /absolute/new/actions/actions-raw.json \
  --output /absolute/new/actions
```

`fetch-actions` 使用只读 `gh api` 获取所有 attempt 及各自 job 的所有分页；`--attempt`
指定一次暂定 snapshot。获取成功不代表测量完整，须检查 `complete_measurement` 和
`successful_observed_checks`。`actions` 遇到缺失/不完整数据返回 2。输入是含 attempt
原始 REST 字段的 `{run, jobs}` 数组，可附带 `requested_at`、`cache_condition` 和来源。
重复 attempt 被拒绝；缺少较早 attempt 时列出缺项，并将累计 runner 时间设为未知。

queue 分别记录 attempt 起点到首个 job，以及每个 job 创建到启动。attempt 1 起点为
`created_at`；重试通常无法取得请求时刻，因此 `run_started_at` **明确标为代理值**。
仅在独立观测到真实请求时刻时提供 `requested_at`。check 完成时间是起点到所有指定
check 完成（默认 `--check Python`，可重复）；它与包含失败/取消 attempt 及其他并行 job
的**所有 job runner 时间之和**分开。`Python` 也是[仓库规则](../../.agents/README.md#github運用ルール初版)规定的merge必需check。

step 分类区分 checkout、uv/cache/install、apt、lint、pytest、post 和其他工作，保留原始
时间戳及结论。时间缺失产生带原因的未知总量，不会作为零耗时成功；skip 保持为 skip。

中位数及 p95 只使用成功且测量完整的 attempt，同时记录 n 和 nearest-rank 公式
`sorted[ceil(0.95*n)-1]`；n<20 的 p95 仅供参考。结果数量及 runner 累计仍包含失败/取消。
不同 revision、runner、cache 或 inventory 的观测只能描述现状，不能证明前后改善；后续
成功不会悄悄取代先前失败。

## 比较条件

比较 job 构成时固定源码、inventory、runner image/CPU/RAM、依赖、worker/thread、cache
和竞争负载。合并测试会改变 inventory，须保留删除 case 到保留的缺陷/边界契约的对应，
以及初始化、fixture/CLI 次数与 setup/call/teardown 时间。时间贡献可能重叠，不能直接当成
job 墙钟缩短量。保留上述完整 CPU 执行入口。候选决策和 baseline 证据见
[baseline.zh-CN.md](baseline.zh-CN.md)。

API 依据：[pytest hooks](https://docs.pytest.org/en/stable/reference/reference.html#hooks)、
[Actions attempt API](https://docs.github.com/en/rest/actions/workflow-runs#get-a-workflow-run-attempt)、
[`setup-uv` outputs](https://github.com/astral-sh/setup-uv#outputs)。

使用[`profile`命令](sharding.zh-CN.md#重新生成实测profile)从完整Actions成果物重新生成权重。保留失败用例，拒绝不完整attempt。
