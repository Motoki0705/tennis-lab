# 完整 CPU 测试分片

[English](sharding.md) · [日本語](sharding.ja.md)

CI 独立执行 lint，并默认将请求的 CPU 目标分给10个普通测试 job，每个使用2个 worker，并增加1个必需的system runtime job。手动dispatch可选 `shards=1/2/4/6/8/10` 和 `workers=2/4`，比较相同源码的调度。明确指定worker数量，不使用依赖主机的 `auto`。手动单job基准上限120分钟，普通shard仍为60分钟。固定名称
`Python` 使用 `always()`，仅在plan、lint、所有测试job和覆盖审计成功后通过，保持merge必需check。
PR选择与main/手动完整请求的正本为[通道注册表与计划](README.zh-CN.md#通道注册表与计划)。仅PR取消之前的run。

## 分配与 fixture

每个 worker 用正常 CPU marker 收集当前源码，包括由通道注册表选中的 court dataset 集成，先记录通道选择前的完整集合，再分配请求的
nodeid，包括未注册的新测试。`system_deps_nodeids` 子集同时记录到event和manifest。标记case始终进入R，普通分片只分配R中未标记case，独立job只选择全部标记case；未标记的相邻case保留文件推迟策略。`cpu-file-times.json` 保存测得的 phase 时间；按成本从大到小分配
给当前最轻的 shard，同成本按名称/index稳定排序。默认按文件保留共享准备。
仅将 function fixture 独立的重文件明确列入 `split_files`，使用各用例 `test_seconds` 分配。
添加前须检查 fixture 共享范围。若后来加入module/class fixture，收集会拒绝该分割文件。

未测文件用已测文件中位数，新分割用例用该文件平均成本估计；manifest 明确列出估计值，
不会因此省略测试。不合法 profile 明确失败。命令见[英文版](sharding.md#assignment-and-fixture-boundaries)，
不分片的完整执行见[README](README.zh-CN.md)。单个 shard 的 `full_cpu_baseline=false`；
只有汇总验证完整覆盖。通道选择在分片前执行；时间或slow marker不决定成员。

## 证据与失败语义

schema-3的 `shard-gw*.json` 保存 `system_deps_nodeids` 以及 `universe_nodeids`（U）、`requested_nodeids`（R）、
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
- marker清单与独立event及全部shard一致，普通分片不混入标记case，独立job全部case均passed（skip/xfail为0）。
- plan、lint、全部matrix和独立runtime job成功，不能把skip/cancel视为成功。

从所有manifest删除同一个case也会被独立的选择前event检出。D不计入skip或成功。
汇总报告 `scoped_cpu_success`，而 `full_cpu_success` 还要求D为空。
U是既有marker与明确排除后的CPU集合，不代表所有环境的全部测试。
历史schema-1/2观测可供profile读取但不能证明新runtime覆盖。schema-1 artifact保留 `full_nodeids` 的含义，可供profile读取；混用schema或与lane plan一起使用会失败。

准备或分配失败、缺失 artifact、失败/取消或其他 HEAD 均使 `Python` 失败。
原有合法 CPU 排除和 setup/module skip 保持可见。matrix 禁用 fail-fast。按[runtime准备契约](README.zh-CN.md#固定的系统运行环境)限制并验证下载；失败不会转换为省略。

部分重试仅在同一run、同一HEAD（`source_sha`）和同一lane plan（`plan_id`）内，
复用固定plan与每个shard的最新结果。未重试的成功shard可保留之前的attempt。
attempt可在上述范围内混用；其他run、HEAD或plan会被拒绝。

plan artifact名为 `cpu-lane-plan-RUN`，测量名为 `cpu-test-measurements-RUN-shard-N`。普通index为0至count-1，独立runtime的index等于count（默认10），不增加普通CPU数量。
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

准备与pin维护以[runtime契约](README.zh-CN.md#固定的系统运行环境)为准。

## 重新生成实测profile

在checkout中使用已认证且能读取Actions产物的 `gh` 运行 `update-profile`。
`--latest-main` 按从新到旧的顺序检查成功的main push，并报告不合格run的原因。
产物仍可用时，可用 `--run-id` 和 `--attempt` 复现指定更新。

命令固定run和HEAD，按artifact ID下载lane plan、最新coverage和所有shard，核对
汇总的下载清单，并重建worker与JUnit审计。仅接受完整成功的main push，要求延期
用例为0并通过独立runtime shard审计。部分重试可按上述attempt规则复用固定plan下
较早成功的shard。证据无效或缺失时保留原输出profile，验证通过后原子替换。

使用测量来源run的调度profile作为比较基准，可复现本次更新。

```bash
profile_work=$(mktemp -d)
git show 8d629a35289f475e514291167435c333876afdd2:scripts/ci_metrics/cpu-file-times.json > "$profile_work/baseline.json"
.venv/bin/python -m scripts.ci_metrics update-profile --run-id 45955929416 --attempt 1 \
  --baseline "$profile_work/baseline.json" --output "$profile_work/cpu-file-times.json" \
  --report "$profile_work/comparison.json" \
  --split-file tests/e2e/agents_ops/test_board_messages.py \
  --split-file tests/e2e/agents_ops/test_parent_session.py \
  --split-file tests/integration/agents_ops/test_storage_tier_crash_boundaries.py \
  --split-file tests/integration/agents_ops/test_storage_tier_recovery_boundaries.py
```

选择最新run时，将 `--run-id … --attempt …` 换成 `--latest-main`。默认比较基准是
checkout中的 `cpu-file-times.json`。省略拆分选项会沿用测量来源的拆分声明。检查fixture
后，重复指定 `--split-file` 可替换列表。若某个声明的文件没有实测用例，需要显式更新列表。
JSON报告包含来源、固定的复现命令、文件差异，以及完整的分组分配和负载。若未测文件或
用例明显增多，或实测负载持续失衡，比较候选并通过常规评审PR发布。命令只写本地文件。

对于历史记录或失败诊断，仍可用 `profile --root artifacts
--output diagnostic.json` 从已下载的shard目录重建权重。完整记录的失败保存在
`source.counts` 和 `unsuccessful_cases` 中。混合attempt时使用最高shard attempt，
有延期用例时拒绝生成。

### 当前profile的离线比较

旧profile来自[run 41240159671](https://github.com/Motoki0705/tennis-lab/actions/runs/41240159671)的
883个文件和14,561个用例。更新采用[run 45955929416](https://github.com/Motoki0705/tennis-lab/actions/runs/45955929416)，
commit为 `8d629a35289f475e514291167435c333876afdd2`。lane plan、11份测量和成功coverage
的run与HEAD一致，所有attempt均为1。1,160个文件共26,189例，26,070例通过、119例合法跳过、
失败与延期均为0。独立job的17例全部通过。新增282个文件权重，更新878个，移出5个没有实测
用例的旧条目。准确的artifact ID及inventory哈希记录在profile的 `source` 中。

A使用旧权重，B用新权重并在实测inventory上沿用原拆分策略，C进一步拆分停止和恢复边界的
两个文件。旧声明中的 `test_merge_gate.py` 提供辅助函数，收集用例为0。B有两个有效拆分
文件，C有四个。A为普通shard的281个文件和28个拆分用例使用未测估值。B和C均无未测普通
文件或拆分用例。

下表为所有worker的setup、call和teardown累计秒数。A实测为来源run按A分配的实测负载。
B/C用同一批观测值估计重新分配的负载。独立job的18.578 phase秒不计入普通shard合计。

| Shard | A估计 | A实测 | B估计 | C估计 |
|---|---:|---:|---:|---:|
| 0 | 610.579 | 725.360 | 1809.091 | 1360.607 |
| 1 | 610.580 | 1322.624 | 1310.776 | 1360.609 |
| 2 | 610.579 | 611.440 | 1310.776 | 1360.608 |
| 3 | 610.579 | 1304.961 | 1310.776 | 1360.609 |
| 4 | 610.579 | 1185.038 | 1310.777 | 1360.608 |
| 5 | 610.578 | 1397.866 | 1310.777 | 1360.607 |
| 6 | 610.579 | 1650.712 | 1310.776 | 1360.608 |
| 7 | 610.580 | 844.635 | 1310.778 | 1360.609 |
| 8 | 610.578 | 3311.545 | 1310.777 | 1360.608 |
| 9 | 610.578 | 1235.024 | 1310.778 | 1360.609 |
| 最小 | 610.578 | 611.440 | 1310.776 | 1360.607 |
| 最大 | 610.580 | 3311.545 | 1809.091 | 1360.609 |
| 平均 | 610.579 | 1358.921 | 1360.608 | 1360.608 |
| 最大 / 平均 | 1.000 | 2.437 | 1.330 | 1.000 |

4,606个恢复边界用例耗时1809.091秒，从A的shard 8移到B的shard 0。1,649个停止边界用例
耗时533.836秒，从A的6移到B的1。C把两个文件分到全部十个普通shard。接下来最重的
整文件是 `test_parent_development.py`（423.166秒，A/B/C分别为1/2/0）、
`test_parent_completion_resume.py`（373.922秒，8/3/1）和 `test_court_dataset.py`
（355.775秒，0/4/2）。按C重新分配实测用例成本的最大累计值为1360.609秒，比A的
实测最大值低58.9%。

两个边界文件使用function scope的 `tmp_path`、`monkeypatch` 与全局的逐用例随机种子
fixture。pytest的session scope临时目录分配器为各用例提供独立根目录。`make_lab` 在
该根目录中存放repo、镜像、模拟 `/proc`、journal和锁。trace hook仅在fork的子进程内
运行，子进程会被join或kill，pytest在teardown时恢复路径改动。实际shard plugin的收集
guard接受了全部6,255个边界用例，每例恰好分配一次，未测拆分用例为0。

## 同条件Actions比较

A使用旧权重，C使用采用的新权重。完整dispatch的源码差异仅为
`cpu-file-times.json`。两个scoped PR分别在各自基准上向
`src/tasks/ball_refiner_3d/__init__.py` 加入同一条注释。
每次运行使用十个4 vCPU普通shard和一个独立runtime job，每个shard有两个
pytest worker和一个native thread。66个测试job均命中uv缓存。
托管runner的CPU型号和Ubuntu镜像版本有所不同，以下为这六次运行的观测结果。

六次首轮attempt全部成功。四次完整运行均请求26,208例，26,089例通过、
119例跳过，`full_cpu_success=true`。两次scoped PR均从26,208例中请求
相同的23,334例，23,219例通过、115例跳过、2,874例延期，
`scoped_cpu_success=true`。根据worker manifest、event和JUnit重跑覆盖审计，
核对了每次运行固定的profile和lane plan。请求与执行的nodeid完全一致，
没有缺失、重复或额外用例，同一scope内的跳过用例也一致。
独立runtime job的17例每次均通过。

下表单位为秒。Test和job分别为普通shard中的最大值，括号内为shard index。
Python从workflow创建计时至必需检查完成。整体耗时从创建计时至Actions最终更新。
Runner合计为全部15个job的执行时间之和。

| 条件 | Actions run | 最长Test (shard) | 最长job (shard) | Python | 整体耗时 | Runner合计 |
|---|---|---:|---:|---:|---:|---:|
| Full A1 | [46045996862](https://github.com/Motoki0705/tennis-lab/actions/runs/46045996862) | 1755 (8) | 1834 (8) | 1941 | 1941 | 9082 |
| Full C1 | [46050353999](https://github.com/Motoki0705/tennis-lab/actions/runs/46050353999) | 933 (5) | 1006 (5) | 1088 | 1089 | 8792 |
| Full A2 | [46052765632](https://github.com/Motoki0705/tennis-lab/actions/runs/46052765632) | 1467 (8) | 1535 (8) | 1607 | 1608 | 8502 |
| Full C2 | [46056231956](https://github.com/Motoki0705/tennis-lab/actions/runs/46056231956) | 991 (1) | 1061 (1) | 1141 | 1141 | 9476 |
| Scoped A | [46058668606](https://github.com/Motoki0705/tennis-lab/actions/runs/46058668606) | 1069 (3) | 1125 (0) | 1208 | 1209 | 6496 |
| Scoped C | [46061213080](https://github.com/Motoki0705/tennis-lab/actions/runs/46061213080) | 706 (2) | 749 (2) | 823 | 824 | 6328 |

准备时间从普通job启动计至Test步骤启动，收集时间从测试命令启动计至所有worker
完成收集，两列均为普通shard的平均值。等待时间分别为dispatch至首个job启动，
以及所有job中从创建到启动的最大值。实测phase秒数是worker的setup、call
和teardown时间之和。

| 条件 | 准备 | 启动至收集完成 | 首个job等待 | 最大job等待 | 最大实测phase | Phase最大值 / 平均值 |
|---|---:|---:|---:|---:|---:|---:|
| Full A1 | 60.5 | 109.1 | 3 | 3 | 3264.667 | 2.409 |
| Full C1 | 59.9 | 109.9 | 4 | 2 | 1552.381 | 1.178 |
| Full A2 | 60.8 | 107.4 | 3 | 3 | 2672.376 | 2.139 |
| Full C2 | 62.3 | 118.5 | 5 | 3 | 1645.171 | 1.155 |
| Scoped A | 39.4 | 110.8 | 3 | 3 | 1945.171 | 2.149 |
| Scoped C | 40.6 | 106.8 | 2 | 3 | 1117.201 | 1.280 |

完整运行每种条件各两次，最长Test的平均值从1611降到962秒（缩短40.3%），
Python检查完成时间从1774降到1114.5秒（缩短37.2%）。两组最长Test分别缩短
46.8%和32.4%，均达到30%目标。Runner总时间平均从8792增到9134秒，
每次完整运行增加342秒（3.9%）。一次scoped对照中，最长Test从1069降到706秒
（缩短34.0%），Python检查提前385秒完成，runner总时间从6496降到6328秒
（减少2.6%）。

每个shard的估计与实测phase时间、实际耗时、runner信息和审计标识见
[比较数据](sharding-comparison.json)。

## 默认配置选择

默认十个普通shard，每个两个worker，另有一个必需的独立runtime job。用于选定默认值的
测量、cache条件和样本限制见[baseline](baseline.zh-CN.md#runner配置比较)。
