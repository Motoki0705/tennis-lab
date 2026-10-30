# 重型测试的准备与评估

[English](heavy-test-cost.md) · [日本語](heavy-test-cost.ja.md)

父议题 [#1390](https://github.com/Motoki0705/tennis-lab/issues/1390) 的子议题
[#1522](https://github.com/Motoki0705/tennis-lab/issues/1522) 保留全部现有merge gate与
coordinates training用例，并增加一个seed隔离测试。
生产代码、默认值、CPU测试选择与分片时间profile均不变。

## 变更与保留的契约

merge fixture在每个pytest worker中，针对每种variant通过真实git和parent/review CLI
创建已完成评审的scenario。session内缓存保存private快照，以及与首个测试分离的Python
metadata。后续每个测试获得新的文件系统副本、独立的GitHub/registry/review记录，以及
独立的git object、ref、index和worktree。重新绑定绝对private路径后，运行真实的
`git worktree repair` 检查迁移后的仓库。若binary state包含private路径，则明确失败。
测试不会获得可写的缓存本体。每次gate操作仍启动真实merge CLI subprocess与fake GitHub
CLI；head/base/diff重新收集和merge调用均不采用mock或缓存。

训练smoke保留真实BLCS生成器、4场rally和4个camera，将模拟限制为256个物理step
（64个输出frame），最短32个frame。固定seed为992，明确检查validation/test保留多个飞行区间。短窗保留为32个frame，长窗上限为64个frame。
已有宽度16、1层的模型、batch 2、roundtrip 4次更新、resume 5次更新、Flow sampling
3 step、每2次更新validation以及最后的不完整epoch均保留。CPU trainer thread明确设为1。
保存的物理诊断使用真实非线性fit，仅在本测试的evaluator边界注入1组初始值、
0次segment单独迭代和1次joint迭代。预测、loss、梯度、optimizer、checkpoint、plot和metric
输出均未被替换。

| 原用例或交互 | 保留位置 / 检测的缺陷 / 独有交互 |
|---|---|
| 重投影与共享生成 | 相同重投影测试与roundtrip内的split检查：真实生成、拒绝覆盖、4条rally独立的train/val/test记录；3D/events/time完全一致，不同camera的可见2D重投影改变 |
| 回归roundtrip | 相同用例：真实normal fit、更新4次后的非空AdamW state、有限的加权loss、由监控指标选择的best/last checkpoint重新加载、TensorBoard events、带seed评估的完全一致与推理文件契约 |
| GAN与GAN-only roundtrip | 保留两个用例：generator权重0/0.5/1/1、discriminator更新3次、重建权重1/0.5/0/0、各项与total一致及最终GAN-only目标函数 |
| Flow roundtrip | 相同用例：真实Flow loss/backward/optimizer、3 step带seed的sampling、best/last重新加载以及预测与事件概率的精确重现 |
| 物理head roundtrip | 相同用例：4种正的物理loss与加权和、积分轨迹数组、field/surface/segment输出及推理schema、真实fit的有限残差与保存的smoke设置 |
| 回归 / Flow / GAN resume | 相同3个用例：更新2次后中断，从native checkpoint恢复到更新5次/epoch 2，最后只有1次更新的epoch、dropout、连续运行与恢复后的全部权重、Torch/Flow RNG和学习曲线完全一致 |
| 物理fit数值收敛 | 已有 `test_physics_evaluation.py` 与 `test_ball_fitting.py` 保持不变：生产多初始值protocol、模拟轨迹的精确fit、非物理残差、多个segment/surface、事件与非法输入。数值精度由这些用例检查，smoke检查lifecycle连接 |
| 人工保留、评审缺失/中断/未确认、projection/head/base/diff一致性、draft/fork/conflict/未知输入 | 保留所有原gate用例与CI状态组合；只修改private副本并调用真实merge CLI，保留各自的拒绝原因以及不启动agent/merge的断言 |
| 独立/依赖PR、部分决定、base前移、过期/失败/响应丢失merge、修复、head竞争与新增PR | 相同3 PR variant与scenario正文；保留可变的真实git历史和CLI重新收集，不合并交互用例 |
| 独立issue、非main默认branch、共享path、已完成的研究依赖、非法auto-merge与无目标 | 相同专用variant / policy / CLI用例，保持已有的最小scenario规模 |
| `test_execution_basis.py` / `test_judgement_execution.py` | 下游scenario不变，包括3 PR/shared variant与显式 `make_gate_env` 调用 |
| 新增隔离测试 | 在副本中提交、增加worktree、修改GitHub状态并删除review证据；副本的真实CLI拒绝操作，原环境dry run成功。原ref/state/result与seed全部bytes不变，git object inode不同，首个测试修改metadata也不会影响template |

## 相同本地环境下的测量

历史 [Actions run 40986313121](https://github.com/Motoki0705/tennis-lab/actions/runs/40986313121) 的主要剩余热点如下。merge gate的setup/call总计1129.879/713.591秒。训练roundtrip为95.1–96.1秒，物理head用例190.7秒，resume为0.95–1.08秒。这些观测用于确定优先顺序，不作为本地配对比较的分母。

| Actions case | Setup s | Call s |
|---|---:|---:|
| `test_all_targets_merge_in_one_run_with_verified_base_advances[True-independent]` | 11.159 | 44.416 |
| `test_all_targets_merge_in_one_run_with_verified_base_advances[False-independent]` | 11.290 | 43.712 |
| `test_new_child_pr_does_not_invalidate_reviewed_prs[multiple]` | 11.188 | 36.792 |

| 文件 | 用例数 | setup秒 | call秒 | 总秒数 | 减少比例 |
| --- | --- | --- | --- | --- | --- |
| `test_merge_gate.py` | 162 → 163 | 1804.797 → 175.366 | 1047.166 → 942.883 | 2852.110 → 1118.353 | 60.8% |
| `test_coordinates_training.py` | 9 → 9 | 15.622 → 0.785 | 1326.628 → 25.534 | 1342.258 → 26.324 | 98.0% |

完整的本地1组对比：优化前171 passed，优化后172 passed。command wall time: 2257.648 → 595.396秒。未删除原用例。

训练文件9个用例的call时间均不超过 **6.372秒**。

数值为不同worker间重叠的pytest经过秒数之和，不是CPU时间，也不是整个CI耗时的预测。
完整保留旧inventory，优化后总量还包括新增隔离测试。
使用2 workers、`--dist=worksteal`、native thread 1并隐藏CUDA。
fit时间已包含于评估/训练时间，不再加到phase时间中。

| 操作 | 优化前 | 优化后 |
| --- | --- | --- |
| test_merge_gate: `fixture_parent_env` | 157 | 158 |
| test_merge_gate: `fixture_gate_env` | 156 | 157 |
| test_merge_gate: `reviewed_gate_preparations` | 156 | 13 |
| test_merge_gate: `seed_restore` | 0 | 145 |
| test_merge_gate: `git` | 2734 | 1396 |
| test_merge_gate: `git_commit` | 601 | 219 |
| test_merge_gate: `worktree_add` | 268 | 28 |
| test_merge_gate: `parent_cli` | 1400 | 142 |
| test_merge_gate: `review_cli` | 168 | 25 |
| test_merge_gate: `gate_cli` | 169 | 171 |
| test_coordinates_training: `model_init` | 14 | 14 |
| test_coordinates_training: `optimizer_step` | 64 | 64 |
| test_coordinates_training: `torch_save` | 56 | 56 |
| test_coordinates_training: `evaluate` | 62 | 62 |
| test_coordinates_training: `physics_fit` | 24 | 24 |
| fake GitHub CLI | 64554 | 29446 |

真实非线性fit：1291.517 → 13.374秒。`git` 统计pytest helper调用次数，不包含CLI subprocess内部的git。`model_init` 统计训练Lightning module的构造次数，不包含checkpoint/推理model的创建次数。

复制的历史fake GitHub调用日志行不计入实际执行次数。
基础 `parent_env` 仍为每个测试初始化独立可写环境。只摊薄已完成评审的准备工作，
模型初始化、optimizer更新、保存评估与checkpoint I/O仍真实运行。

## 缺陷注入与验证

| Fault | 优化前 | 优化后 |
| --- | --- | --- |
| `human_hold` | 3 AssertionError | 3 AssertionError |
| `projection_integrity` | 2 AssertionError | 2 AssertionError |
| `resume_rng` | 3 AssertionError | 3 AssertionError |
| `reconstruction_schedule` | 1 AssertionError | 1 AssertionError |

相关command共 **187个用例pass**：未修改的execution-basis/judgement scenario
**71 + 12个**、全部ball refiner unit **97个**、共享数值fit **5个**、分配/新用例检查
**2个**。生产多初始值protocol及收敛检查得到保留。另用 `--dist=loadfile -n 2`
复现两个timeout scenario、head竞争以及有/无修复的base前移，**5个用例pass**，
覆盖从cache恢复副本的路径。

```bash
.venv/bin/python -m pytest -n 2 --dist=worksteal -q --no-cov \
  tests/e2e/agents_ops/test_execution_basis.py \
  tests/e2e/agents_ops/test_judgement_execution.py \
  tests/unit/tasks/ball_refiner_3d \
  tests/unit/utils/physics/test_ball_fitting.py \
  tests/e2e/development/test_ci_shards.py::test_assignment_uses_cost_and_preserves_files_including_new_tests \
  tests/e2e/development/test_ci_shards.py::test_only_explicit_function_fixture_files_split_by_measured_case_cost
.venv/bin/python -m spin lint
.venv/bin/python -m mypy --follow-imports=skip \
  tests/e2e/agents_ops/test_merge_gate.py \
  tests/integration/tasks/ball_refiner_3d/test_coordinates_training.py \
  tests/support/agents_ops/directory_seed.py
```

lint和3个修改的Python文件mypy均pass。4种缺陷的command在优化前后均以exit 1结束，
检测到相同的 **3 / 2 / 3 / 1个AssertionError**，没有setup/collection error。
依次绕过人工保留、绕过projection一致性检查、重置resume RNG、强制重建权重保持为1。


全CPU收集/分配为 **12783个唯一用例**，四分片 **3389 / 3246 / 3186 / 2962个**，缺失/重复 **0/0**。新用例通过已有未测量case估值获得分配；实际收集也检查fixture scope。这是收集审计，不代表全suite执行成功。首次收集因缺少jsonschema/psutil/requests产生6个error，随后在独立收集overlay安装lock版本，不改变配对测量环境。

临时pytest plugin仅在内存中注入缺陷，不编辑生产source。gate plugin在真实CLI subprocess中
编译修改后的当前policy；训练plugin修改实际Lightning实现。每个缺陷使用新的pytest进程
重建全部seed，因此已准备好的state不会掩盖实现缺陷。准确的plugin、command、各用例结果、
phase、初始化/操作次数和失败尝试均保存在 [机器证据](heavy-test-evidence.json)。

## 复现与限制

在固定source上，使用相同环境和evidence JSON内的observer运行
[测量runner](README.zh-CN.md)：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  .venv/bin/python -m scripts.ci_metrics run --workers 2 --output /absolute/new/observation -- \
  tests/e2e/agents_ops/test_merge_gate.py \
  tests/integration/tasks/ball_refiner_3d/test_coordinates_training.py
```

测量操作次数时，将 `observer_source` 提取为临时 `PYTHONPATH` 上的 `heavy_observer.py`，
设置 `PYTEST_PLUGINS=heavy_observer`，并将 `HEAVY_OBSERVER_DIR` 指向新目录。
缺陷注入使用 `mutation_plugin_source`、`PYTEST_PLUGINS=heavy_mutation`、`HEAVY_MUTATION`、
`HEAVY_MUTATION_DIR` 及evidence中的selector。command结束前保持source不变。

共享本地WSL主机上n=1（AMD Ryzen 5 7430U with Radeon Graphics、Python 3.12、Torch 2.15.0+cpu）；cache状态未知。共享venv缺少spin/pytest-cov，因此成功的前后测量使用相同临时tool overlay；准确版本与主机负载保存在evidence中。环境不同于Actions，所以历史run不作为配对加速比较的分母。原trainer会将Torch intraop thread覆盖为2，优化smoke明确选择1，native library环境变量限制不变。保留最初缺少spin/pytest-cov的尝试，以及严重主机竞争期间中断的37 passed、5 failed、129 incomplete记录。另一次cProfile尝试在长时间生产物理fit中中断，不算作完成的测量。

首次优化后完整观测为170 passed及并发负载下2个CLI 90秒timeout，使用中间的segment 1次/joint 1次fit预算，保留其全部产物。成功的完整重跑没有修改gate实现，最终smoke fit为segment warmup 0次/joint 1次。缺陷注入、5个cache复现用例及数值回归测试也保留记录。已评审准备次数按fixture次数减去seed恢复次数，再加回隔离测试额外的1次恢复计算。

这不是成功的全CPU baseline，也不是4 shard的Actions对比。每个shard/worker会重建seed。
父会话需在集成后更新分片时间profile并实测Actions；本变更不修改workflow/profile。
不排除测试、失败检查或slow marker。
