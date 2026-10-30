# Agent 运维 e2e 的准备成本

[English](agents-e2e-cost.md) · [日本語](agents-e2e-cost.ja.md)

[#1524](https://github.com/Motoki0705/tennis-lab/issues/1524) 保留消息、父会话和开发测试的全部用例，减少准备工作。
每个测试的 git repository/worktree、fake GitHub、registry 和 SQLite 可写状态保持独立。
不改变生产代码、CPU 选择条件或 shard profile。

| 原用例 / 准备工作 | 保留位置及检测的缺陷 | 保留的特有交互 |
|---|---|---|
| 每个消息输入、SQL 约束、reaction 和重试用例都创建已确认的 Herdr 显示所有者 | 相同 nodeid/assert 使用 `message_world`：保留真实 SQLite transaction、registry/通知锁和 fake notifier dispatch，不配置显示所有者 | prose/container/fence × post/reply × 外部 mention 的完整组合，拒绝 self/retired/unassigned，原子性、冻结收件人、broadcast 意图、投递 key、竞争、read/ack/reaction 状态 |
| broadcast、名册及代表性的 post/reply/read/ack/rename 用例 | 原 `world` 保留已确认正在运行的 Herdr 父会话 | 按运行状态筛选 all、投递与真实显示路径、未读数、rename 后的身份。container 矩阵还确认第4个父会话运行，使意外 all 展开产生额外收件人 |
| 下游 board 使用者 | 导出的 `world` 仍配置3个已确认的父会话 | `test_board_doorbell.py` / `test_board_presence.py` / `test_board_web.py` 的显示、socket 和 HTTP 场景 |
| 未指示/无效 review、中断/并发重发、父会话停止、准备/启动前失败恢复 | 相同用例使用 `single_review_env`：1个真实 worktree、公开 prepare/request 和 reviewer subprocess | 全部无效权限/输入值、零启动、已消费 run 的持久性、未消费准备的重试、并发时仅调用1次 |
| 更新/不完整 review 对象以及 agent/parse/partial/发布/head 竞争失败 | 相同用例使用含2个 PR 的 `pair_review_env` | 不同 head/base、删除/添加对象、部分 coverage、PR 投影发布失败和失败后的重试 |
| 指示前 head 变更、收集 snapshot 期间的变更、peer/base 证据及跟随 main | 相同测试准备2个 PR | 真实 git snapshot 和 head 竞争、已 review peer 与剩余 PR、未经 review 的历史、resolution 路径 |
| 大 diff 先 prepare/request，扩展后再次 prepare/request | `test_large_diff_is_available_in_full_without_extra_review` 在唯一公开请求之前扩展 | 完整15,000行 diff、简短 prompt、reviewer 调用1次；旧对象拒绝仍由上述2个 head/snapshot 竞争测试覆盖 |
| 取消的6种情况：3个合并子任务及取消子任务13 | 相同 `test_completed_parent_distinguishes_confirmed_cancellation_from_success` 的6种情况改为合并子任务10及取消子任务11 | 真实 review/gate merge、确认 close/结果与取消 receipt 并存；missing/body/state-reason/open-PR/worker 缺陷仍阻止完成。3 PR 顺序仍由原 merge-gate/sequential-peer 测试覆盖 |
| 撤回采纳：出典/结果设置前后反复 prepare/request | 相同2个用例在首次 review 请求之前配置出典/结果 | 父/子采纳撤回、3个独立 PR 的判断、合格兄弟 PR 的合并、review 仅1次 |
| 每个 fixture 用例重新构造已指示的 review | `ReviewSeed` 在每个 worker 内通过真实公开 CLI 首次构造1/2/3 PR 的各变种，再使用已合并的 `DirectorySeed` 复制私有文件并 repair 真实 git worktree | request 和环境也复制，git object/ref/index、GitHub/registry/review 文件不共享。新增隔离测试破坏原 request、output 和 GitHub 状态，并验证恢复的 review 仍成功 |

保留原有679个用例，并新增1个 seed 隔离用例（共680个）。不删除任何输入分类、post/reply 组合、失败或重试用例。
只有取消的6个 nodeid 去掉冗余 fixture 后缀 `-independent`，保留 `change` 的全部6个值。
共享 `review_env` / `make_review_env` / `review_setup` 的默认3 PR 场景继续供下游使用。

## 已保存 Actions 的审计

run [40986313121](https://github.com/Motoki0705/tennis-lab/actions/runs/40986313121) 的 attempt 1
共有12,782个 JUnit 用例：**12,685 passed / 96 skipped / 1 failed**。
唯一失败是中文 README overview 的数值一致性检查。
4个 shard 的 `pytest.log` / JUnit / phase events 没有其他 failure/error、重复 phase、重跑或 worker 重启。
中文技术术语统一为与英文相同的 `2D` / `3D`，不改变检查本身。

| 文件 | 用例数 | Setup秒 | Call秒 | Teardown秒 |
|---|---:|---:|---:|---:|
| Board messages | 539 | 43.415 | 545.179 | 0.394 |
| Parent session | 97 | 192.810 | 372.032 | 0.072 |
| Parent development | 43 | 61.148 | 288.234 | 0.029 |

这些是历史 runner 观测，不能与本地时间作前后比较。

## 本地测量与验证

| 指标 | 前 | 后 |
|---|---:|---:|
| agents e2e用例数 | 679 | 680 |
| `test_board_messages.py` setup s | 93.346 | 65.762 |
| `test_board_messages.py` call s | 1136.355 | 181.091 |
| `test_board_messages.py` teardown s | 0.320 | 0.234 |
| `test_parent_session.py` setup s | 298.850 | 26.398 |
| `test_parent_session.py` call s | 603.133 | 431.577 |
| `test_parent_session.py` teardown s | 0.047 | 0.042 |
| `test_parent_development.py` setup s | 22.281 | 16.474 |
| `test_parent_development.py` call s | 526.130 | 295.990 |
| `test_parent_development.py` teardown s | 0.020 | 0.019 |
| 3个负责文件的phase总秒数 | 2680.482 | 1017.587 |
| 3个文件＋README的命令经过秒数 | 2695.062 | 1024.909 |
| git commit | 388 | 203 |
| git worktree add / repair | 223 / 9 | 81 / 55 |
| parent CLI | 929 | 549 |
| review / gate CLI | 102 | 103 |
| board CLI | 196 | 196 |
| 实际fake GitHub启动 | 31986 | 17065 |
| fake Herdr journal行数 | 20025 | 2400 |

3个负责文件的phase总时间观测减少 **62.04%**。 达到本地phase总时间减少40%的目标。

在前后两组1 worker版本中临时注入3种缺陷。代码内mention成为收件人时10/10用例失败，接受draft时1/1失败，跳过取消验证时2/2失败。全部run为exit 1、JUnit error 0且注入函数执行计数为正。早期取消injector在到达缺陷前timeout，排除出检测证据，但JSON保留该记录。

当前正常control为 **前723 passed / 后724 passed**，各自包含 **README全部44例**。后组新增1例通过真实review调用验证seed隔离。先行的seed验证17例及单独隔离验证1例也成功。`.venv/bin/python -m spin lint`、3个变更文件的mypy和Ruff format检查也成功。

## 复现与限制

向[测量 runner](README.zh-CN.md) 传入 `--workers 1` 和3个e2e文件及README检查：

```bash
.venv/bin/python -m scripts.ci_metrics run --output /absolute/new/observation --workers 1 -- \
  tests/e2e/agents_ops/test_board_messages.py \
  tests/e2e/agents_ops/test_parent_session.py \
  tests/e2e/agents_ops/test_parent_development.py \
  tests/e2e/development/test_readme_layout.py
```

operation observer 在 seed 外的专用 journal 统计实际 fake GitHub 启动，不重复计入复制的准备日志。它统计由 pytest 直接启动的 git（不含 CLI 后代进程内的 git）、parent/board/review-gate CLI
以及结束时 fake GitHub/Herdr journal。它原样调用所有 subprocess。
[机器可读证据](agents-e2e-evidence.json) 保留 observer 与临时缺陷注入源码、精确 selector、phase 时间、inventory 及全部失败/中断 attempt。

固定1 worker、native thread 1、隐藏 CUDA 和锁定的 CPU 环境，其他会话共享 WSL 主机。
每组比较是 n=1 观测，cache 条件未知；不能证明 Actions 墙钟时间缩短或全 CPU 的中位数/p95。
fixture 时间已经包含在 setup 内，phase 总和表示测试经过时间而非 CPU 时间。
集成后由父会话刷新 profile 并验证完整 CPU inventory 与 CI 性能；本任务不更改 profile/workflow。

前后均使用包含 #1527 的最新 main。每个 worker/shard 仍需首次准备 seed，不能用1 worker 的 cache hit 率推断10 shard 的 CI 收益。早期2 worker 观测保留独立来源，不混入本次比较。

两组串行控制均包含 README 翻译修复，“前后”指 e2e 准备优化。本地仅运行负责的3个文件及 README 检查。
