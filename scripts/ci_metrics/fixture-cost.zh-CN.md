# 减少 merge fixture 准备

[English](fixture-cost.md) · [日本語](fixture-cost.ja.md)

[#1402](https://github.com/Motoki0705/tennis-lab/issues/1402) 对单 PR 判定只准备一个 PR。
独立/依赖顺序、部分判定、base 前进、不确定 merge、修复发布、head 竞争及新 PR 交互仍使用
3个 PR 和真实 git worktree。可变 git/GitHub/review 状态仍按用例隔离。

| 原准备 | 保留的契约 |
|---|---|
| 拒绝、dry-run、单目标也创建3 PR | `pr_count` 和 `make_gate_env` 选择必要场景；保留人工阻止、当前review/head/diff、fork/draft/conflict、未启动agent |
| 先准备review，改变依赖/branch/path后重新准备 | `before_review` 在首次公开请求前固定，保留真实CLI及git snapshot |
| 每个auto-merge异常值均创建git/合意/3 PR/review | 4个policy分类及1个真实CLI检查，验证I/O与私有state创建前拒绝 |
| 父任务输入错误/恢复一律准备3 PR | 缺失目标交互保留3，其余独立错误/恢复用1；保留发布、identity与恢复 |

修改文件为 `tests/e2e/agents_ops/` 下的 `test_merge_gate.py`、`test_parent_session.py`。
不恢复main已废弃的按修改路径追加批准；保留普通review、人工阻止与实际merge opt-in。

## 验证与样本

两文件259例全部通过。4个错误值改成4个policy＋1个CLI用例，因此总数增加1；成果是减少准备。
允许非法auto-merge值的变异被旧4/新5例检出，忽略人工阻止被旧3/新3例检出。
4次运行均为普通assert失败、退出码1，没有setup/收集错误。

| 代表样本 | 原版 | 新版 |
|---|---:|---:|
| 成功用例 | 8 | 9 |
| setup合计秒 | 78.155 | 26.636 |
| call合计秒 | 16.729 | 9.350 |
| 命令单调时钟秒 | 58.828 | 23.558 |
| gate fixture数量 | 8 | 4 |
| git commit / worktree创建 | 40 / 24 | 14 / 6 |
| 父CLI / fake GitHub调用 | 82 / 3391 | 34 / 1255 |
| review prepare / request | 9 / 9 | 4 / 4 |

每版一次本地部分观测，2 worker、原生线程各1、隐藏CUDA、相同依赖；其他验证同时运行。
不能据此声称整个CI或GitHub runner改善。多PR交互仍为3 PR。
[机器证据](fixture-evidence.json)保留版本、结果、phase、调用数和变异。
代表用例、复现selector与计数observer说明见[英文版](fixture-cost.md#reproduction)。
