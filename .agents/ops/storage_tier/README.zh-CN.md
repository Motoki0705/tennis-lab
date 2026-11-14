# 本地 D:/E: 存储分层

[English](README.md) · [日本語](README.ja.md)

本文是本地目录布局与保留规则的依据。D: 是 WSL 根 ext4（`relatime`），保存工作副本。
E: 是挂载到 `/mnt/wsl/tennis-data` 的 HDD ext4（`noatime`），保存权威镜像。
[Google Drive 策略](../../skills/tennis-drive/references/storage-policy.md)保持独立。
任务内部的相对路径遵守 [OUTPUTS.md](../../../src/tasks/OUTPUTS.md)。

## 布局与管理边界

主 checkout 为 `R` 时，`R/<relative>` 在 E: 上对应
`/mnt/wsl/tennis-data/<去掉开头斜杠的R>/<relative>`。

| 位置 | D: | E: | 清理单位 |
|---|---|---|---|
| 主目录 `data/` | 实际目录 | 相同相对路径 | 数据集版本 |
| 主目录 `outputs/` | 实际目录 | 相同相对路径 | task/purpose/experiment/run-id |
| worktree `outputs/` | D: 上的实际目录 | `.claude/worktrees/<name>/outputs/...` | task/purpose/experiment/run-id |
| 主目录 `ckpt/` | 始终完整取回 | 已校验镜像 | 永不清理 |
| worktree `data/`、`ckpt/`、`.venv` | 指向主 checkout 的链接 | 不建立独立副本 | 不单独扫描 |

[config.toml](config.toml) 声明数据集 **container**（其直接子目录为版本）与例外的**单位根目录**。
重叠声明被拒绝；新增任务的命名空间必须先声明。素材与边界模糊的旧目录列为 `unmanaged`，不递归猜测。
输出依据标准四层 run 路径及已知 purpose 发现。experiment 直接包含文件时视为旧 run，不将其子目录当作 run。
E: 上的 worktree 输出在 Git 注册删除后仍可发现。拒绝管理根路径中的符号链接、嵌套目录挂载、特殊文件与单位内所有符号链接。
Linux readlink 即使使用 O_PATH fd 仍会更新旧链接的 atime，因此必须在 readlink 前拒绝，以保持 dry-run 只读。
管理 payload 应使用实际文件或硬链接；本工具不改写已有链接。

硬链接只能用于 D: 内或 E: 内，不能跨 tier。复制保留**单位内**硬链接，单位之间、tier 之间使用独立 inode。
清理只删除目标单位的名字。审计区分逻辑大小、去重后的分配量、外部硬链接、预计可回收量与文件系统空闲差值。
其他名字仍存在时，磁盘块可能完全不释放。hydrate 创建新的 D: inode，不恢复跨单位硬链接。

## CLI

在包含所需实现的 checkout 中运行，dry-run 也必须通过有限
[checkout launcher](../lib/README.md#既存cliと新規jobの接続)。

```bash
export PYTHONPATH="$PWD/.agents/ops"
export PYTHONDONTWRITEBYTECODE=1
S=(.venv/bin/python .agents/ops/checkout_job.py --repo "$PWD" -- .venv/bin/python -m storage_tier)
"${S[@]}" scan
"${S[@]}" mirror data/ball_detection/example-v1 --dry-run
"${S[@]}" mirror data/ball_detection/example-v1
"${S[@]}" verify data/ball_detection/example-v1 --dry-run
"${S[@]}" index data/ball_detection/example-v1 --dry-run
"${S[@]}" hydrate data/ball_detection/example-v1
"${S[@]}" evict data/ball_detection/example-v1 --dry-run
"${S[@]}" cycle --dry-run
```

`--repo`、`--config`、`--queue-dir` 放在子命令前。repo 必须与 launcher 共用 Git common directory；主目录从 Git 推导。
显式参数可以指向已发现单位的内部路径。无参数时 mirror/verify/evict/cycle 处理 D: 单位；index/hydrate 必须显式指定单位。
实机探测也应明确指定单位。`scan` 不读取 payload；`index --dry-run` 仅对所选 E: 单位计算哈希，
报告字节数、文件数、tree digest 与 atime 保持结果。dry-run 不创建锁、manifest、目录、审计或副本。
退出码为 0=成功或保留、1=安全或校验失败、2=参数语法、75=没有 launcher。E: 未挂载时 dry-run 也失败。

迁移后执行 `activate`，建立本地 `.storage-tier/active.json`。未显式选择单位的实际日常 `cycle` 必须先激活。
公共 `PathResolver` 与非 Hydra 输入边界会对
仅存于 E: 的 DATA/CHECKPOINT/ARTIFACT 给出明确错误和 hydrate 命令。只用 stat 诊断 E: 是否存在，不读取或自动取回 payload。
未挂载会单独报错。repo 外的自定义根和绕过公共边界的 consumer 继续使用原有 missing-file 错误。
新 OUTPUT 路径仍可创建；不逐个修改训练入口或 queue 命令。

## 校验与排除条件

`mountpoint`、`findmnt` 必须确认两个不同的 ext4 与指定 atime 选项。E: 缺失绝不解释为空镜像。
数据、目录、manifest 都用 `O_NOATIME` 读取；失败时不退回普通 open。root 所有的虚拟 `/proc` 根目录
使用普通列举，它不包含数据集访问历史。

镜像与认证前按 device/inode 检查 fd、cwd、mmap，覆盖单位外的硬链接别名。无法读取进程或 queue 目录则失败。
存在任何 `jobs/*.job` 或 `running/*.job` 时保留**所有单位**，因为 Python/Hydra/shell 能通过配置间接得到路径，
命令字符串不能证明其他单位未被引用。旧 running 记录由 queue 流程处理，本工具不改写。
近期写入需经过配置的静止时间（默认一小时）；哈希与发布前后的 snapshot 也检测并发变更。
发布后 metadata 必须与已哈希的 staging tree 一致，才能将其 digest 写入 manifest。

镜像先计算 D: 全量 SHA-256，复制到新的 E: staging，再计算 staging 全量 SHA-256 并比较整个 tree。
未记录的现有 E: 只有与 D: 完全一致才可采用。更新已记录镜像前必须确认旧 E: metadata 未变。
使用 Linux `RENAME_NOREPLACE` / `RENAME_EXCHANGE` 发布，manifest 以 atomic JSON 写入
E:`<mirror-repo>/.storage-tier/manifests/`，保存两边的 SHA-256、size/mtime/ctime/device/inode。

只有全部 regular file 的 atime 至少 14 天未更新才可清理，空单位保留。删除前重新哈希 D:，
确认 E: 全部 entry 及记录的 metadata 不变，将 D: 改为同父目录的隔离名，再检查后才删除。
`ckpt/` 永不删除。已过期且有证据的单位在日常镜像更新证据**之前**校验，不能用重新镜像掩盖不一致。
删除时固定已校验 root 的 FD，核对 device/inode 后按 FD 相对删除。隔离目录被替换时，不删除或将其作为
原单位恢复，而保留身份错误供人工检查。即使 device 相同，也按目录 mount ID 拒绝嵌套 bind mount。
hydrate 拒绝已存在的 D: 目标，完成 E: 哈希、复制、D: 哈希后才发布，并将文件 atime 更新为当前时间。

## 审计与恢复

D:`<repo>/.storage-tier/audit.jsonl` 保存校验、取回、失败、删除确认与完成记录，包含 UTC、单位、大小与校验方式。
修改操作按 repo 排他；dry-run 不创建锁。持久删除 intent 位于同目录的 `transactions/`。

```bash
"${S[@]}" recover
```

恢复将隔离单位无覆盖地移回，**绝不恢复执行删除**。原位置被占用或身份改变会明确失败。
部分删除的 tree 会恢复并标记可能不完整，应检查后使用已验证的 E: 副本修复。若崩溃前已删除，则记录事实，
不宣称已实测容量收益。未完成 transaction 阻止后续操作。若 E: exchange 后 manifest 发布中断，旧镜像保留在 staging 名下。
检查两个 tree；只有 D:/E: 全量一致时才显式 `verify`，在证据落盘前保留旧副本。不静默清理 staging 残留。

## worktree 与常驻服务

完成迁移并执行 `git worktree add` 后运行：

```bash
"${S[@]}" prepare-worktree /absolute/main/.claude/worktrees/new-worker --dry-run
"${S[@]}" prepare-worktree /absolute/main/.claude/worktrees/new-worker
```

先确认主目录 data/outputs/ckpt 为 D: 实际目录，并完整验证计划，才创建 outputs 和共享输入、venv 链接。
拒绝既有 outputs 符号链接，不自动切换现有路径。

现有 [336 小时 worktree 期限删除](../worktree_cleanup/README.md) 会独立删除整个 worktree，包括真实 D: outputs。
**在此布局同时启用两种删除策略前，必须先达成并实现生命周期协调。** 本工具不擅自弱化该人类已同意的策略或停用 timer。

迁移、进程检查权限与生命周期冲突解决后再安装启用：

```bash
spin ops install storage-tier --dry-run
spin ops install storage-tier
spin ops on storage-tier
spin ops status storage-tier
spin ops logs storage-tier --lines 100
spin ops off storage-tier
```

user timer 每天 05:30 Asia/Tokyo 执行，jitter 最多 15 分钟。只有失败时通过 `OnFailure` 用 busctl 调用
user bus 的 `org.freedesktop.Notifications.Notify`；成功或仅保留时不通知。主机必须有通知服务，
配送失败会显示为 helper unit 失败。启用前应实测通知配送和完整 `/proc` 检查。
不建立对 system mount unit 的顺序依赖，每次直接检查挂载。install 只链接 unit，启用是独立部署操作。

## 迁移与验证

1. 协调暂停 producer、queue 新任务与持有旧路径的 consumer。启用删除前解决 worktree 期限策略冲突及 user unit
   无法读取部分 `/proc` 的问题。不隐式改变权限、停止其他 session 或安装特权 helper。
2. 清点 E:、活动与等待任务引用、配置间接引用、硬链接别名、E: 绝对路径和已有 D: 副本。
   只取回完整 `ckpt/` 与确定需要的版本、run 单位。
3. 对明确指定且稳定的 E: 单位逐个 `index`。若顺序速度为 80–150MiB/s，432GiB 单 pass 约 49–92 分钟；
   小文件 seek 可能更慢。这是规划假设，并非全盘实测。index 读 E: 一次；hydrate 还需 E: 哈希、复制与 D: 验证。
   不从小型热缓存 probe 推导 HDD 全盘速度。
4. 保存旧 symlink entry，并 rename **链接自身**，不移动 E: payload。为 D: `data/`、`outputs/` 创建空实际目录；
   `ckpt/` 留空缺，由整单位 hydrate 发布。迁移相应 worktree 的 outputs 链接并创建真实 D: 根目录。
   取回选择的单位、验证内容和代表性硬链接后再 activate 输入诊断。已有 D: 副本需显式比较，hydrate 不做合并。
5. 由对应 session 更新活动、等待命令。先做只读 cycle，再以可丢弃单位执行 mirror→evict→hydrate。
   检查 `statvfs` 空闲量、审计、consumer 读取与失败通知，之后才启用 timer。外部硬链接可能阻止容量回收。
   rollback 仅在 consumer 暂停时恢复旧链接，并独立保留 D: 新写入。根 VHDX 迁移、压缩不在范围内。

`tests/integration/agents_ops/test_storage_tier.py` 使用临时真实文件与模拟 mount/proc，并覆盖真实 fd 和硬链接。
CLI/worktree 由 `tests/e2e/agents_ops/test_storage_tier_cli.py` 验证；扩展既有 ops-control 安装测试与 configuration path 错误测试。
