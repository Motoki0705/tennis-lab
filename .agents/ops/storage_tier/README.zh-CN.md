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
| worktree `outputs/` | 链接到 D:`<main>/.worktree-outputs/<name>/` | `.worktree-outputs/<name>/...` | task/purpose/experiment/run-id |
| 指向主 outputs 的既有 alias | 链接到主 `outputs/` | 主 `outputs/...` | 不建立独立单位 |
| 主目录 `ckpt/` | 始终完整取回 | 已校验镜像 | 永不清理 |
| worktree `data/`、`ckpt/`、`.venv` | 指向主 checkout 的链接 | 不建立独立副本 | 不单独扫描 |

[config.toml](config.toml) 声明数据集 **container**（其直接子目录为版本）与例外的**单位根目录**。
重叠声明被拒绝；新增任务的命名空间必须先声明。素材与边界模糊的旧目录列为 `unmanaged`，不递归猜测。
输出依据标准四层 run 路径及已知 purpose 发现。experiment 直接包含文件时视为旧 run，不将其子目录当作 run。
持久目录 `.worktree-outputs/<name>/` 在 Git 注册和 worktree 删除后仍可发现。同名 worktree 复用同一持久目录；
需要独立命名空间时使用新名字。outputs 链接本身是引用，不是独立单位。scan 仅列出引用路径，不读取链接目标，
因为仅 stat 穿过旧 symlink 也会改变其 atime。storage 命令使用主 outputs 或 `.worktree-outputs` 的规范路径。
旧 `.claude/worktrees/<name>/outputs/` 实际目录作为未管理的迁移源，在 `legacy_outputs` 中列出新目标，先迁移再 index。
拒绝管理根路径中的符号链接、嵌套目录挂载、特殊文件与单位内所有符号链接。
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

`--repo`、`--config` 放在子命令前。repo 必须与 launcher 共用 Git common directory；主目录从 Git 推导。
显式参数可以指向已发现单位的内部路径。无参数时 mirror/verify/evict/cycle 处理 D: 单位；index/hydrate 必须显式指定单位。
实机探测也应明确指定单位。`scan` 不读取 payload，并在 `unmanaged_data_sizes_E` 中按未分类 E:`data/<顶层>`
报告 regular file 逻辑字节数、包含目录和 symlink 的分配量。类似 du，在每个顶层内按 inode 去重，不跟随链接目标或嵌套挂载。
目录用 O_NOATIME 读取，计量失败明确报错，不视为零字节；这些资产仍不清理。`index --dry-run` 仅对所选 E: 单位计算哈希，
报告字节数、文件数、tree digest 与 atime 保持结果。dry-run 不创建锁、manifest、目录、审计或副本。
退出码为 0=成功或保留、1=安全或校验失败、2=参数语法、75=没有 launcher。E: 未挂载时 dry-run 也失败。

迁移后执行 `activate`，建立本地 `.storage-tier/active.json`。未显式选择单位的实际日常 `cycle` 必须先激活。
公共 `PathResolver` 与非 Hydra 输入边界会对
仅存于 E: 的 DATA/CHECKPOINT/ARTIFACT 给出明确错误和 hydrate 命令。只用 stat 诊断 E: 是否存在，不读取或自动取回 payload。
未挂载会单独报错。repo 外的自定义根和绕过公共边界的 consumer 继续使用原有 missing-file 错误。
新 OUTPUT 路径仍可创建；不逐个修改训练入口或 queue 命令。

## 校验与排除条件

`mountpoint`、`findmnt` 必须确认两个不同的 ext4 与指定 atime 选项。E: 缺失绝不解释为空镜像。
数据、目录、manifest 都用 `O_NOATIME` 读取；失败时不退回普通 open。虚拟 `/proc` 根目录和 kernel 生成的 metadata
不包含数据集访问历史，使用普通读取。注入的真实测试目录仍用 O_NOATIME。

镜像与认证前按 device/inode 检查 fd、cwd、mmap，覆盖单位外的硬链接别名。仅检查 `/proc/<pid>/status` 中
有效 UID 与执行用户有效 UID 相同的进程。root、CI 等其他 UID 不在范围内；JSON 和审计的 `process_scope` 按 UID 记录排除数。
同 UID 仅在具备独立证据时排除：四个 UID 均等于非 root 调用用户，fd/maps/cwd 为 root:root 所有，
三项读取都返回 EACCES/EPERM，且前后 PID 启动时间、UID、Name、所有者均不变。
显式 `Dumpable: 1` 否定排除；`CoreDumping: 0` 不作为依据。所有者规则见
[proc_pid(5)](https://man7.org/linux/man-pages/man5/proc_pid.5.html)。其他所有者/namespace 映射无法证明时仍失败。
在 `process_scope.excluded_nondumpable` 记录数量、可读 comm（status Name）、PID 和证据。
Z/X 视为已退出并单独计数，不忽略仍存活 PID 的 metadata 缺失。
不检查同 UID 的非 dumpable 进程，例如 Codex CLI 本体、sshd、gpg-agent；读取数据的训练和脚本必须以 dumpable 进程运行。
UID 无法识别、意外 I/O 错误、部分拒绝，以及 dumpable 但不可读的进程仍明确失败。
例如带 capability 的 systemd，即使 fd 目标/maps/cwd 拒绝读取，普通所有者也不能证明非 dumpable。
其他用户的 consumer 不受保护，应让 storage 与数据 consumer
以同一用户运行。完全不读取 training queue；等待或计划任务的输入也可能在 14 天后删除并需要手动 hydrate。
不设置 mtime/ctime 静止时间，仅按 atime 和打开的引用判断；哈希与发布前后 snapshot 继续检查内容稳定性。
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
各审计事件记录其他 UID 的排除范围。失败仅写入 audit 和 service journal，不发送通知。
只读 `index --dry-run` 报告进程检查错误，但不认证 manifest、不授权删除，仍可读取明确指定的小单位。

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

先确认主目录 data/outputs/ckpt 为 D: 实际目录，创建 `<main>/.worktree-outputs/<name>/`、指向它的 outputs 链接与共享输入、venv 链接。
持久根与 data 都必须位于 D:，保证硬链接可用。保留正确的既有持久链接或主 outputs alias；其他既有 outputs 被拒绝，不自动切换。
已有 symlink 时，初始化 dry-run 在读取前拒绝目标验证，避免改变其 atime。
用实际 prepare-worktree 验证已初始化的 worktree；正确的链接目标保持不变。

现有 [336 小时 worktree 期限删除](../worktree_cleanup/README.md) 仅删除 worktree 内的链接，不跟随目标。
worktree 消失后持久 outputs 仍保留，仅由 storage_tier 的校验删除管理，不改变期限删除规则。

完成迁移并确认同 UID 进程可检查后再安装启用：

```bash
spin ops install storage-tier --dry-run
spin ops install storage-tier
spin ops on storage-tier
spin ops status storage-tier
spin ops logs storage-tier --lines 100
spin ops off storage-tier
```

user timer 每天 05:30 Asia/Tokyo 执行，jitter 最多 15 分钟。失败仅写入 journal 与 `audit.jsonl`，
没有通知 helper 或 `OnFailure`。启用前检查执行用户的进程；其他 UID 按合意排除。
不建立对 system mount unit 的顺序依赖，每次直接检查挂载。install 只链接 unit，启用是独立部署操作。

## 迁移与验证

1. 协调暂停 producer、queue 新任务与持有旧路径的 consumer。启用前确认自身 UID 的 dumpable consumer 可检查，
   并核对非 dumpable 的排除证据与记录。带 capability 的 systemd 等无法证明的读取失败仍为错误；
   不能忽略，也不隐式改变权限或停止其他 session。
2. 清点 E:、活动与等待任务引用、配置间接引用、硬链接别名、E: 绝对路径和已有 D: 副本。
   只取回完整 `ckpt/` 与确定需要的版本、run 单位。
3. 将主 outputs alias 归为主 outputs 的引用。独立的旧 E:`.claude/worktrees/<name>/outputs` 在 consumer 暂停时
   rename 到 E:`.worktree-outputs/<name>`，对应关系见 scan 的 `legacy_outputs`。目标已存在时拒绝覆盖、合并，先显式比较。
   旧 manifest 的路径/ctime 不能复用，需按新规范路径重新 index。rename 是第二阶段的修改操作。
   对明确指定且稳定的 E: 单位逐个 `index`。若顺序速度为 80–150MiB/s，432GiB 单 pass 约 49–92 分钟；
   小文件 seek 可能更慢。这是规划假设，并非全盘实测。index 读 E: 一次；hydrate 还需 E: 哈希、复制与 D: 验证。
   不从小型热缓存 probe 推导 HDD 全盘速度。
4. 保存旧 symlink entry，并 rename **链接自身**，不移动 E: payload。为 D: `data/`、`outputs/` 创建空实际目录；
   `ckpt/` 留空缺，由整单位 hydrate 发布。独立 worktree outputs 链接到 D: 持久根，主 alias 链接到 D: 主 outputs。
   按规范路径取回选择的单位、验证内容和代表性硬链接后再 activate 输入诊断。已有 D: 副本需显式比较，hydrate 不做合并。
5. 由对应 session 更新活动、等待命令。先做只读 cycle，再以可丢弃单位执行 mirror→evict→hydrate。
   检查 `statvfs` 空闲量、审计、consumer 读取、同 UID 检查和失败 journal/audit，之后才启用 timer。
   queue 记录和其他 UID 不保护单位。外部硬链接可能阻止容量回收。
   rollback 仅在 consumer 暂停时恢复旧链接，并独立保留 D: 新写入。根 VHDX 迁移、压缩不在范围内。

`tests/integration/agents_ops/test_storage_tier.py` 使用临时真实文件与模拟 mount/proc，并覆盖真实 fd 和硬链接。
CLI/worktree 由 `tests/e2e/agents_ops/test_storage_tier_cli.py` 验证；扩展既有 ops-control 安装测试与 configuration path 错误测试。
