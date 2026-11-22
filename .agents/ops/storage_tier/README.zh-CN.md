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
| 常驻输入 | `resident_units` 声明的实际目录 | 相同相对路径 | 永不清理 |
| 未管理的E:引用 | `unmanaged_data_links` 列出的symlink | 原始文件/目录 | 不作为单位选择 |
| 主目录 `outputs/` | 实际目录 | 相同相对路径 | task/purpose/experiment/run-id |
| worktree `outputs/` | 链接到 D:`<main>/.worktree-outputs/<name>/` | `.worktree-outputs/<name>/...` | task/purpose/experiment/run-id |
| 指向主 outputs 的既有 alias | 链接到主 `outputs/` | 主 `outputs/...` | 不建立独立单位 |
| 主目录 `ckpt/` | 实际容器，逐个取回子目录和根文件 | 相同单位镜像 | 永不清理 |
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

[config.toml](config.toml) 设置 `worktree_outputs_on_mirror = true` 时，worktree 持久输出只放在 E:。此时 D: 的 `<main>/.worktree-outputs` 必须是字面目标恰为其 E: 对应路径的 symlink，目标是挂载上的实际目录；D: 实际目录、其他目标或缺失链接都报错。worktree 的 `outputs` 链接仍指向 `<main>/.worktree-outputs/<name>`。由于 D: 与 E: 经由链接指向同一份 bytes，这些输出在两层都不作为 mirror/evict/hydrate 单元，与 D: data 之间的 hardlink 以 `EXDEV` 失败。未设置该键时要求上述 D: 布局并拒绝链接化的 root。

硬链接只能用于 D: 内或 E: 内，不能跨 tier。复制保留**单位内**硬链接，单位之间、tier 之间使用独立 inode。
清理只删除目标单位的名字。审计区分逻辑大小、去重后的分配量、外部硬链接、预计可回收量与文件系统空闲差值。
其他名字仍存在时，磁盘块可能完全不释放。hydrate 创建新的 D: inode，不恢复跨单位硬链接。

## 常驻输入与未管理的E:引用

`resident_units` 声明永久保留的D:输入：ACCAD、samples、smplh、smplx和
`data/tennis_multiview/{raw,processed}/meiji_4cam`。processed包含 `projects.json`。
它们作为 `resident` 单位复用全量SHA-256的index/mirror/verify/hydrate协议，
但evict始终保留，每日cycle也不清理。payload及祖先必须为实际目录，沿用其他单位的symlink/mount检查。
这不将tennis_multiview其余部分变为受管理数据集；该设计仍属于另一个任务。

`unmanaged_data_links` 明确列举允许的D:引用，包括其余raw/processed条目及两个旧顶层目录。
tennis_multiview/raw/processed本身在D:上为实际容器，仅列出的非驻留子项链接到E:。
也允许 `meiji_3cam.tar` 文件引用。完整清单以[config.toml](config.toml)为准。
dataset、常驻、引用边界不能重复或相互包含。省略两个新键的旧custom配置仍可使用。

已有引用必须为symlink，链接文本与 `Config.mirror / relative_path` 的**绝对路径完全一致**；
拒绝相对或其他拼写。目标必须为配置E:mount上存在的普通文件/实际目录，祖先symlink和嵌套mount也拒绝。
dataset container直属版本root及单位边界/祖先上的未声明symlink在读取target前拒绝。
错误目标、dangling link或引用位置的实际目录在布局验证和实际操作（包括无关单位的显式操作及activate）时失败。
配置布局时允许引用尚未建立。先检查E:挂载；未挂载绝不解释为空外部目录。

引用保持为 `unmanaged`，不沿链接发现单位，也不能明确作为mirror/evict/hydrate的单位。
保留E:未分类目录原有的metadata容量调查。命令输出与写操作audit包含 `unmanaged_links_D`；
引用检查失败也通过stderr保留观察记录。

共享 `PathResolver` 对解析后仍位于DATA root内的路径直接应用普通包含规则，不调用Git、不读取policy，也不进行E:例外验证。
声明名称的链接即使指向D:内部也可能通过此路径边界，但这不证明存储布局有效。
`scan`、`activate`及实际storage操作仍拒绝目标不等于精确E:对应路径的声明链接，包括D:内部目标。
只有需要root外额外权限时才读取同一checkout的 [config.toml](config.toml)，不维护第二份引用名单。
仅对标准DATA root，声明的D:链接精确指向已挂载E:对应路径时才允许。
标准root必须精确等于 `main_checkout / data`；main checkout由runtime的 `PROJECT_ROOT` 查询Git绝对common directory，
取其父目录，与storage CLI使用相同规则。仅basename为data的custom root不获得例外：内部链接保持原包含规则，
外部链接（包括E:对应路径形状）仍被拒绝。runtime worktree尚未创建共享data链接也不改变标准root。
Git repository身份仅在查询成功后按runtime checkout在进程内缓存；每次申请额外权限仍重新验证policy和link内容。
首次必要的Git查询无法执行或缺少metadata时明确报错，普通包含判断无需此查询。

绝对输入中的 `main_checkout/.claude/worktrees/<name>/data` 仅在 `.claude`、`worktrees`、worktree均为实际目录，
且data为直接指向标准root的绝对symlink时接受。不接受任意前缀、链接祖先或root链接链。
仅规范化此root前缀，随后仍检查原声明链接和精确E:目标。main/worktree两种写法得到相同逻辑DATA名与规范E:输入；
alignment的scene-root边界及ClipSource序列化也遵循此契约。仍拒绝越出E:引用subtree、
未声明外部链接、目标不匹配、未挂载及父路径逃逸。解析结果仍是规范E:路径；`PathResolver.relative` 将已验证引用转换回逻辑DATA相对名。
Clip Studio发现、project序列化及review CLI使用此API，在保留保存名的同时不改变合成pipeline的物理workspace。普通实际目录/custom DATA root和其他role保持原包含规则。
迁移后恢复consumer之前，须使用本次审核过的runtime和配置。

Linux通过 `O_PATH|O_NOFOLLOW` fd调用 `readlinkat(fd, "")` 仍可能更新旧链接atime。
**dry-run无变更契约仅豁免已声明未管理symlink entry自身的atime。** 此值不参与闲置判断或其他安全判断。
payload内容与atime、其他管理metadata、状态文件及目录均保持不变，测试比较两端tree的metadata及file hash。
管理payload内和未允许的symlink在读取文本前拒绝。

全局布局验证（scan、activate、未指定单位的dry-run）及实际操作固定父目录/link fd，每条命令读取各链接一次，
记录 `atime_before_ns`、`atime_after_ns`、`atime_changed`、方法及 `verification=verified`。
显式选择管理单位的dry-run不会使用与其不重叠的未管理tree，只检查link entry/祖先，不读取target，
并报告 `verification=deferred-explicit-dry-run`、`verified=false`、`method=lstat-only`。
这不表示全布局已验证；部署前应执行scan。实际写操作仍完整校验target。
全局scan若省略target验证将隐藏错误布局，因此不能省略此读取。

首次迁移将历史outputs仅保留在E:，D:输出根从空目录开始。
对常驻输入进行hash校验复制并创建允许的外部引用，与Clip Studio使用者协调切换期间的暂停和恢复时间，
验证可通过D:原data相对路径打开raw视频和processed的 `projects.json`。
容量验收以Linux文件系统空闲量为准，Windows物理D:差值只单独记录。VHDX sparse/压缩仍不属于本次迁移。

## checkpoint 单位与 CI assets

每个 `ckpt/<顶层目录>` 是独立的 `checkpoint` 单位。根目录的普通文件组成另一个名为
`ckpt` 的 `checkpoint-files` 投影（目前为README.md）；根文件参数也选择此投影，而不是所有子单位。
两类只镜像、校验和取回，永不清理。E:的ckpt容器inode本身不交换、不删除；拒绝旧的整ckpt单位操作。

已有根文件manifest时，hydrate/mirror/verify/index在发布或更新证据前，必须同时匹配E:的metadata与保存的SHA-256。
显式index也不能覆盖不一致的证据。迁移旧整ckpt manifest时，先校验其记录的直属文件子集。

根文件只复制、哈希直属普通文件，再逐文件发布。旧版本和源端已移除文件保留在目标tier的
`.storage-tier/checkpoint-file-history/<transaction>/`，不自动删除历史。子目录不受影响。
发布仅逐文件原子，不保证整个集合原子。D:的 `checkpoint-file-operations/` journal记录已校验的计划/副本entry、旧entry/manifest和进度。
未完成操作会阻止投影继续处理。暂停writer，按照[审计与恢复](#审计与恢复)运行 `recover`，保留journal和历史。
移开journal再index并不能完成恢复。不一致不能授权删除归档inode。
投影忽略容器时间戳，但检查容器device/inode/mode及全部直属文件的metadata/hash。D:/E:各自保留容器权限，不将其作为根文件payload跨tier比较。

迁移旧整ckpt manifest时先运行 `index ckpt`，旧manifest保存在E:的 `.storage-tier/legacy-manifests/`。
迁移中断后重试 `index ckpt`，仅在已有归档与当前旧manifest全内容一致时接受，否则保留两份并失败。
然后对scan报告的每个 `ckpt/<目录>` 单独index，不复用旧tree的hash。创建D:的实际ckpt容器，
逐个hydrate根文件和全部子单位，始终保留E:容器。

mount引用检查只用于evict、旧镜像退役（包括发布前检查）和已校验stage的清理。
index/verify/mirror/hydrate读取保留同UID fd/cwd/mmap检查及前后snapshot，不stat bind目标。
审计用 `mount_scope.inspection_enabled` 记录是否检查。删除时即使只读，root相等或内部bind仍fail closed。
普通目录的index dry-run仍跳过quiet；根文件投影执行不含mount检查的quiet。

runner从同一 `Config.load(...).mirror` 和配置推导bind源，不解析repo的data/ckpt symlink。
CI读取E:data/ckpt，只能看到已镜像的单位；仅在D:的新单位成功镜像后（通常每日cycle后）才可见。
启动要求配置的E:挂载点有效，并重新读取源mount（device编号可能随重启变化），比较filesystem内root及device，不直接将 `/mnt/...` 拼写与FSROOT比较。
已有mount的root/device不一致或允许写入时拒绝。

先完成所有不需要sudo的迁移及验收步骤。当前bind在D:切换后仍指向E:。
之后父会话通知人类，协调CI/runner queue维护窗口并等待job结束；停止活动queue会终止训练，必须先排空。
从已经review的checkout执行；installer支持linked worktree并通过Git共用目录确定main：

```bash
runner_service="$(sudo cat /opt/actions-runner/.service)"
sudo systemctl stop "$runner_service" tennis-lab-training-queue.service
sudo bash scripts/github_actions/install_self_hosted_runner.sh
sudo /opt/tennis-lab-actions/bin/mount_assets.sh start
# 迁移及不需要sudo的验收已完成，恢复通常运行。
sudo systemctl start tennis-lab-training-queue.service "$runner_service"
```

installer更新/opt helper/config及system unit对tennis-data-mount.service的依赖，并执行daemon-reload、enable/start。
原有package/runtime安装也会运行，需预留下载时间。当前bind已指向E:，FSROOT/device不变，验证通过时切换前无需unmount。
重新安装修正repo改成D:后的后续启动，不会自动替换运行中的bind。本阶段只实现手顺，不运行sudo或unit操作。

## CLI

在包含所需实现的 checkout 中运行，dry-run 也必须通过有限
[checkout launcher](../lib/README.md#既存cliと新規jobの接続)。

可从linked worktree加载代码/config，用 `--repo` 选择main，无需更新main checkout。

```bash
export PYTHONPATH="$PWD/.agents/ops" PYTHONDONTWRITEBYTECODE=1
R="$(git rev-parse --path-format=absolute --git-common-dir)"
R="$(dirname "$R")"
S=(.venv/bin/python .agents/ops/checkout_job.py --repo "$PWD" -- .venv/bin/python -m storage_tier --repo "$R")
"${S[@]}" scan
```

user service仍从main导入；启用前main必须具有已review的常驻/引用配置及实现。
worktree CLI成功不表示service已更新。不修改他人的dirty文件，与所有者协调部署。

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

## 临时热副本

读取速度足够时直接使用 E: 上的原件。需要临时复制到 D: 时，在主 checkout 的 `data/` 下使用带版本的只读输入。
复用 [config.toml](config.toml) 中 `dataset_containers` 下的版本，或用 `dataset_units` 声明独立版本的根目录。
复制前，先将声明部署到运行日常 `cycle` 的 checkout。若源数据位于 E: 的其他位置，先在 E: 对应的仓库相对路径
准备固定版本，核对文件清单与源数据的 SHA-256，再 hydrate。`resident_units` 用于需要长期保留在 D: 的输入。
2026-11-14 创建的 `~/tennis-hot` 占用约 147 GiB，未纳入单元清单且没有释放日期，一直留在 D: 直到
[容量故障](https://github.com/Motoki0705/tennis-lab/issues/2361)。

1. 复制前，在负责该工作的 issue 中记录负责人、job 或 run、使用 D: 的原因、E: 源路径和 D: 目标路径、
   预计字节数、使用期限和释放检查日期。日期时间注明时区，将检查安排在计划最后使用的 14 天后。
   负责人应在期限内结束使用，或重新核算容量并在到期前更新记录。为副本保留空间直至释放。
2. 用 [CLI](#cli) 的 `scan` 确认能找到预期的 E: 数据集版本，且 D: 目标路径尚未占用，再用 `index --dry-run`
   估算大小。确认 `/mnt/d` 挂载的是 Windows D:，分别查看物理 D: 和 guest 的可用容量。将整个副本、文件系统和
   VHDX 开销，以及其他任务后续写入的峰值计入预算。guest 必须容纳这些写入，物理 D: 必须仍留有 watchdog 阈值
   （默认 20 GiB）以上的可用空间。watchdog 拦截新启动，运行中的任务和手动复制仍会使用空间。

   ```bash
   unit=data/ball_detection/example-v1
   "${S[@]}" scan
   "${S[@]}" index "$unit" --dry-run
   findmnt --mountpoint /mnt/d
   df -B1 / /mnt/d
   ```

3. 容量检查通过后运行 `"${S[@]}" hydrate "$unit"`，把 `hydrated` 结果和 `.storage-tier/audit.jsonl` 中该单元的记录
   留在负责该工作的 issue。hydrate 校验两端并记录 manifest。任务读取副本，将结果写到 E: 上的 worktree outputs。
4. 完成或取消时，解除排队任务的引用，移除临时创建的单元外硬链接，在 issue 中记录最后使用日期。
   其他硬链接仍保留数据块时，继续把该空间计入副本用量。
   到了释放检查日，确认日常 `cycle` 已回收副本，或运行 `"${S[@]}" evict "$unit" --dry-run`。
   若结果为 `would_evict`，运行 `"${S[@]}" evict "$unit"`，并确认输出为 `evicted`，或用 `scan` 确认副本仅在 E:。
   若仍在使用、atime 未到期或校验失败，在同一 issue 中记录原因和下次检查日期。
   释放遵循[常规的 14 天 atime 和校验规则](#校验与排除条件)。分别记录 Linux 和 Windows 的可用空间。
   回收 Windows 空间可能需要 [trim 和 compact](wsl-vhdx.zh-CN.md)。

## 校验与排除条件

`mountpoint`、`findmnt` 必须确认两个不同的 ext4 与指定 atime 选项。E: 缺失绝不解释为空镜像。
数据、目录、manifest 都用 `O_NOATIME` 读取；失败时不退回普通 open。虚拟 `/proc` 根目录和 kernel 生成的 metadata
不包含数据集访问历史，使用普通读取。注入的真实测试目录仍用 O_NOATIME。

镜像与认证前按 device/inode 检查 fd、cwd、mmap，覆盖单位外的硬链接别名。仅检查 `/proc/<pid>/status` 中
有效 UID 与执行用户有效 UID 相同的进程。root、CI 等其他 UID 不在范围内；JSON 和审计的 `process_scope` 按 UID 记录排除数。
同 UID 仅在具备独立证据时排除：四个 UID 均等于非 root 调用用户，fd/maps/cwd 为 root:root 所有，
三项读取都返回 EACCES/EPERM，且前后 PID 启动时间、UID、Name、所有者均不变。
显式 `Dumpable: 1` 否定排除；`CoreDumping: 0` 不作为依据。所有者规则见
[proc_pid(5)](https://man7.org/linux/man-pages/man5/proc_pid.5.html)。此非 dumpable 推断对其他所有者/namespace 映射仍保守失败。
在 `process_scope.excluded_nondumpable` 记录数量、可读 comm（status Name）、PID 和证据。
Z/X 仅在 `Threads: 1` 且 task 列表只有该 PID 时视为退出并单独计数；zombie leader 仍可能有活动线程
（[proc_pid_fd(5)](https://man7.org/linux/man-pages/man5/proc_pid_fd.5.html)）。不忽略仍存活 PID 的 metadata 缺失。
不检查同 UID 的非 dumpable 进程，例如 Codex CLI 本体、sshd、gpg-agent。
另一种 `permitted-capabilities` 排除以相同 user namespace 为运行前提。
目标 `CapPrm` 必须包含 `/proc/self/status` 的 `CapPrm` 没有的 bit，maps 与 cwd 均返回 EACCES/EPERM；fd 列表可以仍可读。
前后 PID、UID、能力值必须不变。数量、comm、两方能力值和 namespace 前提记录在
`process_scope.excluded_capabilities`；两种规则都不使用进程名白名单。
能力集合依据见 [commoncap](https://github.com/torvalds/linux/blob/v6.6/security/commoncap.c)。
无法建立任何排除证据的 EACCES/EPERM，**间隔50ms最多复查3次**。
每个进程每次检查最多请求150ms的等待，syscall与调度耗时另计。
检查资源前固定PID和 `/proc/<pid>/stat` 启动时间；初始身份不可读或不稳定时保守失败。
每次复查前，仅在确认PID消失且没有后继时计为已退出。
启动时间改变表示后继存在：保留全部已观察到的busy参照，即使未观察到busy也以unknown停止。
原个体退出不能证明后继已被检查。不重新固定后继身份，因为持续PID替换会要求超出既定复查上限的完整检查。
同一个体必须保持UID不变，并重新验证现有thread-group退出证据，再**完整检查fd、cwd、root、mmap及适用的mount参照**。
此前观察到的使用中证据仍保留。UID改变、意外I/O错误、达到上限仍不可检查的存活个体保持unknown，阻止认证或删除。
读取被拒或maps可读本身都不能证明未使用。
在thread-group证明期间、资源检查后或排除证据的最终身份核对时发现替换，也按unknown处理。
不将后继的排除证据当作原固定个体的证据。

`process_scope.rechecks` 记录上限、间隔、复查总次数、请求等待总量 `wait_seconds`（非实测经过时间），
及通过重新检查、退出或既有排除证据得到解决的个体数 `resolved_processes`（按PID/启动时间去重）。
`processes` 保留每次复查事件的身份、首次/最后拒绝、次数及结果，也包括未解决事件。
与最近一次进程清单不同，该历史在单个命令的所有单位检查中累积，写入JSON输出及后续审计记录；下一命令重置。
读取数据的训练和脚本必须 dumpable 且可检查。UID 未知、意外 I/O 错误或两种证据都不成立时仍失败。
namespace 身份始终记录为 `unverified`。`/proc/<pid>/ns/user` 也执行相同 ptrace 检查，因此不打开 handle
（[kernel 实现](https://github.com/torvalds/linux/blob/v6.6/fs/proc/namespaces.c)）。
辅助检查在探测前后读取目标与自身的 `uid_map`。可读且不同或 metadata 非法时失败。
不可读则不因此失败，记录错误；相同只记为 `matched`，不作为 namespace 身份证明。前后观察均保存在审计记录中。
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
删除固定 root FD，并使用包含 hash/atime 的完整 entry 计划。每个 entry 先原子移入同父目录的专用保留区，
再校验实际移入对象后 unlink/rmdir。原名称被重新创建时，新 entry 不会成为 unlink 目标。
不匹配对象无覆盖恢复，无法恢复则保留。跟踪内部 hardlink 因自身操作产生的 ctime/link 数变化，仍重新哈希 payload。
每个 regular file 在 claim 后，必须先通过只读 FD 获取 Linux `F_SETLEASE(F_WRLCK)`，再做最终身份、metadata、SHA-256 校验。
已有 open FD、所有者/权限错误、不支持 lease 或 `/proc/sys/fs/leases-enable` 禁用时，保留对象并失败，绝不降级为无锁删除。
lease 按 inode 生效，也覆盖外部 hardlink 别名；新的 open 请求 break，在持有期间不能写入。
unlink 前必须满足 `F_GETLEASE == F_WRLCK`，unlink 后及完成 fsync/journal 等操作的 `F_UNLCK` 紧前再次检查。
之前的 break 会恢复/保留 entry 并失败；
之后发现 break 则按事故处理，将阶段与确认删除记录写入 journal/audit，操作失败。
SIGIO 通过 Linux `F_SETOWN_EX` 定向到调用 thread，在该 thread 阻塞，释放后回收并恢复原 signal mask。
最后 `F_GETLEASE` 至 `F_UNLCK` 间收到的通知也传播为失败，绝不读取后丢弃并宣称成功。
CLI 在持有 lease 期间专用该 thread 的 SIGIO。

使用中检查按 FD 的 device/inode 匹配，也检测已有 `O_PATH` FD。单位在 filesystem 内的路径从工具自身 namespace 求得：
固定 snapshot root FD，按 mount ID 找到自身 mount root/point，再结合相对路径；quarantine、retained、stage 名称同样适用。
只在删除路径对可检查的同 UID 进程读取 mountinfo，仅对相同 device 且 root 等于单位或位于其**内部**的 mount，
经 `/proc/<pid>/root` stat mountpoint；inode 匹配即判为使用中，包括仅在其他 namespace 可见的单一 file bind。
祖先和外部/兄弟 root 不 stat，直接排除；按 component 区分 `data/a` 与 `data/ab`。
`process_scope.mount_scope` 审计 filesystem 单位路径、检查数、祖先排除数、外部排除数（每进程的观测次数）。
读取/权限错误不作为空列表；`//deleted` 等不可判定 root、范围内 mountpoint 无法到达/读取时检查失败。
范围内root的stat结果不属于已校验单位inode时也记为unknown，因为上层mount可能遮住仍存活的下层file bind。祖先与外部root仍排除。
CI runner 的只读祖先/外部 data bind 即使无法到达，也不属于单位内 inode 参照，不 stat、不判失败。
root 等于单位时，即使只读也仍在范围内。原有进程排除规则适用；无可检查成员进程的 namespace 无法枚举。

在 WSL2 Linux 6.6 实测，O_PATH 和 private file bind 均可在 `nlink=1` 时获取 write lease；
unlink、释放后，通过 `/proc/self/fd/N` 或 bind alias 可以重新打开同一 `nlink=0` inode 并写入。
因此已有参照会事先检测，但前提是 consumer **不在使用中检查后新建 O_PATH/proc-FD/bind 参照**，也不枚举、
操作专用保留区的随机 token 名称。检查、lease 检查和 unlink 不是原子操作，竞争区间包含校验/fsync，不保证仅几微秒。
新持有的参照可在释放后向已 unlink 的 inode 写入；若最后名称已删除，这些写入无法按名称恢复。
检测到的 break 记录为事故，不宣称删除成功。复现实测是 storage 测试中的
`test_opath_reference_is_busy_and_can_reopen_an_unlinked_inode` 和
`test_external_file_bind_reference_is_inspected_and_survives_unlink`。
`nlink > 1` 时其他别名通常保留 payload，除非该名称也被外部删除。
lease 不排他保护同 UID 的目录 entry 操作；身份/link 数检查拒绝检测到的替换，不保证抵抗恶意 namespace 控制。
这会增加 D:/E: 读取和逐 entry 的 journal/directory fsync；完整数据集删除耗时留到第二阶段实测。实现位于 `removal.py`。
隔离目录被替换时不删除或冒充原单位恢复，保留错误供人工检查。
从 repo/E: 基准 mount 到单位 root 的每个路径成分都通过 nofollow FD 与 mount ID 校验，拒绝同 device 的 bind mount。
另按 checkpoint 目录的 device/inode 拒绝 dataset/run 的实际别名。普通文件 hardlink 的 dataset 名称删除不会删除 ckpt 名称。
mount ID 仅为运行期证据，不作为跨 boot 的持久身份。
hydrate 拒绝已存在的 D: 目标，完成 E: 哈希、复制、D: 哈希后才发布，并将文件 atime 更新为当前时间。

## 审计与恢复

D:`<repo>/.storage-tier/audit.jsonl` 保存校验、取回、失败、删除确认与完成记录，包含 UTC、单位、大小与校验方式。
修改操作按 repo 排他；dry-run 不创建锁。持久删除 intent 位于同目录的 `transactions/`。
各审计事件记录其他 UID 的排除范围。失败仅写入 audit 和 service journal，不发送通知。
普通目录的只读index dry-run不调用单位quiet（根文件投影检查fd/cwd/mmap）；两者都不能证明E: mount引用检查成功；
它也不认证 manifest、不授权删除。实机验证须通过 `verify --dry-run` 检查两边，记录 E: `quiet` 通过与 metadata 不变。
布局迁移前，必须区分既有 D:/E: 配对的只读路径映射验证与仍拒绝待迁移布局的普通 CLI 结果。
`removals/<transaction>.jsonl` 持久记录 claim/删除 intent、移入身份、已校验 hash 与确认删除的 entry。
崩溃可留下没有完成记录的 intent；它表示未确认，不能直接算作删除完成。

```bash
"${S[@]}" recover
```

恢复将隔离单位无覆盖地移回，**绝不恢复执行删除**。原位置被占用或身份改变会明确失败。
部分 tree 会移回并标记可能不完整；保留的 entry 无覆盖移入 `recovered-assets/`，完整计划留在
`recovered-transactions/`。不完整/已修改 tree 或保留 payload 会创建 `recovery-required/` 状态，阻止该单位的
mirror/evict/cycle，避免用不完整 D: 覆盖完整 E: 副本。

1. 检查恢复结果、保留对象、journal 与计划，保存全部未镜像修改。身份变化或目标被占用需人工检查。
2. 将部分 tree 保存在管理 root 之外，空出 D: 原位置，再 `hydrate` 完整单位；或明确修复完整 D: 后对 E: 执行 `verify`。
3. 只有全量 SHA-256 hydrate/verify 成功才解除恢复状态。明确核对、整合之前保留修改，恢复绝不继续删除。

若崩溃前已删除，则记录事实，
不宣称已实测容量收益。未完成 transaction 阻止后续操作。mirror/hydrate发布中断时，暂停writer并运行 `recover`，
不要删除journal、强制index或丢弃stage。普通目录在创建stage前，先在D:`publications/`保存 `copying` intent，副本校验后才改为 `prepared`。
根文件在 `checkpoint-file-operations/` 保存已校验副本与逐文件进度。
恢复将稳定的source/destination重新SHA-256与持久计划比较，只继续已校验的剩余发布步骤，绝不继续删除。
`prepared`发布恢复拒绝源变更、计划外目标或未校验stage。写入新证据前，在D:`publication-evidence/`
保存原始旧manifest、journal及恢复时的manifest。旧目录仅移动至目标tier的
`.storage-tier/recovered-publications/`，不删除；根文件旧inode仍在 `checkpoint-file-history/`。
再次哈希D:/E:后才记录新manifest并将receipt标记完成。保留完成receipt和历史，再运行 `verify`。
普通mirror/hydrate/index仍拒绝与已保存证据不一致的内容。若根文件的已校验副本尚未生成就中断，
仅在旧目标内容完整时允许中止，不生成新manifest。旧格式的未完成journal保留并阻止操作，等待人工检查。
绝不静默清理stage残留。

`copying` 中断时，先确认旧目标完整，再将stage连同部分副本及未校验bytes保留至历史并中止操作，不生成新manifest。
恢复后重试mirror/hydrate。即使receipt已为 `complete`，只要旧stage仍在，也必须恢复。
先记录 `archiving` intent，保留同一旧目录到 `recovered-publications/`，再完成receipt。
这覆盖manifest提交后、退役开始前的中断，也适用于旧实现的完成receipt，不重新认证manifest。

若在retained目录创建后、identity记录前中断，恢复先固定并持久记录观察到的identity，再归档。
这是归档custody记录，不是删除payload的校验证据，记录中明确标记。与已有identity不一致仍失败。
JSON原子替换遗留的 `.<record>.<uuid>.tmp` 普通文件予以保留，只读取正式 `.json` record。
其他异常名称、特殊文件或损坏的正式record仍失败。各命令校验已提交的控制JSON，
并在 `uncommitted_json` 中列出保留临时文件的path、bytes和 `committed=false`。
校验证据来自完整的正式record。manifest、transaction或receipt写到一半时，保留临时文件和原已提交record，
保留报告中的临时文件供检查。

D:`audit.jsonl` 或 `removals/<transaction>.jsonl` 的最后一行截断时，普通命令以journal错误失败。
暂停写入后执行 `recover`。恢复先在 D:`.storage-tier/journal-repairs/` 持久保存原journal的SHA-256与完整前缀长度，
将包含截断尾部的整个原文件保存到 `journal-history/`，然后原子发布仅包含以换行结束的JSON object行的有效journal。
完成receipt，并在audit和命令结果中记录 `journal_repaired`、归档path及尾部bytes。
再次中断后，`recover` 校验同一归档内容并继续。完整JSON object若缺少末尾换行，也归入保留尾部。
中间行或正式JSON损坏时失败；检查保留证据、恢复对应的持久record后重试 `recover`，保留所有冲突记录。
publication及ckpt operation的journal使用原子JSON替换，并执行相同的已提交record校验。

[中断边界测试](../../../tests/integration/agents_ops/test_storage_tier_crash_boundaries.py)覆盖下列持久化序列。
[完整边界清单](../../../tests/support/agents_ops/fixtures/storage-crash-boundaries.json)包含JSON临时文件创建/write/flush/fsync/替换、
副本chunk与metadata、写入用 `os.open` flag、`os.write`/`pwrite`/`writev`、stream write/flush、
truncate/ftruncate、rename/claim、删除、journal追加及恢复写入。
连续JSON encoder fragment合并为代表write边界，binary复制保留第一及下一个chunk边界。
[部分写入测试](../../../tests/integration/agents_ops/test_storage_tier_writes.py)检验超过8MiB的复制、大JSON部分写入、
截断追加与修复证据被改变，并验证增加未登记的现有文件write会使trace完整匹配失败。
directory fsync前后的进程中断可见状态与前一修改相同。本验证针对进程退出。
每个中断状态映射到[恢复路径清单](../../../tests/support/agents_ops/fixtures/storage-recovery-boundaries.json)，
[恢复边界测试](../../../tests/integration/agents_ops/test_storage_tier_recovery_boundaries.py)再次中断每条路径，
要求按文档修复后，正式E: payload相等且无参数日常cycle成功。

| 操作 | 持久化顺序 |
|---|---|
| 新建/更新镜像 | copy intent → stage副本 → 已校验copy intent → 发布rename → manifest → 完成receipt → 旧stage退役 |
| hydrate | copy intent → stage副本 → 已校验copy intent → 发布rename → manifest → 完成receipt → 清除恢复marker |
| evict | prepared transaction → 隔离rename → 移动identity → 创建retained → 已校验计划 → entry claim/删除journal → 删除 → audit → 完成transaction → unlink transaction |
| stage退役/清理 | 父发布receipt → prepared删除transaction → 创建retained → 已校验计划 → entry claim/删除 → audit → 完成transaction → unlink |
| ckpt根文件 | prepared journal → 历史/副本 → 发布校验证据 → 逐文件交换/进度 → 旧文件归档/进度 → manifest → 完成journal → audit |
| JSONL修复 | 带原digest/prefix的receipt → 保留完整journal → 原子发布完整前缀 → 完成receipt → audit。再次中断后按保留的原bytes继续 |
| 旧ckpt index | 校验已记录直属文件 → 归档旧manifest → 发布根文件manifest → audit。重试仅接受匹配的已有归档 |
| recover | 还原/保留tree → 记录归档identity → 归档tree/entry → 保存旧证据/计划 → 部分恢复marker或发布校验 → 完成receipt/解除active transaction |

恢复中断后再次运行 `recover`，保留报告的历史。部分D: tree按上述完整hydrate/verify步骤修复，
activate后确认无参数日常 `cycle` 正常。恢复绝不继续删除entry。

`RENAME_EXCHANGE` 后的旧 E: tree 也只按交换前带 hash 的 snapshot，通过相同 claim、lease、持久 journal 路径删除。
旧对象被更新或增加 payload 时保留并使 mirror 失败。recover 将旧 tree/entry 保存在 E: 内的
`<mirror-repo>/.storage-tier/recovered-trees/`、`recovered-assets/`，计划/journal 留在 D:，绝不回滚新发布的镜像。
旧副本删除中断会建立恢复标记；先保存、核对 E: 独有修改，再全量 verify/hydrate。archive 不自动删除。
未发布的 mirror/hydrate stage 也仅在具有完整已校验 snapshot 时通过同一删除路径清理。
不完整/未校验副本先保留staging名称并写入 `stage_preserved`，有receipt时由recover归档。

旧实现的孤立stage可能没有能定位单位的receipt/transaction。recover重新扫描管理root，
此类stage仍在时，以 `unclaimed storage artifacts` 失败。按下列步骤显式保留。

1. 暂停producer，选择scan/recover报告的准确stage路径。
2. 使用正常mount/nofollow检查创建同tier的 `<tier-repo>/.storage-tier/manual-recovery/`。
   通过 `filesystem.rename(stage, archive)` 按原stage名原子移入，使用 `RENAME_NOREPLACE`，目标占用时失败。
   记录两个路径及目录的device/inode。
3. 保留全部bytes供检查。recover后重试原mirror/hydrate/index，或按上述步骤完整verify/修复部分tree，
   再确认无参数日常cycle。孤立stage归档测试固定此步骤。

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
   并核对非 dumpable/能力差的排除证据与记录及相同 user namespace 的运行前提。可读 uid_map 不同或本检查证据不足仍为错误；
   不能忽略，也不隐式改变权限或停止其他 session。
2. 清点 E:、活动与等待任务引用、配置间接引用、硬链接别名、E: 绝对路径和已有 D: 副本。
   逐个取回全部ckpt子目录、根文件投影和需要的版本、run单位。
3. 将主 outputs alias 归为主 outputs 的引用。独立的旧 E:`.claude/worktrees/<name>/outputs` 在 consumer 暂停时
   rename 到 E:`.worktree-outputs/<name>`，对应关系见 scan 的 `legacy_outputs`。目标已存在时拒绝覆盖、合并，先显式比较。
   旧 manifest 的路径/ctime 不能复用，需按新规范路径重新 index。rename 是第二阶段的修改操作。
   对明确指定且稳定的 E: 单位逐个 `index`。若顺序速度为 80–150MiB/s，432GiB 单 pass 约 49–92 分钟；
   小文件 seek 可能更慢。这是规划假设，并非全盘实测。index 读 E: 一次；hydrate 还需 E: 哈希、复制与 D: 验证。
   不从小型热缓存 probe 推导 HDD 全盘速度。
4. 保存旧 symlink entry，并 rename **链接自身**，不移动 E: payload。为 D: `data/`、`outputs/` 创建空实际目录；
   创建实际ckpt容器，逐个hydrate子目录和根文件。独立 worktree outputs 链接到 D: 持久根，主 alias 链接到 D: 主 outputs。
   按规范路径取回选择的单位、验证内容和代表性硬链接后再 activate 输入诊断。已有 D: 副本需显式比较，hydrate 不做合并。
5. 由对应 session 更新活动、等待命令。先做只读 cycle，再以可丢弃单位执行 mirror→evict→hydrate。
   检查 `statvfs` 空闲量、审计、consumer 读取、同 UID 检查和失败 journal/audit，之后才启用 timer。
   queue 记录和其他 UID 不保护单位。外部硬链接可能阻止容量回收。
   rollback 仅在 consumer 暂停时恢复旧链接，并独立保留 D: 新写入。Windows 磁盘维护参见 [Ubuntu VHDX 维护](wsl-vhdx.zh-CN.md)。

`tests/integration/agents_ops/test_storage_tier.py` 使用临时真实文件与模拟 mount/proc，并覆盖真实 fd 和硬链接。
CLI/worktree 由 `tests/e2e/agents_ops/test_storage_tier_cli.py` 验证；扩展既有 ops-control 安装测试与 configuration path 错误测试。
