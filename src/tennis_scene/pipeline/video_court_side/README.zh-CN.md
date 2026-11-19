# 按视频估计固定相机的球场朝向

[English](README.md) · [日本語](README.ja.md)

`court_side.mode=video_shared` 为同一视频的固定相机估计一组180度旋转标记。它要求使用 `court_kp.mode=video_shared`，并沿用共享球场的数据集和视频 ID。默认模式为 `court_side.mode=clip`。
设置 `court_side.evidence=player` 可使用[球员脚点](../../../tasks/court_side/README.md#プレーヤーから判定する)。构建和加载时使用同一种证据。已有视频判定使用默认的球观测。

只保存了六个阈值的旧配置仍按片段处理。现行格式要求同时指定 `mode` 和 `video_shared`。

每个片段使用原有的采样网格，并根据选定的证据评估假设。片段 ID 与片段内的原始帧号共同标识一帧。对于每种朝向组合，系统按各片段的有效多视角帧数加权平均 cost 和 support，累加相机对的共同帧数，再用该证据的阈值作视频级判定。视频产物按内容哈希引用独立诊断对象，其中保存每个片段的有效帧号、假设排名、相机对帧数、margin 和未决原因。没有有效帧时，排名为空，margin 为 null。

## 单片段入口

提供[共享球场配置](../video_court/README.zh-CN.md)和目标 `clip_id`。在 `src.tennis_scene.scripts.run_pipeline` 中加入以下配置以加载已有结果。

```bash
court_side.mode=video_shared \
court_side.video_shared.action=load \
court_side.video_shared.directory=tennis_scene/court_side_shared/match \
court_side.video_shared.artifact_digest="$SIDE_DIGEST"
```

构建时指定 `action=build`、`directory`、`request` 和 `request_sha256`。两个路径均相对于 ARTIFACT 根目录。请求必须列出共享球场的**每个片段**的存储目录，每个目录需包含 `court_calibration` 和所有相机的观测。

| 证据 | 每台相机的输入 | 版本摘要 |
|---|---|---|
| `ball` | `ball_points/<camera>` | `PinnedSideInputs.from_store(directory).digest` |
| `player` | `player_selection/<camera>` | `PinnedSideInputs.from_store(directory, evidence="player").digest` |

该摘要包含视频来源和输入组件的版本，同一目录后续发布的场景可以沿用它。球员证据可从已保存的筛选产物 v2 和 v3 读取观测框及掩码。下面是只有一个片段的视频示例。

```json
{
  "schema": "video_court_side_inputs_v1",
  "clips": {
    "video_000/clip_000": {
      "role": "data",
      "directory": "match/dataset/videos/video_000/clips/clip_000/annotations/tennis_scene",
      "inputs_digest": "<revision digest>"
    }
  },
  "evidence_clip_ids": null
}
```

`role` 可选 `data` 或 `artifact`。默认使用全部片段。显式指定 `evidence_clip_ids` 可限制参与视频判定的片段，系统仍为所有片段计算独立诊断。相同输入的构建会复用结果。`rebuild=true` 显式重新计算并允许修复损坏的对象。加载时校验判定和诊断对象的内容哈希。

## 片段结果与多片段调用

构建和加载也会核对标定所用的球场漂移诊断版本。片段节点核对选定的采样观测、标定矩阵及参考相机。独立判定与视频结果冲突的片段以 `court_side_video_shared_conflict` 失败。未决片段采用视频结果并记录警告。设置 `court_side.video_shared.unobservable_policy=fail` 可以拒绝未决片段。如果视频本身未决，两种策略都会失败。`run.json` 记录两套排名、margin、处理结果和两个对象的摘要。视频产物的标识区分证据类型，节点与数据集标记的缓存标识包含共享判定的摘要及执行策略。

预备计划使用同一共享球场和 `court_side.mode=clip`，为所有片段准备标定与上表列出的输入。将片段 ID 到 `PinnedSideInputs` 的映射传给 `build_video_court_side`，或通过 `prepare_video_court_side` 按配置构建或加载。对各片段调用 `standard_definition` 时传入共享球场和 `video_court_side`。媒体、标定、观测及数值设置必须与证据匹配。通过 `artifact.clip_report(clip_id, policy)` 可获取片段诊断。

复用构建结果或将其绑定到计划时，会核对请求哈希、证据片段集合、所有片段的当前输入版本，以及标定和选定观测的数组校验和，包括仅用于独立诊断的片段。显式加载视频结果时，会校验指定摘要的判定和诊断对象。

构建和加载都会把当前的 `court_kp.video_shared.unobservable_policy` 应用于每个证据片段的所有相机。任何观测被拒绝都会停止采用该结果，并给出片段、相机和漂移原因。调用底层 API 时通过 `court_policy` 传入该设置（默认 `warn`）。未参与视频判定的片段在自身执行或发布时也须通过球场检查。

片段自动缓存和显式加载均要求共享组件生成的输出，并校验输入依赖、共享摘要和两项策略。共享模式拒绝导入或替换 court_side 组件。创建和复用数据集完成标记时，也会检查片段判定及场景导出对该组件的依赖。
