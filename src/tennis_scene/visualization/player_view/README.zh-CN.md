# 球员视角的已保存场景 bundle

[English](README.md) · [日本語](README.ja.md)

此 CPU 导出器将已保存的 SceneResult v2 转成 Blender 与 3DGS 共用的输入，
复用现有网格放置、PC2 和球场几何。头部相机与渲染由下游负责。

```bash
.venv/bin/python -m src.tennis_scene.visualization.player_view \
  --input /absolute/path/to/store \
  --output outputs/tennis_scene/visualize/player_view/example/bundle \
  --smpl-faces /absolute/path/to/body_models/smplh/neutral/model.npz \
  --smpl-joint-regressor /absolute/path/to/body_models/gvhmr/smpl_neutral_J_regressor.pt \
  --smplx-model /absolute/path/to/body_models/smplx/SMPLX_NEUTRAL.npz \
  --frames 10 110 --player-ids 7 42
```

输入支持 NPZ 及其必需的 metadata sidecar、store 目录或 `scene.json`。
store 仅通过 `indexed_scene_path` 解析已采纳的完整导出，并验证校验值与当前依赖链。
缺失或过期的导出会被拒绝。无需原视频、球重投影或推理。
[现有 reveal](../blender/export/README.md) 保留视频 plate 与重投影检查。

显式提供三角面、与网格对应的 24 行 SMPL regressor 和 neutral SMPL-X NPZ。
后者提供 21 个 body pose 关节的骨架；仅读取数值字段，不初始化模型。
派生数值资产包含在 bundle 中，下游无需原模型文件或 store。

`--frames` 为原帧半开区间 `[start,stop)`，省略时为全场景。
`--player-ids` 使用人物 ID，保留指定顺序；省略时按原顺序导出全部人物。
人数可为任意正整数，未知或重复 ID 会被拒绝。已有输出不会覆盖。
验证后在临时目录完成写入，再发布完整目录；失败时删除临时输出。输入只读。

## 格式

`manifest.json` 的 schema 为 `tennis_scene_player_view_bundle_v1`。
`players` 记录 bundle `index`、原 `source_index`、`player_id`、PC2 路径、有效帧数与人物 `scale`。
`frames` 记录 start、排他 stop、原总帧数；`num_frames` 与 `fps` 保留原时间轴。
`arrays` 包含各数组的相对 path、shape、dtype。另有坐标、SMPL、PC2、mask 约定、球场几何、
请求的选择设置、原 archive/metadata 摘要，以及 store index 摘要和采纳的 export 引用。
`config` 中 null 表示全部输入，实际区间和人物见 `frames`、`players`。
`source_metadata.json` 保存原 metadata，包括放置配置与诊断。

P=人数，F=stop-start，V=顶点数，N=面数。数组 key 与 `.npy` 文件 basename 一致。

| key/file | shape | 类型、单位、含义 |
|---|---|---|
| `source_frames`, `source_time_seconds` | F / F | int64 原始零起始帧 / float64 帧÷原 FPS，秒 |
| `player_<index>.pc2` | F,V,3 | float32，已放置网格，m |
| `player_observed`, `player_valid`, `player_heading_valid`, `player_smpl_valid` | P,F | bool，原 v2 mask |
| `player_kp_3d_vis` | P,F,17 | bool，原关节 mask |
| `player_rejection_code`, `player_kp_3d_rejection_code` | P,F / P,F,17 | uint8，原原因 |
| `ball_positions`, `ball_3d_valid`, `ball_rejection_code` | F,3 / F / F | float32 m / bool / uint8 |
| `smpl_body_pose`, `smpl_betas` | P,F,63 / P,10 | float32，父关节相对轴角 rad / 无量纲形状 |
| `smpl_global_orient`, `player_yaw` | P,F,3 / P,F | float32，补偿旋转轴角 / 球场 +Z yaw，rad |
| `player_position`, `smpl_root_local` | P,F,3 | float32，球场 joint0 平移 / 局部网格 row0 回归根点，m |
| `body_to_court_rotation` | P,F,3,3 | float32，解码旋转，列向量 |
| `smpl_faces`, `smpl_joint_regressor` | N,3 / 24,V | int32 面索引 / float32 回归权重 |
| `smplx_parents` | 22 | int32，根父节点=-1 |
| `smplx_rest_joints`, `smplx_joint_shapedirs` | 22,3 / 22,3,10 | float32，单位尺度 SMPL-X 静止关节 m 与形状基底 |

PC2 从 cache frame 0 开始、sample rate 1；sample i 对应 `source_frames[i]`。
不重采样或填补缺失。mesh/球 mask 为 false 时隐藏对象，不从零坐标推断有效性。
无效 PC2/球为零，mesh 无效帧的旋转与 root 也为零。
其他原数组保留各自 mask；root 有效不代表 mesh 有效。

## 放置与头部姿态

world 为右手系球场坐标，单位 m，XY 地面，+Z 向上，X 为宽度、Y 为长度，原点在球网中心下方地面。
局部网格已经应用人物 scale。
与现有 renderer 相同，使用列向量：

```text
A = [[1,0,0], [0,0,-1], [0,1,0]]
R = Rz(player_yaw) @ A @ Rodrigues(smpl_global_orient).T
world_vertex = R @ (saved_local_vertex - smpl_root_local) + player_position
```

`body_to_court_rotation` 保存 R。`smpl_global_orient` 为 renderer 补偿旋转，
不是原 GVHMR incam 旋转。`smpl.parameters.transl` 指向球场 joint0 的 `player_position`；
原 incam transl 不存在于 v2，导出器不会合成它。

下游可沿骨架组合 R 与 body pose 父相对旋转，得到头部方向。
`smpl.body_joint_names` 为 root 加 21 个关节，joint j 使用 pose triple j-1。
骨架会与共用的具名 body 骨架验证。
带形状的 rest joint 为 `smplx_rest_joints + einsum('jck,k->jc', smplx_joint_shapedirs, betas)`。
头部位置可由已放置网格及同捆的 SMPL regressor 回归。
SMPL-X FK 关节与转换后的 SMPL 网格拓扑不同，不可直接假定位置相等。
单位尺度骨架与 offset 使用保存的 scale，不要再次缩放 PC2。
有有效网格的人物必须有已保存的正 scale；仅全部帧无效时允许 `scale: null`。

bundle 保留头部旋转与基于网格的头部位置所需的信息。
头部相机、正面轴、面部 offset 与相机缺失策略由下游定义。
