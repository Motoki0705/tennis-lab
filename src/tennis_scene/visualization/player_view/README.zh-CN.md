# 球员视角的已保存场景 bundle

[English](README.md) · [日本語](README.ja.md)

此 CPU 导出器将已保存的 SceneResult v2 转成 Blender 与 3DGS 共用的输入，
复用现有网格放置、PC2 和球场几何。下述头部相机导出器只读取此 bundle，渲染由下游负责。

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

## 头部相机

只生成一次相机序列，然后把同一份保存的姿态交给 Blender 和 3DGS。
CPU 命令只读取 scene bundle v1，无需原始 store、视频、body model 或推理。
保留所有人物的原始 ID 和时间轴。输出必须是输入 bundle 以外的新目录；
完整结果原子发布，失败时删除临时输出。

```bash
.venv/bin/python -m src.tennis_scene.scripts.export_player_view_cameras \
  --bundle outputs/tennis_scene/visualize/player_view/example/bundle \
  --output outputs/tennis_scene/visualize/player_view/example/cameras \
  --width 1280 --height 720 --horizontal-fov-degrees 90 \
  --face-offset-head-m 0 0.08 0.12 --near-m 0.05 --far-m 100
```

### 配置与方向

上述命令显示原型默认值。面部 offset 相对于网格回归的 head 关节，采用
**单位尺度 SMPL 头部轴**，单位为应用人物 scale 之前的米。
near/far 是球场米单位的相机 +Z 深度，与人物 scale 无关。manifest 保存所有设置。
像素为正方形、skew 为零、主点为 `(width/2,height/2)`，
`fx=fy=width/(2*tan(horizontal_fov/2))`，不使用镜头畸变。

在选择 `head` 之前验证具名骨架与网格关节顺序。
joint j 由 pose triple j-1 控制，旋转链依次为
`spine1 → spine2 → spine3 → neck → head`。列向量采用以下公式：

```text
H = body_to_court_rotation @ R_spine1 @ R_spine2 @ R_spine3 @ R_neck @ R_head
head_position = smpl_joint_regressor[mesh_head_row] @ placed_PC2_vertices
camera_center = head_position + H @ (player_scale * face_offset_head_m)
camera_to_court[:3,:3] = H @ diag(-1,-1,1)
camera_to_court[:3,3] = camera_center
```

SMPL/SMPL-X 的 neutral 头部轴为 +Y 向上、+Z 面向前、+X 朝人物左侧。
因此 OpenCV 相机的 +X 右、+Y 下、+Z 前对应头部的 −X、−Y、+Z。
正面方向已用上游 [vertex ID](https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py)
指定的 neutral 模型鼻、眼、耳位置核对。保存的 63 值 pose 是线性 SMPL-X→SMPL
拓扑转换前使用的 SMPL-X body 关节旋转，并非 SMPL 的 69 值 pose。
旋转 FK 使用该骨架，位置使用**转换且放置后的 SMPL 网格**。
不要用 SMPL-X FK 位置替代，不要重新居中或缩放 PC2，
也不要把编码的 global orientation 当作头部旋转。

每个原始帧使用静态针孔并保留 roll。不额外平滑、插值、稳定或追踪球。
头部姿态允许相机朝向球场外，或让对手、球离开画面。

### 相机输出 schema

`manifest.json` 使用 `tennis_scene_player_view_head_cameras_v1`，记录输入 bundle
引用和已使用文件的 digest、原始帧区间、FPS、ID/index 对应、人物 scale、有效数与
无效原因计数、配置、轴和投影、头部计算方式及缺失策略。
各 `players[index].path` 为可独立读取的 NPZ（F 帧）。

| Key | Shape | 类型与含义 |
|---|---|---|
| `K` | 3,3 | float64，固定内参，像素 |
| `camera_to_court` | F,4,4 | float64，OpenCV 相机→球场齐次变换，平移单位米 |
| `camera_valid` | F | bool，使用姿态前必须检查 |
| `camera_rejection_code` | F | uint8，下述原因码 |
| `head_position` | F,3 | float64，从已放置网格回归的 head 位置，米 |
| `source_frames` | F | int64，原始零起始帧号 |
| `source_time_seconds` | F | float64，原始帧/FPS |
| `source_player_rejection_code` | F | uint8，保留输入人物原因，与相机原因独立 |

对于有效 T，`X_camera=T[:3,:3].T @ (X_court-T[:3,3])`，然后使用 K 投影。
+Z 为深度。画面和 near/far 裁剪由渲染器依据保存配置处理，不修改姿态。
无效变换和 head 位置**全部为 NaN**，不使用单位矩阵或零姿态；K 始终有定义。

原因按优先级为 `0 VALID`、`1 MESH_INVALID`、`2 HEAD_POSE_NONFINITE`、
`3 PLACEMENT_ROTATION_INVALID`、`4 HEAD_POSITION_NONFINITE`。
网格缺失 mask 优先于填零数值检查。相机需要有限的正旋转及头部链 pose。
骨架、regressor 或时间轴不一致会拒绝整个导出。
仅完全缺失的人物允许 null scale，其全部相机保持无效。
不回退到身体 yaw、相邻帧或固定相机。

### 读取相机

```python
from pathlib import Path
from src.tennis_scene.visualization.player_view.camera_bundle import load_head_cameras
from src.tennis_scene.visualization.blender.export.camera import to_blender_camera

saved = load_head_cameras(Path("outputs/tennis_scene/visualize/player_view/example/cameras"), player_id=7)
index = 0  # cache index；原始帧为 saved.source_frames[index]
if saved.series.camera_valid[index]:
    camera = saved.series.pinhole(index)
    blender_pose = to_blender_camera(camera, sensor_width_mm=36.0)
```

reader 验证保存的变换和 mask，无需输入 bundle。
`pinhole()` 拒绝无效 sample；渲染器应保留其时间槽并显示相机缺失帧。
Blender 相机旋转为 `T[:3,:3] @ diag(1,-1,-1)`，位置为 `T[:3,3]`。
使用已保存的分辨率、内参和裁剪值，两个 backend 均不重新计算头部运动。
