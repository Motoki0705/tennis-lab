# 固定输入的网球3D评估

[English](README.md) · [日本語](README.ja.md)

`ball3d/` 使用相同的历史component，在CPU上重放各方法，并用独立的2D观测评分。
它不改变标准pipeline的默认行为，也不执行检测器推理。

| 模块 | 契约 |
|---|---|
| `bundle.py` | 提取并验证固定输入、原identity/source哈希和完整依赖descriptor |
| `contracts.py` | 仅含选定视角的方法输入与带mask的3D结果 |
| `baseline.py` | 从保存的检测观测重放consensus、LOESS、三角测量，并单独核对原始热图解码 |
| `runner.py` | 全视角及三个双视角LOVO折，带类型和checksum的保存/加载 |
| `annotations.py` | 仅供评分的原视频像素标签、复核状态和来源 |
| `scoring.py` | 分段重投影、覆盖率、运动/事件诊断及标注几何参考 |
| [比较可视化](../visualization/ball3d/) | N种方法的RGB叠加、俯视/侧视轨迹、图表、裁剪图和可携带HTML |

## 固定输入与来源

`create_bundle` 复制原 `scene.json`、选定component的descriptor和数组，以及传递依赖的全部descriptor。
它保留原path、identity、代码来源和checksum，不把历史store冒充为当前pipeline的运行。
原store保持不变；未选中依赖的数组不是重放所需输入，因此不复制。

加载器验证source文档、依赖链、文件清单、数组checksum/类型、camera/frame轴、
evidence窗口来源、court side、标定及观测mask。标注import来源及未声明文件会被拒绝。
交换bundle时，通过 `load_bundle(path, expected_sha256=...)` 固定外部记录的digest。
使用 `verify_videos` 按camera ID明确绑定移动后的视频；历史视频路径仅是来源记录，不用于自动查找。

`FixedInputBundle` 属于评估侧。其 `baseline_reference` 是用于核对重放的历史预测。
仅把 `inputs` 传给方法；`MethodInput` 不含参考3D、标注、标注缺失状态或store路径。

## 方法接口

实现 `BallMethod.name`、`config` 和 `__call__(inputs) -> Ball3DResult`，连同bundle/code的digest
传给 `run_method` 或 `run_comparison`。runner没有标注参数。LOVO仅提供保留的两个
`ViewInput`，不提供被排除camera的热图或观测。所有方法共享来源运行的固定标定。

每个view包含aligned camera、local标定/side、模型evidence、阈值/gate之后的历史检测点、
接纳的ball点及其记录的重放设置。坐标是原视频像素；热图保留自身H/W网格，候选像素使用
检测器的W−1/H−1约定。保留完整、从0开始的source时间轴，数组只读。

结果包含球场米制 `positions (T,3)`、`valid (T)` 及显式camera顺序的 `used_views (V,T)`。
无效坐标为0，有效点需要至少两个已使用视角。可选字段为PSD协方差 `(T,3,3)`、
更新后的camera、每个camera的秒单位offset和诊断。改变标定仍须保留原球场坐标规范。
camera的frame `t` 对应世界时间 `t/fps + offset`。小数offset的投影仅在相邻有效3D样本间插值，
不跨越缺失也不外推。保留被排除camera的原标定和零offset。

baseline从保存的解码/gate后观测重新计算production consensus、LOESS和三角测量。
`verify_baseline` 单独重解码原生热图并核对历史点、3D和mask，分别记录容差与逐位相等。
`without_loess` 保持相同consensus接纳mask，使用原始已接纳检测坐标，仅隔离LOESS的坐标变化。

## 评分标签

`load_annotations` 读取 `tennis_lab_ball2d_annotations.v1` JSON：

- `clip_id`、有序 `camera_ids`、`num_frames`、`fps`、`source_size_wh`、按camera索引的
  `video_sha256`，以及含 `kind: direct_video_review`、`model`、`effort` 的 `provenance`。
- `observations` 包含 `camera_id`、`frame_index`、`status` 和可选 `uv_px`、`reviewed`、`evidence`。
  状态为 `observed`、`occluded`、`out_of_frame`、`unknown`、`unreviewed`。
  已复核状态需要证据；只有observed有坐标，未列出的记录保持unreviewed。
- 可选 `events` 含 `frame_index`、`kind`（`bounce`/`hit`）和 `evidence`。

拒绝重复、错误source/轴、未复核观测、图像外坐标及检测器/测试fixture来源。
schema记录复核证据，但本身不能证明声称的图像复核确实发生。单元fixture不能充当实际评估标签。

默认前半 `[0,505)` 用于调整，后半 `[505,1010)` 为主要评估，全片仅作参考。
只评分可见observed标签；LOVO仅评分被排除视角。合并每个残差，不平均各折中位数。
输出中位数/p90、严格超过20px的比例、observed/已评分数量与覆盖率、全视角重投影、有效3D率、
逐frame粗大重投影率、相邻位移/加速度、事件附近误差及完整运行时间。
运动stencil不跨越缺失或分段边界。默认诊断阈值为65m/s、500m/s²和事件半径3frame，
会被记录，未在评估后半段调整。

`annotation_oracle` 是明确的评分专用路径，每折仅使用保留视角的标签三角测量。
近似标定和标签噪声使它只能作为几何参考，不能保证数学误差下界。
缺少标签/事件时返回不可用与覆盖率，不制造零值；没有独立3D真值就不能测量独立3D精度。

## 复现与测试

clip专用入口、测试设计和实际证据位于
[`knowledge/runs/run-ball3d-evaluation-foundation-20261103`](../../../knowledge/runs/run-ball3d-evaluation-foundation-20261103/)。
把repo加入 `PYTHONPATH` 后运行 `run.py --help`。`bundle` 和 `compare` 要求显式本地输入与新输出目录。
仅在独立直接复核标签到达后添加 `--annotations`。实际资产、全片解码与重放验证独立于轻量CI fixture。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/evaluation/ball3d \
  tests/unit/tennis_scene/visualization/test_ball3d_comparison.py \
  tests/unit/tennis_scene/pipeline/test_artifact_codec.py \
  tests/integration/tennis_scene/test_ball3d_bundle.py \
  tests/integration/tennis_scene/test_ball3d_media.py
```
