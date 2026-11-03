# 固定输入的网球3D评估

[English](README.md) · [日本語](README.ja.md)

`ball3d/` 使用相同的历史component，在CPU上重放各方法，并用用户提供的2D参考评分。
它不改变标准pipeline的默认行为，也不执行检测器推理。

| 模块 | 契约 |
|---|---|
| `bundle.py` | 提取并验证固定输入、原identity/source哈希和完整依赖descriptor |
| `contracts.py` | 仅含选定视角的方法输入与带mask的3D结果 |
| `baseline.py` | 从保存的检测观测重放consensus、LOESS、三角测量，并单独核对原始热图解码 |
| `runner.py` | 全视角及三个双视角LOVO折，带类型和checksum的保存/加载 |
| `annotations.py` | 将公共reader选择结果转换为评分专用数组 |
| `scoring.py` | 分段重投影、覆盖率、运动/事件诊断及标注几何参考 |
| [比较可视化](../visualization/ball3d/) | N种方法的RGB叠加、俯视/侧视轨迹、图表、裁剪图和可携带HTML |

`segment_physics.py` 使用保留视角的初始轨迹运行可选的 [分段物理与相对时钟solver](../ball_trajectory/README.zh-CN.md)，并输出相同结果格式。

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

可选的 [`TopKMultiHypothesis`](ball3d/topk.py) adapter调用production的
[top-K关联](../ball_association/README.zh-CN.md)，没有额外的loader或评分逻辑。

可选[候选GMM方法](../ball_gmm/README.zh-CN.md)调用production方式A。
`ball3d/uncertainty.py` 在相同评分camera/frame集合上评估投影协方差与椭圆覆盖率。
非零时间offset需要时间联合后验，因此明确拒绝。

实现 `BallMethod.name`、`config`、bool `requires_court_lines` 和 `__call__(inputs) -> Ball3DResult`，连同bundle/code的digest
传给 `run_method` 或 `run_comparison`。runner没有标注参数。LOVO仅提供保留的两个
`ViewInput`，不提供被排除camera的热图或观测。所有方法共享来源运行的固定标定。

每个view包含aligned camera、local标定/side、模型evidence、阈值/gate之后的历史检测点、
接纳的ball点及其记录的重放设置。坐标是原视频像素；热图保留自身H/W网格，候选像素使用
检测器的W−1/H−1约定。保留完整、从0开始的source时间轴，数组只读。

`ViewInput.court` 按camera保留raw KP14像素、score/valid、原homography的使用点和模板方向，
只记录实际测量frame（本次为0），不把fitted/broadcast坐标当作raw观测。
验证camera/frame/native尺寸和字段缺失；LOVO仅传入选定view的court对象。
现有bundle的已验证标定diagnostics包含这些raw KP，原bytes/digest保持不变。
LINE有类型化表示，但历史store只保存support摘要。`court.lines=None` 附有明确缺失原因。
每种方法必须声明bool `requires_court_lines`；为true时，runner在调用方法前拒绝选定view中缺失的LINE。
也可调用 `court.require_lines()` 或 `load_bundle(..., require_court_lines=True)` 显式要求LINE。
baseline声明false，使用固定camera；不把摘要转为观测，也不重新推理。

结果包含球场米制 `positions (T,3)`、`valid (T)` 及显式camera顺序的 `used_views (V,T)`。
无效坐标为0，有效点需要至少两个已使用视角。可选字段为PSD协方差 `(T,3,3)`、
更新后的camera、每个camera的秒单位offset和诊断。改变标定仍须保留原球场坐标规范。
camera的frame `t` 对应世界时间 `t/fps + offset`。小数offset的投影仅在相邻有效3D样本间插值，
不跨越缺失也不外推。保留被排除camera的原标定和零offset。

baseline从保存的解码/gate后观测重新计算production consensus、LOESS和三角测量。
`verify_baseline` 单独重解码原生热图并核对历史点、3D和mask，分别记录容差与逐位相等。
`without_loess` 保持相同consensus接纳mask，使用原始已接纳检测坐标，仅隔离LOESS的坐标变化。

## 评分标签

`annotations.py` 是严格的 [user-reference reader](../ground_truth/README.md) 的轻量数组适配层。
`load_annotations(path, bundle, expected_sha256=...)` 从已验证的bundle和视频哈希构造
`ReferenceIdentity`，调用 `load_user_reference`，并通过 `UserReference.select` 明确选择camera/frame轴。
不维护第二套JSON schema或来源验证。原始CSV必须与参考JSON一起保留。

评分仅使用 `valid_for_scoring`。用户批准的 `estimated` 点参与评分，并保留其类别，
不得改名为 `observed`。缺失和排除的坐标保持NaN，不做插值。
每个fold和时间段均报告 `all`、`observed`、`estimated` 的分布、可评分/已评分数量和覆盖率。
保留按camera记录的全部事件原行、注释和估计区间。

`score_comparison(runs, bundle, annotations)` 和 `error_samples(run, bundle, annotations)`
必须接收已验证的 `FixedInputBundle`，核对结果bundle digest和camera/frame轴，以及参考的
clip ID、camera、frame数、fps、native尺寸和视频哈希。`annotation_oracle`、`plot_comparison`、
`render_extremes` 同样接收bundle，拒绝混入尺寸相同的另一输入结果。

默认时间段为调参 `[0,505)`、主要评估 `[505,1010)` 和补充的全片段。
LOVO只评估被排除camera，汇总每个残差，而不平均各fold的中位数。
指标包括中位数/p90、严格大于20px的比例、全视角误差、有效3D比例、粗大重投影frame比例、
相邻位移/加速度及异常比例、事件窗口误差和完整运行CPU时间。
粗大frame比例的分母是可评分frame，分子是至少一个残差大于20px的frame；没有独立的3D真值。
运动计算不跨缺失frame或时间段边界。默认诊断阈值65m/s、500m/s²、事件半径3frame被记录，未在后半段调参。

事件指标使用每种事件的原camera行窗口的并集，即使同一事件的记录相差1frame，也只计一次残差。
`source_row_count` 不表示物理事件数量；`physical_event_count` 保持null，不推测事件对应关系。

`paired_error_summary(reference, candidate, bundle, annotations, ...)` 使用与 `error_samples`
相同的bundle/结果/参考identity校验。两组的交集与全方法共同LOVO cohort不同。
它按全fold或被排除camera，比较两组LOVO残差中相同camera/frame的交集。
保留各组评分数量并报告参考减比较的差值；coverage不同时，它是不可相加的诊断。
clip专用误差分解保存在
[`knowledge/runs/run-ball3d-error-decomposition-20261103`](../../../knowledge/runs/run-ball3d-error-decomposition-20261103/)。

`annotation_oracle` 仅供评分，在同一固定camera下对保留camera的可评分参考（含估计）做三角测量，
不会调用预测方法。近似标定和参考噪声使它成为几何参考，而不是数学上保证的误差下限。

HTML主表和独立指标图使用同一表格数据，在LOVO误差旁显示已评分/可评分数及coverage，
全视角有效3D比例使用独立标题。第二张表仅比较所有列出方法（包括所提供的oracle）共同可评分的
(排除camera, 原frame)交集。表格附近说明各方法评分集合不同，以及交集比较仍以可用性为条件。
保存交集identity和观测/估计分类；交集为空时误差不可用，不填0。

可视化支持N种方法：绿色十字表示直接观测，白色菱形表示批准的估计，彩色圆表示预测重投影。
裁剪图按LOVO误差选择，并用彩色斜十字显示LOVO预测。轨迹不跨缺失frame连接。
HTML嵌入视频、图、表和指标，支持单文件查看；相邻文件仅用于可选下载。
复用通用视频reader和frame选择。`MultiviewSceneRenderer` 要求SMPL/SceneResult，
因此仅比较ball时单独组合俯视/侧视面板，不生成虚构的人体mesh。

## 复现与测试

clip专用入口、测试设计和实际证据位于
[`knowledge/runs/run-ball3d-foundation-review-20261103`](../../../knowledge/runs/run-ball3d-foundation-review-20261103/)。
把repo加入 `PYTHONPATH` 后运行 `run.py --help`。`bundle` 和 `compare` 要求显式本地输入与新输出目录。
compare需要可信记录中的 `--bundle-sha256`、`--annotations` 和 `--annotations-sha256`，以及核对历史权重的 `--checkpoints`。修正后的复现还需要 `--prior-comparison`，核对预测数组与v1一致。实际资产、全片解码与重放验证独立于轻量CI fixture。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/evaluation/ball3d \
  tests/unit/tennis_scene/visualization/test_ball3d_comparison.py \
  tests/unit/tennis_scene/pipeline/test_artifact_codec.py \
  tests/integration/tennis_scene/test_ball3d_bundle.py \
  tests/integration/tennis_scene/test_ball3d_media.py
```
