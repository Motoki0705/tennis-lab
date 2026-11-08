# 场景工作流中的空白球标注

[English](README.md) · [日本語](README.ja.md)

`run_pipeline` 默认使用 `ball_detection.method=model`。选择 `gpt_annotation` 后，
工作流连接原始视频注册、无预测标注、超时后自动继续、正式重读已完成的v3标注、固定几何重建，
以及可选的active-ball数据集发布。此路径不提交额外训练；人物和身体功能遵循各自的设置。

## 固定请求

```bash
.venv/bin/python -m src.tasks.annotation schema --type scene_annotation
```

`SceneAnnotationRequest` 包含：

- `configuration`：[本地视频配置](../../../tasks/annotation/README.md#ローカル動画の白紙入力登録)，
  包括显式执行计划、预算scope的新建/复用、worker与服务成本、每个相机的原视频路径和
  `VideoAnnotationInput`，以及 `publish_dataset`。
- `policy`：具有ID、版本和探索指示的 `AnnotationPolicy`。
- `publication`：发布时的 `ProductPlan`，否则为 `null`。生成策略必须使用
  `target_semantics=active_ball`、原Q90 JPEG，以及按原始录制分配的显式split或分割recipe。
  可以向兼容的active-ball snapshot追加数据。

同一录制的所有相机和相关clip共享recording ID。场景请求为每个相机指定一个原视频，
从源frame 0开始覆盖场景的全部frame。按注册契约固定原始bytes、分辨率、实际解码PTS、
time base和FPS。显式 `max_frames` 也必须匹配标注scope。预算和worker成本由操作方根据
实际输入确定；不能将fixture预算或test clock用于生产。

```python
from src.tasks.annotation.scene_workflow import SceneAnnotationRequest
from src.tasks.base.annotation.artifacts import write_document
from src.tasks.base.annotation.paths import AnnotationLayout

layout = AnnotationLayout.production()
# 按上述类型和输入契约准备这些对象。不发布时output_product为None。
request = SceneAnnotationRequest(configuration=local_configuration,
    policy=exploration_policy, publication=output_product)
reference = write_document(layout,
    layout.path("campaigns", "scene-inputs", "request.json"), request)
```

请求在campaign内不可变。注册拒绝视频identity、时间轴、资源scope和发布选择的变化。
探索策略在worker准入前固定，并在多个attempt之间保持一致。

## 运行完整工作流

视频路径相对于DATA，request/bundle相对于ARTIFACT。两个digest参数使用保存的引用。
`clip_id` 必须匹配固定bundle manifest中的clip ID。

```bash
.venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  'video_paths=[samples/cam0.mp4,samples/cam1.mp4,samples/cam2.mp4]' \
  'camera_ids=[cam0,cam1,cam2]' 'clip_id=meiji_3cam/video_000/clip_000' \
  ball_detection.method=gpt_annotation ball_detection.checkpoint=null \
  direct_annotation.request=chat_annotation/campaigns/scene-inputs/request.json \
  direct_annotation.request_sha256="$REQUEST_SHA256" \
  direct_annotation.bundle=tennis_scene/evaluate/ball3d_comparison/clip000_20261103/fixed_input_bundle_v1/bundle.json \
  direct_annotation.bundle_sha256="$BUNDLE_SHA256" \
  direct_annotation.court_detection=verified_calibration_frame0 \
  direct_annotation.publish_dataset=true
```

指定clip000 bundle省略了court数组，因此必须显式选择 `verified_calibration_frame0`。
其他bundle默认使用 `stored_artifact`。court检测、校准、方向和相机布局成为load-only外部输入；
[import契约](../imports/README.zh-CN.md)验证身份对应和来源。这些输入和ball均不需要checkpoint。
GPU功能仍遵循repo的training queue规则。CPU球单独运行使用
`device=cpu person_observations.enabled=false player_reconstruction.enabled=false gvhmr.enabled=false`。

## 收据与恢复

component store的 `run.json` 分别报告 `annotation`、`scene`、`dataset` 和 `learning`。
scene metadata保留原有 `ok`/`partial` 含义。每相机完成引用和测量引用保存到campaign下的
`scene-workflow/annotations.json`；场景artifact和archive保存到 `scene-completion.json`。
可选发布使用 `scene-workflow/publication-request.json` 和 `publication.json`。
`dataset=not_requested` 视为成功；`learning=not_requested` 不启动训练。

正常超时会自动继续。无进展、quota、保存损坏或外部服务故障需要
[local controller恢复](../../../tasks/annotation/README.md#白紙注釈の自動継続)。
统一标注CLI的 `run`、`tick`、`status`、`recover`、`resume-budget`、`report`
接受保存的 `LocalAnnotationRuntime` 及其digest。

仅发布失败时，给原场景命令添加 `direct_annotation.resume=publication`，保持request、几何和store一致。
该操作验证已完成的标注和scene export，然后继续publisher，不构建scene stages或准入worker。
publisher提交之后、workflow收据保存之前的中断也会回到原snapshot。新发布请求不覆盖已保存标注。

## 探索策略比较与测量

使用 `blank-register --operation PATH --sha256 VALUE` 提交 `BlankConditionRegistration`
（source plan、独立condition ID、带版本policy）。CLI schema为 `blank_condition` 和
`blank_comparison`。将返回的 `BlankCondition` 中的runtime交给
`run --runtime PATH --sha256 VALUE --max-seconds N`，只在benchmark namespace中处理对应相机输入。
复用共享图像和预算；进度与revision相互隔离。plan将发布固定为false。

`blank-compare --operation PATH --sha256 VALUE` 接受 `BlankComparisonRequest`，用condition引用
以及可选的父进程专用参考标注保存不可变report。生产完成时同样保存 `BlankRunMeasurements`，
关联policy/run、attempt、图像请求和重访、坐标coverage、响应usage，以及指定参考时的像素误差。
缺失token usage保持null。区分synthetic、real、mixed和未启动worker，保持 `acceptance=unverified`。
比较run不能进入scene发布composition，也不会自动发布常规数据集。

## 汇总与后续验收

`annotation-summary.json` 保存整体和每相机计数、状态分布、unknown/未处理的半开区间。
`summarize_video_annotations` 使用最终状态；坐标不可用指已处理但没有可用坐标的行，
包括无坐标遮挡、画外和no_active_ball。未处理单独计数。
`summarize_reference` 使用最终 `reference_kind`，不会把原 `source_status=unknown`
但已采用推定坐标的点计入最终unknown。

`clip000_acceptance(report)` 仅对cam0/cam1/cam2的frame 0–1009全部3030项应用总计限制：
不可用≤81、最终unknown≤25、未处理=0。小fixture返回 `not_applicable`。
工作流保存计数，不自动增加识别质量gate。

代表性CPU E2E使用3相机×4frame合成视频和固定几何。仅模型响应和5400秒时间边界为fake；
注册、服务、revision保存、重读、几何、scene export、publisher和dataset reader均实际执行。
验证推定点贡献及少于两个可用view的缺测。integration验证不发布/发布及workflow发布收据前中断。
这些结果不能证明实际GPT识别性能。

后续#1714需要旧标注、原视频/frame/PTS identity、录制group/split和迁移来源，不能捏造v3状态。
后续#1715需要三个原视频、固定plan/policy/预算、clip000固定bundle及引用，以及父进程专用的
`human_annotations/ground_truth_v1/ground_truth.json`。比较探索策略后选择生产policy，保留全部3030项判断、
汇总、新3D数组/mask/来源、snapshot与重读结果、overlay。本次未执行真实标注、迁移、识别验收或模型训练。
