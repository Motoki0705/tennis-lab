# 单摄像机空白标注

[English](README.md) · [日本語](README.ja.md)

`BallVideoWorker` 接收已注册的BlankAnnotationContext，实现 `ball_video_annotation / 3 / review`。
不使用预测receipt、overlay或confirm_predictions。状态及来源见 [v3规范](../README.md)。

## 工具与探索策略

[`instructions.md`](instructions.md) 定义唯一比赛有效球、来源保留、减少unknown及工具协议。
AnnotationPolicy提供探索策略。时间序列总览、任意参考点、crop/放大、推断和自我修正使用相同工具，不固定浏览顺序。
AnnotationRun固定context、plan、runtime、策略和tool/worker版本。model、effort、代码及资源由plan固定，
worker/service条件由runtime固定；后续attempt不能替换这些条件。

`camera-tools.v2` 要求handoff、rows和image带有 `read_id`。client每次调用自动生成一个，
编辑后重新读取相同范围会返回当前revision及进度。使用原 `read_id` 和parameters重发相同请求时，
返回原snapshot。编辑仍使用 `expected_revision` 的CAS契约。

父服务验证camera、原frame/PTS、crop及所有权后返回plain图像。frame参数始终使用**原视频frame编号**，
包括非零起点。只能请求或引用提供区间内的同camera frame。receipt保留native crop、缩放比例及原PTS。
运行边界见 [private input boundary](../../../../base/annotation/workers/README.md#private-input-workers)。

## 父进程集成与持久进度

将保存的LocalAnnotationPlan及BallVideoWorker交给WorkerPool，Codex binary/home、review timeout、
scope/资源预留和code revision采用runtime固定值。production仍需共用ResourceManager。ServiceBroker接收inbox，
Operations在CPU render stage执行ReviewOperation。空白plan仅允许camera service；比较使用benchmark-stage。
绕过仅依赖stdlib的input/tools.py也不能绕过父服务验证。

每次video_edit验证完整v3与固定输入，写入不可变payload/command/revision，再以SQLite CAS更新head后回复。
相同重发复用同一revision；即使最终响应中断，也保留已commit进度、旧标注及图像/tool来源。

```python
from src.tasks.ball_detection.annotation.video.agent import VideoRevisionStore
from src.tasks.ball_detection.annotation.video.usage import is_unprocessed

store = VideoRevisionStore(layout, task.run)  # task: saved AgentTask
revision, completion = store.current()
annotation = store.annotation(revision)
remaining = tuple(row.frame_index for row in annotation.frames if is_unprocessed(row))
```

video_handoff分页返回未处理frame，不包含最终unknown。后续attempt的initial_revision为保存的head。
超时自动继续属于 #1709；本模块不调度attempt或自动重试。

最终AgentDecision指定当前revision。机器验证worker/rollout identity、语法、输入/frame/PTS、状态及来源。
推断frame不需要独立目视或人工QA。未处理数为0即可保留unknown并完成。WorkOutcome.adoption指向VideoCompletion，
其中annotation为scene/dataset使用的确切v3 artifact。completion发布、seal及最终receipt中断均复用原完成记录；
dataset发布是独立步骤。

## 探索策略比较

[空白benchmark API](../../../../annotation/benchmark/blank.py) 复用现有namespace和共用telemetry的usage汇总。

```python
from src.tasks.annotation.benchmark.blank import (
    compare_blank_conditions,
    register_blank_condition,
)

condition_a = register_blank_condition(layout, source_plan, "overview", policy_a)
condition_b = register_blank_condition(layout, source_plan, "adaptive", policy_b)
# Run each with its saved plan/runtime/policy and benchmark layout.
report = compare_blank_conditions(
    layout, (condition_a, condition_b), reference_annotations=references
)
```

condition共用图像和frame identity，worker/revision账本独立。固定phase=benchmark并禁止发布，不改变正常dataset
或进度。参考标注仅供父进程评分，绝不进入worker输入。

报告关联attempt/thread/响应ID、input/output/cache token、revision与已处理/unknown/可用坐标、图像请求/展示/
重看及native pixel误差。请求统计已commit图像service；请求重看统计重复原frame的tile；响应重看统计重复展示的
原frame。误差只计算双方有坐标的配对，保留配对数及camera/原frame/PTS结果。缺失usage或参考坐标保持null，
已知小计单独保留。共用telemetry去重usage。区分synthetic、real、mixed和未启动，acceptance为unverified。
测试使用合成视频、真实文件/进程、fake worker及合成usage；GPT准确率、token节省和真实视频验收留待 #1715。
