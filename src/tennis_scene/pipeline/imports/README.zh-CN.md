# 已保存的场景输入

[English](README.md) · [日本語](README.ja.md)

`BallAnnotationImporter`读取任务保存的v3标注；`FixedGeometryImporter`将已保存的
球场检测、标定、朝向和相机配置注册为固定外部输入。两者通过`bind_import` /
`publish_import`发布新的不可变场景产物，不启动GPT、模型、标定或训练数据发布。
工作流与CLI的组合由调用方负责。

## 契约

| 输入 | 场景产物schema | 含义 |
|---|---|---|
| 每台相机保存的v3 | `ball_annotations` v1 | 使用任务的reader和使用mask |
| 标注点传递 | `ball_annotation_points` v1 | 保持坐标，独立保留位置可用、直接观测和估计mask |
| 标注三角测量 | `ball_annotation_trajectory` v1 | 新3D、贡献view、缺失原因与标注来源 |
| 固定bundle | 现有几何输出schema | 原依赖作为来源保存的外部输入 |

[任务的v3契约、reader和usage API](../../../tasks/ball_detection/annotation/video/README.md)
是唯一依据，场景层不复制状态表。传入视频注册时固定的`VideoAnnotationInput`，
不得从标注文档自身重新提取期望输入。导入会核对源视频bytes的identity、相机、图像大小、FPS、
clip的完整frame轴和实际PTS。标注必须覆盖`[0, source.num_frames)`，源视频本身可以更长。
拒绝未处理行；已处理的`unknown`以缺失位置保留。
实际bytes与解码PTS由视频注册阶段验证；导入阶段不解码视频。

`BallAnnotationOutput`保存任务文档、原path与digest、原PTS、native pixel、任务点类型字符串
（无位置时为空），以及独立的`position_usable`、`observed`、`estimated`。
confidence是位置接受值0/1，不是模型概率。support、遮挡、连续性断点、notes和有效比赛球
的目标定义保留在任务文档中。保存产物的派生字段也与任务API核对。

## 几何与点的策略

观测与估计都使用权重**1**，不改变标注坐标。对全部可用view执行DLT与pixel重投影
最小二乘优化。**不平滑、不限制插值时长、不按重投影误差筛选view、不设置速度/高度边界，
也不按轨迹长度过滤**。模型/v2路径继续保持原有的仅观测点LOESS、consensus、重建策略和schema。

少于两个位置可用view时，保存零坐标、缺失mask、`missing_reasons="insufficient_views"`
及rejection code 1。几何求解失败（包括退化射线或非正深度）保存为
`invalid_multiview_geometry`及通用solver失败code 7。缺失点没有贡献view。
`position_usable`记录可用输入（包括单view时刻）；`used_views` / `trajectory.inliers`
记录对有效3D的贡献。有效点使用全部可用view，不静默删除输入点。重投影误差仅用于诊断。
3D产物也保存每个view的mask、点类型、原标注文档/引用与策略。

## 固定bundle与用法

使用包含`bundle.json`和`scene.json`的固定`tennis_lab_ball3d_fixed_inputs.v1`。
格式源自[evaluation bundle reader](../../evaluation/ball3d/bundle.py)。
必须显式提供期望的manifest digest。发布前检查文件/数组checksum、源identity、clip、
相机顺序、标定view的source index、球场frame、参考相机和朝向/配置的一致性。
源视频path可以跨机器迁移，bytes、相机identity和时间轴不能改变。
请求的clip使用manifest的clip ID核对；原scene的clip ID和path继续作为历史metadata保存。

`bind_nodes`要求几何node已声明load，并将其改为没有当前依赖的外部输入。
不会将旧side/alignment重新绑定到当前ball点。新descriptor保存bundle引用、原artifact引用、
原identity、provenance、依赖和历史代码来源。原bundle只读，目标store必须独立。
不导入旧ball_points或旧3D；目标store已有旧产物时，也按新的标注schema重新生成。

默认`court_detection="stored_artifact"`要求原球场数组。指定的比较bundle省略了这些数组，
但保留descriptor和标定中的静态broadcast。显式选择
`court_detection="verified_calibration_frame0"`后，只有从frame 0恢复的NPY之shape、dtype、
bytes都与原数组checksum一致才接受。否则失败，不自动fallback或重新推理球场。

```python
from src.tennis_scene.pipeline.imports import BallAnnotationImporter, FixedGeometryImporter
from src.tennis_scene.pipeline.runner import ComponentRunner

# source/store/nodes由调用方的clip recipe构建。
# 几何与ball_detection node必须预先声明load。
fixed = FixedGeometryImporter(bundle_json, pinned_bundle_digest,
                             court_detection="verified_calibration_frame0")
balls = BallAnnotationImporter(annotation_paths, registered_inputs)
nodes = fixed.bind_nodes(nodes)
nodes = balls.bind_nodes(nodes)
fixed.publish(nodes, store, source)
balls.publish(nodes, store, source)
runner = ComponentRunner(nodes, store)
runner.run(targets=["ball_triangulation"])
result = runner.output("ball_triangulation")
```

绑定还固定期望的import identity。改变bundle或标注revision后，加载旧采用artifact会失败；
应先导入指定revision。最终scene assembly接受公共ball输出base契约，binder声明标注trajectory端口。

## 旧v2 import

`ball_annotations.py::import_ball_annotations`仍是[component qualification](../../../../tests/benchmarks/component_pipeline.py)
使用的临时v2导入。读取`<directory>/<camera>_annotations.json`，保存数字点类型，
只有observed点作为真实测量。不替换为v3使用语义，也不转换旧标注。
导入记录`origin=import`，node声明的上游artifact依赖仍然必须存在。

## 验证

```bash
.venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/pipeline/test_imports.py \
  tests/unit/tennis_scene/pipeline/test_video_ball_imports.py \
  tests/unit/tennis_scene/pipeline/components/test_ball_points.py \
  tests/integration/tennis_scene/test_ball_annotation_imports.py
```

用小型非对称相机与独立投影的已知3D验证保存v3、估计点贡献、缺失、保存恢复、无重新执行的
固定几何、输入/数组篡改拒绝，以及含误差view的等权重拟合。它们不证明真实视频识别精度或验收通过。
