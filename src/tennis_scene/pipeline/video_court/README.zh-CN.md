# 视频共享球场产物

[English](README.md) · [日本語](README.ja.md)

默认仍为 `court_kp.mode=clip`。`video_shared` 按明确指定的dataset、video和camera生成一个公共H。模型候选初始化多帧RGB白线拟合，有限线段端点和中间发球线验证语义对应。从等价解中按残差选择，不平均H系数。[共享几何](../../../tasks/court_detection/geometry/video_shared/README.zh-CN.md)描述算法和单位。

build当前使用固定hash的**冻结模型候选**，不运行新的神经网络推理；调用方负责模型执行。输入缺失时不会切换模型，因此准备过程可仅使用CPU。

在 `court_kp.video_shared` 中明确选择获取方式。

|字段|build|load|
|---|---|---|
|`action`|`build`|`load`|
|`directory`|相对ARTIFACT root的store|同一store|
|`dataset_id`, `video_id`|必须明确范围|必须明确范围|
|`request`, `request_sha256`|相对ARTIFACT root的request及其hash|`null`|
|`artifact_digest`|`null`|准备API返回的digest|
|`rebuild`|显式重算，允许修复损坏对象|`false`|

命令、request JSON和候选字段的唯一完整规格见[英文输入契约](README.md)。必须指定每个camera的候选manifest `{path, sha256}`。clip选择为null表示同一video的全部clip；空、重复、外部或未知ID会失败。采样使用 `floor(numerator*(N-1)/denominator)`，拒绝重复帧。bootstrap为null时使用fit clip中的首、中、末clip。准备过程核对源视频和RGB hash。有限但未通过模型接纳条件的H可以作为初值。

共用API为 `prepare_video_court(config, resolver, code_identity=...)`。可将产物传给每个 `ClipExecution` 的 `standard_definition(..., video_court=artifact)`。orchestrator在单clip及现有dataset入口准备并绑定它。未来批处理CLI的连接属于另一任务；这里不实现CLI整体或模型生命周期。

`run.json`记录bundle及camera digest。load不需要Court checkpoint，不检测、不build、不自动修复。camera内容hash进入clip court node identity，bundle camera引用进入dataset marker identity。改变一个camera会失效其court和下游node，同一实现下独立的人物和球root可复用。既有全局Python fingerprint在代码改变时仍使所有组件cache失效。

公共H不成立、局部漂移不可观测或超过限制时显式失败。一致的3px以上移动产生warning；可信局部H与公共H的平均差超过10px或最大差超过25px时失败。阈值按1920×1080缩放。稀疏诊断帧不能排除帧间移动。不分段，也不替换为局部H。

共享观测使用schema v3，记录应用到frame 0的坐标和视频来源，校正不将它改称直接frame 0观测。固定几何import、direct annotation和旧ball3d reader显式拒绝共享来源。

## 不修改共享H的独立诊断

`drift_build.build_video_drift` 接收原成果物、所有片段的 `ClipSource`、显式的 `frame_plan[camera][clip]` 帧列表、来源记录和 `DriftConfig`，在CPU上准备[法向偏移诊断](../../../tasks/court_detection/geometry/video_drift/README.zh-CN.md)。测量前后验证视频哈希、归属与时间轴，不重新拟合H，结果写入独立存储。调用方在评价前固定帧和排除集合。

加载原H时设 `action=load`，并同时指定 `court_kp.video_shared.drift_directory` 和 `drift_digest`。两者都为null时显式使用原v1内嵌诊断。缺少其中一项、对象损坏或共享H不匹配均报错，不退回旧诊断。多片段计划使用orchestrator的 `prepare_video_court()` 返回的同一诊断成果物及一致配置。

片段缓存包含原camera digest与诊断camera digest；dataset marker也包含诊断引用。receipt和球场观测保留两个digest及每帧判定。仅诊断改变时重新计算对应球场节点及下游，共享H内容哈希不变。原build格式和v1诊断保留用于历史复现。
