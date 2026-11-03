# Top-K网球候选关联

`associate_topk(..., candidate_uv_px=...)` 可接收以保留camera ID为key的显式重新解码候选像素。
检查shape、图像范围和无效slot为0。原始cell、score、mask和heatmap出自保持不变；
省略该可选参数时保持原有计算。

[English](README.md) · [日本語](README.ja.md)

在标准pipeline中选择 `ball_reconstruction.method=topk_multihypothesis`。
默认仍是 `triangulation`。配置位于 `ball_reconstruction.topk`，由 `TopKConfig` 验证。
只用保存的gate前候选，不用RGB、重新推理、dense map、GT或学习参数。
像素阈值按相对1920×1080的对角线缩放，运动单位是m/s、m/s²，缺测限制用source frame。
所有默认值均由Hydra composition提供。评估端也从
`PipelineRuntimeConfig.from_config(composed, bind_inputs=False)` 的结果构造
`TopKMultiHypothesis(runtime.ball_topk, min_frames=runtime.ball_min_frames)`。
配置类型与adapter均不补全缺失的设置。

所有valid且K≤8的候选均参与，没有score阈值。0 score仅在计算log时取1e-8。
枚举全部camera-pair候选组合，以等权DLT三角测量；每个pair还提出按几何和score选择
其他view候选的扩展，同时保留pair-only。要求选中camera的正depth、seed pair射线角≥1°、
重投影阈值、高度[-0.2,20]m及|X/Y|≤40m。按平均平方重投影/sigma²、平均负log score
与省略view成本排序，合并重复index组合后保留 `max_hypotheses` 个。
这是pair种子提案，并不枚举所有view的笛卡尔积。

时间关联使用含速度、加速度、缺测状态的beam动态规划。超过速度上限的链接被禁止。
加速度是速度差除以平均时间间隔，平方成本在 `turn_cost_cap` 截断，使bounce/hit无需
事件标签也只有有限成本。每个(上一个,再前一个)假设对保留最低成本历史，再按
`beam_width` 剪枝。这是近似二阶Viterbi，全局最优路径可能被剪掉；同分按稳定候选和成本顺序。

null支付 `missing_frame_cost`，在 `max_missing_frames` 内按实际frame间隔保留运动历史。
超限后清除历史并支付 `restart_cost`，显式重新捕获。不对缺测或端点插值。
有效frame少于 `ball_reconstruction.min_frames` 时整条轨迹无效。

`AssociationResult` 保存选中index（未使用-1）、raw像素、重投影、假设数、emission cost、
exp(-cost)与重捕获mask。score是未校准相对权重，不是存在概率。
无效3D/像素/重投影填0，有效3D至少使用2view。不进行选后2D LOESS。
optional component保存typed结果和原ball payload；评估adapter调用完全相同的函数。

只复制小候选数组，不复制heatmap/patch。pair提案、假设上限和beam宽度限制计算量。
[基盘renderer](../evaluation/README.zh-CN.md)流式读取视频，optional显示灰色候选排名与
彩色选中square；ring仍表示3D重投影。默认绘制保持不变。

合成契约在 `tests/unit/tennis_scene/ball_association/test_topk.py`；配置、component/codec、
LOVO隔离及main数组回归在 `tests/integration/tennis_scene/test_ball3d_topk.py`。
实clip命令及记录在 `knowledge/runs/run-ball3d-topk-multihypothesis-20261103/`。
