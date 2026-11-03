# 候选GMM球重建

[English](README.md) · [日本語](README.ja.md)

在标准pipeline中设置 `ball_reconstruction.method=candidate_gmm`。
默认仍为 `triangulation`。只有球重建读取 `ball_detections` v2保存的证据；
side/alignment仍使用ball points。计算在CPU执行，无学习型协方差模型。
设置定义于 `CandidateGMMConfig` 和 `configs/pipeline.yaml`。
默认值仅由Hydra的YAML composition提供；Python配置和评估adapter要求显式传入设置，
replay脚本同样遵守此契约。
在 `ball_reconstruction.candidate_gmm` 下选择 `estimator=map|mean` 和
`temporal=none|kalman|rts`。显式提供标定后的 `covariance_scale`、
`presence_slope`、`presence_bias`；production不读取真值。
clip实验及仅前半段的标定位于
[`knowledge/runs/run-ball3d-candidate-gmm-20261103`](../../../knowledge/runs/run-ball3d-candidate-gmm-20261103/)。

## 分布与mask

每camera/frame按score检查候选，保留最先满足3×3 log概率二次拟合满秩且precision
正定的K=2或3个候选。保留交叉曲率项，subpixel最优点限制在原cell内。
协方差特征值由 `min_variance_cells2` / `max_variance_cells2` 显式限制并记录。
边缘支持不足、平坦图和鞍点会被拒绝，不替换为各向同性协方差。
均值和完整协方差使用 `(W-1)/(w-1)`、`(H-1)/(h-1)` 转为原始像素。
不会复制完整dense heatmap。

成分权重为归一化peak score。支持概率代理为
`sigmoid(presence_slope*logit(max retained score)+presence_bias)`，score端点
限制在[1e-7,1-1e-7]。无可用peak时为0。它表示可用候选支持的代理，无法由这些
标签识别真实球存在或独立的遮挡概率。

调用现有[方式A](../../utils/geometry/probabilistic_triangulation/README.md)。
必须提供米制球场prior：`prior_mean_m` 默认[0,0,2]，`prior_std_m` 默认[8,16,6]，
对角协方差。最坏枚举量为 `(K+1)^V`，超预算在求解前报错。
优化失败使整个frame成为带原因的缺测，不丢弃单个失败成分或切换solver。
契约及预算错误直接抛异常。

仅输出至少两个active camera条件下的后验，其原始概率质量须达到
`min_multiview_mass`，并保存该质量。prior-only和单眼项不算三角测量观测。
`map` 为最大权重成分的MAP点，不是整个混合分布的mode；`mean` 使用包含成分间
方差的混合moments。MAP协方差只描述选中成分。MAP的使用view为其subset，
mean使用正权重subset的并集。

共用等速度Kalman/RTS采用秒、米及显式加速度/初速度尺度，在缺测处重置。
保持valid及used-view mask，不输出填补缺口的预测或推断事件标签。
仍可能过度平滑反弹和击球。

## 输出与评估

`CandidateGMMOutput` 保留通常的 `ball`，另存MAP/mean后验、选中协方差、
支持质量、候选index/均值/完整协方差/权重、曲率诊断和solver失败。
2D诊断点为最接近输出投影的保留候选均值；其距离不作为额外gate。
scene export使用 `ball`，详细协方差保留在component artifact和共用评估结果中。

评估adapter调用相同production算法，输入不含真值、真值mask或held-out view。
每fold仅用保留view的[0,505)标定，在主评估前冻结；20px内局部候选用于协方差尺度，
可评分前半点用于score函数。共用评估报告全view/LOVO误差及可用性coverage。
`score_uncertainty` 复用其评分camera/frame和来源验证，按区间/fold/参考类型
统计50/90/95%椭圆覆盖率。pinhole Jacobian只传播posterior协方差，未纳入参考噪声、
标定误差、非线性尾部或mode选择错误。非零时间offset明确拒绝。
椭圆覆盖率条件于已评分预测，不等同于可用性覆盖率。
共用renderer在有协方差时为camera/top/side添加95%椭圆，HTML可嵌入补充图。
合成unit/integration验证曲率、camera顺序、缺测、标定隔离、投影、平滑、配置/codec
及与main固定数组完全一致的默认行为。
