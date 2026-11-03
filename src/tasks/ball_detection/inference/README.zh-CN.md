# Ball推理与原生热图解码

[English](README.md) · [日本語](README.ja.md)

`predictor.py` 负责checkpoint推理，`trajectory_gate.py` 负责单点时序筛选。
`heatmap_decoding.py` 在CPU上重新解码已选择的原生sigmoid热图，不执行模型，也不接收真值。

场景设置为 `ball_detection.decoding`。默认值仅由
[pipeline.yaml](../../../tennis_scene/configs/pipeline.yaml)的composition提供，Python配置的每个字段均须显式传入。
比较脚本也compose同一个YAML，只覆盖试验明确指定的差异。默认路径保持predictor数组逐元素相同。
其他方法要求 `ball_detection.subpixel_refine=true`。

| method | 局部估计器 |
|---|---|
| `legacy` | 原来的轴向三点log抛物线 |
| `axis_logit` | logit域的轴向三点二次拟合 |
| `quadratic_log` / `quadratic_logit` | 3×3/5×5含XY交叉项的最小二乘二次曲面 |
| `centroid` | 减去patch最小值后加权的质心 |
| `softargmax` | 对减背景概率的log除以温度后做空间softmax；T=1等于centroid |
| `gaussian` | 5×5背景加各向异性Gaussian非线性拟合；Cholesky保证precision正定 |

```bash
# 与默认pipeline一样，通过共享training queue运行GPU推理。
.venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  'video_paths=[cam0.mp4,cam1.mp4,cam2.mp4]' \
  'camera_ids=[cam0,cam1,cam2]' \
  ball_detection.decoding.method=quadratic_log \
  ball_detection.decoding.patch_size=5
```

坐标为x/(W−1), y/(H−1)。保留原始格点peak score、dense map与top-K含义。
top-K继续使用原predictor的解码器；只有单一观测使用选定方法。
阈值、gate、consensus、LOESS与三角测量复用原组件，完整设置保存在component identity中。

边界patch只使用真实格点。二次拟合要求rank 6、负定Hessian且中心距格点peak不超过1格。
Gaussian中心同样限制在1格内，最多100次函数求值。退化、非凹、中心超界、优化失败或patch不足时，
保留格点并返回 `FitStatus` 原因。轴向拟合保留边界/退化轴，标记部分修正。
scene记录各状态数量，replay结果保存每个camera/frame的状态码。

`coordinate_correction=stride` 使用半像素中心假设 `((u*(N−1)+0.5)*s−0.5)/(N*s−1)`，
不代表已证明学习模型存在此偏差。`affine` 使用显式常数 `u*scale_xy+offset_xy`，结果clip到[0,1]。
基于真值的系数拟合仅属于解码器之外的前半段调参，方法不接收真值数组。

float32 sigmoid饱和后无法恢复丢失的信息。logit拟合将概率clip到[1e−7,1−1e−7]。
边界质心存在截断偏差；Gaussian假设单峰加常数背景。CPU replay每次最多复制64张map，不缓存全部RGB帧。

公共adapter为 `src.tennis_scene.evaluation.ball3d.heatmap_decoding.HeatmapDecodingMethod`，
复用[固定输入runner与评分](../../../tennis_scene/evaluation/README.zh-CN.md)及production组件。
clip专用比较保存在 `knowledge/runs/run-i1559-heatmap-decoding-20261103/`。
