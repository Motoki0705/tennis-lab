# Dense heatmap似然

[English](README.md) · [日本語](README.ja.md)

该可选CPU方法从保存的模型heatmap直接估计球的球场米制3D位置。
默认仍为 `triangulation`，通过 `ball_reconstruction.method=dense_likelihood` 选择新方法。
参数和默认值位于 [`pipeline.yaml`](../configs/pipeline.yaml) 的 `ball_reconstruction.dense_likelihood`。
[`DenseLikelihoodConfig`](config.py) 要求显式传入全部值，并验证组合后的配置。

新组件读取门控前的 `ball_detections` v2；side/alignment仍读取原来的 `ball_points`。
3D估计不使用2D LOESS、检测接受mask、标注或标注事件。
评估adapter调用相同的 `reconstruct_dense`，并使用[固定输入runner和评分契约](../evaluation/README.md)。

## 似然与插值

sigmoid概率先以FP32 epsilon的一半（约5.96e-8）对称截断，再转换为有限logit。
诊断保存恰好等于1的像素数量；该处理不能恢复饱和丢失的曲率。
`quadratic_log` 使用log概率，因此温度和支持drop使用该域的单位。

`bicubic_logit` 使用logit的张量积三次样条。
`quadratic_logit` / `quadratic_log` 对最近格点的真实3×3邻域拟合含xy交叉项的二次曲面；
边缘只使用真实的单侧样本。邻域切换可能产生不连续，线搜索失败时输出缺失，不返回初值。
本方法没有双线性插值。

插值值f、温度T、native像素数N对应的每view目标为
`log(epsilon + (1-epsilon) * exp(f/T - logsumexp(grid/T) + log(N)))`。
这是信号/均匀背景混合密度相对于均匀密度的对数比；最大化所有view的总和。
画面外或相机后的点贡献 `log(epsilon)` 和零梯度。平坦或低峰map不提供位置支持。
错误view的惩罚有界；未校准sigmoid分数不被视为存在概率。

`view_temperatures` / `view_backgrounds` 为 `[camera_id, value]` 序列，保持LOVO中的身份对应；
未指定相机使用global参数。有效点至少需要两个信息充分、在画面内、距native峰值不超过
`support_logit_drop * temperature` 的view，以及非退化射线角度。

## 搜索与时间

`initialization=candidates` 三角化所有有效候选pair，以dense目标排序，再从有限个分离初值连续优化；
最终位置不限制在候选坐标。`initialization=coarse` 不读取候选坐标，而从球场volume格点及其细分邻域搜索。
两者均不保证全局最优。范围和预算明确；粗网格超过200000点即拒绝。

`temporal_weight>0` 对已有观测支持的frame窗与m/s²单位的3D加速度Cauchy prior联合优化。
输入3D加速度变化点和缺失均切断窗口，不填补缺失。鲁棒prior限制漏检冲击的影响，但噪声可能导致过分割。
窗口互不重叠且不超过 `window_size`。不拟合物理弹道或同步。
不收敛时输出带原因代码的缺失，不自动退回独立估计。

只对单frame或短窗口进行log变换，不复制整段heatmap。volume、梯度和插值工作区由参数预算限制。

## 结果与验证

`DenseTrajectory` 保存完整source时间轴上的float32位置、有效mask、使用view、对数似然、变化边界、
收敛、饱和数量和拒绝代码。无效位置/使用view为0/false。
标准组件将诊断与现有ball轨迹一起保存；2D位置是dense 3D mode在支持map中的投影，不是原检测峰值。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/ball_likelihood \
  tests/integration/tennis_scene/test_ball3d_dense_likelihood.py
```

clip专用调参、命令、图和结论位于
[`knowledge/runs/run-ball3d-dense-likelihood-20261103`](../../../knowledge/runs/run-ball3d-dense-likelihood-20261103/)。
调参结果不改变默认配置；采用决定等待人工综合比较。
