# 分段物理轨迹与相机相对时间

[English](README.md) · [日本語](README.ja.md)

可选的CPU方法在球场米制坐标中拟合 `dX/dt=V`、`dV/dt=(0,0,-g)-k|V|V`。
每段含初始位置和速度六个参数，全片共享非负阻力系数和相机相对时间。标定保持固定。
选择 `ball_reconstruction.method=segment_physics` 以及
`ball_reconstruction.segment_physics.observation=candidates` 或 `heatmap`。
默认仍为 `triangulation`。参数以 `configs/pipeline.yaml` 为准。
可选组件同时依赖已接受的 `ball_points` 与门控前 `ball_detections` v2，要求模型证据，
拒绝人工标注输入。初始轨迹、场地朝向及对齐保持标准pipeline。

`segments.py` 在baseline三维点左右分别拟合二次曲线，以速度变化阈值和NMS选择事件候选。
低处从下降转为上升标为bounce，其余标为hit候选；它们并非已验证的事件。
长缺失区间导致分段，最大段长产生的切分不计为事件。内部拟合可跨短缺失，输出有效性及使用视角
仍不能超过baseline的真实观测支持。

`fit.py` 使用有界稀疏least squares和soft-L1同时优化全部参数。候选模式选择
距离²/sigma²−2log(score)最小的有效峰，拟合两个坐标残差和分数惩罚。
heatmap模式使用原生图双线性似然相对峰值的−2log平方根，并加入图像外连续惩罚。
每条观测只读取四个cell，不复制大图。RK4积分不使用检测器、GPU、视频或真值。

offset以秒保存，相机帧f观测 `X(f/fps+offset)`。字典序第一台**保留相机**固定为零，
其余offset绝对值小于一帧。拟合排除每段两端各一帧，避免offset查询跨越事件不连续处。
输出仍在帧时刻。独立飞行段可在事件处有位置间隙，间隙及竖直反弹系数会记录，未强制反弹定律。

可选组件在常规 `ball` 字段之外保存fit和诊断。
[评估adapter](../evaluation/ball3d/segment_physics.py)调用同一solver，按基盘格式输出offset。
每次LOVO仅拟合保留相机；基盘评分中被排除相机仍为零offset。
不同fold可能使用不同的相对时间基准，本方法不能辨识绝对同步。

诊断包含分段、事件、图像残差、物理加速度、状态、阻力、offset、收敛状态及信息矩阵秩。
只有满秩时才保存局部曲率近似的offset协方差和标准误；秩不足有明确原因，边界命中会标记。
这些量以分段及候选分配为条件，不是已校准的置信区间。
未收敛返回明确状态及无效输出，不会静默改用baseline。
平坦图、短飞行及标定误差可能导致失败；不处理Magnus、rolling shutter或标定精化。

clip实验、测试设计及实测参见
[`knowledge/runs/run-i1565-segment-physics-offset-20261103`](../../../knowledge/runs/run-i1565-segment-physics-offset-20261103/)。
