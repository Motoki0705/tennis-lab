# Court/ball bundle adjustment

[English](README.md) · [日本語](README.ja.md)

在标准pipeline中设置 `ball_reconstruction.calibration=court_ball_ba`。
默认 `fixed` 保留原组件、schema及数值输出。可选精化仅用于球路径，人物继续使用原alignment。
球artifact采用schema v2，scene metadata记录球专用camera与时间偏移。

`bundle_adjustment.py::refine_court_ball` 联合优化每camera的7个变量（共同焦距倍率1、
旋转3、camera中心3）、相对时间与每个对应的3D点。固定主点、宽高比、零畸变，保持
`PinholeCamera` 的无畸变像素契约。米制CourtKP14固定空间gauge；所选camera中字典序
第一台的时间偏移为零。camera帧t观测世界时间t/fps + offset_seconds。各LOVO fold仅使用
保留view独立拟合，时间基准随fold变化；不估计被排除camera的校准或时间。

共用raw court类型及历史diagnostics reader位于 `court.py`，评估侧导入相同定义。
仅用原homography选择且valid的KP拟合，其余valid raw KP留作holdout。原最终H已映射回
原template，不能重复施加PROSAC `x_reflected`，只施加已决定的half-turn。四个非共线点为
7变量提供至少8残差，并额外要求数值Jacobian rank=7。不要求raw LINE。holdout仍以历史
检测、点选择和初始校准为条件，不等于独立的人类court真值。

球对应来自标准consensus、LOESS与三角测量的inlier。每个拟合view要求相邻两帧也是
inlier；优化前固定支持集合，不能通过偏移时间删除残差。在整个clip上均匀选择最多
`max_ball_frames` 个对应，要求运动及与基准camera连通。用相邻接受2D观测的分段线性
插值处理fractional time。最终按t-offset重新采样，再执行原三角测量与速度gate。
不外推或跨缺失插值，coverage可能下降；拟合3D点不填补输出缺失。
球artifact与 `SceneResult.ball_uv/ball_vis` 保留原源帧2D像素和mask。
重采样像素/可见性仅用于3D求解与显式 `resampled_uv_px/resampled_visibility` 诊断。

court/ball残差除以像素scale后采用soft-L1。score不作为已校准概率。prior标准差为
log焦距0.05、旋转0.02 rad、中心0.5 m、时间0.5帧，乘以共同的无量纲prior weight。
camera边界为4个prior单位，时间边界由 `max_offset_frames < 1` 限制。scale、prior与CPU预算
在 `ball_reconstruction.bundle_adjustment` 中显式配置。优化失败、退化或支持不足抛出异常。

`CalibrationRefinement` 保存camera顺序、秒单位offset、源拟合帧及solver/支持/边界诊断。
评估adapter通过原 `Ball3DResult` 与typed runner保存。GT、真值mask、视频路径及被排除view
的evidence不进入solver。clip指标、raw KP holdout叠加图和复现脚本记录在 `knowledge/runs/`。
