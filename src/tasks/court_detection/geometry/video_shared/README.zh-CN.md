# 共享球场几何

[English](README.md) · [日本語](README.ja.md)

用于固定相机观测的CPU组件。输入是明确指定的模型初值和原始RGB测量，
不加载检测器、参考单应矩阵、手工点或相机专属遮罩。数据集选择、scene配置、
成果发布及GPU调度由调用方负责。

- `candidates.read_frozen_candidates`验证固定manifest的内容及范围，保留模型采纳标记，
  只应用一次裁剪平移。被拒绝的有限H可作初值；不会补造null H。
- `observations.measure_frame`从白色条带的两侧边缘提出投影网格，并用其他线验证。
  `endpoints.measure_endings`测量有限线段内外。看不到端点不代表线条在这里结束。
- `fit.fit_common_court`对线、clip、frame等权处理，要求底线和发球线的端点证据。
  失败时`SharedFit.matrix`为None；单独的`fitted_matrix`仅用于诊断。优化收敛不足以采纳结果。
- `consensus.merge_measurements`双向核对实际条带，在原始frame/线分组内合并测量。
  不平均H的矩阵系数，也不把重复初值当作新frame。
- `drift.inspect_drift`考虑线的相关不确定性，利用诊断局部H报告警告、失败或不可观测。
  不替换共享H。3/10/25px按1920×1080的对角线缩放；`SharedPaintConfig`和`PaintConfig`
  中的白线测量容差使用原图像素。

线序及球场米坐标遵循`geometry.homography`；像素是未经归一化的原图坐标。
线的身份、空间支持、时间支持及drift分别检查。几何成功不等于独立精度验证。
不可见的线属于外推；静止的白色条带仍可能是球网。

CPU契约测试：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/unit/tasks/court_detection/geometry/test_video_shared.py
```
