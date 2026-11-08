# NHT 公开渲染数据

[English](README.md) · [日本語](README.ja.md)

`NHTRenderClient`仅使用`nht-render`的公开文件。请求/结果 schema 和 CUDA
运行环境由[NHT](../../../../third_party/nht/README.md#standard-renderer)维护。
以下语义针对已采用且启用了 view encoding 的 B00–B03 导出。

`rgb.npy`为采用训练图像编码的 float32 解码 RGB。当前渲染器先累积谐波特征，
再通过学习到的非线性 shader 解码，因此不保证输出为线性光、直通 alpha 或预乘
alpha 的 RGB。`alpha.npy`是单独累积的 Gaussian 不透明度，并非最近物体或球网的遮罩。

`depth.npy`是以 canonical NHT scene 单位表示的 float32 **期望相机 z 深度**：
`sum(w_i * z_i) / sum(w_i)`。它既不是最近表面深度，也不是沿像素射线的距离。
空累积的深度为零。`nht_depth_to_metric`除以显式指定的
`nht_scene_units_per_metre`，仅转换单位，不改变深度统计量的含义。near/far 来自
导出的 runtime config，单位同样为 NHT scene 单位。

这三个数组不足以确定如何把不透明前景插入半透明背景内部。比较前景深度与期望
深度，或把整个背景的 alpha 当作球网不透明度，均不属于支持的遮挡契约。当前公开
请求没有前景深度输入，结果也没有以深度为条件的 RGB 贡献和透射率输出。

拟议契约与实测限制见[合成研究记录](../../../../knowledge/runs/run-player-depth-compositing-20261108/README.md)，
它们尚未实现为公开 schema 扩展。
[C5 的命名近似契约](../../../../knowledge/runs/run-player-depth-compositing-20261108/contract.md#c5-approximation)
同样属于实现提案，不改变上述现有数组的语义。
