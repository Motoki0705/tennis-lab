# 官方 MSDA 执行边界

`OfficialMSDeformAttn` 使用官方 Deformable DETR Python module 和 CUDA autograd
function。来源及仅限导入的修改见 [origin.json](vendor/origin.json) 和
[NOTICE](vendor/NOTICE.md)。CUDA 使用现有 DINO 扩展；实验 build manifest 必须记录
与官方 CUDA/C++ 源码的对比和兼容补丁。

构建与选择二进制文件见 [安装说明](../../../../../../docs/installation/README.zh-CN.md#dino-cuda-extension)。
支持在 CPU 上构造和加载权重。执行必须明确设置 `TENNIS_LAB_DINO_EXTENSION`，使用
CUDA、64维 attention head、有效 shapes/starts，并满足 batch/im2col 整除条件。
不提供 CPU 或 reference 回退。

FP32/BF16 输入显式转换为连续 FP32；官方 Value 投影、sampling 和输出投影在禁用
autocast 后运行。输出为 FP32，转换保留梯度。这不是 BF16 native MSDA。执行边界拒绝
FP16/double；不变的底层官方函数仍可用于 double 精度数值梯度验证。

显式禁用此边界的编译，调用方必须允许并记录 graph break。
vendor 中的 PyTorch reference 仅供验证。
