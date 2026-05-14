#include <torch/extension.h>

torch::Tensor window_gather_forward_cuda(
    torch::Tensor tensor,
    torch::Tensor indices);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "window_gather_forward",
        &window_gather_forward_cuda,
        "Time-local window gather forward (CUDA)");
}