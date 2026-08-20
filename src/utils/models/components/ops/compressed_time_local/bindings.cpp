#include <torch/extension.h>

#include <c10/util/Optional.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> compressed_time_local_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor query_valid,
    torch::Tensor key_valid,
    c10::optional<torch::Tensor> query_phasors_real,
    c10::optional<torch::Tensor> key_phasors_real,
    int64_t compression_ratio,
    int64_t window_radius);

std::vector<torch::Tensor> compressed_time_local_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor query_valid,
    torch::Tensor key_valid,
    torch::Tensor logsumexp,
    c10::optional<torch::Tensor> query_phasors_real,
    c10::optional<torch::Tensor> key_phasors_real,
    int64_t compression_ratio,
    int64_t window_radius);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "forward",
        &compressed_time_local_forward_cuda,
        "Compressed time-local attention forward (CUDA)");
    module.def(
        "backward",
        &compressed_time_local_backward_cuda,
        "Compressed time-local attention backward (CUDA)");
}
