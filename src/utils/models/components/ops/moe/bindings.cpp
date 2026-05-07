#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> moe_dispatch_forward_cuda(
    torch::Tensor tokens,
    torch::Tensor expert_indices,
    int64_t num_experts,
    int64_t capacity);

torch::Tensor moe_dispatch_backward_cuda(
    torch::Tensor grad_expert_inputs,
    torch::Tensor expert_indices,
    torch::Tensor locations,
    torch::Tensor combine_mask,
    int64_t num_tokens);

torch::Tensor moe_combine_forward_cuda(
    torch::Tensor expert_outputs,
    torch::Tensor expert_indices,
    torch::Tensor locations,
    torch::Tensor expert_weights,
    torch::Tensor combine_mask);

std::vector<torch::Tensor> moe_combine_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor expert_outputs,
    torch::Tensor expert_indices,
    torch::Tensor locations,
    torch::Tensor expert_weights,
    torch::Tensor combine_mask);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_dispatch_forward", &moe_dispatch_forward_cuda, "MoE dispatch forward (CUDA)");
    m.def("moe_dispatch_backward", &moe_dispatch_backward_cuda, "MoE dispatch backward (CUDA)");
    m.def("moe_combine_forward", &moe_combine_forward_cuda, "MoE combine forward (CUDA)");
    m.def("moe_combine_backward", &moe_combine_backward_cuda, "MoE combine backward (CUDA)");
}