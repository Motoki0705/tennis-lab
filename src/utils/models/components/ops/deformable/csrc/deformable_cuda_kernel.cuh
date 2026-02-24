#pragma once

#include <torch/extension.h>

at::Tensor ms_deform_attn_forward_cuda_launcher(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights);

std::vector<at::Tensor> ms_deform_attn_backward_cuda_launcher(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const at::Tensor& grad_output);
