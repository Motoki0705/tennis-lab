#include <torch/extension.h>

// Forward declarations from deformable_cuda.cpp
at::Tensor ms_deform_attn_forward_cuda(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights);

std::vector<at::Tensor> ms_deform_attn_backward_cuda(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const at::Tensor& grad_output);

at::Tensor ms_deform_attn_forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights) {
  TORCH_CHECK(value.is_cuda(), "value must be CUDA tensor for ms_deform_attn_forward.");
  return ms_deform_attn_forward_cuda(
      value, spatial_shapes, level_start_index, sampling_locations, attention_weights);
}

std::vector<at::Tensor> ms_deform_attn_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const at::Tensor& grad_output) {
  TORCH_CHECK(value.is_cuda(), "value must be CUDA tensor for ms_deform_attn_backward.");
  return ms_deform_attn_backward_cuda(
      value,
      spatial_shapes,
      level_start_index,
      sampling_locations,
      attention_weights,
      grad_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "MSDeformAttn forward (CUDA)");
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "MSDeformAttn backward (CUDA)");
}
