#include "deformable_cuda_kernel.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__device__ inline scalar_t read_value(
    const scalar_t* value,
    int64_t b,
    int64_t s,
    int64_t h,
    int64_t c,
    int64_t S,
    int64_t H,
    int64_t D) {
  const int64_t idx = (((b * S + s) * H + h) * D + c);
  return value[idx];
}

template <typename scalar_t>
__device__ inline void atomic_add_value(
    scalar_t* grad_value,
    scalar_t v,
    int64_t b,
    int64_t s,
    int64_t h,
    int64_t c,
    int64_t S,
    int64_t H,
    int64_t D) {
  const int64_t idx = (((b * S + s) * H + h) * D + c);
  atomicAdd(&grad_value[idx], v);
}

template <typename scalar_t>
__global__ void msda_forward_kernel(
    const scalar_t* __restrict__ value,            // (B,S,H,D)
    const int64_t* __restrict__ spatial_shapes,    // (L,2)
    const int64_t* __restrict__ level_start_index, // (L)
    const scalar_t* __restrict__ sampling_locations, // (B,Q,H,L,P,2)
    const scalar_t* __restrict__ attention_weights,  // (B,Q,H,L,P)
    scalar_t* __restrict__ output,                   // (B,Q,H,D)
    int64_t B,
    int64_t S,
    int64_t H,
    int64_t D,
    int64_t Q,
    int64_t L,
    int64_t P) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = B * Q * H * D;
  if (idx >= total) {
    return;
  }

  int64_t tmp = idx;
  const int64_t c = tmp % D;
  tmp /= D;
  const int64_t h = tmp % H;
  tmp /= H;
  const int64_t q = tmp % Q;
  const int64_t b = tmp / Q;

  scalar_t acc = scalar_t(0);

  for (int64_t l = 0; l < L; ++l) {
    const int64_t h_l = spatial_shapes[l * 2 + 0];
    const int64_t w_l = spatial_shapes[l * 2 + 1];
    const int64_t start = level_start_index[l];

    for (int64_t p = 0; p < P; ++p) {
      const int64_t loc_base = (((((b * Q + q) * H + h) * L + l) * P + p) * 2);
      const scalar_t loc_x = sampling_locations[loc_base + 0];
      const scalar_t loc_y = sampling_locations[loc_base + 1];
      const int64_t w_base = ((((b * Q + q) * H + h) * L + l) * P + p);
      const scalar_t attn = attention_weights[w_base];

      const scalar_t x = loc_x * static_cast<scalar_t>(w_l) - static_cast<scalar_t>(0.5);
      const scalar_t y = loc_y * static_cast<scalar_t>(h_l) - static_cast<scalar_t>(0.5);

      const int64_t x0 = static_cast<int64_t>(floorf(static_cast<float>(x)));
      const int64_t y0 = static_cast<int64_t>(floorf(static_cast<float>(y)));
      const int64_t x1 = x0 + 1;
      const int64_t y1 = y0 + 1;

      const scalar_t dx = x - static_cast<scalar_t>(x0);
      const scalar_t dy = y - static_cast<scalar_t>(y0);

      const scalar_t w00 = (static_cast<scalar_t>(1) - dx) * (static_cast<scalar_t>(1) - dy);
      const scalar_t w01 = dx * (static_cast<scalar_t>(1) - dy);
      const scalar_t w10 = (static_cast<scalar_t>(1) - dx) * dy;
      const scalar_t w11 = dx * dy;

      scalar_t v00 = scalar_t(0);
      scalar_t v01 = scalar_t(0);
      scalar_t v10 = scalar_t(0);
      scalar_t v11 = scalar_t(0);

      if (x0 >= 0 && x0 < w_l && y0 >= 0 && y0 < h_l) {
        const int64_t s = start + y0 * w_l + x0;
        v00 = read_value(value, b, s, h, c, S, H, D);
      }
      if (x1 >= 0 && x1 < w_l && y0 >= 0 && y0 < h_l) {
        const int64_t s = start + y0 * w_l + x1;
        v01 = read_value(value, b, s, h, c, S, H, D);
      }
      if (x0 >= 0 && x0 < w_l && y1 >= 0 && y1 < h_l) {
        const int64_t s = start + y1 * w_l + x0;
        v10 = read_value(value, b, s, h, c, S, H, D);
      }
      if (x1 >= 0 && x1 < w_l && y1 >= 0 && y1 < h_l) {
        const int64_t s = start + y1 * w_l + x1;
        v11 = read_value(value, b, s, h, c, S, H, D);
      }

      const scalar_t sampled = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
      acc += sampled * attn;
    }
  }

  output[idx] = acc;
}

template <typename scalar_t>
__global__ void msda_backward_value_kernel(
    const scalar_t* __restrict__ grad_output,       // (B,Q,H,D)
    const scalar_t* __restrict__ sampling_locations, // (B,Q,H,L,P,2)
    const scalar_t* __restrict__ attention_weights,  // (B,Q,H,L,P)
    const int64_t* __restrict__ spatial_shapes,      // (L,2)
    const int64_t* __restrict__ level_start_index,   // (L)
    scalar_t* __restrict__ grad_value,               // (B,S,H,D)
    int64_t B,
    int64_t S,
    int64_t H,
    int64_t D,
    int64_t Q,
    int64_t L,
    int64_t P) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = B * Q * H * D;
  if (idx >= total) {
    return;
  }

  int64_t tmp = idx;
  const int64_t c = tmp % D;
  tmp /= D;
  const int64_t h = tmp % H;
  tmp /= H;
  const int64_t q = tmp % Q;
  const int64_t b = tmp / Q;

  const scalar_t g = grad_output[idx];

  for (int64_t l = 0; l < L; ++l) {
    const int64_t h_l = spatial_shapes[l * 2 + 0];
    const int64_t w_l = spatial_shapes[l * 2 + 1];
    const int64_t start = level_start_index[l];

    for (int64_t p = 0; p < P; ++p) {
      const int64_t loc_base = (((((b * Q + q) * H + h) * L + l) * P + p) * 2);
      const scalar_t loc_x = sampling_locations[loc_base + 0];
      const scalar_t loc_y = sampling_locations[loc_base + 1];
      const int64_t w_base = ((((b * Q + q) * H + h) * L + l) * P + p);
      const scalar_t attn = attention_weights[w_base];

      const scalar_t x = loc_x * static_cast<scalar_t>(w_l) - static_cast<scalar_t>(0.5);
      const scalar_t y = loc_y * static_cast<scalar_t>(h_l) - static_cast<scalar_t>(0.5);

      const int64_t x0 = static_cast<int64_t>(floorf(static_cast<float>(x)));
      const int64_t y0 = static_cast<int64_t>(floorf(static_cast<float>(y)));
      const int64_t x1 = x0 + 1;
      const int64_t y1 = y0 + 1;

      const scalar_t dx = x - static_cast<scalar_t>(x0);
      const scalar_t dy = y - static_cast<scalar_t>(y0);

      const scalar_t w00 = (static_cast<scalar_t>(1) - dx) * (static_cast<scalar_t>(1) - dy);
      const scalar_t w01 = dx * (static_cast<scalar_t>(1) - dy);
      const scalar_t w10 = (static_cast<scalar_t>(1) - dx) * dy;
      const scalar_t w11 = dx * dy;

      const scalar_t scale = g * attn;

      if (x0 >= 0 && x0 < w_l && y0 >= 0 && y0 < h_l) {
        const int64_t s = start + y0 * w_l + x0;
        atomic_add_value(grad_value, scale * w00, b, s, h, c, S, H, D);
      }
      if (x1 >= 0 && x1 < w_l && y0 >= 0 && y0 < h_l) {
        const int64_t s = start + y0 * w_l + x1;
        atomic_add_value(grad_value, scale * w01, b, s, h, c, S, H, D);
      }
      if (x0 >= 0 && x0 < w_l && y1 >= 0 && y1 < h_l) {
        const int64_t s = start + y1 * w_l + x0;
        atomic_add_value(grad_value, scale * w10, b, s, h, c, S, H, D);
      }
      if (x1 >= 0 && x1 < w_l && y1 >= 0 && y1 < h_l) {
        const int64_t s = start + y1 * w_l + x1;
        atomic_add_value(grad_value, scale * w11, b, s, h, c, S, H, D);
      }
    }
  }
}

template <typename scalar_t>
__global__ void msda_backward_loc_attn_kernel(
    const scalar_t* __restrict__ value,              // (B,S,H,D)
    const scalar_t* __restrict__ grad_output,        // (B,Q,H,D)
    const scalar_t* __restrict__ sampling_locations, // (B,Q,H,L,P,2)
    const scalar_t* __restrict__ attention_weights,  // (B,Q,H,L,P)
    const int64_t* __restrict__ spatial_shapes,      // (L,2)
    const int64_t* __restrict__ level_start_index,   // (L)
    scalar_t* __restrict__ grad_sampling_locations,  // (B,Q,H,L,P,2)
    scalar_t* __restrict__ grad_attention_weights,   // (B,Q,H,L,P)
    int64_t B,
    int64_t S,
    int64_t H,
    int64_t D,
    int64_t Q,
    int64_t L,
    int64_t P) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = B * Q * H * L * P;
  if (idx >= total) {
    return;
  }

  int64_t tmp = idx;
  const int64_t p = tmp % P;
  tmp /= P;
  const int64_t l = tmp % L;
  tmp /= L;
  const int64_t h = tmp % H;
  tmp /= H;
  const int64_t q = tmp % Q;
  const int64_t b = tmp / Q;

  const int64_t h_l = spatial_shapes[l * 2 + 0];
  const int64_t w_l = spatial_shapes[l * 2 + 1];
  const int64_t start = level_start_index[l];

  const int64_t loc_base = (((((b * Q + q) * H + h) * L + l) * P + p) * 2);
  const scalar_t loc_x = sampling_locations[loc_base + 0];
  const scalar_t loc_y = sampling_locations[loc_base + 1];

  const scalar_t x = loc_x * static_cast<scalar_t>(w_l) - static_cast<scalar_t>(0.5);
  const scalar_t y = loc_y * static_cast<scalar_t>(h_l) - static_cast<scalar_t>(0.5);

  const int64_t x0 = static_cast<int64_t>(floorf(static_cast<float>(x)));
  const int64_t y0 = static_cast<int64_t>(floorf(static_cast<float>(y)));
  const int64_t x1 = x0 + 1;
  const int64_t y1 = y0 + 1;

  const scalar_t dx = x - static_cast<scalar_t>(x0);
  const scalar_t dy = y - static_cast<scalar_t>(y0);

  const scalar_t attn = attention_weights[((((b * Q + q) * H + h) * L + l) * P + p)];

  scalar_t grad_attn = scalar_t(0);
  scalar_t grad_x = scalar_t(0);
  scalar_t grad_y = scalar_t(0);

  for (int64_t c = 0; c < D; ++c) {
    const scalar_t go = grad_output[(((b * Q + q) * H + h) * D + c)];

    scalar_t v00 = scalar_t(0);
    scalar_t v01 = scalar_t(0);
    scalar_t v10 = scalar_t(0);
    scalar_t v11 = scalar_t(0);

    if (x0 >= 0 && x0 < w_l && y0 >= 0 && y0 < h_l) {
      const int64_t s = start + y0 * w_l + x0;
      v00 = read_value(value, b, s, h, c, S, H, D);
    }
    if (x1 >= 0 && x1 < w_l && y0 >= 0 && y0 < h_l) {
      const int64_t s = start + y0 * w_l + x1;
      v01 = read_value(value, b, s, h, c, S, H, D);
    }
    if (x0 >= 0 && x0 < w_l && y1 >= 0 && y1 < h_l) {
      const int64_t s = start + y1 * w_l + x0;
      v10 = read_value(value, b, s, h, c, S, H, D);
    }
    if (x1 >= 0 && x1 < w_l && y1 >= 0 && y1 < h_l) {
      const int64_t s = start + y1 * w_l + x1;
      v11 = read_value(value, b, s, h, c, S, H, D);
    }

    const scalar_t sampled =
        v00 * (static_cast<scalar_t>(1) - dx) * (static_cast<scalar_t>(1) - dy) +
        v01 * dx * (static_cast<scalar_t>(1) - dy) +
        v10 * (static_cast<scalar_t>(1) - dx) * dy +
        v11 * dx * dy;

    const scalar_t dsdx =
        (v01 - v00) * (static_cast<scalar_t>(1) - dy) + (v11 - v10) * dy;
    const scalar_t dsdy =
        (v10 - v00) * (static_cast<scalar_t>(1) - dx) + (v11 - v01) * dx;

    grad_attn += go * sampled;
    grad_x += go * attn * dsdx * static_cast<scalar_t>(w_l);
    grad_y += go * attn * dsdy * static_cast<scalar_t>(h_l);
  }

  grad_attention_weights[((((b * Q + q) * H + h) * L + l) * P + p)] = grad_attn;
  grad_sampling_locations[loc_base + 0] = grad_x;
  grad_sampling_locations[loc_base + 1] = grad_y;
}

at::Tensor ms_deform_attn_forward_cuda_launcher(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights) {
  CHECK_INPUT(value);
  CHECK_INPUT(spatial_shapes);
  CHECK_INPUT(level_start_index);
  CHECK_INPUT(sampling_locations);
  CHECK_INPUT(attention_weights);

  TORCH_CHECK(value.dim() == 4, "value must be (B,S,H,D)");
  TORCH_CHECK(spatial_shapes.dim() == 2 && spatial_shapes.size(1) == 2, "spatial_shapes must be (L,2)");
  TORCH_CHECK(level_start_index.dim() == 1, "level_start_index must be (L,)");
  TORCH_CHECK(sampling_locations.dim() == 6, "sampling_locations must be (B,Q,H,L,P,2)");
  TORCH_CHECK(attention_weights.dim() == 5, "attention_weights must be (B,Q,H,L,P)");

  const auto B = value.size(0);
  const auto S = value.size(1);
  const auto H = value.size(2);
  const auto D = value.size(3);
  const auto Q = sampling_locations.size(1);
  const auto L = sampling_locations.size(3);
  const auto P = sampling_locations.size(4);

  auto out = at::zeros({B, Q, H, D}, value.options());

  const int threads = 256;
  const int64_t total = B * Q * H * D;
  const int blocks = static_cast<int>((total + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.scalar_type(), "msda_forward_cuda", [&] {
    msda_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        value.data_ptr<scalar_t>(),
        spatial_shapes.data_ptr<int64_t>(),
        level_start_index.data_ptr<int64_t>(),
        sampling_locations.data_ptr<scalar_t>(),
        attention_weights.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        B,
        S,
        H,
        D,
        Q,
        L,
        P);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<at::Tensor> ms_deform_attn_backward_cuda_launcher(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights,
    const at::Tensor& grad_output) {
  CHECK_INPUT(value);
  CHECK_INPUT(spatial_shapes);
  CHECK_INPUT(level_start_index);
  CHECK_INPUT(sampling_locations);
  CHECK_INPUT(attention_weights);
  CHECK_INPUT(grad_output);

  TORCH_CHECK(value.dim() == 4, "value must be (B,S,H,D)");
  TORCH_CHECK(grad_output.dim() == 4, "grad_output must be (B,Q,H,D)");

  const auto B = value.size(0);
  const auto S = value.size(1);
  const auto H = value.size(2);
  const auto D = value.size(3);
  const auto Q = sampling_locations.size(1);
  const auto L = sampling_locations.size(3);
  const auto P = sampling_locations.size(4);

  auto grad_value = at::zeros_like(value);
  auto grad_sampling_locations = at::zeros_like(sampling_locations);
  auto grad_attention_weights = at::zeros_like(attention_weights);

  const int threads = 256;
  const int64_t total_value = B * Q * H * D;
  const int blocks_value = static_cast<int>((total_value + threads - 1) / threads);
  const int64_t total_loc = B * Q * H * L * P;
  const int blocks_loc = static_cast<int>((total_loc + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.scalar_type(), "msda_backward_cuda", [&] {
    msda_backward_value_kernel<scalar_t><<<blocks_value, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        grad_output.data_ptr<scalar_t>(),
        sampling_locations.data_ptr<scalar_t>(),
        attention_weights.data_ptr<scalar_t>(),
        spatial_shapes.data_ptr<int64_t>(),
        level_start_index.data_ptr<int64_t>(),
        grad_value.data_ptr<scalar_t>(),
        B,
        S,
        H,
        D,
        Q,
        L,
        P);

    msda_backward_loc_attn_kernel<scalar_t><<<blocks_loc, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        value.data_ptr<scalar_t>(),
        grad_output.data_ptr<scalar_t>(),
        sampling_locations.data_ptr<scalar_t>(),
        attention_weights.data_ptr<scalar_t>(),
        spatial_shapes.data_ptr<int64_t>(),
        level_start_index.data_ptr<int64_t>(),
        grad_sampling_locations.data_ptr<scalar_t>(),
        grad_attention_weights.data_ptr<scalar_t>(),
        B,
        S,
        H,
        D,
        Q,
        L,
        P);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_value, grad_sampling_locations, grad_attention_weights};
}
