#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK((x).scalar_type() == at::kLong, #x " must be int64")

template <typename scalar_t>
__global__ void window_gather_forward_kernel(
    const scalar_t* __restrict__ tensor,
    const int64_t* __restrict__ indices,
    scalar_t* __restrict__ output,
    int64_t batch_size,
    int64_t num_heads,
    int64_t seq_len,
    int64_t window_size,
    int64_t hidden_dim) {
    const int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_values = batch_size * num_heads * seq_len * window_size * hidden_dim;
    if (linear_idx >= total_values) {
        return;
    }

    int64_t residual = linear_idx;
    const int64_t hidden_idx = residual % hidden_dim;
    residual /= hidden_dim;
    const int64_t window_idx = residual % window_size;
    residual /= window_size;
    const int64_t time_idx = residual % seq_len;
    residual /= seq_len;
    const int64_t head_idx = residual % num_heads;
    const int64_t batch_idx = residual / num_heads;

    const int64_t source_time_idx = indices[time_idx * window_size + window_idx];
    const int64_t source_offset =
        (((batch_idx * num_heads + head_idx) * seq_len + source_time_idx) * hidden_dim) +
        hidden_idx;
    output[linear_idx] = tensor[source_offset];
}

torch::Tensor window_gather_forward_cuda(
    torch::Tensor tensor,
    torch::Tensor indices) {
    CHECK_CUDA(tensor);
    CHECK_CUDA(indices);
    CHECK_CONTIGUOUS(tensor);
    CHECK_CONTIGUOUS(indices);
    CHECK_LONG(indices);
    TORCH_CHECK(tensor.dim() == 4, "tensor must have shape [B, H, T, D]");
    TORCH_CHECK(indices.dim() == 2, "indices must have shape [T, W]");
    TORCH_CHECK(
        tensor.size(2) == indices.size(0),
        "indices first dimension must equal tensor sequence length");

    const c10::cuda::CUDAGuard device_guard(tensor.device());
    const int64_t batch_size = tensor.size(0);
    const int64_t num_heads = tensor.size(1);
    const int64_t seq_len = tensor.size(2);
    const int64_t hidden_dim = tensor.size(3);
    const int64_t window_size = indices.size(1);

    auto output = torch::empty(
        {batch_size, num_heads, seq_len, window_size, hidden_dim},
        tensor.options());
    const int64_t total_values = output.numel();
    if (total_values == 0) {
        return output;
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total_values + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        tensor.scalar_type(),
        "window_gather_forward_cuda",
        [&] {
            window_gather_forward_kernel<scalar_t>
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    tensor.data_ptr<scalar_t>(),
                    indices.data_ptr<int64_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    num_heads,
                    seq_len,
                    window_size,
                    hidden_dim);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}