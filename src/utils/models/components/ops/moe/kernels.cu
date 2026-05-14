#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK((x).scalar_type() == at::kLong, #x " must be int64")

template <typename scalar_t>
__global__ void moe_dispatch_locations_kernel(
    const int64_t* __restrict__ expert_indices,
    int64_t* __restrict__ locations,
    bool* __restrict__ combine_mask,
    int64_t* __restrict__ expert_counts,
    int64_t num_tokens,
    int64_t top_k,
    int64_t capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    const int64_t total_pairs = num_tokens * top_k;
    for (int64_t pair_idx = 0; pair_idx < total_pairs; ++pair_idx) {
        const int64_t expert_idx = expert_indices[pair_idx];
        const int64_t slot = expert_counts[expert_idx];
        locations[pair_idx] = -1;
        combine_mask[pair_idx] = false;
        if (slot < capacity) {
            locations[pair_idx] = slot;
            combine_mask[pair_idx] = true;
            expert_counts[expert_idx] = slot + 1;
        }
    }
}

template <typename scalar_t>
__global__ void moe_dispatch_copy_kernel(
    const scalar_t* __restrict__ tokens,
    const int64_t* __restrict__ expert_indices,
    scalar_t* __restrict__ expert_inputs,
    const int64_t* __restrict__ locations,
    const bool* __restrict__ combine_mask,
    int64_t total_values,
    int64_t top_k,
    int64_t hidden_dim,
    int64_t capacity) {
    const int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear_idx >= total_values) {
        return;
    }

    const int64_t pair_idx = linear_idx / hidden_dim;
    if (!combine_mask[pair_idx]) {
        return;
    }
    const int64_t token_idx = pair_idx / top_k;
    const int64_t hidden_idx = linear_idx - pair_idx * hidden_dim;
    const int64_t expert_idx = expert_indices[pair_idx];
    const int64_t slot = locations[pair_idx];
    expert_inputs[(expert_idx * capacity + slot) * hidden_dim + hidden_idx] =
        tokens[token_idx * hidden_dim + hidden_idx];
}

template <typename scalar_t>
__global__ void moe_dispatch_backward_kernel(
    const scalar_t* __restrict__ grad_expert_inputs,
    const int64_t* __restrict__ expert_indices,
    const int64_t* __restrict__ locations,
    const bool* __restrict__ combine_mask,
    scalar_t* __restrict__ grad_tokens,
    int64_t num_tokens,
    int64_t top_k,
    int64_t hidden_dim,
    int64_t capacity) {
    const int64_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) {
        return;
    }

    for (int64_t hidden_idx = 0; hidden_idx < hidden_dim; ++hidden_idx) {
        float acc = 0.0f;
        for (int64_t top_idx = 0; top_idx < top_k; ++top_idx) {
            const int64_t pair_idx = token_idx * top_k + top_idx;
            if (!combine_mask[pair_idx]) {
                continue;
            }
            const int64_t expert_idx = expert_indices[pair_idx];
            const int64_t slot = locations[pair_idx];
            const int64_t source_idx = (expert_idx * capacity + slot) * hidden_dim + hidden_idx;
            acc += static_cast<float>(grad_expert_inputs[source_idx]);
        }
        grad_tokens[token_idx * hidden_dim + hidden_idx] = static_cast<scalar_t>(acc);
    }
}

template <typename scalar_t>
__global__ void moe_combine_forward_kernel(
    const scalar_t* __restrict__ expert_outputs,
    const int64_t* __restrict__ expert_indices,
    const int64_t* __restrict__ locations,
    const scalar_t* __restrict__ expert_weights,
    const bool* __restrict__ combine_mask,
    scalar_t* __restrict__ output,
    int64_t num_tokens,
    int64_t top_k,
    int64_t hidden_dim,
    int64_t capacity) {
    const int64_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) {
        return;
    }

    for (int64_t hidden_idx = 0; hidden_idx < hidden_dim; ++hidden_idx) {
        float acc = 0.0f;
        for (int64_t top_idx = 0; top_idx < top_k; ++top_idx) {
            const int64_t pair_idx = token_idx * top_k + top_idx;
            if (!combine_mask[pair_idx]) {
                continue;
            }
            const int64_t expert_idx = expert_indices[pair_idx];
            const int64_t slot = locations[pair_idx];
            const int64_t source_idx = (expert_idx * capacity + slot) * hidden_dim + hidden_idx;
            acc += static_cast<float>(expert_outputs[source_idx]) *
                   static_cast<float>(expert_weights[pair_idx]);
        }
        output[token_idx * hidden_dim + hidden_idx] = static_cast<scalar_t>(acc);
    }
}

template <typename scalar_t>
__global__ void moe_combine_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ expert_outputs,
    const int64_t* __restrict__ expert_indices,
    const int64_t* __restrict__ locations,
    const scalar_t* __restrict__ expert_weights,
    const bool* __restrict__ combine_mask,
    scalar_t* __restrict__ grad_expert_outputs,
    scalar_t* __restrict__ grad_expert_weights,
    int64_t num_tokens,
    int64_t top_k,
    int64_t hidden_dim,
    int64_t capacity) {
    const int64_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_pairs = num_tokens * top_k;
    if (pair_idx >= total_pairs) {
        return;
    }
    if (!combine_mask[pair_idx]) {
        grad_expert_weights[pair_idx] = static_cast<scalar_t>(0.0f);
        return;
    }

    const int64_t token_idx = pair_idx / top_k;
    const int64_t expert_idx = expert_indices[pair_idx];
    const int64_t slot = locations[pair_idx];
    float weight_grad = 0.0f;
    for (int64_t hidden_idx = 0; hidden_idx < hidden_dim; ++hidden_idx) {
        const int64_t expert_offset = (expert_idx * capacity + slot) * hidden_dim + hidden_idx;
        const int64_t output_offset = token_idx * hidden_dim + hidden_idx;
        const float grad_value = static_cast<float>(grad_output[output_offset]);
        grad_expert_outputs[expert_offset] = static_cast<scalar_t>(
            grad_value * static_cast<float>(expert_weights[pair_idx]));
        weight_grad += grad_value * static_cast<float>(expert_outputs[expert_offset]);
    }
    grad_expert_weights[pair_idx] = static_cast<scalar_t>(weight_grad);
}

std::vector<torch::Tensor> moe_dispatch_forward_cuda(
    torch::Tensor tokens,
    torch::Tensor expert_indices,
    int64_t num_experts,
    int64_t capacity) {
    CHECK_CUDA(tokens);
    CHECK_CUDA(expert_indices);
    CHECK_CONTIGUOUS(tokens);
    CHECK_CONTIGUOUS(expert_indices);
    CHECK_LONG(expert_indices);
    TORCH_CHECK(tokens.dim() == 2, "tokens must have shape [tokens, hidden]");
    TORCH_CHECK(expert_indices.dim() == 2, "expert_indices must have shape [tokens, top_k]");
    TORCH_CHECK(tokens.size(0) == expert_indices.size(0), "token dimension mismatch");
    TORCH_CHECK(num_experts > 0, "num_experts must be positive");
    TORCH_CHECK(capacity >= 0, "capacity must be non-negative");

    const c10::cuda::CUDAGuard device_guard(tokens.device());
    const int64_t num_tokens = tokens.size(0);
    const int64_t hidden_dim = tokens.size(1);
    const int64_t top_k = expert_indices.size(1);
    auto expert_inputs = torch::zeros({num_experts, capacity, hidden_dim}, tokens.options());
    auto locations = torch::full(expert_indices.sizes(), -1, expert_indices.options());
    auto combine_mask = torch::zeros(expert_indices.sizes(), expert_indices.options().dtype(torch::kBool));
    auto expert_counts = torch::zeros({num_experts}, expert_indices.options());

    const int64_t total_pairs = num_tokens * top_k;
    if (total_pairs > 0) {
        const int threads = 256;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, tokens.scalar_type(), "moe_dispatch_forward_cuda", [&] {
            moe_dispatch_locations_kernel<scalar_t><<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
                expert_indices.data_ptr<int64_t>(),
                locations.data_ptr<int64_t>(),
                combine_mask.data_ptr<bool>(),
                expert_counts.data_ptr<int64_t>(),
                num_tokens,
                top_k,
                capacity);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            const int64_t total_values = total_pairs * hidden_dim;
            const int blocks = static_cast<int>((total_values + threads - 1) / threads);
            moe_dispatch_copy_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                tokens.data_ptr<scalar_t>(),
                expert_indices.data_ptr<int64_t>(),
                expert_inputs.data_ptr<scalar_t>(),
                locations.data_ptr<int64_t>(),
                combine_mask.data_ptr<bool>(),
                total_values,
                top_k,
                hidden_dim,
                capacity);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
    return {expert_inputs, locations, combine_mask, expert_counts};
}

torch::Tensor moe_dispatch_backward_cuda(
    torch::Tensor grad_expert_inputs,
    torch::Tensor expert_indices,
    torch::Tensor locations,
    torch::Tensor combine_mask,
    int64_t num_tokens) {
    CHECK_CUDA(grad_expert_inputs);
    CHECK_CUDA(expert_indices);
    CHECK_CUDA(locations);
    CHECK_CUDA(combine_mask);
    CHECK_CONTIGUOUS(grad_expert_inputs);
    CHECK_CONTIGUOUS(expert_indices);
    CHECK_CONTIGUOUS(locations);
    CHECK_CONTIGUOUS(combine_mask);
    CHECK_LONG(expert_indices);
    CHECK_LONG(locations);

    const c10::cuda::CUDAGuard device_guard(grad_expert_inputs.device());
    const int64_t capacity = grad_expert_inputs.size(1);
    const int64_t hidden_dim = grad_expert_inputs.size(2);
    const int64_t top_k = expert_indices.size(1);
    auto grad_tokens = torch::zeros({num_tokens, hidden_dim}, grad_expert_inputs.options());
    if (num_tokens > 0) {
        const int threads = 256;
        const int blocks = static_cast<int>((num_tokens + threads - 1) / threads);
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, grad_expert_inputs.scalar_type(), "moe_dispatch_backward_cuda", [&] {
            moe_dispatch_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_expert_inputs.data_ptr<scalar_t>(),
                expert_indices.data_ptr<int64_t>(),
                locations.data_ptr<int64_t>(),
                combine_mask.data_ptr<bool>(),
                grad_tokens.data_ptr<scalar_t>(),
                num_tokens,
                top_k,
                hidden_dim,
                capacity);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
    return grad_tokens;
}

torch::Tensor moe_combine_forward_cuda(
    torch::Tensor expert_outputs,
    torch::Tensor expert_indices,
    torch::Tensor locations,
    torch::Tensor expert_weights,
    torch::Tensor combine_mask) {
    CHECK_CUDA(expert_outputs);
    CHECK_CUDA(expert_indices);
    CHECK_CUDA(locations);
    CHECK_CUDA(expert_weights);
    CHECK_CUDA(combine_mask);
    CHECK_CONTIGUOUS(expert_outputs);
    CHECK_CONTIGUOUS(expert_indices);
    CHECK_CONTIGUOUS(locations);
    CHECK_CONTIGUOUS(expert_weights);
    CHECK_CONTIGUOUS(combine_mask);
    CHECK_LONG(expert_indices);
    CHECK_LONG(locations);

    const c10::cuda::CUDAGuard device_guard(expert_outputs.device());
    const int64_t num_tokens = expert_indices.size(0);
    const int64_t top_k = expert_indices.size(1);
    const int64_t hidden_dim = expert_outputs.size(2);
    const int64_t capacity = expert_outputs.size(1);
    auto output = torch::zeros({num_tokens, hidden_dim}, expert_outputs.options());
    if (num_tokens > 0) {
        const int threads = 256;
        const int blocks = static_cast<int>((num_tokens + threads - 1) / threads);
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, expert_outputs.scalar_type(), "moe_combine_forward_cuda", [&] {
            moe_combine_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                expert_outputs.data_ptr<scalar_t>(),
                expert_indices.data_ptr<int64_t>(),
                locations.data_ptr<int64_t>(),
                expert_weights.data_ptr<scalar_t>(),
                combine_mask.data_ptr<bool>(),
                output.data_ptr<scalar_t>(),
                num_tokens,
                top_k,
                hidden_dim,
                capacity);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
    return output;
}

std::vector<torch::Tensor> moe_combine_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor expert_outputs,
    torch::Tensor expert_indices,
    torch::Tensor locations,
    torch::Tensor expert_weights,
    torch::Tensor combine_mask) {
    CHECK_CUDA(grad_output);
    CHECK_CUDA(expert_outputs);
    CHECK_CUDA(expert_indices);
    CHECK_CUDA(locations);
    CHECK_CUDA(expert_weights);
    CHECK_CUDA(combine_mask);
    CHECK_CONTIGUOUS(grad_output);
    CHECK_CONTIGUOUS(expert_outputs);
    CHECK_CONTIGUOUS(expert_indices);
    CHECK_CONTIGUOUS(locations);
    CHECK_CONTIGUOUS(expert_weights);
    CHECK_CONTIGUOUS(combine_mask);
    CHECK_LONG(expert_indices);
    CHECK_LONG(locations);

    const c10::cuda::CUDAGuard device_guard(expert_outputs.device());
    const int64_t num_tokens = expert_indices.size(0);
    const int64_t top_k = expert_indices.size(1);
    const int64_t hidden_dim = expert_outputs.size(2);
    const int64_t capacity = expert_outputs.size(1);
    auto grad_expert_outputs = torch::zeros_like(expert_outputs);
    auto grad_expert_weights = torch::zeros_like(expert_weights);
    const int64_t total_pairs = num_tokens * top_k;
    if (total_pairs > 0) {
        const int threads = 256;
        const int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, expert_outputs.scalar_type(), "moe_combine_backward_cuda", [&] {
            moe_combine_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_output.data_ptr<scalar_t>(),
                expert_outputs.data_ptr<scalar_t>(),
                expert_indices.data_ptr<int64_t>(),
                locations.data_ptr<int64_t>(),
                expert_weights.data_ptr<scalar_t>(),
                combine_mask.data_ptr<bool>(),
                grad_expert_outputs.data_ptr<scalar_t>(),
                grad_expert_weights.data_ptr<scalar_t>(),
                num_tokens,
                top_k,
                hidden_dim,
                capacity);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
    return {grad_expert_outputs, grad_expert_weights};
}