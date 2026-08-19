#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#define CHECK_CUDA(tensor) TORCH_CHECK((tensor).is_cuda(), #tensor " must be CUDA")
#define CHECK_CONTIGUOUS(tensor) \
    TORCH_CHECK((tensor).is_contiguous(), #tensor " must be contiguous")
#define CHECK_BOOL(tensor) \
    TORCH_CHECK((tensor).scalar_type() == at::kBool, #tensor " must be bool")

namespace {

constexpr int kWarpSize = 32;
constexpr int64_t kMaximumWindowRadius = 64;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <typename scalar_t>
__global__ void compressed_time_local_forward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const bool* __restrict__ query_valid,
    const bool* __restrict__ key_valid,
    scalar_t* __restrict__ output,
    float* __restrict__ logsumexp,
    int* __restrict__ invalid_row,
    int64_t heads,
    int64_t query_length,
    int64_t key_length,
    int64_t head_dim,
    int64_t compression_ratio,
    int64_t window_radius) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t query_index = row % query_length;
    const int64_t head_index = (row / query_length) % heads;
    const int64_t batch_index = row / (heads * query_length);
    if (!query_valid[batch_index * query_length + query_index]) {
        return;
    }

    const int64_t query_offset = row * head_dim;
    const int64_t center = query_index / compression_ratio;
    const int window_width = static_cast<int>(2 * window_radius + 1);
    extern __shared__ float shared[];
    float* scores = shared;
    float* statistics = shared + window_width;
    float running_max = -INFINITY;
    float denominator = 0.0f;
    bool any_valid = false;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    for (int window_index = 0; window_index < window_width; ++window_index) {
        const int64_t key_index = center + window_index - window_radius;
        const bool index_valid = key_index >= 0 && key_index < key_length;
        const bool valid = index_valid &&
            key_valid[batch_index * key_length + key_index];
        float dot = 0.0f;
        if (valid) {
            const int64_t key_offset =
                ((batch_index * heads + head_index) * key_length + key_index) *
                head_dim;
            for (int64_t feature = lane; feature < head_dim; feature += kWarpSize) {
                dot += static_cast<float>(query[query_offset + feature]) *
                    static_cast<float>(key[key_offset + feature]);
            }
        }
        dot = warp_sum(dot);
        if (lane == 0) {
            if (valid) {
                const float score = dot * scale;
                scores[window_index] = score;
                if (!any_valid) {
                    running_max = score;
                    denominator = 1.0f;
                    any_valid = true;
                } else if (score > running_max) {
                    denominator = denominator * expf(running_max - score) + 1.0f;
                    running_max = score;
                } else {
                    denominator += expf(score - running_max);
                }
            } else {
                scores[window_index] = -INFINITY;
            }
        }
        __syncwarp();
    }

    if (lane == 0) {
        statistics[0] = running_max;
        statistics[1] = denominator;
        if (!any_valid) {
            atomicExch(invalid_row, 1);
        } else {
            logsumexp[row] = running_max + logf(denominator);
        }
    }
    __syncwarp();
    if (statistics[1] == 0.0f) {
        return;
    }

    for (int64_t feature = lane; feature < head_dim; feature += kWarpSize) {
        float accumulated = 0.0f;
        for (int window_index = 0; window_index < window_width; ++window_index) {
            const float score = scores[window_index];
            if (score == -INFINITY) {
                continue;
            }
            const int64_t key_index = center + window_index - window_radius;
            const int64_t value_offset =
                ((batch_index * heads + head_index) * key_length + key_index) *
                    head_dim +
                feature;
            const float probability =
                expf(score - statistics[0]) / statistics[1];
            accumulated += probability * static_cast<float>(value[value_offset]);
        }
        output[query_offset + feature] = static_cast<scalar_t>(accumulated);
    }
}

template <typename scalar_t>
__global__ void compressed_time_local_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const bool* __restrict__ query_valid,
    const bool* __restrict__ key_valid,
    const scalar_t* __restrict__ output,
    const float* __restrict__ logsumexp,
    scalar_t* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int64_t heads,
    int64_t query_length,
    int64_t key_length,
    int64_t head_dim,
    int64_t compression_ratio,
    int64_t window_radius) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t query_index = row % query_length;
    const int64_t head_index = (row / query_length) % heads;
    const int64_t batch_index = row / (heads * query_length);
    if (!query_valid[batch_index * query_length + query_index]) {
        return;
    }
    const int64_t query_offset = row * head_dim;
    const int64_t center = query_index / compression_ratio;
    const int window_width = static_cast<int>(2 * window_radius + 1);
    const float scale = rsqrtf(static_cast<float>(head_dim));
    extern __shared__ float shared[];
    float* probabilities = shared + 1;
    float* score_gradients = probabilities + window_width;

    float delta = 0.0f;
    for (int64_t feature = lane; feature < head_dim; feature += kWarpSize) {
        delta += static_cast<float>(grad_output[query_offset + feature]) *
            static_cast<float>(output[query_offset + feature]);
    }
    delta = warp_sum(delta);
    if (lane == 0) {
        shared[0] = delta;
    }
    __syncwarp();

    for (int window_index = 0; window_index < window_width; ++window_index) {
        const int64_t key_index = center + window_index - window_radius;
        const bool index_valid = key_index >= 0 && key_index < key_length;
        const bool valid = index_valid &&
            key_valid[batch_index * key_length + key_index];
        if (!valid) {
            if (lane == 0) {
                probabilities[window_index] = 0.0f;
                score_gradients[window_index] = 0.0f;
            }
            __syncwarp();
            continue;
        }
        const int64_t key_offset =
            ((batch_index * heads + head_index) * key_length + key_index) *
            head_dim;
        float score = 0.0f;
        float probability_gradient = 0.0f;
        for (int64_t feature = lane; feature < head_dim; feature += kWarpSize) {
            score += static_cast<float>(query[query_offset + feature]) *
                static_cast<float>(key[key_offset + feature]);
            probability_gradient +=
                static_cast<float>(grad_output[query_offset + feature]) *
                static_cast<float>(value[key_offset + feature]);
        }
        score = warp_sum(score);
        probability_gradient = warp_sum(probability_gradient);
        if (lane == 0) {
            const float probability =
                expf(score * scale - logsumexp[row]);
            probabilities[window_index] = probability;
            score_gradients[window_index] =
                probability * (probability_gradient - shared[0]) * scale;
        }
        __syncwarp();
    }

    for (int64_t feature = lane; feature < head_dim; feature += kWarpSize) {
        float query_gradient = 0.0f;
        for (int window_index = 0; window_index < window_width; ++window_index) {
            if (probabilities[window_index] == 0.0f) {
                continue;
            }
            const int64_t key_index = center + window_index - window_radius;
            const int64_t key_offset =
                ((batch_index * heads + head_index) * key_length + key_index) *
                    head_dim +
                feature;
            const float score_gradient = score_gradients[window_index];
            query_gradient += score_gradient * static_cast<float>(key[key_offset]);
            atomicAdd(
                grad_key + key_offset,
                score_gradient * static_cast<float>(query[query_offset + feature]));
            atomicAdd(
                grad_value + key_offset,
                probabilities[window_index] *
                    static_cast<float>(grad_output[query_offset + feature]));
        }
        grad_query[query_offset + feature] = static_cast<scalar_t>(query_gradient);
    }
}

void validate_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& query_valid,
    const torch::Tensor& key_valid,
    int64_t compression_ratio,
    int64_t window_radius) {
    CHECK_CUDA(query);
    CHECK_CUDA(key);
    CHECK_CUDA(value);
    CHECK_CUDA(query_valid);
    CHECK_CUDA(key_valid);
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(key);
    CHECK_CONTIGUOUS(value);
    CHECK_CONTIGUOUS(query_valid);
    CHECK_CONTIGUOUS(key_valid);
    CHECK_BOOL(query_valid);
    CHECK_BOOL(key_valid);
    TORCH_CHECK(query.dim() == 4, "query must have shape [N,H,T,Dh]");
    TORCH_CHECK(key.dim() == 4, "key must have shape [N,H,Tc,Dh]");
    TORCH_CHECK(value.sizes() == key.sizes(), "value shape must equal key shape");
    TORCH_CHECK(
        query.size(0) == key.size(0) && query.size(1) == key.size(1) &&
            query.size(3) == key.size(3),
        "query/key batch, head, and head dimensions must match");
    TORCH_CHECK(query.numel() > 0 && key.numel() > 0, "tensor dimensions must be positive");
    TORCH_CHECK(
        query.scalar_type() == key.scalar_type() &&
            query.scalar_type() == value.scalar_type(),
        "query, key, and value dtypes must match");
    TORCH_CHECK(
        query.scalar_type() == at::kFloat || query.scalar_type() == at::kHalf ||
            query.scalar_type() == at::kBFloat16,
        "CUDA attention supports float32, float16, and bfloat16");
    TORCH_CHECK(
        query.device() == key.device() && query.device() == value.device() &&
            query.device() == query_valid.device() &&
            query.device() == key_valid.device(),
        "all tensors must be on the same CUDA device");
    TORCH_CHECK(
        query_valid.dim() == 2 && query_valid.size(0) == query.size(0) &&
            query_valid.size(1) == query.size(2),
        "query_valid shape mismatch");
    TORCH_CHECK(
        key_valid.dim() == 2 && key_valid.size(0) == key.size(0) &&
            key_valid.size(1) == key.size(2),
        "key_valid shape mismatch");
    TORCH_CHECK(compression_ratio >= 2, "compression_ratio must be at least 2");
    const int64_t expected_key_length =
        (query.size(2) + compression_ratio - 1) / compression_ratio;
    TORCH_CHECK(
        key.size(2) == expected_key_length,
        "key length must equal ceil(query length / compression_ratio)");
    TORCH_CHECK(
        window_radius >= 0 && window_radius <= kMaximumWindowRadius,
        "window_radius must be in [0, 64]");
}

}  // namespace

std::vector<torch::Tensor> compressed_time_local_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor query_valid,
    torch::Tensor key_valid,
    int64_t compression_ratio,
    int64_t window_radius) {
    validate_inputs(
        query,
        key,
        value,
        query_valid,
        key_valid,
        compression_ratio,
        window_radius);
    const c10::cuda::CUDAGuard device_guard(query.device());
    const int64_t rows = query.size(0) * query.size(1) * query.size(2);
    auto output = torch::zeros_like(query);
    auto logsumexp = torch::zeros(
        {query.size(0), query.size(1), query.size(2)},
        query.options().dtype(torch::kFloat));
    auto invalid_row = torch::zeros({1}, query.options().dtype(torch::kInt));
    const int shared_bytes = static_cast<int>((2 * window_radius + 3) * sizeof(float));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        query.scalar_type(),
        "compressed_time_local_forward_cuda",
        [&] {
            compressed_time_local_forward_kernel<scalar_t>
                <<<rows, kWarpSize, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    value.data_ptr<scalar_t>(),
                    query_valid.data_ptr<bool>(),
                    key_valid.data_ptr<bool>(),
                    output.data_ptr<scalar_t>(),
                    logsumexp.data_ptr<float>(),
                    invalid_row.data_ptr<int>(),
                    query.size(1),
                    query.size(2),
                    key.size(2),
                    query.size(3),
                    compression_ratio,
                    window_radius);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, logsumexp, invalid_row};
}

std::vector<torch::Tensor> compressed_time_local_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor query_valid,
    torch::Tensor key_valid,
    torch::Tensor output,
    torch::Tensor logsumexp,
    int64_t compression_ratio,
    int64_t window_radius) {
    validate_inputs(
        query,
        key,
        value,
        query_valid,
        key_valid,
        compression_ratio,
        window_radius);
    CHECK_CUDA(grad_output);
    CHECK_CUDA(output);
    CHECK_CUDA(logsumexp);
    CHECK_CONTIGUOUS(grad_output);
    CHECK_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(logsumexp);
    TORCH_CHECK(grad_output.sizes() == query.sizes(), "grad_output shape mismatch");
    TORCH_CHECK(output.sizes() == query.sizes(), "output shape mismatch");
    TORCH_CHECK(grad_output.scalar_type() == query.scalar_type(), "grad_output dtype mismatch");
    TORCH_CHECK(output.scalar_type() == query.scalar_type(), "output dtype mismatch");
    TORCH_CHECK(logsumexp.scalar_type() == at::kFloat, "logsumexp must be float32");
    TORCH_CHECK(
        logsumexp.dim() == 3 && logsumexp.size(0) == query.size(0) &&
            logsumexp.size(1) == query.size(1) &&
            logsumexp.size(2) == query.size(2),
        "logsumexp shape mismatch");
    TORCH_CHECK(
        grad_output.device() == query.device() && output.device() == query.device() &&
            logsumexp.device() == query.device(),
        "backward tensors must share the query device");

    const c10::cuda::CUDAGuard device_guard(query.device());
    const int64_t rows = query.size(0) * query.size(1) * query.size(2);
    auto grad_query = torch::zeros_like(query);
    auto float_options = query.options().dtype(torch::kFloat);
    auto grad_key_float = torch::zeros(key.sizes(), float_options);
    auto grad_value_float = torch::zeros(value.sizes(), float_options);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        query.scalar_type(),
        "compressed_time_local_backward_cuda",
        [&] {
            compressed_time_local_backward_kernel<scalar_t>
                <<<
                    rows,
                    kWarpSize,
                    (4 * window_radius + 3) * sizeof(float),
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output.data_ptr<scalar_t>(),
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    value.data_ptr<scalar_t>(),
                    query_valid.data_ptr<bool>(),
                    key_valid.data_ptr<bool>(),
                    output.data_ptr<scalar_t>(),
                    logsumexp.data_ptr<float>(),
                    grad_query.data_ptr<scalar_t>(),
                    grad_key_float.data_ptr<float>(),
                    grad_value_float.data_ptr<float>(),
                    query.size(1),
                    query.size(2),
                    key.size(2),
                    query.size(3),
                    compression_ratio,
                    window_radius);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        grad_query,
        grad_key_float.to(key.scalar_type()),
        grad_value_float.to(value.scalar_type())};
}
