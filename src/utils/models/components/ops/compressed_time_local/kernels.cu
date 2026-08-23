#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Optional.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <array>
#include <algorithm>
#include <vector>

#define CHECK_CUDA(tensor) TORCH_CHECK((tensor).is_cuda(), #tensor " must be CUDA")
#define CHECK_CONTIGUOUS(tensor) \
    TORCH_CHECK((tensor).is_contiguous(), #tensor " must be contiguous")
#define CHECK_BOOL(tensor) \
    TORCH_CHECK((tensor).scalar_type() == at::kBool, #tensor " must be bool")

namespace {

constexpr int kWarpSize = 32;
constexpr int kCachedFeaturesPerLane = 4;
constexpr int64_t kMaximumCachedHeadDim =
    kWarpSize * kCachedFeaturesPerLane;
constexpr int64_t kMaximumWindowRadius = 64;

void check_supported_feature_layout(
    const torch::Tensor& tensor,
    const char* tensor_name) {
    TORCH_CHECK(
        tensor.stride(3) == 1,
        tensor_name,
        " must have unit feature stride");
    std::array<int64_t, 4> dimensions = {0, 1, 2, 3};
    std::sort(
        dimensions.begin(),
        dimensions.end(),
        [&tensor](int64_t left, int64_t right) {
            return tensor.stride(left) < tensor.stride(right);
        });
    int64_t required_span = 1;
    for (const int64_t dimension : dimensions) {
        TORCH_CHECK(
            tensor.stride(dimension) > 0,
            tensor_name,
            " must have positive strides");
        if (tensor.size(dimension) <= 1) {
            continue;
        }
        TORCH_CHECK(
            tensor.stride(dimension) >= required_span,
            tensor_name,
            " must be a supported non-overlapping positive-stride layout");
        required_span +=
            (tensor.size(dimension) - 1) * tensor.stride(dimension);
    }
}

torch::Tensor empty_compact_nhtd_like(const torch::Tensor& tensor) {
    return torch::empty(
               {tensor.size(0), tensor.size(2), tensor.size(1), tensor.size(3)},
               tensor.options())
        .transpose(1, 2);
}

void validate_phasors(
    const torch::Tensor& phasors,
    const torch::Tensor& source,
    const char* phasor_name) {
    CHECK_CUDA(phasors);
    TORCH_CHECK(
        phasors.scalar_type() == at::kFloat,
        phasor_name,
        " must be a float32 real view");
    TORCH_CHECK(!phasors.requires_grad(), phasor_name, " must not require gradients");
    TORCH_CHECK(phasors.dim() == 5, phasor_name, " must have rank 5");
    TORCH_CHECK(
        phasors.device() == source.device(),
        phasor_name,
        " must be on the source device");
    TORCH_CHECK(
        phasors.size(0) == 1 || phasors.size(0) == source.size(0),
        phasor_name,
        " batch dimension must be 1 or match the source");
    TORCH_CHECK(
        phasors.size(1) == source.size(2),
        phasor_name,
        " sequence dimension must match the source");
    TORCH_CHECK(
        phasors.size(2) == 1 || phasors.size(2) == source.size(1),
        phasor_name,
        " head dimension must be 1 or match the source");
    TORCH_CHECK(
        source.size(3) % 2 == 0 && phasors.size(3) == source.size(3) / 2,
        phasor_name,
        " pair dimension must equal head_dim / 2");
    TORCH_CHECK(phasors.size(4) == 2, phasor_name, " final dimension must be 2");
    for (int64_t dimension = 0; dimension < phasors.dim(); ++dimension) {
        TORCH_CHECK(
            phasors.stride(dimension) >= 0,
            phasor_name,
            " must have non-negative strides");
    }
    TORCH_CHECK(
        phasors.stride(4) == 1,
        phasor_name,
        " real/imaginary component stride must be 1");
}

template <typename scalar_t>
__device__ __forceinline__ float load_rotated_component(
    const scalar_t* __restrict__ source,
    int64_t source_offset,
    int64_t feature,
    const float* __restrict__ phasors,
    int64_t batch_index,
    int64_t head_index,
    int64_t sequence_index,
    int64_t phasor_batches,
    int64_t phasor_heads,
    int64_t phasor_batch_stride,
    int64_t phasor_time_stride,
    int64_t phasor_head_stride,
    int64_t phasor_pair_stride) {
    if (phasors == nullptr) {
        return static_cast<float>(source[source_offset + feature]);
    }
    const int64_t pair = feature / 2;
    const int64_t pair_feature = pair * 2;
    const int64_t phasor_offset =
        (phasor_batches == 1 ? 0 : batch_index) * phasor_batch_stride +
        sequence_index * phasor_time_stride +
        (phasor_heads == 1 ? 0 : head_index) * phasor_head_stride +
        pair * phasor_pair_stride;
    const float real = static_cast<float>(source[source_offset + pair_feature]);
    const float imaginary =
        static_cast<float>(source[source_offset + pair_feature + 1]);
    const float cosine = phasors[phasor_offset];
    const float sine = phasors[phasor_offset + 1];
    const float rotated = feature % 2 == 0
        ? real * cosine - imaginary * sine
        : real * sine + imaginary * cosine;
    return static_cast<float>(static_cast<scalar_t>(rotated));
}

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
    const float* __restrict__ query_phasors,
    const float* __restrict__ key_phasors,
    const bool* __restrict__ query_valid,
    const bool* __restrict__ key_valid,
    scalar_t* __restrict__ output,
    float* __restrict__ logsumexp,
    int* __restrict__ invalid_row,
    int heads,
    int key_heads,
    int query_length,
    int key_length,
    int head_dim,
    int compression_ratio,
    int window_radius,
    int64_t query_batch_stride,
    int64_t query_head_stride,
    int64_t query_time_stride,
    int64_t output_batch_stride,
    int64_t output_head_stride,
    int64_t output_time_stride,
    int64_t query_phasor_batches,
    int64_t query_phasor_heads,
    int64_t query_phasor_batch_stride,
    int64_t query_phasor_time_stride,
    int64_t query_phasor_head_stride,
    int64_t query_phasor_pair_stride,
    int64_t key_phasor_batches,
    int64_t key_phasor_heads,
    int64_t key_phasor_batch_stride,
    int64_t key_phasor_time_stride,
    int64_t key_phasor_head_stride,
    int64_t key_phasor_pair_stride) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int query_index = row % query_length;
    const int head_index = (row / query_length) % heads;
    const int key_head_index = key_heads == 1 ? 0 : head_index;
    const int batch_index = row / (heads * query_length);
    const int64_t query_offset =
        batch_index * query_batch_stride + head_index * query_head_stride +
        query_index * query_time_stride;
    const int64_t output_offset =
        batch_index * output_batch_stride + head_index * output_head_stride +
        query_index * output_time_stride;
    if (!query_valid[batch_index * query_length + query_index]) {
        for (int feature = lane; feature < head_dim; feature += kWarpSize) {
            output[output_offset + feature] = static_cast<scalar_t>(0.0f);
        }
        if (lane == 0) {
            logsumexp[row] = 0.0f;
        }
        return;
    }

    const int center = query_index / compression_ratio;
    const int window_width = static_cast<int>(2 * window_radius + 1);
    extern __shared__ float shared[];
    float* scores = shared;
    float* statistics = shared + window_width;
    float running_max = -INFINITY;
    float denominator = 0.0f;
    bool any_valid = false;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float cached_query[kCachedFeaturesPerLane];
    if (head_dim <= kMaximumCachedHeadDim) {
        for (int slot = 0; slot < kCachedFeaturesPerLane; ++slot) {
            const int feature = lane + slot * kWarpSize;
            if (feature < head_dim) {
                cached_query[slot] = load_rotated_component(
                    query,
                    query_offset,
                    feature,
                    query_phasors,
                    batch_index,
                    head_index,
                    query_index,
                    query_phasor_batches,
                    query_phasor_heads,
                    query_phasor_batch_stride,
                    query_phasor_time_stride,
                    query_phasor_head_stride,
                    query_phasor_pair_stride);
            }
        }
    }

    for (int window_index = 0; window_index < window_width; ++window_index) {
        const int key_index = center + window_index - window_radius;
        const bool index_valid = key_index >= 0 && key_index < key_length;
        const bool valid = index_valid &&
            key_valid[batch_index * key_length + key_index];
        float dot = 0.0f;
        if (valid) {
            const int64_t key_offset =
                ((static_cast<int64_t>(batch_index) * key_heads + key_head_index) *
                     key_length +
                 key_index) *
                head_dim;
            for (int feature = lane; feature < head_dim; feature += kWarpSize) {
                const float rotated_query = head_dim <= kMaximumCachedHeadDim
                    ? cached_query[feature / kWarpSize]
                    : load_rotated_component(
                          query,
                          query_offset,
                          feature,
                          query_phasors,
                          batch_index,
                          head_index,
                          query_index,
                          query_phasor_batches,
                          query_phasor_heads,
                          query_phasor_batch_stride,
                          query_phasor_time_stride,
                          query_phasor_head_stride,
                          query_phasor_pair_stride);
                const float rotated_key = load_rotated_component(
                    key,
                    key_offset,
                    feature,
                    key_phasors,
                    batch_index,
                    key_head_index,
                    key_index,
                    key_phasor_batches,
                    key_phasor_heads,
                    key_phasor_batch_stride,
                    key_phasor_time_stride,
                    key_phasor_head_stride,
                    key_phasor_pair_stride);
                dot += rotated_query * rotated_key;
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
            logsumexp[row] = 0.0f;
            atomicExch(invalid_row, 1);
        } else {
            logsumexp[row] = running_max + logf(denominator);
        }
    }
    __syncwarp();
    if (statistics[1] == 0.0f) {
        for (int feature = lane; feature < head_dim; feature += kWarpSize) {
            output[output_offset + feature] = static_cast<scalar_t>(0.0f);
        }
        return;
    }
    for (int window_index = lane; window_index < window_width;
         window_index += kWarpSize) {
        const float score = scores[window_index];
        scores[window_index] = score == -INFINITY
            ? 0.0f
            : expf(score - statistics[0]) / statistics[1];
    }
    __syncwarp();

    for (int feature = lane; feature < head_dim; feature += kWarpSize) {
        float accumulated = 0.0f;
        for (int window_index = 0; window_index < window_width; ++window_index) {
            const float probability = scores[window_index];
            if (probability == 0.0f) {
                continue;
            }
            const int key_index = center + window_index - window_radius;
            const int64_t value_offset =
                ((static_cast<int64_t>(batch_index) * key_heads + key_head_index) *
                     key_length +
                 key_index) *
                    head_dim +
                feature;
            accumulated += probability * static_cast<float>(value[value_offset]);
        }
        output[output_offset + feature] = static_cast<scalar_t>(accumulated);
    }
}

template <typename scalar_t>
__global__ void compressed_time_local_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const float* __restrict__ query_phasors,
    const float* __restrict__ key_phasors,
    const bool* __restrict__ query_valid,
    const bool* __restrict__ key_valid,
    const float* __restrict__ logsumexp,
    scalar_t* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    int64_t heads,
    int64_t key_heads,
    int64_t query_length,
    int64_t key_length,
    int64_t head_dim,
    int64_t compression_ratio,
    int64_t window_radius,
    int64_t grad_output_batch_stride,
    int64_t grad_output_head_stride,
    int64_t grad_output_time_stride,
    int64_t query_batch_stride,
    int64_t query_head_stride,
    int64_t query_time_stride,
    int64_t grad_query_batch_stride,
    int64_t grad_query_head_stride,
    int64_t grad_query_time_stride,
    int64_t query_phasor_batches,
    int64_t query_phasor_heads,
    int64_t query_phasor_batch_stride,
    int64_t query_phasor_time_stride,
    int64_t query_phasor_head_stride,
    int64_t query_phasor_pair_stride,
    int64_t key_phasor_batches,
    int64_t key_phasor_heads,
    int64_t key_phasor_batch_stride,
    int64_t key_phasor_time_stride,
    int64_t key_phasor_head_stride,
    int64_t key_phasor_pair_stride) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t query_index = row % query_length;
    const int64_t head_index = (row / query_length) % heads;
    const int64_t key_head_index = key_heads == 1 ? 0 : head_index;
    const int64_t batch_index = row / (heads * query_length);
    const int64_t grad_query_offset =
        batch_index * grad_query_batch_stride +
        head_index * grad_query_head_stride +
        query_index * grad_query_time_stride;
    if (!query_valid[batch_index * query_length + query_index]) {
        for (int64_t feature = lane; feature < head_dim; feature += kWarpSize) {
            grad_query[grad_query_offset + feature] = static_cast<scalar_t>(0.0f);
        }
        return;
    }
    const int64_t grad_output_offset =
        batch_index * grad_output_batch_stride +
        head_index * grad_output_head_stride +
        query_index * grad_output_time_stride;
    const int64_t query_offset =
        batch_index * query_batch_stride + head_index * query_head_stride +
        query_index * query_time_stride;
    const int64_t center = query_index / compression_ratio;
    const int window_width = static_cast<int>(2 * window_radius + 1);
    const float scale = rsqrtf(static_cast<float>(head_dim));
    extern __shared__ float shared[];
    float* probabilities = shared;
    float* probability_gradients = probabilities + window_width;
    float* scaled_score_gradients = probability_gradients;
    float cached_query[kCachedFeaturesPerLane];
    float cached_grad_output[kCachedFeaturesPerLane];
    if (head_dim <= kMaximumCachedHeadDim) {
        for (int slot = 0; slot < kCachedFeaturesPerLane; ++slot) {
            const int64_t feature = lane + slot * kWarpSize;
            if (feature < head_dim) {
                cached_query[slot] = load_rotated_component(
                    query,
                    query_offset,
                    feature,
                    query_phasors,
                    batch_index,
                    head_index,
                    query_index,
                    query_phasor_batches,
                    query_phasor_heads,
                    query_phasor_batch_stride,
                    query_phasor_time_stride,
                    query_phasor_head_stride,
                    query_phasor_pair_stride);
                cached_grad_output[slot] =
                    static_cast<float>(grad_output[grad_output_offset + feature]);
            }
        }
    }

    for (int window_index = 0; window_index < window_width; ++window_index) {
        const int64_t key_index = center + window_index - window_radius;
        const bool index_valid = key_index >= 0 && key_index < key_length;
        const bool valid = index_valid &&
            key_valid[batch_index * key_length + key_index];
        if (!valid) {
            if (lane == 0) {
                probabilities[window_index] = 0.0f;
                probability_gradients[window_index] = 0.0f;
            }
            __syncwarp();
            continue;
        }
        const int64_t key_offset =
            ((static_cast<int64_t>(batch_index) * key_heads + key_head_index) *
                 key_length +
             key_index) *
            head_dim;
        float score = 0.0f;
        float probability_gradient = 0.0f;
        for (int64_t feature = lane; feature < head_dim; feature += kWarpSize) {
            const int64_t feature_slot = feature / kWarpSize;
            const float rotated_query = head_dim <= kMaximumCachedHeadDim
                ? cached_query[feature_slot]
                : load_rotated_component(
                      query,
                      query_offset,
                      feature,
                      query_phasors,
                      batch_index,
                      head_index,
                      query_index,
                      query_phasor_batches,
                      query_phasor_heads,
                      query_phasor_batch_stride,
                      query_phasor_time_stride,
                      query_phasor_head_stride,
                      query_phasor_pair_stride);
            const float rotated_key = load_rotated_component(
                key,
                key_offset,
                feature,
                key_phasors,
                batch_index,
                key_head_index,
                key_index,
                key_phasor_batches,
                key_phasor_heads,
                key_phasor_batch_stride,
                key_phasor_time_stride,
                key_phasor_head_stride,
                key_phasor_pair_stride);
            score += rotated_query * rotated_key;
            const float output_gradient = head_dim <= kMaximumCachedHeadDim
                ? cached_grad_output[feature_slot]
                : static_cast<float>(grad_output[grad_output_offset + feature]);
            probability_gradient +=
                output_gradient * static_cast<float>(value[key_offset + feature]);
        }
        score = warp_sum(score);
        probability_gradient = warp_sum(probability_gradient);
        if (lane == 0) {
            const float probability =
                expf(score * scale - logsumexp[row]);
            probabilities[window_index] = probability;
            probability_gradients[window_index] = probability_gradient;
        }
        __syncwarp();
    }

    if (lane == 0) {
        float delta = 0.0f;
        for (int window_index = 0; window_index < window_width; ++window_index) {
            delta = fmaf(
                probabilities[window_index],
                probability_gradients[window_index],
                delta);
        }
        for (int window_index = 0; window_index < window_width; ++window_index) {
            const float score_gradient =
                probabilities[window_index] *
                (probability_gradients[window_index] - delta);
            scaled_score_gradients[window_index] = score_gradient * scale;
        }
    }
    __syncwarp();

    for (int64_t feature = lane; feature < head_dim; feature += kWarpSize) {
        float query_gradient = 0.0f;
        for (int window_index = 0; window_index < window_width; ++window_index) {
            if (probabilities[window_index] == 0.0f) {
                continue;
            }
            const int64_t key_index = center + window_index - window_radius;
            const int64_t key_offset =
                ((static_cast<int64_t>(batch_index) * key_heads + key_head_index) *
                     key_length +
                 key_index) *
                    head_dim +
                feature;
            const int64_t key_row_offset = key_offset - feature;
            const float scaled_score_gradient = scaled_score_gradients[window_index];
            const int64_t feature_slot = feature / kWarpSize;
            const float rotated_query = head_dim <= kMaximumCachedHeadDim
                ? cached_query[feature_slot]
                : load_rotated_component(
                      query,
                      query_offset,
                      feature,
                      query_phasors,
                      batch_index,
                      head_index,
                      query_index,
                      query_phasor_batches,
                      query_phasor_heads,
                      query_phasor_batch_stride,
                      query_phasor_time_stride,
                      query_phasor_head_stride,
                      query_phasor_pair_stride);
            const float rotated_key = load_rotated_component(
                key,
                key_row_offset,
                feature,
                key_phasors,
                batch_index,
                key_head_index,
                key_index,
                key_phasor_batches,
                key_phasor_heads,
                key_phasor_batch_stride,
                key_phasor_time_stride,
                key_phasor_head_stride,
                key_phasor_pair_stride);
            query_gradient +=
                scaled_score_gradient * rotated_key;
            atomicAdd(
                grad_key + key_offset,
                scaled_score_gradient * rotated_query);
            atomicAdd(
                grad_value + key_offset,
                probabilities[window_index] *
                    (head_dim <= kMaximumCachedHeadDim
                         ? cached_grad_output[feature_slot]
                         : static_cast<float>(
                               grad_output[grad_output_offset + feature])));
        }
        grad_query[grad_query_offset + feature] =
            static_cast<scalar_t>(query_gradient);
    }
}

template <typename scalar_t>
__device__ __forceinline__ void store_inverse_rotated_pair(
    const scalar_t* __restrict__ rotated_gradient,
    scalar_t* __restrict__ raw_gradient,
    int64_t rotated_offset,
    int64_t raw_offset,
    const float* __restrict__ phasors,
    int64_t batch_index,
    int64_t head_index,
    int64_t sequence_index,
    int64_t pair_index,
    int64_t phasor_batches,
    int64_t phasor_heads,
    int64_t phasor_batch_stride,
    int64_t phasor_time_stride,
    int64_t phasor_head_stride,
    int64_t phasor_pair_stride) {
    const int64_t phasor_offset =
        (phasor_batches == 1 ? 0 : batch_index) * phasor_batch_stride +
        sequence_index * phasor_time_stride +
        (phasor_heads == 1 ? 0 : head_index) * phasor_head_stride +
        pair_index * phasor_pair_stride;
    const float cosine = phasors[phasor_offset];
    const float sine = phasors[phasor_offset + 1];
    const float rotated_real =
        static_cast<float>(rotated_gradient[rotated_offset]);
    const float rotated_imaginary =
        static_cast<float>(rotated_gradient[rotated_offset + 1]);
    raw_gradient[raw_offset] = static_cast<scalar_t>(
        rotated_real * cosine + rotated_imaginary * sine);
    raw_gradient[raw_offset + 1] = static_cast<scalar_t>(
        -rotated_real * sine + rotated_imaginary * cosine);
}

template <typename scalar_t>
__global__ void inverse_rope_gradients_kernel(
    const scalar_t* __restrict__ rotated_grad_query,
    const scalar_t* __restrict__ rotated_grad_key,
    scalar_t* __restrict__ raw_grad_query,
    scalar_t* __restrict__ raw_grad_key,
    const float* __restrict__ query_phasors,
    const float* __restrict__ key_phasors,
    int64_t query_pairs_count,
    int64_t key_pairs_count,
    int64_t query_heads,
    int64_t query_length,
    int64_t key_heads,
    int64_t key_length,
    int64_t pairs_per_head,
    int64_t rotated_query_batch_stride,
    int64_t rotated_query_head_stride,
    int64_t rotated_query_time_stride,
    int64_t raw_query_batch_stride,
    int64_t raw_query_head_stride,
    int64_t raw_query_time_stride,
    int64_t query_phasor_batches,
    int64_t query_phasor_heads,
    int64_t query_phasor_batch_stride,
    int64_t query_phasor_time_stride,
    int64_t query_phasor_head_stride,
    int64_t query_phasor_pair_stride,
    int64_t key_phasor_batches,
    int64_t key_phasor_heads,
    int64_t key_phasor_batch_stride,
    int64_t key_phasor_time_stride,
    int64_t key_phasor_head_stride,
    int64_t key_phasor_pair_stride) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < query_pairs_count) {
        int64_t remaining = index;
        const int64_t pair_index = remaining % pairs_per_head;
        remaining /= pairs_per_head;
        const int64_t sequence_index = remaining % query_length;
        remaining /= query_length;
        const int64_t head_index = remaining % query_heads;
        const int64_t batch_index = remaining / query_heads;
        const int64_t rotated_offset =
            batch_index * rotated_query_batch_stride +
            head_index * rotated_query_head_stride +
            sequence_index * rotated_query_time_stride + pair_index * 2;
        const int64_t raw_offset =
            batch_index * raw_query_batch_stride +
            head_index * raw_query_head_stride +
            sequence_index * raw_query_time_stride + pair_index * 2;
        store_inverse_rotated_pair(
            rotated_grad_query,
            raw_grad_query,
            rotated_offset,
            raw_offset,
            query_phasors,
            batch_index,
            head_index,
            sequence_index,
            pair_index,
            query_phasor_batches,
            query_phasor_heads,
            query_phasor_batch_stride,
            query_phasor_time_stride,
            query_phasor_head_stride,
            query_phasor_pair_stride);
        return;
    }
    const int64_t key_index = index - query_pairs_count;
    if (key_index >= key_pairs_count) {
        return;
    }
    int64_t remaining = key_index;
    const int64_t pair_index = remaining % pairs_per_head;
    remaining /= pairs_per_head;
    const int64_t sequence_index = remaining % key_length;
    remaining /= key_length;
    const int64_t head_index = remaining % key_heads;
    const int64_t batch_index = remaining / key_heads;
    const int64_t contiguous_offset =
        (((batch_index * key_heads + head_index) * key_length + sequence_index) *
             pairs_per_head +
         pair_index) *
        2;
    store_inverse_rotated_pair(
        rotated_grad_key,
        raw_grad_key,
        contiguous_offset,
        contiguous_offset,
        key_phasors,
        batch_index,
        head_index,
        sequence_index,
        pair_index,
        key_phasor_batches,
        key_phasor_heads,
        key_phasor_batch_stride,
        key_phasor_time_stride,
        key_phasor_head_stride,
        key_phasor_pair_stride);
}

void validate_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& query_valid,
    const torch::Tensor& key_valid,
    const c10::optional<torch::Tensor>& query_phasors_real,
    const c10::optional<torch::Tensor>& key_phasors_real,
    int64_t compression_ratio,
    int64_t window_radius) {
    CHECK_CUDA(query);
    CHECK_CUDA(key);
    CHECK_CUDA(value);
    CHECK_CUDA(query_valid);
    CHECK_CUDA(key_valid);
    CHECK_CONTIGUOUS(key);
    CHECK_CONTIGUOUS(value);
    CHECK_CONTIGUOUS(query_valid);
    CHECK_CONTIGUOUS(key_valid);
    CHECK_BOOL(query_valid);
    CHECK_BOOL(key_valid);
    TORCH_CHECK(query.dim() == 4, "query must have shape [N,H,T,Dh]");
    TORCH_CHECK(key.dim() == 4, "key must have shape [N,H,Tc,Dh]");
    check_supported_feature_layout(query, "query");
    TORCH_CHECK(value.sizes() == key.sizes(), "value shape must equal key shape");
    TORCH_CHECK(
        query.size(0) == key.size(0) && query.size(3) == key.size(3) &&
            (key.size(1) == 1 || query.size(1) == key.size(1)),
        "query/key batch and head dimensions must match, and key heads must be 1 "
        "or equal query heads");
    TORCH_CHECK(query.numel() > 0 && key.numel() > 0, "tensor dimensions must be positive");
    constexpr int64_t maximum_int = std::numeric_limits<int>::max();
    TORCH_CHECK(
        query.size(0) <= maximum_int && query.size(1) <= maximum_int &&
            query.size(2) <= maximum_int && query.size(3) <= maximum_int &&
            key.size(1) <= maximum_int && key.size(2) <= maximum_int &&
            query.numel() / query.size(3) <= maximum_int &&
            compression_ratio <= maximum_int,
        "CUDA attention dimensions, row count, and compression ratio must fit int32");
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
    TORCH_CHECK(
        query_phasors_real.has_value() == key_phasors_real.has_value(),
        "query_phasors_real and key_phasors_real must both be present or absent");
    if (query_phasors_real.has_value()) {
        validate_phasors(query_phasors_real.value(), query, "query_phasors_real");
        validate_phasors(key_phasors_real.value(), key, "key_phasors_real");
    }
}

}  // namespace

std::vector<torch::Tensor> compressed_time_local_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor query_valid,
    torch::Tensor key_valid,
    c10::optional<torch::Tensor> query_phasors_real,
    c10::optional<torch::Tensor> key_phasors_real,
    int64_t compression_ratio,
    int64_t window_radius) {
    validate_inputs(
        query,
        key,
        value,
        query_valid,
        key_valid,
        query_phasors_real,
        key_phasors_real,
        compression_ratio,
        window_radius);
    const c10::cuda::CUDAGuard device_guard(query.device());
    const int64_t rows = query.size(0) * query.size(1) * query.size(2);
    auto output = empty_compact_nhtd_like(query);
    auto logsumexp = torch::empty(
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
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().data_ptr<float>()
                        : nullptr,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().data_ptr<float>()
                        : nullptr,
                    query_valid.data_ptr<bool>(),
                    key_valid.data_ptr<bool>(),
                    output.data_ptr<scalar_t>(),
                    logsumexp.data_ptr<float>(),
                    invalid_row.data_ptr<int>(),
                    query.size(1),
                    key.size(1),
                    query.size(2),
                    key.size(2),
                    query.size(3),
                    compression_ratio,
                    window_radius,
                    query.stride(0),
                    query.stride(1),
                    query.stride(2),
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().size(0)
                        : 0,
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().size(2)
                        : 0,
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().stride(0)
                        : 0,
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().stride(1)
                        : 0,
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().stride(2)
                        : 0,
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().stride(3)
                        : 0,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().size(0)
                        : 0,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().size(2)
                        : 0,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().stride(0)
                        : 0,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().stride(1)
                        : 0,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().stride(2)
                        : 0,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().stride(3)
                        : 0);
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
    torch::Tensor logsumexp,
    c10::optional<torch::Tensor> query_phasors_real,
    c10::optional<torch::Tensor> key_phasors_real,
    int64_t compression_ratio,
    int64_t window_radius) {
    validate_inputs(
        query,
        key,
        value,
        query_valid,
        key_valid,
        query_phasors_real,
        key_phasors_real,
        compression_ratio,
        window_radius);
    CHECK_CUDA(grad_output);
    CHECK_CUDA(logsumexp);
    CHECK_CONTIGUOUS(logsumexp);
    TORCH_CHECK(grad_output.sizes() == query.sizes(), "grad_output shape mismatch");
    TORCH_CHECK(grad_output.scalar_type() == query.scalar_type(), "grad_output dtype mismatch");
    check_supported_feature_layout(grad_output, "grad_output");
    TORCH_CHECK(logsumexp.scalar_type() == at::kFloat, "logsumexp must be float32");
    TORCH_CHECK(
        logsumexp.dim() == 3 && logsumexp.size(0) == query.size(0) &&
            logsumexp.size(1) == query.size(1) &&
            logsumexp.size(2) == query.size(2),
        "logsumexp shape mismatch");
    TORCH_CHECK(
        grad_output.device() == query.device() && logsumexp.device() == query.device(),
        "backward tensors must share the query device");

    const c10::cuda::CUDAGuard device_guard(query.device());
    const int64_t rows = query.size(0) * query.size(1) * query.size(2);
    auto grad_query = empty_compact_nhtd_like(query);
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
                    (4 * window_radius + 2) * sizeof(float),
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output.data_ptr<scalar_t>(),
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    value.data_ptr<scalar_t>(),
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().data_ptr<float>()
                        : nullptr,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().data_ptr<float>()
                        : nullptr,
                    query_valid.data_ptr<bool>(),
                    key_valid.data_ptr<bool>(),
                    logsumexp.data_ptr<float>(),
                    grad_query.data_ptr<scalar_t>(),
                    grad_key_float.data_ptr<float>(),
                    grad_value_float.data_ptr<float>(),
                    query.size(1),
                    key.size(1),
                    query.size(2),
                    key.size(2),
                    query.size(3),
                    compression_ratio,
                    window_radius,
                    grad_output.stride(0),
                    grad_output.stride(1),
                    grad_output.stride(2),
                    query.stride(0),
                    query.stride(1),
                    query.stride(2),
                    grad_query.stride(0),
                    grad_query.stride(1),
                    grad_query.stride(2),
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().size(0)
                        : 0,
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().size(2)
                        : 0,
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().stride(0)
                        : 0,
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().stride(1)
                        : 0,
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().stride(2)
                        : 0,
                    query_phasors_real.has_value()
                        ? query_phasors_real.value().stride(3)
                        : 0,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().size(0)
                        : 0,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().size(2)
                        : 0,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().stride(0)
                        : 0,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().stride(1)
                        : 0,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().stride(2)
                        : 0,
                    key_phasors_real.has_value()
                        ? key_phasors_real.value().stride(3)
                        : 0);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto grad_key = grad_key_float.to(key.scalar_type());
    auto grad_value = grad_value_float.to(value.scalar_type());
    if (!query_phasors_real.has_value()) {
        return {grad_query, grad_key, grad_value};
    }

    auto raw_grad_query = empty_compact_nhtd_like(query);
    auto raw_grad_key = torch::empty(key.sizes(), key.options());
    const int64_t pairs_per_head = query.size(3) / 2;
    const int64_t query_pairs_count =
        query.size(0) * query.size(1) * query.size(2) * pairs_per_head;
    const int64_t key_pairs_count =
        key.size(0) * key.size(1) * key.size(2) * pairs_per_head;
    const int64_t total_pairs = query_pairs_count + key_pairs_count;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        query.scalar_type(),
        "inverse_rope_gradients_cuda",
        [&] {
            inverse_rope_gradients_kernel<scalar_t>
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    grad_query.data_ptr<scalar_t>(),
                    grad_key.data_ptr<scalar_t>(),
                    raw_grad_query.data_ptr<scalar_t>(),
                    raw_grad_key.data_ptr<scalar_t>(),
                    query_phasors_real.value().data_ptr<float>(),
                    key_phasors_real.value().data_ptr<float>(),
                    query_pairs_count,
                    key_pairs_count,
                    query.size(1),
                    query.size(2),
                    key.size(1),
                    key.size(2),
                    pairs_per_head,
                    grad_query.stride(0),
                    grad_query.stride(1),
                    grad_query.stride(2),
                    raw_grad_query.stride(0),
                    raw_grad_query.stride(1),
                    raw_grad_query.stride(2),
                    query_phasors_real.value().size(0),
                    query_phasors_real.value().size(2),
                    query_phasors_real.value().stride(0),
                    query_phasors_real.value().stride(1),
                    query_phasors_real.value().stride(2),
                    query_phasors_real.value().stride(3),
                    key_phasors_real.value().size(0),
                    key_phasors_real.value().size(2),
                    key_phasors_real.value().stride(0),
                    key_phasors_real.value().stride(1),
                    key_phasors_real.value().stride(2),
                    key_phasors_real.value().stride(3));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {raw_grad_query, raw_grad_key, grad_value};
}
