/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Hipsr Slice keeps starts, ends, axes, and steps on the host. This wrapper
// reads those windows in place and launches the same slice kernel without a
// device-to-host copy or stream synchronization.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <hip/hip_runtime.h>

static constexpr int kHipsrSliceRuntimeMaxRank = 8;

static int hipsr_slice_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

int wrap_hipsr_slice(RuntimeState *state, void *data, void *starts, void *ends,
                     void *axes, void *steps, void *output,
                     const int64_t *data_shape, int64_t data_rank,
                     const int64_t *output_shape, int64_t output_rank,
                     int64_t starts_num_elements, int64_t axes_num_elements,
                     int64_t steps_num_elements, int64_t data_type) {
  OP_PROFILE(
      "slice",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:K%lld:%s", (long long)data_rank,
                 (long long)starts_num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  if (!state || !starts || !ends || !output || !data_shape || !output_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_hipsr_slice: null required argument\n");
    return -1;
  }
  if (data_rank <= 0 || data_rank != output_rank) {
    fprintf(stderr,
            "[REAL] wrap_hipsr_slice: invalid ranks (data_rank=%lld, "
            "output_rank=%lld)\n",
            (long long)data_rank, (long long)output_rank);
    return -1;
  }

  int64_t data_num_elements = 1;
  for (int d = 0; d < data_rank; ++d) {
    data_num_elements *= data_shape[d];
  }
  if (!data || data_num_elements == 0) {
    int64_t out_num_elements = 1;
    for (int d = 0; d < output_rank; ++d) {
      out_num_elements *= output_shape[d];
    }
    int64_t elem_size = hipdnn_ep_datatype_size(data_type);
    if (elem_size <= 0) {
      fprintf(stderr,
              "[REAL] wrap_hipsr_slice: empty-input path -- unsupported "
              "data_type=%s(%lld)\n",
              hipdnn_ep_datatype_name(data_type), (long long)data_type);
      return -1;
    }
    if (output && out_num_elements > 0) {
      hipStream_t stream =
          static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
      hipError_t err = hipMemsetAsync(output, 0,
                                      static_cast<size_t>(out_num_elements) *
                                          static_cast<size_t>(elem_size),
                                      stream);
      if (err != hipSuccess) {
        fprintf(stderr,
                "[REAL] wrap_hipsr_slice: empty-input hipMemsetAsync "
                "failed: %s\n",
                hipGetErrorString(err));
        return -1;
      }
    }
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_hipsr_slice: empty input (data=%p "
        "data_num_elements=%lld) -- zeroed output and returning success\n",
        data, (long long)data_num_elements);
    return 0;
  }
  if (data_rank > kHipsrSliceRuntimeMaxRank) {
    fprintf(stderr, "[REAL] wrap_hipsr_slice: data_rank=%lld exceeds max %d\n",
            (long long)data_rank, kHipsrSliceRuntimeMaxRank);
    return -1;
  }
  if (starts_num_elements <= 0) {
    fprintf(stderr,
            "[REAL] wrap_hipsr_slice: starts_num_elements=%lld must be > 0\n",
            (long long)starts_num_elements);
    return -1;
  }

  int hip_dtype = hipsr_slice_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_hipsr_slice: unsupported data_type=%s(%lld) "
            "(supported: f16, f32, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  const int64_t K = starts_num_elements;
  if (axes && axes_num_elements > 0 && axes_num_elements != K) {
    fprintf(stderr,
            "[REAL] wrap_hipsr_slice: axes_num_elements(%lld) != "
            "starts_num_elements(%lld)\n",
            (long long)axes_num_elements, (long long)K);
    return -1;
  }
  if (steps && steps_num_elements > 0 && steps_num_elements != K) {
    fprintf(stderr,
            "[REAL] wrap_hipsr_slice: steps_num_elements(%lld) != "
            "starts_num_elements(%lld)\n",
            (long long)steps_num_elements, (long long)K);
    return -1;
  }

  const auto *starts_host = static_cast<const int64_t *>(starts);
  const auto *ends_host = static_cast<const int64_t *>(ends);
  const auto *axes_host =
      axes && axes_num_elements > 0 ? static_cast<const int64_t *>(axes)
                                   : nullptr;
  const auto *steps_host =
      steps && steps_num_elements > 0 ? static_cast<const int64_t *>(steps)
                                     : nullptr;

  int64_t start_per_axis[kHipsrSliceRuntimeMaxRank] = {};
  int64_t step_per_axis[kHipsrSliceRuntimeMaxRank];
  for (int d = 0; d < data_rank; ++d) {
    step_per_axis[d] = 1;
  }

  bool axis_set[kHipsrSliceRuntimeMaxRank] = {};
  for (int64_t k = 0; k < K; ++k) {
    int64_t axis = axes_host ? axes_host[k] : k;
    if (axis < 0) {
      axis += data_rank;
    }
    if (axis < 0 || axis >= data_rank) {
      fprintf(stderr,
              "[REAL] wrap_hipsr_slice: axis=%lld out of range [0, %lld)\n",
              (long long)axis, (long long)data_rank);
      return -1;
    }
    if (axis_set[axis]) {
      fprintf(stderr, "[REAL] wrap_hipsr_slice: duplicate axis %lld\n",
              (long long)axis);
      return -1;
    }
    axis_set[axis] = true;

    int64_t dim = data_shape[axis];
    int64_t start = starts_host[k];
    int64_t end = ends_host[k];
    int64_t step = steps_host ? steps_host[k] : 1;
    if (step == 0) {
      fprintf(stderr, "[REAL] wrap_hipsr_slice: zero step on axis %lld\n",
              (long long)axis);
      return -1;
    }

    if (start < 0) {
      start += dim;
    }
    if (end < 0) {
      end += dim;
    }
    if (step > 0) {
      start = std::clamp<int64_t>(start, 0, dim);
    } else {
      start = std::clamp<int64_t>(start, 0, dim - 1);
    }

    start_per_axis[axis] = start;
    step_per_axis[axis] = step;
  }

  int64_t logical_extent[kHipsrSliceRuntimeMaxRank];
  for (int d = 0; d < data_rank; ++d) {
    logical_extent[d] = output_shape[d];
  }

  for (int d = 0; d < data_rank; ++d) {
    if (!axis_set[d]) {
      continue;
    }
    int64_t dim = data_shape[d];
    int64_t start = start_per_axis[d];
    int64_t step = step_per_axis[d];
    int64_t end = 0;
    for (int64_t k = 0; k < K; ++k) {
      int64_t axis = axes_host ? axes_host[k] : k;
      if (axis < 0) {
        axis += data_rank;
      }
      if (axis == d) {
        end = ends_host[k];
        break;
      }
    }
    if (end < 0) {
      end += dim;
    }
    if (step > 0) {
      end = std::clamp<int64_t>(end, 0, dim);
    } else {
      end = std::clamp<int64_t>(end, -1, dim - 1);
    }

    int64_t expected;
    if (step > 0) {
      expected = (end - start + step - 1) / step;
    } else {
      expected = (end - start + step + 1) / step;
    }
    expected = std::max<int64_t>(expected, 0);
    if (expected > output_shape[d]) {
      fprintf(stderr,
              "[REAL] wrap_hipsr_slice: derived output extent on axis %d "
              "(%lld) > IR output_shape (%lld) -- aborting "
              "(start=%lld end=%lld step=%lld dim=%lld data_rank=%lld "
              "K=%lld)\n",
              d, (long long)expected, (long long)output_shape[d],
              (long long)start, (long long)end, (long long)step, (long long)dim,
              (long long)data_rank, (long long)K);
      return -1;
    }
    if (output_shape[d] == 0 && dim > 0) {
      static std::atomic<bool> warned{false};
      if (!warned.exchange(true)) {
        fprintf(stderr,
                "[REAL] wrap_hipsr_slice: zero-capacity output on sliced axis "
                "%d with a non-empty input dim (%lld) -- the compile-time "
                "extent collapsed to 0; a consumer that appends into this "
                "buffer (e.g. a hip.loop Concat accumulator) will fail\n",
                d, (long long)dim);
      }
    }
    logical_extent[d] = expected;
  }

  for (int d = 0; d < data_rank; ++d) {
    if (!axis_set[d]) {
      start_per_axis[d] = 0;
      step_per_axis[d] = 1;
    }
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_hipsr_slice: rank=%lld, K=%lld, data_type=%s -> hip_slice\n",
      (long long)data_rank, (long long)K,
      hipdnn_ep_datatype_name(data_type));

  return hip_slice(hipdnn_ep_state_get_stream(state), data, output, data_shape,
                   output_shape, logical_extent, start_per_axis, step_per_axis,
                   static_cast<int>(data_rank), hip_dtype);
}
