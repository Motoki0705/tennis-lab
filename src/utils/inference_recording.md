# Inference execution recording

`inference_recording.py` records execution provenance separately from configuration,
component outputs and cache keys. Recording is disabled until an
`InferenceRecorder` context is entered. Importing it does not import PyTorch or
initialize CUDA. Enabling it on the same source revision leaves artifact bytes,
artifact IDs and execution keys unchanged. Installing instrumentation itself is
a source change and follows the pipeline's ordinary code fingerprint rules.

```python
from pathlib import Path
from src.utils.inference_recording import InferenceRecorder, model_event, recording_context, span

with InferenceRecorder(Path("outputs/measurement.json"), devices=("cuda:0",)):
    with recording_context(clip_id="video_000/clip_000", camera_id="cam0", node="detector/cam0"):
        with model_event("load", "detector", weights={"checkpoint": checkpoint},
                         device="cuda:0", dtype="float32", instance="detector-pass-1"):
            model.load()
        try:
            with span("inference", model="detector", batch_shape=list(images.shape)):
                result = model(images)
        finally:
            with model_event("unload", "detector", weights={"checkpoint": checkpoint},
                             device="cuda:0", dtype="float32", instance="detector-pass-1"):
                model.unload()
```

The inference runtime should emit model events at actual load/unload boundaries,
using the same instance ID for both. Borrowing a resident model emits no load.
`BaseInferenceModel.load`/`unload` and the instrumented component lifecycle methods
already emit these events; the runtime should not wrap them in another model event.
Use `recording_context` to attach pass, clip, camera or node attributes, and
`span` for bounded work. `measured` decorates ordinary functions; generators need
spans around their work, ending **before** each yield. The video prefetch iterator
copies the execution context to its producer thread. Contexts are local to an
execution; recorders cannot be nested or reused. Exceptions retain their type and
produce error spans and a failed JSON record.

`batch_shape` is the actual model input, including the final partial batch.
`batch_items` defaults to its leading dimension. ViTPose records original crop
count separately from the doubled flip-test model batch. GVHMR records the
number of sequences and their shared time dimension. Empty detections produce no inference
batch; node outcomes distinguish execution, cache hits and load-only use.

For video batching, `batch_dispatches` summarizes adapter dispatch counts,
logical batch sizes, fill fraction, cross-source and cross-clip dispatches,
and peak queue/source counts. `batch_shapes` reports the forward shapes
recorded by the models. `model_loads_by_pass` and `video_reads_by_pass` count
loads and decoder opens by explicit video pass. Video metadata probes are
outside the decoder-open count.

The versioned `inference_recording_v1` JSON contains raw spans, parent and thread
IDs, relative wall-clock start/end times, inclusive thread and process CPU time
in nanoseconds, status, model assets with file SHA-256, device/dtype, batch shape
counts, resources and summaries. CPU clocks are cumulative counters at each
boundary. Process CPU time includes concurrent work on all process threads.
Weight hashes are
computed once per recorder; subsequent events verify file metadata, including
ctime. A changed asset fails explicitly. Hashing added by recording is measured
as `measurement`, outside model load time. No record is embedded in artifacts.

With `devices` configured, CUDA timings default to synchronized wall times. Transfer,
inference and lifecycle spans synchronize the configured devices at both
boundaries. CUDA preprocessing declares its device. The recorder writes each
added synchronization as a `cuda_sync` child span. Use `synchronize_cuda=False`
to time the host without the added barriers, while retaining CUDA allocator
and NVML measurements. `transfer` includes implicit waits in host reads of CUDA
tensors. `decode_queue_wait` labels prefetch queue reads and writes separately.
Eager
CLIP unload events describe the existing reference-drop and memory-cleanup
boundary; other local owners may retain tensors until their function returns.
AFLink and body root-regressor assets have no explicit unload in the
existing component; their loads are recorded without inventing an unload.

`stage_inclusive_seconds` sums spans and can exceed wall time due to nesting and
prefetch overlap. `stage_wall_seconds` partitions the measured interval without
double counting: deepest active span, then GPU work, then latest start wins.
Coarse clip/node/video-pass spans do not win attribution. `unattributed` covers uninstrumented
work and boundary overhead. Component CPU work is enclosed in a postprocess span;
explicit decode, preprocess, transfer and inference spans take precedence.
The partition sums to `total_wall_seconds`. It is an attribution convention,
not a utilization estimate. JSON serialization and final publication of the
recorder itself occur after its measured interval.

Each `resource_samples` row uses the span clock and includes its sampling cost.
Sampling runs every 0.5 seconds by default, with initial and final readings.
Shorter intervals increase CPU and driver query cost and can repeat the same
NVML reading. Process and per-thread CPU percentages use one logical core as
100%, while each system logical core has its own percentage. The first reading
for a thread establishes its baseline. The recorder labels the main, decode
prefetch and sampler threads by native ID, and labels other threads as
`native`. RSS and host memory are in bytes. Linux process-lifetime peak RSS can
include earlier repeats. GPU utilization and used memory come from NVML for the
device UUID, including activity and allocations from other processes. PyTorch
allocated/reserved peaks are separate allocator counts. Sampling may miss
short peaks.

`timings` reports interval unions by stage, node, clip/node and model. Within a
row it unions wall and process CPU intervals and sums the CPU interval union of
each thread. Work shared by a cross-source batch belongs to every participating
node. Rows can therefore overlap. GPU coverage and mean utilization are weighted by
wall overlap between adjacent samples. GPU active fraction counts the covered
intervals whose final sample has nonzero GPU utilization. Missing telemetry is
excluded from coverage. NVML measures utilization over a preceding device
window, so these estimates have sample-level resolution.

`cpu_during_gpu_idle` ranks operations by estimated exclusive thread CPU time
during covered intervals where all selected GPUs sampled zero utilization.
It subtracts same-thread nested spans and apportions the remaining thread CPU
time in proportion to idle overlap. It also reports the corresponding wall time
summed across threads. Child work in another thread is attributed to that thread.
GPU launches, synchronization and queue waits have separate stages.

`temporary_directory(path)` samples only caller-owned temporary outputs, including
their completed size before atomic publication. It never scans the shared
dataset or unrelated output directories. Resource sampling failures are reported
in the JSON and raise after saving it. These measurements are not GPU admission
control; run GPU work through the shared training queue.
