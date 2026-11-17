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
count separately from the doubled flip-test model batch. GVHMR records one
sequence with its actual time dimension. Empty detections produce no inference
batch; node outcomes distinguish execution, cache hits and load-only use.

The versioned `inference_recording_v1` JSON contains raw spans, parent and thread
IDs, relative nanosecond start/end times, status, model assets with file SHA-256,
device/dtype, batch shape counts, resources and summaries. Weight hashes are
computed once per recorder; subsequent events verify file metadata, including
ctime. A changed asset fails explicitly. Hashing added by recording is measured
as `measurement`, outside model load time. No record is embedded in artifacts.

CUDA timings are synchronized wall times when `devices` is supplied. Transfer,
inference and lifecycle spans synchronize the configured devices at both
boundaries; CUDA preprocessing also declares its device. Synchronization changes
execution overhead, so comparisons must use the same recorder settings. It does
not change precision, model lifetime, batch limits or inference order. Eager
CLIP unload events describe the existing reference-drop and memory-cleanup
boundary; other local owners may retain tensors until their function returns.
AFLink and body root-regressor assets have no explicit unload in the
existing component; their loads are recorded without inventing an unload.

`stage_inclusive_seconds` sums spans and can exceed wall time due to nesting and
prefetch overlap. `stage_wall_seconds` partitions the measured interval without
double counting: deepest active span, then GPU work, then latest start wins.
Coarse clip/node spans do not win attribution. `unattributed` covers uninstrumented
work and boundary overhead. Component CPU work is enclosed in a postprocess span;
explicit decode, preprocess, transfer and inference spans take precedence.
The partition sums to `total_wall_seconds`. It is an attribution convention,
not a utilization estimate. JSON serialization and final publication of the
recorder itself occur after its measured interval.

RAM includes sampled process RSS and Linux process-lifetime peak RSS (which can
include earlier repeats). CUDA allocated/reserved peaks are PyTorch allocator
high-water marks reset at recorder entry, excluding other processes, driver
memory and allocations outside PyTorch. RAM sampling may miss short peaks.
`temporary_directory(path)` samples only caller-owned temporary outputs, including
their completed size before atomic publication. It never scans the shared
dataset or unrelated output directories. Resource sampling failures are reported
in the JSON and raise after saving it. These measurements are not GPU admission
control; run GPU work through the shared training queue.
