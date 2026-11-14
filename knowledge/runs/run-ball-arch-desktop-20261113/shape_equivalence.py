"""Full T64/720p VJP equivalence with a memory-bounded independent reference.

The uncheckpointed end-to-end model is known to exceed this GPU's capacity.
This diagnostic releases the CNN graph before the query reference and applies
its saved feature VJP to a fresh CNN graph. This is the ordinary chain rule,
not a training path; production never detaches a feature or gradient.
"""
import json
import time
from dataclasses import replace

from rebuild_dataset import write
from state import model_config


def check(root):
    import torch

    from src.tasks.ball_detection.model_io.mdd_temporal import build_temporal_model

    torch.set_num_threads(2)
    torch.manual_seed(1859)
    plan = json.loads((root / "original/plan.json").read_text())
    config = model_config(plan, "T1", "encoder_msda")
    original = root / "original/T1-q06000-q12000/checkpoint-00005500-cddcef20"
    latest = json.loads((original / "latest.json").read_text())
    saved = torch.load(original / latest["checkpoint"], map_location="cpu", weights_only=True)
    weights = saved["model_artifact"]["state_dict"]
    rgb = torch.randint(256, (1, 64, 3, 720, 1280), dtype=torch.uint8, device="cuda")
    times = torch.arange(64, device="cuda")[None].float() / 30
    probe = torch.linspace(-.7, 1.3, 128, device="cuda").reshape(1, 64, 2)
    reference = build_temporal_model(replace(config, activation_checkpointing=False)).cuda().train()
    reference.load_state_dict(weights, strict=True)
    stage_inputs, features_cpu = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        value = reference.mdd(rgb)
        for index, stage in enumerate(reference.encoder.stages):
            stage_inputs.append(value.cpu())
            value = stage(value)
            if index >= 2:
                features_cpu.append(value.transpose(1, 2).cpu())
    del reference, value
    torch.cuda.empty_cache()
    results, query_gradients, feature_gradients, outputs = [], [], [], []
    rng = torch.cuda.get_rng_state()
    for enabled in (False, True):
        model = build_temporal_model(replace(config, activation_checkpointing=enabled)).cuda().train()
        model.load_state_dict(weights, strict=True)
        features = tuple(value.cuda().requires_grad_() for value in features_cpu)
        # Replace computation only at the explicit query component input; invoke
        # the actual detector forward/prepare_memory/blocks/head in both paths.
        model.encoder.forward = lambda _value, features=features: features
        model.mdd.forward = lambda value: value
        torch.cuda.set_rng_state(rng)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(rgb, times)
            loss = (prediction * probe).sum()
        forward_rng = torch.cuda.get_rng_state()
        loss.backward()
        torch.cuda.synchronize()
        if not torch.equal(forward_rng, torch.cuda.get_rng_state()):
            raise ValueError("Checkpoint replay advanced the full-shape CUDA RNG")
        outputs.append(prediction.detach().cpu())
        query_gradients.append({n:p.grad.cpu().clone() for n,p in model.named_parameters() if not n.startswith("encoder.") and p.grad is not None})
        expected_names = {n for n,_ in model.named_parameters() if not n.startswith("encoder.")}
        if set(query_gradients[-1]) != expected_names:
            raise ValueError("A full-shape query parameter gradient is missing")
        if not all(bool(torch.isfinite(value).all()) for value in query_gradients[-1].values()):
            raise ValueError("A full-shape query gradient is nonfinite")
        feature_gradients.append(tuple(value.grad.cpu().clone() if value.grad is not None else None for value in features))
        results.append(dict(component="query", activation_checkpointing=enabled,
                            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                            seconds=time.perf_counter()-started))
        del model, features, prediction, loss
        torch.cuda.empty_cache()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-5, atol=2e-6)
    maximum_query_error = 0.
    for name, value in query_gradients[0].items():
        other = query_gradients[1][name]
        torch.testing.assert_close(value, other, rtol=1e-2, atol=5e-5, msg=name)
        maximum_query_error = max(maximum_query_error, float((value-other).abs().max()))
    for a,b in zip(feature_gradients[0], feature_gradients[1], strict=True):
        if a is None or b is None:
            if a is not b:
                raise ValueError("Pyramid feature gradient connectivity changed")
        else:
            torch.testing.assert_close(a,b,rtol=1e-2,atol=2e-6)
    # Apply the identical upstream feature VJP to actual full-size CNN executions.
    cnn_gradients = []
    for enabled in (False, True):
        model = build_temporal_model(replace(config, activation_checkpointing=enabled)).cuda().train()
        model.load_state_dict(weights, strict=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if enabled:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = model.encoder(model.mdd(rgb))
            values, gradients = [], []
            for value, upstream in zip(features, feature_gradients[0], strict=True):
                if upstream is not None:
                    values.append(value)
                    gradients.append(upstream.cuda())
            torch.autograd.backward(values, gradients)
            del features, values, gradients
        else:
            # A stage-local uncheckpointed reference avoids retaining the whole
            # CNN graph. Add every pyramid VJP at its exact stage output.
            incoming = None
            for index in reversed(range(len(model.encoder.stages))):
                value = stage_inputs[index].cuda().requires_grad_()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model.encoder.stages[index](value)
                gradient = torch.zeros_like(output) if incoming is None else incoming.cuda()
                if index >= 2 and feature_gradients[0][index-2] is not None:
                    gradient = gradient + feature_gradients[0][index-2].cuda().transpose(1, 2)
                output.backward(gradient)
                incoming = value.grad.cpu()
                del value, output, gradient
        torch.cuda.synchronize()
        cnn_gradients.append({n:p.grad.cpu().clone() for n,p in model.named_parameters() if n.startswith("encoder.") and p.grad is not None})
        if set(cnn_gradients[-1]) != {n for n,_ in model.named_parameters() if n.startswith("encoder.")}:
            raise ValueError("A full-shape CNN parameter gradient is missing")
        if not all(bool(torch.isfinite(value).all()) for value in cnn_gradients[-1].values()):
            raise ValueError("A full-shape CNN gradient is nonfinite")
        results.append(dict(component="encoder", activation_checkpointing=enabled,
                            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                            seconds=time.perf_counter()-started))
        del model
        torch.cuda.empty_cache()
    maximum_cnn_error = 0.
    for name, value in cnn_gradients[0].items():
        other = cnn_gradients[1][name]
        torch.testing.assert_close(value, other, rtol=1e-2, atol=5e-5, msg=name)
        maximum_cnn_error=max(maximum_cnn_error,float((value-other).abs().max()))
    report = dict(complete=True, shape=[1,64,3,720,1280], model="T1 saved 11500 query updates",
                  precision="BF16; official MSDA FP32", reference="component VJP chain rule; uncheckpointed stage-local CNN plus uncheckpointed full query",
                  actual_production_detaches=False, all_parameter_gradients_compared=True,
                  query_parameter_count=len(query_gradients[0]), cnn_parameter_count=len(cnn_gradients[0]),
                  maximum_query_absolute_gradient_error=maximum_query_error,
                  maximum_cnn_absolute_gradient_error=maximum_cnn_error, samples=results)
    write(root / "full-shape-equivalence.json",report)
    return report
