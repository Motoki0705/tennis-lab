"""Batch ranges and dynamic shape contracts for the inference benchmarks."""


def batch_slices(count, batch_size):
    """Preserve input order and keep the last batch at its actual size."""
    if not 1 <= batch_size <= count:
        raise ValueError(f"Batch size {batch_size} is invalid for {count} items")
    return tuple(
        slice(start, min(start + batch_size, count))
        for start in range(0, count, batch_size)
    )


def batch_profile_points(optimum, maximum):
    if not 1 <= optimum <= maximum or maximum == 1:
        raise ValueError(f"Invalid dynamic batch profile: 1/{optimum}/{maximum}")
    return 1, optimum, maximum


def require_dynamic_axis(value, axis, symbol):
    dims = value.type.tensor_type.shape.dim
    if len(dims) <= axis or dims[axis].dim_param != symbol:
        raise ValueError(f"Export lost the {symbol} axis for {value.name}")


def require_engine_profile(engine, expected):
    if engine.num_optimization_profiles != 1:
        raise ValueError("Expected exactly one TensorRT optimization profile")
    actual = {
        name: [list(shape) for shape in engine.get_tensor_profile_shape(name, 0)]
        for name in expected
    }
    if actual != {
        name: [list(shape) for shape in shapes] for name, shapes in expected.items()
    }:
        raise ValueError(f"TensorRT engine profile {actual} differs from {expected}")
    return actual
