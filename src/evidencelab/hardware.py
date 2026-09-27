"""Resolve precision once per session; never switch it silently during a run."""
import platform
import socket


def choose_dtype(requested, native_bf16):
    if requested not in {"auto", "bfloat16", "float16"}:
        raise ValueError("dtype must be auto, bfloat16 or float16")
    if requested == "bfloat16" and not native_bf16:
        raise RuntimeError("Explicit bfloat16 requested but this GPU lacks native BF16; use a new float16 run")
    return ("bfloat16" if native_bf16 else "float16") if requested == "auto" else requested


def execution_profile(config):
    base = {"node": socket.gethostname(), "python_full": platform.python_version()}
    if config.backend == "mock":
        return {**base, "resolved_dtype": "mock", "gpu": None}
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Require exactly one allocated CUDA GPU; do not override CUDA_VISIBLE_DEVICES")
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = choose_dtype(config.dtype, native_bf16)
    # Test driver/wheel/kernel compatibility before downloading model weights.
    tensor = torch.ones((8, 8), device="cuda", dtype=getattr(torch, dtype))
    if not torch.isfinite(tensor @ tensor).all().item():
        raise RuntimeError("CUDA precision smoke check returned nonfinite values")
    torch.cuda.synchronize()
    return {**base, "resolved_dtype": dtype, "native_bf16": native_bf16,
            "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "torch_cuda": torch.version.cuda}
