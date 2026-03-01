# MobiEdit Edge

`edge` is the on-device runtime module of MobiEdit, **built on top of the mllm engine** for efficient LLM inference and lightweight editing on mobile and edge devices.

## Main Features

- C/C++ on-device inference runtime targeting ARM, x86, and Qualcomm NPU.
- Supports both text and multimodal model deployment/inference.
- Includes quantization-friendly execution paths for better mobile memory/latency efficiency.
- Integrates zero-order (forward-only) editing for low-cost personalization updates.

## Zero-Order Pipeline

The current zero-order workflow focuses on the `Qwen2.5-1.5B` path:

- End-to-end perturbation training + inference loop in `examples/demo_qwen_rome_fwd.cpp`.
- `ZeroOrderOptimizer` manages random perturbations and registered tensors via `applyPerturbation/removePerturbation`.
- QNN decoding path is implemented in `src/models/qwen/modeling_qwen_npu_rome.hpp`.
- After updating an input tensor, call `input_tensor.setTtype(INPUT_TENSOR)` to refresh model input pointers.
- For each step, positive/negative perturbations are sampled by `group_size`, and updates are applied from averaged loss estimates.

## Key Files

- `examples/demo_qwen_rome_fwd.cpp`
- `src/ZeroOrderOptimize.hpp`
- `src/ZoroOrderOptimize.cpp`
- `src/models/qwen/modeling_qwen_npu_rome.hpp`

## Quick Build

```bash
cmake -S . -B build
cmake --build build -j
```

After building, you can run on-device experiments with `demo_qwen_rome_fwd` and related scripts.
