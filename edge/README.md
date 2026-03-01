# MobiEdit Edge

`edge` 是 MobiEdit 的端侧运行时模块，**基于 mllm 引擎**构建，用于在移动/边缘设备上执行高效 LLM 推理与轻量编辑流程。

## 主要功能

- 基于 C/C++ 的端侧推理引擎，面向 ARM/x86/Qualcomm NPU。
- 支持文本与多模态模型的端侧部署与推理。
- 支持量化与移动端内存/性能友好的执行路径。
- 集成零阶（Forward-only）编辑能力，用于低开销个性化更新。

## Zero-Order（参考 `zo.md`）

当前零阶优化流程以 `Qwen2.5-1.5B` 路径为主，核心能力包括：

- 在 `examples/demo_qwen_rome_fwd.cpp` 中完成扰动训练与推理闭环。
- `ZeroOrderOptimizer` 管理随机扰动与注册张量，并通过 `applyPerturbation/removePerturbation` 执行加减扰动。
- QNN 解码路径位于 `src/models/qwen/modeling_qwen_npu_rome.hpp`。
- 对 `input tensor` 更新后，需要重新设置 `input_tensor.setTtype(INPUT_TENSOR)` 以确保模型指针同步。
- 训练时按 `group_size` 执行正负扰动，基于 loss 均值估计梯度并进行学习率更新。

## 关键文件

- `examples/demo_qwen_rome_fwd.cpp`
- `src/ZeroOrderOptimize.hpp`
- `src/ZoroOrderOptimize.cpp`
- `src/models/qwen/modeling_qwen_npu_rome.hpp`
- `zo.md`

## 快速构建

```bash
cmake -S . -B build
cmake --build build -j
```

构建完成后，可基于 `demo_qwen_rome_fwd` 和相关脚本进行端侧实验。
