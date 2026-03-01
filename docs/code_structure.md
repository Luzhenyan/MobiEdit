# MobiEdit Code Structure

This document describes how the unified MobiEdit project is organized.

## Top-level layout

```text
MobiEdit/
  README.md
  docs/
    code_structure.md
  scripts/
    bootstrap.sh
  server/
    README.md
    DemoData/
    mobiedit/
      easyeditor/
      hparams/
      examples/
  edge/
    README.md
    src/
    include/
    scripts/
    tools/
  reference/
    FwdLLM/
```

## Module responsibilities

### server (algorithm side)

- Source: original `MobiEdit` repository.
- Runtime: Python-based training/editing pipelines.
- Main code path:
  - `server/mobiedit/easyeditor`: editing methods, editors, model wrappers.
  - `server/mobiedit/hparams`: method/model hyper-parameters.
  - `server/mobiedit/examples`: runnable scripts (for example ROME runs).
  - `server/DemoData`: sample benchmark data.

### edge (device side)

- Source: original `mllm` repository.
- Runtime: C/C++ inference engine for mobile and edge devices.
- Main code path:
  - `edge/src`: model runtime core, backend implementations.
  - `edge/include`: public headers.
  - `edge/scripts`: build and run scripts for Linux/Android/NPU.
  - `edge/tools`: conversion and utility tooling.

### reference/FwdLLM (reference style)

- Source: original `FwdLLM` repository.
- Purpose: reference architecture and documentation style.
- Not part of the primary runtime path of this integrated project.

## Recommended workflow

1. Develop and test editing methods under `server`.
2. Export/convert required model artifacts for on-device usage.
3. Integrate and benchmark runtime behavior under `edge`.
4. Keep cross-component assumptions documented at this top level.
