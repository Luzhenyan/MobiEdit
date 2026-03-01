# MobiEdit
MobiEdit is the first mobile knowledge editing framework that enables efficient LLM personalization on commercial off-the-shelf (COTS) mobile devices.
MobiEdit is organized as a two-part project:

- `server`: algorithm implementation and model editing pipeline running on server-class machines.
- `edge`: on-device multimodal LLM inference runtime for mobile/edge deployment.



## Installation

Run commands from the `MobiEdit/` directory.

### 1) Server environment (Python)

```bash
cd server/mobiedit
conda create -n mobiedit-server python=3.9.7 -y
conda activate mobiedit-server
pip install -r requirements.txt
```

### 2) Edge environment (C/C++)

```bash
cd edge
cmake -S . -B build
cmake --build build -j
```

For Android/NPU workflows, use scripts under `edge/scripts`.

## Code Structure of MobiEdit

- `server`: server-side model editing codebase.
- `edge`: edge/mobile inference runtime.
- `docs`: project-level documentation.
- `scripts`: project-level helper scripts.

Detailed mapping: [docs/code_structure.md](./docs/code_structure.md)

## Quick Start

### Server-side editing demo

```bash
cd server/mobiedit/examples
sh run_ROME.sh
```

### Edge-side runtime build demo

```bash
cd edge
cmake -S . -B build
cmake --build build -j
```

### Helper script

```bash
./scripts/bootstrap.sh all
```
