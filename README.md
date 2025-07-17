# MobiEdit

A mobile-friendly model editing framework for efficient knowledge modification in language models.

## � Quick Start

### Requirements

- Python 3.9+
- Conda package manager

### Installation

1. **Create Environment**
   ```bash
   conda create -n MobiEdit python=3.9.7
   conda activate MobiEdit
   ```

2. **Install Dependencies**
   ```bash
   cd mobiedit
   pip install -r requirements.txt
   ```

### Configuration

Before running the model, you need to configure the model path:

1. Navigate to the configuration file:
   ```
   mobiedit/hparams/ROME_new/qwen2.5-rotate-3b-0-zo-quan.yaml
   ```

2. Update the `model_name` parameter with your local model path

3. Configure key parameters:
   - `quantize`: Set to `true` to enable model quantization
   - `use_zo`: Set to `true` to enable zeroth-order (forward-only) training

### Usage

Run the ROME editing method:

```bash
cd mobiedit/examples
sh run_ROME.sh
```

## 🔧 Technical Details

### Architecture Overview

MobiEdit implements a mobile-friendly model editing framework with the following key components:

#### Model Loading and Quantization
- **Location**: `mobiedit/easyeditor/editors/editor.py` (lines 206-259)
- **Function**: Loads Qwen2.5 model and performs weight quantization
- **Purpose**: Reduces model size for mobile deployment while maintaining performance

#### ROME Method Implementation
- **Location**: `mobiedit/easyeditor/models/rome/` directory
- **Core Algorithm**: Implements Rank-One Model Editing (ROME) for efficient knowledge modification

#### Edit Vector Training
- **Location**: `mobiedit/easyeditor/models/rome/compute_v.py`
- **Function**: Handles training of edit vectors (delta)
- **Forward Training**: When `use_zo=true`, calls `compute_v_zo` function for Forward Training

#### Activation Quantization and Calibration
- **Location**: `compute_v.py` (lines 593-619)
- **Process**: 
  1. Calls functions in `../quantization/quantizer.py`
  2. Quantizes activation values
  3. Performs calibration for optimal quantization

#### Forward Training
- **Location**: `mobiedit/easyeditor/models/rome/zo.py`
- **Function**: `forward_grad_step` function
- **Purpose**: Implements forward-only training for memory-efficient optimization






