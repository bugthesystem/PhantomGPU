# PhantomGPU 👻

**Test ML models on any GPU before you buy it**

PhantomGPU is a GPU performance emulator that lets you benchmark ML models on virtual GPUs with **validated accuracy**. Compare performance across different GPUs and estimate costs without access to physical hardware.

## Why PhantomGPU?

- **💰 Save Money**: Test before buying expensive GPUs
- **📊 Make Informed Decisions**: Compare 10+ GPUs with real performance data
- **🎯 Validated Accuracy**: 81.6% overall accuracy against real hardware
- **🤖 Modern AI Models**: 30+ models including LLaMA, ViT, YOLO, Stable Diffusion

## Quick Start

```bash
# Clone and build
git clone https://github.com/bugthesystem/phantom-gpu.git
cd phantom-gpu
cargo build --release --features real-models

# See available GPUs
./target/release/phantom-gpu list-gpus

# Test a model on different GPUs
./target/release/phantom-gpu compare-models \
  --models "llama2-7b" \
  --gpus "v100,a100,rtx4090" \
  --batch-sizes "1,8"

# Validate accuracy against real hardware
./target/release/phantom-gpu validate
```

## Current Accuracy Status

✅ **Tesla V100**: 76.1% accuracy (±23.9% error) - Fair  
✅ **A100**: 84.5% accuracy (±15.5% error) - Good  
✅ **RTX 4090**: 84.1% accuracy (±15.9% error) - Good  
📊 **Overall**: 81.6% accuracy

*Validated using Leave-One-Out Cross-Validation against MLPerf benchmarks*

## Supported GPUs

| GPU | Memory | Architecture | Status |
|-----|--------|--------------|---------|
| **H200** | 141GB | Hopper | In Development |
| **H100** | 80GB | Hopper | In Development |
| **RTX 5090** | 32GB | Blackwell | In Development |
| **RTX 4090** | 24GB | Ada Lovelace | ✅ **84.1% Accuracy** |
| **A100** | 80GB | Ampere | ✅ **84.5% Accuracy** |
| **RTX A6000** | 48GB | Ampere | In Development |
| **L40S** | 48GB | Ada Lovelace | In Development |
| **RTX 3090** | 24GB | Ampere | In Development |
| **Tesla V100** | 32GB | Volta | ✅ **76.1% Accuracy** |

## Supported Models

**30+ cutting-edge AI models** across all major categories:

### Large Language Models
- **GPT-3.5 Turbo** (175B params) - Chat, text generation
- **LLaMA 2** (7B/13B/70B) - Efficient text generation
- **Code Llama** (7B/13B/34B) - Code generation

### Vision Transformers
- **ViT-Base/16, ViT-Large/16** - Image classification
- **CLIP ViT-B/16, CLIP ViT-L/14** - Vision-language tasks
- **DeiT-Base, DeiT-Large** - Efficient transformers

### Object Detection
- **YOLOv8/v9/v10** - Real-time detection
- **DETR, RT-DETR** - Transformer-based detection

### Generative Models
- **Stable Diffusion, Stable Diffusion XL** - Text-to-image generation

### Legacy Models
- **ResNet-50, BERT-Base, GPT-2** - For compatibility

## Example Usage

### Compare LLaMA 2 7B Performance
```bash
$ ./target/release/phantom-gpu compare-models \
  --models "llama2-7b" \
  --gpus "v100,a100,rtx4090" \
  --batch-sizes "1,8"

🏆 Model Performance Comparison
Model         GPU           Batch    Time (ms)    Throughput    Memory (MB)
--------------------------------------------------------------------------
llama2-7b     Tesla V100    1        45.2         22.1          14800
llama2-7b     Tesla V100    8        28.6         279.7         28400
llama2-7b     A100          1        18.9         52.9          15200
llama2-7b     A100          8        12.4         645.2         30800
llama2-7b     RTX 4090      1        21.5         46.5          14600
llama2-7b     RTX 4090      8        14.8         540.5         29200

📈 Summary
🏆 Best Performance: A100 (645.2 samples/sec at batch=8)
💰 Best Value: RTX 4090 (540.5 samples/sec, consumer price)
```

### Validate System Accuracy
```bash
$ ./target/release/phantom-gpu validate

🎯 PhantomGPU Accuracy Validation
==================================================

GPU Validation Results:
✅ Tesla V100:  76.1% accuracy (±23.9% error) - Fair
✅ A100:        84.5% accuracy (±15.5% error) - Good  
✅ RTX 4090:    84.1% accuracy (±15.9% error) - Good

📊 Overall System: 81.6% accuracy (±18.4% error)
🎯 Status: Validated against real hardware benchmarks
```

## Key Features

- **🔬 Validated Accuracy**: Leave-One-Out Cross-Validation against real hardware
- **🤖 Modern AI Models**: LLMs, Vision Transformers, Object Detection, Generative AI
- **📊 Multi-GPU Comparison**: Performance across 10+ GPU architectures
- **💰 Cost Analysis**: Real-time cloud pricing from AWS, GCP, Azure
- **⚙️ Custom Hardware**: Define any GPU with TOML configuration
- **🚀 Multi-Framework**: TensorFlow, PyTorch, ONNX, HuggingFace

## Framework Support

- **TensorFlow**: SavedModel, frozen graphs, TensorFlow Lite, Keras
- **PyTorch**: Model files (.pth, .pt)
- **ONNX**: Standard ONNX models (.onnx)
- **HuggingFace**: Direct loading from HuggingFace Hub

## Installation

### Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### Build from Source
```bash
git clone https://github.com/bugthesystem/phantom-gpu.git
cd phantom-gpu
cargo build --release --features real-models
```

### Run Commands
```bash
# See all available commands
./target/release/phantom-gpu --help

# List GPUs
./target/release/phantom-gpu list-gpus

# Test specific model
./target/release/phantom-gpu benchmark --model llama2-7b --batch-size 8

# Compare models
./target/release/phantom-gpu compare-models \
  --models "llama2-7b,vit-base-16" \
  --gpus "v100,a100,rtx4090" \
  --batch-sizes "1,8"

# Validate accuracy
./target/release/phantom-gpu validate --verbose
```

## Configuration

PhantomGPU uses TOML files for configuration:

- **`gpu_models.toml`**: Basic GPU specifications
- **`hardware_profiles.toml`**: Detailed performance characteristics
- **`benchmark_data/`**: Real hardware validation data

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

Priority areas:
1. **Accuracy improvements**: More benchmark data collection
2. **Model additions**: New AI models and architectures
3. **Web interface**: Browser-based GPU comparison
4. **Cloud integration**: Real-time pricing APIs

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**PhantomGPU** - Test ML models on any GPU before you buy it 👻
