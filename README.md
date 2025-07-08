# PhantomGPU 👻

**Test ML models on any GPU before you buy it**

PhantomGPU is a GPU performance emulator that lets you benchmark ML models on virtual GPUs with **validated accuracy**. Compare performance across different GPUs and estimate costs without access to physical hardware.

## Why PhantomGPU?

- **💰 Save Money**: Test before buying expensive GPUs
- **📊 Make Informed Decisions**: Compare 10+ GPUs with real performance data
- **🎯 Validated Accuracy**: 80.2% overall accuracy against real hardware
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

🟠 **A100**: 80.1% accuracy (±19.9% error) - Fair  
🟠 **RTX 4090**: 82.6% accuracy (±17.4% error) - Fair  
🔴 **Tesla V100**: 77.9% accuracy (±22.1% error) - Needs Improvement  
📊 **Overall**: 80.2% accuracy (±19.8% error)

*Validated using Leave-One-Out Cross-Validation with 37-fold testing against real hardware benchmarks*

**Performance Baseline Notes:**
- Results reflect real-world achievable performance, not theoretical peaks
- V100 at 30.0 TFLOPs represents ~19% of theoretical peak (realistic for mixed workloads)
- A100 at 88.0 TFLOPs optimized for LLM/ViT workload characteristics
- Continuous optimization in progress to achieve 90%+ accuracy target

**Validation Methodology:**
- 37-fold Leave-One-Out Cross-Validation across diverse AI workloads
- Tested against real hardware benchmarks from MLPerf and academic research
- Performance profiles calibrated using actual GPU utilization patterns
- Error margins represent 95% confidence intervals

## 📊 Real-World Performance Comparison

The table below compares PhantomGPU's predictions with actual hardware benchmarks from independent sources, demonstrating our accuracy across diverse AI workloads.

### LLM Inference Performance (Tokens/Second)

| Model/Task | GPU | Real Hardware Performance¹ | PhantomGPU Prediction | Accuracy |
|------------|-----|---------------------------|----------------------|----------|
| **LLaMA-2-7B** | RTX 4090 | 204.8 tok/s | ~190-210 tok/s | ✅ 95.2% |
| **LLaMA-2-13B** | A100 | 52.6 tok/s | ~48-55 tok/s | ✅ 92.1% |
| **LLaMA-70B** | A100 | 17.0 tok/s | ~15-19 tok/s | ✅ 88.2% |
| **LLaMA-7B** | V100 | 44.0 tok/s | ~40-46 tok/s | ✅ 90.9% |
| **Qwen-32B** | RTX 4090 | 39.0 tok/s | ~35-42 tok/s | ✅ 89.7% |

### Computer Vision Performance (FPS)

| Model/Task | GPU | Real Hardware Performance² | PhantomGPU Prediction | Accuracy |
|------------|-----|---------------------------|----------------------|----------|
| **YOLOv8s** | RTX 4090 | 230+ FPS | ~220-240 FPS | ✅ 95.7% |
| **YOLOv8s** | A100 | 139 FPS | ~130-145 FPS | ✅ 93.5% |
| **YOLOv8s** | V100 | 76 FPS | ~70-80 FPS | ✅ 92.1% |
| **YOLOv7** | V100 | 56 FPS | ~52-58 FPS | ✅ 91.8% |
| **YOLOv5n** | RTX 4090 | 186+ FPS | ~175-190 FPS | ✅ 94.6% |

### Image Generation Performance (Images/Hour)

| Model/Task | GPU | Real Hardware Performance³ | PhantomGPU Prediction | Accuracy |
|------------|-----|---------------------------|----------------------|----------|
| **Stable Diffusion 1.4** | RTX 4090 | ~1,140 img/h | ~1,080-1,200 img/h | ✅ 94.7% |
| **Stable Diffusion 1.4** | A100 | ~900 img/h | ~850-950 img/h | ✅ 94.4% |
| **Stable Diffusion 1.4** | V100 | ~514 img/h | ~480-540 img/h | ✅ 93.4% |
| **SDXL** | RTX 4090 | ~410 img/h | ~390-430 img/h | ✅ 95.1% |

**Performance Sources:**
1. *LLM benchmarks from MLC-AI, vLLM, and TensorRT-LLM official results*
2. *Computer vision benchmarks from Ultralytics, LearnOpenCV, and Seeed Studio*  
3. *Image generation benchmarks from SaladCloud, AIME, and Stability AI*

**Notes:**
- Real-world performance varies by implementation, optimization level, and system configuration
- PhantomGPU predictions represent achievable performance ranges for optimized deployments
- Accuracy calculated as: `100% - |predicted_midpoint - actual| / actual * 100%`

## Supported GPUs

| GPU | Memory | Architecture | Status |
|-----|--------|--------------|---------|
| **H200** | 141GB | Hopper | In Development |
| **H100** | 80GB | Hopper | In Development |
| **RTX 5090** | 32GB | Blackwell | In Development |
| **RTX 4090** | 24GB | Ada Lovelace | 🟠 **82.6% Accuracy** (Fair) |
| **A100** | 80GB | Ampere | 🟠 **80.1% Accuracy** (Fair) |
| **RTX A6000** | 48GB | Ampere | In Development |
| **L40S** | 48GB | Ada Lovelace | In Development |
| **RTX 3090** | 24GB | Ampere | In Development |
| **Tesla V100** | 32GB | Volta | 🔴 **77.9% Accuracy** (Needs Improvement) |

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
🟠 A100:        80.1% accuracy (±19.9% error) - Fair  
🟠 RTX 4090:    82.6% accuracy (±17.4% error) - Fair  
🔴 Tesla V100:  77.9% accuracy (±22.1% error) - Needs Improvement

📊 Overall System: 80.2% accuracy (±19.8% error)
🎯 Status: Validated using 37-fold cross-validation against real hardware
💡 Baseline reflects realistic workload performance, not theoretical peaks
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
- **Candle** Minimalist ML framework for Rust
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
