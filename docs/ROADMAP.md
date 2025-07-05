# PhantomGPU Roadmap

## Current Status ✅

**Core Features Complete:**
- Multi-framework support (TensorFlow, PyTorch, ONNX, HuggingFace)
- Real hardware modeling with TOML configuration
- Multi-GPU performance comparison
- Cost analysis and GPU recommendations
- CLI interface with comprehensive features

**Framework Support:**
- ✅ TensorFlow (SavedModel, frozen graphs, TensorFlow Lite, Keras)
- ✅ ONNX models
- ✅ HuggingFace Hub integration
- ✅ PyTorch (basic support)

**Hardware Modeling:**
- ✅ 10+ production GPUs (V100, A100, RTX 4090, etc.)
- ✅ Custom GPU definitions via TOML
- ✅ Thermal and memory modeling
- ✅ Precision support (FP32, FP16, INT8)

## Phase 1

### Dynamic GPU Support 🔧
**Goal**: Remove CLI limitations
- Load any GPU name from TOML profiles
- Runtime GPU discovery
- Better error messages for missing profiles

### AI-Powered Optimization 🧠
- Automated model optimization suggestions
- Batch size and precision recommendations
- Hardware-model matching algorithms

### Energy & Sustainability 🌱
- Power consumption modeling
- Carbon footprint analysis
- Green deployment recommendations

## Not Planned

We explicitly avoid these to maintain focus:
- Game/graphics workloads (ML-focused only)
- Real GPU monitoring (prediction-focused)
- Multiple configuration formats (TOML is sufficient)
- Distributed system complexity (keep core simple

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

Priority areas for contributions:
1. Web interface development (TypeScript/React)
2. Cloud provider API integrations
3. Additional ML framework support
4. Documentation and examples

---

*Updated: January 2025*
