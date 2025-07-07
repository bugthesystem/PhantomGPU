//! Command-line interface definitions and argument parsing

use clap::{ Parser, Subcommand, ValueEnum };
use crate::gpu_config::GpuModel;

#[derive(Parser)]
#[command(name = "phantom-gpu")]
#[command(about = "🚀 Phantom GPU - Advanced GPU Emulator for ML Workloads")]
#[command(version = "1.0.0")]
#[command(author = "Phantom GPU Team")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,

    /// GPU model to emulate
    #[arg(short, long, value_enum, default_value = "v100")]
    pub gpu: GpuType,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Train neural networks using Candle framework
    Train {
        /// Model architecture to train
        #[arg(short, long, value_enum, default_value = "cnn")]
        model: ModelType,

        /// Batch size for training
        #[arg(short, long, default_value = "32")]
        batch_size: usize,

        /// Number of epochs
        #[arg(short, long, default_value = "3")]
        epochs: usize,

        /// Number of batches per epoch
        #[arg(long, default_value = "20")]
        batches: usize,
    },

    /// Load and benchmark pre-trained models
    Benchmark {
        /// Model to benchmark
        #[arg(short, long, value_enum, default_value = "resnet50")]
        model: PretrainedModel,

        /// Batch size for inference
        #[arg(short, long, default_value = "16")]
        batch_size: usize,

        /// Number of inference runs
        #[arg(short, long, default_value = "100")]
        runs: usize,
    },

    /// Compare performance across multiple GPU models
    Compare {
        /// Models to compare
        #[arg(short, long, value_enum)]
        gpus: Vec<GpuType>,

        /// Model to test on all GPUs
        #[arg(short, long, value_enum, default_value = "resnet50")]
        model: PretrainedModel,

        /// Batch size for comparison
        #[arg(short, long, default_value = "32")]
        batch_size: usize,
    },

    /// Estimate cloud costs for training workloads
    Cost {
        /// Model to estimate costs for
        #[arg(short, long, value_enum, default_value = "resnet50")]
        model: PretrainedModel,

        /// Training duration in hours
        #[arg(long, default_value = "24")]
        hours: f64,

        /// Cloud provider
        #[arg(short, long, value_enum, default_value = "aws")]
        provider: CloudProvider,
    },

    /// Run distributed training simulation
    Distributed {
        /// Number of GPUs to simulate
        #[arg(short, long, default_value = "4")]
        num_gpus: usize,

        /// Model for distributed training
        #[arg(short, long, value_enum, default_value = "resnet50")]
        model: PretrainedModel,

        /// Epochs for distributed training
        #[arg(short, long, default_value = "5")]
        epochs: usize,
    },

    /// Run all benchmark suites
    Suite {
        /// Include experimental features
        #[arg(long)]
        experimental: bool,
    },

    /// List available GPU models (basic specs)
    ListGpus,

    #[cfg(feature = "real-models")]
    /// List detailed hardware profiles for realistic performance modeling
    ListHardware {
        /// Show detailed specifications for each profile
        #[arg(long)]
        verbose: bool,
    },

    #[cfg(feature = "real-models")]
    /// Validate PhantomGPU accuracy against real hardware benchmarks
    Validate {
        /// GPU to validate (e.g., "v100", "a100", "rtx4090")
        #[arg(long)]
        gpu: Option<String>,

        /// Path to custom benchmark data file
        #[arg(long)]
        benchmark_data: Option<String>,

        /// Show detailed validation report
        #[arg(long)]
        verbose: bool,
    },

    #[cfg(feature = "real-models")]
    /// Calibrate performance models using real benchmark data
    Calibrate {
        /// GPU to calibrate
        #[arg(long)]
        gpu: String,

        /// Path to benchmark data file
        #[arg(long)]
        benchmark_data: String,

        /// Output path for calibrated model
        #[arg(long)]
        output: Option<String>,
    },

    #[cfg(feature = "real-models")]
    /// Run stress tests for edge cases and extreme scenarios
    StressTest {
        /// Enable verbose output
        #[arg(long)]
        verbose: bool,

        /// Load custom edge case data
        #[arg(long)]
        edge_cases: Option<String>,
    },

    #[cfg(feature = "real-models")]
    /// Load and benchmark a real model from file or Hub
    LoadModel {
        /// Model source (file path or Hugging Face model ID)
        #[arg(short, long)]
        model: String,

        /// Model format
        #[arg(short, long, value_enum, default_value = "auto")]
        format: ModelFormat,

        /// Batch size for benchmarking
        #[arg(short, long, default_value = "16")]
        batch_size: usize,

        /// Number of inference runs
        #[arg(short, long, default_value = "100")]
        runs: usize,
    },

    #[cfg(feature = "pytorch")]
    /// Compare PyTorch vs Candle performance
    FrameworkCompare {
        /// Batch size for comparison
        #[arg(short, long, default_value = "32")]
        batch_size: usize,
    },

    #[cfg(feature = "real-models")]
    /// Compare multiple models across different GPUs
    CompareModels {
        /// Model sources (file paths or Hugging Face model IDs)
        #[arg(short, long, value_delimiter = ',')]
        models: Vec<String>,

        /// GPU types to compare
        #[arg(
            short,
            long,
            value_enum,
            value_delimiter = ',',
            default_values = &["rtx4090", "a100", "h100"]
        )]
        gpus: Vec<GpuType>,

        /// Batch sizes to test
        #[arg(short, long, value_delimiter = ',', default_values = &["1", "8", "32"])]
        batch_sizes: Vec<usize>,

        /// Output format
        #[arg(short, long, value_enum, default_value = "table")]
        output: OutputFormat,

        /// Include cost analysis
        #[arg(long)]
        include_cost: bool,

        /// Fast mode - instant results for demos (skips realistic timing)
        #[arg(long)]
        fast_mode: bool,

        /// Show progress indicators
        #[arg(long, default_value = "true")]
        show_progress: bool,

        /// Precision for inference (FP32, FP16, INT8)
        #[arg(long, value_enum, default_value = "fp32")]
        precision: Precision,

        /// Use real hardware performance modeling (slower but more accurate)
        #[arg(long)]
        real_hardware: bool,

        /// Path to custom hardware profiles TOML file
        #[arg(long)]
        hardware_profiles: Option<String>,
    },

    #[cfg(feature = "real-models")]
    /// Recommend optimal GPU for a specific model and use case
    RecommendGpu {
        /// Model source (file path or Hugging Face model ID)
        #[arg(short, long)]
        model: String,

        /// Maximum budget per hour (USD)
        #[arg(short, long)]
        budget: Option<f64>,

        /// Target batch size
        #[arg(long, default_value = "16")]
        batch_size: usize,

        /// Target throughput (samples/second)
        #[arg(long)]
        target_throughput: Option<f64>,

        /// Workload type
        #[arg(short, long, value_enum, default_value = "inference")]
        workload: WorkloadType,

        /// Cloud providers to consider
        #[arg(long, value_enum, value_delimiter = ',', default_values = &["aws", "gcp", "azure"])]
        cloud_providers: Vec<CloudProvider>,

        /// Fast mode - instant results for demos
        #[arg(long)]
        fast_mode: bool,
    },
}

#[derive(ValueEnum, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum GpuType {
    V100,
    A100,
    Rtx4090,
    Rtx3090,
    H100,
    H200,
    A6000,
    L40s,
    Rtx5090,
    RtxPro6000,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum ModelType {
    Cnn,
    ResNet,
    Transformer,
    Gpt,
    Bert,
    #[cfg(feature = "pytorch")]
    /// PyTorch CNN for CIFAR-10
    PytorchCnn,
    #[cfg(feature = "pytorch")]
    /// PyTorch ResNet for ImageNet
    PytorchResnet,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum PretrainedModel {
    // ===== LEGACY MODELS =====
    ResNet50,
    BertBase,
    Gpt2,
    Llama7b,
    StableDiffusion,
    DistilbertReal,

    // ===== LARGE LANGUAGE MODELS =====
    #[value(name = "gpt35-turbo")]
    Gpt35Turbo,
    #[value(name = "llama2-7b")]
    Llama2_7b,
    #[value(name = "llama2-13b")]
    Llama2_13b,
    #[value(name = "llama2-70b")]
    Llama2_70b,
    #[value(name = "codellama-7b")]
    CodeLlama7b,
    #[value(name = "codellama-13b")]
    CodeLlama13b,
    #[value(name = "codellama-34b")]
    CodeLlama34b,

    // ===== POPULAR MODERN LLMS =====
    #[value(name = "qwen25-7b")]
    Qwen25_7b,
    #[value(name = "qwen25-14b")]
    Qwen25_14b,
    #[value(name = "mistral-7b")]
    Mistral7b,
    #[value(name = "mistral-22b")]
    Mistral22b,
    #[value(name = "deepseek-v3")]
    DeepSeekV3,
    #[value(name = "phi35-mini")]
    Phi35Mini,
    #[value(name = "phi35-medium")]
    Phi35Medium,
    #[value(name = "gemma2-9b")]
    Gemma2_9b,
    #[value(name = "gemma2-27b")]
    Gemma2_27b,

    // ===== VISION TRANSFORMERS =====
    #[value(name = "vit-base-16")]
    VitBase16,
    #[value(name = "vit-large-16")]
    VitLarge16,
    #[value(name = "vit-huge-14")]
    VitHuge14,
    #[value(name = "deit-base")]
    DeitBase,
    #[value(name = "deit-large")]
    DeitLarge,
    #[value(name = "clip-vit-b-32")]
    ClipVitB32,
    #[value(name = "clip-vit-b-16")]
    ClipVitB16,
    #[value(name = "clip-vit-l-14")]
    ClipVitL14,

    // ===== MODERN OBJECT DETECTION =====
    #[value(name = "yolov9")]
    YoloV9,
    #[value(name = "yolov10")]
    YoloV10,
    #[value(name = "detr")]
    Detr,
    #[value(name = "rt-detr")]
    RtDetr,
    #[value(name = "yolov8n")]
    YoloV8n,
    #[value(name = "yolov8s")]
    YoloV8s,
    #[value(name = "yolov8m")]
    YoloV8m,
    #[value(name = "yolov8l")]
    YoloV8l,
    #[value(name = "yolov8x")]
    YoloV8x,

    // ===== GENERATIVE MODELS =====
    #[value(name = "stable-diffusion-xl")]
    StableDiffusionXl,
    #[value(name = "yolo-v8")]
    YoloV8,

    // Real model support
    #[cfg(feature = "real-models")]
    OnnxResnet50,
    #[cfg(feature = "real-models")]
    OnnxBert,
    #[cfg(feature = "real-models")]
    OnnxGpt2,
    #[cfg(feature = "real-models")]
    OnnxMobilenet,
    #[cfg(feature = "real-models")]
    HfDistilbert,
    #[cfg(feature = "real-models")]
    HfBert,
    #[cfg(feature = "real-models")]
    HfGpt2,
    #[cfg(feature = "real-models")]
    HfT5Small,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum CloudProvider {
    Aws,
    Gcp,
    Azure,
}

#[cfg(feature = "real-models")]
#[derive(ValueEnum, Clone, Debug)]
pub enum ModelFormat {
    Auto,
    Onnx,
    PyTorch,
    HuggingFace,
    #[cfg(feature = "tensorflow")]
    TensorFlow,
}

#[cfg(feature = "real-models")]
#[derive(ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    Table,
    Json,
    Csv,
    Markdown,
}

#[cfg(feature = "real-models")]
#[derive(ValueEnum, Clone, Debug)]
pub enum WorkloadType {
    Inference,
    Training,
    BatchProcessing,
    RealTime,
}

#[cfg(feature = "real-models")]
#[derive(ValueEnum, Clone, Debug, Copy)]
pub enum Precision {
    FP32,
    FP16,
    INT8,
}

#[cfg(feature = "real-models")]
impl Precision {
    pub fn to_real_hardware_precision(&self) -> crate::real_hardware_model::Precision {
        match self {
            Precision::FP32 => crate::real_hardware_model::Precision::FP32,
            Precision::FP16 => crate::real_hardware_model::Precision::FP16,
            Precision::INT8 => crate::real_hardware_model::Precision::INT8,
        }
    }
}

impl GpuType {
    pub fn to_gpu_model(&self) -> GpuModel {
        let gpu_manager = crate::gpu_config::GpuModelManager
            ::load()
            .expect("Failed to load GPU configuration");

        gpu_manager
            .get_gpu_by_type(self)
            .expect(&format!("GPU type {:?} not found in configuration", self))
    }
}
