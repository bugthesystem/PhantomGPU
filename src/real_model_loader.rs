//! Real Model Loading Support
//!
//! This module provides support for loading real ML models from various sources:
//! - ONNX models (local files and remote URLs)
//! - TensorFlow SavedModel and .pb files
//! - Hugging Face Hub models
//! - PyTorch .pth/.pt files
//!
//! This enables benchmarking thousands of real-world models instead of synthetic ones.

use crate::errors::{ PhantomGpuError, PhantomResult };
use crate::models::ModelConfig;
use serde::{ Deserialize, Serialize };
use std::path::{ Path, PathBuf };
use tracing::{ info, warn };

/// Supported model formats for real model loading
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelFormat {
    /// ONNX format (.onnx files)
    Onnx,
    /// TensorFlow SavedModel directory
    TensorFlowSavedModel,
    /// TensorFlow frozen graph (.pb file)
    TensorFlowFrozen,
    /// TensorFlow Lite format (.tflite files)
    TensorFlowLite,
    /// Keras format (.h5 files)
    TensorFlowKeras,
    /// PyTorch state dict (.pth/.pt files)
    PyTorch,
    /// Hugging Face Hub model (transformers format)
    HuggingFace,
}

/// Information about a loaded real model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealModelInfo {
    pub name: String,
    pub format: ModelFormat,
    pub file_path: Option<PathBuf>,
    pub hub_id: Option<String>,
    pub model_size_mb: f64,
    pub parameter_count: Option<u64>,
    pub input_shapes: Vec<Vec<i64>>,
    pub output_shapes: Vec<Vec<i64>>,
    pub estimated_memory_mb: f64,
    pub estimated_flops_g: f64,
}

/// Real model loader for different formats
pub struct RealModelLoader {
    pub cache_dir: PathBuf,
}

impl RealModelLoader {
    /// Create a new real model loader with specified cache directory
    pub fn new(cache_dir: Option<PathBuf>) -> Self {
        let cache_dir = cache_dir.unwrap_or_else(|| {
            std::env::temp_dir().join("phantom_gpu_models")
        });

        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir).unwrap_or_else(|e| {
                warn!("Failed to create cache directory: {}", e);
            });
        }

        Self { cache_dir }
    }

    /// Load a model from an ONNX file
    #[cfg(feature = "onnx")]
    pub async fn load_onnx<P: AsRef<Path>>(&self, path: P) -> PhantomResult<RealModelInfo> {
        let path = path.as_ref();
        info!("Loading ONNX model from: {}", path.display());

        // For now, use a simplified approach for ONNX model metadata extraction
        // We'll read the file and use basic heuristics to estimate the model info
        info!("📊 Analyzing ONNX model metadata: {}", path.display());

        // Get basic model metadata from file size and common patterns
        let input_shapes = vec![vec![1, 3, 224, 224]]; // Common image input shape
        let output_shapes = vec![vec![1, 1000]]; // Common classification output

        let file_size_mb = std::fs
            ::metadata(path)
            .map(|m| (m.len() as f64) / 1024.0 / 1024.0)
            .unwrap_or(0.0);

        let estimated_memory_mb = Self::estimate_memory_from_shapes(&input_shapes, &output_shapes);
        let estimated_flops_g = Self::estimate_flops_from_model_size(file_size_mb);

        Ok(RealModelInfo {
            name: path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            format: ModelFormat::Onnx,
            file_path: Some(path.to_path_buf()),
            hub_id: None,
            model_size_mb: file_size_mb,
            parameter_count: Some(((file_size_mb * 1024.0 * 1024.0) / 4.0) as u64), // Assume FP32
            input_shapes,
            output_shapes,
            estimated_memory_mb,
            estimated_flops_g,
        })
    }

    /// Load a model from Hugging Face Hub
    #[cfg(feature = "huggingface")]
    pub async fn load_huggingface(&self, model_id: &str) -> PhantomResult<RealModelInfo> {
        use hf_hub::api::tokio::Api;

        info!("Loading model from Hugging Face Hub: {}", model_id);

        let api = Api::new().map_err(|e|
            PhantomGpuError::ModelLoadError(format!("HF Hub API error: {}", e))
        )?;

        let repo = api.model(model_id.to_string());

        // Try to find model files (ONNX preferred, then PyTorch)
        let model_files = vec!["model.onnx", "pytorch_model.bin", "model.safetensors"];
        let mut downloaded_file = None;
        let mut model_format = ModelFormat::HuggingFace;

        for filename in model_files {
            match repo.get(filename).await {
                Ok(file_path) => {
                    downloaded_file = Some(file_path);
                    model_format = match filename {
                        "model.onnx" => ModelFormat::Onnx,
                        _ => ModelFormat::PyTorch,
                    };
                    break;
                }
                Err(_) => {
                    continue;
                }
            }
        }

        let file_path = downloaded_file.ok_or_else(||
            PhantomGpuError::ModelLoadError(
                "No compatible model file found in Hugging Face repo".to_string()
            )
        )?;

        // Get model info from config.json if available
        let (estimated_memory_mb, parameter_count) = match repo.get("config.json").await {
            Ok(config_path) => {
                let config_content = tokio::fs
                    ::read_to_string(config_path).await
                    .unwrap_or_default();
                Self::parse_hf_config(&config_content)
            }
            Err(_) => (128.0, None), // Default fallback
        };

        let file_size_mb = tokio::fs
            ::metadata(&file_path).await
            .map(|m| (m.len() as f64) / 1024.0 / 1024.0)
            .unwrap_or(0.0);

        let estimated_flops_g = Self::estimate_flops_from_model_size(file_size_mb);

        Ok(RealModelInfo {
            name: model_id.to_string(),
            format: model_format,
            file_path: Some(file_path),
            hub_id: Some(model_id.to_string()),
            model_size_mb: file_size_mb,
            parameter_count,
            input_shapes: vec![vec![1, 512]], // Default for most transformer models
            output_shapes: vec![vec![1, 512, 768]], // Default transformer output
            estimated_memory_mb,
            estimated_flops_g,
        })
    }

    /// Load a TensorFlow model with PyO3-based parsing
    pub async fn load_tensorflow<P: AsRef<Path>>(&self, path: P) -> PhantomResult<RealModelInfo> {
        let path = path.as_ref();
        info!("Loading TensorFlow model from: {}", path.display());

        // Use PyO3-based TensorFlow parser
        #[cfg(feature = "tensorflow")]
        {
            use crate::tensorflow_parser::TensorFlowParser;

            let path_str = path.to_string_lossy().to_string();

            if path.is_dir() && path.join("saved_model.pb").exists() {
                info!("🔍 Detected SavedModel format");
            } else if path.is_file() && path.extension().map_or(false, |ext| ext == "pb") {
                info!("🔍 Detected frozen graph format");
            } else if path.is_file() && path.extension().map_or(false, |ext| ext == "tflite") {
                info!("🔍 Detected TensorFlow Lite format");
            } else if path.is_file() && path.extension().map_or(false, |ext| ext == "h5") {
                info!("🔍 Detected Keras format");
            } else {
                return Err(
                    PhantomGpuError::ModelLoadError(
                        "Invalid TensorFlow model format. Expected SavedModel directory, .pb, .tflite, or .h5 file".to_string()
                    )
                );
            }

            let tf_analysis = TensorFlowParser::parse_model(&path_str)?;

            // Convert TensorFlowAnalysis to RealModelInfo
            let format = match tf_analysis.format.as_str() {
                "SavedModel" => ModelFormat::TensorFlowSavedModel,
                "FrozenGraph" => ModelFormat::TensorFlowFrozen,
                "TensorFlowLite" => ModelFormat::TensorFlowLite,
                "Keras" => ModelFormat::TensorFlowKeras,
                _ => ModelFormat::TensorFlowFrozen,
            };

            Ok(RealModelInfo {
                name: tf_analysis.model_name,
                format,
                file_path: Some(path.to_path_buf()),
                hub_id: None,
                model_size_mb: tf_analysis.model_size_mb,
                parameter_count: Some(tf_analysis.total_parameters as u64),
                input_shapes: tf_analysis.input_shapes,
                output_shapes: tf_analysis.output_shapes,
                estimated_memory_mb: tf_analysis.estimated_memory_mb,
                estimated_flops_g: (
                    tf_analysis.layers
                        .iter()
                        .map(|l| l.flops)
                        .sum::<i64>() as f64
                ) / 1e9, // Convert to GFLOPs
            })
        }

        // Fallback implementation when tensorflow feature is not enabled
        #[cfg(not(feature = "tensorflow"))]
        {
            warn!("⚠️ TensorFlow support not enabled. Using basic file analysis.");
            warn!("💡 Enable with: cargo build --features tensorflow");

            // Determine format from path structure
            let format = if path.is_dir() && path.join("saved_model.pb").exists() {
                ModelFormat::TensorFlowSavedModel
            } else {
                ModelFormat::TensorFlowFrozen
            };

            // Calculate model size
            let model_size_mb = if path.is_dir() {
                Self::calculate_directory_size(path)? / 1024.0 / 1024.0
            } else {
                std::fs
                    ::metadata(path)
                    .map(|m| (m.len() as f64) / 1024.0 / 1024.0)
                    .unwrap_or(0.0)
            };

            let estimated_memory_mb = Self::estimate_memory_from_file_size(model_size_mb);
            let estimated_flops_g = Self::estimate_flops_from_model_size(model_size_mb);

            Ok(RealModelInfo {
                name: path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("tensorflow_model")
                    .to_string(),
                format,
                file_path: Some(path.to_path_buf()),
                hub_id: None,
                model_size_mb,
                parameter_count: Some(((model_size_mb * 1024.0 * 1024.0) / 4.0) as u64),
                input_shapes: vec![vec![1, 224, 224, 3]], // Default for image models
                output_shapes: vec![vec![1, 1000]], // Default classification output
                estimated_memory_mb,
                estimated_flops_g,
            })
        }
    }

    /// Load a model from various sources (auto-detects format)
    pub async fn load_from_source(&self, source: &str) -> PhantomResult<RealModelInfo> {
        info!("Auto-detecting model format for: {}", source);

        // Check if it's a local file or directory first
        let path = std::path::Path::new(source);
        if path.exists() {
            if source.ends_with(".onnx") {
                #[cfg(feature = "onnx")]
                {
                    return self.load_onnx(source).await;
                }
                #[cfg(not(feature = "onnx"))]
                {
                    return Err(
                        PhantomGpuError::ModelLoadError(
                            "ONNX support not enabled. Use --features onnx".to_string()
                        )
                    );
                }
            } else if
                source.ends_with(".pb") ||
                source.ends_with(".tflite") ||
                source.ends_with(".h5") ||
                (path.is_dir() && path.join("saved_model.pb").exists())
            {
                return self.load_tensorflow(source).await;
            } else if source.ends_with(".pth") || source.ends_with(".pt") {
                // For now, treat PyTorch files as generic models
                return self.load_pytorch_placeholder(source).await;
            }
        }

        // Check if it's a Hugging Face model ID (contains a slash and no file extension)
        if
            source.contains('/') &&
            !source.ends_with(".onnx") &&
            !source.ends_with(".pb") &&
            !source.ends_with(".tflite") &&
            !source.ends_with(".h5") &&
            !source.ends_with(".pth") &&
            !source.ends_with(".pt")
        {
            #[cfg(feature = "huggingface")]
            {
                return self.load_huggingface(source).await;
            }
            #[cfg(not(feature = "huggingface"))]
            {
                return Err(
                    PhantomGpuError::ModelLoadError(
                        "Hugging Face support not enabled. Use --features huggingface".to_string()
                    )
                );
            }
        }

        // Fallback: try to load as Hugging Face model
        #[cfg(feature = "huggingface")]
        {
            self.load_huggingface(source).await
        }
        #[cfg(not(feature = "huggingface"))]
        {
            Err(
                PhantomGpuError::ModelLoadError(
                    format!("Could not detect model format for: {}. Ensure the file exists or use a valid Hugging Face model ID.", source)
                )
            )
        }
    }

    /// Load a PyTorch model (placeholder implementation)
    pub async fn load_pytorch_placeholder(&self, source: &str) -> PhantomResult<RealModelInfo> {
        let path = std::path::Path::new(source);
        let file_size_mb = std::fs
            ::metadata(path)
            .map(|m| (m.len() as f64) / 1024.0 / 1024.0)
            .unwrap_or(0.0);

        Ok(RealModelInfo {
            name: path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("pytorch_model")
                .to_string(),
            format: ModelFormat::PyTorch,
            file_path: Some(path.to_path_buf()),
            hub_id: None,
            model_size_mb: file_size_mb,
            parameter_count: Some(((file_size_mb * 1024.0 * 1024.0) / 4.0) as u64),
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            estimated_memory_mb: file_size_mb * 2.0,
            estimated_flops_g: file_size_mb * 0.5,
        })
    }

    /// Convert RealModelInfo to ModelConfig for emulation
    pub fn to_model_config(&self, model_info: &RealModelInfo, batch_size: usize) -> ModelConfig {
        // Convert i64 input shapes to usize
        let input_shape: Vec<usize> = model_info.input_shapes
            .get(0)
            .unwrap_or(&vec![3, 224, 224]) // Default image shape
            .iter()
            .map(|&x| if x > 0 { x as usize } else { 1 })
            .collect();

        ModelConfig {
            name: model_info.name.clone(),
            batch_size,
            input_shape,
            parameters_m: ((model_info.parameter_count.unwrap_or(0) as f64) / 1_000_000.0) as f32,
            flops_per_sample_g: model_info.estimated_flops_g as f32,
            model_type: Self::infer_model_type(&model_info.name, &model_info.format),
            precision: "fp32".to_string(), // Default precision
        }
    }

    /// Estimate memory usage from tensor shapes
    fn estimate_memory_from_shapes(input_shapes: &[Vec<i64>], output_shapes: &[Vec<i64>]) -> f64 {
        let input_elements: i64 = input_shapes
            .iter()
            .map(|shape| shape.iter().product::<i64>())
            .sum();

        let output_elements: i64 = output_shapes
            .iter()
            .map(|shape| shape.iter().product::<i64>())
            .sum();

        // Estimate memory: (inputs + outputs + intermediate) * 4 bytes (FP32)
        let total_elements = input_elements + output_elements;
        let memory_bytes = (total_elements as f64) * 4.0 * 2.0; // 2x for intermediate activations
        memory_bytes / 1024.0 / 1024.0 // Convert to MB
    }

    /// Estimate memory from file size
    fn estimate_memory_from_file_size(file_size_mb: f64) -> f64 {
        // Model weights + activations (rough estimate: 2x model size)
        file_size_mb * 2.0
    }

    /// Estimate FLOPs from model size (very rough heuristic)
    fn estimate_flops_from_model_size(model_size_mb: f64) -> f64 {
        // Rough heuristic: 1MB model size ≈ 0.5 GFLOPs for inference
        model_size_mb * 0.5
    }

    /// Parse Hugging Face config.json for model info
    fn parse_hf_config(config_content: &str) -> (f64, Option<u64>) {
        match serde_json::from_str::<serde_json::Value>(config_content) {
            Ok(config) => {
                let hidden_size = config
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(768);

                let num_layers = config
                    .get("num_hidden_layers")
                    .or_else(|| config.get("num_layers"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(12);

                let vocab_size = config
                    .get("vocab_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(30522);

                // Rough parameter estimation for transformer models
                let estimated_params =
                    hidden_size * hidden_size * num_layers * 12 + vocab_size * hidden_size;
                let estimated_memory_mb = ((estimated_params * 4) as f64) / 1024.0 / 1024.0; // FP32

                (estimated_memory_mb, Some(estimated_params))
            }
            Err(_) => (128.0, None), // Fallback
        }
    }

    /// Calculate total size of a directory
    fn calculate_directory_size(dir: &Path) -> PhantomResult<f64> {
        let mut total_size = 0;

        fn visit_dir(dir: &Path, total_size: &mut u64) -> std::io::Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    visit_dir(&path, total_size)?;
                } else {
                    *total_size += entry.metadata()?.len();
                }
            }
            Ok(())
        }

        visit_dir(dir, &mut total_size).map_err(|e|
            PhantomGpuError::ModelLoadError(format!("Directory size calculation failed: {}", e))
        )?;

        Ok(total_size as f64)
    }

    /// Estimate compute intensity based on model format
    fn estimate_compute_intensity(format: &ModelFormat) -> f64 {
        match format {
            ModelFormat::Onnx => 0.7,
            ModelFormat::TensorFlowSavedModel | ModelFormat::TensorFlowFrozen => 0.6,
            ModelFormat::TensorFlowLite => 0.5, // Optimized for mobile/edge
            ModelFormat::TensorFlowKeras => 0.6,
            ModelFormat::PyTorch => 0.8,
            ModelFormat::HuggingFace => 0.9, // Transformers are typically compute-intensive
        }
    }

    /// Estimate memory intensity from input shapes
    fn estimate_memory_intensity(input_shapes: &[Vec<i64>]) -> f64 {
        let total_input_size: i64 = input_shapes
            .iter()
            .map(|shape| shape.iter().product::<i64>())
            .sum();

        // Larger inputs typically mean more memory-intensive workloads
        if total_input_size > 1_000_000 {
            0.8
        } else if total_input_size > 100_000 {
            0.6
        } else {
            0.4
        }
    }

    /// Convert model format to model type string
    fn format_to_model_type(format: &ModelFormat) -> String {
        match format {
            ModelFormat::Onnx => "ONNX".to_string(),
            ModelFormat::TensorFlowSavedModel => "TensorFlow SavedModel".to_string(),
            ModelFormat::TensorFlowFrozen => "TensorFlow Frozen".to_string(),
            ModelFormat::TensorFlowLite => "TensorFlow Lite".to_string(),
            ModelFormat::TensorFlowKeras => "Keras".to_string(),
            ModelFormat::PyTorch => "PyTorch".to_string(),
            ModelFormat::HuggingFace => "Hugging Face Transformers".to_string(),
        }
    }

    /// Infer model type for bottleneck analysis based on name and format
    fn infer_model_type(name: &str, format: &ModelFormat) -> String {
        let name_lower = name.to_lowercase();

        // Check for transformer/LLM models
        if
            name_lower.contains("bert") ||
            name_lower.contains("gpt") ||
            name_lower.contains("transformer") ||
            name_lower.contains("distilbert") ||
            name_lower.contains("bart") ||
            name_lower.contains("t5") ||
            name_lower.contains("llama") ||
            name_lower.contains("mistral") ||
            name_lower.contains("qwen") ||
            name_lower.contains("gemma") ||
            name_lower.contains("phi") ||
            name_lower.contains("deepseek") ||
            format == &ModelFormat::HuggingFace
        {
            return "transformer".to_string();
        }

        // Check for CNN models
        if
            name_lower.contains("resnet") ||
            name_lower.contains("alexnet") ||
            name_lower.contains("vgg") ||
            name_lower.contains("mobilenet") ||
            name_lower.contains("efficientnet") ||
            name_lower.contains("inception") ||
            name_lower.contains("yolo") ||
            name_lower.contains("cnn") ||
            name_lower.contains("vision")
        {
            return "cnn".to_string();
        }

        // Check for RNN models
        if name_lower.contains("lstm") || name_lower.contains("gru") || name_lower.contains("rnn") {
            return "rnn".to_string();
        }

        // Default based on format
        match format {
            ModelFormat::HuggingFace => "transformer".to_string(),
            _ => "cnn".to_string(), // Default to CNN for unknown models
        }
    }
}

/// Popular real models for quick testing
pub struct PopularRealModels;

impl PopularRealModels {
    /// Get a list of popular ONNX models for testing
    pub fn popular_onnx_models() -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "ResNet50",
                "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx",
            ),
            (
                "BERT Base",
                "https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx",
            ),
            (
                "GPT-2",
                "https://github.com/onnx/models/raw/main/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx",
            ),
            (
                "MobileNetV2",
                "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
            ),
            ("DistilBERT", "https://huggingface.co/distilbert-base-uncased/resolve/main/model.onnx")
        ]
    }

    /// Get a list of popular Hugging Face models
    pub fn popular_huggingface_models() -> Vec<&'static str> {
        vec![
            "distilbert-base-uncased",
            "bert-base-uncased",
            "gpt2",
            "microsoft/DialoGPT-medium",
            "facebook/bart-base",
            "t5-small",
            "google/electra-small-discriminator",
            "microsoft/CodeBERT-base",
            "sentence-transformers/all-MiniLM-L6-v2",
            "facebook/opt-125m"
        ]
    }

    /// Download and cache a popular model
    pub async fn download_popular_model(
        model_name: &str,
        cache_dir: &Path
    ) -> PhantomResult<PathBuf> {
        let models = Self::popular_onnx_models();

        if let Some((_, url)) = models.iter().find(|(name, _)| *name == model_name) {
            let file_name = format!("{}.onnx", model_name.to_lowercase().replace(" ", "_"));
            let cache_path = cache_dir.join(&file_name);

            if cache_path.exists() {
                info!("Using cached model: {}", cache_path.display());
                return Ok(cache_path);
            }

            // TODO: Implement model downloading with proper network dependencies
            warn!("Model downloading not yet implemented. Please download manually from: {}", url);
            Err(
                PhantomGpuError::ModelLoadError(
                    format!(
                        "Please download {} manually from {} to {}",
                        model_name,
                        url,
                        cache_path.display()
                    )
                )
            )
        } else {
            Err(PhantomGpuError::ModelLoadError(format!("Unknown popular model: {}", model_name)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_loader_creation() {
        let loader = RealModelLoader::new(None);
        assert!(
            loader.cache_dir.exists() ||
                loader.cache_dir.to_string_lossy().contains("phantom_gpu_models")
        );
    }

    #[test]
    fn test_popular_models_list() {
        let onnx_models = PopularRealModels::popular_onnx_models();
        assert!(!onnx_models.is_empty());
        assert!(onnx_models.iter().any(|(name, _)| name.contains("ResNet")));

        let hf_models = PopularRealModels::popular_huggingface_models();
        assert!(!hf_models.is_empty());
        assert!(hf_models.iter().any(|name| name.contains("bert")));
    }

    #[test]
    fn test_memory_estimation() {
        let input_shapes = vec![vec![1, 3, 224, 224]]; // Typical image input
        let output_shapes = vec![vec![1, 1000]]; // ImageNet classes

        let memory_mb = RealModelLoader::estimate_memory_from_shapes(&input_shapes, &output_shapes);
        assert!(memory_mb > 0.0);
        assert!(memory_mb < 1000.0); // Reasonable bound
    }
}
