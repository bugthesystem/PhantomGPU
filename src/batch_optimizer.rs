//! Batch Size Optimization Engine
//!
//! This module provides intelligent batch size recommendations based on:
//! - GPU memory constraints
//! - Compute efficiency curves
//! - Model architecture characteristics
//! - Throughput vs. memory trade-offs
//! - Architecture-specific optimizations

use serde::{ Deserialize, Serialize };
use std::collections::HashMap;
use crate::gpu_config::GpuModel;
use crate::errors::PhantomResult;

/// Batch optimization result with recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOptimizationResult {
    pub gpu_name: String,
    pub model_name: String,
    pub optimal_batch_size: usize,
    pub max_safe_batch_size: usize,
    pub memory_utilization: f64,
    pub throughput_samples_per_sec: f64,
    pub memory_efficiency: f64,
    pub compute_efficiency: f64,
    pub recommendations: Vec<String>,
    pub batch_analysis: Vec<BatchAnalysis>,
}

/// Analysis for a specific batch size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchAnalysis {
    pub batch_size: usize,
    pub memory_usage_mb: f64,
    pub memory_utilization: f64,
    pub throughput_samples_per_sec: f64,
    pub latency_ms: f64,
    pub compute_efficiency: f64,
    pub is_viable: bool,
    pub notes: Vec<String>,
}

/// Model memory and compute characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub name: String,
    pub base_memory_mb: f64,
    pub memory_per_sample_mb: f64,
    pub compute_intensity: f64,
    pub memory_bandwidth_sensitivity: f64,
    pub tensor_core_compatibility: f64,
    pub model_type: ModelType,
}

/// Model type for optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub enum ModelType {
    LargeLanguageModel,
    ConvolutionalNeuralNetwork,
    Transformer,
    VisionTransformer,
    ObjectDetection,
    GenerativeModel,
}

/// Batch optimization engine
pub struct BatchOptimizer {
    model_profiles: HashMap<String, ModelProfile>,
    gpu_memory_efficiency: HashMap<String, f64>,
    architecture_efficiency: HashMap<String, HashMap<ModelType, f64>>,
}

impl BatchOptimizer {
    /// Create a new batch optimizer
    pub fn new() -> Self {
        let mut optimizer = Self {
            model_profiles: HashMap::new(),
            gpu_memory_efficiency: HashMap::new(),
            architecture_efficiency: HashMap::new(),
        };

        optimizer.load_model_profiles();
        optimizer.load_gpu_characteristics();
        optimizer
    }

    /// Load predefined model profiles
    fn load_model_profiles(&mut self) {
        // Large Language Models
        self.model_profiles.insert("llama2-7b".to_string(), ModelProfile {
            name: "LLaMA 2 7B".to_string(),
            base_memory_mb: 13000.0,
            memory_per_sample_mb: 4.0,
            compute_intensity: 0.8,
            memory_bandwidth_sensitivity: 0.9,
            tensor_core_compatibility: 0.95,
            model_type: ModelType::LargeLanguageModel,
        });

        self.model_profiles.insert("llama2-13b".to_string(), ModelProfile {
            name: "LLaMA 2 13B".to_string(),
            base_memory_mb: 26000.0,
            memory_per_sample_mb: 6.0,
            compute_intensity: 0.85,
            memory_bandwidth_sensitivity: 0.95,
            tensor_core_compatibility: 0.95,
            model_type: ModelType::LargeLanguageModel,
        });

        self.model_profiles.insert("gpt35-turbo".to_string(), ModelProfile {
            name: "GPT-3.5 Turbo".to_string(),
            base_memory_mb: 15000.0,
            memory_per_sample_mb: 5.0,
            compute_intensity: 0.85,
            memory_bandwidth_sensitivity: 0.9,
            tensor_core_compatibility: 0.9,
            model_type: ModelType::LargeLanguageModel,
        });

        // Vision Transformers
        self.model_profiles.insert("vit-base-16".to_string(), ModelProfile {
            name: "ViT-Base/16".to_string(),
            base_memory_mb: 350.0,
            memory_per_sample_mb: 3.0,
            compute_intensity: 0.7,
            memory_bandwidth_sensitivity: 0.6,
            tensor_core_compatibility: 0.8,
            model_type: ModelType::VisionTransformer,
        });

        self.model_profiles.insert("vit-large-16".to_string(), ModelProfile {
            name: "ViT-Large/16".to_string(),
            base_memory_mb: 1200.0,
            memory_per_sample_mb: 4.5,
            compute_intensity: 0.75,
            memory_bandwidth_sensitivity: 0.7,
            tensor_core_compatibility: 0.85,
            model_type: ModelType::VisionTransformer,
        });

        // CNNs
        self.model_profiles.insert("resnet50".to_string(), ModelProfile {
            name: "ResNet-50".to_string(),
            base_memory_mb: 200.0,
            memory_per_sample_mb: 1.5,
            compute_intensity: 0.6,
            memory_bandwidth_sensitivity: 0.4,
            tensor_core_compatibility: 0.7,
            model_type: ModelType::ConvolutionalNeuralNetwork,
        });

        self.model_profiles.insert("yolov8".to_string(), ModelProfile {
            name: "YOLO v8".to_string(),
            base_memory_mb: 100.0,
            memory_per_sample_mb: 2.0,
            compute_intensity: 0.65,
            memory_bandwidth_sensitivity: 0.5,
            tensor_core_compatibility: 0.6,
            model_type: ModelType::ObjectDetection,
        });

        // Transformers
        self.model_profiles.insert("bert-base".to_string(), ModelProfile {
            name: "BERT-Base".to_string(),
            base_memory_mb: 400.0,
            memory_per_sample_mb: 2.0,
            compute_intensity: 0.7,
            memory_bandwidth_sensitivity: 0.8,
            tensor_core_compatibility: 0.9,
            model_type: ModelType::Transformer,
        });
    }

    /// Load GPU memory and compute characteristics
    fn load_gpu_characteristics(&mut self) {
        // Memory efficiency factors (how well GPU utilizes memory)
        self.gpu_memory_efficiency.insert("Tesla V100".to_string(), 0.85);
        self.gpu_memory_efficiency.insert("A100".to_string(), 0.92);
        self.gpu_memory_efficiency.insert("H100".to_string(), 0.95);
        self.gpu_memory_efficiency.insert("RTX 4090".to_string(), 0.88);
        self.gpu_memory_efficiency.insert("RTX 5090".to_string(), 0.9);

        // Architecture efficiency for different model types
        let mut volta_efficiency = HashMap::new();
        volta_efficiency.insert(ModelType::LargeLanguageModel, 0.6);
        volta_efficiency.insert(ModelType::ConvolutionalNeuralNetwork, 0.8);
        volta_efficiency.insert(ModelType::Transformer, 0.75);
        volta_efficiency.insert(ModelType::VisionTransformer, 0.5);
        volta_efficiency.insert(ModelType::ObjectDetection, 0.7);
        volta_efficiency.insert(ModelType::GenerativeModel, 0.6);
        self.architecture_efficiency.insert("Tesla V100".to_string(), volta_efficiency);

        let mut ampere_efficiency = HashMap::new();
        ampere_efficiency.insert(ModelType::LargeLanguageModel, 0.9);
        ampere_efficiency.insert(ModelType::ConvolutionalNeuralNetwork, 0.85);
        ampere_efficiency.insert(ModelType::Transformer, 0.9);
        ampere_efficiency.insert(ModelType::VisionTransformer, 0.8);
        ampere_efficiency.insert(ModelType::ObjectDetection, 0.85);
        ampere_efficiency.insert(ModelType::GenerativeModel, 0.85);
        self.architecture_efficiency.insert("A100".to_string(), ampere_efficiency);

        let mut hopper_efficiency = HashMap::new();
        hopper_efficiency.insert(ModelType::LargeLanguageModel, 0.95);
        hopper_efficiency.insert(ModelType::ConvolutionalNeuralNetwork, 0.9);
        hopper_efficiency.insert(ModelType::Transformer, 0.95);
        hopper_efficiency.insert(ModelType::VisionTransformer, 0.9);
        hopper_efficiency.insert(ModelType::ObjectDetection, 0.9);
        hopper_efficiency.insert(ModelType::GenerativeModel, 0.9);
        self.architecture_efficiency.insert("H100".to_string(), hopper_efficiency);

        let mut ada_efficiency = HashMap::new();
        ada_efficiency.insert(ModelType::LargeLanguageModel, 0.7);
        ada_efficiency.insert(ModelType::ConvolutionalNeuralNetwork, 0.9);
        ada_efficiency.insert(ModelType::Transformer, 0.8);
        ada_efficiency.insert(ModelType::VisionTransformer, 0.75);
        ada_efficiency.insert(ModelType::ObjectDetection, 0.85);
        ada_efficiency.insert(ModelType::GenerativeModel, 0.8);
        self.architecture_efficiency.insert("RTX 4090".to_string(), ada_efficiency);

        let mut blackwell_efficiency = HashMap::new();
        blackwell_efficiency.insert(ModelType::LargeLanguageModel, 0.85);
        blackwell_efficiency.insert(ModelType::ConvolutionalNeuralNetwork, 0.92);
        blackwell_efficiency.insert(ModelType::Transformer, 0.88);
        blackwell_efficiency.insert(ModelType::VisionTransformer, 0.85);
        blackwell_efficiency.insert(ModelType::ObjectDetection, 0.9);
        blackwell_efficiency.insert(ModelType::GenerativeModel, 0.88);
        self.architecture_efficiency.insert("RTX 5090".to_string(), blackwell_efficiency);
    }

    /// Optimize batch size for a specific model and GPU
    pub fn optimize_batch_size(
        &self,
        gpu_model: &GpuModel,
        model_name: &str,
        target_memory_utilization: f64
    ) -> PhantomResult<BatchOptimizationResult> {
        let model_profile = self.model_profiles
            .get(model_name)
            .ok_or_else(|| crate::errors::PhantomGpuError::InvalidModel {
                reason: format!("Model profile not found: {}", model_name),
            })?;

        let memory_efficiency = self.gpu_memory_efficiency.get(&gpu_model.name).unwrap_or(&0.85);

        let architecture_efficiency = self.architecture_efficiency
            .get(&gpu_model.name)
            .and_then(|arch| arch.get(&model_profile.model_type))
            .unwrap_or(&0.8);

        let available_memory = (gpu_model.memory_gb as f64) * 1024.0 * memory_efficiency;
        let mut batch_analyses = Vec::new();

        // Test batch sizes from 1 to reasonable maximum
        let max_theoretical_batch = ((available_memory - model_profile.base_memory_mb) /
            model_profile.memory_per_sample_mb) as usize;
        let max_test_batch = std::cmp::min(max_theoretical_batch, 1024);

        let mut optimal_batch_size = 1;
        let mut best_score = 0.0;
        let mut max_safe_batch_size = 1;

        for batch_size in (1..=max_test_batch).step_by(if max_test_batch > 64 { 8 } else { 1 }) {
            let analysis = self.analyze_batch_size(
                gpu_model,
                model_profile,
                batch_size,
                *memory_efficiency,
                *architecture_efficiency,
                available_memory
            );

            if analysis.is_viable {
                max_safe_batch_size = batch_size;

                // Score based on throughput and memory efficiency
                let score = analysis.throughput_samples_per_sec * analysis.memory_utilization;
                if score > best_score {
                    best_score = score;
                    optimal_batch_size = batch_size;
                }
            }

            batch_analyses.push(analysis);
        }

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            gpu_model,
            model_profile,
            optimal_batch_size,
            max_safe_batch_size,
            &batch_analyses
        );

        let optimal_analysis = batch_analyses
            .iter()
            .find(|a| a.batch_size == optimal_batch_size)
            .unwrap();

        Ok(BatchOptimizationResult {
            gpu_name: gpu_model.name.clone(),
            model_name: model_profile.name.clone(),
            optimal_batch_size,
            max_safe_batch_size,
            memory_utilization: optimal_analysis.memory_utilization,
            throughput_samples_per_sec: optimal_analysis.throughput_samples_per_sec,
            memory_efficiency: optimal_analysis.memory_utilization,
            compute_efficiency: optimal_analysis.compute_efficiency,
            recommendations,
            batch_analysis: batch_analyses,
        })
    }

    /// Analyze a specific batch size
    fn analyze_batch_size(
        &self,
        gpu_model: &GpuModel,
        model_profile: &ModelProfile,
        batch_size: usize,
        memory_efficiency: f64,
        architecture_efficiency: f64,
        available_memory: f64
    ) -> BatchAnalysis {
        let memory_usage =
            model_profile.base_memory_mb + (batch_size as f64) * model_profile.memory_per_sample_mb;
        let memory_utilization = memory_usage / available_memory;

        let is_viable = memory_utilization <= 0.95; // Leave 5% headroom

        let mut notes = Vec::new();

        if !is_viable {
            notes.push("Exceeds GPU memory capacity".to_string());
        }

        if memory_utilization > 0.9 {
            notes.push("High memory pressure - risk of OOM".to_string());
        }

        if memory_utilization < 0.3 {
            notes.push("Low memory utilization - consider larger batch".to_string());
        }

        // Calculate throughput based on compute efficiency curve
        let base_throughput = (gpu_model.compute_tflops as f64) * 10.0; // samples/sec baseline
        let batch_efficiency = self.calculate_batch_efficiency(batch_size, model_profile);
        let compute_efficiency = architecture_efficiency * batch_efficiency;

        let throughput_samples_per_sec =
            base_throughput * compute_efficiency * model_profile.compute_intensity;
        let latency_ms = ((batch_size as f64) / throughput_samples_per_sec) * 1000.0;

        // Add performance notes
        if batch_efficiency < 0.7 {
            notes.push("Inefficient batch size for this model".to_string());
        }

        if batch_size == 1 {
            notes.push("Single sample - high latency, low throughput".to_string());
        }

        if batch_size >= 64 {
            notes.push("Large batch - high throughput, high latency".to_string());
        }

        BatchAnalysis {
            batch_size,
            memory_usage_mb: memory_usage,
            memory_utilization,
            throughput_samples_per_sec,
            latency_ms,
            compute_efficiency,
            is_viable,
            notes,
        }
    }

    /// Calculate batch efficiency curve
    fn calculate_batch_efficiency(&self, batch_size: usize, model_profile: &ModelProfile) -> f64 {
        let batch_f = batch_size as f64;

        // Different models have different optimal batch sizes
        let optimal_batch = match model_profile.model_type {
            ModelType::LargeLanguageModel => 32.0,
            ModelType::ConvolutionalNeuralNetwork => 64.0,
            ModelType::Transformer => 16.0,
            ModelType::VisionTransformer => 32.0,
            ModelType::ObjectDetection => 8.0,
            ModelType::GenerativeModel => 16.0,
        };

        // Efficiency curve: ramps up to optimal, then slowly decreases
        if batch_f <= optimal_batch {
            // Ramp up phase
            0.5 + 0.5 * (batch_f / optimal_batch)
        } else {
            // Diminishing returns phase
            1.0 - 0.3 * ((batch_f - optimal_batch) / (optimal_batch * 2.0)).min(1.0)
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        gpu_model: &GpuModel,
        model_profile: &ModelProfile,
        optimal_batch: usize,
        max_safe_batch: usize,
        analyses: &[BatchAnalysis]
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Memory recommendations
        if let Some(optimal_analysis) = analyses.iter().find(|a| a.batch_size == optimal_batch) {
            if optimal_analysis.memory_utilization > 0.8 {
                recommendations.push(
                    "Consider gradient checkpointing to reduce memory usage".to_string()
                );
            }

            if optimal_analysis.memory_utilization < 0.5 {
                recommendations.push(
                    "GPU memory is underutilized - consider larger batch or model parallelism".to_string()
                );
            }
        }

        // Architecture-specific recommendations
        if let Some(arch) = &gpu_model.architecture {
            match arch.as_str() {
                "Ampere" | "Hopper" => {
                    if matches!(model_profile.model_type, ModelType::LargeLanguageModel) {
                        recommendations.push(
                            "Use mixed precision (FP16/BF16) for better throughput".to_string()
                        );
                    }
                }
                "Ada Lovelace" => {
                    recommendations.push(
                        "Consider using DLSS 3 Frame Generation for applicable workloads".to_string()
                    );
                }
                "Volta" => {
                    if optimal_batch > 32 {
                        recommendations.push(
                            "Volta architecture performs better with smaller batch sizes".to_string()
                        );
                    }
                }
                _ => {}
            }
        }

        // Model-specific recommendations
        match model_profile.model_type {
            ModelType::LargeLanguageModel => {
                if optimal_batch < 16 {
                    recommendations.push(
                        "Consider sequence parallelism for large language models".to_string()
                    );
                }
            }
            ModelType::ConvolutionalNeuralNetwork => {
                if optimal_batch > 128 {
                    recommendations.push(
                        "Very large CNN batches may cause numerical instability".to_string()
                    );
                }
            }
            ModelType::VisionTransformer => {
                recommendations.push(
                    "Vision Transformers benefit from larger image patches with smaller batches".to_string()
                );
            }
            _ => {}
        }

        // Performance recommendations
        if optimal_batch != max_safe_batch {
            recommendations.push(
                format!("Consider batch size {} for maximum throughput if latency is not critical", max_safe_batch)
            );
        }

        recommendations
    }

    /// Get model profile
    pub fn get_model_profile(&self, model_name: &str) -> Option<&ModelProfile> {
        self.model_profiles.get(model_name)
    }

    /// List available models
    pub fn list_models(&self) -> Vec<String> {
        self.model_profiles.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_optimizer_creation() {
        let optimizer = BatchOptimizer::new();
        assert!(!optimizer.model_profiles.is_empty());
        assert!(!optimizer.gpu_memory_efficiency.is_empty());
    }

    #[test]
    fn test_batch_size_optimization() {
        let optimizer = BatchOptimizer::new();
        let gpu_model = GpuModel {
            name: "A100".to_string(),
            memory_gb: 80.0,
            compute_tflops: 312.0,
            memory_bandwidth_gbps: 1935.0,
            architecture: Some("Ampere".to_string()),
            release_year: Some(2020),
        };

        let result = optimizer.optimize_batch_size(&gpu_model, "llama2-7b", 0.8);
        assert!(result.is_ok());

        let optimization = result.unwrap();
        assert!(optimization.optimal_batch_size > 0);
        assert!(optimization.max_safe_batch_size >= optimization.optimal_batch_size);
    }

    #[test]
    fn test_batch_efficiency_calculation() {
        let optimizer = BatchOptimizer::new();
        let profile = optimizer.get_model_profile("llama2-7b").unwrap();

        let efficiency_1 = optimizer.calculate_batch_efficiency(1, profile);
        let efficiency_32 = optimizer.calculate_batch_efficiency(32, profile);
        let efficiency_128 = optimizer.calculate_batch_efficiency(128, profile);

        assert!(efficiency_32 > efficiency_1);
        assert!(efficiency_32 > efficiency_128);
    }
}
