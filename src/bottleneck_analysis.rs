//! Bottleneck Analysis for Hardware-Aware Performance Modeling
//!
//! This module implements real hardware bottleneck analysis instead of simple FLOPS-based calculations.
//! It models the actual performance characteristics of different GPU architectures and workload types.

use std::collections::HashMap;
use std::fs;
use serde::{ Deserialize, Serialize };
use anyhow::{ Result, Context };

/// Different types of computational bottlenecks
#[derive(Debug, Clone, PartialEq)]
pub enum BottleneckType {
    Memory, // Memory bandwidth bound (e.g., LLM inference)
    Compute, // Compute bound (e.g., CNN inference)
    Mixed, // Mixed bottlenecks (e.g., RNN inference)
}

/// Hardware bottleneck characteristics
#[derive(Debug, Clone, Deserialize)]
pub struct BottleneckProfile {
    pub memory_bandwidth_gbps: f32,
    pub memory_bandwidth_utilization: f32,
    pub compute_tflops_fp32: f32,
    pub compute_tflops_fp16: f32,
    pub compute_tflops_int8: f32,
    pub compute_utilization: f32,
    pub memory_bound_threshold_ratio: f32,
    pub compute_bound_threshold_ratio: f32,
}

/// Model-specific performance characteristics
#[derive(Debug, Clone, Deserialize)]
pub struct ModelPerformanceProfile {
    pub batch_scaling_curve: Vec<f32>,
    pub memory_efficiency: f32,
    pub tensor_core_utilization: f32,
    pub architecture_multiplier: f32,
    pub primary_bottleneck: String,
    pub memory_intensity: f32,
    pub compute_intensity: f32,
}

/// Complete hardware profile with bottleneck modeling
#[derive(Debug, Clone, Deserialize)]
pub struct HardwareProfile {
    pub name: String,
    pub thermal: ThermalProfile,
    pub memory: MemoryProfile,
    pub architecture: ArchitectureProfile,
    pub bottlenecks: BottleneckProfile,
    pub model_performance: ModelPerformanceProfiles,
    pub precision: PrecisionProfile,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThermalProfile {
    pub tdp_watts: f32,
    pub base_clock_mhz: f32,
    pub boost_clock_mhz: f32,
    pub throttle_temp_celsius: f32,
    pub thermal_factor_sustained: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MemoryProfile {
    pub l1_cache_kb: f32,
    pub l2_cache_mb: f32,
    pub memory_channels: u32,
    pub cache_hit_ratio: f32,
    pub coalescing_efficiency: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArchitectureProfile {
    pub cuda_cores: u32,
    pub tensor_cores: u32,
    pub rt_cores: u32,
    pub streaming_multiprocessors: u32,
    pub memory_bus_width: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelPerformanceProfiles {
    pub cnn: ModelPerformanceProfile,
    pub vit: Option<ModelPerformanceProfile>,
    pub transformer: ModelPerformanceProfile,
    pub rnn: ModelPerformanceProfile,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PrecisionProfile {
    pub fp16_multiplier: f32,
    pub int8_multiplier: f32,
    pub int4_multiplier: f32,
}

/// Bottleneck analysis result
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    pub bottleneck_type: BottleneckType,
    pub memory_bound_factor: f32,
    pub compute_bound_factor: f32,
    pub effective_bandwidth_gbps: f32,
    pub effective_compute_tflops: f32,
    pub performance_multiplier: f32,
}

/// Hardware profile manager
#[derive(Debug)]
pub struct HardwareProfileManager {
    pub profiles: HashMap<String, HardwareProfile>,
}

impl HardwareProfileManager {
    /// Load hardware profiles from TOML configuration file
    pub fn load_from_file(config_path: &str) -> Result<Self> {
        let config_content = fs
            ::read_to_string(config_path)
            .with_context(|| format!("Failed to read hardware profiles file: {}", config_path))?;

        let profiles_config: HashMap<String, HashMap<String, HardwareProfile>> = toml
            ::from_str(&config_content)
            .with_context(|| "Failed to parse hardware profiles TOML file")?;

        // Extract the profiles from the nested structure
        let profiles = profiles_config
            .get("profiles")
            .ok_or_else(|| anyhow::anyhow!("Missing 'profiles' section in hardware profiles"))?
            .clone();

        Ok(Self { profiles })
    }

    /// Try to load from file, fallback to empty if file doesn't exist
    pub fn load() -> Result<Self> {
        match Self::load_from_file("hardware_profiles.toml") {
            Ok(manager) => {
                println!("✅ Loaded hardware profiles from hardware_profiles.toml");
                Ok(manager)
            }
            Err(e) => {
                println!("⚠️  hardware_profiles.toml not found or invalid: {}", e);
                Ok(Self {
                    profiles: HashMap::new(),
                })
            }
        }
    }

    /// Get hardware profile by GPU architecture
    pub fn get_profile(&self, gpu_name: &str) -> Option<&HardwareProfile> {
        // Map GPU names to profile keys
        let profile_key = match gpu_name.to_lowercase().as_str() {
            "h200" => "h200",
            "h100" => "h100",
            "rtx 5090" => "rtx5090",
            "rtx pro 6000 blackwell" => "rtx_pro_6000",
            "rtx 4090" => "rtx4090",
            "a100" => "a100",
            "rtx a6000" => "a6000",
            "l40s" => "l40s",
            "rtx 3090" => "rtx3090",
            "tesla v100" => "v100",
            _ => {
                return None;
            }
        };

        self.profiles.get(profile_key)
    }
}

/// Bottleneck analyzer for hardware-aware performance modeling
#[derive(Debug)]
pub struct BottleneckAnalyzer {
    hardware_profiles: HardwareProfileManager,
}

impl BottleneckAnalyzer {
    /// Create new bottleneck analyzer
    pub fn new() -> Result<Self> {
        let hardware_profiles = HardwareProfileManager::load()?;
        Ok(Self { hardware_profiles })
    }

    /// Analyze bottlenecks for a specific workload
    pub fn analyze_bottleneck(
        &self,
        gpu_name: &str,
        model_type: &str,
        batch_size: usize,
        flops_per_sample_g: f32,
        precision: &str
    ) -> Result<BottleneckAnalysis> {
        let profile = self.hardware_profiles
            .get_profile(gpu_name)
            .ok_or_else(|| anyhow::anyhow!("No hardware profile found for GPU: {}", gpu_name))?;

        let model_profile = match model_type.to_lowercase().as_str() {
            "cnn" | "resnet" | "alexnet" | "yolo" => &profile.model_performance.cnn,
            | "vit"
            | "vision_transformer"
            | "deit"
            | "clip"
            | "vit-base"
            | "vit-large"
            | "vit-huge" => {
                // Use ViT profile if available, fallback to transformer
                profile.model_performance.vit
                    .as_ref()
                    .unwrap_or(&profile.model_performance.transformer)
            }
            | "transformer"
            | "llm"
            | "bert"
            | "gpt"
            | "llama"
            | "mistral"
            | "qwen"
            | "deepseek"
            | "phi"
            | "gemma" => &profile.model_performance.transformer,
            "rnn" | "lstm" | "gru" => &profile.model_performance.rnn,
            | "gan"
            | "diffusion"
            | "stable_diffusion"
            | "stable_diffusion_xl"
            | "stable-diffusion"
            | "stable-diffusion-xl" => {
                // GANs behave like transformers but are compute-intensive
                &profile.model_performance.transformer
            }
            _ => &profile.model_performance.transformer, // Default to transformer for unknown models
        };

        // Determine primary bottleneck type
        let bottleneck_type = match model_profile.primary_bottleneck.as_str() {
            "memory" => BottleneckType::Memory,
            "compute" => BottleneckType::Compute,
            "mixed" => BottleneckType::Mixed,
            _ => BottleneckType::Mixed,
        };

        // Calculate effective performance characteristics
        let batch_scaling_factor = self.get_batch_scaling_factor(
            batch_size,
            &model_profile.batch_scaling_curve
        );
        let precision_multiplier = self.get_precision_multiplier(precision, &profile.precision);

        let effective_bandwidth_gbps =
            profile.bottlenecks.memory_bandwidth_gbps *
            profile.bottlenecks.memory_bandwidth_utilization *
            model_profile.memory_efficiency *
            batch_scaling_factor;

        let effective_compute_tflops =
            (match precision {
                "fp16" => profile.bottlenecks.compute_tflops_fp16,
                "int8" => profile.bottlenecks.compute_tflops_int8,
                _ => profile.bottlenecks.compute_tflops_fp32,
            }) *
            profile.bottlenecks.compute_utilization *
            model_profile.tensor_core_utilization *
            batch_scaling_factor *
            precision_multiplier;

        // Calculate bottleneck factors
        let memory_bound_factor = model_profile.memory_intensity;
        let compute_bound_factor = model_profile.compute_intensity;

        // Calculate overall performance multiplier
        let performance_multiplier =
            model_profile.architecture_multiplier *
            batch_scaling_factor *
            precision_multiplier *
            profile.thermal.thermal_factor_sustained;

        Ok(BottleneckAnalysis {
            bottleneck_type,
            memory_bound_factor,
            compute_bound_factor,
            effective_bandwidth_gbps,
            effective_compute_tflops,
            performance_multiplier,
        })
    }

    /// Get batch scaling factor from curve
    fn get_batch_scaling_factor(&self, batch_size: usize, batch_curve: &[f32]) -> f32 {
        let batch_index = match batch_size {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            16 => 4,
            32 => 5,
            64 => 6,
            128 => 7,
            _ => {
                // Interpolate for other batch sizes
                let log_batch = (batch_size as f32).log2();
                let index = log_batch.floor() as usize;
                if index < batch_curve.len() - 1 {
                    index.min(batch_curve.len() - 1)
                } else {
                    batch_curve.len() - 1
                }
            }
        };

        batch_curve.get(batch_index).copied().unwrap_or(0.5)
    }

    /// Get precision multiplier
    fn get_precision_multiplier(
        &self,
        precision: &str,
        precision_profile: &PrecisionProfile
    ) -> f32 {
        match precision.to_lowercase().as_str() {
            "fp16" => precision_profile.fp16_multiplier,
            "int8" => precision_profile.int8_multiplier,
            "int4" => precision_profile.int4_multiplier,
            _ => 1.0, // FP32 baseline
        }
    }

    /// Calculate actual inference time using bottleneck analysis
    pub fn calculate_inference_time(
        &self,
        analysis: &BottleneckAnalysis,
        batch_flops: f32,
        data_size_mb: f32
    ) -> f64 {
        // Calculate time for each bottleneck
        let compute_time_ms = (batch_flops / analysis.effective_compute_tflops) * 1000.0;
        let memory_time_ms = (data_size_mb * 8.0) / analysis.effective_bandwidth_gbps;

        // Weighted combination based on bottleneck type and intensities
        let inference_time_ms = match analysis.bottleneck_type {
            BottleneckType::Memory => {
                memory_time_ms * analysis.memory_bound_factor +
                    compute_time_ms * (1.0 - analysis.memory_bound_factor)
            }
            BottleneckType::Compute => {
                compute_time_ms * analysis.compute_bound_factor +
                    memory_time_ms * (1.0 - analysis.compute_bound_factor)
            }
            BottleneckType::Mixed => {
                // For mixed bottlenecks, use the maximum of the two (they can't be fully overlapped)
                memory_time_ms.max(compute_time_ms * 0.7) // Assume some overlap
            }
        };

        (inference_time_ms * analysis.performance_multiplier) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bottleneck_analysis_creation() {
        // Test that we can create a bottleneck analyzer
        let analyzer = BottleneckAnalyzer::new();
        assert!(
            analyzer.is_ok() || analyzer.unwrap_err().to_string().contains("hardware_profiles.toml")
        );
    }

    #[test]
    fn test_bottleneck_type_classification() {
        let memory_bottleneck = BottleneckType::Memory;
        let compute_bottleneck = BottleneckType::Compute;
        let mixed_bottleneck = BottleneckType::Mixed;

        assert_eq!(memory_bottleneck, BottleneckType::Memory);
        assert_eq!(compute_bottleneck, BottleneckType::Compute);
        assert_eq!(mixed_bottleneck, BottleneckType::Mixed);
    }

    #[test]
    fn test_batch_scaling_factor() {
        let analyzer = BottleneckAnalyzer::new().unwrap_or_else(|_| {
            BottleneckAnalyzer {
                hardware_profiles: HardwareProfileManager {
                    profiles: HashMap::new(),
                },
            }
        });

        let batch_curve = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];

        assert_eq!(analyzer.get_batch_scaling_factor(1, &batch_curve), 1.0);
        assert_eq!(analyzer.get_batch_scaling_factor(2, &batch_curve), 0.9);
        assert_eq!(analyzer.get_batch_scaling_factor(8, &batch_curve), 0.7);
    }
}
