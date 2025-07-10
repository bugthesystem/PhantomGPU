//! Model configurations and emulation profiles

use crate::bottleneck_analysis::{ BottleneckAnalyzer, BottleneckType };
use serde::{ Deserialize, Serialize };
use std::collections::HashMap;
use crate::gpu_config::GpuModel;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub batch_size: usize,
    pub input_shape: Vec<usize>,
    pub parameters_m: f32, // Parameters in millions
    pub flops_per_sample_g: f32, // FLOPS per sample in billions
    pub model_type: String, // Model type for bottleneck analysis
    pub precision: String, // Precision (fp32, fp16, int8, int4)
}

impl ModelConfig {
    pub fn resnet50(batch_size: usize) -> Self {
        Self {
            name: "ResNet50".to_string(),
            batch_size,
            input_shape: vec![3, 224, 224],
            parameters_m: 25.6,
            flops_per_sample_g: 4.1,
            model_type: "cnn".to_string(),
            precision: "fp32".to_string(),
        }
    }

    pub fn alexnet(batch_size: usize) -> Self {
        Self {
            name: "AlexNet".to_string(),
            batch_size,
            input_shape: vec![3, 224, 224],
            parameters_m: 61.0,
            flops_per_sample_g: 0.7,
            model_type: "cnn".to_string(),
            precision: "fp32".to_string(),
        }
    }

    /// Set model precision
    pub fn with_precision(mut self, precision: &str) -> Self {
        self.precision = precision.to_string();
        self
    }

    /// Set model type for bottleneck analysis
    pub fn with_model_type(mut self, model_type: &str) -> Self {
        self.model_type = model_type.to_string();
        self
    }
}

#[derive(Debug, Clone)]
pub struct EmulationProfile {
    pub forward_time_ms: f64,
    pub backward_time_ms: f64,
    pub data_transfer_time_ms: f64,
    pub memory_usage_mb: f64,
    pub preprocessing_time_ms: f64,
    pub bottleneck_type: Option<String>,
    pub performance_analysis: Option<String>,
}

impl EmulationProfile {
    /// Estimate profile based on model characteristics and GPU specs using bottleneck analysis
    pub fn estimate(model: &ModelConfig, gpu: &crate::gpu_config::GpuModel) -> Self {
        // Try to use bottleneck-aware analysis first
        if let Ok(analyzer) = BottleneckAnalyzer::new() {
            Self::estimate_with_bottleneck_analysis(model, gpu, &analyzer)
        } else {
            // Fallback to simple FLOPS-based calculation with warning
            println!("⚠️  Bottleneck analysis unavailable, using simplified FLOPS model");
            Self::estimate_fallback(model, gpu)
        }
    }

    /// Advanced bottleneck-aware estimation
    fn estimate_with_bottleneck_analysis(
        model: &ModelConfig,
        gpu: &crate::gpu_config::GpuModel,
        analyzer: &BottleneckAnalyzer
    ) -> Self {
        let batch_flops = model.flops_per_sample_g * (model.batch_size as f32);

        // Perform bottleneck analysis
        let analysis = analyzer
            .analyze_bottleneck(
                &gpu.name,
                &model.model_type,
                model.batch_size,
                model.flops_per_sample_g,
                &model.precision
            )
            .unwrap_or_else(|_| {
                // Fallback analysis if hardware profile not found
                crate::bottleneck_analysis::BottleneckAnalysis {
                    bottleneck_type: BottleneckType::Mixed,
                    memory_bound_factor: 0.5,
                    compute_bound_factor: 0.5,
                    effective_bandwidth_gbps: gpu.memory_bandwidth_gbps * 0.7,
                    effective_compute_tflops: gpu.compute_tflops * 0.8,
                    performance_multiplier: 1.0,
                }
            });

        // Calculate data size for memory transfer estimation
        let input_size_mb =
            ((model.batch_size as f32) *
                (model.input_shape.iter().product::<usize>() as f32) *
                4.0) /
            (1024.0 * 1024.0);

        // Use bottleneck-aware calculation for forward time
        let forward_time_ms = analyzer.calculate_inference_time(
            &analysis,
            batch_flops,
            input_size_mb
        );

        // Backward pass estimation (varies by model type and bottleneck)
        let backward_multiplier = match analysis.bottleneck_type {
            BottleneckType::Memory => 1.8, // Memory-bound models have lower backward multiplier
            BottleneckType::Compute => 2.2, // Compute-bound models have higher backward multiplier
            BottleneckType::Mixed => 2.0, // Mixed models use average multiplier
        };
        let backward_time_ms = forward_time_ms * backward_multiplier;

        // Data transfer time using effective bandwidth
        let data_transfer_time_ms = ((input_size_mb / analysis.effective_bandwidth_gbps) *
            8.0 *
            1000.0) as f64;

        // Memory usage estimation (consider precision)
        let precision_factor = match model.precision.as_str() {
            "fp16" => 2.0,
            "int8" => 1.0,
            "int4" => 0.5,
            _ => 4.0, // fp32
        };
        let memory_usage_mb = (model.parameters_m * precision_factor + // Parameters
            input_size_mb * 2.0) as f64; // Input + activations

        // Preprocessing time (varies by model complexity and bottleneck)
        let preprocessing_multiplier = match model.model_type.as_str() {
            "cnn" => 0.05, // CNNs have minimal preprocessing
            "transformer" => 0.15, // Transformers have more preprocessing
            "rnn" => 0.1, // RNNs have moderate preprocessing
            _ => 0.1,
        };
        let preprocessing_time_ms = forward_time_ms * preprocessing_multiplier;

        // Performance analysis summary
        let bottleneck_type_str = match analysis.bottleneck_type {
            BottleneckType::Memory => "Memory-bound",
            BottleneckType::Compute => "Compute-bound",
            BottleneckType::Mixed => "Mixed bottlenecks",
        };

        let performance_analysis = format!(
            "{} | {:.1}% memory intensity | Effective BW: {:.0} GB/s | Effective compute: {:.1} TFLOPS",
            bottleneck_type_str,
            analysis.memory_bound_factor * 100.0,
            analysis.effective_bandwidth_gbps,
            analysis.effective_compute_tflops
        );

        Self {
            forward_time_ms,
            backward_time_ms,
            data_transfer_time_ms,
            memory_usage_mb,
            preprocessing_time_ms,
            bottleneck_type: Some(bottleneck_type_str.to_string()),
            performance_analysis: Some(performance_analysis),
        }
    }

    /// Fallback to simple FLOPS-based calculation (for compatibility)
    fn estimate_fallback(model: &ModelConfig, gpu: &crate::gpu_config::GpuModel) -> Self {
        let batch_flops = model.flops_per_sample_g * (model.batch_size as f32);
        let forward_time_ms = ((batch_flops / gpu.compute_tflops) * 1000.0) as f64;
        let backward_time_ms = forward_time_ms * 2.0; // Backward pass ~2x forward

        // Data transfer estimation (input tensor size)
        let input_size_mb =
            ((model.batch_size as f32) *
                (model.input_shape.iter().product::<usize>() as f32) *
                4.0) /
            (1024.0 * 1024.0);
        let data_transfer_time_ms = ((input_size_mb / gpu.memory_bandwidth_gbps) *
            8.0 *
            1000.0) as f64;

        // Memory usage estimation
        let memory_usage_mb = (model.parameters_m * 4.0 + // Parameters (fp32)
            input_size_mb * 2.0) as f64; // Input + activations

        // Preprocessing (varies by model complexity)
        let preprocessing_time_ms = forward_time_ms * 0.1; // ~10% of compute time

        Self {
            forward_time_ms,
            backward_time_ms,
            data_transfer_time_ms,
            memory_usage_mb,
            preprocessing_time_ms,
            bottleneck_type: Some("Compute-bound (simplified)".to_string()),
            performance_analysis: Some("Using simplified FLOPS-based calculation".to_string()),
        }
    }
}

// Add gaming workload support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingWorkloadConfig {
    pub game_name: String,
    pub resolution: (u32, u32),
    pub ray_tracing: bool,
    pub dlss_mode: DLSSMode,
    pub fsr_mode: FSRMode,
    pub graphics_settings: GraphicsSettings,
    pub target_fps: f64,
    pub scene_complexity: f64, // 0.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DLSSMode {
    Off,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FSRMode {
    Off,
    UltraQuality,
    Quality,
    Balanced,
    Performance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicsSettings {
    pub texture_quality: Quality,
    pub shadow_quality: Quality,
    pub anti_aliasing: AntiAliasing,
    pub anisotropic_filtering: u8,
    pub variable_rate_shading: bool,
    pub mesh_shaders: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Quality {
    Low,
    Medium,
    High,
    Ultra,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AntiAliasing {
    Off,
    FXAA,
    MSAA2x,
    MSAA4x,
    MSAA8x,
    TAA,
}

impl GamingWorkloadConfig {
    /// Create Cyberpunk 2077 4K RT workload
    pub fn cyberpunk_4k_rt() -> Self {
        Self {
            game_name: "Cyberpunk 2077".to_string(),
            resolution: (3840, 2160),
            ray_tracing: true,
            dlss_mode: DLSSMode::Quality,
            fsr_mode: FSRMode::Off,
            graphics_settings: GraphicsSettings {
                texture_quality: Quality::Ultra,
                shadow_quality: Quality::High,
                anti_aliasing: AntiAliasing::TAA,
                anisotropic_filtering: 16,
                variable_rate_shading: true,
                mesh_shaders: true,
            },
            target_fps: 60.0,
            scene_complexity: 0.8,
        }
    }

    /// Create competitive gaming workload (Valorant)
    pub fn valorant_competitive() -> Self {
        Self {
            game_name: "Valorant".to_string(),
            resolution: (1920, 1080),
            ray_tracing: false,
            dlss_mode: DLSSMode::Off,
            fsr_mode: FSRMode::Off,
            graphics_settings: GraphicsSettings {
                texture_quality: Quality::High,
                shadow_quality: Quality::Low,
                anti_aliasing: AntiAliasing::FXAA,
                anisotropic_filtering: 8,
                variable_rate_shading: false,
                mesh_shaders: false,
            },
            target_fps: 240.0,
            scene_complexity: 0.3,
        }
    }

    /// Convert gaming workload to ModelConfig for emulator
    pub fn to_model_config(&self) -> ModelConfig {
        let _pixel_count = (self.resolution.0 * self.resolution.1) as f64;

        // Calculate FLOPS based on graphics pipeline
        let base_flops_per_frame = self.calculate_base_flops();
        let rt_flops = if self.ray_tracing { self.calculate_ray_tracing_flops() } else { 0.0 };
        let post_processing_flops = self.calculate_post_processing_flops();

        let total_flops_per_frame = base_flops_per_frame + rt_flops + post_processing_flops;

        // Target batch size = frames in flight (typically 2-3)
        let frames_in_flight = if self.target_fps >= 120.0 { 2 } else { 3 };

        // Convert to FLOPS per sample in billions
        let flops_per_sample_g = (total_flops_per_frame /
            (frames_in_flight as f64) /
            1_000_000_000.0) as f32;

        // Estimate "parameters" based on GPU state complexity
        let memory_usage_mb = self.calculate_memory_usage();
        let parameters_m = (memory_usage_mb / 4.0) as f32; // Assuming 4 bytes per parameter

        ModelConfig {
            name: format!("Gaming_{}", self.game_name.replace(" ", "_")),
            batch_size: frames_in_flight,
            input_shape: vec![self.resolution.0 as usize, self.resolution.1 as usize, 4], // RGBA
            parameters_m,
            flops_per_sample_g,
            model_type: "gaming".to_string(),
            precision: "fp32".to_string(), // Gaming typically uses FP32
        }
    }

    /// Calculate base rasterization FLOPS
    fn calculate_base_flops(&self) -> f64 {
        let pixel_count = (self.resolution.0 * self.resolution.1) as f64;
        let scene_complexity = self.scene_complexity;

        // Base shader calculations: vertex + fragment shaders (scaled for realism)
        let vertex_flops = pixel_count * 10.0; // ~10 FLOPS per vertex shader (reduced from 100)
        let fragment_flops = pixel_count * 30.0 * scene_complexity; // ~30 FLOPS per fragment shader (reduced from 300)

        // Graphics settings impact
        let texture_multiplier = match self.graphics_settings.texture_quality {
            Quality::Ultra => 1.5,
            Quality::High => 1.2,
            Quality::Medium => 1.0,
            Quality::Low => 0.8,
        };

        let shadow_multiplier = match self.graphics_settings.shadow_quality {
            Quality::Ultra => 2.0,
            Quality::High => 1.5,
            Quality::Medium => 1.0,
            Quality::Low => 0.5,
        };

        let aa_multiplier = match self.graphics_settings.anti_aliasing {
            AntiAliasing::MSAA8x => 8.0,
            AntiAliasing::MSAA4x => 4.0,
            AntiAliasing::MSAA2x => 2.0,
            AntiAliasing::TAA => 1.3,
            AntiAliasing::FXAA => 1.1,
            AntiAliasing::Off => 1.0,
        };

        (vertex_flops + fragment_flops) * texture_multiplier * shadow_multiplier * aa_multiplier
    }

    /// Calculate ray tracing FLOPS
    fn calculate_ray_tracing_flops(&self) -> f64 {
        if !self.ray_tracing {
            return 0.0;
        }

        let pixel_count = (self.resolution.0 * self.resolution.1) as f64;
        let scene_complexity = self.scene_complexity;

        // Ray tracing is expensive but scaled for realism
        // ~200-500 FLOPS per ray, 1-2 rays per pixel (reduced from 1000-5000 FLOPS)
        let rays_per_pixel = 1.5 * scene_complexity; // 1-1.5 rays per pixel
        let flops_per_ray = 300.0; // Average ray tracing FLOPS (reduced from 2000)

        pixel_count * rays_per_pixel * flops_per_ray
    }

    /// Calculate post-processing FLOPS (DLSS, FSR, etc.)
    fn calculate_post_processing_flops(&self) -> f64 {
        let pixel_count = (self.resolution.0 * self.resolution.1) as f64;
        let mut post_flops = 0.0;

        // DLSS neural network inference (scaled for realism)
        match self.dlss_mode {
            DLSSMode::Off => {}
            DLSSMode::Quality => {
                post_flops += pixel_count * 5.0;
            } // 5 FLOPS per pixel (reduced from 50)
            DLSSMode::Balanced => {
                post_flops += pixel_count * 4.5;
            }
            DLSSMode::Performance => {
                post_flops += pixel_count * 4.0;
            }
            DLSSMode::UltraPerformance => {
                post_flops += pixel_count * 3.5;
            }
        }

        // FSR upscaling
        match self.fsr_mode {
            FSRMode::Off => {}
            _ => {
                post_flops += pixel_count * 2.0;
            } // 2 FLOPS per pixel for FSR (reduced from 20)
        }

        post_flops
    }

    /// Calculate memory usage for gaming workload
    fn calculate_memory_usage(&self) -> f64 {
        let pixel_count = (self.resolution.0 * self.resolution.1) as f64;
        let scene_complexity = self.scene_complexity;

        // Frame buffers (color, depth, stencil)
        let frame_buffer_mb = (pixel_count * 4.0 * 4.0) / (1024.0 * 1024.0); // 4 bytes per pixel, 4 buffers

        // Textures - depends on quality and scene complexity
        let texture_mb = match self.graphics_settings.texture_quality {
            Quality::Ultra => 8000.0 * scene_complexity,
            Quality::High => 6000.0 * scene_complexity,
            Quality::Medium => 4000.0 * scene_complexity,
            Quality::Low => 2000.0 * scene_complexity,
        };

        // Ray tracing data structures
        let rt_mb = if self.ray_tracing {
            2000.0 * scene_complexity // BVH + ray data
        } else {
            0.0
        };

        // DLSS model weights
        let dlss_mb = if matches!(self.dlss_mode, DLSSMode::Off) {
            0.0
        } else {
            150.0 // DLSS model size
        };

        frame_buffer_mb + texture_mb + rt_mb + dlss_mb
    }
}
