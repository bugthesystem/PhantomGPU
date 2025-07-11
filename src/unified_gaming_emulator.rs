use crate::emulator::RustGPUEmu;
use crate::models::GamingWorkloadConfig;
use crate::gpu_config::GpuModel;
use crate::errors::PhantomGpuError;
use crate::gaming_benchmark_calibrator::GamingBenchmarkCalibrator;
use serde::{ Deserialize, Serialize };
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingPerformanceResult {
    pub avg_fps: f64,
    pub one_percent_low: f64,
    pub frame_time_ms: f64,
    pub frame_time_consistency: FrameTimeConsistency,
    pub gpu_utilization: f64,
    pub memory_utilization: f64,
    pub power_consumption: f64,
    pub temperature: f64,
    pub memory_usage_mb: f64,
    pub bottleneck_analysis: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FrameTimeConsistency {
    Excellent, // < 2ms variation
    Good, // < 5ms variation
    Acceptable, // < 10ms variation
    Poor, // > 10ms variation
}

/// Unified gaming emulator using the core RustGPUEmu
pub struct UnifiedGamingEmulator {
    emulator: RustGPUEmu,
    game_profiles: HashMap<String, GamingProfile>,
    calibrator: GamingBenchmarkCalibrator,
}

#[derive(Debug, Clone)]
struct GamingProfile {
    base_complexity: f64,
    rt_scaling_factor: f64,
    memory_intensity: f64,
    thermal_multiplier: f64,
}

impl UnifiedGamingEmulator {
    pub fn new(gpu_model: GpuModel) -> Self {
        let emulator = RustGPUEmu::new(gpu_model);
        let mut game_profiles = HashMap::new();

        // Initialize game profiles with realistic complexity data
        game_profiles.insert("Cyberpunk 2077".to_string(), GamingProfile {
            base_complexity: 0.9,
            rt_scaling_factor: 0.5,
            memory_intensity: 0.8,
            thermal_multiplier: 1.2,
        });

        game_profiles.insert("Valorant".to_string(), GamingProfile {
            base_complexity: 0.3,
            rt_scaling_factor: 1.0, // No RT support
            memory_intensity: 0.3,
            thermal_multiplier: 0.7,
        });

        game_profiles.insert("Apex Legends".to_string(), GamingProfile {
            base_complexity: 0.6,
            rt_scaling_factor: 0.7,
            memory_intensity: 0.6,
            thermal_multiplier: 1.0,
        });

        game_profiles.insert("Overwatch 2".to_string(), GamingProfile {
            base_complexity: 0.5,
            rt_scaling_factor: 1.0, // No RT support
            memory_intensity: 0.5,
            thermal_multiplier: 0.8,
        });

        game_profiles.insert("Fortnite".to_string(), GamingProfile {
            base_complexity: 0.5,
            rt_scaling_factor: 0.8,
            memory_intensity: 0.5,
            thermal_multiplier: 0.9,
        });

        game_profiles.insert("Call of Duty: Modern Warfare III".to_string(), GamingProfile {
            base_complexity: 0.7,
            rt_scaling_factor: 0.75,
            memory_intensity: 0.6,
            thermal_multiplier: 1.0,
        });

        game_profiles.insert("Hogwarts Legacy".to_string(), GamingProfile {
            base_complexity: 0.8,
            rt_scaling_factor: 0.6,
            memory_intensity: 0.9,
            thermal_multiplier: 1.1,
        });

        Self {
            emulator,
            game_profiles,
            calibrator: GamingBenchmarkCalibrator::new(),
        }
    }

    /// Predict gaming performance using core emulator
    pub async fn predict_gaming_performance(
        &mut self,
        workload: &GamingWorkloadConfig,
        ambient_temp: f64
    ) -> Result<GamingPerformanceResult, PhantomGpuError> {
        // Get game profile for complexity adjustments
        let game_profile = self.game_profiles
            .get(&workload.game_name)
            .cloned()
            .unwrap_or_else(|| GamingProfile {
                base_complexity: 0.7,
                rt_scaling_factor: 0.8,
                memory_intensity: 0.6,
                thermal_multiplier: 1.0,
            });

        // Apply game-specific complexity adjustments
        let mut adjusted_workload = workload.clone();
        adjusted_workload.scene_complexity *= game_profile.base_complexity;

        // Convert to ModelConfig for emulator
        let model_config = adjusted_workload.to_model_config();

        println!("🎮 Gaming Emulation: {}", workload.game_name);
        println!("   Resolution: {}x{}", workload.resolution.0, workload.resolution.1);
        println!("   FLOPS per frame: {:.2}B", model_config.flops_per_sample_g);
        println!("   Memory usage: {:.1}MB", model_config.parameters_m * 4.0);

        // Run emulation for resource analysis (NOT for frame timing)
        let _emulator_duration = self.emulator
            .emulate_forward(&model_config).await
            .map_err(|e|
                PhantomGpuError::ModelLoadError(format!("Gaming emulation failed: {}", e))
            )?;

        // Calculate realistic gaming performance based on GPU capabilities
        let gaming_result = self.calculate_realistic_gaming_performance(
            &model_config,
            workload,
            &game_profile,
            ambient_temp
        )?;

        Ok(gaming_result)
    }

    /// Calculate realistic gaming performance based on GPU capabilities and workload
    fn calculate_realistic_gaming_performance(
        &self,
        model_config: &crate::models::ModelConfig,
        workload: &GamingWorkloadConfig,
        game_profile: &GamingProfile,
        ambient_temp: f64
    ) -> Result<GamingPerformanceResult, PhantomGpuError> {
        // GPU specifications
        let gpu_tflops = self.emulator.gpu_model.compute_tflops as f64;
        let gpu_memory_bandwidth_gbps = self.emulator.gpu_model.memory_bandwidth_gbps as f64;
        let gpu_memory_gb = self.emulator.gpu_model.memory_gb as f64;

        // Workload requirements
        let flops_per_frame = (model_config.flops_per_sample_g as f64) * 1e9; // Convert to FLOPS
        let memory_per_frame_gb = ((model_config.parameters_m as f64) * 4.0) / 1000.0; // MB to GB

        // Apply game-specific optimization factors
        let optimization_factor = self.get_game_optimization_factor(&workload.game_name);
        let dlss_factor = self.get_dlss_performance_factor(&workload.dlss_mode);
        let fsr_factor = self.get_fsr_performance_factor(&workload.fsr_mode);

        // Calculate effective requirements
        let effective_flops = flops_per_frame / (optimization_factor * dlss_factor * fsr_factor);
        let effective_memory_bandwidth = memory_per_frame_gb / optimization_factor;

        // Calculate performance bottlenecks
        let compute_limited_fps = (gpu_tflops * 1e12) / effective_flops;
        let memory_limited_fps =
            (gpu_memory_bandwidth_gbps * 1000.0) / (effective_memory_bandwidth * 8.0 * 60.0); // Assume 60x memory reuse

        // Take the minimum (bottleneck)
        let theoretical_fps = compute_limited_fps.min(memory_limited_fps);

        // Apply realistic scaling factors
        let driver_efficiency = 0.85; // Driver overhead
        let thermal_throttling = self.get_thermal_throttling_factor(ambient_temp);
        let power_limit_factor = self.get_power_limit_factor(&workload.game_name);

        let realistic_fps =
            theoretical_fps * driver_efficiency * thermal_throttling * power_limit_factor;

        // 🎯 NEW: Use benchmark calibrator for much more accurate FPS predictions
        let avg_fps = self.calibrator.calibrate_gaming_performance(
            workload,
            &self.emulator.gpu_model,
            realistic_fps
        );

        println!("🎮 Gaming Performance Calculation:");
        println!("   Theoretical FPS: {:.1}", theoretical_fps);
        println!("   Realistic FPS: {:.1}", realistic_fps);
        println!("   Calibrated FPS: {:.1}", avg_fps);

        // Calculate derived metrics
        let frame_time_ms = 1000.0 / avg_fps;
        let one_percent_low = avg_fps * 0.78; // 78% of average is typical for 1% low

        // Calculate frame time consistency
        let frame_time_consistency = self.calculate_frame_time_consistency(
            frame_time_ms,
            workload.scene_complexity,
            game_profile
        );

        // Calculate GPU utilization based on performance vs target
        let target_frame_time_ms = 1000.0 / workload.target_fps;
        let gpu_utilization = if avg_fps >= workload.target_fps {
            (target_frame_time_ms / frame_time_ms).min(1.0)
        } else {
            0.95 // High utilization when struggling to meet target
        };

        // Get memory utilization from emulator
        let (memory_used_mb, _memory_total_mb, memory_utilization_percent) =
            self.emulator.get_memory_info();
        let memory_utilization = (memory_utilization_percent / 100.0).max(
            memory_per_frame_gb / gpu_memory_gb
        );

        // Calculate power consumption
        let power_consumption = self.calculate_gaming_power_consumption(
            gpu_utilization,
            workload.scene_complexity,
            game_profile
        );

        // Calculate temperature
        let temperature = self.calculate_gaming_temperature(
            power_consumption,
            ambient_temp,
            game_profile
        );

        // Determine bottleneck analysis
        let bottleneck_analysis = if compute_limited_fps < memory_limited_fps {
            Some(
                format!(
                    "Compute-bound | {:.1}% GPU utilization | Effective compute: {:.1} TFLOPS",
                    gpu_utilization * 100.0,
                    gpu_tflops * gpu_utilization
                )
            )
        } else {
            Some(
                format!(
                    "Memory-bound | {:.1}% memory utilization | Effective BW: {:.0} GB/s",
                    memory_utilization * 100.0,
                    gpu_memory_bandwidth_gbps * memory_utilization
                )
            )
        };

        Ok(GamingPerformanceResult {
            avg_fps,
            one_percent_low,
            frame_time_ms,
            frame_time_consistency,
            gpu_utilization,
            memory_utilization,
            power_consumption,
            temperature,
            memory_usage_mb: memory_used_mb,
            bottleneck_analysis,
        })
    }

    fn get_game_optimization_factor(&self, game_name: &str) -> f64 {
        match game_name {
            "Valorant" => 3.5, // Highly optimized competitive game
            "Overwatch 2" => 2.8, // Well-optimized esports title
            "Fortnite" => 2.5, // Good optimization across platforms
            "Apex Legends" => 2.2, // Decent optimization
            "Call of Duty: Modern Warfare III" => 1.8, // Modern but demanding
            "Hogwarts Legacy" => 1.4, // Poor optimization at launch
            "Cyberpunk 2077" => 1.2, // Notoriously demanding
            _ => 1.8, // Default assumption
        }
    }

    fn get_dlss_performance_factor(&self, dlss_mode: &crate::models::DLSSMode) -> f64 {
        match dlss_mode {
            crate::models::DLSSMode::Off => 1.0,
            crate::models::DLSSMode::Quality => 1.5,
            crate::models::DLSSMode::Balanced => 1.7,
            crate::models::DLSSMode::Performance => 2.0,
            crate::models::DLSSMode::UltraPerformance => 2.5,
        }
    }

    fn get_fsr_performance_factor(&self, fsr_mode: &crate::models::FSRMode) -> f64 {
        match fsr_mode {
            crate::models::FSRMode::Off => 1.0,
            crate::models::FSRMode::UltraQuality => 1.2,
            crate::models::FSRMode::Quality => 1.3,
            crate::models::FSRMode::Balanced => 1.5,
            crate::models::FSRMode::Performance => 1.8,
        }
    }

    fn get_thermal_throttling_factor(&self, ambient_temp: f64) -> f64 {
        if ambient_temp < 20.0 {
            1.0
        } else if ambient_temp < 30.0 {
            0.98
        } else if ambient_temp < 35.0 {
            0.95
        } else {
            0.9 // Significant throttling in hot environments
        }
    }

    fn get_power_limit_factor(&self, game_name: &str) -> f64 {
        match game_name {
            "Valorant" | "Overwatch 2" => 1.0, // Not power limited
            "Fortnite" | "Apex Legends" => 0.98,
            "Call of Duty: Modern Warfare III" => 0.95,
            "Hogwarts Legacy" | "Cyberpunk 2077" => 0.92, // Power hungry games
            _ => 0.96,
        }
    }

    fn clamp_fps_to_realistic_range(
        &self,
        calculated_fps: f64,
        game_name: &str,
        resolution: (u32, u32)
    ) -> f64 {
        let gpu_tier = self.get_gpu_tier();

        // Define realistic FPS ranges based on GPU tier, game, and resolution
        let (min_fps, max_fps) = match (gpu_tier, game_name, resolution) {
            // RTX 4090 ranges
            (GPUTier::Flagship, "Valorant", (1920, 1080)) => (300.0, 500.0),
            (GPUTier::Flagship, "Valorant", (2560, 1440)) => (250.0, 400.0),
            (GPUTier::Flagship, "Cyberpunk 2077", (2560, 1440)) => (60.0, 120.0),
            (GPUTier::Flagship, "Cyberpunk 2077", (3840, 2160)) => (40.0, 80.0),
            (GPUTier::Flagship, "Fortnite", (1920, 1080)) => (150.0, 300.0),
            (GPUTier::Flagship, "Apex Legends", (2560, 1440)) => (120.0, 200.0),
            (GPUTier::Flagship, "Overwatch 2", (2560, 1440)) => (150.0, 250.0),

            // RTX 4080 ranges
            (GPUTier::HighEnd, "Valorant", (1920, 1080)) => (250.0, 400.0),
            (GPUTier::HighEnd, "Cyberpunk 2077", (2560, 1440)) => (50.0, 90.0),
            (GPUTier::HighEnd, "Overwatch 2", (2560, 1440)) => (130.0, 200.0),

            // RTX 5090 ranges (extrapolated)
            (GPUTier::NextGen, "Valorant", (1920, 1080)) => (350.0, 600.0),
            (GPUTier::NextGen, "Cyberpunk 2077", (3840, 2160)) => (60.0, 100.0),

            // Default ranges
            (GPUTier::Flagship, _, (1920, 1080)) => (80.0, 300.0),
            (GPUTier::Flagship, _, (2560, 1440)) => (60.0, 200.0),
            (GPUTier::Flagship, _, (3840, 2160)) => (30.0, 120.0),
            (GPUTier::HighEnd, _, (1920, 1080)) => (60.0, 250.0),
            (GPUTier::HighEnd, _, (2560, 1440)) => (45.0, 150.0),
            (GPUTier::HighEnd, _, (3840, 2160)) => (25.0, 80.0),
            (GPUTier::NextGen, _, (1920, 1080)) => (100.0, 400.0),
            (GPUTier::NextGen, _, (2560, 1440)) => (80.0, 300.0),
            (GPUTier::NextGen, _, (3840, 2160)) => (50.0, 150.0),
            _ => (30.0, 200.0),
        };

        calculated_fps.max(min_fps).min(max_fps)
    }

    fn get_gpu_tier(&self) -> GPUTier {
        match self.emulator.gpu_model.name.as_str() {
            "RTX 4090" => GPUTier::Flagship,
            "RTX 4080" => GPUTier::HighEnd,
            "RTX 5090" => GPUTier::NextGen,
            _ => {
                if self.emulator.gpu_model.compute_tflops > 80.0 {
                    GPUTier::NextGen
                } else if self.emulator.gpu_model.compute_tflops > 40.0 {
                    GPUTier::Flagship
                } else {
                    GPUTier::HighEnd
                }
            }
        }
    }

    fn calculate_frame_time_consistency(
        &self,
        frame_time_ms: f64,
        scene_complexity: f64,
        game_profile: &GamingProfile
    ) -> FrameTimeConsistency {
        // Calculate variance based on scene complexity and game characteristics
        let base_variance = scene_complexity * 0.3; // 30% max variance
        let game_variance = game_profile.base_complexity * 0.2; // Game-specific variance

        let total_variance = base_variance + game_variance;
        let frame_time_variation = frame_time_ms * total_variance;

        match frame_time_variation {
            v if v < 2.0 => FrameTimeConsistency::Excellent,
            v if v < 5.0 => FrameTimeConsistency::Good,
            v if v < 10.0 => FrameTimeConsistency::Acceptable,
            _ => FrameTimeConsistency::Poor,
        }
    }

    fn calculate_gaming_power_consumption(
        &self,
        gpu_utilization: f64,
        scene_complexity: f64,
        game_profile: &GamingProfile
    ) -> f64 {
        let gpu_tdp = (self.emulator.gpu_model.compute_tflops as f64) * 15.0; // Estimate: 15W per TFLOP
        let base_power = gpu_tdp * 0.3; // 30% base power
        let rendering_power = gpu_tdp * 0.6 * scene_complexity; // Up to 60% for rendering
        let utilization_scaling = gpu_utilization.powf(0.8); // Sub-linear scaling

        let total_power =
            (base_power + rendering_power * utilization_scaling) * game_profile.thermal_multiplier;
        total_power
    }

    fn calculate_gaming_temperature(
        &self,
        power_consumption: f64,
        ambient_temp: f64,
        game_profile: &GamingProfile
    ) -> f64 {
        let thermal_resistance = 0.15; // K/W typical for gaming GPUs
        let temperature_rise =
            power_consumption * thermal_resistance * game_profile.thermal_multiplier;
        ambient_temp + temperature_rise
    }

    /// Get emulator memory information
    pub fn get_memory_info(&self) -> (f64, f64, f64) {
        self.emulator.get_memory_info()
    }

    /// Get emulator stats
    pub fn get_stats(&self) -> String {
        self.emulator.get_stats()
    }

    /// Reset emulator memory
    pub fn reset_memory(&mut self) {
        self.emulator.reset_memory()
    }

    /// Get supported games
    pub fn get_supported_games(&self) -> Vec<String> {
        self.game_profiles.keys().cloned().collect()
    }

    /// Predict frame generation performance
    pub async fn predict_frame_generation(
        &mut self,
        workload: &GamingWorkloadConfig,
        ambient_temp: f64
    ) -> Result<FrameGenerationResult, PhantomGpuError> {
        // Check if GPU supports frame generation
        let gpu_arch = self.emulator.gpu_model.architecture
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("Unknown");

        let supports_frame_gen = matches!(gpu_arch, "Ada Lovelace" | "Blackwell");

        if !supports_frame_gen {
            return Ok(FrameGenerationResult {
                supported: false,
                base_fps: 0.0,
                generated_fps: 0.0,
                frame_generation_ratio: 1.0,
                latency_penalty_ms: 0.0,
                quality_impact: FrameGenerationQuality::NotSupported,
            });
        }

        // Calculate frame generation scaling
        let generation_ratio = match gpu_arch {
            "Blackwell" => 2.2, // Enhanced frame generation
            "Ada Lovelace" => 1.8, // DLSS 3 Frame Generation
            _ => 1.0,
        };

        // Apply game compatibility factor
        let game_compatibility = self.get_frame_generation_compatibility(&workload.game_name);
        let effective_ratio = generation_ratio * game_compatibility;

        // Get base performance
        let base_result = self.predict_gaming_performance(workload, ambient_temp).await?;
        let generated_fps = base_result.avg_fps * effective_ratio;

        // Calculate latency penalty
        let latency_penalty = self.calculate_frame_generation_latency_penalty(
            base_result.avg_fps,
            effective_ratio,
            &workload.dlss_mode
        );

        // Assess quality impact
        let quality_impact = self.assess_frame_generation_quality(
            base_result.avg_fps,
            effective_ratio,
            workload.scene_complexity
        );

        Ok(FrameGenerationResult {
            supported: true,
            base_fps: base_result.avg_fps,
            generated_fps,
            frame_generation_ratio: effective_ratio,
            latency_penalty_ms: latency_penalty,
            quality_impact,
        })
    }

    fn get_frame_generation_compatibility(&self, game_name: &str) -> f64 {
        match game_name {
            "Cyberpunk 2077" => 0.9,
            "Call of Duty: Modern Warfare III" => 0.8,
            "Hogwarts Legacy" => 0.85,
            "Fortnite" => 0.7,
            "Apex Legends" => 0.6,
            "Overwatch 2" => 0.5,
            "Valorant" => 0.0, // Competitive integrity
            _ => 0.75,
        }
    }

    fn calculate_frame_generation_latency_penalty(
        &self,
        base_fps: f64,
        _scaling: f64,
        dlss_mode: &crate::models::DLSSMode
    ) -> f64 {
        let base_latency_penalty = match dlss_mode {
            crate::models::DLSSMode::Performance => 12.0,
            crate::models::DLSSMode::Balanced => 8.0,
            crate::models::DLSSMode::Quality => 6.0,
            crate::models::DLSSMode::Off => 10.0,
            crate::models::DLSSMode::UltraPerformance => 15.0,
        };

        let fps_factor = (60.0 / base_fps.max(30.0)).min(2.0);
        base_latency_penalty * fps_factor
    }

    fn assess_frame_generation_quality(
        &self,
        base_fps: f64,
        scaling: f64,
        scene_complexity: f64
    ) -> FrameGenerationQuality {
        let quality_score = base_fps * (1.0 - scene_complexity * 0.3) * (scaling - 1.0);

        match quality_score {
            score if score > 60.0 => FrameGenerationQuality::Excellent,
            score if score > 40.0 => FrameGenerationQuality::Good,
            score if score > 20.0 => FrameGenerationQuality::Acceptable,
            _ => FrameGenerationQuality::Poor,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameGenerationResult {
    pub supported: bool,
    pub base_fps: f64,
    pub generated_fps: f64,
    pub frame_generation_ratio: f64,
    pub latency_penalty_ms: f64,
    pub quality_impact: FrameGenerationQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FrameGenerationQuality {
    NotSupported,
    Poor,
    Acceptable,
    Good,
    Excellent,
}

#[derive(Debug, Clone)]
enum GPUTier {
    HighEnd, // RTX 4080 class
    Flagship, // RTX 4090 class
    NextGen, // RTX 5090 class
}

/// Helper functions for common gaming scenarios
impl UnifiedGamingEmulator {
    pub fn create_competitive_workload(target_fps: f64) -> GamingWorkloadConfig {
        GamingWorkloadConfig {
            game_name: "Valorant".to_string(),
            resolution: (1920, 1080),
            ray_tracing: false,
            dlss_mode: crate::models::DLSSMode::Off,
            fsr_mode: crate::models::FSRMode::Off,
            graphics_settings: crate::models::GraphicsSettings {
                texture_quality: crate::models::Quality::High,
                shadow_quality: crate::models::Quality::Low,
                anti_aliasing: crate::models::AntiAliasing::FXAA,
                anisotropic_filtering: 8,
                variable_rate_shading: false,
                mesh_shaders: false,
            },
            target_fps,
            scene_complexity: 0.3,
        }
    }

    pub fn create_visual_quality_workload(resolution: (u32, u32)) -> GamingWorkloadConfig {
        GamingWorkloadConfig {
            game_name: "Cyberpunk 2077".to_string(),
            resolution,
            ray_tracing: true,
            dlss_mode: crate::models::DLSSMode::Quality,
            fsr_mode: crate::models::FSRMode::Off,
            graphics_settings: crate::models::GraphicsSettings {
                texture_quality: crate::models::Quality::Ultra,
                shadow_quality: crate::models::Quality::High,
                anti_aliasing: crate::models::AntiAliasing::TAA,
                anisotropic_filtering: 16,
                variable_rate_shading: true,
                mesh_shaders: true,
            },
            target_fps: 60.0,
            scene_complexity: 0.8,
        }
    }
}
