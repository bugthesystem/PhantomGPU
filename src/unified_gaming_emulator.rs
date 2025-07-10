use crate::emulator::RustGPUEmu;
use crate::models::GamingWorkloadConfig;
use crate::gpu_config::GpuModel;
use crate::errors::PhantomGpuError;
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

        // Run emulation for frame rendering
        let frame_duration = self.emulator
            .emulate_forward(&model_config).await
            .map_err(|e|
                PhantomGpuError::ModelLoadError(format!("Gaming emulation failed: {}", e))
            )?;

        // Convert frame time to FPS
        let frame_time_ms = frame_duration.as_millis() as f64;
        let avg_fps = if frame_time_ms > 0.0 { 1000.0 / frame_time_ms } else { 0.0 };

        // Calculate 1% low (typically 75-85% of average)
        let one_percent_low = avg_fps * 0.78;

        // Calculate frame time consistency based on workload complexity
        let frame_time_consistency = self.calculate_frame_time_consistency(
            frame_time_ms,
            workload.scene_complexity,
            &game_profile
        );

        // Get resource utilization from emulator
        let (memory_used_mb, _memory_total_mb, memory_utilization_percent) =
            self.emulator.get_memory_info();
        let memory_utilization = memory_utilization_percent / 100.0;

        // Calculate GPU utilization based on target FPS vs achieved FPS
        let target_frame_time_ms = 1000.0 / workload.target_fps;
        let gpu_utilization = (target_frame_time_ms / frame_time_ms).min(1.0);

        // Calculate power consumption using emulator's thermal system
        let power_consumption = self.calculate_gaming_power_consumption(
            gpu_utilization,
            workload.scene_complexity,
            &game_profile
        );

        // Calculate temperature
        let temperature = self.calculate_gaming_temperature(
            power_consumption,
            ambient_temp,
            &game_profile
        );

        // Get bottleneck analysis if available
        let bottleneck_analysis = self.emulator
            .get_or_create_profile(&model_config)
            .performance_analysis.clone();

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
        let gpu_tdp = self.emulator.gpu_model.compute_tflops * 15.0; // Estimate: 15W per TFLOP
        let base_power = gpu_tdp * 0.3; // 30% base power
        let rendering_power = gpu_tdp * 0.6 * (scene_complexity as f32); // Up to 60% for rendering
        let utilization_scaling = (gpu_utilization as f32).powf(0.8); // Sub-linear scaling

        let total_power =
            (base_power + rendering_power * utilization_scaling) *
            (game_profile.thermal_multiplier as f32);
        total_power as f64
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
