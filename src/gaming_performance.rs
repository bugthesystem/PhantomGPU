use crate::gpu_config::GpuModel;
use crate::errors::PhantomGpuError;
use serde::{ Deserialize, Serialize };
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingPerformance {
    pub avg_fps: f64,
    pub one_percent_low: f64,
    pub frame_time_ms: f64,
    pub frame_time_consistency: FrameTimeConsistency,
    pub gpu_utilization: f64,
    pub memory_utilization: f64,
    pub power_consumption: f64,
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FrameTimeConsistency {
    Good, // < 5ms variation
    Acceptable, // < 10ms variation
    Poor, // > 10ms variation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DLSSMode {
    Off,
    Quality, // 1800p→4K
    Balanced, // 1440p→4K
    Performance, // 1080p→4K
    UltraPerformance, // 720p→4K
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FSRMode {
    Off,
    UltraQuality, // 1.3x upscaling
    Quality, // 1.5x upscaling
    Balanced, // 1.7x upscaling
    Performance, // 2.0x upscaling
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingWorkload {
    pub game_name: String,
    pub resolution: (u32, u32),
    pub ray_tracing: bool,
    pub dlss_mode: DLSSMode,
    pub fsr_mode: FSRMode,
    pub target_fps: f64,
    pub scene_complexity: f64, // 0.0 to 1.0
    pub graphics_settings: GraphicsSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicsSettings {
    pub texture_quality: Quality,
    pub shadow_quality: Quality,
    pub anti_aliasing: AntiAliasing,
    pub anisotropic_filtering: u8, // 1x, 2x, 4x, 8x, 16x
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingGPUFeatures {
    pub rops: u32, // Raster Operations units
    pub tmus: u32, // Texture Mapping Units
    pub rt_cores: u32, // Ray Tracing cores
    pub rt_generation: u8, // RT core generation (1-4)
    pub tensor_cores: u32, // For DLSS
    pub tensor_generation: u8, // Tensor core generation
    pub vrs_support: bool, // Variable Rate Shading
    pub mesh_shader_support: bool,
    pub av1_encode: bool, // For streaming
    pub av1_decode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameGenerationResult {
    pub supported: bool,
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

// TOML configuration structures
#[derive(Debug, Deserialize)]
struct GameProfilesConfig {
    gpu_features: HashMap<String, GamingGPUFeatures>,
    games: HashMap<String, GameConfigEntry>,
    upscaling: UpscalingConfig,
    frame_generation: FrameGenerationConfig,
    resolution_scaling: HashMap<String, f64>,
    graphics_settings: GraphicsSettingsConfig,
    validation: Option<HashMap<String, HashMap<String, ValidationEntry>>>,
}

#[derive(Debug, Deserialize)]
struct GameConfigEntry {
    description: String,
    rt_performance_impact: f64,
    memory_intensity: f64,
    compute_intensity: f64,
    texture_streaming: f64,
    scene_complexity_variance: f64,
    base_performance: HashMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct UpscalingConfig {
    dlss: HashMap<String, f64>,
    fsr: HashMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct FrameGenerationConfig {
    compatibility: HashMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct GraphicsSettingsConfig {
    texture_quality: HashMap<String, f64>,
    shadow_quality: HashMap<String, f64>,
    anti_aliasing: HashMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct ValidationEntry {
    resolution: String,
    ray_tracing: bool,
    dlss: String,
    expected_fps: f64,
    tolerance: f64,
    source: String,
}

pub struct GamingPerformanceEngine {
    gpu_features: HashMap<String, GamingGPUFeatures>,
    game_profiles: HashMap<String, GameProfile>,
    upscaling_config: UpscalingConfig,
    frame_generation_config: FrameGenerationConfig,
    resolution_scaling: HashMap<String, f64>,
    graphics_settings_config: GraphicsSettingsConfig,
    validation_data: HashMap<String, HashMap<String, ValidationEntry>>,
}

#[derive(Debug, Clone)]
struct GameProfile {
    base_performance: HashMap<String, f64>, // GPU -> base FPS at 1080p
    rt_performance_impact: f64, // 0.0 to 1.0 (multiplier)
    memory_intensity: f64, // 0.0 to 1.0
    compute_intensity: f64, // 0.0 to 1.0
    texture_streaming: f64, // 0.0 to 1.0
    scene_complexity_variance: f64, // FPS variance due to scene complexity
}

impl GamingPerformanceEngine {
    pub fn new() -> Self {
        // Try to load from TOML file, fallback to hardcoded data
        match Self::load_from_config() {
            Ok(engine) => engine,
            Err(_) => {
                // Fallback to hardcoded initialization
                let mut engine = Self {
                    gpu_features: HashMap::new(),
                    game_profiles: HashMap::new(),
                    upscaling_config: UpscalingConfig {
                        dlss: HashMap::new(),
                        fsr: HashMap::new(),
                    },
                    frame_generation_config: FrameGenerationConfig {
                        compatibility: HashMap::new(),
                    },
                    resolution_scaling: HashMap::new(),
                    graphics_settings_config: GraphicsSettingsConfig {
                        texture_quality: HashMap::new(),
                        shadow_quality: HashMap::new(),
                        anti_aliasing: HashMap::new(),
                    },
                    validation_data: HashMap::new(),
                };

                engine.initialize_gpu_features();
                engine.initialize_game_profiles();
                engine.initialize_fallback_config();
                engine
            }
        }
    }

    fn load_from_config() -> Result<Self, PhantomGpuError> {
        let toml_content = fs
            ::read_to_string("game_profiles.toml")
            .map_err(|e| PhantomGpuError::ConfigError {
                message: format!("Failed to read game_profiles.toml: {}", e),
            })?;

        let config: GameProfilesConfig = toml
            ::from_str(&toml_content)
            .map_err(|e| PhantomGpuError::ConfigError {
                message: format!("Failed to parse game_profiles.toml: {}", e),
            })?;

        let mut game_profiles = HashMap::new();
        for (name, entry) in config.games {
            game_profiles.insert(name, GameProfile {
                base_performance: entry.base_performance,
                rt_performance_impact: entry.rt_performance_impact,
                memory_intensity: entry.memory_intensity,
                compute_intensity: entry.compute_intensity,
                texture_streaming: entry.texture_streaming,
                scene_complexity_variance: entry.scene_complexity_variance,
            });
        }

        Ok(Self {
            gpu_features: config.gpu_features,
            game_profiles,
            upscaling_config: config.upscaling,
            frame_generation_config: config.frame_generation,
            resolution_scaling: config.resolution_scaling,
            graphics_settings_config: config.graphics_settings,
            validation_data: config.validation.unwrap_or_default(),
        })
    }

    fn initialize_gpu_features(&mut self) {
        // NVIDIA RTX 4090 (Ada Lovelace)
        self.gpu_features.insert("RTX 4090".to_string(), GamingGPUFeatures {
            rops: 176,
            tmus: 512,
            rt_cores: 128,
            rt_generation: 3,
            tensor_cores: 512,
            tensor_generation: 4,
            vrs_support: true,
            mesh_shader_support: true,
            av1_encode: true,
            av1_decode: true,
        });

        // NVIDIA RTX 5090 (Blackwell)
        self.gpu_features.insert("RTX 5090".to_string(), GamingGPUFeatures {
            rops: 192,
            tmus: 608,
            rt_cores: 170,
            rt_generation: 4,
            tensor_cores: 680,
            tensor_generation: 5,
            vrs_support: true,
            mesh_shader_support: true,
            av1_encode: true,
            av1_decode: true,
        });

        // NVIDIA RTX 4080
        self.gpu_features.insert("RTX 4080".to_string(), GamingGPUFeatures {
            rops: 112,
            tmus: 304,
            rt_cores: 76,
            rt_generation: 3,
            tensor_cores: 304,
            tensor_generation: 4,
            vrs_support: true,
            mesh_shader_support: true,
            av1_encode: true,
            av1_decode: true,
        });

        // NVIDIA RTX 5080
        self.gpu_features.insert("RTX 5080".to_string(), GamingGPUFeatures {
            rops: 128,
            tmus: 384,
            rt_cores: 96,
            rt_generation: 4,
            tensor_cores: 384,
            tensor_generation: 5,
            vrs_support: true,
            mesh_shader_support: true,
            av1_encode: true,
            av1_decode: true,
        });

        // AMD RX 7900 XTX (RDNA 3)
        self.gpu_features.insert("RX 7900 XTX".to_string(), GamingGPUFeatures {
            rops: 192,
            tmus: 384,
            rt_cores: 96, // Ray accelerators
            rt_generation: 2,
            tensor_cores: 0, // Uses Matrix units instead
            tensor_generation: 0,
            vrs_support: true,
            mesh_shader_support: true,
            av1_encode: true,
            av1_decode: true,
        });
    }

    fn initialize_fallback_config(&mut self) {
        // Initialize fallback upscaling configuration
        self.upscaling_config.dlss.insert("quality".to_string(), 1.4);
        self.upscaling_config.dlss.insert("balanced".to_string(), 1.7);
        self.upscaling_config.dlss.insert("performance".to_string(), 2.3);
        self.upscaling_config.dlss.insert("ultra_performance".to_string(), 3.0);

        self.upscaling_config.fsr.insert("ultra_quality".to_string(), 1.3);
        self.upscaling_config.fsr.insert("quality".to_string(), 1.5);
        self.upscaling_config.fsr.insert("balanced".to_string(), 1.7);
        self.upscaling_config.fsr.insert("performance".to_string(), 2.0);

        // Initialize fallback frame generation configuration
        self.frame_generation_config.compatibility.insert("Cyberpunk 2077".to_string(), 0.9);
        self.frame_generation_config.compatibility.insert(
            "Call of Duty: Modern Warfare III".to_string(),
            0.8
        );
        self.frame_generation_config.compatibility.insert("Fortnite".to_string(), 0.7);
        self.frame_generation_config.compatibility.insert("Hogwarts Legacy".to_string(), 0.85);

        // Initialize fallback resolution scaling
        self.resolution_scaling.insert("1920x1080".to_string(), 1.0);
        self.resolution_scaling.insert("2560x1440".to_string(), 0.65);
        self.resolution_scaling.insert("3440x1440".to_string(), 0.55);
        self.resolution_scaling.insert("3840x2160".to_string(), 0.35);

        // Initialize fallback graphics settings
        self.graphics_settings_config.texture_quality.insert("low".to_string(), 1.05);
        self.graphics_settings_config.texture_quality.insert("medium".to_string(), 1.0);
        self.graphics_settings_config.texture_quality.insert("high".to_string(), 0.95);
        self.graphics_settings_config.texture_quality.insert("ultra".to_string(), 0.9);

        self.graphics_settings_config.shadow_quality.insert("low".to_string(), 1.08);
        self.graphics_settings_config.shadow_quality.insert("medium".to_string(), 1.0);
        self.graphics_settings_config.shadow_quality.insert("high".to_string(), 0.94);
        self.graphics_settings_config.shadow_quality.insert("ultra".to_string(), 0.88);

        self.graphics_settings_config.anti_aliasing.insert("off".to_string(), 1.0);
        self.graphics_settings_config.anti_aliasing.insert("fxaa".to_string(), 0.98);
        self.graphics_settings_config.anti_aliasing.insert("msaa_2x".to_string(), 0.92);
        self.graphics_settings_config.anti_aliasing.insert("msaa_4x".to_string(), 0.85);
        self.graphics_settings_config.anti_aliasing.insert("msaa_8x".to_string(), 0.75);
        self.graphics_settings_config.anti_aliasing.insert("taa".to_string(), 0.96);
    }

    fn initialize_game_profiles(&mut self) {
        // Cyberpunk 2077 - Very demanding, excellent RT showcase
        let mut cyberpunk_base = HashMap::new();
        cyberpunk_base.insert("RTX 4090".to_string(), 120.0);
        cyberpunk_base.insert("RTX 5090".to_string(), 160.0);
        cyberpunk_base.insert("RTX 4080".to_string(), 95.0);
        cyberpunk_base.insert("RTX 5080".to_string(), 130.0);
        cyberpunk_base.insert("RX 7900 XTX".to_string(), 110.0);

        self.game_profiles.insert("Cyberpunk 2077".to_string(), GameProfile {
            base_performance: cyberpunk_base,
            rt_performance_impact: 0.5, // 50% performance hit with RT
            memory_intensity: 0.8,
            compute_intensity: 0.9,
            texture_streaming: 0.9,
            scene_complexity_variance: 0.3,
        });

        // Call of Duty: Modern Warfare III - Competitive FPS
        let mut cod_base = HashMap::new();
        cod_base.insert("RTX 4090".to_string(), 180.0);
        cod_base.insert("RTX 5090".to_string(), 240.0);
        cod_base.insert("RTX 4080".to_string(), 150.0);
        cod_base.insert("RTX 5080".to_string(), 200.0);
        cod_base.insert("RX 7900 XTX".to_string(), 170.0);

        self.game_profiles.insert("Call of Duty: Modern Warfare III".to_string(), GameProfile {
            base_performance: cod_base,
            rt_performance_impact: 0.75, // Less RT impact than Cyberpunk
            memory_intensity: 0.6,
            compute_intensity: 0.7,
            texture_streaming: 0.7,
            scene_complexity_variance: 0.2,
        });

        // Fortnite - Battle Royale with good optimization
        let mut fortnite_base = HashMap::new();
        fortnite_base.insert("RTX 4090".to_string(), 200.0);
        fortnite_base.insert("RTX 5090".to_string(), 260.0);
        fortnite_base.insert("RTX 4080".to_string(), 170.0);
        fortnite_base.insert("RTX 5080".to_string(), 220.0);
        fortnite_base.insert("RX 7900 XTX".to_string(), 190.0);

        self.game_profiles.insert("Fortnite".to_string(), GameProfile {
            base_performance: fortnite_base,
            rt_performance_impact: 0.8,
            memory_intensity: 0.5,
            compute_intensity: 0.6,
            texture_streaming: 0.6,
            scene_complexity_variance: 0.4, // High variance due to building/destruction
        });

        // Hogwarts Legacy - Open world with demanding graphics
        let mut hogwarts_base = HashMap::new();
        hogwarts_base.insert("RTX 4090".to_string(), 90.0);
        hogwarts_base.insert("RTX 5090".to_string(), 130.0);
        hogwarts_base.insert("RTX 4080".to_string(), 70.0);
        hogwarts_base.insert("RTX 5080".to_string(), 100.0);
        hogwarts_base.insert("RX 7900 XTX".to_string(), 85.0);

        self.game_profiles.insert("Hogwarts Legacy".to_string(), GameProfile {
            base_performance: hogwarts_base,
            rt_performance_impact: 0.6,
            memory_intensity: 0.9,
            compute_intensity: 0.8,
            texture_streaming: 0.9,
            scene_complexity_variance: 0.3,
        });
    }

    pub fn predict_gaming_performance(
        &self,
        gpu_config: &GpuModel,
        workload: &GamingWorkload,
        ambient_temp: f64
    ) -> Result<GamingPerformance, PhantomGpuError> {
        let gpu_features = self.gpu_features
            .get(&gpu_config.name)
            .ok_or_else(|| PhantomGpuError::GpuNotFound { gpu_name: gpu_config.name.clone() })?;

        let game_profile = self.game_profiles
            .get(&workload.game_name)
            .ok_or_else(|| PhantomGpuError::ModelLoadError(workload.game_name.clone()))?;

        // Step 1: Get base performance at 1080p
        let base_fps = game_profile.base_performance.get(&gpu_config.name).cloned().unwrap_or(60.0); // Default fallback

        // Step 2: Apply resolution scaling
        let resolution_scaled_fps = self.apply_resolution_scaling(base_fps, workload.resolution)?;

        // Step 3: Apply ray tracing impact
        let rt_scaled_fps = if workload.ray_tracing {
            self.apply_ray_tracing_impact(resolution_scaled_fps, gpu_features, game_profile)
        } else {
            resolution_scaled_fps
        };

        // Step 4: Apply DLSS/FSR upscaling
        let upscaled_fps = self.apply_upscaling(
            rt_scaled_fps,
            &workload.dlss_mode,
            &workload.fsr_mode,
            gpu_features
        )?;

        // Step 5: Apply graphics settings impact
        let settings_scaled_fps = self.apply_graphics_settings(
            upscaled_fps,
            &workload.graphics_settings,
            gpu_features
        );

        // Step 6: Apply scene complexity variance
        let final_fps = self.apply_scene_complexity(
            settings_scaled_fps,
            workload.scene_complexity,
            game_profile
        );

        // Step 7: Calculate frame time metrics
        let frame_time_ms = 1000.0 / final_fps;
        let one_percent_low = final_fps * 0.75; // Typical 1% low relationship
        let frame_time_consistency = self.calculate_frame_time_consistency(final_fps, game_profile);

        // Step 8: Estimate resource utilization
        let (gpu_utilization, memory_utilization) = self.estimate_resource_utilization(
            final_fps,
            workload,
            gpu_config,
            game_profile
        );

        // Step 9: Calculate power consumption and temperature
        let power_consumption = self.calculate_gaming_power_consumption(
            gpu_utilization,
            workload.scene_complexity,
            gpu_config
        );
        let temperature = self.estimate_gaming_temperature(
            power_consumption,
            ambient_temp,
            gpu_config
        );

        Ok(GamingPerformance {
            avg_fps: final_fps,
            one_percent_low,
            frame_time_ms,
            frame_time_consistency,
            gpu_utilization,
            memory_utilization,
            power_consumption,
            temperature,
        })
    }

    fn apply_resolution_scaling(
        &self,
        base_fps: f64,
        resolution: (u32, u32)
    ) -> Result<f64, PhantomGpuError> {
        let resolution_key = format!("{}x{}", resolution.0, resolution.1);

        let scaling_factor = if let Some(&factor) = self.resolution_scaling.get(&resolution_key) {
            factor
        } else {
            // Fallback to calculation for unknown resolutions
            let pixel_count = (resolution.0 as f64) * (resolution.1 as f64);
            let baseline_pixels = 1920.0 * 1080.0; // 1080p baseline

            let scaling_factor = baseline_pixels / pixel_count;

            // Apply non-linear scaling (not all workload is pixel-bound)
            let pixel_bound_factor = 0.7; // 70% of workload is pixel-bound
            let cpu_bound_factor = 1.0 - pixel_bound_factor;

            pixel_bound_factor * scaling_factor + cpu_bound_factor
        };

        Ok(base_fps * scaling_factor)
    }

    fn apply_ray_tracing_impact(
        &self,
        base_fps: f64,
        gpu_features: &GamingGPUFeatures,
        game_profile: &GameProfile
    ) -> f64 {
        let rt_efficiency = match gpu_features.rt_generation {
            4 => 1.0, // Latest RT cores (Blackwell)
            3 => 0.85, // Ada Lovelace
            2 => 0.7, // RDNA 3 / Turing
            1 => 0.5, // First-gen RT
            _ => 0.3, // Software RT
        };

        let effective_rt_impact = game_profile.rt_performance_impact * rt_efficiency;
        base_fps * effective_rt_impact
    }

    fn apply_upscaling(
        &self,
        base_fps: f64,
        dlss_mode: &DLSSMode,
        fsr_mode: &FSRMode,
        gpu_features: &GamingGPUFeatures
    ) -> Result<f64, PhantomGpuError> {
        let mut fps = base_fps;

        // Apply DLSS if available and enabled
        if gpu_features.tensor_cores > 0 {
            let dlss_scaling = match dlss_mode {
                DLSSMode::Off => 1.0,
                DLSSMode::Quality =>
                    self.upscaling_config.dlss.get("quality").copied().unwrap_or(1.4),
                DLSSMode::Balanced =>
                    self.upscaling_config.dlss.get("balanced").copied().unwrap_or(1.7),
                DLSSMode::Performance =>
                    self.upscaling_config.dlss.get("performance").copied().unwrap_or(2.3),
                DLSSMode::UltraPerformance =>
                    self.upscaling_config.dlss.get("ultra_performance").copied().unwrap_or(3.0),
            };

            let dlss_efficiency = match gpu_features.tensor_generation {
                5 => 0.98, // Blackwell - improved efficiency
                4 => 0.95, // Ada Lovelace
                3 => 0.92, // Ampere
                _ => 0.9, // Older generations
            };

            fps *= dlss_scaling * dlss_efficiency;
        }

        // Apply FSR if DLSS is off (they're mutually exclusive)
        if matches!(dlss_mode, DLSSMode::Off) {
            let fsr_scaling = match fsr_mode {
                FSRMode::Off => 1.0,
                FSRMode::UltraQuality =>
                    self.upscaling_config.fsr.get("ultra_quality").copied().unwrap_or(1.3),
                FSRMode::Quality =>
                    self.upscaling_config.fsr.get("quality").copied().unwrap_or(1.5),
                FSRMode::Balanced =>
                    self.upscaling_config.fsr.get("balanced").copied().unwrap_or(1.7),
                FSRMode::Performance =>
                    self.upscaling_config.fsr.get("performance").copied().unwrap_or(2.0),
            };

            fps *= fsr_scaling * 0.93; // FSR has slightly lower efficiency than DLSS
        }

        Ok(fps)
    }

    fn apply_graphics_settings(
        &self,
        base_fps: f64,
        settings: &GraphicsSettings,
        gpu_features: &GamingGPUFeatures
    ) -> f64 {
        let mut fps = base_fps;

        // Texture quality impact
        let texture_impact = match settings.texture_quality {
            Quality::Ultra =>
                self.graphics_settings_config.texture_quality.get("ultra").copied().unwrap_or(0.9),
            Quality::High =>
                self.graphics_settings_config.texture_quality.get("high").copied().unwrap_or(0.95),
            Quality::Medium =>
                self.graphics_settings_config.texture_quality.get("medium").copied().unwrap_or(1.0),
            Quality::Low =>
                self.graphics_settings_config.texture_quality.get("low").copied().unwrap_or(1.05),
        };
        fps *= texture_impact;

        // Shadow quality impact
        let shadow_impact = match settings.shadow_quality {
            Quality::Ultra =>
                self.graphics_settings_config.shadow_quality.get("ultra").copied().unwrap_or(0.88),
            Quality::High =>
                self.graphics_settings_config.shadow_quality.get("high").copied().unwrap_or(0.94),
            Quality::Medium =>
                self.graphics_settings_config.shadow_quality.get("medium").copied().unwrap_or(1.0),
            Quality::Low =>
                self.graphics_settings_config.shadow_quality.get("low").copied().unwrap_or(1.08),
        };
        fps *= shadow_impact;

        // Anti-aliasing impact
        let aa_impact = match settings.anti_aliasing {
            AntiAliasing::Off =>
                self.graphics_settings_config.anti_aliasing.get("off").copied().unwrap_or(1.0),
            AntiAliasing::FXAA =>
                self.graphics_settings_config.anti_aliasing.get("fxaa").copied().unwrap_or(0.98),
            AntiAliasing::TAA =>
                self.graphics_settings_config.anti_aliasing.get("taa").copied().unwrap_or(0.96),
            AntiAliasing::MSAA2x =>
                self.graphics_settings_config.anti_aliasing.get("msaa_2x").copied().unwrap_or(0.92),
            AntiAliasing::MSAA4x =>
                self.graphics_settings_config.anti_aliasing.get("msaa_4x").copied().unwrap_or(0.85),
            AntiAliasing::MSAA8x =>
                self.graphics_settings_config.anti_aliasing.get("msaa_8x").copied().unwrap_or(0.75),
        };
        fps *= aa_impact;

        // Variable Rate Shading boost
        if settings.variable_rate_shading && gpu_features.vrs_support {
            fps *= 1.15; // 15% performance boost
        }

        // Mesh shaders boost
        if settings.mesh_shaders && gpu_features.mesh_shader_support {
            fps *= 1.1; // 10% performance boost for complex geometry
        }

        fps
    }

    fn apply_scene_complexity(
        &self,
        base_fps: f64,
        complexity: f64,
        game_profile: &GameProfile
    ) -> f64 {
        let complexity_impact = 1.0 - complexity * game_profile.scene_complexity_variance;
        base_fps * complexity_impact
    }

    fn calculate_frame_time_consistency(
        &self,
        avg_fps: f64,
        game_profile: &GameProfile
    ) -> FrameTimeConsistency {
        let variance_factor = game_profile.scene_complexity_variance;
        let frame_time_variation = (1000.0 / avg_fps) * variance_factor;

        if frame_time_variation < 5.0 {
            FrameTimeConsistency::Good
        } else if frame_time_variation < 10.0 {
            FrameTimeConsistency::Acceptable
        } else {
            FrameTimeConsistency::Poor
        }
    }

    fn estimate_resource_utilization(
        &self,
        fps: f64,
        workload: &GamingWorkload,
        gpu_config: &GpuModel,
        game_profile: &GameProfile
    ) -> (f64, f64) {
        // GPU utilization based on target FPS vs achieved FPS
        let target_ratio = fps / workload.target_fps.max(30.0);
        let gpu_utilization = (target_ratio * 0.8).min(1.0);

        // Memory utilization based on resolution and game requirements
        let resolution_factor =
            ((workload.resolution.0 * workload.resolution.1) as f64) / (1920.0 * 1080.0);
        let memory_utilization = (game_profile.memory_intensity * resolution_factor * 0.6).min(1.0);

        (gpu_utilization, memory_utilization)
    }

    fn calculate_gaming_power_consumption(
        &self,
        gpu_utilization: f64,
        scene_complexity: f64,
        gpu_config: &GpuModel
    ) -> f64 {
        let estimated_tdp = gpu_config.compute_tflops * 15.0; // Rough estimate: 15W per TFLOP
        let base_power = estimated_tdp * 0.3; // 30% TDP for basic operations
        let rendering_power = estimated_tdp * 0.6 * (scene_complexity as f32); // Up to 60% for rendering
        let utilization_scaling = gpu_utilization.powf(0.8); // Sub-linear scaling

        (base_power + rendering_power * (utilization_scaling as f32)) as f64
    }

    fn estimate_gaming_temperature(
        &self,
        power_consumption: f64,
        ambient_temp: f64,
        gpu_config: &GpuModel
    ) -> f64 {
        let thermal_resistance = 0.15; // K/W typical for gaming GPUs
        let temperature_rise = power_consumption * thermal_resistance;
        ambient_temp + temperature_rise
    }

    pub fn predict_frame_generation(&self, base_fps: f64, gpu_arch: &str) -> f64 {
        match gpu_arch {
            "Ada Lovelace" => base_fps * 1.8, // DLSS 3 Frame Generation
            "Blackwell" => base_fps * 2.2, // Enhanced frame generation
            _ => base_fps, // No frame generation support
        }
    }

    pub fn predict_frame_generation_advanced(
        &self,
        workload: &GamingWorkload,
        gpu_features: &GamingGPUFeatures,
        base_fps: f64
    ) -> Result<FrameGenerationResult, PhantomGpuError> {
        let frame_gen_support = gpu_features.tensor_generation >= 4; // Requires Ada Lovelace+

        if !frame_gen_support {
            return Ok(FrameGenerationResult {
                supported: false,
                generated_fps: base_fps,
                frame_generation_ratio: 1.0,
                latency_penalty_ms: 0.0,
                quality_impact: FrameGenerationQuality::NotSupported,
            });
        }

        // Frame generation scaling based on architecture
        let base_scaling = match gpu_features.tensor_generation {
            5 => 2.2, // Blackwell - improved frame generation
            4 => 1.8, // Ada Lovelace - DLSS 3
            _ => 1.0, // No support
        };

        // Adjust scaling based on game characteristics
        let game_compatibility = self.get_frame_generation_compatibility(&workload.game_name);
        let effective_scaling = base_scaling * game_compatibility;

        // Calculate latency penalty
        let latency_penalty = self.calculate_frame_generation_latency_penalty(
            base_fps,
            effective_scaling,
            &workload.dlss_mode
        );

        // Determine quality impact
        let quality_impact = self.assess_frame_generation_quality(
            base_fps,
            effective_scaling,
            workload.scene_complexity
        );

        Ok(FrameGenerationResult {
            supported: true,
            generated_fps: base_fps * effective_scaling,
            frame_generation_ratio: effective_scaling,
            latency_penalty_ms: latency_penalty,
            quality_impact,
        })
    }

    fn get_frame_generation_compatibility(&self, game_name: &str) -> f64 {
        self.frame_generation_config.compatibility.get(game_name).copied().unwrap_or(0.75) // Default compatibility
    }

    fn calculate_frame_generation_latency_penalty(
        &self,
        base_fps: f64,
        scaling: f64,
        dlss_mode: &DLSSMode
    ) -> f64 {
        let base_latency_penalty = match dlss_mode {
            DLSSMode::Performance => 12.0, // Higher penalty with aggressive DLSS
            DLSSMode::Balanced => 8.0,
            DLSSMode::Quality => 6.0,
            DLSSMode::Off => 10.0,
            DLSSMode::UltraPerformance => 15.0,
        };

        // Lower base FPS = higher latency penalty
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

        if quality_score > 60.0 {
            FrameGenerationQuality::Excellent
        } else if quality_score > 40.0 {
            FrameGenerationQuality::Good
        } else if quality_score > 20.0 {
            FrameGenerationQuality::Acceptable
        } else {
            FrameGenerationQuality::Poor
        }
    }

    pub fn get_supported_games(&self) -> Vec<String> {
        self.game_profiles.keys().cloned().collect()
    }

    pub fn get_supported_gpus(&self) -> Vec<String> {
        self.gpu_features.keys().cloned().collect()
    }

    pub fn get_gpu_features(&self, gpu_name: &str) -> Option<&GamingGPUFeatures> {
        self.gpu_features.get(gpu_name)
    }
}

impl Default for GamingPerformanceEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions for common gaming scenarios
impl GamingPerformanceEngine {
    pub fn optimize_for_competitive_gaming(
        &self,
        gpu_config: &GpuModel,
        target_fps: f64
    ) -> Result<GamingWorkload, PhantomGpuError> {
        Ok(GamingWorkload {
            game_name: "Call of Duty: Modern Warfare III".to_string(),
            resolution: (1920, 1080), // 1080p for competitive
            ray_tracing: false, // Disabled for max FPS
            dlss_mode: DLSSMode::Performance, // Max performance
            fsr_mode: FSRMode::Off,
            target_fps,
            scene_complexity: 0.5, // Medium complexity
            graphics_settings: GraphicsSettings {
                texture_quality: Quality::Medium,
                shadow_quality: Quality::Low,
                anti_aliasing: AntiAliasing::FXAA,
                anisotropic_filtering: 8,
                variable_rate_shading: true,
                mesh_shaders: false,
            },
        })
    }

    pub fn optimize_for_visual_quality(
        &self,
        gpu_config: &GpuModel,
        resolution: (u32, u32)
    ) -> Result<GamingWorkload, PhantomGpuError> {
        Ok(GamingWorkload {
            game_name: "Cyberpunk 2077".to_string(),
            resolution,
            ray_tracing: true, // Enable RT for visual quality
            dlss_mode: DLSSMode::Quality, // Balance quality and performance
            fsr_mode: FSRMode::Off,
            target_fps: 60.0, // Cinematic experience
            scene_complexity: 0.8, // High complexity
            graphics_settings: GraphicsSettings {
                texture_quality: Quality::Ultra,
                shadow_quality: Quality::High,
                anti_aliasing: AntiAliasing::TAA,
                anisotropic_filtering: 16,
                variable_rate_shading: false,
                mesh_shaders: true,
            },
        })
    }
}
