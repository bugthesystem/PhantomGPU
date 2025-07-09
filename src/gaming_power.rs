use crate::gpu_config::GpuModel;
use crate::gaming_performance::{ GamingWorkload, Quality, AntiAliasing, DLSSMode, FSRMode };
use crate::errors::PhantomGpuError;
use serde::{ Deserialize, Serialize };
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingPowerProfile {
    pub workload_type: PowerWorkloadType,
    pub base_power_percentage: f64, // Base power consumption as % of TDP
    pub rendering_power_percentage: f64, // Max rendering power as % of TDP
    pub memory_power_percentage: f64, // Memory subsystem power as % of TDP
    pub io_power_percentage: f64, // I/O and display power as % of TDP
    pub efficiency_curve: PowerEfficiencyCurve,
    pub scene_complexity_impact: f64, // How much scene complexity affects power
    pub fps_scaling_factor: f64, // How FPS target affects power consumption
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerWorkloadType {
    Gaming,
    Compute,
    ContentCreation,
    Streaming,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerEfficiencyCurve {
    pub idle_efficiency: f64, // Efficiency at idle
    pub optimal_efficiency: f64, // Peak efficiency point
    pub optimal_utilization: f64, // Utilization at peak efficiency
    pub max_load_efficiency: f64, // Efficiency at max load
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingPowerConsumption {
    pub total_power_watts: f64,
    pub base_power_watts: f64,
    pub rendering_power_watts: f64,
    pub memory_power_watts: f64,
    pub io_power_watts: f64,
    pub cooling_power_watts: f64,
    pub efficiency_percentage: f64,
    pub power_per_fps: f64, // Watts per FPS
    pub performance_per_watt: f64, // FPS per Watt
    pub estimated_battery_life_hours: Option<f64>, // For mobile GPUs
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerScenarioAnalysis {
    pub scenario_name: String,
    pub power_consumption: GamingPowerConsumption,
    pub thermal_impact: f64,
    pub sustainability_score: f64, // 0-100 scale
    pub cost_per_hour: f64, // Based on electricity rates
    pub carbon_footprint_kg: f64, // CO2 equivalent per hour
}

pub struct GamingPowerEngine {
    power_profiles: HashMap<String, GamingPowerProfile>,
    gpu_power_specs: HashMap<String, GPUPowerSpecs>,
    electricity_rate_per_kwh: f64,
    carbon_intensity_kg_per_kwh: f64,
}

#[derive(Debug, Clone)]
struct GPUPowerSpecs {
    tdp: f64,
    base_power: f64,
    peak_power: f64,
    memory_power: f64,
    cooling_power: f64,
    power_efficiency: f64,
    voltage_domains: Vec<VoltageDomain>,
}

#[derive(Debug, Clone)]
struct VoltageDomain {
    name: String,
    voltage: f64,
    current_capacity: f64,
    power_rail: f64,
}

impl GamingPowerEngine {
    pub fn new() -> Self {
        Self {
            power_profiles: HashMap::new(),
            gpu_power_specs: HashMap::new(),
            electricity_rate_per_kwh: 0.12, // $0.12/kWh default
            carbon_intensity_kg_per_kwh: 0.5, // 0.5 kg CO2/kWh default
        }
    }

    pub fn with_electricity_rate(mut self, rate: f64) -> Self {
        self.electricity_rate_per_kwh = rate;
        self
    }

    pub fn with_carbon_intensity(mut self, intensity: f64) -> Self {
        self.carbon_intensity_kg_per_kwh = intensity;
        self
    }

    pub fn initialize(&mut self) {
        self.initialize_power_profiles();
        self.initialize_gpu_power_specs();
    }

    fn initialize_power_profiles(&mut self) {
        // Gaming power profile
        self.power_profiles.insert("Gaming".to_string(), GamingPowerProfile {
            workload_type: PowerWorkloadType::Gaming,
            base_power_percentage: 0.25, // 25% TDP for basic operations
            rendering_power_percentage: 0.6, // Up to 60% TDP for rendering
            memory_power_percentage: 0.15, // 15% TDP for memory
            io_power_percentage: 0.05, // 5% TDP for I/O
            efficiency_curve: PowerEfficiencyCurve {
                idle_efficiency: 0.3,
                optimal_efficiency: 0.85,
                optimal_utilization: 0.75,
                max_load_efficiency: 0.7,
            },
            scene_complexity_impact: 0.4, // Scene complexity can add 40% more power
            fps_scaling_factor: 0.3, // FPS target affects power by 30%
        });

        // Compute power profile
        self.power_profiles.insert("Compute".to_string(), GamingPowerProfile {
            workload_type: PowerWorkloadType::Compute,
            base_power_percentage: 0.2,
            rendering_power_percentage: 0.75, // Higher sustained compute load
            memory_power_percentage: 0.2,
            io_power_percentage: 0.03,
            efficiency_curve: PowerEfficiencyCurve {
                idle_efficiency: 0.25,
                optimal_efficiency: 0.9,
                optimal_utilization: 0.85,
                max_load_efficiency: 0.85,
            },
            scene_complexity_impact: 0.1,
            fps_scaling_factor: 0.1,
        });

        // Content creation power profile
        self.power_profiles.insert("ContentCreation".to_string(), GamingPowerProfile {
            workload_type: PowerWorkloadType::ContentCreation,
            base_power_percentage: 0.3,
            rendering_power_percentage: 0.55,
            memory_power_percentage: 0.25,
            io_power_percentage: 0.08,
            efficiency_curve: PowerEfficiencyCurve {
                idle_efficiency: 0.35,
                optimal_efficiency: 0.8,
                optimal_utilization: 0.7,
                max_load_efficiency: 0.75,
            },
            scene_complexity_impact: 0.3,
            fps_scaling_factor: 0.2,
        });

        // Streaming power profile
        self.power_profiles.insert("Streaming".to_string(), GamingPowerProfile {
            workload_type: PowerWorkloadType::Streaming,
            base_power_percentage: 0.35,
            rendering_power_percentage: 0.45,
            memory_power_percentage: 0.2,
            io_power_percentage: 0.12, // Higher I/O for encoding
            efficiency_curve: PowerEfficiencyCurve {
                idle_efficiency: 0.4,
                optimal_efficiency: 0.75,
                optimal_utilization: 0.65,
                max_load_efficiency: 0.7,
            },
            scene_complexity_impact: 0.25,
            fps_scaling_factor: 0.35, // Higher FPS for streaming quality
        });
    }

    fn initialize_gpu_power_specs(&mut self) {
        // RTX 4090 power specs
        self.gpu_power_specs.insert("RTX 4090".to_string(), GPUPowerSpecs {
            tdp: 450.0,
            base_power: 50.0,
            peak_power: 480.0,
            memory_power: 80.0,
            cooling_power: 25.0,
            power_efficiency: 0.88,
            voltage_domains: vec![
                VoltageDomain {
                    name: "GPU Core".to_string(),
                    voltage: 1.0,
                    current_capacity: 400.0,
                    power_rail: 400.0,
                },
                VoltageDomain {
                    name: "Memory".to_string(),
                    voltage: 2.5,
                    current_capacity: 32.0,
                    power_rail: 80.0,
                }
            ],
        });

        // RTX 5090 power specs (improved efficiency)
        self.gpu_power_specs.insert("RTX 5090".to_string(), GPUPowerSpecs {
            tdp: 500.0,
            base_power: 45.0,
            peak_power: 525.0,
            memory_power: 75.0,
            cooling_power: 30.0,
            power_efficiency: 0.92,
            voltage_domains: vec![
                VoltageDomain {
                    name: "GPU Core".to_string(),
                    voltage: 0.9,
                    current_capacity: 450.0,
                    power_rail: 450.0,
                },
                VoltageDomain {
                    name: "Memory".to_string(),
                    voltage: 2.2,
                    current_capacity: 34.0,
                    power_rail: 75.0,
                }
            ],
        });

        // RTX 4080 power specs
        self.gpu_power_specs.insert("RTX 4080".to_string(), GPUPowerSpecs {
            tdp: 320.0,
            base_power: 35.0,
            peak_power: 350.0,
            memory_power: 60.0,
            cooling_power: 20.0,
            power_efficiency: 0.86,
            voltage_domains: vec![
                VoltageDomain {
                    name: "GPU Core".to_string(),
                    voltage: 1.0,
                    current_capacity: 280.0,
                    power_rail: 280.0,
                },
                VoltageDomain {
                    name: "Memory".to_string(),
                    voltage: 2.5,
                    current_capacity: 24.0,
                    power_rail: 60.0,
                }
            ],
        });

        // RTX 5080 power specs
        self.gpu_power_specs.insert("RTX 5080".to_string(), GPUPowerSpecs {
            tdp: 360.0,
            base_power: 32.0,
            peak_power: 380.0,
            memory_power: 55.0,
            cooling_power: 22.0,
            power_efficiency: 0.89,
            voltage_domains: vec![
                VoltageDomain {
                    name: "GPU Core".to_string(),
                    voltage: 0.95,
                    current_capacity: 320.0,
                    power_rail: 320.0,
                },
                VoltageDomain {
                    name: "Memory".to_string(),
                    voltage: 2.3,
                    current_capacity: 24.0,
                    power_rail: 55.0,
                }
            ],
        });
    }

    pub fn calculate_gaming_power_consumption(
        &self,
        gpu_config: &GpuModel,
        workload: &GamingWorkload,
        target_fps: f64,
        actual_fps: f64
    ) -> Result<GamingPowerConsumption, PhantomGpuError> {
        let profile = self.power_profiles
            .get("Gaming")
            .ok_or_else(||
                PhantomGpuError::ModelLoadError("Gaming profile not found".to_string())
            )?;

        let gpu_specs = self.gpu_power_specs
            .get(&gpu_config.name)
            .ok_or_else(|| PhantomGpuError::GpuNotFound { gpu_name: gpu_config.name.clone() })?;

        // Calculate base power consumption
        let base_power = gpu_specs.base_power;

        // Calculate rendering power based on scene complexity and settings
        let rendering_load = self.calculate_rendering_load(workload, target_fps, actual_fps)?;
        let rendering_power = rendering_load * gpu_specs.tdp * profile.rendering_power_percentage;

        // Calculate memory power based on resolution and textures
        let memory_load = self.calculate_memory_load(workload)?;
        let memory_power = memory_load * gpu_specs.memory_power;

        // Calculate I/O power for display output and data transfer
        let io_load = self.calculate_io_load(workload)?;
        let io_power = io_load * gpu_specs.tdp * profile.io_power_percentage;

        // Calculate cooling power based on total thermal load
        let total_compute_power = rendering_power + memory_power + io_power;
        let cooling_power = (total_compute_power * 0.15).min(gpu_specs.cooling_power);

        // Calculate total power consumption
        let total_power = base_power + rendering_power + memory_power + io_power + cooling_power;

        // Calculate efficiency metrics
        let efficiency = self.calculate_power_efficiency(total_power, gpu_specs.tdp, profile);
        let power_per_fps = total_power / actual_fps.max(1.0);
        let performance_per_watt = actual_fps / total_power;

        // Estimate battery life for mobile scenarios (hypothetical)
        let battery_life = if gpu_config.name.contains("Mobile") {
            Some(100.0 / total_power) // Assume 100Wh battery
        } else {
            None
        };

        Ok(GamingPowerConsumption {
            total_power_watts: total_power,
            base_power_watts: base_power,
            rendering_power_watts: rendering_power,
            memory_power_watts: memory_power,
            io_power_watts: io_power,
            cooling_power_watts: cooling_power,
            efficiency_percentage: efficiency * 100.0,
            power_per_fps,
            performance_per_watt,
            estimated_battery_life_hours: battery_life,
        })
    }

    fn calculate_rendering_load(
        &self,
        workload: &GamingWorkload,
        target_fps: f64,
        actual_fps: f64
    ) -> Result<f64, PhantomGpuError> {
        let mut load = 0.4; // Base rendering load

        // Resolution impact
        let pixel_count = (workload.resolution.0 as f64) * (workload.resolution.1 as f64);
        let resolution_factor = pixel_count / (1920.0 * 1080.0); // Normalized to 1080p
        load *= resolution_factor.powf(0.8); // Sub-linear scaling

        // Ray tracing impact
        if workload.ray_tracing {
            load *= 1.6; // 60% increase for ray tracing
        }

        // Graphics settings impact
        load *= self.calculate_graphics_settings_load(&workload.graphics_settings)?;

        // DLSS/FSR efficiency
        load *= self.calculate_upscaling_efficiency(&workload.dlss_mode, &workload.fsr_mode)?;

        // Scene complexity impact
        load *= 1.0 + workload.scene_complexity * 0.5;

        // FPS target impact
        let fps_scaling = (target_fps / 60.0).min(2.0); // Cap at 2x for 120fps
        load *= fps_scaling.powf(0.6); // Sub-linear FPS scaling

        // Utilization based on actual vs target FPS
        let utilization = (actual_fps / target_fps.max(30.0)).min(1.2);
        load *= utilization;

        Ok(load.min(1.0))
    }

    fn calculate_memory_load(&self, workload: &GamingWorkload) -> Result<f64, PhantomGpuError> {
        let mut load = 0.3; // Base memory load

        // Resolution impact on memory bandwidth
        let pixel_count = (workload.resolution.0 as f64) * (workload.resolution.1 as f64);
        let resolution_factor = pixel_count / (1920.0 * 1080.0);
        load *= resolution_factor.powf(0.7);

        // Texture quality impact
        let texture_multiplier = match workload.graphics_settings.texture_quality {
            Quality::Ultra => 1.4,
            Quality::High => 1.2,
            Quality::Medium => 1.0,
            Quality::Low => 0.8,
        };
        load *= texture_multiplier;

        // Anti-aliasing impact
        let aa_multiplier = match workload.graphics_settings.anti_aliasing {
            AntiAliasing::Off => 1.0,
            AntiAliasing::FXAA => 1.1,
            AntiAliasing::TAA => 1.2,
            AntiAliasing::MSAA2x => 1.3,
            AntiAliasing::MSAA4x => 1.6,
            AntiAliasing::MSAA8x => 2.2,
        };
        load *= aa_multiplier;

        // Ray tracing memory impact
        if workload.ray_tracing {
            load *= 1.3;
        }

        Ok(load.min(1.0))
    }

    fn calculate_io_load(&self, workload: &GamingWorkload) -> Result<f64, PhantomGpuError> {
        let mut load = 0.2; // Base I/O load

        // Resolution impact on display output
        let pixel_count = (workload.resolution.0 as f64) * (workload.resolution.1 as f64);
        let resolution_factor = pixel_count / (1920.0 * 1080.0);
        load *= resolution_factor.powf(0.5);

        // Target FPS impact on display refresh
        let fps_factor = (workload.target_fps / 60.0).min(4.0);
        load *= fps_factor.powf(0.3);

        Ok(load.min(1.0))
    }

    fn calculate_graphics_settings_load(
        &self,
        settings: &crate::gaming_performance::GraphicsSettings
    ) -> Result<f64, PhantomGpuError> {
        let mut multiplier = 1.0;

        // Texture quality
        multiplier *= match settings.texture_quality {
            Quality::Ultra => 1.2,
            Quality::High => 1.1,
            Quality::Medium => 1.0,
            Quality::Low => 0.9,
        };

        // Shadow quality
        multiplier *= match settings.shadow_quality {
            Quality::Ultra => 1.3,
            Quality::High => 1.15,
            Quality::Medium => 1.0,
            Quality::Low => 0.85,
        };

        // Variable Rate Shading efficiency
        if settings.variable_rate_shading {
            multiplier *= 0.88; // 12% power reduction
        }

        // Mesh shaders efficiency
        if settings.mesh_shaders {
            multiplier *= 0.92; // 8% power reduction for complex geometry
        }

        Ok(multiplier)
    }

    fn calculate_upscaling_efficiency(
        &self,
        dlss_mode: &DLSSMode,
        fsr_mode: &FSRMode
    ) -> Result<f64, PhantomGpuError> {
        let dlss_efficiency = match dlss_mode {
            DLSSMode::Off => 1.0,
            DLSSMode::Quality => 0.85, // 15% power reduction
            DLSSMode::Balanced => 0.75, // 25% power reduction
            DLSSMode::Performance => 0.65, // 35% power reduction
            DLSSMode::UltraPerformance => 0.55, // 45% power reduction
        };

        let fsr_efficiency = match fsr_mode {
            FSRMode::Off => 1.0,
            FSRMode::UltraQuality => 0.9,
            FSRMode::Quality => 0.82,
            FSRMode::Balanced => 0.75,
            FSRMode::Performance => 0.68,
        };

        // Use DLSS if available, otherwise FSR
        Ok(if !matches!(dlss_mode, DLSSMode::Off) { dlss_efficiency } else { fsr_efficiency })
    }

    fn calculate_power_efficiency(
        &self,
        total_power: f64,
        tdp: f64,
        profile: &GamingPowerProfile
    ) -> f64 {
        let utilization = total_power / tdp;
        let optimal_point = profile.efficiency_curve.optimal_utilization;

        if utilization <= optimal_point {
            let ratio = utilization / optimal_point;
            profile.efficiency_curve.idle_efficiency +
                ratio *
                    (profile.efficiency_curve.optimal_efficiency -
                        profile.efficiency_curve.idle_efficiency)
        } else {
            let ratio = (utilization - optimal_point) / (1.0 - optimal_point);
            profile.efficiency_curve.optimal_efficiency -
                ratio *
                    (profile.efficiency_curve.optimal_efficiency -
                        profile.efficiency_curve.max_load_efficiency)
        }
    }

    pub fn analyze_power_scenarios(
        &self,
        gpu_config: &GpuModel,
        base_workload: &GamingWorkload
    ) -> Result<Vec<PowerScenarioAnalysis>, PhantomGpuError> {
        let mut scenarios = Vec::new();

        // Competitive gaming scenario
        let competitive_workload = GamingWorkload {
            resolution: (1920, 1080),
            ray_tracing: false,
            dlss_mode: DLSSMode::Performance,
            target_fps: 240.0,
            graphics_settings: crate::gaming_performance::GraphicsSettings {
                texture_quality: Quality::Medium,
                shadow_quality: Quality::Low,
                anti_aliasing: AntiAliasing::FXAA,
                variable_rate_shading: true,
                mesh_shaders: false,
                anisotropic_filtering: 8,
            },
            ..base_workload.clone()
        };

        let competitive_power = self.calculate_gaming_power_consumption(
            gpu_config,
            &competitive_workload,
            240.0,
            220.0
        )?;

        scenarios.push(PowerScenarioAnalysis {
            scenario_name: "Competitive Gaming".to_string(),
            power_consumption: competitive_power.clone(),
            thermal_impact: competitive_power.total_power_watts * 0.8,
            sustainability_score: 85.0,
            cost_per_hour: (competitive_power.total_power_watts * self.electricity_rate_per_kwh) /
            1000.0,
            carbon_footprint_kg: (competitive_power.total_power_watts *
                self.carbon_intensity_kg_per_kwh) /
            1000.0,
        });

        // 4K Max quality scenario
        let max_quality_workload = GamingWorkload {
            resolution: (3840, 2160),
            ray_tracing: true,
            dlss_mode: DLSSMode::Quality,
            target_fps: 60.0,
            graphics_settings: crate::gaming_performance::GraphicsSettings {
                texture_quality: Quality::Ultra,
                shadow_quality: Quality::Ultra,
                anti_aliasing: AntiAliasing::TAA,
                variable_rate_shading: false,
                mesh_shaders: true,
                anisotropic_filtering: 16,
            },
            ..base_workload.clone()
        };

        let max_quality_power = self.calculate_gaming_power_consumption(
            gpu_config,
            &max_quality_workload,
            60.0,
            55.0
        )?;

        scenarios.push(PowerScenarioAnalysis {
            scenario_name: "4K Max Quality".to_string(),
            power_consumption: max_quality_power.clone(),
            thermal_impact: max_quality_power.total_power_watts * 0.95,
            sustainability_score: 45.0,
            cost_per_hour: (max_quality_power.total_power_watts * self.electricity_rate_per_kwh) /
            1000.0,
            carbon_footprint_kg: (max_quality_power.total_power_watts *
                self.carbon_intensity_kg_per_kwh) /
            1000.0,
        });

        // Balanced 1440p scenario
        let balanced_workload = GamingWorkload {
            resolution: (2560, 1440),
            ray_tracing: true,
            dlss_mode: DLSSMode::Balanced,
            target_fps: 120.0,
            graphics_settings: crate::gaming_performance::GraphicsSettings {
                texture_quality: Quality::High,
                shadow_quality: Quality::High,
                anti_aliasing: AntiAliasing::TAA,
                variable_rate_shading: true,
                mesh_shaders: true,
                anisotropic_filtering: 16,
            },
            ..base_workload.clone()
        };

        let balanced_power = self.calculate_gaming_power_consumption(
            gpu_config,
            &balanced_workload,
            120.0,
            110.0
        )?;

        scenarios.push(PowerScenarioAnalysis {
            scenario_name: "Balanced 1440p".to_string(),
            power_consumption: balanced_power.clone(),
            thermal_impact: balanced_power.total_power_watts * 0.85,
            sustainability_score: 70.0,
            cost_per_hour: (balanced_power.total_power_watts * self.electricity_rate_per_kwh) /
            1000.0,
            carbon_footprint_kg: (balanced_power.total_power_watts *
                self.carbon_intensity_kg_per_kwh) /
            1000.0,
        });

        Ok(scenarios)
    }
}

impl Default for GamingPowerEngine {
    fn default() -> Self {
        let mut engine = Self::new();
        engine.initialize();
        engine
    }
}
