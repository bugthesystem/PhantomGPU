use crate::gpu_config::GpuModel;
use crate::gaming_performance::{ GamingWorkload, GamingPerformance };
use crate::errors::PhantomGpuError;
use serde::{ Deserialize, Serialize };
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingThermalProfile {
    pub workload_type: ThermalWorkloadType,
    pub temperature_pattern: TemperaturePattern,
    pub thermal_spikes: Vec<ThermalSpike>,
    pub cooling_efficiency: f64,
    pub thermal_mass: f64,
    pub ambient_sensitivity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalWorkloadType {
    Gaming,
    Compute,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperaturePattern {
    pub base_temperature: f64,
    pub peak_temperature: f64,
    pub temperature_variance: f64,
    pub thermal_cycles_per_minute: f64,
    pub steady_state_time: f64, // seconds to reach steady state
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalSpike {
    pub trigger: SpikeTrigger,
    pub temperature_increase: f64,
    pub duration_seconds: f64,
    pub recovery_time_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpikeTrigger {
    SceneTransition,
    ExplosionEffect,
    RayTracingIntensive,
    ShaderCompilation,
    AssetStreaming,
    MultiplayerJoin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingThermalState {
    pub current_temperature: f64,
    pub thermal_load: f64,
    pub cooling_demand: f64,
    pub thermal_throttling: bool,
    pub performance_scaling: f64,
    pub fan_speed_percentage: f64,
    pub thermal_history: Vec<ThermalDataPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalDataPoint {
    pub timestamp: f64,
    pub temperature: f64,
    pub power_draw: f64,
    pub fan_speed: f64,
    pub performance_level: f64,
}

pub struct GamingThermalEngine {
    thermal_profiles: HashMap<String, GamingThermalProfile>,
    gpu_thermal_specs: HashMap<String, GPUThermalSpecs>,
}

#[derive(Debug, Clone)]
struct GPUThermalSpecs {
    tj_max: f64, // Maximum junction temperature
    throttle_temp: f64, // Throttling start temperature
    target_temp: f64, // Target operating temperature
    thermal_resistance: f64, // °C/W
    thermal_capacitance: f64, // J/°C
    cooling_curves: CoolingCurves,
}

#[derive(Debug, Clone)]
struct CoolingCurves {
    fan_curve: Vec<(f64, f64)>, // (temperature, fan_speed_percentage)
    thermal_curve: Vec<(f64, f64)>, // (power_percentage, temp_rise)
}

impl GamingThermalEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            thermal_profiles: HashMap::new(),
            gpu_thermal_specs: HashMap::new(),
        };

        engine.initialize_thermal_profiles();
        engine.initialize_gpu_thermal_specs();
        engine
    }

    fn initialize_thermal_profiles(&mut self) {
        // Gaming thermal profile - variable load with spikes
        self.thermal_profiles.insert("Gaming".to_string(), GamingThermalProfile {
            workload_type: ThermalWorkloadType::Gaming,
            temperature_pattern: TemperaturePattern {
                base_temperature: 35.0, // Lower idle temperature
                peak_temperature: 75.0, // Peak gaming temperature
                temperature_variance: 15.0, // High variance due to scene changes
                thermal_cycles_per_minute: 3.0, // Frequent temperature changes
                steady_state_time: 45.0, // Longer time to steady state
            },
            thermal_spikes: vec![
                ThermalSpike {
                    trigger: SpikeTrigger::SceneTransition,
                    temperature_increase: 5.0,
                    duration_seconds: 2.0,
                    recovery_time_seconds: 8.0,
                },
                ThermalSpike {
                    trigger: SpikeTrigger::ExplosionEffect,
                    temperature_increase: 8.0,
                    duration_seconds: 1.0,
                    recovery_time_seconds: 5.0,
                },
                ThermalSpike {
                    trigger: SpikeTrigger::RayTracingIntensive,
                    temperature_increase: 12.0,
                    duration_seconds: 5.0,
                    recovery_time_seconds: 15.0,
                },
                ThermalSpike {
                    trigger: SpikeTrigger::ShaderCompilation,
                    temperature_increase: 15.0,
                    duration_seconds: 3.0,
                    recovery_time_seconds: 10.0,
                }
            ],
            cooling_efficiency: 0.85, // Good cooling efficiency
            thermal_mass: 1.0, // Standard thermal mass
            ambient_sensitivity: 1.2, // Higher sensitivity to ambient temperature
        });

        // Compute thermal profile - sustained high load
        self.thermal_profiles.insert("Compute".to_string(), GamingThermalProfile {
            workload_type: ThermalWorkloadType::Compute,
            temperature_pattern: TemperaturePattern {
                base_temperature: 45.0, // Higher base temperature
                peak_temperature: 85.0, // Higher sustained temperature
                temperature_variance: 3.0, // Low variance - steady load
                thermal_cycles_per_minute: 0.5, // Infrequent changes
                steady_state_time: 120.0, // Quick to steady state
            },
            thermal_spikes: vec![ThermalSpike {
                trigger: SpikeTrigger::AssetStreaming,
                temperature_increase: 3.0,
                duration_seconds: 10.0,
                recovery_time_seconds: 30.0,
            }],
            cooling_efficiency: 0.95, // Better cooling efficiency at sustained load
            thermal_mass: 1.0,
            ambient_sensitivity: 1.0, // Standard sensitivity
        });

        // Mixed workload profile
        self.thermal_profiles.insert("Mixed".to_string(), GamingThermalProfile {
            workload_type: ThermalWorkloadType::Mixed,
            temperature_pattern: TemperaturePattern {
                base_temperature: 40.0,
                peak_temperature: 80.0,
                temperature_variance: 8.0,
                thermal_cycles_per_minute: 2.0,
                steady_state_time: 60.0,
            },
            thermal_spikes: vec![ThermalSpike {
                trigger: SpikeTrigger::MultiplayerJoin,
                temperature_increase: 7.0,
                duration_seconds: 4.0,
                recovery_time_seconds: 12.0,
            }],
            cooling_efficiency: 0.9,
            thermal_mass: 1.0,
            ambient_sensitivity: 1.1,
        });
    }

    fn initialize_gpu_thermal_specs(&mut self) {
        // RTX 4090 thermal specs
        self.gpu_thermal_specs.insert("RTX 4090".to_string(), GPUThermalSpecs {
            tj_max: 90.0,
            throttle_temp: 83.0,
            target_temp: 75.0,
            thermal_resistance: 0.12, // °C/W
            thermal_capacitance: 500.0, // J/°C
            cooling_curves: CoolingCurves {
                fan_curve: vec![
                    (30.0, 0.0), // 0% fan below 30°C
                    (50.0, 30.0), // 30% fan at 50°C
                    (70.0, 60.0), // 60% fan at 70°C
                    (80.0, 90.0), // 90% fan at 80°C
                    (85.0, 100.0) // 100% fan at 85°C
                ],
                thermal_curve: vec![
                    (0.0, 0.0),
                    (0.3, 15.0), // 15°C rise at 30% power
                    (0.6, 35.0), // 35°C rise at 60% power
                    (1.0, 60.0) // 60°C rise at 100% power
                ],
            },
        });

        // RTX 5090 thermal specs (improved cooling)
        self.gpu_thermal_specs.insert("RTX 5090".to_string(), GPUThermalSpecs {
            tj_max: 92.0,
            throttle_temp: 85.0,
            target_temp: 77.0,
            thermal_resistance: 0.1, // Better thermal design
            thermal_capacitance: 600.0,
            cooling_curves: CoolingCurves {
                fan_curve: vec![
                    (30.0, 0.0),
                    (50.0, 25.0),
                    (70.0, 55.0),
                    (80.0, 85.0),
                    (87.0, 100.0)
                ],
                thermal_curve: vec![(0.0, 0.0), (0.3, 12.0), (0.6, 30.0), (1.0, 55.0)],
            },
        });

        // RTX 4080 thermal specs
        self.gpu_thermal_specs.insert("RTX 4080".to_string(), GPUThermalSpecs {
            tj_max: 90.0,
            throttle_temp: 83.0,
            target_temp: 75.0,
            thermal_resistance: 0.15,
            thermal_capacitance: 400.0,
            cooling_curves: CoolingCurves {
                fan_curve: vec![
                    (30.0, 0.0),
                    (50.0, 35.0),
                    (70.0, 65.0),
                    (80.0, 95.0),
                    (85.0, 100.0)
                ],
                thermal_curve: vec![(0.0, 0.0), (0.3, 18.0), (0.6, 40.0), (1.0, 65.0)],
            },
        });
    }

    pub fn simulate_gaming_thermal_session(
        &self,
        gpu_config: &GpuModel,
        workload: &GamingWorkload,
        session_duration_minutes: f64,
        ambient_temp: f64
    ) -> Result<GamingThermalState, PhantomGpuError> {
        let profile = self.thermal_profiles
            .get("Gaming")
            .ok_or_else(||
                PhantomGpuError::ModelLoadError("Gaming profile not found".to_string())
            )?;

        let gpu_specs = self.gpu_thermal_specs
            .get(&gpu_config.name)
            .ok_or_else(|| PhantomGpuError::GpuNotFound { gpu_name: gpu_config.name.clone() })?;

        let mut thermal_state = GamingThermalState {
            current_temperature: ambient_temp + 10.0, // Start slightly above ambient
            thermal_load: 0.0,
            cooling_demand: 0.0,
            thermal_throttling: false,
            performance_scaling: 1.0,
            fan_speed_percentage: 0.0,
            thermal_history: Vec::new(),
        };

        let time_steps = (session_duration_minutes * 60.0) as usize; // 1-second intervals

        for step in 0..time_steps {
            let time_seconds = step as f64;

            // Calculate gaming load pattern (varies with time)
            let gaming_load = self.calculate_gaming_load_pattern(
                time_seconds,
                workload,
                session_duration_minutes * 60.0
            );

            // Apply thermal spikes based on gaming events
            let spike_temp_increase = self.calculate_thermal_spikes(
                time_seconds,
                workload,
                profile
            );

            // Update thermal state
            thermal_state = self.update_thermal_state(
                thermal_state,
                gaming_load,
                spike_temp_increase,
                ambient_temp,
                gpu_specs,
                profile
            );

            // Record thermal history every 10 seconds
            if step % 10 == 0 {
                thermal_state.thermal_history.push(ThermalDataPoint {
                    timestamp: time_seconds,
                    temperature: thermal_state.current_temperature,
                    power_draw: gaming_load * 450.0, // Assume 450W max for gaming GPU
                    fan_speed: thermal_state.fan_speed_percentage,
                    performance_level: thermal_state.performance_scaling,
                });
            }
        }

        Ok(thermal_state)
    }

    fn calculate_gaming_load_pattern(
        &self,
        time_seconds: f64,
        workload: &GamingWorkload,
        total_duration: f64
    ) -> f64 {
        let base_load = 0.4; // 40% base load
        let scene_variance = 0.4; // Up to 40% additional load

        // Create realistic gaming load pattern
        let scene_cycle = (time_seconds / 30.0) * 2.0 * std::f64::consts::PI; // 30-second cycles
        let action_intensity = workload.scene_complexity;

        // Gaming sessions typically start intense, then settle
        let session_factor = if time_seconds < 300.0 {
            1.0 - (time_seconds / 300.0) * 0.2 // Reduce by 20% over first 5 minutes
        } else {
            0.8 + 0.2 * (time_seconds / total_duration) // Gradually increase again
        };

        let scene_load = scene_cycle.sin().abs() * action_intensity * scene_variance;

        ((base_load + scene_load) * session_factor).min(1.0)
    }

    fn calculate_thermal_spikes(
        &self,
        time_seconds: f64,
        workload: &GamingWorkload,
        profile: &GamingThermalProfile
    ) -> f64 {
        let mut total_spike = 0.0;

        for spike in &profile.thermal_spikes {
            let spike_probability = match spike.trigger {
                SpikeTrigger::SceneTransition => 0.05, // 5% chance per second
                SpikeTrigger::ExplosionEffect => 0.02, // 2% chance per second
                SpikeTrigger::RayTracingIntensive => if workload.ray_tracing { 0.03 } else { 0.0 }
                SpikeTrigger::ShaderCompilation => 0.01,
                SpikeTrigger::AssetStreaming => 0.015,
                SpikeTrigger::MultiplayerJoin => 0.005,
            };

            // Simulate random thermal spikes
            let spike_hash = (time_seconds * 1000.0) as u64;
            let spike_chance = ((spike_hash % 1000) as f64) / 1000.0;

            if spike_chance < spike_probability {
                let spike_intensity = workload.scene_complexity * spike.temperature_increase;
                total_spike += spike_intensity;
            }
        }

        total_spike
    }

    fn update_thermal_state(
        &self,
        mut state: GamingThermalState,
        gaming_load: f64,
        spike_temp_increase: f64,
        ambient_temp: f64,
        gpu_specs: &GPUThermalSpecs,
        profile: &GamingThermalProfile
    ) -> GamingThermalState {
        // Calculate power dissipation
        let power_draw = gaming_load * 450.0; // Assume 450W max for gaming GPU

        // Apply thermal model
        let target_temp = ambient_temp + power_draw * gpu_specs.thermal_resistance;
        let temp_diff = target_temp - state.current_temperature;
        let time_constant = gpu_specs.thermal_resistance * gpu_specs.thermal_capacitance;

        // Exponential thermal response
        let temp_change = temp_diff * (1.0 - (-1.0 / time_constant).exp());
        state.current_temperature += temp_change;

        // Apply thermal spikes
        state.current_temperature += spike_temp_increase;

        // Calculate fan speed
        state.fan_speed_percentage = self.calculate_fan_speed(
            state.current_temperature,
            &gpu_specs.cooling_curves.fan_curve
        );

        // Check for thermal throttling
        if state.current_temperature > gpu_specs.throttle_temp {
            state.thermal_throttling = true;
            let throttle_factor =
                (gpu_specs.tj_max - state.current_temperature) /
                (gpu_specs.tj_max - gpu_specs.throttle_temp);
            state.performance_scaling = throttle_factor.max(0.4); // Minimum 40% performance
        } else {
            state.thermal_throttling = false;
            state.performance_scaling = 1.0;
        }

        state.thermal_load = gaming_load;
        state.cooling_demand = state.fan_speed_percentage / 100.0;

        state
    }

    fn calculate_fan_speed(&self, temperature: f64, fan_curve: &[(f64, f64)]) -> f64 {
        if temperature <= fan_curve[0].0 {
            return fan_curve[0].1;
        }

        for i in 1..fan_curve.len() {
            if temperature <= fan_curve[i].0 {
                let t1 = fan_curve[i - 1].0;
                let f1 = fan_curve[i - 1].1;
                let t2 = fan_curve[i].0;
                let f2 = fan_curve[i].1;

                // Linear interpolation
                let ratio = (temperature - t1) / (t2 - t1);
                return f1 + ratio * (f2 - f1);
            }
        }

        fan_curve.last().unwrap().1
    }

    pub fn compare_gaming_vs_compute_thermal(
        &self,
        gpu_config: &GpuModel,
        duration_minutes: f64,
        ambient_temp: f64
    ) -> Result<ThermalComparison, PhantomGpuError> {
        let gaming_workload = GamingWorkload {
            game_name: "Cyberpunk 2077".to_string(),
            resolution: (2560, 1440),
            ray_tracing: true,
            dlss_mode: crate::gaming_performance::DLSSMode::Quality,
            fsr_mode: crate::gaming_performance::FSRMode::Off,
            target_fps: 60.0,
            scene_complexity: 0.8,
            graphics_settings: Default::default(),
        };

        let gaming_thermal = self.simulate_gaming_thermal_session(
            gpu_config,
            &gaming_workload,
            duration_minutes,
            ambient_temp
        )?;

        // Simulate compute thermal (simplified)
        let compute_thermal = self.simulate_compute_thermal_session(
            gpu_config,
            duration_minutes,
            ambient_temp
        )?;

        Ok(ThermalComparison {
            temperature_difference: compute_thermal.current_temperature -
            gaming_thermal.current_temperature,
            power_difference: compute_thermal.thermal_load - gaming_thermal.thermal_load,
            gaming_thermal,
            compute_thermal,
        })
    }

    fn simulate_compute_thermal_session(
        &self,
        gpu_config: &GpuModel,
        duration_minutes: f64,
        ambient_temp: f64
    ) -> Result<GamingThermalState, PhantomGpuError> {
        let profile = self.thermal_profiles
            .get("Compute")
            .ok_or_else(||
                PhantomGpuError::ModelLoadError("Compute profile not found".to_string())
            )?;

        let gpu_specs = self.gpu_thermal_specs
            .get(&gpu_config.name)
            .ok_or_else(|| PhantomGpuError::GpuNotFound { gpu_name: gpu_config.name.clone() })?;

        // Compute workload: steady 90% load
        let compute_load = 0.9;
        let target_temp = ambient_temp + compute_load * 450.0 * gpu_specs.thermal_resistance;

        Ok(GamingThermalState {
            current_temperature: target_temp,
            thermal_load: compute_load,
            cooling_demand: 0.9,
            thermal_throttling: target_temp > gpu_specs.throttle_temp,
            performance_scaling: if target_temp > gpu_specs.throttle_temp {
                0.8
            } else {
                1.0
            },
            fan_speed_percentage: 85.0,
            thermal_history: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalComparison {
    pub gaming_thermal: GamingThermalState,
    pub compute_thermal: GamingThermalState,
    pub temperature_difference: f64,
    pub power_difference: f64,
}

impl Default for crate::gaming_performance::GraphicsSettings {
    fn default() -> Self {
        Self {
            texture_quality: crate::gaming_performance::Quality::High,
            shadow_quality: crate::gaming_performance::Quality::High,
            anti_aliasing: crate::gaming_performance::AntiAliasing::TAA,
            anisotropic_filtering: 16,
            variable_rate_shading: false,
            mesh_shaders: false,
        }
    }
}

impl Default for GamingThermalEngine {
    fn default() -> Self {
        Self::new()
    }
}
