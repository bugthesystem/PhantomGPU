//! Real-Time Thermal Modeling for GPU Performance Degradation
//!
//! This module implements sophisticated thermal modeling that affects GPU performance:
//! - Temperature tracking based on workload intensity
//! - Thermal throttling when temperature limits are exceeded
//! - Performance scaling curves based on temperature
//! - Cooling simulation with thermal time constants
//! - Architecture-specific thermal characteristics

use serde::{ Deserialize, Serialize };
use std::collections::HashMap;
use std::time::{ Duration, Instant };

/// Thermal state of a GPU at a specific moment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalState {
    pub current_temp_celsius: f64,
    pub ambient_temp_celsius: f64,
    pub thermal_load: f64, // 0.0 to 1.0 - current thermal load
    pub throttling_active: bool,
    pub performance_multiplier: f64, // 1.0 = no throttling, lower = throttled
    #[serde(skip)]
    pub last_update: Option<Instant>,
}

/// Thermal profile for a specific GPU architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalProfile {
    pub gpu_name: String,
    pub tdp_watts: f64,
    pub base_temp_celsius: f64,
    pub throttle_temp_celsius: f64,
    pub shutdown_temp_celsius: f64,
    pub thermal_resistance: f64, // °C/W - how much temp rises per watt
    pub thermal_capacitance: f64, // J/°C - thermal mass
    pub cooling_time_constant: f64, // seconds - how fast it cools
    pub performance_curve: Vec<(f64, f64)>, // (temp, performance_multiplier)
}

/// Real-time thermal modeling engine
pub struct ThermalModelingEngine {
    thermal_states: HashMap<String, ThermalState>,
    thermal_profiles: HashMap<String, ThermalProfile>,
    ambient_temperature: f64,
    thermal_update_interval: Duration,
}

impl ThermalModelingEngine {
    /// Create a new thermal modeling engine
    pub fn new(ambient_temp: f64) -> Self {
        let mut engine = Self {
            thermal_states: HashMap::new(),
            thermal_profiles: HashMap::new(),
            ambient_temperature: ambient_temp,
            thermal_update_interval: Duration::from_millis(100), // Update every 100ms
        };

        // Load predefined thermal profiles
        engine.load_thermal_profiles();
        engine
    }

    /// Load thermal profiles for various GPU architectures
    fn load_thermal_profiles(&mut self) {
        // A100 Thermal Profile
        self.thermal_profiles.insert("A100".to_string(), ThermalProfile {
            gpu_name: "A100".to_string(),
            tdp_watts: 400.0,
            base_temp_celsius: 30.0,
            throttle_temp_celsius: 80.0,
            shutdown_temp_celsius: 90.0,
            thermal_resistance: 0.12, // °C/W
            thermal_capacitance: 2000.0, // J/°C
            cooling_time_constant: 45.0, // seconds
            performance_curve: vec![
                (30.0, 1.0), // Base performance
                (50.0, 1.0), // Still full performance
                (70.0, 0.95), // Slight performance drop
                (80.0, 0.85), // Throttling begins
                (85.0, 0.7), // Heavy throttling
                (90.0, 0.4) // Emergency throttling
            ],
        });

        // H100 Thermal Profile
        self.thermal_profiles.insert("H100".to_string(), ThermalProfile {
            gpu_name: "H100".to_string(),
            tdp_watts: 700.0,
            base_temp_celsius: 32.0,
            throttle_temp_celsius: 85.0,
            shutdown_temp_celsius: 95.0,
            thermal_resistance: 0.08, // Better cooling than A100
            thermal_capacitance: 2500.0, // J/°C
            cooling_time_constant: 35.0, // Faster cooling
            performance_curve: vec![
                (32.0, 1.0), // Base performance
                (55.0, 1.0), // Still full performance
                (75.0, 0.98), // Minimal performance drop
                (85.0, 0.9), // Throttling begins
                (90.0, 0.75), // Heavy throttling
                (95.0, 0.5) // Emergency throttling
            ],
        });

        // RTX 4090 Thermal Profile
        self.thermal_profiles.insert("RTX 4090".to_string(), ThermalProfile {
            gpu_name: "RTX 4090".to_string(),
            tdp_watts: 450.0,
            base_temp_celsius: 25.0,
            throttle_temp_celsius: 83.0,
            shutdown_temp_celsius: 93.0,
            thermal_resistance: 0.15, // °C/W
            thermal_capacitance: 1800.0, // J/°C
            cooling_time_constant: 40.0, // seconds
            performance_curve: vec![
                (25.0, 1.0), // Base performance
                (45.0, 1.0), // Still full performance
                (65.0, 0.97), // Slight performance drop
                (83.0, 0.87), // Throttling begins
                (88.0, 0.72), // Heavy throttling
                (93.0, 0.45) // Emergency throttling
            ],
        });

        // RTX 5090 Thermal Profile
        self.thermal_profiles.insert("RTX 5090".to_string(), ThermalProfile {
            gpu_name: "RTX 5090".to_string(),
            tdp_watts: 600.0,
            base_temp_celsius: 28.0,
            throttle_temp_celsius: 87.0,
            shutdown_temp_celsius: 97.0,
            thermal_resistance: 0.1, // Better cooling than RTX 4090
            thermal_capacitance: 2200.0, // J/°C
            cooling_time_constant: 30.0, // Faster cooling
            performance_curve: vec![
                (28.0, 1.0), // Base performance
                (50.0, 1.0), // Still full performance
                (70.0, 0.98), // Minimal performance drop
                (87.0, 0.92), // Throttling begins
                (92.0, 0.8), // Heavy throttling
                (97.0, 0.6) // Emergency throttling
            ],
        });

        // Tesla V100 Thermal Profile
        self.thermal_profiles.insert("Tesla V100".to_string(), ThermalProfile {
            gpu_name: "Tesla V100".to_string(),
            tdp_watts: 300.0,
            base_temp_celsius: 35.0,
            throttle_temp_celsius: 80.0,
            shutdown_temp_celsius: 90.0,
            thermal_resistance: 0.18, // °C/W
            thermal_capacitance: 1500.0, // J/°C
            cooling_time_constant: 55.0, // Slower cooling
            performance_curve: vec![
                (35.0, 1.0), // Base performance
                (55.0, 1.0), // Still full performance
                (70.0, 0.93), // Performance drop
                (80.0, 0.82), // Throttling begins
                (85.0, 0.65), // Heavy throttling
                (90.0, 0.35) // Emergency throttling
            ],
        });
    }

    /// Initialize thermal state for a GPU
    pub fn initialize_gpu(&mut self, gpu_name: &str) {
        let profile = self.thermal_profiles.get(gpu_name);

        let initial_temp = if let Some(profile) = profile {
            profile.base_temp_celsius
        } else {
            self.ambient_temperature + 10.0 // Default offset
        };

        self.thermal_states.insert(gpu_name.to_string(), ThermalState {
            current_temp_celsius: initial_temp,
            ambient_temp_celsius: self.ambient_temperature,
            thermal_load: 0.0,
            throttling_active: false,
            performance_multiplier: 1.0,
            last_update: Some(Instant::now()),
        });
    }

    /// Update thermal state based on workload intensity
    pub fn update_thermal_state(
        &mut self,
        gpu_name: &str,
        workload_intensity: f64
    ) -> Option<ThermalState> {
        let profile = self.thermal_profiles.get(gpu_name)?.clone();
        let state = self.thermal_states.get_mut(gpu_name)?;

        let now = Instant::now();
        let dt = if let Some(last_update) = state.last_update {
            now.duration_since(last_update).as_secs_f64()
        } else {
            0.1 // Default dt
        };

        // Calculate power dissipation based on workload
        let power_dissipation = profile.tdp_watts * workload_intensity;

        // Temperature rise due to workload
        let temp_rise = power_dissipation * profile.thermal_resistance;

        // Thermal dynamics simulation
        let target_temp = self.ambient_temperature + temp_rise;
        let temp_diff = target_temp - state.current_temp_celsius;

        // Exponential approach to target temperature
        let temp_change = temp_diff * (1.0 - (-dt / profile.cooling_time_constant).exp());
        state.current_temp_celsius += temp_change;

        // Update thermal load
        state.thermal_load = workload_intensity;

        // Calculate performance multiplier based on temperature
        let performance_curve = profile.performance_curve.clone();
        state.performance_multiplier = Self::calculate_performance_multiplier_static(
            state.current_temp_celsius,
            &performance_curve
        );

        // Check if throttling is active
        state.throttling_active = state.current_temp_celsius >= profile.throttle_temp_celsius;

        state.last_update = Some(now);

        Some(state.clone())
    }

    /// Calculate performance multiplier based on temperature and performance curve
    fn calculate_performance_multiplier_static(temp: f64, curve: &[(f64, f64)]) -> f64 {
        if curve.is_empty() {
            return 1.0;
        }

        // Find the appropriate range in the curve
        for i in 0..curve.len() - 1 {
            let (temp1, perf1) = curve[i];
            let (temp2, perf2) = curve[i + 1];

            if temp >= temp1 && temp <= temp2 {
                // Linear interpolation
                let ratio = (temp - temp1) / (temp2 - temp1);
                return perf1 + ratio * (perf2 - perf1);
            }
        }

        // Temperature is outside the curve range
        if temp < curve[0].0 {
            curve[0].1 // Return first performance value
        } else {
            curve[curve.len() - 1].1 // Return last performance value
        }
    }

    /// Get current thermal state for a GPU
    pub fn get_thermal_state(&self, gpu_name: &str) -> Option<&ThermalState> {
        self.thermal_states.get(gpu_name)
    }

    /// Get thermal profile for a GPU
    pub fn get_thermal_profile(&self, gpu_name: &str) -> Option<&ThermalProfile> {
        self.thermal_profiles.get(gpu_name)
    }

    /// Simulate cooling period (no workload)
    pub fn simulate_cooling(
        &mut self,
        gpu_name: &str,
        duration_seconds: f64
    ) -> Option<ThermalState> {
        self.update_thermal_state(gpu_name, 0.0)?;

        let profile = self.thermal_profiles.get(gpu_name)?;
        let state = self.thermal_states.get_mut(gpu_name)?;

        // Fast cooling simulation
        let target_temp = self.ambient_temperature + 5.0; // Idle temperature
        let temp_diff = target_temp - state.current_temp_celsius;
        let temp_change =
            temp_diff * (1.0 - (-duration_seconds / profile.cooling_time_constant).exp());

        state.current_temp_celsius += temp_change;
        state.thermal_load = 0.0;
        state.throttling_active = false;
        state.performance_multiplier = 1.0;

        Some(state.clone())
    }

    /// Set ambient temperature
    pub fn set_ambient_temperature(&mut self, temp: f64) {
        self.ambient_temperature = temp;

        // Update all thermal states
        for state in self.thermal_states.values_mut() {
            state.ambient_temp_celsius = temp;
        }
    }

    /// Get thermal summary for all GPUs
    pub fn get_thermal_summary(&self) -> HashMap<String, ThermalSummary> {
        let mut summary = HashMap::new();

        for (gpu_name, state) in &self.thermal_states {
            if let Some(profile) = self.thermal_profiles.get(gpu_name) {
                summary.insert(gpu_name.clone(), ThermalSummary {
                    gpu_name: gpu_name.clone(),
                    current_temp: state.current_temp_celsius,
                    throttle_temp: profile.throttle_temp_celsius,
                    thermal_load: state.thermal_load,
                    performance_multiplier: state.performance_multiplier,
                    throttling_active: state.throttling_active,
                    thermal_headroom: profile.throttle_temp_celsius - state.current_temp_celsius,
                });
            }
        }

        summary
    }
}

/// Summary of thermal state for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalSummary {
    pub gpu_name: String,
    pub current_temp: f64,
    pub throttle_temp: f64,
    pub thermal_load: f64,
    pub performance_multiplier: f64,
    pub throttling_active: bool,
    pub thermal_headroom: f64,
}

/// Thermal impact on performance result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalPerformanceResult {
    pub base_throughput: f64,
    pub thermal_adjusted_throughput: f64,
    pub thermal_impact: f64, // Percentage impact (negative for throttling)
    pub thermal_state: ThermalState,
}

impl ThermalPerformanceResult {
    pub fn new(base_throughput: f64, thermal_state: ThermalState) -> Self {
        let thermal_adjusted_throughput = base_throughput * thermal_state.performance_multiplier;
        let thermal_impact =
            ((thermal_adjusted_throughput - base_throughput) / base_throughput) * 100.0;

        Self {
            base_throughput,
            thermal_adjusted_throughput,
            thermal_impact,
            thermal_state,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_modeling_initialization() {
        let mut engine = ThermalModelingEngine::new(25.0);
        engine.initialize_gpu("A100");

        let state = engine.get_thermal_state("A100").unwrap();
        assert_eq!(state.current_temp_celsius, 30.0);
        assert_eq!(state.performance_multiplier, 1.0);
        assert!(!state.throttling_active);
    }

    #[test]
    fn test_thermal_throttling() {
        let mut engine = ThermalModelingEngine::new(25.0);
        engine.initialize_gpu("A100");

        // Simulate high workload
        let state = engine.update_thermal_state("A100", 1.0).unwrap();

        // Should heat up and potentially throttle
        assert!(state.current_temp_celsius > 30.0);
        assert!(state.thermal_load > 0.0);
    }

    #[test]
    fn test_performance_multiplier_calculation() {
        let engine = ThermalModelingEngine::new(25.0);
        let curve = vec![(30.0, 1.0), (80.0, 0.85), (90.0, 0.4)];

        // Test interpolation
        let perf = ThermalModelingEngine::calculate_performance_multiplier_static(55.0, &curve);
        assert!(perf > 0.85 && perf <= 1.0);
    }
}
