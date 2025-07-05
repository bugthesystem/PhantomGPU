use serde::{ Deserialize, Serialize };
use std::collections::HashMap;
use std::fs;
use tracing::{ info, warn, error };

/// Load and validate hardware profiles from a TOML file
pub fn load_hardware_profiles(file_path: &str) -> Result<RealHardwareProfiles, String> {
    let loader = HardwareProfileLoader::new(file_path).map_err(|e| e.to_string())?;

    let mut profiles_map = HashMap::new();
    for name in loader.list_profiles() {
        if let Some(profile) = loader.get_profile(&name) {
            profiles_map.insert(name, profile.clone());
        }
    }

    Ok(RealHardwareProfiles { profiles: profiles_map })
}

/// TOML-configurable thermal characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConfig {
    pub tdp_watts: f64,
    pub base_clock_mhz: f64,
    pub boost_clock_mhz: f64,
    pub throttle_temp_celsius: f64,
    pub thermal_factor_sustained: f64,
}

/// TOML-configurable memory hierarchy characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub l1_cache_kb: f64,
    pub l2_cache_mb: f64,
    pub memory_channels: u32,
    pub cache_hit_ratio: f64,
    pub coalescing_efficiency: f64,
}

/// TOML-configurable architecture details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    pub cuda_cores: u32,
    pub tensor_cores: u32,
    pub rt_cores: u32,
    pub streaming_multiprocessors: u32,
    pub memory_bus_width: u32,
}

/// TOML-configurable model type performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceConfig {
    pub batch_scaling_curve: Vec<f64>, // batch sizes 1,2,4,8,16,32,64,128
    pub memory_efficiency: f64,
    pub tensor_core_utilization: f64,
    pub architecture_multiplier: f64,
}

/// TOML-configurable precision multipliers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionConfig {
    pub fp16_multiplier: f64,
    pub int8_multiplier: f64,
    pub int4_multiplier: f64,
}

/// TOML-configurable model performance for different types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTypePerformanceConfig {
    pub cnn: ModelPerformanceConfig,
    pub transformer: ModelPerformanceConfig,
    pub rnn: ModelPerformanceConfig,
}

/// Complete hardware profile loaded from TOML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfileConfig {
    pub name: String,
    pub thermal: ThermalConfig,
    pub memory: MemoryConfig,
    pub architecture: ArchitectureConfig,
    pub model_performance: ModelTypePerformanceConfig,
    pub precision: PrecisionConfig,
}

/// Root TOML configuration structure
#[derive(Debug, Serialize, Deserialize)]
pub struct RealHardwareProfiles {
    pub profiles: HashMap<String, HardwareProfileConfig>,
}

/// Hardware profile loader with caching and error handling
pub struct HardwareProfileLoader {
    profiles: HashMap<String, HardwareProfileConfig>,
    config_path: String,
}

impl HardwareProfileLoader {
    /// Create a new profile loader with the specified config file path
    pub fn new(config_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut loader = Self {
            profiles: HashMap::new(),
            config_path: config_path.to_string(),
        };

        loader.load_profiles()?;
        Ok(loader)
    }

    /// Load hardware profiles from TOML configuration file
    fn load_profiles(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Loading real hardware profiles from: {}", self.config_path);

        // Read TOML file
        let config_content = match fs::read_to_string(&self.config_path) {
            Ok(content) => content,
            Err(e) => {
                error!("Failed to read hardware profiles config: {}", e);
                warn!("Falling back to default hardcoded profiles");
                return self.load_default_profiles();
            }
        };

        // Parse TOML
        let config: RealHardwareProfiles = match toml::from_str(&config_content) {
            Ok(config) => config,
            Err(e) => {
                error!("Failed to parse hardware profiles TOML: {}", e);
                warn!("Falling back to default hardcoded profiles");
                return self.load_default_profiles();
            }
        };

        // Store profiles
        self.profiles = config.profiles;

        info!(
            "Successfully loaded {} hardware profiles: {}",
            self.profiles.len(),
            self.profiles.keys().cloned().collect::<Vec<_>>().join(", ")
        );

        Ok(())
    }

    /// Fallback to hardcoded profiles if TOML loading fails
    fn load_default_profiles(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        warn!("Using minimal hardcoded hardware profiles as fallback");

        // Create minimal V100 profile as fallback
        let v100_profile = HardwareProfileConfig {
            name: "Tesla V100".to_string(),
            thermal: ThermalConfig {
                tdp_watts: 300.0,
                base_clock_mhz: 1245.0,
                boost_clock_mhz: 1380.0,
                throttle_temp_celsius: 83.0,
                thermal_factor_sustained: 1.0,
            },
            memory: MemoryConfig {
                l1_cache_kb: 128.0,
                l2_cache_mb: 6.0,
                memory_channels: 4,
                cache_hit_ratio: 0.85,
                coalescing_efficiency: 0.8,
            },
            architecture: ArchitectureConfig {
                cuda_cores: 5120,
                tensor_cores: 640,
                rt_cores: 0,
                streaming_multiprocessors: 80,
                memory_bus_width: 4096,
            },
            model_performance: ModelTypePerformanceConfig {
                cnn: ModelPerformanceConfig {
                    batch_scaling_curve: vec![1.0, 0.85, 0.72, 0.6, 0.5, 0.42, 0.35, 0.3],
                    memory_efficiency: 0.75,
                    tensor_core_utilization: 0.65,
                    architecture_multiplier: 1.25,
                },
                transformer: ModelPerformanceConfig {
                    batch_scaling_curve: vec![1.0, 0.82, 0.7, 0.6, 0.5, 0.42, 0.35, 0.28],
                    memory_efficiency: 0.7,
                    tensor_core_utilization: 0.8,
                    architecture_multiplier: 1.3,
                },
                rnn: ModelPerformanceConfig {
                    batch_scaling_curve: vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.42, 0.35],
                    memory_efficiency: 0.65,
                    tensor_core_utilization: 0.45,
                    architecture_multiplier: 1.15,
                },
            },
            precision: PrecisionConfig {
                fp16_multiplier: 1.8,
                int8_multiplier: 2.2,
                int4_multiplier: 3.0,
            },
        };

        self.profiles.insert("v100".to_string(), v100_profile);

        Ok(())
    }

    /// Get a hardware profile by GPU identifier
    pub fn get_profile(&self, gpu_id: &str) -> Option<&HardwareProfileConfig> {
        self.profiles.get(gpu_id)
    }

    /// List all available hardware profile names
    pub fn list_profiles(&self) -> Vec<String> {
        self.profiles.keys().cloned().collect()
    }

    /// Get the count of loaded profiles
    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }

    /// Reload profiles from the configuration file
    pub fn reload_profiles(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Reloading hardware profiles from configuration");
        self.profiles.clear();
        self.load_profiles()
    }

    /// Validate that a profile has sensible values
    pub fn validate_profile(&self, gpu_id: &str) -> Result<(), String> {
        let profile = self
            .get_profile(gpu_id)
            .ok_or_else(|| format!("Profile '{}' not found", gpu_id))?;

        // Validate thermal characteristics
        if profile.thermal.tdp_watts <= 0.0 || profile.thermal.tdp_watts > 1000.0 {
            return Err(format!("Invalid TDP: {} watts", profile.thermal.tdp_watts));
        }

        if
            profile.thermal.base_clock_mhz <= 0.0 ||
            profile.thermal.boost_clock_mhz < profile.thermal.base_clock_mhz
        {
            return Err(
                format!(
                    "Invalid clock speeds: base {} MHz, boost {} MHz",
                    profile.thermal.base_clock_mhz,
                    profile.thermal.boost_clock_mhz
                )
            );
        }

        // Validate precision multipliers
        if profile.precision.fp16_multiplier < 1.0 || profile.precision.fp16_multiplier > 10.0 {
            return Err(format!("Invalid FP16 multiplier: {}", profile.precision.fp16_multiplier));
        }

        // Validate batch scaling curves
        for curve in &[
            &profile.model_performance.cnn.batch_scaling_curve,
            &profile.model_performance.transformer.batch_scaling_curve,
            &profile.model_performance.rnn.batch_scaling_curve,
        ] {
            if curve.is_empty() || curve[0] != 1.0 {
                return Err("Batch scaling curve must start with 1.0".to_string());
            }

            for value in curve.iter() {
                if *value <= 0.0 || *value > 1.0 {
                    return Err(format!("Invalid batch scaling value: {}", value));
                }
            }
        }

        Ok(())
    }
}

/// Convert our TOML config to the existing real hardware model types
impl From<&HardwareProfileConfig> for crate::real_hardware_model::RealHardwareProfile {
    fn from(config: &HardwareProfileConfig) -> Self {
        use crate::real_hardware_model::ModelTypePerformance;
        use crate::gpu_config::GpuModel;

        // Convert batch scaling curve from Vec<f64> to Vec<(usize, f64)>
        let convert_batch_scaling = |curve: &Vec<f64>| -> Vec<(usize, f64)> {
            curve
                .iter()
                .enumerate()
                .map(|(i, &efficiency)| {
                    let batch_size = 1 << i; // 1, 2, 4, 8, 16, 32, 64, 128
                    (batch_size, efficiency)
                })
                .collect()
        };

        Self {
            gpu_model: GpuModel {
                name: config.name.clone(),
                memory_gb: 0.0, // Will be filled from basic GPU config
                compute_tflops: 0.0, // Will be filled from basic GPU config
                memory_bandwidth_gbps: 0.0, // Will be filled from basic GPU config
                architecture: None, // Will be filled from basic GPU config
                release_year: None, // Will be filled from basic GPU config
            },
            thermal_design_power: config.thermal.tdp_watts,
            base_clock_mhz: config.thermal.base_clock_mhz,
            boost_clock_mhz: config.thermal.boost_clock_mhz,
            thermal_throttle_temp: config.thermal.throttle_temp_celsius,
            l1_cache_kb: config.memory.l1_cache_kb,
            l2_cache_mb: config.memory.l2_cache_mb,
            memory_channels: config.memory.memory_channels as usize,
            memory_bus_width: config.architecture.memory_bus_width as usize,
            cuda_cores: config.architecture.cuda_cores as usize,
            tensor_cores: if config.architecture.tensor_cores > 0 {
                Some(config.architecture.tensor_cores as usize)
            } else {
                None
            },
            rt_cores: if config.architecture.rt_cores > 0 {
                Some(config.architecture.rt_cores as usize)
            } else {
                None
            },
            streaming_multiprocessors: config.architecture.streaming_multiprocessors as usize,
            cnn_performance: ModelTypePerformance {
                batch_scaling: convert_batch_scaling(
                    &config.model_performance.cnn.batch_scaling_curve
                ),
                memory_efficiency: std::collections::HashMap::from([
                    ("small".to_string(), config.model_performance.cnn.memory_efficiency * 0.8),
                    ("medium".to_string(), config.model_performance.cnn.memory_efficiency),
                    ("large".to_string(), config.model_performance.cnn.memory_efficiency * 1.1),
                ]),
                fp32_multiplier: 1.0,
                fp16_multiplier: config.precision.fp16_multiplier,
                int8_multiplier: config.precision.int8_multiplier,
                tensor_core_utilization: config.model_performance.cnn.tensor_core_utilization,
                memory_bound_ratio: 1.0 - config.model_performance.cnn.memory_efficiency,
            },
            transformer_performance: ModelTypePerformance {
                batch_scaling: convert_batch_scaling(
                    &config.model_performance.transformer.batch_scaling_curve
                ),
                memory_efficiency: std::collections::HashMap::from([
                    (
                        "small".to_string(),
                        config.model_performance.transformer.memory_efficiency * 0.8,
                    ),
                    ("medium".to_string(), config.model_performance.transformer.memory_efficiency),
                    (
                        "large".to_string(),
                        config.model_performance.transformer.memory_efficiency * 1.1,
                    ),
                ]),
                fp32_multiplier: 1.0,
                fp16_multiplier: config.precision.fp16_multiplier,
                int8_multiplier: config.precision.int8_multiplier,
                tensor_core_utilization: config.model_performance.transformer.tensor_core_utilization,
                memory_bound_ratio: 1.0 - config.model_performance.transformer.memory_efficiency,
            },
            rnn_performance: ModelTypePerformance {
                batch_scaling: convert_batch_scaling(
                    &config.model_performance.rnn.batch_scaling_curve
                ),
                memory_efficiency: std::collections::HashMap::from([
                    ("small".to_string(), config.model_performance.rnn.memory_efficiency * 0.8),
                    ("medium".to_string(), config.model_performance.rnn.memory_efficiency),
                    ("large".to_string(), config.model_performance.rnn.memory_efficiency * 1.1),
                ]),
                fp32_multiplier: 1.0,
                fp16_multiplier: config.precision.fp16_multiplier,
                int8_multiplier: config.precision.int8_multiplier,
                tensor_core_utilization: config.model_performance.rnn.tensor_core_utilization,
                memory_bound_ratio: 1.0 - config.model_performance.rnn.memory_efficiency,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_profiles_load() {
        let mut loader = HardwareProfileLoader {
            profiles: HashMap::new(),
            config_path: "nonexistent.toml".to_string(),
        };

        assert!(loader.load_default_profiles().is_ok());
        assert!(loader.get_profile("v100").is_some());
    }

    #[test]
    fn test_profile_validation() {
        let mut loader = HardwareProfileLoader {
            profiles: HashMap::new(),
            config_path: "nonexistent.toml".to_string(),
        };

        loader.load_default_profiles().unwrap();
        assert!(loader.validate_profile("v100").is_ok());
        assert!(loader.validate_profile("nonexistent").is_err());
    }
}
