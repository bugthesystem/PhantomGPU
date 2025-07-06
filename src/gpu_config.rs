// GPU Configuration Loader
// Loads GPU models from gpu_models.toml configuration file

use std::collections::HashMap;
use std::fs;
use serde::{ Deserialize, Serialize };
use anyhow::{ Result, Context };

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuModel {
    pub name: String,
    pub memory_gb: f32,
    pub compute_tflops: f32,
    pub memory_bandwidth_gbps: f32,
    pub architecture: Option<String>,
    pub release_year: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct GpuConfig {
    gpus: HashMap<String, GpuModel>,
    defaults: Option<DefaultConfig>,
}

#[derive(Debug, Deserialize)]
struct DefaultConfig {
    gpu: String,
}

/// GPU Model Manager - loads and provides access to GPU configurations
pub struct GpuModelManager {
    pub models: HashMap<String, GpuModel>,
    default_gpu: String,
}

impl GpuModelManager {
    /// Load GPU models from TOML configuration file
    pub fn load_from_file(config_path: &str) -> Result<Self> {
        let config_content = fs
            ::read_to_string(config_path)
            .with_context(|| format!("Failed to read GPU config file: {}", config_path))?;

        let config: GpuConfig = toml
            ::from_str(&config_content)
            .with_context(|| "Failed to parse GPU config TOML file")?;

        let default_gpu = config.defaults
            .and_then(|d| Some(d.gpu))
            .unwrap_or_else(|| "v100".to_string());

        Ok(Self {
            models: config.gpus,
            default_gpu,
        })
    }

    /// Load GPU models from embedded default configuration
    pub fn load_default() -> Result<Self> {
        // Fallback default configuration if file doesn't exist
        let default_config =
            r#"
[gpus.v100]
name = "Tesla V100"
memory_gb = 32.0
compute_tflops = 15.7
memory_bandwidth_gbps = 900.0
architecture = "Volta"

[gpus.a100]
name = "A100"
memory_gb = 80.0
compute_tflops = 19.5
memory_bandwidth_gbps = 1935.0
architecture = "Ampere"

[gpus.rtx4090]
name = "RTX 4090"
memory_gb = 24.0
compute_tflops = 35.0
memory_bandwidth_gbps = 1008.0
architecture = "Ada Lovelace"

[gpus.rtx3090]
name = "RTX 3090"
memory_gb = 24.0
compute_tflops = 35.6
memory_bandwidth_gbps = 936.0
architecture = "Ampere"

[gpus.t4]
name = "Tesla T4"
memory_gb = 16.0
compute_tflops = 8.1
memory_bandwidth_gbps = 320.0
architecture = "Turing"

[defaults]
gpu = "v100"
"#;

        let config: GpuConfig = toml
            ::from_str(default_config)
            .with_context(|| "Failed to parse embedded GPU config")?;

        Ok(Self {
            models: config.gpus,
            default_gpu: config.defaults.unwrap().gpu,
        })
    }

    /// Try to load from file, fallback to defaults if file doesn't exist
    pub fn load() -> Result<Self> {
        match Self::load_from_file("gpu_models.toml") {
            Ok(manager) => {
                println!("✅ Loaded GPU models from gpu_models.toml");
                Ok(manager)
            }
            Err(_) => {
                println!("⚠️  gpu_models.toml not found, using embedded defaults");
                Self::load_default()
            }
        }
    }

    /// Get GPU model by key
    pub fn get_gpu(&self, gpu_key: &str) -> Option<&GpuModel> {
        self.models.get(gpu_key)
    }

    /// Get default GPU model
    pub fn get_default_gpu(&self) -> Option<&GpuModel> {
        self.models.get(&self.default_gpu)
    }

    /// List all available GPU models
    pub fn list_gpus(&self) -> Vec<(&String, &GpuModel)> {
        self.models.iter().collect()
    }

    /// Get GPU model by CLI enum value
    pub fn get_gpu_by_type(&self, gpu_type: &crate::cli::GpuType) -> Option<GpuModel> {
        let key = match gpu_type {
            crate::cli::GpuType::V100 => "v100",
            crate::cli::GpuType::A100 => "a100",
            crate::cli::GpuType::Rtx4090 => "rtx4090",
            crate::cli::GpuType::Rtx3090 => "rtx3090",
            crate::cli::GpuType::H100 => "h100",
            crate::cli::GpuType::H200 => "h200",
            crate::cli::GpuType::A6000 => "a6000",
            crate::cli::GpuType::L40s => "l40s",
            crate::cli::GpuType::Rtx5090 => "rtx5090",
            crate::cli::GpuType::RtxPro6000 => "rtx_pro_6000",
        };

        self.get_gpu(key).cloned()
    }

    /// Print available GPUs
    pub fn print_available_gpus(&self) {
        println!("\n🖥️  Available GPU Models:");
        println!("{}", "=".repeat(80));
        println!(
            "{:<12} {:<20} {:>8} {:>10} {:>12} {:<12}",
            "Key",
            "Name",
            "Memory",
            "TFLOPS",
            "Bandwidth",
            "Architecture"
        );
        println!("{}", "-".repeat(80));

        let mut gpus: Vec<_> = self.models.iter().collect();
        gpus.sort_by_key(|(k, _)| k.as_str());

        for (key, gpu) in gpus {
            println!(
                "{:<12} {:<20} {:>6.0}GB {:>8.1}T {:>9.0}GB/s {:<12}",
                key,
                gpu.name,
                gpu.memory_gb,
                gpu.compute_tflops,
                gpu.memory_bandwidth_gbps,
                gpu.architecture.as_deref().unwrap_or("Unknown")
            );
        }

        println!("\nDefault GPU: {}", self.default_gpu);
    }
}

/// Generate list of available GPU types for CLI
pub fn get_available_gpu_types() -> Vec<String> {
    // This could be dynamically generated from the config file
    vec![
        "v100".to_string(),
        "a100".to_string(),
        "rtx4090".to_string(),
        "rtx3090".to_string(),
        "t4".to_string(),
        "rtx4080".to_string(),
        "h100".to_string(),
        "a6000".to_string(),
        "l40s".to_string(),
        "mi300x".to_string()
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_default_config() {
        let manager = GpuModelManager::load_default().unwrap();

        // Test basic functionality
        assert!(manager.get_gpu("v100").is_some());
        assert!(manager.get_gpu("a100").is_some());
        assert!(manager.get_default_gpu().is_some());

        let v100 = manager.get_gpu("v100").unwrap();
        assert_eq!(v100.name, "Tesla V100");
        assert_eq!(v100.memory_gb, 32.0);
        assert_eq!(v100.compute_tflops, 15.7);
    }

    #[test]
    fn test_list_gpus() {
        let manager = GpuModelManager::load_default().unwrap();
        let gpus = manager.list_gpus();

        assert!(gpus.len() >= 5); // At least the default 5 GPUs
    }
}
