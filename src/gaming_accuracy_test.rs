use crate::unified_gaming_emulator::UnifiedGamingEmulator;
use crate::models::{
    GamingWorkloadConfig,
    DLSSMode,
    FSRMode,
    GraphicsSettings,
    Quality,
    AntiAliasing,
};
use crate::gpu_config::GpuModel;
use crate::errors::PhantomResult;
use std::collections::HashMap;
use serde::{ Deserialize, Serialize };
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeasurement {
    pub resolution: String,
    pub preset: String,
    pub ray_tracing: bool,
    pub dlss_mode: String,
    pub fsr_mode: String,
    pub avg_fps: f64,
    pub one_percent_low: f64,
    pub frame_time_ms: f64,
    pub memory_usage_mb: f64,
    pub gpu_utilization_percent: f64,
    pub power_usage_watts: f64,
    pub temperature_celsius: f64,
    pub runs: u32,
    pub std_dev_fps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkEntry {
    pub gpu_name: String,
    pub gpu_architecture: String,
    pub game_name: String,
    pub game_engine: String,
    pub resolutions: Vec<String>,
    pub measurements: Vec<BenchmarkMeasurement>,
    pub system_info: HashMap<String, serde_json::Value>,
    pub timestamp: String,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct GamingAccuracyResult {
    pub gpu_name: String,
    pub game_name: String,
    pub predicted_fps: f64,
    pub expected_fps: f64,
    pub accuracy_percentage: f64,
    pub error_percentage: f64,
    pub within_tolerance: bool,
    pub tolerance: f64,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct GamingAccuracyReport {
    pub results: Vec<GamingAccuracyResult>,
    pub overall_accuracy: f64,
    pub tests_passed: usize,
    pub total_tests: usize,
}

pub struct GamingAccuracyValidator {
    gpu_models: HashMap<String, GpuModel>,
    benchmark_data: Vec<BenchmarkEntry>,
    edge_cases: Vec<BenchmarkEntry>,
}

impl GamingAccuracyValidator {
    pub fn new() -> Self {
        let mut gpu_models = HashMap::new();

        // RTX 4090
        gpu_models.insert("RTX 4090".to_string(), GpuModel {
            name: "RTX 4090".to_string(),
            memory_gb: 24.0,
            memory_bandwidth_gbps: 1008.0,
            compute_tflops: 42.0,
            architecture: Some("Ada Lovelace".to_string()),
            release_year: Some(2022),
        });

        // RTX 5090
        gpu_models.insert("RTX 5090".to_string(), GpuModel {
            name: "RTX 5090".to_string(),
            memory_gb: 32.0,
            memory_bandwidth_gbps: 1792.0,
            compute_tflops: 125.0,
            architecture: Some("Blackwell".to_string()),
            release_year: Some(2025),
        });

        // RTX 4080
        gpu_models.insert("RTX 4080".to_string(), GpuModel {
            name: "RTX 4080".to_string(),
            memory_gb: 16.0,
            memory_bandwidth_gbps: 716.8,
            compute_tflops: 48.7,
            architecture: Some("Ada Lovelace".to_string()),
            release_year: Some(2022),
        });

        // Load benchmark data
        let benchmark_data = Self::load_benchmark_data();
        let edge_cases = Self::load_edge_cases();

        Self {
            gpu_models,
            benchmark_data,
            edge_cases,
        }
    }

    fn load_benchmark_data() -> Vec<BenchmarkEntry> {
        match fs::read_to_string("benchmark_data/gaming_benchmarks.json") {
            Ok(content) => {
                match serde_json::from_str::<Vec<BenchmarkEntry>>(&content) {
                    Ok(data) => data,
                    Err(e) => {
                        eprintln!("Warning: Failed to parse gaming benchmark data: {}", e);
                        Vec::new()
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Failed to load gaming benchmark data: {}", e);
                Vec::new()
            }
        }
    }

    fn load_edge_cases() -> Vec<BenchmarkEntry> {
        match fs::read_to_string("benchmark_data/gaming_edge_cases.json") {
            Ok(content) => {
                match serde_json::from_str::<Vec<BenchmarkEntry>>(&content) {
                    Ok(data) => data,
                    Err(e) => {
                        eprintln!("Warning: Failed to parse gaming edge cases: {}", e);
                        Vec::new()
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Failed to load gaming edge cases: {}", e);
                Vec::new()
            }
        }
    }

    pub async fn run_validation_tests(&self) -> PhantomResult<GamingAccuracyReport> {
        let mut results = Vec::new();

        // Test against main benchmark data
        for entry in &self.benchmark_data {
            if let Some(gpu_model) = self.gpu_models.get(&entry.gpu_name) {
                for measurement in &entry.measurements {
                    let result = self.run_single_benchmark_test(
                        gpu_model,
                        entry,
                        measurement,
                        false // not edge case
                    ).await?;
                    results.push(result);
                }
            }
        }

        // Test against edge cases
        for entry in &self.edge_cases {
            if let Some(gpu_model) = self.gpu_models.get(&entry.gpu_name) {
                for measurement in &entry.measurements {
                    let result = self.run_single_benchmark_test(
                        gpu_model,
                        entry,
                        measurement,
                        true // is edge case
                    ).await?;
                    results.push(result);
                }
            }
        }

        // Calculate overall accuracy
        let overall_accuracy = if results.is_empty() {
            0.0
        } else {
            results
                .iter()
                .map(|r| r.accuracy_percentage)
                .sum::<f64>() / (results.len() as f64)
        };

        let tests_passed = results
            .iter()
            .filter(|r| r.within_tolerance)
            .count();
        let total_tests = results.len();

        Ok(GamingAccuracyReport {
            results,
            overall_accuracy,
            tests_passed,
            total_tests,
        })
    }

    async fn run_single_benchmark_test(
        &self,
        gpu_model: &GpuModel,
        entry: &BenchmarkEntry,
        measurement: &BenchmarkMeasurement,
        is_edge_case: bool
    ) -> PhantomResult<GamingAccuracyResult> {
        let mut emulator = UnifiedGamingEmulator::new(gpu_model.clone());

        // Parse resolution
        let resolution = Self::parse_resolution(&measurement.resolution)?;

        // Create workload config from benchmark data
        let workload = GamingWorkloadConfig {
            game_name: entry.game_name.clone(),
            resolution,
            ray_tracing: measurement.ray_tracing,
            dlss_mode: Self::parse_dlss_mode(&measurement.dlss_mode),
            fsr_mode: Self::parse_fsr_mode(&measurement.fsr_mode),
            graphics_settings: Self::parse_graphics_settings(&measurement.preset),
            target_fps: measurement.avg_fps, // Use actual FPS as target
            scene_complexity: Self::estimate_scene_complexity(&entry.game_name),
        };

        let performance = emulator.predict_gaming_performance(&workload, 25.0).await?;

        // Calculate tolerance based on whether it's an edge case
        let tolerance = if is_edge_case { 0.25 } else { 0.15 }; // 25% tolerance for edge cases, 15% for normal cases

        let accuracy = calculate_accuracy(performance.avg_fps, measurement.avg_fps);
        let error = calculate_error(performance.avg_fps, measurement.avg_fps);
        let within_tolerance = is_within_tolerance(
            performance.avg_fps,
            measurement.avg_fps,
            tolerance
        );

        Ok(GamingAccuracyResult {
            gpu_name: entry.gpu_name.clone(),
            game_name: entry.game_name.clone(),
            predicted_fps: performance.avg_fps,
            expected_fps: measurement.avg_fps,
            accuracy_percentage: accuracy,
            error_percentage: error,
            within_tolerance,
            tolerance,
            source: entry.source.clone(),
        })
    }

    fn parse_resolution(resolution_str: &str) -> PhantomResult<(u32, u32)> {
        let parts: Vec<&str> = resolution_str.split('x').collect();
        if parts.len() != 2 {
            return Err(crate::errors::PhantomGpuError::ConfigError {
                message: format!("Invalid resolution format: {}", resolution_str),
            });
        }

        let width = parts[0]
            .parse::<u32>()
            .map_err(|_| crate::errors::PhantomGpuError::ConfigError {
                message: format!("Invalid width in resolution: {}", parts[0]),
            })?;

        let height = parts[1]
            .parse::<u32>()
            .map_err(|_| crate::errors::PhantomGpuError::ConfigError {
                message: format!("Invalid height in resolution: {}", parts[1]),
            })?;

        Ok((width, height))
    }

    fn parse_dlss_mode(dlss_str: &str) -> DLSSMode {
        match dlss_str.to_lowercase().as_str() {
            "off" => DLSSMode::Off,
            "quality" => DLSSMode::Quality,
            "balanced" => DLSSMode::Balanced,
            "performance" => DLSSMode::Performance,
            "ultraperformance" => DLSSMode::UltraPerformance,
            _ => DLSSMode::Off,
        }
    }

    fn parse_fsr_mode(fsr_str: &str) -> FSRMode {
        match fsr_str.to_lowercase().as_str() {
            "off" => FSRMode::Off,
            "ultraulity" => FSRMode::UltraQuality,
            "quality" => FSRMode::Quality,
            "balanced" => FSRMode::Balanced,
            "performance" => FSRMode::Performance,
            _ => FSRMode::Off,
        }
    }

    fn parse_graphics_settings(preset: &str) -> GraphicsSettings {
        match preset.to_lowercase().as_str() {
            "ultra" | "psycho" | "maximum" =>
                GraphicsSettings {
                    texture_quality: Quality::Ultra,
                    shadow_quality: Quality::High,
                    anti_aliasing: AntiAliasing::TAA,
                    anisotropic_filtering: 16,
                    variable_rate_shading: true,
                    mesh_shaders: true,
                },
            "high" | "epic" =>
                GraphicsSettings {
                    texture_quality: Quality::High,
                    shadow_quality: Quality::High,
                    anti_aliasing: AntiAliasing::TAA,
                    anisotropic_filtering: 16,
                    variable_rate_shading: true,
                    mesh_shaders: false,
                },
            "medium" =>
                GraphicsSettings {
                    texture_quality: Quality::Medium,
                    shadow_quality: Quality::Medium,
                    anti_aliasing: AntiAliasing::FXAA,
                    anisotropic_filtering: 8,
                    variable_rate_shading: false,
                    mesh_shaders: false,
                },
            "low" | "performance" =>
                GraphicsSettings {
                    texture_quality: Quality::Low,
                    shadow_quality: Quality::Low,
                    anti_aliasing: AntiAliasing::FXAA,
                    anisotropic_filtering: 4,
                    variable_rate_shading: false,
                    mesh_shaders: false,
                },
            _ =>
                GraphicsSettings {
                    texture_quality: Quality::High,
                    shadow_quality: Quality::High,
                    anti_aliasing: AntiAliasing::TAA,
                    anisotropic_filtering: 16,
                    variable_rate_shading: true,
                    mesh_shaders: false,
                },
        }
    }

    fn estimate_scene_complexity(game_name: &str) -> f64 {
        match game_name {
            "Cyberpunk 2077" => 0.9,
            "Hogwarts Legacy" => 0.8,
            "Call of Duty: Modern Warfare III" => 0.7,
            "Apex Legends" => 0.6,
            "Fortnite" => 0.5,
            "Overwatch 2" => 0.5,
            "Valorant" => 0.3,
            _ => 0.6,
        }
    }
}

fn calculate_accuracy(predicted: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        return 0.0;
    }
    let error = (predicted - expected).abs() / expected;
    ((1.0 - error) * 100.0).max(0.0)
}

fn calculate_error(predicted: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        return 100.0;
    }
    ((predicted - expected).abs() / expected) * 100.0
}

fn is_within_tolerance(predicted: f64, expected: f64, tolerance: f64) -> bool {
    if expected == 0.0 {
        return predicted == 0.0;
    }
    let error = (predicted - expected).abs() / expected;
    error <= tolerance
}

impl GamingAccuracyReport {
    pub fn format_detailed_report(&self) -> String {
        let mut report = String::new();

        report.push_str(
            &format!(
                "🎯 Gaming Accuracy Test Results\n\
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\
            📊 Overall Accuracy: {:.1}%\n\
            ✅ Passed: {}/{} tests\n\n\
            📋 Detailed Results:\n",
                self.overall_accuracy,
                self.tests_passed,
                self.total_tests
            )
        );

        for result in &self.results {
            let status = if result.within_tolerance { "✅ PASS" } else { "❌ FAIL" };
            report.push_str(
                &format!(
                    "  {} - {}: {:.1}% accuracy (±{:.1}% error) - {} [{}]\n",
                    result.gpu_name,
                    result.game_name,
                    result.accuracy_percentage,
                    result.error_percentage,
                    status,
                    result.source
                )
            );
        }

        report
    }
}

pub async fn run_gaming_accuracy_test() -> PhantomResult<GamingAccuracyReport> {
    let validator = GamingAccuracyValidator::new();
    validator.run_validation_tests().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gaming_accuracy_validation() {
        let result = run_gaming_accuracy_test().await;
        match result {
            Ok(report) => {
                println!("Gaming Accuracy Test Results:");
                println!("{}", report.format_detailed_report());

                // We expect reasonable accuracy (>50% overall)
                assert!(
                    report.overall_accuracy > 50.0,
                    "Overall accuracy ({:.1}%) should be greater than 50%",
                    report.overall_accuracy
                );

                // We expect at least some tests to pass
                assert!(report.tests_passed > 0, "At least one test should pass");
            }
            Err(e) => {
                panic!("Gaming accuracy test failed: {}", e);
            }
        }
    }
}
