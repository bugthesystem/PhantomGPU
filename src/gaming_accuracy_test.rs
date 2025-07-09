use crate::gaming_performance::{
    GamingPerformanceEngine,
    GamingWorkload,
    DLSSMode,
    FSRMode,
    GraphicsSettings,
    Quality,
    AntiAliasing,
};
use crate::gpu_config::GpuModel;
use crate::errors::PhantomGpuError;
use std::collections::HashMap;

#[derive(Debug)]
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

#[derive(Debug)]
pub struct GamingAccuracyReport {
    pub results: Vec<GamingAccuracyResult>,
    pub overall_accuracy: f64,
    pub passed_tests: usize,
    pub total_tests: usize,
}

pub struct GamingAccuracyValidator {
    gaming_engine: GamingPerformanceEngine,
    gpu_models: HashMap<String, GpuModel>,
}

impl GamingAccuracyValidator {
    pub fn new() -> Result<Self, PhantomGpuError> {
        // Load GPU models from config
        let gpu_models = load_gpu_models()?;

        Ok(Self {
            gaming_engine: GamingPerformanceEngine::new(),
            gpu_models,
        })
    }

    pub fn validate_all_predictions(&self) -> Result<GamingAccuracyReport, PhantomGpuError> {
        let mut results = Vec::new();

        // Test RTX 4090 with Cyberpunk 2077 at 1440p
        if let Some(gpu_model) = self.gpu_models.get("RTX 4090") {
            let workload = GamingWorkload {
                game_name: "Cyberpunk 2077".to_string(),
                resolution: (2560, 1440),
                ray_tracing: true,
                dlss_mode: DLSSMode::Quality,
                fsr_mode: FSRMode::Off,
                target_fps: 60.0,
                scene_complexity: 0.7,
                graphics_settings: GraphicsSettings {
                    texture_quality: Quality::High,
                    shadow_quality: Quality::High,
                    anti_aliasing: AntiAliasing::TAA,
                    anisotropic_filtering: 16,
                    variable_rate_shading: true,
                    mesh_shaders: true,
                },
            };

            let performance = self.gaming_engine.predict_gaming_performance(
                gpu_model,
                &workload,
                25.0
            )?;

            results.push(GamingAccuracyResult {
                gpu_name: "RTX 4090".to_string(),
                game_name: "Cyberpunk 2077".to_string(),
                predicted_fps: performance.avg_fps,
                expected_fps: 85.0, // From TOML validation data
                accuracy_percentage: calculate_accuracy(performance.avg_fps, 85.0),
                error_percentage: calculate_error(performance.avg_fps, 85.0),
                within_tolerance: is_within_tolerance(performance.avg_fps, 85.0, 0.15),
                tolerance: 0.15,
                source: "TechPowerUp, January 2025".to_string(),
            });
        }

        // Test RTX 4090 with Fortnite at 1080p
        if let Some(gpu_model) = self.gpu_models.get("RTX 4090") {
            let workload = GamingWorkload {
                game_name: "Fortnite".to_string(),
                resolution: (1920, 1080),
                ray_tracing: false,
                dlss_mode: DLSSMode::Off,
                fsr_mode: FSRMode::Off,
                target_fps: 144.0,
                scene_complexity: 0.5,
                graphics_settings: GraphicsSettings {
                    texture_quality: Quality::High,
                    shadow_quality: Quality::Medium,
                    anti_aliasing: AntiAliasing::FXAA,
                    anisotropic_filtering: 8,
                    variable_rate_shading: true,
                    mesh_shaders: false,
                },
            };

            let performance = self.gaming_engine.predict_gaming_performance(
                gpu_model,
                &workload,
                25.0
            )?;

            results.push(GamingAccuracyResult {
                gpu_name: "RTX 4090".to_string(),
                game_name: "Fortnite".to_string(),
                predicted_fps: performance.avg_fps,
                expected_fps: 195.0, // From TOML validation data
                accuracy_percentage: calculate_accuracy(performance.avg_fps, 195.0),
                error_percentage: calculate_error(performance.avg_fps, 195.0),
                within_tolerance: is_within_tolerance(performance.avg_fps, 195.0, 0.1),
                tolerance: 0.1,
                source: "Competitive Gaming Benchmarks".to_string(),
            });
        }

        // Test RTX 5090 with Cyberpunk 2077 at 4K
        if let Some(gpu_model) = self.gpu_models.get("RTX 5090") {
            let workload = GamingWorkload {
                game_name: "Cyberpunk 2077".to_string(),
                resolution: (3840, 2160),
                ray_tracing: true,
                dlss_mode: DLSSMode::Quality,
                fsr_mode: FSRMode::Off,
                target_fps: 60.0,
                scene_complexity: 0.8,
                graphics_settings: GraphicsSettings {
                    texture_quality: Quality::Ultra,
                    shadow_quality: Quality::High,
                    anti_aliasing: AntiAliasing::TAA,
                    anisotropic_filtering: 16,
                    variable_rate_shading: true,
                    mesh_shaders: true,
                },
            };

            let performance = self.gaming_engine.predict_gaming_performance(
                gpu_model,
                &workload,
                25.0
            )?;

            results.push(GamingAccuracyResult {
                gpu_name: "RTX 5090".to_string(),
                game_name: "Cyberpunk 2077".to_string(),
                predicted_fps: performance.avg_fps,
                expected_fps: 75.0, // From TOML validation data
                accuracy_percentage: calculate_accuracy(performance.avg_fps, 75.0),
                error_percentage: calculate_error(performance.avg_fps, 75.0),
                within_tolerance: is_within_tolerance(performance.avg_fps, 75.0, 0.12),
                tolerance: 0.12,
                source: "NVIDIA Internal Benchmarks".to_string(),
            });
        }

        // Calculate overall accuracy
        let passed_tests = results
            .iter()
            .filter(|r| r.within_tolerance)
            .count();
        let total_tests = results.len();
        let overall_accuracy = if total_tests > 0 {
            results
                .iter()
                .map(|r| r.accuracy_percentage)
                .sum::<f64>() / (total_tests as f64)
        } else {
            0.0
        };

        Ok(GamingAccuracyReport {
            results,
            overall_accuracy,
            passed_tests,
            total_tests,
        })
    }
}

fn calculate_accuracy(predicted: f64, actual: f64) -> f64 {
    let error = (predicted - actual).abs();
    let accuracy = (1.0 - error / actual.max(1.0)) * 100.0;
    accuracy.max(0.0).min(100.0)
}

fn calculate_error(predicted: f64, actual: f64) -> f64 {
    ((predicted - actual).abs() / actual.max(1.0)) * 100.0
}

fn is_within_tolerance(predicted: f64, actual: f64, tolerance: f64) -> bool {
    let error_ratio = (predicted - actual).abs() / actual.max(1.0);
    error_ratio <= tolerance
}

fn load_gpu_models() -> Result<HashMap<String, GpuModel>, PhantomGpuError> {
    let mut gpu_models = HashMap::new();

    // RTX 4090
    gpu_models.insert("RTX 4090".to_string(), GpuModel {
        name: "RTX 4090".to_string(),
        memory_gb: 24.0,
        memory_bandwidth_gbps: 1008.0,
        compute_tflops: 83.0,
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

    Ok(gpu_models)
}

pub fn run_gaming_accuracy_test() -> Result<GamingAccuracyReport, PhantomGpuError> {
    let validator = GamingAccuracyValidator::new()?;
    validator.validate_all_predictions()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaming_accuracy_validation() {
        let result = run_gaming_accuracy_test();
        match result {
            Ok(report) => {
                println!("Gaming Accuracy Test Results:");
                println!("Overall Accuracy: {:.1}%", report.overall_accuracy);
                println!("Passed: {}/{} tests", report.passed_tests, report.total_tests);

                for result in &report.results {
                    println!(
                        "  {} - {}: {:.1}% accuracy (±{:.1}% error) - {}",
                        result.gpu_name,
                        result.game_name,
                        result.accuracy_percentage,
                        result.error_percentage,
                        if result.within_tolerance {
                            "✅ PASS"
                        } else {
                            "❌ FAIL"
                        }
                    );
                }

                // Assert that we have reasonable accuracy
                assert!(report.overall_accuracy > 50.0, "Overall accuracy should be above 50%");
                assert!(report.passed_tests > 0, "At least one test should pass");
            }
            Err(e) => {
                println!("Gaming accuracy test failed: {:?}", e);
                panic!("Gaming accuracy test failed");
            }
        }
    }
}
