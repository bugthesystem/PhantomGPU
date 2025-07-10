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

        Self {
            gpu_models,
        }
    }

    pub async fn run_validation_tests(&self) -> PhantomResult<GamingAccuracyReport> {
        let mut results = Vec::new();

        // Test RTX 4090 with Cyberpunk 2077 at 1440p with RT and DLSS Quality
        if let Some(gpu_model) = self.gpu_models.get("RTX 4090") {
            let mut emulator = UnifiedGamingEmulator::new(gpu_model.clone());
            let workload = GamingWorkloadConfig {
                game_name: "Cyberpunk 2077".to_string(),
                resolution: (2560, 1440),
                ray_tracing: true,
                dlss_mode: DLSSMode::Quality,
                fsr_mode: FSRMode::Off,
                graphics_settings: GraphicsSettings {
                    texture_quality: Quality::Ultra,
                    shadow_quality: Quality::High,
                    anti_aliasing: AntiAliasing::TAA,
                    anisotropic_filtering: 16,
                    variable_rate_shading: true,
                    mesh_shaders: true,
                },
                target_fps: 60.0,
                scene_complexity: 0.8,
            };

            let performance = emulator.predict_gaming_performance(&workload, 25.0).await?;

            results.push(GamingAccuracyResult {
                gpu_name: "RTX 4090".to_string(),
                game_name: "Cyberpunk 2077".to_string(),
                predicted_fps: performance.avg_fps,
                expected_fps: 85.0, // From TOML validation data
                accuracy_percentage: calculate_accuracy(performance.avg_fps, 85.0),
                error_percentage: calculate_error(performance.avg_fps, 85.0),
                within_tolerance: is_within_tolerance(performance.avg_fps, 85.0, 0.15),
                tolerance: 0.15,
                source: "TechPowerUp RTX 4090 Review".to_string(),
            });
        }

        // Test RTX 4090 with Fortnite at 1080p without RT
        if let Some(gpu_model) = self.gpu_models.get("RTX 4090") {
            let mut emulator = UnifiedGamingEmulator::new(gpu_model.clone());
            let workload = GamingWorkloadConfig {
                game_name: "Fortnite".to_string(),
                resolution: (1920, 1080),
                ray_tracing: false,
                dlss_mode: DLSSMode::Off,
                fsr_mode: FSRMode::Off,
                graphics_settings: GraphicsSettings {
                    texture_quality: Quality::High,
                    shadow_quality: Quality::Medium,
                    anti_aliasing: AntiAliasing::TAA,
                    anisotropic_filtering: 8,
                    variable_rate_shading: false,
                    mesh_shaders: false,
                },
                target_fps: 120.0,
                scene_complexity: 0.5,
            };

            let performance = emulator.predict_gaming_performance(&workload, 25.0).await?;

            results.push(GamingAccuracyResult {
                gpu_name: "RTX 4090".to_string(),
                game_name: "Fortnite".to_string(),
                predicted_fps: performance.avg_fps,
                expected_fps: 200.0, // From TOML validation data
                accuracy_percentage: calculate_accuracy(performance.avg_fps, 200.0),
                error_percentage: calculate_error(performance.avg_fps, 200.0),
                within_tolerance: is_within_tolerance(performance.avg_fps, 200.0, 0.12),
                tolerance: 0.12,
                source: "Competitive Gaming Benchmarks".to_string(),
            });
        }

        // Test RTX 5090 with Cyberpunk 2077 at 4K with RT and DLSS Quality
        if let Some(gpu_model) = self.gpu_models.get("RTX 5090") {
            let mut emulator = UnifiedGamingEmulator::new(gpu_model.clone());
            let workload = GamingWorkloadConfig {
                game_name: "Cyberpunk 2077".to_string(),
                resolution: (3840, 2160),
                ray_tracing: true,
                dlss_mode: DLSSMode::Quality,
                fsr_mode: FSRMode::Off,
                graphics_settings: GraphicsSettings {
                    texture_quality: Quality::Ultra,
                    shadow_quality: Quality::High,
                    anti_aliasing: AntiAliasing::TAA,
                    anisotropic_filtering: 16,
                    variable_rate_shading: true,
                    mesh_shaders: true,
                },
                target_fps: 60.0,
                scene_complexity: 0.8,
            };

            let performance = emulator.predict_gaming_performance(&workload, 25.0).await?;

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

        // Test RTX 4090 with Apex Legends at 1440p
        if let Some(gpu_model) = self.gpu_models.get("RTX 4090") {
            let mut emulator = UnifiedGamingEmulator::new(gpu_model.clone());
            let workload = GamingWorkloadConfig {
                game_name: "Apex Legends".to_string(),
                resolution: (2560, 1440),
                ray_tracing: false,
                dlss_mode: DLSSMode::Quality,
                fsr_mode: FSRMode::Off,
                graphics_settings: GraphicsSettings {
                    texture_quality: Quality::High,
                    shadow_quality: Quality::Medium,
                    anti_aliasing: AntiAliasing::TAA,
                    anisotropic_filtering: 8,
                    variable_rate_shading: true,
                    mesh_shaders: false,
                },
                target_fps: 144.0,
                scene_complexity: 0.6,
            };

            let performance = emulator.predict_gaming_performance(&workload, 25.0).await?;

            results.push(GamingAccuracyResult {
                gpu_name: "RTX 4090".to_string(),
                game_name: "Apex Legends".to_string(),
                predicted_fps: performance.avg_fps,
                expected_fps: 165.0, // From TOML validation data
                accuracy_percentage: calculate_accuracy(performance.avg_fps, 165.0),
                error_percentage: calculate_error(performance.avg_fps, 165.0),
                within_tolerance: is_within_tolerance(performance.avg_fps, 165.0, 0.1),
                tolerance: 0.1,
                source: "Hardware Unboxed, December 2024".to_string(),
            });
        }

        // Test RTX 5090 with Valorant at 1080p
        if let Some(gpu_model) = self.gpu_models.get("RTX 5090") {
            let mut emulator = UnifiedGamingEmulator::new(gpu_model.clone());
            let workload = GamingWorkloadConfig {
                game_name: "Valorant".to_string(),
                resolution: (1920, 1080),
                ray_tracing: false,
                dlss_mode: DLSSMode::Off,
                fsr_mode: FSRMode::Off,
                graphics_settings: GraphicsSettings {
                    texture_quality: Quality::High,
                    shadow_quality: Quality::Low,
                    anti_aliasing: AntiAliasing::FXAA,
                    anisotropic_filtering: 8,
                    variable_rate_shading: false,
                    mesh_shaders: false,
                },
                target_fps: 240.0,
                scene_complexity: 0.3,
            };

            let performance = emulator.predict_gaming_performance(&workload, 25.0).await?;

            results.push(GamingAccuracyResult {
                gpu_name: "RTX 5090".to_string(),
                game_name: "Valorant".to_string(),
                predicted_fps: performance.avg_fps,
                expected_fps: 380.0, // From TOML validation data
                accuracy_percentage: calculate_accuracy(performance.avg_fps, 380.0),
                error_percentage: calculate_error(performance.avg_fps, 380.0),
                within_tolerance: is_within_tolerance(performance.avg_fps, 380.0, 0.08),
                tolerance: 0.08,
                source: "Pro Gaming Benchmarks".to_string(),
            });
        }

        // Test RTX 4080 with Overwatch 2 at 1440p
        if let Some(gpu_model) = self.gpu_models.get("RTX 4080") {
            let mut emulator = UnifiedGamingEmulator::new(gpu_model.clone());
            let workload = GamingWorkloadConfig {
                game_name: "Overwatch 2".to_string(),
                resolution: (2560, 1440),
                ray_tracing: false,
                dlss_mode: DLSSMode::Quality,
                fsr_mode: FSRMode::Off,
                graphics_settings: GraphicsSettings {
                    texture_quality: Quality::Ultra,
                    shadow_quality: Quality::High,
                    anti_aliasing: AntiAliasing::TAA,
                    anisotropic_filtering: 16,
                    variable_rate_shading: true,
                    mesh_shaders: true,
                },
                target_fps: 165.0,
                scene_complexity: 0.5,
            };

            let performance = emulator.predict_gaming_performance(&workload, 25.0).await?;

            results.push(GamingAccuracyResult {
                gpu_name: "RTX 4080".to_string(),
                game_name: "Overwatch 2".to_string(),
                predicted_fps: performance.avg_fps,
                expected_fps: 190.0, // From TOML validation data
                accuracy_percentage: calculate_accuracy(performance.avg_fps, 190.0),
                error_percentage: calculate_error(performance.avg_fps, 190.0),
                within_tolerance: is_within_tolerance(performance.avg_fps, 190.0, 0.12),
                tolerance: 0.12,
                source: "Competitive Gaming Reviews".to_string(),
            });
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

        Ok(GamingAccuracyReport {
            results,
            overall_accuracy,
            tests_passed,
            total_tests: 6, // Updated total
        })
    }
}

// Helper functions
fn calculate_accuracy(predicted: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        if predicted == 0.0 { 100.0 } else { 0.0 }
    } else {
        let error = (predicted - expected).abs() / expected;
        ((1.0 - error) * 100.0).max(0.0)
    }
}

fn calculate_error(predicted: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        if predicted == 0.0 { 0.0 } else { 100.0 }
    } else {
        (((predicted - expected).abs() / expected) * 100.0).min(100.0)
    }
}

fn is_within_tolerance(predicted: f64, expected: f64, tolerance: f64) -> bool {
    if expected == 0.0 {
        predicted == 0.0
    } else {
        let error = (predicted - expected).abs() / expected;
        error <= tolerance
    }
}

// Format results for display
impl GamingAccuracyReport {
    pub fn format_detailed_report(&self) -> String {
        let mut report = String::new();
        report.push_str("🎯 PhantomGPU Gaming Accuracy Report\n");
        report.push_str("==================================================\n\n");

        for result in &self.results {
            let status = if result.within_tolerance { "✅ PASS" } else { "❌ FAIL" };
            report.push_str(&format!("🎮 {} - {}\n", result.gpu_name, result.game_name));
            report.push_str(
                &format!(
                    "  Predicted: {:.1} FPS | Expected: {:.1} FPS\n",
                    result.predicted_fps,
                    result.expected_fps
                )
            );
            report.push_str(
                &format!(
                    "  Accuracy: {:.1}% (±{:.1}% error) - {}\n",
                    result.accuracy_percentage,
                    result.error_percentage,
                    status
                )
            );
            report.push_str(&format!("  Source: {}\n\n", result.source));
        }

        report.push_str(&format!("📊 Overall Accuracy: {:.1}%\n", self.overall_accuracy));
        report.push_str(&format!("✅ Passed: {}/{} tests\n", self.tests_passed, self.total_tests));

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
                println!("Overall Accuracy: {:.1}%", report.overall_accuracy);
                println!("Passed: {}/{} tests", report.tests_passed, report.total_tests);

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
                assert!(report.tests_passed > 0, "At least one test should pass");
            }
            Err(e) => {
                println!("Gaming accuracy test failed: {:?}", e);
                panic!("Gaming accuracy test failed");
            }
        }
    }
}
