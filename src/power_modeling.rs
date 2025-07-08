use crate::gpu_config::GpuModel;
use crate::thermal_modeling::{ ThermalModelingEngine, ThermalState };
use crate::errors::{ PhantomGpuError, PhantomResult };
use serde::{ Deserialize, Serialize };
use std::collections::HashMap;
use std::time::Duration;

/// Power consumption result for a specific workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerConsumptionResult {
    pub gpu_name: String,
    pub workload_type: String,
    pub base_power_watts: f64,
    pub compute_power_watts: f64,
    pub memory_power_watts: f64,
    pub cooling_power_watts: f64,
    pub total_power_watts: f64,
    pub duration_seconds: f64,
    pub energy_consumption_wh: f64,
    pub energy_cost_usd: f64,
    pub thermal_impact: ThermalPowerImpact,
}

/// Power efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerEfficiencyMetrics {
    pub gpu_name: String,
    pub workload_type: String,
    pub performance_score: f64,
    pub power_consumption_watts: f64,
    pub efficiency_score: f64, // Performance per watt
    pub efficiency_rating: String, // "Excellent", "Good", "Fair", "Poor"
    pub samples_per_watt: f64,
    pub operations_per_joule: f64,
    pub comparison_metrics: PowerComparisonMetrics,
}

/// Power comparison metrics against other GPUs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerComparisonMetrics {
    pub efficiency_rank: usize,
    pub efficiency_percentile: f64,
    pub power_vs_average: f64, // +/- percentage vs average
    pub performance_vs_average: f64,
    pub cost_efficiency_score: f64,
}

/// Thermal impact on power consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalPowerImpact {
    pub base_temp_celsius: f64,
    pub peak_temp_celsius: f64,
    pub thermal_throttling_detected: bool,
    pub power_scaling_factor: f64,
    pub cooling_overhead_watts: f64,
    pub thermal_efficiency_loss: f64,
}

/// Power profile for different GPU architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerProfile {
    pub architecture: String,
    pub tdp_watts: f64,
    pub base_power_watts: f64,
    pub compute_power_factor: f64,
    pub memory_power_factor: f64,
    pub cooling_efficiency: f64,
    pub power_scaling_curve: PowerScalingCurve,
    pub thermal_design_point: f64,
}

/// Power scaling curve for different utilization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerScalingCurve {
    pub idle_power_ratio: f64,
    pub low_util_power_ratio: f64,
    pub medium_util_power_ratio: f64,
    pub high_util_power_ratio: f64,
    pub max_power_ratio: f64,
}

/// Workload power characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadPowerProfile {
    pub workload_name: String,
    pub compute_intensity: f64,
    pub memory_intensity: f64,
    pub utilization_pattern: String, // "Constant", "Bursty", "Periodic"
    pub thermal_generation_factor: f64,
    pub power_efficiency_modifier: f64,
}

/// Power modeling engine
pub struct PowerModelingEngine {
    power_profiles: HashMap<String, PowerProfile>,
    workload_profiles: HashMap<String, WorkloadPowerProfile>,
    thermal_engine: ThermalModelingEngine,
    energy_cost_per_kwh: f64,
}

impl PowerModelingEngine {
    /// Create a new power modeling engine
    pub fn new() -> Self {
        let mut engine = Self {
            power_profiles: HashMap::new(),
            workload_profiles: HashMap::new(),
            thermal_engine: ThermalModelingEngine::new(25.0), // Default ambient temperature
            energy_cost_per_kwh: 0.12, // Default US average
        };

        engine.load_power_profiles();
        engine.load_workload_profiles();
        engine
    }

    /// Load power profiles for different GPU architectures
    fn load_power_profiles(&mut self) {
        // Ampere architecture (A100, RTX 4090, etc.)
        self.power_profiles.insert("Ampere".to_string(), PowerProfile {
            architecture: "Ampere".to_string(),
            tdp_watts: 400.0,
            base_power_watts: 80.0,
            compute_power_factor: 0.85,
            memory_power_factor: 0.75,
            cooling_efficiency: 0.88,
            power_scaling_curve: PowerScalingCurve {
                idle_power_ratio: 0.15,
                low_util_power_ratio: 0.35,
                medium_util_power_ratio: 0.65,
                high_util_power_ratio: 0.85,
                max_power_ratio: 1.0,
            },
            thermal_design_point: 83.0,
        });

        // Hopper architecture (H100, H200)
        self.power_profiles.insert("Hopper".to_string(), PowerProfile {
            architecture: "Hopper".to_string(),
            tdp_watts: 700.0,
            base_power_watts: 120.0,
            compute_power_factor: 0.9,
            memory_power_factor: 0.8,
            cooling_efficiency: 0.92,
            power_scaling_curve: PowerScalingCurve {
                idle_power_ratio: 0.12,
                low_util_power_ratio: 0.3,
                medium_util_power_ratio: 0.6,
                high_util_power_ratio: 0.82,
                max_power_ratio: 1.0,
            },
            thermal_design_point: 90.0,
        });

        // Ada Lovelace architecture (RTX 4090, RTX 4080, etc.)
        self.power_profiles.insert("Ada Lovelace".to_string(), PowerProfile {
            architecture: "Ada Lovelace".to_string(),
            tdp_watts: 450.0,
            base_power_watts: 90.0,
            compute_power_factor: 0.82,
            memory_power_factor: 0.78,
            cooling_efficiency: 0.85,
            power_scaling_curve: PowerScalingCurve {
                idle_power_ratio: 0.18,
                low_util_power_ratio: 0.4,
                medium_util_power_ratio: 0.68,
                high_util_power_ratio: 0.88,
                max_power_ratio: 1.0,
            },
            thermal_design_point: 85.0,
        });

        // Volta architecture (V100)
        self.power_profiles.insert("Volta".to_string(), PowerProfile {
            architecture: "Volta".to_string(),
            tdp_watts: 300.0,
            base_power_watts: 70.0,
            compute_power_factor: 0.78,
            memory_power_factor: 0.72,
            cooling_efficiency: 0.8,
            power_scaling_curve: PowerScalingCurve {
                idle_power_ratio: 0.2,
                low_util_power_ratio: 0.42,
                medium_util_power_ratio: 0.7,
                high_util_power_ratio: 0.9,
                max_power_ratio: 1.0,
            },
            thermal_design_point: 80.0,
        });

        // Blackwell architecture (RTX 5090, etc.)
        self.power_profiles.insert("Blackwell".to_string(), PowerProfile {
            architecture: "Blackwell".to_string(),
            tdp_watts: 600.0,
            base_power_watts: 100.0,
            compute_power_factor: 0.88,
            memory_power_factor: 0.82,
            cooling_efficiency: 0.9,
            power_scaling_curve: PowerScalingCurve {
                idle_power_ratio: 0.14,
                low_util_power_ratio: 0.32,
                medium_util_power_ratio: 0.62,
                high_util_power_ratio: 0.84,
                max_power_ratio: 1.0,
            },
            thermal_design_point: 88.0,
        });
    }

    /// Load workload power profiles
    fn load_workload_profiles(&mut self) {
        // Large Language Models
        self.workload_profiles.insert("LLM".to_string(), WorkloadPowerProfile {
            workload_name: "Large Language Model".to_string(),
            compute_intensity: 0.95,
            memory_intensity: 0.85,
            utilization_pattern: "Constant".to_string(),
            thermal_generation_factor: 0.9,
            power_efficiency_modifier: 0.88,
        });

        // Computer Vision
        self.workload_profiles.insert("CV".to_string(), WorkloadPowerProfile {
            workload_name: "Computer Vision".to_string(),
            compute_intensity: 0.85,
            memory_intensity: 0.75,
            utilization_pattern: "Bursty".to_string(),
            thermal_generation_factor: 0.8,
            power_efficiency_modifier: 0.92,
        });

        // Training workloads
        self.workload_profiles.insert("Training".to_string(), WorkloadPowerProfile {
            workload_name: "Model Training".to_string(),
            compute_intensity: 0.98,
            memory_intensity: 0.9,
            utilization_pattern: "Constant".to_string(),
            thermal_generation_factor: 0.95,
            power_efficiency_modifier: 0.85,
        });

        // Inference workloads
        self.workload_profiles.insert("Inference".to_string(), WorkloadPowerProfile {
            workload_name: "Model Inference".to_string(),
            compute_intensity: 0.7,
            memory_intensity: 0.6,
            utilization_pattern: "Bursty".to_string(),
            thermal_generation_factor: 0.7,
            power_efficiency_modifier: 0.95,
        });

        // Mixed workloads
        self.workload_profiles.insert("Mixed".to_string(), WorkloadPowerProfile {
            workload_name: "Mixed Workload".to_string(),
            compute_intensity: 0.8,
            memory_intensity: 0.7,
            utilization_pattern: "Periodic".to_string(),
            thermal_generation_factor: 0.75,
            power_efficiency_modifier: 0.9,
        });
    }

    /// Calculate power consumption for a workload
    pub fn calculate_power_consumption(
        &self,
        gpu_model: &GpuModel,
        workload_type: &str,
        duration_seconds: f64,
        performance_score: f64,
        thermal_state: Option<&ThermalState>
    ) -> PhantomResult<PowerConsumptionResult> {
        let unknown_arch = "Unknown".to_string();
        let architecture = gpu_model.architecture.as_ref().unwrap_or(&unknown_arch);
        let power_profile = self.power_profiles
            .get(architecture)
            .ok_or_else(|| PhantomGpuError::InvalidModel {
                reason: format!("No power profile found for architecture: {}", architecture),
            })?;

        let workload_profile = self.workload_profiles
            .get(workload_type)
            .ok_or_else(|| PhantomGpuError::InvalidModel {
                reason: format!("No workload profile found for: {}", workload_type),
            })?;

        // Calculate base power consumption
        let base_power = power_profile.base_power_watts;

        // Calculate compute power based on utilization
        let compute_utilization = workload_profile.compute_intensity;
        let compute_power =
            power_profile.tdp_watts * power_profile.compute_power_factor * compute_utilization;

        // Calculate memory power
        let memory_utilization = workload_profile.memory_intensity;
        let memory_power =
            power_profile.tdp_watts * power_profile.memory_power_factor * memory_utilization * 0.3; // Memory typically 30% of compute power

        // Calculate cooling power
        let cooling_power =
            (base_power + compute_power + memory_power) * (1.0 - power_profile.cooling_efficiency);

        // Apply thermal scaling if thermal state is provided
        let thermal_impact = if let Some(thermal_state) = thermal_state {
            self.calculate_thermal_power_impact(thermal_state, power_profile)
        } else {
            ThermalPowerImpact {
                base_temp_celsius: 25.0,
                peak_temp_celsius: 70.0,
                thermal_throttling_detected: false,
                power_scaling_factor: 1.0,
                cooling_overhead_watts: cooling_power * 0.1,
                thermal_efficiency_loss: 0.0,
            }
        };

        // Calculate total power with thermal scaling
        let total_power =
            (base_power + compute_power + memory_power + cooling_power) *
            thermal_impact.power_scaling_factor;

        // Calculate energy consumption
        let energy_consumption_wh = total_power * (duration_seconds / 3600.0);
        let energy_cost_usd = energy_consumption_wh * (self.energy_cost_per_kwh / 1000.0);

        Ok(PowerConsumptionResult {
            gpu_name: gpu_model.name.clone(),
            workload_type: workload_type.to_string(),
            base_power_watts: base_power,
            compute_power_watts: compute_power,
            memory_power_watts: memory_power,
            cooling_power_watts: cooling_power,
            total_power_watts: total_power,
            duration_seconds,
            energy_consumption_wh,
            energy_cost_usd,
            thermal_impact,
        })
    }

    /// Calculate thermal impact on power consumption
    fn calculate_thermal_power_impact(
        &self,
        thermal_state: &ThermalState,
        power_profile: &PowerProfile
    ) -> ThermalPowerImpact {
        let temp_diff = thermal_state.current_temp_celsius - thermal_state.ambient_temp_celsius;
        let thermal_headroom =
            power_profile.thermal_design_point - thermal_state.current_temp_celsius;

        // Determine if thermal throttling is occurring
        let thermal_throttling_detected =
            thermal_state.current_temp_celsius > power_profile.thermal_design_point;

        // Calculate power scaling factor
        let power_scaling_factor = if thermal_throttling_detected {
            // Reduce power when thermal throttling occurs
            let throttle_factor = (thermal_headroom / 10.0).max(0.5).min(1.0);
            throttle_factor
        } else {
            1.0
        };

        // Calculate cooling overhead
        let cooling_overhead_watts = (temp_diff / 50.0) * power_profile.tdp_watts * 0.1;

        // Calculate thermal efficiency loss
        let thermal_efficiency_loss = if thermal_throttling_detected {
            (thermal_state.current_temp_celsius - power_profile.thermal_design_point) / 10.0
        } else {
            0.0
        };

        ThermalPowerImpact {
            base_temp_celsius: thermal_state.ambient_temp_celsius,
            peak_temp_celsius: thermal_state.current_temp_celsius,
            thermal_throttling_detected,
            power_scaling_factor,
            cooling_overhead_watts,
            thermal_efficiency_loss,
        }
    }

    /// Calculate power efficiency metrics
    pub fn calculate_power_efficiency(
        &self,
        gpu_model: &GpuModel,
        workload_type: &str,
        performance_score: f64,
        power_consumption: &PowerConsumptionResult,
        comparison_gpus: &[GpuModel]
    ) -> PhantomResult<PowerEfficiencyMetrics> {
        // Calculate basic efficiency metrics
        let efficiency_score = performance_score / power_consumption.total_power_watts;
        let samples_per_watt = performance_score / power_consumption.total_power_watts;
        let operations_per_joule =
            performance_score / (power_consumption.total_power_watts * 3600.0);

        // Calculate efficiency rating
        let efficiency_rating = match efficiency_score {
            e if e >= 2.0 => "Excellent".to_string(),
            e if e >= 1.5 => "Good".to_string(),
            e if e >= 1.0 => "Fair".to_string(),
            _ => "Poor".to_string(),
        };

        // Calculate comparison metrics
        let comparison_metrics = self.calculate_comparison_metrics(
            gpu_model,
            workload_type,
            efficiency_score,
            power_consumption.total_power_watts,
            performance_score,
            comparison_gpus
        )?;

        Ok(PowerEfficiencyMetrics {
            gpu_name: gpu_model.name.clone(),
            workload_type: workload_type.to_string(),
            performance_score,
            power_consumption_watts: power_consumption.total_power_watts,
            efficiency_score,
            efficiency_rating,
            samples_per_watt,
            operations_per_joule,
            comparison_metrics,
        })
    }

    /// Calculate comparison metrics against other GPUs
    fn calculate_comparison_metrics(
        &self,
        gpu_model: &GpuModel,
        workload_type: &str,
        efficiency_score: f64,
        power_consumption: f64,
        performance_score: f64,
        comparison_gpus: &[GpuModel]
    ) -> PhantomResult<PowerComparisonMetrics> {
        let mut efficiency_scores = Vec::new();
        let mut power_consumptions = Vec::new();
        let mut performance_scores = Vec::new();

        // Calculate metrics for comparison GPUs
        for gpu in comparison_gpus {
            if gpu.name == gpu_model.name {
                continue; // Skip self
            }

            // Estimate efficiency for comparison GPU
            let estimated_power = self.estimate_power_consumption(gpu, workload_type)?;
            let estimated_performance = self.estimate_performance_score(gpu, workload_type)?;
            let estimated_efficiency = estimated_performance / estimated_power;

            efficiency_scores.push(estimated_efficiency);
            power_consumptions.push(estimated_power);
            performance_scores.push(estimated_performance);
        }

        // Add current GPU to comparison
        efficiency_scores.push(efficiency_score);
        power_consumptions.push(power_consumption);
        performance_scores.push(performance_score);

        // Calculate ranking and percentiles
        efficiency_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let efficiency_rank =
            efficiency_scores
                .iter()
                .position(|&x| x == efficiency_score)
                .unwrap_or(0) + 1;
        let efficiency_percentile =
            100.0 * (1.0 - (efficiency_rank as f64) / (efficiency_scores.len() as f64));

        // Calculate averages
        let avg_power = power_consumptions.iter().sum::<f64>() / (power_consumptions.len() as f64);
        let avg_performance =
            performance_scores.iter().sum::<f64>() / (performance_scores.len() as f64);
        let avg_efficiency =
            efficiency_scores.iter().sum::<f64>() / (efficiency_scores.len() as f64);

        let power_vs_average = ((power_consumption - avg_power) / avg_power) * 100.0;
        let performance_vs_average =
            ((performance_score - avg_performance) / avg_performance) * 100.0;
        let cost_efficiency_score = (efficiency_score / avg_efficiency) * 100.0;

        Ok(PowerComparisonMetrics {
            efficiency_rank,
            efficiency_percentile,
            power_vs_average,
            performance_vs_average,
            cost_efficiency_score,
        })
    }

    /// Estimate power consumption for a GPU
    fn estimate_power_consumption(
        &self,
        gpu_model: &GpuModel,
        workload_type: &str
    ) -> PhantomResult<f64> {
        let unknown_arch = "Unknown".to_string();
        let architecture = gpu_model.architecture.as_ref().unwrap_or(&unknown_arch);
        let power_profile = self.power_profiles
            .get(architecture)
            .ok_or_else(|| PhantomGpuError::InvalidModel {
                reason: format!("No power profile found for architecture: {}", architecture),
            })?;

        let workload_profile = self.workload_profiles
            .get(workload_type)
            .ok_or_else(|| PhantomGpuError::InvalidModel {
                reason: format!("No workload profile found for: {}", workload_type),
            })?;

        // Simple estimation based on TDP and workload intensity
        let estimated_power =
            power_profile.base_power_watts +
            power_profile.tdp_watts * workload_profile.compute_intensity * 0.8;

        Ok(estimated_power)
    }

    /// Estimate performance score for a GPU
    fn estimate_performance_score(
        &self,
        gpu_model: &GpuModel,
        workload_type: &str
    ) -> PhantomResult<f64> {
        let workload_profile = self.workload_profiles
            .get(workload_type)
            .ok_or_else(|| PhantomGpuError::InvalidModel {
                reason: format!("No workload profile found for: {}", workload_type),
            })?;

        // Estimate based on compute TFLOPS and memory bandwidth
        let compute_score = (gpu_model.compute_tflops as f64) * workload_profile.compute_intensity;
        let memory_score =
            (gpu_model.memory_bandwidth_gbps as f64) * workload_profile.memory_intensity * 0.1;

        Ok(compute_score + memory_score)
    }

    /// Set energy cost per kWh
    pub fn set_energy_cost(&mut self, cost_per_kwh: f64) {
        self.energy_cost_per_kwh = cost_per_kwh;
    }

    /// Get available workload types
    pub fn get_workload_types(&self) -> Vec<String> {
        self.workload_profiles.keys().cloned().collect()
    }

    /// Get available architectures
    pub fn get_architectures(&self) -> Vec<String> {
        self.power_profiles.keys().cloned().collect()
    }

    /// Generate power efficiency report
    pub fn generate_power_report(
        &self,
        results: &[PowerConsumptionResult],
        efficiency_metrics: &[PowerEfficiencyMetrics]
    ) -> String {
        let mut report = String::new();
        report.push_str("🔋 Power Efficiency Report\n");
        report.push_str(&"=".repeat(50));
        report.push_str("\n\n");

        // Power consumption summary
        report.push_str("📊 Power Consumption Summary:\n");
        for result in results {
            report.push_str(
                &format!(
                    "   • {}: {:.1}W ({:.2} Wh over {:.1}s)\n",
                    result.gpu_name,
                    result.total_power_watts,
                    result.energy_consumption_wh,
                    result.duration_seconds
                )
            );
        }

        // Efficiency rankings
        report.push_str("\n⚡ Efficiency Rankings:\n");
        let mut sorted_efficiency = efficiency_metrics.to_vec();
        sorted_efficiency.sort_by(|a, b|
            b.efficiency_score.partial_cmp(&a.efficiency_score).unwrap()
        );

        for (rank, metrics) in sorted_efficiency.iter().enumerate() {
            let rank_symbol = match rank {
                0 => "🥇",
                1 => "🥈",
                2 => "🥉",
                _ => "  ",
            };
            report.push_str(
                &format!(
                    "   {} {}: {:.2} perf/W ({})\n",
                    rank_symbol,
                    metrics.gpu_name,
                    metrics.efficiency_score,
                    metrics.efficiency_rating
                )
            );
        }

        // Cost analysis
        report.push_str("\n💰 Energy Cost Analysis:\n");
        for result in results {
            report.push_str(
                &format!(
                    "   • {}: ${:.4} ({:.2}¢/hour)\n",
                    result.gpu_name,
                    result.energy_cost_usd,
                    (result.energy_cost_usd * 100.0) / (result.duration_seconds / 3600.0)
                )
            );
        }

        report
    }
}

impl Default for PowerModelingEngine {
    fn default() -> Self {
        Self::new()
    }
}
