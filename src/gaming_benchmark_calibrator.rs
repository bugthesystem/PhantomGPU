use crate::models::GamingWorkloadConfig;
use crate::gpu_config::GpuModel;
use crate::errors::PhantomResult;
use serde::{ Deserialize, Serialize };
use std::collections::HashMap;
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkEntry {
    pub gpu_name: String,
    pub gpu_architecture: String,
    pub game_name: String,
    pub measurements: Vec<BenchmarkMeasurement>,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct CalibrationFactor {
    pub base_performance_multiplier: f64,
    pub resolution_scaling: HashMap<String, f64>,
    pub ray_tracing_penalty: f64,
    pub dlss_boost: HashMap<String, f64>,
    pub confidence_score: f64,
}

pub struct GamingBenchmarkCalibrator {
    benchmark_data: Vec<BenchmarkEntry>,
    edge_cases: Vec<BenchmarkEntry>,
    calibration_factors: HashMap<String, HashMap<String, CalibrationFactor>>, // GPU -> Game -> Factors
}

impl GamingBenchmarkCalibrator {
    pub fn new() -> Self {
        let benchmark_data = Self::load_benchmark_data();
        let edge_cases = Self::load_edge_cases();

        println!("🎯 Gaming Benchmark Calibrator Initialization:");
        println!("   Benchmark entries loaded: {}", benchmark_data.len());
        println!("   Edge case entries loaded: {}", edge_cases.len());

        let calibration_factors = Self::compute_calibration_factors(&benchmark_data, &edge_cases);

        println!("   Calibration factors computed for {} GPUs", calibration_factors.len());
        for (gpu, games) in &calibration_factors {
            println!("   {} -> {} games calibrated", gpu, games.len());
        }

        Self {
            benchmark_data,
            edge_cases,
            calibration_factors,
        }
    }

    fn load_benchmark_data() -> Vec<BenchmarkEntry> {
        match fs::read_to_string("benchmark_data/gaming_benchmarks.json") {
            Ok(content) => {
                match serde_json::from_str::<Vec<BenchmarkEntry>>(&content) {
                    Ok(data) => data,
                    Err(e) => {
                        println!("Warning: Failed to parse gaming benchmark data: {}", e);
                        Vec::new()
                    }
                }
            }
            Err(_) => Vec::new(),
        }
    }

    fn load_edge_cases() -> Vec<BenchmarkEntry> {
        match fs::read_to_string("benchmark_data/gaming_edge_cases.json") {
            Ok(content) => {
                match serde_json::from_str::<Vec<BenchmarkEntry>>(&content) {
                    Ok(data) => data,
                    Err(e) => {
                        println!("Warning: Failed to parse gaming edge cases: {}", e);
                        Vec::new()
                    }
                }
            }
            Err(_) => Vec::new(),
        }
    }

    fn compute_calibration_factors(
        benchmark_data: &[BenchmarkEntry],
        edge_cases: &[BenchmarkEntry]
    ) -> HashMap<String, HashMap<String, CalibrationFactor>> {
        let mut factors = HashMap::new();

        // Combine benchmark data and edge cases
        let all_data: Vec<&BenchmarkEntry> = benchmark_data
            .iter()
            .chain(edge_cases.iter())
            .collect();

        for entry in all_data {
            let gpu_factors = factors.entry(entry.gpu_name.clone()).or_insert_with(HashMap::new);

            if let Some(calibration) = Self::analyze_game_performance(entry) {
                gpu_factors.insert(entry.game_name.clone(), calibration);
            }
        }

        factors
    }

    fn analyze_game_performance(entry: &BenchmarkEntry) -> Option<CalibrationFactor> {
        if entry.measurements.is_empty() {
            return None;
        }

        // Analyze resolution scaling
        let mut resolution_scaling = HashMap::new();
        let mut dlss_boost = HashMap::new();
        let mut ray_tracing_measurements = Vec::new();
        let mut base_measurements = Vec::new();

        for measurement in &entry.measurements {
            // Track resolution scaling
            resolution_scaling.insert(measurement.resolution.clone(), measurement.avg_fps);

            // Track DLSS performance boost
            if measurement.dlss_mode != "Off" {
                dlss_boost.insert(measurement.dlss_mode.clone(), measurement.avg_fps);
            }

            // Separate ray tracing vs non-ray tracing
            if measurement.ray_tracing {
                ray_tracing_measurements.push(measurement.avg_fps);
            } else {
                base_measurements.push(measurement.avg_fps);
            }
        }

        // Calculate base performance multiplier (average FPS at 1080p)
        let base_1080p_fps = resolution_scaling
            .get("1920x1080")
            .copied()
            .unwrap_or(
                entry.measurements
                    .iter()
                    .filter(|m| !m.ray_tracing && m.dlss_mode == "Off")
                    .map(|m| m.avg_fps)
                    .next()
                    .unwrap_or(60.0)
            );

        // Normalize resolution scaling relative to 1080p
        let mut normalized_resolution_scaling = HashMap::new();
        for (resolution, fps) in resolution_scaling.iter() {
            normalized_resolution_scaling.insert(resolution.clone(), fps / base_1080p_fps);
        }

        // Calculate ray tracing penalty
        let ray_tracing_penalty = if
            !ray_tracing_measurements.is_empty() &&
            !base_measurements.is_empty()
        {
            let avg_rt_fps: f64 =
                ray_tracing_measurements.iter().sum::<f64>() /
                (ray_tracing_measurements.len() as f64);
            let avg_base_fps: f64 =
                base_measurements.iter().sum::<f64>() / (base_measurements.len() as f64);
            avg_rt_fps / avg_base_fps // Ratio of RT to non-RT performance
        } else {
            0.7 // Default 30% penalty for ray tracing
        };

        // Normalize DLSS boost
        let mut normalized_dlss_boost = HashMap::new();
        for (dlss_mode, fps) in dlss_boost.iter() {
            normalized_dlss_boost.insert(dlss_mode.clone(), fps / base_1080p_fps);
        }

        // Calculate confidence score based on measurement count and variance
        let confidence_score = ((entry.measurements.len() as f64) / 10.0).min(1.0) * 0.9 + 0.1;

        Some(CalibrationFactor {
            base_performance_multiplier: base_1080p_fps / 100.0, // Normalize to ~1.0 for 100 FPS baseline
            resolution_scaling: normalized_resolution_scaling,
            ray_tracing_penalty,
            dlss_boost: normalized_dlss_boost,
            confidence_score,
        })
    }

    /// Calibrate gaming performance prediction using benchmark data
    pub fn calibrate_gaming_performance(
        &self,
        workload: &GamingWorkloadConfig,
        gpu_model: &GpuModel,
        theoretical_fps: f64
    ) -> f64 {
        // Get calibration factors for this GPU and game
        let gpu_factors = self.calibration_factors.get(&gpu_model.name);
        let game_factors = gpu_factors.and_then(|factors| factors.get(&workload.game_name));

        println!("🎯 Calibrating {} on {}:", workload.game_name, gpu_model.name);
        println!("   Theoretical FPS: {:.1}", theoretical_fps);

        let raw_calibrated = if let Some(factors) = game_factors {
            println!(
                "   Using calibration factors (confidence: {:.1}%)",
                factors.confidence_score * 100.0
            );
            self.apply_calibration_factors(workload, gpu_model, theoretical_fps, factors)
        } else {
            println!("   No calibration factors found, interpolating...");
            self.interpolate_performance(workload, gpu_model, theoretical_fps)
        };

        // Apply safety cap to prevent absurdly high FPS predictions
        let capped_fps = self.apply_safety_cap(raw_calibrated, workload, &gpu_model.name);
        println!("   Final FPS: {:.1}", capped_fps);
        capped_fps
    }

    fn apply_safety_cap(&self, fps: f64, workload: &GamingWorkloadConfig, gpu_name: &str) -> f64 {
        // Define reasonable maximum FPS based on game and GPU
        let max_fps = match workload.game_name.as_str() {
            "Valorant" => {
                match gpu_name {
                    "RTX 5090" => 600.0,
                    "RTX 4090" => 500.0,
                    "RTX 4080" => 400.0,
                    _ => 350.0,
                }
            }
            "Overwatch 2" => {
                match gpu_name {
                    "RTX 5090" => 400.0,
                    "RTX 4090" => 350.0,
                    "RTX 4080" => 300.0,
                    _ => 250.0,
                }
            }
            "Fortnite" | "Apex Legends" => {
                match gpu_name {
                    "RTX 5090" => 300.0,
                    "RTX 4090" => 250.0,
                    "RTX 4080" => 200.0,
                    _ => 180.0,
                }
            }
            "Call of Duty: Modern Warfare III" => {
                match gpu_name {
                    "RTX 5090" => 250.0,
                    "RTX 4090" => 200.0,
                    "RTX 4080" => 150.0,
                    _ => 120.0,
                }
            }
            "Cyberpunk 2077" | "Hogwarts Legacy" => {
                match gpu_name {
                    "RTX 5090" => 150.0,
                    "RTX 4090" => 120.0,
                    "RTX 4080" => 100.0,
                    _ => 80.0,
                }
            }
            _ => 200.0, // Default cap
        };

        // Apply resolution penalty to cap
        let resolution_penalty = self.get_resolution_penalty(workload.resolution);
        let adjusted_cap = max_fps * resolution_penalty;

        fps.min(adjusted_cap)
    }

    fn apply_calibration_factors(
        &self,
        workload: &GamingWorkloadConfig,
        gpu_model: &GpuModel,
        theoretical_fps: f64,
        factors: &CalibrationFactor
    ) -> f64 {
        // 🎯 NEW APPROACH: Use benchmark data as primary source, not theoretical scaling

        // Find the closest matching benchmark measurement
        if let Some(benchmark_fps) = self.find_closest_benchmark_fps(workload, &gpu_model.name) {
            println!("   Found benchmark FPS: {:.1}", benchmark_fps);

            // Apply minor adjustments for differences from benchmark conditions
            let mut adjusted_fps = benchmark_fps;

            // Apply DLSS boost if different from benchmark
            let dlss_key = format!("{:?}", workload.dlss_mode);
            if let Some(dlss_factor) = factors.dlss_boost.get(&dlss_key) {
                adjusted_fps *= dlss_factor;
            } else {
                adjusted_fps *= self.get_default_dlss_scaling(&workload.dlss_mode);
            }

            // Apply ray tracing penalty if different from benchmark
            if workload.ray_tracing {
                adjusted_fps *= factors.ray_tracing_penalty;
            }

            println!("   Benchmark-based FPS: {:.1}", adjusted_fps);

            // High confidence in benchmark data - use it directly with minor blending
            let benchmark_confidence = 0.95;
            adjusted_fps * benchmark_confidence + theoretical_fps * (1.0 - benchmark_confidence)
        } else {
            // Fallback to original scaling approach when no benchmark data
            self.apply_scaling_based_calibration(workload, gpu_model, theoretical_fps, factors)
        }
    }

    fn find_closest_benchmark_fps(
        &self,
        workload: &GamingWorkloadConfig,
        gpu_name: &str
    ) -> Option<f64> {
        // Search benchmark data for closest match
        let target_resolution = format!("{}x{}", workload.resolution.0, workload.resolution.1);
        let target_dlss = format!("{:?}", workload.dlss_mode);

        for entry in &self.benchmark_data {
            if entry.gpu_name == gpu_name && entry.game_name == workload.game_name {
                for measurement in &entry.measurements {
                    // Look for exact match first
                    if
                        measurement.resolution == target_resolution &&
                        measurement.dlss_mode == target_dlss &&
                        measurement.ray_tracing == workload.ray_tracing
                    {
                        return Some(measurement.avg_fps);
                    }
                }

                // Look for close match (same resolution, different DLSS/RT)
                for measurement in &entry.measurements {
                    if measurement.resolution == target_resolution {
                        return Some(measurement.avg_fps);
                    }
                }

                // Use any measurement from this GPU/game combination
                if let Some(measurement) = entry.measurements.first() {
                    return Some(measurement.avg_fps);
                }
            }
        }

        // Check edge cases too
        for entry in &self.edge_cases {
            if entry.gpu_name == gpu_name && entry.game_name == workload.game_name {
                if let Some(measurement) = entry.measurements.first() {
                    return Some(measurement.avg_fps);
                }
            }
        }

        None
    }

    fn apply_scaling_based_calibration(
        &self,
        workload: &GamingWorkloadConfig,
        gpu_model: &GpuModel,
        theoretical_fps: f64,
        factors: &CalibrationFactor
    ) -> f64 {
        // Original scaling approach as fallback
        let gpu_performance_factor = self.get_gpu_performance_factor(&gpu_model.name);
        let mut calibrated_fps =
            factors.base_performance_multiplier * gpu_performance_factor * 100.0;

        // Apply resolution scaling
        let resolution_key = format!("{}x{}", workload.resolution.0, workload.resolution.1);
        if let Some(resolution_factor) = factors.resolution_scaling.get(&resolution_key) {
            calibrated_fps *= resolution_factor;
        } else {
            calibrated_fps *= self.interpolate_resolution_scaling(workload.resolution, factors);
        }

        // Apply ray tracing penalty
        if workload.ray_tracing {
            calibrated_fps *= factors.ray_tracing_penalty;
        }

        // Apply DLSS boost
        let dlss_key = format!("{:?}", workload.dlss_mode);
        if let Some(dlss_factor) = factors.dlss_boost.get(&dlss_key) {
            calibrated_fps *= dlss_factor;
        } else {
            calibrated_fps *= self.get_default_dlss_scaling(&workload.dlss_mode);
        }

        // Ultra-aggressive theoretical value reduction for extreme cases
        let theoretical_blend = if theoretical_fps > 10000.0 {
            0.01 // Only 1% theoretical for extreme values
        } else if theoretical_fps > 1000.0 {
            0.05 // 5% theoretical for very high values
        } else {
            0.15 // 15% theoretical for normal high values
        };

        calibrated_fps * (1.0 - theoretical_blend) + theoretical_fps * theoretical_blend
    }

    fn interpolate_performance(
        &self,
        workload: &GamingWorkloadConfig,
        gpu_model: &GpuModel,
        theoretical_fps: f64
    ) -> f64 {
        // Try to find similar benchmark data first
        if let Some(similar_fps) = self.find_similar_benchmark_fps(workload, &gpu_model.name) {
            println!("   Found similar benchmark FPS: {:.1}", similar_fps);
            return similar_fps;
        }

        // Fallback to aggressive scaling of theoretical values
        let similar_gpu_factor = self.find_similar_gpu_performance(&gpu_model.name);
        let similar_game_factor = self.find_similar_game_performance(&workload.game_name);

        // Apply aggressive scaling to bring down unrealistic theoretical values
        let realistic_baseline = self.get_realistic_baseline_fps(workload, &gpu_model.name);
        let scaling_factor = similar_gpu_factor * similar_game_factor;

        let interpolated_fps = realistic_baseline * scaling_factor;

        // Ultra-aggressive theoretical value reduction for extreme cases
        let theoretical_blend = if theoretical_fps > 10000.0 {
            0.01 // Only 1% theoretical for extreme values
        } else if theoretical_fps > 1000.0 {
            0.05 // 5% theoretical for very high values
        } else {
            0.1 // 10% theoretical for normal high values
        };

        interpolated_fps * (1.0 - theoretical_blend) + theoretical_fps * theoretical_blend
    }

    fn find_similar_benchmark_fps(
        &self,
        workload: &GamingWorkloadConfig,
        gpu_name: &str
    ) -> Option<f64> {
        // Look for similar games on the same GPU
        let similar_games = self.get_similar_games(&workload.game_name);

        for entry in &self.benchmark_data {
            if entry.gpu_name == gpu_name && similar_games.contains(&entry.game_name.as_str()) {
                if let Some(measurement) = entry.measurements.first() {
                    println!(
                        "   Found similar game {} -> {:.1} FPS",
                        entry.game_name,
                        measurement.avg_fps
                    );
                    return Some(measurement.avg_fps);
                }
            }
        }

        // Look for same game on similar GPUs
        let similar_gpus = self.get_similar_gpus(gpu_name);

        for entry in &self.benchmark_data {
            if
                entry.game_name == workload.game_name &&
                similar_gpus.contains(&entry.gpu_name.as_str())
            {
                if let Some(measurement) = entry.measurements.first() {
                    // Scale based on GPU tier difference
                    let scaling =
                        self.get_gpu_performance_factor(gpu_name) /
                        self.get_gpu_performance_factor(&entry.gpu_name);
                    let scaled_fps = measurement.avg_fps * scaling;
                    println!(
                        "   Found similar GPU {} -> {:.1} FPS (scaled from {:.1})",
                        entry.gpu_name,
                        scaled_fps,
                        measurement.avg_fps
                    );
                    return Some(scaled_fps);
                }
            }
        }

        println!("   No similar benchmarks found for {} on {}", workload.game_name, gpu_name);
        None
    }

    fn get_realistic_baseline_fps(&self, workload: &GamingWorkloadConfig, gpu_name: &str) -> f64 {
        // More conservative baseline FPS expectations aligned with actual benchmark data
        let gpu_tier_multiplier = self.get_gpu_performance_factor(gpu_name);
        let resolution_penalty = self.get_resolution_penalty(workload.resolution);

        let game_baseline = match workload.game_name.as_str() {
            "Valorant" => 300.0, // Well-optimized competitive (reduced from 400)
            "Overwatch 2" => 180.0, // Competitive esports (reduced from 200)
            "Fortnite" => 120.0, // Battle royale (reduced from 150)
            "Apex Legends" => 100.0, // BR with good optimization (reduced from 120)
            "Call of Duty: Modern Warfare III" => 90.0, // Modern FPS (reduced from 100)
            "Cyberpunk 2077" => 70.0, // Demanding open world (reduced from 80)
            "Hogwarts Legacy" => 60.0, // Poorly optimized (reduced from 70)
            _ => 90.0, // Default (reduced from 100)
        };

        // Apply additional conservative factor for ultra-high theoretical values
        let conservative_factor = if workload.game_name == "Valorant" {
            0.7 // Extra conservative for Valorant to counter extreme theoretical values
        } else {
            1.0
        };

        game_baseline * gpu_tier_multiplier * resolution_penalty * conservative_factor
    }

    fn get_similar_games(&self, game_name: &str) -> Vec<&str> {
        match game_name {
            "Valorant" => vec!["Overwatch 2", "Fortnite"],
            "Overwatch 2" => vec!["Valorant", "Apex Legends"],
            "Fortnite" => vec!["Apex Legends", "Valorant"],
            "Apex Legends" => vec!["Fortnite", "Overwatch 2"],
            "Call of Duty: Modern Warfare III" => vec!["Apex Legends", "Overwatch 2"],
            "Cyberpunk 2077" => vec!["Hogwarts Legacy"],
            "Hogwarts Legacy" => vec!["Cyberpunk 2077"],
            _ => vec![],
        }
    }

    fn get_similar_gpus(&self, gpu_name: &str) -> Vec<&str> {
        match gpu_name {
            "RTX 4090" => vec!["RTX 5090", "RTX 4080"],
            "RTX 4080" => vec!["RTX 4090", "RTX 5090"],
            "RTX 5090" => vec!["RTX 4090", "RTX 4080"],
            _ => vec![],
        }
    }

    fn get_resolution_penalty(&self, resolution: (u32, u32)) -> f64 {
        let pixel_count = (resolution.0 * resolution.1) as f64;
        let base_1080p = 1920.0 * 1080.0;

        match pixel_count {
            p if p <= base_1080p => 1.0,
            p if p <= 2560.0 * 1440.0 => 0.7,
            p if p <= 3840.0 * 2160.0 => 0.4,
            _ => 0.25,
        }
    }

    fn get_gpu_performance_factor(&self, gpu_name: &str) -> f64 {
        match gpu_name {
            "RTX 5090" => 1.3, // Next-gen performance
            "RTX 4090" => 1.0, // Baseline flagship
            "RTX 4080" => 0.75, // High-end but not flagship
            _ => {
                // Estimate based on TFLOPS or other specs
                0.8
            }
        }
    }

    fn interpolate_resolution_scaling(
        &self,
        resolution: (u32, u32),
        factors: &CalibrationFactor
    ) -> f64 {
        let pixel_count = (resolution.0 * resolution.1) as f64;

        // Common resolution pixel counts
        let res_1080p = 1920.0 * 1080.0;
        let res_1440p = 2560.0 * 1440.0;
        let res_4k = 3840.0 * 2160.0;

        if pixel_count <= res_1080p {
            factors.resolution_scaling.get("1920x1080").copied().unwrap_or(1.0)
        } else if pixel_count <= res_1440p {
            let factor_1080p = factors.resolution_scaling.get("1920x1080").copied().unwrap_or(1.0);
            let factor_1440p = factors.resolution_scaling.get("2560x1440").copied().unwrap_or(0.7);
            Self::lerp(
                factor_1080p,
                factor_1440p,
                (pixel_count - res_1080p) / (res_1440p - res_1080p)
            )
        } else {
            let factor_1440p = factors.resolution_scaling.get("2560x1440").copied().unwrap_or(0.7);
            let factor_4k = factors.resolution_scaling.get("3840x2160").copied().unwrap_or(0.4);
            Self::lerp(factor_1440p, factor_4k, (pixel_count - res_1440p) / (res_4k - res_1440p))
        }
    }

    fn get_default_dlss_scaling(&self, dlss_mode: &crate::models::DLSSMode) -> f64 {
        match dlss_mode {
            crate::models::DLSSMode::Off => 1.0,
            crate::models::DLSSMode::Quality => 1.25,
            crate::models::DLSSMode::Balanced => 1.4,
            crate::models::DLSSMode::Performance => 1.7,
            crate::models::DLSSMode::UltraPerformance => 2.0,
        }
    }

    fn find_similar_gpu_performance(&self, gpu_name: &str) -> f64 {
        // Find GPUs with similar architecture or performance tier
        match gpu_name {
            name if name.contains("RTX 50") => 1.2, // Next-gen estimate
            name if name.contains("RTX 40") => 1.0, // Current gen
            name if name.contains("RTX 30") => 0.8, // Previous gen
            _ => 0.9,
        }
    }

    fn find_similar_game_performance(&self, game_name: &str) -> f64 {
        // Categorize games by complexity and optimization level
        match game_name {
            "Valorant" | "Overwatch 2" => 1.5, // Well-optimized competitive games
            "Fortnite" | "Apex Legends" => 1.2, // Decent optimization
            "Call of Duty: Modern Warfare III" => 1.0, // Modern demanding
            "Cyberpunk 2077" | "Hogwarts Legacy" => 0.8, // Poorly optimized demanding games
            _ => 1.0,
        }
    }

    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + (b - a) * t.clamp(0.0, 1.0)
    }

    /// Get calibration statistics
    pub fn get_calibration_stats(&self) -> String {
        let total_entries = self.benchmark_data.len() + self.edge_cases.len();
        let total_measurements: usize = self.benchmark_data
            .iter()
            .chain(self.edge_cases.iter())
            .map(|entry| entry.measurements.len())
            .sum();

        let calibrated_gpus = self.calibration_factors.len();
        let calibrated_games: usize = self.calibration_factors
            .values()
            .map(|games| games.len())
            .sum();

        format!(
            "Gaming Benchmark Calibrator Stats:\n\
            • Total benchmark entries: {}\n\
            • Total measurements: {}\n\
            • Calibrated GPUs: {}\n\
            • Calibrated game combinations: {}",
            total_entries,
            total_measurements,
            calibrated_gpus,
            calibrated_games
        )
    }
}
