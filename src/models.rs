//! Model configurations and emulation profiles

use crate::bottleneck_analysis::{ BottleneckAnalyzer, BottleneckType };

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub batch_size: usize,
    pub input_shape: Vec<usize>,
    pub parameters_m: f32, // Parameters in millions
    pub flops_per_sample_g: f32, // FLOPS per sample in billions
    pub model_type: String, // Model type for bottleneck analysis
    pub precision: String, // Precision (fp32, fp16, int8, int4)
}

impl ModelConfig {
    pub fn resnet50(batch_size: usize) -> Self {
        Self {
            name: "ResNet50".to_string(),
            batch_size,
            input_shape: vec![3, 224, 224],
            parameters_m: 25.6,
            flops_per_sample_g: 4.1,
            model_type: "cnn".to_string(),
            precision: "fp32".to_string(),
        }
    }

    pub fn alexnet(batch_size: usize) -> Self {
        Self {
            name: "AlexNet".to_string(),
            batch_size,
            input_shape: vec![3, 224, 224],
            parameters_m: 61.0,
            flops_per_sample_g: 0.7,
            model_type: "cnn".to_string(),
            precision: "fp32".to_string(),
        }
    }

    /// Set model precision
    pub fn with_precision(mut self, precision: &str) -> Self {
        self.precision = precision.to_string();
        self
    }

    /// Set model type for bottleneck analysis
    pub fn with_model_type(mut self, model_type: &str) -> Self {
        self.model_type = model_type.to_string();
        self
    }
}

#[derive(Debug, Clone)]
pub struct EmulationProfile {
    pub forward_time_ms: f64,
    pub backward_time_ms: f64,
    pub data_transfer_time_ms: f64,
    pub memory_usage_mb: f64,
    pub preprocessing_time_ms: f64,
    pub bottleneck_type: Option<String>,
    pub performance_analysis: Option<String>,
}

impl EmulationProfile {
    /// Estimate profile based on model characteristics and GPU specs using bottleneck analysis
    pub fn estimate(model: &ModelConfig, gpu: &crate::gpu_config::GpuModel) -> Self {
        // Try to use bottleneck-aware analysis first
        if let Ok(analyzer) = BottleneckAnalyzer::new() {
            Self::estimate_with_bottleneck_analysis(model, gpu, &analyzer)
        } else {
            // Fallback to simple FLOPS-based calculation with warning
            println!("⚠️  Bottleneck analysis unavailable, using simplified FLOPS model");
            Self::estimate_fallback(model, gpu)
        }
    }

    /// Advanced bottleneck-aware estimation
    fn estimate_with_bottleneck_analysis(
        model: &ModelConfig,
        gpu: &crate::gpu_config::GpuModel,
        analyzer: &BottleneckAnalyzer
    ) -> Self {
        let batch_flops = model.flops_per_sample_g * (model.batch_size as f32);

        // Perform bottleneck analysis
        let analysis = analyzer
            .analyze_bottleneck(
                &gpu.name,
                &model.model_type,
                model.batch_size,
                model.flops_per_sample_g,
                &model.precision
            )
            .unwrap_or_else(|_| {
                // Fallback analysis if hardware profile not found
                crate::bottleneck_analysis::BottleneckAnalysis {
                    bottleneck_type: BottleneckType::Mixed,
                    memory_bound_factor: 0.5,
                    compute_bound_factor: 0.5,
                    effective_bandwidth_gbps: gpu.memory_bandwidth_gbps * 0.7,
                    effective_compute_tflops: gpu.compute_tflops * 0.8,
                    performance_multiplier: 1.0,
                }
            });

        // Calculate data size for memory transfer estimation
        let input_size_mb =
            ((model.batch_size as f32) *
                (model.input_shape.iter().product::<usize>() as f32) *
                4.0) /
            (1024.0 * 1024.0);

        // Use bottleneck-aware calculation for forward time
        let forward_time_ms = analyzer.calculate_inference_time(
            &analysis,
            batch_flops,
            input_size_mb
        );

        // Backward pass estimation (varies by model type and bottleneck)
        let backward_multiplier = match analysis.bottleneck_type {
            BottleneckType::Memory => 1.8, // Memory-bound models have lower backward multiplier
            BottleneckType::Compute => 2.2, // Compute-bound models have higher backward multiplier
            BottleneckType::Mixed => 2.0, // Mixed models use average multiplier
        };
        let backward_time_ms = forward_time_ms * backward_multiplier;

        // Data transfer time using effective bandwidth
        let data_transfer_time_ms = ((input_size_mb / analysis.effective_bandwidth_gbps) *
            8.0 *
            1000.0) as f64;

        // Memory usage estimation (consider precision)
        let precision_factor = match model.precision.as_str() {
            "fp16" => 2.0,
            "int8" => 1.0,
            "int4" => 0.5,
            _ => 4.0, // fp32
        };
        let memory_usage_mb = (model.parameters_m * precision_factor + // Parameters
            input_size_mb * 2.0) as f64; // Input + activations

        // Preprocessing time (varies by model complexity and bottleneck)
        let preprocessing_multiplier = match model.model_type.as_str() {
            "cnn" => 0.05, // CNNs have minimal preprocessing
            "transformer" => 0.15, // Transformers have more preprocessing
            "rnn" => 0.1, // RNNs have moderate preprocessing
            _ => 0.1,
        };
        let preprocessing_time_ms = forward_time_ms * preprocessing_multiplier;

        // Performance analysis summary
        let bottleneck_type_str = match analysis.bottleneck_type {
            BottleneckType::Memory => "Memory-bound",
            BottleneckType::Compute => "Compute-bound",
            BottleneckType::Mixed => "Mixed bottlenecks",
        };

        let performance_analysis = format!(
            "{} | {:.1}% memory intensity | Effective BW: {:.0} GB/s | Effective compute: {:.1} TFLOPS",
            bottleneck_type_str,
            analysis.memory_bound_factor * 100.0,
            analysis.effective_bandwidth_gbps,
            analysis.effective_compute_tflops
        );

        Self {
            forward_time_ms,
            backward_time_ms,
            data_transfer_time_ms,
            memory_usage_mb,
            preprocessing_time_ms,
            bottleneck_type: Some(bottleneck_type_str.to_string()),
            performance_analysis: Some(performance_analysis),
        }
    }

    /// Fallback to simple FLOPS-based calculation (for compatibility)
    fn estimate_fallback(model: &ModelConfig, gpu: &crate::gpu_config::GpuModel) -> Self {
        let batch_flops = model.flops_per_sample_g * (model.batch_size as f32);
        let forward_time_ms = ((batch_flops / gpu.compute_tflops) * 1000.0) as f64;
        let backward_time_ms = forward_time_ms * 2.0; // Backward pass ~2x forward

        // Data transfer estimation (input tensor size)
        let input_size_mb =
            ((model.batch_size as f32) *
                (model.input_shape.iter().product::<usize>() as f32) *
                4.0) /
            (1024.0 * 1024.0);
        let data_transfer_time_ms = ((input_size_mb / gpu.memory_bandwidth_gbps) *
            8.0 *
            1000.0) as f64;

        // Memory usage estimation
        let memory_usage_mb = (model.parameters_m * 4.0 + // Parameters (fp32)
            input_size_mb * 2.0) as f64; // Input + activations

        // Preprocessing (varies by model complexity)
        let preprocessing_time_ms = forward_time_ms * 0.1; // ~10% of compute time

        Self {
            forward_time_ms,
            backward_time_ms,
            data_transfer_time_ms,
            memory_usage_mb,
            preprocessing_time_ms,
            bottleneck_type: Some("Compute-bound (simplified)".to_string()),
            performance_analysis: Some("Using simplified FLOPS-based calculation".to_string()),
        }
    }
}
