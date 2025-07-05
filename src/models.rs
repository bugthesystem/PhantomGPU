//! Model configurations and emulation profiles

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub batch_size: usize,
    pub input_shape: Vec<usize>,
    pub parameters_m: f32, // Parameters in millions
    pub flops_per_sample_g: f32, // FLOPS per sample in billions
}

impl ModelConfig {
    pub fn resnet50(batch_size: usize) -> Self {
        Self {
            name: "ResNet50".to_string(),
            batch_size,
            input_shape: vec![3, 224, 224],
            parameters_m: 25.6,
            flops_per_sample_g: 4.1,
        }
    }

    pub fn alexnet(batch_size: usize) -> Self {
        Self {
            name: "AlexNet".to_string(),
            batch_size,
            input_shape: vec![3, 224, 224],
            parameters_m: 61.0,
            flops_per_sample_g: 0.7,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmulationProfile {
    pub forward_time_ms: f64,
    pub backward_time_ms: f64,
    pub data_transfer_time_ms: f64,
    pub memory_usage_mb: f64,
    pub preprocessing_time_ms: f64,
}

impl EmulationProfile {
    /// Estimate profile based on model characteristics and GPU specs
    pub fn estimate(model: &ModelConfig, gpu: &crate::gpu_config::GpuModel) -> Self {
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
        }
    }
}
