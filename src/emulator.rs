//! Core GPU emulation and multi-GPU simulation

use std::time::{ Duration, Instant };
use std::collections::HashMap;
use std::sync::{ Arc, atomic::{ AtomicU64, Ordering } };
use tokio::time::sleep;
use futures::future::join_all;
use tracing::info;

use crate::gpu_config::GpuModel;
use crate::models::{ ModelConfig, EmulationProfile };

#[derive(Debug)]
pub struct RustGPUEmu {
    pub gpu_model: GpuModel,
    pub profile_cache: HashMap<String, EmulationProfile>,
    pub memory_used_bytes: Arc<AtomicU64>, // Thread-safe atomic memory tracking
    pub is_busy: bool,
    pub total_operations: usize,
}

impl RustGPUEmu {
    pub fn new(gpu_model: GpuModel) -> Self {
        Self {
            gpu_model,
            profile_cache: HashMap::new(),
            memory_used_bytes: Arc::new(AtomicU64::new(0)),
            is_busy: false,
            total_operations: 0,
        }
    }

    pub fn get_or_create_profile(&mut self, model: &ModelConfig) -> EmulationProfile {
        let key = format!("{}_{}", model.name, model.batch_size);

        if let Some(profile) = self.profile_cache.get(&key) {
            profile.clone()
        } else {
            let profile = EmulationProfile::estimate(model, &self.gpu_model);
            self.profile_cache.insert(key, profile.clone());
            profile
        }
    }

    pub async fn emulate_forward(&mut self, model: &ModelConfig) -> Result<Duration, String> {
        if self.is_busy {
            return Err("GPU is busy".to_string());
        }

        let profile = self.get_or_create_profile(model);

        // Check memory constraints
        let current_memory_bytes = self.memory_used_bytes.load(Ordering::Relaxed);
        let current_memory_mb = (current_memory_bytes as f64) / (1024.0 * 1024.0);
        let needed_memory_bytes = (profile.memory_usage_mb * 1024.0 * 1024.0) as u64;
        let total_memory_mb = (self.gpu_model.memory_gb * 1024.0) as f64;

        if current_memory_mb + profile.memory_usage_mb > (total_memory_mb as f64) {
            return Err(
                format!(
                    "OOM: Need {:.1}MB, Available {:.1}MB",
                    profile.memory_usage_mb,
                    total_memory_mb - current_memory_mb
                )
            );
        }

        self.is_busy = true;
        let new_total = self.memory_used_bytes.fetch_add(needed_memory_bytes, Ordering::Relaxed);
        tracing::debug!(
            "GPU {} allocated {:.1}MB, total: {:.1}MB",
            self.gpu_model.name,
            profile.memory_usage_mb,
            ((new_total + needed_memory_bytes) as f64) / (1024.0 * 1024.0)
        );

        let start = Instant::now();

        // Emulate data transfer (host -> GPU)
        println!(
            "📤 Transferring data: {:.2}MB",
            ((model.batch_size as f32) *
                (model.input_shape.iter().product::<usize>() as f32) *
                4.0) /
                (1024.0 * 1024.0)
        );
        sleep(Duration::from_millis(profile.data_transfer_time_ms as u64)).await;

        // Emulate forward computation
        println!("⚡ Forward pass: {} (batch_size: {})", model.name, model.batch_size);
        sleep(Duration::from_millis(profile.forward_time_ms as u64)).await;

        let elapsed = start.elapsed();
        self.is_busy = false;
        self.total_operations += 1;

        println!(
            "✅ Forward completed: {:.2}ms (estimated: {:.2}ms)",
            elapsed.as_millis(),
            profile.data_transfer_time_ms + profile.forward_time_ms
        );

        Ok(elapsed)
    }

    pub async fn emulate_backward(&mut self, model: &ModelConfig) -> Result<Duration, String> {
        if self.is_busy {
            return Err("GPU is busy".to_string());
        }

        let profile = self.get_or_create_profile(model);

        self.is_busy = true;
        let start = Instant::now();

        println!("🔙 Backward pass: {}", model.name);
        sleep(Duration::from_millis(profile.backward_time_ms as u64)).await;

        let elapsed = start.elapsed();
        self.is_busy = false;

        println!(
            "✅ Backward completed: {:.2}ms (estimated: {:.2}ms)",
            elapsed.as_millis(),
            profile.backward_time_ms
        );

        Ok(elapsed)
    }

    pub async fn emulate_training_step(&mut self, model: &ModelConfig) -> Result<Duration, String> {
        let step_start = Instant::now();

        self.emulate_forward(model).await?;
        self.emulate_backward(model).await?;

        let total_time = step_start.elapsed();
        println!("🎯 Training step completed: {:.2}ms\n", total_time.as_millis());

        Ok(total_time)
    }

    pub fn get_memory_info(&self) -> (f64, f64, f64) {
        let used_bytes = self.memory_used_bytes.load(Ordering::Relaxed);
        let used = (used_bytes as f64) / (1024.0 * 1024.0); // Convert bytes to MB
        let total = (self.gpu_model.memory_gb * 1024.0) as f64;
        let utilization = (used / total) * 100.0;
        (used, total, utilization)
    }

    pub fn reset_memory(&mut self) {
        self.memory_used_bytes.store(0, Ordering::Relaxed);
    }

    pub fn get_stats(&self) -> String {
        let (used, total, util) = self.get_memory_info();
        format!(
            "GPU: {} | Memory: {:.1}/{:.1}MB ({:.1}%) | Operations: {}",
            self.gpu_model.name,
            used,
            total,
            util,
            self.total_operations
        )
    }
}

/// Multi-GPU emulation for distributed scenarios
pub struct MultiGPUEmulator {
    pub emulators: Vec<RustGPUEmu>,
    pub communication_latency_ms: f64,
}

impl MultiGPUEmulator {
    pub fn new(gpu_models: Vec<GpuModel>, communication_latency_ms: f64) -> Self {
        let emulators = gpu_models
            .into_iter()
            .map(|model| RustGPUEmu::new(model))
            .collect();

        Self {
            emulators,
            communication_latency_ms,
        }
    }

    pub async fn emulate_data_parallel_training(
        &mut self,
        model: &ModelConfig,
        epochs: usize
    ) -> Result<Vec<Duration>, String> {
        let mut epoch_times = Vec::new();
        let gpus_count = self.emulators.len();

        info!("🔥 Starting distributed training on {} GPUs", gpus_count);
        info!("Model: {} | Batch size per GPU: {}", model.name, model.batch_size);
        println!("🔥 Starting distributed training on {} GPUs", gpus_count);
        println!("Model: {} | Batch size per GPU: {}\n", model.name, model.batch_size);

        for epoch in 0..epochs {
            let epoch_start = Instant::now();

            // Simulate data parallel training
            let mut gpu_futures = Vec::new();

            for (gpu_id, emulator) in self.emulators.iter_mut().enumerate() {
                println!("GPU {} starting epoch {}", gpu_id, epoch + 1);
                gpu_futures.push(emulator.emulate_training_step(model));
            }

            // Wait for all GPUs to complete their forward/backward passes (in parallel!)
            let gpu_results = join_all(gpu_futures).await;
            let mut max_gpu_time = Duration::ZERO;
            for result in gpu_results {
                let gpu_time = result?;
                max_gpu_time = max_gpu_time.max(gpu_time);
            }

            // Simulate gradient synchronization across GPUs
            println!("🔄 Synchronizing gradients across {} GPUs", gpus_count);
            sleep(Duration::from_millis(self.communication_latency_ms as u64)).await;

            let epoch_time = epoch_start.elapsed();
            epoch_times.push(epoch_time);

            println!(
                "📊 Epoch {} completed: {:.2}ms (GPU work: {:.2}ms, Communication: {:.2}ms)\n",
                epoch + 1,
                epoch_time.as_millis(),
                max_gpu_time.as_millis(),
                self.communication_latency_ms
            );
        }

        Ok(epoch_times)
    }
}
