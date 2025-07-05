//! Benchmark suite and testing scenarios based on GPEmu paper

use crate::emulator::{ RustGPUEmu, MultiGPUEmulator };
use crate::models::ModelConfig;
use crate::gpu_config;

/// Demo scenarios based on GPEmu paper
pub struct BenchmarkSuite;

impl BenchmarkSuite {
    /// Reproduce GPEmu's data stall analysis (Section 3.1)
    pub async fn data_stall_analysis() {
        println!("🧪 Data Stall Analysis (GPEmu Section 3.1)");
        println!("{}", "=".repeat(50));

        let gpu_manager = gpu_config::GpuModelManager
            ::load()
            .expect("Failed to load GPU configuration");
        let v100_model = gpu_manager.get_gpu("v100").expect("V100 GPU model not found").clone();
        let mut gpu = RustGPUEmu::new(v100_model);
        let models = vec![ModelConfig::alexnet(64), ModelConfig::resnet50(64)];

        for model in models {
            println!("\n📈 Testing model: {}", model.name);

            let _step_time = gpu.emulate_training_step(&model).await.unwrap();
            let profile = gpu.get_or_create_profile(&model);

            let compute_time = profile.forward_time_ms + profile.backward_time_ms;
            let data_time = profile.data_transfer_time_ms + profile.preprocessing_time_ms;
            let data_stall_ratio = (data_time / (compute_time + data_time)) * 100.0;

            println!("   Compute time: {:.2}ms", compute_time);
            println!("   Data time: {:.2}ms", data_time);
            println!("   Data stall ratio: {:.1}%", data_stall_ratio);

            gpu.reset_memory();
        }
    }

    /// Test distributed training (GPEmu Section 3.4)
    pub async fn distributed_training_test() {
        println!("\n🌐 Distributed Training Test (GPEmu Section 3.4)");
        println!("{}", "=".repeat(50));

        let gpu_manager = gpu_config::GpuModelManager
            ::load()
            .expect("Failed to load GPU configuration");
        let v100_model = gpu_manager.get_gpu("v100").expect("V100 GPU model not found").clone();
        let gpu_configs = vec![
            v100_model.clone(),
            v100_model.clone(),
            v100_model.clone(),
            v100_model
        ];

        let mut cluster = MultiGPUEmulator::new(gpu_configs, 10.0); // 10ms network latency
        let model = ModelConfig::resnet50(32); // 32 per GPU = 128 total batch size

        let epoch_times = cluster.emulate_data_parallel_training(&model, 3).await.unwrap();

        let avg_time =
            (epoch_times.iter().sum::<std::time::Duration>().as_millis() as f64) /
            (epoch_times.len() as f64);
        println!("📊 Average epoch time: {:.2}ms", avg_time);
        println!("📊 Effective throughput: {:.1} samples/sec", (128.0 * 1000.0) / avg_time);
    }

    /// Compare different GPU models (practical scenario)
    pub async fn gpu_comparison() {
        println!("\n🔋 GPU Model Comparison");
        println!("{}", "=".repeat(50));

        let gpu_manager = gpu_config::GpuModelManager
            ::load()
            .expect("Failed to load GPU configuration");
        let v100_model = gpu_manager.get_gpu("v100").expect("V100 GPU model not found").clone();
        let rtx4090_model = gpu_manager
            .get_gpu("rtx4090")
            .expect("RTX 4090 GPU model not found")
            .clone();
        let gpus = vec![v100_model, rtx4090_model];

        let model = ModelConfig::resnet50(128);

        for gpu_model in gpus {
            println!("\n🖥️  Testing: {}", gpu_model.name);
            let mut emulator = RustGPUEmu::new(gpu_model);

            let step_time = emulator.emulate_training_step(&model).await.unwrap();
            println!("   Training step time: {:.2}ms", step_time.as_millis());
            println!("   {}", emulator.get_stats());
        }
    }
}
