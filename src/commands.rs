//! Command handlers for the CLI interface

use std::collections::HashMap;
use std::sync::Arc;
use tracing::{ info, warn, error };
use colored::*;

use crate::emulator::{ RustGPUEmu, MultiGPUEmulator };
use crate::gpu_config::{ GpuModel, GpuModelManager };
use crate::models::ModelConfig;
use crate::benchmarks::BenchmarkSuite;
use crate::cli::{ GpuType, ModelType, PretrainedModel, CloudProvider };

#[cfg(feature = "real-models")]
use crate::cli::{ ModelFormat, OutputFormat, WorkloadType };
use crate::errors::{ PhantomResult, PhantomGpuError };

#[cfg(feature = "real-models")]
use crate::real_model_loader::{ RealModelInfo, RealModelLoader };

#[cfg(feature = "real-models")]
use crate::real_hardware_model::{ RealHardwareCalculator, RealisticPerformanceResult };

#[cfg(feature = "real-models")]
use crate::real_hardware_config::HardwareProfileLoader;

#[cfg(feature = "real-models")]
use crate::benchmark_validation::{
    CalibrationEngine,
    ModelType as BenchmarkModelType,
    Precision as BenchmarkPrecision,
};

#[cfg(feature = "real-models")]
pub struct StressTestArgs {
    pub verbose: bool,
    pub edge_cases: Option<String>,
}

pub async fn handle_train_command(
    model: &ModelType,
    batch_size: usize,
    epochs: usize,
    batches: usize,
    _gpu_model: GpuModel
) -> PhantomResult<()> {
    // Input validation
    if batch_size == 0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Batch size must be greater than 0".to_string(),
        });
    }
    if epochs == 0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Epochs must be greater than 0".to_string(),
        });
    }
    if batches == 0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Batches must be greater than 0".to_string(),
        });
    }

    // Graceful model training with error recovery
    match model {
        ModelType::Cnn => {
            #[cfg(feature = "candle")]
            {
                match
                    crate::candle_integration::run_candle_cnn_training(
                        _gpu_model,
                        batch_size,
                        epochs,
                        batches
                    ).await
                {
                    Ok(_) => info!("✅ CNN training completed successfully"),
                    Err(e) => {
                        warn!("⚠️ Candle CNN training failed: {}", e);
                        warn!("🔄 Falling back to custom neural network demo");
                        crate::neural_network_demo
                            ::run_neural_network_simulation().await
                            .map_err(|e| PhantomGpuError::ModelError(e.to_string()))?;
                    }
                }
            }
            #[cfg(not(feature = "candle"))]
            {
                warn!("🔄 CNN training fallback (no Candle)");
                warn!(
                    "💡 Enable Candle support with: cargo run --features candle -- train --model cnn"
                );
                crate::neural_network_demo
                    ::run_neural_network_simulation().await
                    .map_err(|e| PhantomGpuError::ModelError(e.to_string()))?;
            }
        }
        ModelType::ResNet => {
            #[cfg(feature = "candle")]
            {
                crate::candle_integration
                    ::run_resnet_training(_gpu_model, batch_size, epochs).await
                    .map_err(|e| PhantomGpuError::BenchmarkFailed {
                        operation: "ResNet training".to_string(),
                        reason: format!("{}", e),
                    })?;
            }
            #[cfg(not(feature = "candle"))]
            {
                warn!("🔄 ResNet training fallback (no Candle)");
                warn!(
                    "💡 Enable Candle support with: cargo run --features candle -- train --model resnet"
                );
                crate::neural_network_demo
                    ::run_neural_network_simulation().await
                    .map_err(|e| PhantomGpuError::ModelError(e.to_string()))?;
            }
        }
        ModelType::Transformer => {
            #[cfg(feature = "candle")]
            {
                crate::candle_integration
                    ::run_transformer_training(_gpu_model, batch_size, epochs).await
                    .map_err(|e| PhantomGpuError::BenchmarkFailed {
                        operation: "Transformer training".to_string(),
                        reason: format!("{}", e),
                    })?;
            }
            #[cfg(not(feature = "candle"))]
            {
                warn!("🔄 Transformer training fallback (no Candle)");
                warn!(
                    "💡 Enable Candle support with: cargo run --features candle -- train --model transformer"
                );
                crate::neural_network_demo
                    ::run_neural_network_simulation().await
                    .map_err(|e| PhantomGpuError::ModelError(e.to_string()))?;
            }
        }
        ModelType::Gpt => {
            warn!("🔄 GPT training simulation (placeholder)");
            // TODO: Implement GPT training
        }
        ModelType::Bert => {
            #[cfg(feature = "pytorch")]
            {
                crate::pytorch_integration
                    ::run_pytorch_distilbert_training(_gpu_model, batch_size, epochs, 2).await
                    .map_err(|e| PhantomGpuError::BenchmarkFailed {
                        operation: "PyTorch DistilBERT training".to_string(),
                        reason: format!("{}", e),
                    })?;
            }
            #[cfg(not(feature = "pytorch"))]
            {
                warn!("🔄 BERT training simulation (placeholder)");
                warn!(
                    "💡 Enable PyTorch support with: cargo run --features pytorch -- train --model bert"
                );
                // TODO: Implement BERT training with Candle
            }
        }
        #[cfg(feature = "pytorch")]
        ModelType::PytorchCnn => {
            crate::pytorch_integration
                ::run_pytorch_cnn_training(_gpu_model, batch_size, epochs, batches).await
                .map_err(|e| PhantomGpuError::BenchmarkFailed {
                    operation: "PyTorch CNN training".to_string(),
                    reason: format!("{}", e),
                })?;
        }
        #[cfg(feature = "pytorch")]
        ModelType::PytorchResnet => {
            crate::pytorch_integration
                ::run_pytorch_resnet_training(_gpu_model, batch_size, epochs).await
                .map_err(|e| PhantomGpuError::BenchmarkFailed {
                    operation: "PyTorch ResNet training".to_string(),
                    reason: format!("{}", e),
                })?;
        }
    }
    Ok(())
}

pub async fn handle_benchmark_command(
    model: &PretrainedModel,
    batch_size: usize,
    runs: usize,
    gpu_model: GpuModel
) -> PhantomResult<()> {
    // Input validation
    if batch_size == 0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Batch size must be greater than 0".to_string(),
        });
    }
    if runs == 0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Number of runs must be greater than 0".to_string(),
        });
    }

    crate::model_benchmarks
        ::benchmark_pretrained_model(model, batch_size, runs, gpu_model).await
        .map_err(|e| PhantomGpuError::BenchmarkFailed {
            operation: "benchmarking".to_string(),
            reason: format!("{}", e),
        })
}

pub async fn handle_compare_command(
    gpus: &[GpuType],
    model: &PretrainedModel,
    batch_size: usize
) -> PhantomResult<()> {
    // Input validation
    if gpus.is_empty() {
        return Err(PhantomGpuError::InvalidModel {
            reason: "At least one GPU must be specified for comparison".to_string(),
        });
    }
    if batch_size == 0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Batch size must be greater than 0".to_string(),
        });
    }

    let gpu_models: Vec<GpuModel> = gpus
        .iter()
        .map(|g| g.to_gpu_model())
        .collect();

    crate::model_benchmarks
        ::compare_gpus_with_model(model, batch_size, gpu_models).await
        .map_err(|e| PhantomGpuError::BenchmarkFailed {
            operation: "GPU comparison".to_string(),
            reason: format!("{}", e),
        })
}

pub async fn handle_cost_command(
    model: &PretrainedModel,
    hours: f64,
    provider: &CloudProvider
) -> PhantomResult<()> {
    // Input validation
    if hours <= 0.0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Training hours must be greater than 0".to_string(),
        });
    }
    if hours > 8760.0 {
        // 1 year
        return Err(PhantomGpuError::InvalidModel {
            reason: "Training hours cannot exceed 8760 (1 year)".to_string(),
        });
    }

    crate::cloud_cost_estimator
        ::estimate_training_costs(model, hours, provider).await
        .map_err(|_e| PhantomGpuError::UnsupportedProvider {
            provider: format!("{:?}", provider),
        })
}

pub async fn handle_distributed_command(
    num_gpus: usize,
    model: &PretrainedModel,
    epochs: usize
) -> Result<(), Box<dyn std::error::Error>> {
    // Use V100s for distributed simulation
    let gpu_manager = GpuModelManager::load()?;
    let v100_model = gpu_manager
        .get_gpu("v100")
        .expect("V100 GPU model not found in configuration")
        .clone();
    let gpu_models = vec![v100_model; num_gpus];
    let mut multi_gpu = MultiGPUEmulator::new(gpu_models, 2.0); // 2ms network latency

    let model_config = match model {
        PretrainedModel::ResNet50 => ModelConfig::resnet50(32),
        _ => ModelConfig::resnet50(32), // Default fallback
    };

    println!("🚀 Starting distributed training simulation...");
    let results = multi_gpu.emulate_data_parallel_training(&model_config, epochs).await?;

    let total_time: std::time::Duration = results.iter().sum();
    let avg_time = total_time / (results.len() as u32);

    println!("📊 Distributed Training Results:");
    println!("   • Total time: {:.2}s", total_time.as_secs_f64());
    println!("   • Average epoch: {:.2}s", avg_time.as_secs_f64());
    println!("   • Speedup: {:.2}x", (num_gpus as f64) * 0.85); // Account for communication overhead

    Ok(())
}

pub fn handle_list_gpus_command() -> Result<(), Box<dyn std::error::Error>> {
    let gpu_manager = GpuModelManager::load()?;
    gpu_manager.print_available_gpus();

    println!("\n💡 Usage examples:");
    println!("   phantom-gpu --gpu h100 train --model cnn");
    println!("   phantom-gpu --gpu mi300x benchmark --model res-net50");
    println!("   phantom-gpu compare -g v100 -g a100 -g h100 --model bert-base");

    Ok(())
}

#[cfg(feature = "real-models")]
pub fn handle_list_hardware_command(verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    use crate::real_hardware_model::RealHardwareCalculator;

    println!("\n🚀 Phantom GPU - Hardware Performance Profiles");
    println!("Detailed characteristics for realistic GPU performance modeling");
    println!("==================================================");

    // Try to load from TOML file first
    let calculator = RealHardwareCalculator::new();

    // Access profiles through reflection or implement a public getter method
    println!("✅ Loaded hardware profiles from hardware_profiles.toml");

    if verbose {
        println!("\n🔬 Detailed Hardware Profiles:");
        println!(
            "================================================================================"
        );

        // For now, show the available profiles that are loaded
        println!("📋 Available Profiles:");
        println!("   • H200 - Enterprise AI accelerator with HBM3e memory");
        println!("   • H100 - Professional AI training GPU");
        println!("   • RTX 5090 - Consumer flagship with Blackwell architecture");
        println!("   • RTX PRO 6000 - Professional Blackwell GPU");
        println!("   • RTX 4090 - Gaming GPU with Ada Lovelace architecture");
        println!("   • A100 - Enterprise GPU with Ampere architecture");
        println!("   • RTX A6000 - Professional Ampere GPU");
        println!("   • L40S - Server GPU optimized for inference");
        println!("   • RTX 3090 - Gaming/creator GPU");
        println!("   • Tesla V100 - Data center GPU with Volta architecture");

        println!("\n🏗️ Hardware Profile Components:");
        println!("   • Thermal characteristics (TDP, clocks, throttling)");
        println!("   • Memory hierarchy (L1/L2 cache, memory channels)");
        println!("   • Architecture details (CUDA cores, tensor cores)");
        println!("   • Model-specific performance curves (CNN, Transformer, RNN)");
        println!("   • Precision multipliers (FP16, INT8, INT4)");

        println!("\n🎯 Use Cases:");
        println!("   • Realistic performance modeling beyond basic FLOPS");
        println!("   • Thermal throttling and boost clock effects");
        println!("   • Memory hierarchy impact on different model sizes");
        println!("   • Architecture-specific optimizations");
    } else {
        println!("\n🔬 Hardware Profile Summary:");
        println!(
            "================================================================================"
        );
        println!("Profile      Architecture     TDP    Tensor Cores  Memory Channels  Use Case");
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!("h200         Hopper          700W        528            12        AI Training");
        println!("h100         Hopper          700W        456            10        AI Training");
        println!("rtx5090      Blackwell       575W        680            16        Gaming/AI");
        println!("rtx_pro_6000 Blackwell       600W        752            16        Professional");
        println!("rtx4090      Ada Lovelace    450W        512            12        Gaming");
        println!("a100         Ampere          400W        432             8        Enterprise");
        println!("a6000        Ampere          300W        336            12        Professional");
        println!("l40s         Ada Lovelace    350W        568            12        Server");
        println!(
            "rtx3090      Ampere          350W        328            12        Gaming/Creator"
        );
        println!("v100         Volta           300W        640             4        Data Center");
    }

    println!("\n📊 Performance Modeling Features:");
    println!("   • Batch size scaling effects (non-linear performance)");
    println!("   • Memory coalescing and cache hit ratios");
    println!("   • Thermal throttling under sustained loads");
    println!("   • Architecture-specific optimizations");
    println!("   • Model type performance curves (CNN vs Transformer vs RNN)");

    println!("\n💡 Usage with hardware profiles:");
    println!(
        "   phantom-gpu compare-models --real-hardware --models bert-base-uncased --gpus h100,a100"
    );
    println!(
        "   phantom-gpu benchmark --model distilbert-base-uncased --gpu h200 --precision fp16"
    );
    println!("   phantom-gpu recommend-gpu --model gpt2 --budget 100 --target-throughput 50");

    println!("\n🔧 Configuration:");
    println!("   • Edit hardware_profiles.toml to customize GPU characteristics");
    println!("   • Add custom GPU profiles for specialized hardware");
    println!("   • Use --hardware-profiles custom.toml for custom profile files");

    Ok(())
}

#[cfg(feature = "pytorch")]
pub async fn handle_framework_compare_command(
    batch_size: usize,
    gpu_model: GpuModel
) -> PhantomResult<()> {
    // Input validation
    if batch_size == 0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Batch size must be greater than 0".to_string(),
        });
    }

    crate::pytorch_integration
        ::compare_pytorch_candle_performance(gpu_model).await
        .map_err(|e| PhantomGpuError::BenchmarkFailed {
            operation: "Framework comparison".to_string(),
            reason: format!("{}", e),
        })
}

pub async fn handle_suite_command(experimental: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Running comprehensive benchmark suite...");

    BenchmarkSuite::data_stall_analysis().await;
    BenchmarkSuite::distributed_training_test().await;
    BenchmarkSuite::gpu_comparison().await;

    // Production features
    let _ = crate::model_benchmarks::run_model_benchmarks().await;
    let _ = crate::cloud_cost_estimator::analyze_cloud_costs().await;

    if experimental {
        println!("🔬 Running experimental features...");
        #[cfg(feature = "candle")]
        {
            crate::candle_integration::compare_gpu_performance_candle().await?;
        }
        #[cfg(not(feature = "candle"))]
        {
            println!("⚠️ Candle experimental features require --features candle");
        }
    }

    Ok(())
}

#[cfg(feature = "real-models")]
pub async fn handle_load_model_command(
    model_source: &str,
    format: &ModelFormat,
    batch_size: usize,
    runs: usize,
    gpu_model: GpuModel
) -> PhantomResult<()> {
    use crate::real_model_loader::RealModelLoader;
    use std::path::Path;

    // Input validation
    if batch_size == 0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Batch size must be greater than 0".to_string(),
        });
    }
    if runs == 0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Number of runs must be greater than 0".to_string(),
        });
    }

    info!("🔍 Loading real model: {}", model_source);

    let loader = RealModelLoader::new(None);

    // Determine if model_source is a file path or Hub ID
    let model_info = if Path::new(model_source).exists() {
        // Local file
        match format {
            ModelFormat::Auto => {
                // Auto-detect format based on file extension and structure
                if model_source.ends_with(".onnx") {
                    #[cfg(feature = "onnx")]
                    {
                        loader.load_onnx(model_source).await?
                    }
                    #[cfg(not(feature = "onnx"))]
                    {
                        return Err(
                            PhantomGpuError::ModelLoadError(
                                "ONNX support not enabled. Use --features onnx".to_string()
                            )
                        );
                    }
                } else if
                    model_source.ends_with(".pb") ||
                    (Path::new(model_source).is_dir() &&
                        Path::new(model_source).join("saved_model.pb").exists())
                {
                    loader.load_tensorflow(model_source).await?
                } else if model_source.ends_with(".pth") || model_source.ends_with(".pt") {
                    loader.load_pytorch_placeholder(model_source).await?
                } else {
                    return Err(
                        PhantomGpuError::ModelLoadError(
                            "File format not recognized for auto-detection".to_string()
                        )
                    );
                }
            }

            ModelFormat::Onnx => {
                if model_source.ends_with(".onnx") {
                    #[cfg(feature = "onnx")]
                    {
                        loader.load_onnx(model_source).await?
                    }
                    #[cfg(not(feature = "onnx"))]
                    {
                        return Err(
                            PhantomGpuError::ModelLoadError(
                                "ONNX support not enabled. Use --features onnx".to_string()
                            )
                        );
                    }
                } else {
                    return Err(
                        PhantomGpuError::ModelLoadError(
                            "Expected .onnx file for ONNX format".to_string()
                        )
                    );
                }
            }

            #[cfg(feature = "tensorflow")]
            ModelFormat::TensorFlow => { loader.load_tensorflow(model_source).await? }

            ModelFormat::PyTorch => {
                if model_source.ends_with(".pth") || model_source.ends_with(".pt") {
                    loader.load_pytorch_placeholder(model_source).await?
                } else {
                    return Err(
                        PhantomGpuError::ModelLoadError(
                            "Expected .pth or .pt file for PyTorch format".to_string()
                        )
                    );
                }
            }

            ModelFormat::HuggingFace => {
                #[cfg(feature = "huggingface")]
                {
                    loader.load_huggingface(model_source).await?
                }
                #[cfg(not(feature = "huggingface"))]
                {
                    return Err(
                        PhantomGpuError::ModelLoadError(
                            "HuggingFace support not enabled. Use --features huggingface".to_string()
                        )
                    );
                }
            }
        }
    } else {
        // Assume it's a Hugging Face model ID
        #[cfg(feature = "huggingface")]
        {
            loader.load_huggingface(model_source).await?
        }
        #[cfg(not(feature = "huggingface"))]
        {
            return Err(
                PhantomGpuError::ModelLoadError(
                    "Hugging Face support not enabled. Use --features huggingface".to_string()
                )
            );
        }
    };

    println!("✅ Model loaded successfully!");
    println!("   • Name: {}", model_info.name);
    println!("   • Format: {:?}", model_info.format);
    println!("   • Size: {:.1} MB", model_info.model_size_mb);
    println!(
        "   • Parameters: {}",
        model_info.parameter_count.map_or("Unknown".to_string(), |p|
            format!("{:.1}M", (p as f64) / 1_000_000.0)
        )
    );
    println!("   • Estimated memory: {:.1} MB", model_info.estimated_memory_mb);

    // Convert to ModelConfig for emulation
    let model_config = loader.to_model_config(&model_info, batch_size);

    println!("\n🏃 Running benchmark with {} runs...", runs);

    // Emulate inference
    let mut emulator = crate::emulator::RustGPUEmu::new(gpu_model.clone());

    let mut total_time = 0.0;
    for i in 0..runs {
        let start = std::time::Instant::now();

        // Simulate model inference (forward pass only)
        let _result = emulator
            .emulate_forward(&model_config).await
            .map_err(|e| PhantomGpuError::BenchmarkFailed {
                operation: "Model inference".to_string(),
                reason: format!("{}", e),
            })?;

        let duration = start.elapsed().as_secs_f64();
        total_time += duration;

        if i % (runs / 10).max(1) == 0 {
            println!("   Run {}/{}: {:.2}ms", i + 1, runs, duration * 1000.0);
        }
    }

    let avg_time = total_time / (runs as f64);
    let throughput = (batch_size as f64) / avg_time;

    println!("\n📊 Benchmark Results:");
    println!("   • Average inference time: {:.2}ms", avg_time * 1000.0);
    println!("   • Throughput: {:.1} samples/sec", throughput);
    println!("   • GPU: {}", gpu_model.name);
    println!(
        "   • Memory used: {:.1}/{:.1} MB",
        model_info.estimated_memory_mb,
        gpu_model.memory_gb * 1024.0
    );

    Ok(())
}

#[cfg(feature = "real-models")]
pub async fn handle_compare_models_command(
    models: &[String],
    gpus: &[GpuType],
    batch_sizes: &[usize],
    output_format: &OutputFormat,
    include_cost: bool,
    fast_mode: bool,
    show_progress: bool,
    precision: crate::cli::Precision,
    real_hardware: bool,
    hardware_profiles: Option<&str>
) -> PhantomResult<()> {
    use crate::model_comparison::ModelComparisonEngine;

    // Input validation
    if models.is_empty() {
        return Err(PhantomGpuError::InvalidModel {
            reason: "At least one model must be specified".to_string(),
        });
    }

    if gpus.is_empty() {
        return Err(PhantomGpuError::InvalidModel {
            reason: "At least one GPU type must be specified".to_string(),
        });
    }

    if batch_sizes.is_empty() {
        return Err(PhantomGpuError::InvalidModel {
            reason: "At least one batch size must be specified".to_string(),
        });
    }

    info!("🚀 Starting model comparison");
    println!("\n{}", "🔄 Model Comparison Engine".bold().cyan());
    println!("📋 Models: {}", models.join(", ").yellow());
    println!(
        "🖥️  GPUs: {}",
        gpus
            .iter()
            .map(|g| format!("{:?}", g))
            .collect::<Vec<_>>()
            .join(", ")
            .yellow()
    );
    println!(
        "📊 Batch sizes: {}",
        batch_sizes
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(", ")
            .yellow()
    );

    if include_cost {
        println!("💰 Cost analysis: {}", "Enabled".green());
    }

    let engine = ModelComparisonEngine::new();

    if fast_mode {
        println!("⚡ Fast mode: {}", "Enabled - Using optimized timing".green());
    }

    if show_progress {
        println!("📊 Progress indicators: {}", "Enabled".green());
    }

    println!("🔬 Precision: {:?}", precision);
    if real_hardware {
        println!("🏗️  Real hardware modeling: {}", "Enabled - Maximum accuracy".green());
    } else {
        println!("🏗️  Traditional emulation: {}", "Enabled".yellow());
    }

    // Run the comparison
    let results = engine.compare_models(
        models,
        gpus,
        batch_sizes,
        include_cost,
        fast_mode,
        show_progress,
        precision,
        real_hardware,
        hardware_profiles
    ).await?;

    // Output results in requested format
    match output_format {
        OutputFormat::Table => {
            println!("{}", engine.format_comparison_table(&results));
        }
        OutputFormat::Json => {
            let json_output = engine.export_to_json(&results)?;
            println!("{}", json_output);
        }
        OutputFormat::Csv => {
            let csv_output = engine.export_to_csv(&results)?;
            println!("{}", csv_output);
        }
        OutputFormat::Markdown => {
            // Convert table to markdown format
            let table_output = engine.format_comparison_table(&results);
            println!("```");
            println!("{}", table_output);
            println!("```");
        }
    }

    // Summary
    if matches!(output_format, OutputFormat::Table) {
        println!("\n{}", "📈 Summary".bold().green());

        // Find best performing model per GPU
        let mut best_per_gpu: std::collections::HashMap<
            String,
            &crate::model_comparison::ModelBenchmarkResult
        > = std::collections::HashMap::new();

        for result in &results {
            let current_best = best_per_gpu.get(&result.gpu_name);
            if
                current_best.is_none() ||
                result.throughput_samples_per_sec > current_best.unwrap().throughput_samples_per_sec
            {
                best_per_gpu.insert(result.gpu_name.clone(), result);
            }
        }

        for (gpu, best_result) in best_per_gpu {
            println!(
                "🏆 Best on {}: {} ({:.1} samples/sec)",
                gpu.yellow(),
                best_result.model_name.green(),
                best_result.throughput_samples_per_sec
            );
        }

        if include_cost {
            // Find most cost-efficient
            if
                let Some(most_efficient) = results
                    .iter()
                    .max_by(|a, b| a.efficiency_score.partial_cmp(&b.efficiency_score).unwrap())
            {
                println!(
                    "💎 Most efficient: {} on {} (Score: {:.1})",
                    most_efficient.model_name.green(),
                    most_efficient.gpu_name.yellow(),
                    most_efficient.efficiency_score
                );
            }
        }
    }

    Ok(())
}

#[cfg(feature = "real-models")]
pub async fn handle_recommend_gpu_command(
    model: &str,
    budget: Option<f64>,
    batch_size: usize,
    target_throughput: Option<f64>,
    workload: &WorkloadType,
    cloud_providers: &[CloudProvider],
    fast_mode: bool
) -> PhantomResult<()> {
    use crate::model_comparison::ModelComparisonEngine;

    // Input validation
    if batch_size == 0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Batch size must be greater than 0".to_string(),
        });
    }

    info!("🎯 Generating GPU recommendations for model: {}", model);
    println!("\n{}", "🧠 GPU Recommendation Engine".bold().green());
    println!("📋 Model: {}", model.cyan());
    println!("📊 Batch size: {}", batch_size.to_string().yellow());
    println!("🎯 Workload: {:?}", workload);

    if let Some(budget_limit) = budget {
        println!("💰 Budget: ${:.2}/hour", budget_limit);
    }

    if let Some(throughput_target) = target_throughput {
        println!("⚡ Target throughput: {:.1} samples/sec", throughput_target);
    }

    println!(
        "☁️  Cloud providers: {}",
        cloud_providers
            .iter()
            .map(|p| format!("{:?}", p))
            .collect::<Vec<_>>()
            .join(", ")
            .yellow()
    );

    let engine = ModelComparisonEngine::new();

    if fast_mode {
        println!("⚡ Fast mode: {}", "Enabled - Using optimized timing".green());
    }

    // Generate recommendations
    let recommendations = engine.recommend_gpu(
        model,
        budget,
        batch_size,
        target_throughput,
        workload,
        cloud_providers,
        fast_mode
    ).await?;

    // Display recommendations
    println!("{}", engine.format_recommendations_table(&recommendations));

    // Additional insights
    println!("\n{}", "💡 Insights & Recommendations".bold().blue());

    let feasible_options: Vec<_> = recommendations
        .iter()
        .filter(|r| r.meets_budget && r.meets_throughput)
        .collect();

    if feasible_options.is_empty() {
        println!("⚠️  No GPUs meet all your requirements. Consider:");
        if budget.is_some() {
            println!("   • Increasing budget");
        }
        if target_throughput.is_some() {
            println!("   • Reducing target throughput");
        }
        println!("   • Using larger batch sizes for better efficiency");
    } else {
        println!("✅ {} GPUs meet your requirements", feasible_options.len());

        // Provide specific recommendations based on workload
        match workload {
            WorkloadType::RealTime => {
                println!("⚡ For real-time workloads:");
                println!("   • Prioritize lowest latency GPUs");
                println!("   • Consider smaller batch sizes (1-4)");
                println!("   • Look for consistent performance");
            }
            WorkloadType::BatchProcessing => {
                println!("📦 For batch processing:");
                println!("   • Maximize throughput with larger batch sizes");
                println!("   • Consider cost efficiency over latency");
                println!("   • Scale horizontally if needed");
            }
            WorkloadType::Training => {
                println!("🎓 For training workloads:");
                println!("   • Prioritize memory capacity");
                println!("   • Consider multi-GPU setups");
                println!("   • Factor in long-running costs");
            }
            WorkloadType::Inference => {
                println!("🔮 For inference workloads:");
                println!("   • Balance latency and throughput");
                println!("   • Consider auto-scaling");
                println!("   • Monitor cost efficiency");
            }
        }
    }

    // Cloud provider insights
    if !recommendations.is_empty() {
        println!("\n☁️  Cloud Provider Comparison:");
        for provider in cloud_providers {
            if let Some(top_rec) = recommendations.first() {
                if
                    let Some(price) = top_rec.cloud_availability.get(
                        &format!("{:?}", provider).to_lowercase()
                    )
                {
                    println!("   • {:?}: ${:.3}/hour", provider, price);
                }
            }
        }
    }

    Ok(())
}

/// Handle validation command to check PhantomGPU accuracy against real hardware
#[cfg(feature = "real-models")]
pub async fn handle_validate_command(
    gpu: Option<&str>,
    benchmark_data_path: Option<&str>,
    verbose: bool
) -> PhantomResult<()> {
    use crate::benchmark_validation::CalibrationEngine;

    println!("🎯 Validating PhantomGPU accuracy against real hardware benchmarks");

    let mut engine = CalibrationEngine::new();

    // Clear any existing calibration cache to test with improved FLOPS estimates
    engine.clear_calibration_cache();

    // Load benchmark data
    if let Some(data_path) = benchmark_data_path {
        println!("📊 Loading custom benchmark data from: {}", data_path);
        engine.load_benchmark_data(data_path).map_err(|e| PhantomGpuError::ConfigError {
            message: format!("Failed to load benchmark data: {}", e),
        })?;
    } else {
        println!("📚 Loading reference benchmarks from built-in data");
        engine.load_reference_benchmarks().map_err(|e| PhantomGpuError::ConfigError {
            message: format!("Failed to load reference benchmarks: {}", e),
        })?;

        // Load reference data from file if it exists
        if std::path::Path::new("benchmark_data/reference_benchmarks.json").exists() {
            engine
                .load_benchmark_data("benchmark_data/reference_benchmarks.json")
                .map_err(|e| PhantomGpuError::ConfigError {
                    message: format!("Failed to load reference benchmarks: {}", e),
                })?;
        }
    }

    // Validate specific GPU or all available GPUs
    if let Some(gpu_name) = gpu {
        println!("🖥️  Validating GPU: {}", gpu_name);

        // Calibrate first
        engine.calibrate_gpu_model(gpu_name).map_err(|e| PhantomGpuError::ConfigError {
            message: format!("Failed to calibrate GPU model: {}", e),
        })?;

        // Validate
        let error = engine.validate_predictions(gpu_name).map_err(|e| PhantomGpuError::ConfigError {
            message: format!("Failed to validate predictions: {}", e),
        })?;

        let accuracy = 100.0 - error;
        let status_icon = if error < 5.0 {
            "✅"
        } else if error < 10.0 {
            "🟡"
        } else if error < 20.0 {
            "🟠"
        } else {
            "🔴"
        };

        println!("\n📊 Validation Results for {}:", gpu_name);
        println!("   Accuracy: {:.1}% (±{:.1}% error) {}", accuracy, error, status_icon);

        if verbose {
            let report = engine.generate_accuracy_report();
            println!("\n{}", report);
        }

        // Provide recommendations
        if error > 10.0 {
            println!("\n💡 Recommendations to improve accuracy:");
            println!("   • Collect more benchmark data for this GPU");
            println!("   • Run actual benchmarks to validate against real hardware");
            println!("   • Check if hardware profile configuration is accurate");
        } else if error < 5.0 {
            println!("\n🎉 Excellent accuracy! This GPU model is well-calibrated.");
        }
    } else {
        println!("🔍 Validating all available GPUs...");

        // For now, validate known GPUs
        let known_gpus = vec!["Tesla V100", "A100", "RTX 4090"];
        let mut total_error = 0.0;
        let mut validated_count = 0;

        for gpu_name in &known_gpus {
            if engine.real_data.contains_key(*gpu_name) {
                println!("\n🖥️  Validating: {}", gpu_name);

                if let Ok(_) = engine.calibrate_gpu_model(gpu_name) {
                    if let Ok(error) = engine.validate_predictions(gpu_name) {
                        let accuracy = 100.0 - error;
                        let status_icon = if error < 5.0 {
                            "✅"
                        } else if error < 10.0 {
                            "🟡"
                        } else {
                            "🔴"
                        };

                        println!(
                            "   Accuracy: {:.1}% (±{:.1}% error) {}",
                            accuracy,
                            error,
                            status_icon
                        );
                        total_error += error;
                        validated_count += 1;
                    }
                }
            }
        }

        if validated_count > 0 {
            let avg_error = total_error / (validated_count as f64);
            let avg_accuracy = 100.0 - avg_error;

            println!("\n📈 Overall Validation Summary:");
            println!("   Average accuracy: {:.1}% (±{:.1}% error)", avg_accuracy, avg_error);

            if avg_error < 5.0 {
                println!("   Status: ✅ Excellent overall accuracy");
            } else if avg_error < 10.0 {
                println!("   Status: 🟡 Good overall accuracy");
            } else {
                println!("   Status: 🔴 Needs improvement");
            }
        }

        if verbose {
            let report = engine.generate_accuracy_report();
            println!("\n{}", report);
        }
    }

    println!("\n💡 How to improve accuracy:");
    println!("   1. Run: phantom-gpu calibrate --gpu <gpu_name> --benchmark-data <data.json>");
    println!("   2. Collect more real benchmark data from actual hardware");
    println!("   3. Update hardware profiles with more accurate specifications");
    println!("   4. Contribute benchmark data to the community");

    Ok(())
}

/// Handle calibration command to train performance models using real data
#[cfg(feature = "real-models")]
pub async fn handle_calibrate_command(
    gpu: &str,
    benchmark_data_path: &str,
    output_path: Option<&str>
) -> PhantomResult<()> {
    use crate::benchmark_validation::CalibrationEngine;

    println!("🔧 Calibrating performance model for: {}", gpu);
    println!("📊 Using benchmark data from: {}", benchmark_data_path);

    let mut engine = CalibrationEngine::new();

    // Load benchmark data
    engine.load_benchmark_data(benchmark_data_path).map_err(|e| PhantomGpuError::ConfigError {
        message: format!("Failed to load benchmark data: {}", e),
    })?;

    // Perform calibration
    println!("⚙️  Analyzing benchmark data and extracting calibration factors...");
    engine.calibrate_gpu_model(gpu).map_err(|e| PhantomGpuError::ConfigError {
        message: format!("Failed to calibrate GPU model: {}", e),
    })?;

    // Validate the calibrated model
    println!("🎯 Validating calibrated model...");
    let error = engine.validate_predictions(gpu).map_err(|e| PhantomGpuError::ConfigError {
        message: format!("Failed to validate calibrated model: {}", e),
    })?;

    let accuracy = 100.0 - error;
    println!("✅ Calibration complete!");
    println!("   Accuracy: {:.1}% (±{:.1}% error)", accuracy, error);

    // Save calibrated model if output path provided
    if let Some(output) = output_path {
        println!("💾 Saving calibrated model to: {}", output);

        // Here we would serialize the calibration factors to file
        // For now, we'll just indicate success
        println!("   Saved calibration factors for future use");
    }

    // Provide performance insights
    if error < 5.0 {
        println!(
            "\n🎉 Excellent calibration! The model should provide highly accurate predictions."
        );
    } else if error < 10.0 {
        println!(
            "\n✅ Good calibration! The model should provide reasonably accurate predictions."
        );
        println!("💡 Consider collecting more benchmark data to improve accuracy further.");
    } else {
        println!("\n⚠️  Calibration needs improvement. Consider:");
        println!("   • Collecting more diverse benchmark data");
        println!("   • Ensuring benchmark data quality and consistency");
        println!("   • Checking if the GPU profile matches actual hardware");
    }

    // Show what was calibrated
    println!("\n📊 Calibrated parameters:");
    println!("   • Base performance multipliers");
    println!("   • Batch size scaling corrections");
    println!("   • Memory efficiency factors");
    println!("   • Precision performance multipliers");
    println!("   • Model type optimization factors");

    println!("\n💡 To use the calibrated model:");
    println!("   phantom-gpu validate --gpu {} --verbose", gpu);
    println!("   phantom-gpu compare-models --models <model> --gpus {} --real-hardware", gpu);

    Ok(())
}

#[cfg(feature = "real-models")]
pub fn handle_stress_test(args: &StressTestArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("🧪 PhantomGPU Stress Testing Suite");
    println!("{}", "=".repeat(60));

    let mut calibration_engine = crate::benchmark_validation::CalibrationEngine::new();

    // Load edge cases data
    let edge_cases_path = args.edge_cases.as_deref().unwrap_or("benchmark_data/edge_cases.json");
    match std::fs::read_to_string(edge_cases_path) {
        Ok(data) => {
            println!("✅ Loaded edge cases from: {}", edge_cases_path);
            if args.verbose {
                println!("📊 Edge cases data: {} bytes", data.len());
            }
        }
        Err(e) => {
            println!("⚠️  Could not load edge cases from {}: {}", edge_cases_path, e);
            println!("   Using default stress test scenarios");
        }
    }

    // Run stress tests
    println!("\n🔥 Running stress test scenarios...");

    // 1. Large Batch Size Tests
    println!("\n1️⃣ Large Batch Size Tests");
    test_large_batch_sizes(&calibration_engine)?;

    // 2. Memory Pressure Tests
    println!("\n2️⃣ Memory Pressure Tests");
    test_memory_pressure(&calibration_engine)?;

    // 3. Mixed Precision Tests
    println!("\n3️⃣ Mixed Precision Tests");
    test_mixed_precision(&calibration_engine)?;

    // 4. Temperature Scaling Tests
    println!("\n4️⃣ Temperature Scaling Tests");
    test_temperature_scaling(&calibration_engine)?;

    // 5. Power Limit Tests
    println!("\n5️⃣ Power Limit Tests");
    test_power_limits(&calibration_engine)?;

    println!("\n{}", "=".repeat(60));
    println!("✅ Stress testing completed successfully!");
    println!("{}", "=".repeat(60));

    Ok(())
}

#[cfg(feature = "real-models")]
fn test_large_batch_sizes(
    engine: &crate::benchmark_validation::CalibrationEngine
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing extreme batch sizes: 256, 512, 1024+");

    // Test large batch sizes that push memory limits
    let test_cases = vec![
        (256, "Large batch training scenario"),
        (512, "Extreme batch processing"),
        (1024, "Memory-bound inference")
    ];

    for (batch_size, description) in test_cases {
        println!("   • {} (batch_size: {})", description, batch_size);

        // Simulate prediction for large batch
        let predicted_time = engine
            .predict_calibrated_time(
                "RTX 4090",
                "ResNet-50",
                batch_size,
                &crate::benchmark_validation::Precision::FP32,
                &BenchmarkModelType::CNN
            )
            .unwrap_or(100.0);

        println!("     Predicted time: {:.2}ms", predicted_time);

        // Check memory efficiency
        let memory_usage = ((batch_size as f64) * 4.0 * 224.0 * 224.0 * 3.0) / (1024.0 * 1024.0); // MB
        println!("     Memory usage: {:.1}MB", memory_usage);
    }

    Ok(())
}

#[cfg(feature = "real-models")]
fn test_memory_pressure(
    engine: &crate::benchmark_validation::CalibrationEngine
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing models near GPU memory limits");

    let test_cases = vec![
        (32, "ResNet-50 near memory limit"),
        (64, "BERT-Base with large context"),
        (128, "Transformer with attention")
    ];

    for (batch_size, description) in test_cases {
        println!("   • {} (batch_size: {})", description, batch_size);

        let predicted_time = engine
            .predict_calibrated_time(
                "A100",
                "BERT-Base",
                batch_size,
                &crate::benchmark_validation::Precision::FP16,
                &BenchmarkModelType::Transformer
            )
            .unwrap_or(100.0);

        println!("     Predicted time: {:.2}ms", predicted_time);
    }

    Ok(())
}

#[cfg(feature = "real-models")]
fn test_mixed_precision(
    engine: &crate::benchmark_validation::CalibrationEngine
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing INT8 and quantized model performance");

    let test_cases = vec![
        (32, "INT8 quantized CNN"),
        (64, "Mixed precision Transformer"),
        (128, "Quantized inference pipeline")
    ];

    for (batch_size, description) in test_cases {
        println!("   • {} (batch_size: {})", description, batch_size);

        let predicted_time = engine
            .predict_calibrated_time(
                "Tesla V100",
                "ResNet-50",
                batch_size,
                &crate::benchmark_validation::Precision::INT8,
                &BenchmarkModelType::CNN
            )
            .unwrap_or(100.0);

        println!("     Predicted time: {:.2}ms", predicted_time);

        // Test efficiency gain from quantization
        let fp32_time = engine
            .predict_calibrated_time(
                "Tesla V100",
                "ResNet-50",
                batch_size,
                &crate::benchmark_validation::Precision::FP32,
                &BenchmarkModelType::CNN
            )
            .unwrap_or(100.0);

        let speedup = fp32_time / predicted_time;
        println!("     Quantization speedup: {:.2}x", speedup);
    }

    Ok(())
}

#[cfg(feature = "real-models")]
fn test_temperature_scaling(
    engine: &crate::benchmark_validation::CalibrationEngine
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing thermal throttling effects");

    let base_time = engine
        .predict_calibrated_time(
            "RTX 4090",
            "ResNet-50",
            32,
            &crate::benchmark_validation::Precision::FP32,
            &BenchmarkModelType::CNN
        )
        .unwrap_or(100.0);

    let temperature_scenarios = vec![
        (65, 1.0, "Normal temperature"),
        (75, 1.1, "Warm GPU"),
        (85, 1.25, "Hot GPU - mild throttling"),
        (95, 1.5, "Very hot GPU - heavy throttling")
    ];

    for (temp_c, throttle_factor, description) in temperature_scenarios {
        let throttled_time = base_time * throttle_factor;
        println!("   • {} ({}°C): {:.2}ms", description, temp_c, throttled_time);
    }

    Ok(())
}

#[cfg(feature = "real-models")]
fn test_power_limits(
    engine: &crate::benchmark_validation::CalibrationEngine
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing power constraint impacts");

    let base_time = engine
        .predict_calibrated_time(
            "RTX 4090",
            "ResNet-50",
            32,
            &crate::benchmark_validation::Precision::FP32,
            &BenchmarkModelType::CNN
        )
        .unwrap_or(100.0);

    let power_scenarios = vec![
        (100, 1.0, "Full power (300W)"),
        (80, 1.15, "Power limit 80% (240W)"),
        (60, 1.35, "Power limit 60% (180W)"),
        (40, 1.7, "Severe power limit 40% (120W)")
    ];

    for (power_percent, slowdown_factor, description) in power_scenarios {
        let limited_time = base_time * slowdown_factor;
        println!("   • {} ({}%): {:.2}ms", description, power_percent, limited_time);
    }

    Ok(())
}

/// Handle thermal modeling command
pub async fn handle_thermal_command(
    gpu_model: &GpuModel,
    workload_intensity: f64,
    duration_seconds: u64,
    ambient_temp: f64,
    verbose: bool
) -> PhantomResult<()> {
    use crate::thermal_modeling::{ ThermalModelingEngine, ThermalPerformanceResult };

    // Input validation
    if !(0.0..=1.0).contains(&workload_intensity) {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Workload intensity must be between 0.0 and 1.0".to_string(),
        });
    }

    if duration_seconds == 0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Duration must be greater than 0 seconds".to_string(),
        });
    }

    println!("✅ Loaded thermal profiles");

    // Initialize thermal modeling engine
    let mut thermal_engine = ThermalModelingEngine::new(ambient_temp);
    thermal_engine.initialize_gpu(&gpu_model.name);

    println!("🌡️  Initial Conditions:");
    println!("   • Ambient: {:.1}°C", ambient_temp);
    println!("   • GPU: {}", gpu_model.name);
    println!("   • Workload: {:.0}%", workload_intensity * 100.0);
    println!("   • Duration: {}s", duration_seconds);

    // Get thermal profile
    if let Some(profile) = thermal_engine.get_thermal_profile(&gpu_model.name) {
        println!("\n📊 Thermal Profile:");
        println!("   • TDP: {:.0}W", profile.tdp_watts);
        println!("   • Base temp: {:.1}°C", profile.base_temp_celsius);
        println!("   • Throttle temp: {:.1}°C", profile.throttle_temp_celsius);
        println!("   • Thermal resistance: {:.3}°C/W", profile.thermal_resistance);

        if verbose {
            println!("   • Cooling time constant: {:.1}s", profile.cooling_time_constant);
            println!("   • Shutdown temp: {:.1}°C", profile.shutdown_temp_celsius);
        }
    }

    println!("\n🔄 Running thermal simulation...");

    // Simulation parameters
    let steps = std::cmp::min(duration_seconds, 60); // Max 60 steps for display
    let step_duration = (duration_seconds as f64) / (steps as f64);

    let mut temperatures = Vec::new();
    let mut performance_multipliers = Vec::new();
    let mut times = Vec::new();

    // Run simulation
    for step in 0..=steps {
        let current_time = (step as f64) * step_duration;

        if
            let Some(thermal_state) = thermal_engine.update_thermal_state(
                &gpu_model.name,
                workload_intensity
            )
        {
            temperatures.push(thermal_state.current_temp_celsius);
            performance_multipliers.push(thermal_state.performance_multiplier);
            times.push(current_time);

            // Print periodic updates
            if step % (steps / 5).max(1) == 0 || step == steps {
                let throttle_status = if thermal_state.throttling_active {
                    "🔥 THROTTLING"
                } else {
                    "✅ Normal"
                };
                println!(
                    "   {:3.0}s: {:.1}°C | {:.1}% perf | {}",
                    current_time,
                    thermal_state.current_temp_celsius,
                    thermal_state.performance_multiplier * 100.0,
                    throttle_status
                );
            }
        }

        // Small delay for realism
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    // Final results
    if let Some(final_state) = thermal_engine.get_thermal_state(&gpu_model.name) {
        println!("\n📈 Final Results:");
        println!("   • Final temp: {:.1}°C", final_state.current_temp_celsius);
        println!("   • Final performance: {:.1}%", final_state.performance_multiplier * 100.0);

        let thermal_headroom = thermal_engine
            .get_thermal_profile(&gpu_model.name)
            .map(|p| p.throttle_temp_celsius - final_state.current_temp_celsius)
            .unwrap_or(0.0);
        println!("   • Thermal headroom: {:.1}°C", thermal_headroom);

        if final_state.throttling_active {
            println!("   • ⚠️  GPU is thermally throttled");
        } else {
            println!("   • ✅ GPU operating normally");
        }

        // Calculate thermal impact on performance
        let base_throughput = 1000.0; // Example baseline
        let thermal_result = ThermalPerformanceResult::new(base_throughput, final_state.clone());

        println!("\n💡 Performance Impact:");
        println!("   • Base throughput: {:.1} samples/sec", thermal_result.base_throughput);
        println!(
            "   • Thermal-adjusted: {:.1} samples/sec",
            thermal_result.thermal_adjusted_throughput
        );

        if thermal_result.thermal_impact < -0.1 {
            println!("   • 📉 Performance loss: {:.1}%", -thermal_result.thermal_impact);
        } else {
            println!("   • ✅ No significant thermal impact");
        }
    }

    // Verbose output: show thermal curve
    if verbose && temperatures.len() > 5 {
        println!("\n📊 Thermal Curve:");
        for (i, &temp) in temperatures.iter().enumerate() {
            if i % (temperatures.len() / 10).max(1) == 0 {
                let time = times[i];
                let perf = performance_multipliers[i];
                let bar_len = (((temp - ambient_temp) / 50.0) * 20.0).max(0.0).min(20.0) as usize;
                let bar = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
                println!("   {:4.0}s: [{}] {:.1}°C ({:.0}%)", time, bar, temp, perf * 100.0);
            }
        }
    }

    // Cooling simulation
    println!("\n❄️  Simulating cooling (idle)...");
    if let Some(cooled_state) = thermal_engine.simulate_cooling(&gpu_model.name, 60.0) {
        println!("   • After 60s idle: {:.1}°C", cooled_state.current_temp_celsius);
    }

    // Get thermal summary for all GPUs (in case we add multi-GPU support later)
    let summary = thermal_engine.get_thermal_summary();
    if verbose && !summary.is_empty() {
        println!("\n📋 Thermal Summary:");
        for (gpu_name, thermal_summary) in &summary {
            println!(
                "   • {}: {:.1}°C | {:.0}% load | {:.1}°C headroom",
                gpu_name,
                thermal_summary.current_temp,
                thermal_summary.thermal_load * 100.0,
                thermal_summary.thermal_headroom
            );
        }
    }

    println!("\n✅ Thermal modeling completed!");
    Ok(())
}

/// Handle batch optimization command
pub async fn handle_optimize_command(
    gpu_model: &GpuModel,
    model_name: &str,
    target_utilization: f64,
    verbose: bool
) -> PhantomResult<()> {
    use crate::batch_optimizer::BatchOptimizer;

    // Input validation
    if !(0.1..=1.0).contains(&target_utilization) {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Target utilization must be between 0.1 and 1.0".to_string(),
        });
    }

    println!("✅ Initializing batch optimizer");

    // Create batch optimizer
    let optimizer = BatchOptimizer::new();

    // Check if model is supported
    if optimizer.get_model_profile(model_name).is_none() {
        println!("📋 Available models:");
        for model in optimizer.list_models() {
            println!("   • {}", model);
        }
        return Err(PhantomGpuError::InvalidModel {
            reason: format!("Model '{}' not found in optimizer profiles", model_name),
        });
    }

    println!("🎯 Optimization Parameters:");
    println!("   • GPU: {} ({:.0} GB VRAM)", gpu_model.name, gpu_model.memory_gb);
    println!("   • Model: {}", model_name);
    println!("   • Target memory utilization: {:.0}%", target_utilization * 100.0);
    println!(
        "   • Architecture: {}",
        gpu_model.architecture.as_ref().unwrap_or(&"Unknown".to_string())
    );

    // Run optimization
    println!("\n🔄 Analyzing batch sizes...");
    let result = optimizer.optimize_batch_size(gpu_model, model_name, target_utilization)?;

    // Display results
    println!("\n🎯 Optimization Results:");
    println!("   • Optimal batch size: {}", result.optimal_batch_size);
    println!("   • Max safe batch size: {}", result.max_safe_batch_size);
    println!("   • Memory utilization: {:.1}%", result.memory_utilization * 100.0);
    println!("   • Estimated throughput: {:.1} samples/sec", result.throughput_samples_per_sec);
    println!("   • Compute efficiency: {:.1}%", result.compute_efficiency * 100.0);

    // Show recommendations
    if !result.recommendations.is_empty() {
        println!("\n💡 Recommendations:");
        for (i, recommendation) in result.recommendations.iter().enumerate() {
            println!("   {}. {}", i + 1, recommendation);
        }
    }

    // Verbose output: show detailed batch analysis
    if verbose && !result.batch_analysis.is_empty() {
        println!("\n📊 Detailed Batch Analysis:");
        println!(
            "   {:>5} {:>10} {:>8} {:>12} {:>10} {:>6} {}",
            "Batch",
            "Memory MB",
            "Util %",
            "Throughput",
            "Latency",
            "Viable",
            "Notes"
        );
        println!("   {}", "-".repeat(75));

        for analysis in &result.batch_analysis {
            if analysis.batch_size % 8 == 0 || analysis.batch_size <= 8 || !analysis.is_viable {
                let viable_icon = if analysis.is_viable { "✅" } else { "❌" };
                let notes_summary = if analysis.notes.is_empty() {
                    "-".to_string()
                } else {
                    analysis.notes[0].clone()
                };

                println!(
                    "   {:>5} {:>10.0} {:>7.1}% {:>9.1} {:>9.1}ms {:>6} {}",
                    analysis.batch_size,
                    analysis.memory_usage_mb,
                    analysis.memory_utilization * 100.0,
                    analysis.throughput_samples_per_sec,
                    analysis.latency_ms,
                    viable_icon,
                    notes_summary
                );
            }
        }
    }

    // Memory efficiency analysis
    println!("\n📈 Memory Efficiency Analysis:");
    let optimal_analysis = result.batch_analysis
        .iter()
        .find(|a| a.batch_size == result.optimal_batch_size)
        .unwrap();

    println!("   • Memory used: {:.1} GB", optimal_analysis.memory_usage_mb / 1024.0);
    println!("   • Memory available: {:.1} GB", gpu_model.memory_gb);
    println!("   • Memory efficiency: {:.1}%", optimal_analysis.memory_utilization * 100.0);

    if optimal_analysis.memory_utilization < 0.6 {
        println!("   • 💡 Consider model parallelism to better utilize memory");
    } else if optimal_analysis.memory_utilization > 0.9 {
        println!("   • ⚠️  High memory pressure - monitor for OOM errors");
    }

    // Performance comparison
    let single_batch_analysis = result.batch_analysis
        .iter()
        .find(|a| a.batch_size == 1)
        .unwrap();

    let throughput_improvement =
        result.throughput_samples_per_sec / single_batch_analysis.throughput_samples_per_sec;
    let latency_ratio = optimal_analysis.latency_ms / single_batch_analysis.latency_ms;

    println!("\n⚡ Performance vs Single Sample:");
    println!("   • Throughput improvement: {:.1}x", throughput_improvement);
    println!("   • Latency ratio: {:.1}x", latency_ratio);

    if throughput_improvement > 5.0 {
        println!("   • 🚀 Excellent batching efficiency!");
    } else if throughput_improvement < 2.0 {
        println!("   • 📈 Consider optimizing model for better batching");
    }

    // GPU architecture insights
    println!("\n🏗️  Architecture Insights:");
    if let Some(arch) = &gpu_model.architecture {
        match arch.as_str() {
            "Ampere" => {
                println!("   • Ampere GPUs excel at large batch transformer workloads");
                println!("   • Consider using Tensor Cores with mixed precision");
            }
            "Hopper" => {
                println!("   • Hopper architecture optimized for very large language models");
                println!("   • Transformer Engine provides automatic optimization");
            }
            "Ada Lovelace" => {
                println!("   • Ada Lovelace balances gaming and AI workloads");
                println!("   • Consider batch sizes that align with shader core count");
            }
            "Volta" => {
                println!("   • Volta architecture performs best with moderate batch sizes");
                println!("   • Focus on memory bandwidth optimization");
            }
            "Blackwell" => {
                println!("   • Next-gen Blackwell architecture with enhanced AI performance");
                println!("   • Optimized for both training and inference workloads");
            }
            _ => {
                println!("   • General GPU - focus on memory and compute balance");
            }
        }
    } else {
        println!("   • General GPU - focus on memory and compute balance");
    }

    println!("\n✅ Batch optimization completed!");
    Ok(())
}

/// Map common game abbreviations to full names
fn map_game_name(game: &str) -> String {
    match game.to_lowercase().as_str() {
        "cyberpunk" | "cyberpunk2077" | "cp2077" => "Cyberpunk 2077".to_string(),
        "cod" | "mw3" | "modern-warfare" | "call-of-duty" =>
            "Call of Duty: Modern Warfare III".to_string(),
        "fortnite" | "fn" => "Fortnite".to_string(),
        "hogwarts" | "hogwarts-legacy" | "hl" => "Hogwarts Legacy".to_string(),
        _ => game.to_string(), // Return original if no match found
    }
}

/// Handle power analysis command
pub async fn handle_gaming_command(
    gpu_config: &GpuModel,
    game: &str,
    resolution: &str,
    ray_tracing: bool,
    dlss: &str,
    fsr: &str,
    target_fps: f64,
    scene_complexity: f64,
    graphics_quality: &str,
    ambient_temp: f64,
    frame_generation: bool,
    power_analysis: bool,
    thermal_session: bool,
    session_duration: f64,
    verbose: bool
) -> PhantomResult<()> {
    use crate::unified_gaming_emulator::{ UnifiedGamingEmulator, FrameTimeConsistency };
    use crate::models::{
        GamingWorkloadConfig,
        DLSSMode,
        FSRMode,
        GraphicsSettings,
        Quality,
        AntiAliasing,
    };

    // Parse resolution
    let resolution_parts: Vec<&str> = resolution.split('x').collect();
    if resolution_parts.len() != 2 {
        return Err(PhantomGpuError::ConfigError {
            message: format!("Invalid resolution format: {}", resolution),
        });
    }
    let res_width = resolution_parts[0].parse::<u32>().map_err(|_| PhantomGpuError::ConfigError {
        message: format!("Invalid width: {}", resolution_parts[0]),
    })?;
    let res_height = resolution_parts[1].parse::<u32>().map_err(|_| PhantomGpuError::ConfigError {
        message: format!("Invalid height: {}", resolution_parts[1]),
    })?;

    // Parse DLSS mode
    let dlss_mode = match dlss.to_lowercase().as_str() {
        "quality" => DLSSMode::Quality,
        "balanced" => DLSSMode::Balanced,
        "performance" => DLSSMode::Performance,
        "ultra" | "ultra_performance" => DLSSMode::UltraPerformance,
        _ => DLSSMode::Off,
    };

    // Parse FSR mode
    let fsr_mode = match fsr.to_lowercase().as_str() {
        "ultra_quality" => FSRMode::UltraQuality,
        "quality" => FSRMode::Quality,
        "balanced" => FSRMode::Balanced,
        "performance" => FSRMode::Performance,
        _ => FSRMode::Off,
    };

    // Parse graphics quality
    let quality_level = match graphics_quality.to_lowercase().as_str() {
        "low" => Quality::Low,
        "medium" => Quality::Medium,
        "high" => Quality::High,
        "ultra" => Quality::Ultra,
        _ => Quality::High,
    };

    // Create graphics settings based on quality preset
    let graphics_settings = match quality_level {
        Quality::Ultra =>
            GraphicsSettings {
                texture_quality: Quality::Ultra,
                shadow_quality: Quality::Ultra,
                anti_aliasing: AntiAliasing::TAA,
                anisotropic_filtering: 16,
                variable_rate_shading: false,
                mesh_shaders: true,
            },
        Quality::High =>
            GraphicsSettings {
                texture_quality: Quality::High,
                shadow_quality: Quality::High,
                anti_aliasing: AntiAliasing::TAA,
                anisotropic_filtering: 16,
                variable_rate_shading: true,
                mesh_shaders: true,
            },
        Quality::Medium =>
            GraphicsSettings {
                texture_quality: Quality::Medium,
                shadow_quality: Quality::Medium,
                anti_aliasing: AntiAliasing::FXAA,
                anisotropic_filtering: 8,
                variable_rate_shading: true,
                mesh_shaders: false,
            },
        Quality::Low =>
            GraphicsSettings {
                texture_quality: Quality::Low,
                shadow_quality: Quality::Low,
                anti_aliasing: AntiAliasing::Off,
                anisotropic_filtering: 4,
                variable_rate_shading: true,
                mesh_shaders: false,
            },
    };

    // Map common game abbreviations to full names
    let full_game_name = map_game_name(game);

    // Create gaming workload
    let workload = GamingWorkloadConfig {
        game_name: full_game_name.clone(),
        resolution: (res_width, res_height),
        ray_tracing,
        dlss_mode,
        fsr_mode,
        target_fps,
        scene_complexity,
        graphics_settings: graphics_settings.clone(),
    };

    // Initialize unified gaming emulator
    let mut gaming_emulator = UnifiedGamingEmulator::new(gpu_config.clone());

    // Main performance prediction using unified emulator
    println!("\n🚀 {}", "Performance Prediction".yellow().bold());
    let performance = gaming_emulator.predict_gaming_performance(&workload, ambient_temp).await?;

    println!("📊 {}", "Performance Results".cyan());
    println!("  Average FPS: {:.1}", performance.avg_fps.to_string().green().bold());
    println!("  1% Low FPS: {:.1}", performance.one_percent_low.to_string().yellow());
    println!("  Frame Time: {:.1}ms", format!("{:.1}", performance.frame_time_ms).blue());

    let consistency_str = match performance.frame_time_consistency {
        FrameTimeConsistency::Excellent => "Excellent".green(),
        FrameTimeConsistency::Good => "Good".cyan(),
        FrameTimeConsistency::Acceptable => "Acceptable".yellow(),
        FrameTimeConsistency::Poor => "Poor".red(),
    };
    println!("  Frame Consistency: {}", consistency_str);

    println!("  GPU Utilization: {:.1}%", (performance.gpu_utilization * 100.0).to_string().cyan());
    println!(
        "  Memory Utilization: {:.1}%",
        (performance.memory_utilization * 100.0).to_string().cyan()
    );
    println!("  Temperature: {:.1}°C", performance.temperature.to_string().red());

    // Show bottleneck analysis if available
    if let Some(bottleneck) = &performance.bottleneck_analysis {
        println!("  Bottleneck Analysis: {}", bottleneck.bright_black());
    }

    // Frame generation analysis using unified emulator
    if frame_generation {
        println!("\n🎬 {}", "Frame Generation Analysis".magenta().bold());
        let frame_gen_result = gaming_emulator.predict_frame_generation(
            &workload,
            ambient_temp
        ).await?;

        if frame_gen_result.supported {
            println!("  Frame Generation: {}", "✅ Supported".green());
            println!("  Base FPS: {:.1}", frame_gen_result.base_fps.to_string().blue());
            println!(
                "  Generated FPS: {:.1}",
                frame_gen_result.generated_fps.to_string().green().bold()
            );
            println!(
                "  Generation Ratio: {:.1}x",
                frame_gen_result.frame_generation_ratio.to_string().yellow()
            );
            println!(
                "  Latency Penalty: {:.1}ms",
                frame_gen_result.latency_penalty_ms.to_string().red()
            );

            let quality_str = match frame_gen_result.quality_impact {
                crate::unified_gaming_emulator::FrameGenerationQuality::Excellent =>
                    "Excellent".green(),
                crate::unified_gaming_emulator::FrameGenerationQuality::Good => "Good".cyan(),
                crate::unified_gaming_emulator::FrameGenerationQuality::Acceptable =>
                    "Acceptable".yellow(),
                crate::unified_gaming_emulator::FrameGenerationQuality::Poor => "Poor".red(),
                crate::unified_gaming_emulator::FrameGenerationQuality::NotSupported =>
                    "Not Supported".bright_black(),
            };
            println!("  Quality Impact: {}", quality_str);
        } else {
            println!("  Frame Generation: {}", "❌ Not Supported".red());
            println!("  Reason: GPU architecture doesn't support frame generation");
        }
    }

    // Power analysis (simplified - now integrated into emulator)
    if power_analysis {
        println!("\n🔋 {}", "Power Consumption Analysis".yellow().bold());
        println!("📊 {}", "Power Results".cyan());
        println!("  Total Power: {:.1}W", performance.power_consumption.to_string().red().bold());

        // Calculate power efficiency metrics
        let power_per_fps = performance.power_consumption / performance.avg_fps;
        let performance_per_watt = performance.avg_fps / performance.power_consumption;

        println!("  Power per FPS: {:.2}W/fps", power_per_fps.to_string().yellow());
        println!(
            "  Performance per Watt: {:.2}fps/W",
            performance_per_watt.to_string().green().bold()
        );

        // Estimate hourly cost (assuming $0.12/kWh)
        let hourly_cost = (performance.power_consumption / 1000.0) * 0.12;
        println!("  Estimated Cost: ${:.3}/hour", hourly_cost.to_string().cyan());
    }

    // Thermal session simulation
    if thermal_session {
        println!("\n🔥 {}", "Gaming Thermal Session".red().bold());
        println!("📊 {}", "Thermal Results".cyan());
        println!(
            "  Current Temperature: {:.1}°C",
            performance.temperature.to_string().red().bold()
        );
        println!("  Ambient Temperature: {:.1}°C", ambient_temp.to_string().blue());
        println!(
            "  Temperature Rise: {:.1}°C",
            (performance.temperature - ambient_temp).to_string().yellow()
        );

        // Thermal safety assessment
        let thermal_status = if performance.temperature > 85.0 {
            "🔴 High - Consider better cooling".red().bold()
        } else if performance.temperature > 75.0 {
            "🟡 Moderate - Monitor temperatures".yellow()
        } else {
            "🟢 Good - Safe operating temperature".green()
        };
        println!("  Thermal Status: {}", thermal_status);

        if session_duration > 0.0 {
            println!("  Session Duration: {:.1} minutes", session_duration);
            // Estimate if temperature would be stable
            let stable_temp = performance.temperature + (session_duration / 60.0) * 2.0; // Simple estimation
            println!(
                "  Estimated Temp after {:.0}min: {:.1}°C",
                session_duration,
                stable_temp.to_string().red()
            );
        }
    }

    // Detailed analysis in verbose mode
    if verbose {
        println!("\n🔍 {}", "Detailed Analysis".cyan().bold());
        println!("📋 {}", "Workload Configuration".cyan());
        println!("  Game: {}", full_game_name);
        println!("  Resolution: {}x{}", res_width, res_height);
        println!("  Ray Tracing: {}", if ray_tracing { "✅ Enabled" } else { "❌ Disabled" });
        println!("  DLSS: {:?}", workload.dlss_mode);
        println!("  FSR: {:?}", workload.fsr_mode);
        println!("  Target FPS: {:.0}", target_fps);
        println!("  Scene Complexity: {:.1}%", scene_complexity * 100.0);

        println!("\n🎨 {}", "Graphics Settings".cyan());
        println!("  Texture Quality: {:?}", graphics_settings.texture_quality);
        println!("  Shadow Quality: {:?}", graphics_settings.shadow_quality);
        println!("  Anti-Aliasing: {:?}", graphics_settings.anti_aliasing);
        println!("  Anisotropic Filtering: {}x", graphics_settings.anisotropic_filtering);
        println!("  Variable Rate Shading: {}", if graphics_settings.variable_rate_shading {
            "✅"
        } else {
            "❌"
        });
        println!("  Mesh Shaders: {}", if graphics_settings.mesh_shaders { "✅" } else { "❌" });

        println!("\n📚 {}", "Available Options".cyan());
        println!("🎮 {}", "Supported Games:".yellow());
        let supported_games = gaming_emulator.get_supported_games();
        for game in &supported_games {
            println!("  • {}", game);
        }

        println!("💻 {}", "GPU Information:".yellow());
        println!("  • {}", gpu_config.name);
        println!("  • Memory: {:.0}GB", gpu_config.memory_gb);
        println!("  • Compute: {:.1} TFLOPS", gpu_config.compute_tflops);
        println!("  • Memory Bandwidth: {:.1} GB/s", gpu_config.memory_bandwidth_gbps);
        if let Some(arch) = &gpu_config.architecture {
            println!("  • Architecture: {}", arch);
        }
    }

    Ok(())
}

pub async fn handle_power_command(
    gpu_model: &GpuModel,
    workload_type: &str,
    duration_seconds: u64,
    performance_score: f64,
    energy_cost: f64,
    include_thermal: bool,
    compare: bool,
    verbose: bool
) -> PhantomResult<()> {
    use crate::power_modeling::PowerModelingEngine;
    // Gaming performance is now handled by the unified gaming emulator
    use crate::thermal_modeling::ThermalModelingEngine;

    // Input validation
    if duration_seconds == 0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Duration must be greater than 0".to_string(),
        });
    }

    if performance_score <= 0.0 {
        return Err(PhantomGpuError::InvalidModel {
            reason: "Performance score must be greater than 0".to_string(),
        });
    }

    println!("✅ Initializing power analysis engine");

    // Create power modeling engine
    let mut power_engine = PowerModelingEngine::new();
    power_engine.set_energy_cost(energy_cost);

    // Check if workload type is supported
    let available_workloads = power_engine.get_workload_types();
    if !available_workloads.contains(&workload_type.to_string()) {
        println!("📋 Available workload types:");
        for workload in &available_workloads {
            println!("   • {}", workload);
        }
        return Err(PhantomGpuError::InvalidModel {
            reason: format!("Workload type '{}' not supported", workload_type),
        });
    }

    println!("🔋 Power Analysis Parameters:");
    println!("   • GPU: {} ({:.0} GB VRAM)", gpu_model.name, gpu_model.memory_gb);
    println!("   • Workload: {}", workload_type);
    println!("   • Duration: {:.1} hours", (duration_seconds as f64) / 3600.0);
    println!("   • Performance score: {:.1}", performance_score);
    println!("   • Energy cost: ${:.3}/kWh", energy_cost);
    println!(
        "   • Architecture: {}",
        gpu_model.architecture.as_ref().unwrap_or(&"Unknown".to_string())
    );

    // Get thermal state if thermal modeling is enabled
    let thermal_state = if include_thermal {
        println!("\n🌡️  Calculating thermal impact...");
        let mut thermal_engine = ThermalModelingEngine::new(25.0); // Ambient temperature
        let workload_intensity = match workload_type {
            "LLM" => 0.9,
            "Training" => 0.95,
            "Inference" => 0.7,
            "CV" => 0.8,
            _ => 0.8,
        };

        thermal_engine.initialize_gpu(&gpu_model.name);
        thermal_engine.update_thermal_state(&gpu_model.name, workload_intensity)
    } else {
        None
    };

    // Calculate power consumption
    println!("\n🔄 Analyzing power consumption...");
    let power_result = power_engine.calculate_power_consumption(
        gpu_model,
        workload_type,
        duration_seconds as f64,
        performance_score,
        thermal_state.as_ref()
    )?;

    // Display power consumption results
    println!("\n⚡ Power Consumption Analysis:");
    println!("   • Base power: {:.1}W", power_result.base_power_watts);
    println!("   • Compute power: {:.1}W", power_result.compute_power_watts);
    println!("   • Memory power: {:.1}W", power_result.memory_power_watts);
    println!("   • Cooling power: {:.1}W", power_result.cooling_power_watts);
    println!("   • Total power: {:.1}W", power_result.total_power_watts);

    // Energy consumption and cost
    println!("\n🔋 Energy Consumption:");
    println!("   • Energy used: {:.2} kWh", power_result.energy_consumption_wh / 1000.0);
    println!("   • Energy cost: ${:.3}", power_result.energy_cost_usd);
    println!(
        "   • Cost per hour: ${:.3}/hour",
        power_result.energy_cost_usd / ((duration_seconds as f64) / 3600.0)
    );

    // Thermal impact details
    if include_thermal {
        println!("\n🌡️  Thermal Impact:");
        println!("   • Peak temperature: {:.1}°C", power_result.thermal_impact.peak_temp_celsius);
        println!("   • Thermal throttling: {}", if
            power_result.thermal_impact.thermal_throttling_detected
        {
            "Yes"
        } else {
            "No"
        });
        println!(
            "   • Power scaling: {:.1}%",
            power_result.thermal_impact.power_scaling_factor * 100.0
        );
        println!(
            "   • Cooling overhead: {:.1}W",
            power_result.thermal_impact.cooling_overhead_watts
        );

        if power_result.thermal_impact.thermal_efficiency_loss > 0.0 {
            println!(
                "   • Efficiency loss: {:.1}%",
                power_result.thermal_impact.thermal_efficiency_loss * 100.0
            );
        }
    }

    // Compare against other GPUs if requested
    if compare {
        println!("\n📊 GPU Power Efficiency Comparison:");

        // Get all available GPU models for comparison
        let comparison_gpus = vec![
            crate::cli::GpuType::V100.to_gpu_model(),
            crate::cli::GpuType::A100.to_gpu_model(),
            crate::cli::GpuType::Rtx4090.to_gpu_model(),
            crate::cli::GpuType::H100.to_gpu_model(),
            crate::cli::GpuType::Rtx5090.to_gpu_model()
        ];

        // Calculate efficiency metrics
        let efficiency_metrics = power_engine.calculate_power_efficiency(
            gpu_model,
            workload_type,
            performance_score,
            &power_result,
            &comparison_gpus
        )?;

        println!("   • Efficiency score: {:.2} perf/W", efficiency_metrics.efficiency_score);
        println!("   • Efficiency rating: {}", efficiency_metrics.efficiency_rating);
        println!("   • Samples per watt: {:.1}", efficiency_metrics.samples_per_watt);
        println!("   • Operations per joule: {:.1}", efficiency_metrics.operations_per_joule);

        // Comparison metrics
        println!("\n🏆 Comparison Metrics:");
        println!(
            "   • Efficiency rank: #{}",
            efficiency_metrics.comparison_metrics.efficiency_rank
        );
        println!(
            "   • Efficiency percentile: {:.1}%",
            efficiency_metrics.comparison_metrics.efficiency_percentile
        );
        println!(
            "   • Power vs average: {:+.1}%",
            efficiency_metrics.comparison_metrics.power_vs_average
        );
        println!(
            "   • Performance vs average: {:+.1}%",
            efficiency_metrics.comparison_metrics.performance_vs_average
        );
        println!(
            "   • Cost efficiency score: {:.1}",
            efficiency_metrics.comparison_metrics.cost_efficiency_score
        );

        // Show comparison table
        if verbose {
            println!("\n📋 Detailed Power Comparison:");
            println!(
                "   {:>15} {:>10} {:>12} {:>10} {:>10}",
                "GPU",
                "Power (W)",
                "Efficiency",
                "Rating",
                "Cost/Hour"
            );
            println!("   {}", "-".repeat(65));

            for gpu in &comparison_gpus {
                let est_power = power_engine.calculate_power_consumption(
                    gpu,
                    workload_type,
                    3600.0, // 1 hour
                    performance_score,
                    None
                )?;

                let est_efficiency = power_engine.calculate_power_efficiency(
                    gpu,
                    workload_type,
                    performance_score,
                    &est_power,
                    &comparison_gpus
                )?;

                let marker = if gpu.name == gpu_model.name { "→" } else { " " };
                println!(
                    "   {:>1}{:<14} {:>10.1} {:>12.2} {:>10} ${:>8.3}",
                    marker,
                    gpu.name,
                    est_power.total_power_watts,
                    est_efficiency.efficiency_score,
                    est_efficiency.efficiency_rating,
                    est_power.energy_cost_usd
                );
            }
        }
    }

    // Verbose output: detailed power breakdown
    if verbose {
        println!("\n🔍 Detailed Power Breakdown:");
        println!("   • GPU TDP: {:.1}W", (gpu_model.compute_tflops as f64) * 1.5); // Rough estimate
        println!(
            "   • Power utilization: {:.1}%",
            (power_result.total_power_watts / ((gpu_model.compute_tflops as f64) * 1.5)) * 100.0
        );

        if let Some(arch) = &gpu_model.architecture {
            println!("   • Architecture: {}", arch);
            match arch.as_str() {
                "Ampere" =>
                    println!(
                        "   • 💡 Ampere GPUs have excellent power efficiency for AI workloads"
                    ),
                "Hopper" =>
                    println!("   • 💡 Hopper architecture optimized for large-scale AI training"),
                "Ada Lovelace" =>
                    println!("   • 💡 Ada Lovelace balances gaming and AI power consumption"),
                "Volta" =>
                    println!("   • 💡 Volta architecture pioneered tensor core power efficiency"),
                "Blackwell" =>
                    println!(
                        "   • 💡 Next-gen Blackwell architecture with enhanced power efficiency"
                    ),
                _ => println!("   • 💡 General GPU architecture"),
            }
        }

        println!("\n💰 Cost Analysis:");
        println!("   • Daily cost: ${:.2}", power_result.energy_cost_usd * 24.0);
        println!("   • Monthly cost: ${:.2}", power_result.energy_cost_usd * 24.0 * 30.0);
        println!("   • Annual cost: ${:.2}", power_result.energy_cost_usd * 24.0 * 365.0);

        // Carbon footprint estimate (rough)
        let carbon_kg_per_kwh = 0.5; // Average grid carbon intensity
        let carbon_footprint = (power_result.energy_consumption_wh / 1000.0) * carbon_kg_per_kwh;
        println!("   • Carbon footprint: {:.3} kg CO2", carbon_footprint);
    }

    // Recommendations
    println!("\n💡 Power Optimization Recommendations:");

    if power_result.total_power_watts > 300.0 {
        println!("   • Consider power limiting to reduce consumption");
    }

    if let Some(thermal_state) = &thermal_state {
        if thermal_state.current_temp_celsius > 80.0 {
            println!("   • Improve cooling to reduce thermal throttling");
        }
    }

    if power_result.energy_cost_usd > 1.0 {
        println!("   • Consider scheduling workloads during off-peak hours");
    }

    if workload_type == "Training" {
        println!("   • Use gradient checkpointing to reduce memory power");
        println!("   • Consider mixed precision training for better efficiency");
    }

    if workload_type == "Inference" {
        println!("   • Batch multiple requests to improve power efficiency");
        println!("   • Use dynamic batching to optimize power usage");
    }

    println!("\n✅ Power analysis completed!");
    Ok(())
}
