//! Command handlers for the CLI interface

use tracing::{ warn, info };
use colored::Colorize;

use crate::cli::{ ModelType, PretrainedModel, CloudProvider, GpuType };
#[cfg(feature = "real-models")]
use crate::cli::{ ModelFormat, OutputFormat, WorkloadType };
use crate::errors::{ PhantomGpuError, PhantomResult };
use crate::emulator::MultiGPUEmulator;
use crate::models::ModelConfig;
use crate::benchmarks::BenchmarkSuite;
use crate::gpu_config::{ GpuModel, GpuModelManager };

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
