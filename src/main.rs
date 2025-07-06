//! Phantom GPU Emulator - Advanced GPU Emulation for ML Workloads
//!
//! A comprehensive GPU emulator that provides realistic performance modeling
//! for machine learning workloads across different GPU architectures.

use clap::Parser;
use colored::*;

// Module declarations
pub mod cli;
pub mod errors;
pub mod models;
pub mod emulator;
pub mod benchmarks;
pub mod metrics;
pub mod commands;

// Core modules
pub mod gpu_config;
pub mod neural_network_demo;
pub mod model_loader;
pub mod model_benchmarks;
pub mod cloud_cost_estimator;

// Real model support
#[cfg(feature = "real-models")]
pub mod real_model_loader;

#[cfg(feature = "real-models")]
pub mod model_comparison;

#[cfg(feature = "real-models")]
pub mod real_hardware_model;

#[cfg(feature = "real-models")]
pub mod real_hardware_config;

#[cfg(feature = "tensorflow")]
pub mod tensorflow_parser;

// ML Framework integrations
#[cfg(feature = "candle")]
pub mod candle_integration;

#[cfg(feature = "pytorch")]
pub mod pytorch_integration;

// Re-exports for convenience
pub use errors::{ PhantomGpuError, PhantomResult, handle_error };
pub use cli::{ Cli, Commands };
pub use gpu_config::GpuModel;

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        handle_error(&e);
    }
}

async fn run() -> PhantomResult<()> {
    let cli = Cli::parse();

    // Initialize logging based on verbosity
    let level = if cli.verbose { tracing::Level::DEBUG } else { tracing::Level::INFO };
    tracing_subscriber::fmt().with_max_level(level).init();

    // Print banner
    println!("\n{}", "🚀 Phantom GPU - Advanced GPU Emulator".cyan().bold());
    println!("{}", "Real ML workloads on virtual GPUs".bright_black());
    println!("{}", "=".repeat(50).bright_blue());

    // Execute command using the commands module
    match &cli.command {
        Commands::Train { model, batch_size, epochs, batches } => {
            let gpu_model = cli.gpu.to_gpu_model();
            println!(
                "\n{}",
                format!(
                    "🧠 Training {} on {}",
                    format!("{:?}", model).cyan(),
                    gpu_model.name.yellow()
                ).bold()
            );
            commands::handle_train_command(model, *batch_size, *epochs, *batches, gpu_model).await?;
        }

        Commands::Benchmark { model, batch_size, runs } => {
            let gpu_model = cli.gpu.to_gpu_model();
            println!(
                "\n{}",
                format!(
                    "⚡ Benchmarking {} on {}",
                    format!("{:?}", model).cyan(),
                    gpu_model.name.yellow()
                ).bold()
            );
            commands::handle_benchmark_command(model, *batch_size, *runs, gpu_model).await?;
        }

        Commands::Compare { gpus, model, batch_size } => {
            println!(
                "\n{}",
                format!("📊 Comparing GPUs with {}", format!("{:?}", model).cyan()).bold()
            );
            commands::handle_compare_command(gpus, model, *batch_size).await?;
        }

        Commands::Cost { model, hours, provider } => {
            println!(
                "\n{}",
                format!(
                    "💰 Cost estimation for {} ({:.1}h on {:?})",
                    format!("{:?}", model).cyan(),
                    hours,
                    provider
                ).bold()
            );
            commands::handle_cost_command(model, *hours, provider).await?;
        }

        Commands::Distributed { num_gpus, model, epochs } => {
            println!(
                "\n{}",
                format!(
                    "🌐 Distributed training: {} on {} GPUs",
                    format!("{:?}", model).cyan(),
                    num_gpus
                ).bold()
            );
            commands
                ::handle_distributed_command(*num_gpus, model, *epochs).await
                .map_err(|e| PhantomGpuError::BenchmarkFailed {
                    operation: "Distributed training".to_string(),
                    reason: format!("{}", e),
                })?;
        }

        Commands::Suite { experimental } => {
            println!("\n{}", "🧪 Running full benchmark suite".bold());
            commands
                ::handle_suite_command(*experimental).await
                .map_err(|e| PhantomGpuError::BenchmarkFailed {
                    operation: "Benchmark suite".to_string(),
                    reason: format!("{}", e),
                })?;
        }

        Commands::ListGpus => {
            commands::handle_list_gpus_command().map_err(|e| PhantomGpuError::ConfigError {
                message: format!("Failed to list GPUs: {}", e),
            })?;
        }

        #[cfg(feature = "real-models")]
        Commands::LoadModel { model, format, batch_size, runs } => {
            let gpu_model = cli.gpu.to_gpu_model();
            println!(
                "\n{}",
                format!(
                    "🤖 Loading real model: {} on {}",
                    model.cyan(),
                    gpu_model.name.yellow()
                ).bold()
            );
            commands::handle_load_model_command(
                model,
                format,
                *batch_size,
                *runs,
                gpu_model
            ).await?;
        }

        #[cfg(feature = "real-models")]
        Commands::CompareModels {
            models,
            gpus,
            batch_sizes,
            output,
            include_cost,
            fast_mode,
            show_progress,
            precision,
            real_hardware,
            hardware_profiles,
        } => {
            commands::handle_compare_models_command(
                models,
                gpus,
                batch_sizes,
                output,
                *include_cost,
                *fast_mode,
                *show_progress,
                *precision,
                *real_hardware,
                hardware_profiles.as_deref()
            ).await?;
        }

        #[cfg(feature = "real-models")]
        Commands::RecommendGpu {
            model,
            budget,
            batch_size,
            target_throughput,
            workload,
            cloud_providers,
            fast_mode,
        } => {
            commands::handle_recommend_gpu_command(
                model,
                *budget,
                *batch_size,
                *target_throughput,
                workload,
                cloud_providers,
                *fast_mode
            ).await?;
        }

        #[cfg(feature = "pytorch")]
        Commands::FrameworkCompare { batch_size } => {
            let gpu_model = cli.gpu.to_gpu_model();
            println!(
                "\n{}",
                format!("🥊 Framework Comparison on {}", gpu_model.name.yellow()).bold()
            );
            commands::handle_framework_compare_command(*batch_size, gpu_model).await?;
        }
    }

    println!("\n{}", "✅ Operation completed successfully!".green().bold());
    Ok(())
}
