//! Error handling for Phantom GPU Emulator

use colored::*;

/// Production-ready error handling with recovery suggestions
#[derive(thiserror::Error, Debug)]
pub enum PhantomGpuError {
    #[error(
        "GPU '{gpu_name}' out of memory: need {needed_mb:.1}MB, available {available_mb:.1}MB"
    )] OutOfMemory {
        gpu_name: String,
        needed_mb: f64,
        available_mb: f64,
    },

    #[error("GPU '{gpu_name}' is busy with operation '{current_task}'")] DeviceBusy {
        gpu_name: String,
        current_task: String,
    },

    #[error("Model configuration invalid: {reason}")] InvalidModel {
        reason: String,
    },

    #[error("GPU model '{gpu_name}' not found in configuration")] GpuNotFound {
        gpu_name: String,
    },

    #[error("Benchmark failed: {operation} - {reason}")] BenchmarkFailed {
        operation: String,
        reason: String,
    },

    #[error("Cloud provider '{provider}' not supported")] UnsupportedProvider {
        provider: String,
    },

    #[error("Configuration error: {message}")] ConfigError {
        message: String,
    },

    #[error("IO error: {0}")] IoError(#[from] std::io::Error),

    #[error("Model loading error: {0}")] ModelError(String),

    #[error("Model loading failed: {0}")] ModelLoadError(String),
}

pub type PhantomResult<T> = Result<T, PhantomGpuError>;

// Convert common errors to our structured types
impl From<String> for PhantomGpuError {
    fn from(msg: String) -> Self {
        PhantomGpuError::ConfigError { message: msg }
    }
}

/// Graceful error handler with recovery suggestions
pub fn handle_error(error: &PhantomGpuError) -> ! {
    eprintln!("\n{}", error.to_string().red().bold());

    match error {
        PhantomGpuError::OutOfMemory { .. } => {
            eprintln!("{}", "💡 Try: Reduce batch size or use a simpler model".yellow());
            eprintln!("{}", "💡 Example: cargo run -- train --batch-size 8".cyan());
        }
        PhantomGpuError::DeviceBusy { .. } => {
            eprintln!("{}", "💡 Try: Wait for the current operation to complete".yellow());
            eprintln!("{}", "💡 Example: cargo run -- train --wait".cyan());
        }
        PhantomGpuError::InvalidModel { .. } => {
            eprintln!("{}", "💡 Try: Check your input parameters".yellow());
            eprintln!("{}", "💡 Example: cargo run -- --help".cyan());
        }
        PhantomGpuError::GpuNotFound { .. } => {
            eprintln!(
                "{}",
                "💡 Try: Check gpu_models.toml exists or run with default GPU models".yellow()
            );
            eprintln!("{}", "💡 Example: cargo run -- list-gpus".cyan());
        }
        PhantomGpuError::BenchmarkFailed { .. } => {
            eprintln!("{}", "💡 Try: Reduce batch size or use a simpler model".yellow());
            eprintln!("{}", "💡 Example: cargo run -- train --batch-size 8".cyan());
        }
        PhantomGpuError::UnsupportedProvider { .. } => {
            eprintln!("{}", "💡 Try: Check internet connection for cloud cost estimation".yellow());
        }
        PhantomGpuError::ConfigError { .. } => {
            eprintln!("{}", "💡 Try: Check your configuration file".yellow());
            eprintln!("{}", "💡 Example: cargo run -- --help".cyan());
        }
        PhantomGpuError::IoError(_) => {
            eprintln!("{}", "💡 Try: Check your file system permissions".yellow());
            eprintln!("{}", "💡 Example: cargo run -- --help".cyan());
        }
        PhantomGpuError::ModelError(_) => {
            eprintln!("{}", "💡 Try: Please report this bug on GitHub".yellow());
            eprintln!("{}", "💡 Include: Command line arguments and error details".cyan());
        }
        PhantomGpuError::ModelLoadError(_) => {
            eprintln!("{}", "💡 Try: Check model file exists and format is supported".yellow());
            eprintln!("{}", "💡 Supported: ONNX (.onnx), PyTorch (.pth), Hugging Face Hub".cyan());
        }
    }

    std::process::exit(1);
}
