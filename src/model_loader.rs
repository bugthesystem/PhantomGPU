//! Model Loading Interface
//!
//! This module provides a unified interface for loading ML models from various sources.
//! It conditionally re-exports functionality from the real_model_loader module when
//! the real-models feature is enabled.

// Re-export everything from real_model_loader for the main interface when feature is enabled
#[cfg(feature = "real-models")]
pub use crate::real_model_loader::{
    ModelFormat,
    RealModelInfo,
    RealModelLoader,
    PopularRealModels,
};

// For backwards compatibility with existing code
#[cfg(feature = "real-models")]
pub type ModelInfo = RealModelInfo;
#[cfg(feature = "real-models")]
pub type ModelLoader = RealModelLoader;
#[cfg(feature = "real-models")]
pub type PopularModels = PopularRealModels;

/// Convenience function to create a new model loader
#[cfg(feature = "real-models")]
pub fn new_model_loader(cache_dir: Option<std::path::PathBuf>) -> RealModelLoader {
    RealModelLoader::new(cache_dir)
}

/// Convenience function to load a model from any source
#[cfg(feature = "real-models")]
pub async fn load_model_from_source(source: &str) -> crate::errors::PhantomResult<RealModelInfo> {
    let loader = RealModelLoader::new(None);
    loader.load_from_source(source).await
}

/// Convenience function to get popular models list
#[cfg(feature = "real-models")]
pub fn get_popular_huggingface_models() -> Vec<&'static str> {
    PopularRealModels::popular_huggingface_models()
}

/// Convenience function to get popular ONNX models list
#[cfg(feature = "real-models")]
pub fn get_popular_onnx_models() -> Vec<(&'static str, &'static str)> {
    PopularRealModels::popular_onnx_models()
}
