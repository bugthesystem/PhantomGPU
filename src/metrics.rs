//! Performance monitoring and metrics collection system

use std::collections::HashMap;
use std::time::{ Duration, Instant };
use std::sync::{ Arc, Mutex };

/// Performance metrics collection and reporting
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub operation_counts: HashMap<String, u64>,
    pub operation_times: HashMap<String, Vec<f64>>,
    pub memory_usage_history: Vec<f64>,
    pub throughput_history: Vec<f64>,
    pub gpu_utilization_history: Vec<f64>,
    pub start_time: Instant,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            operation_counts: HashMap::new(),
            operation_times: HashMap::new(),
            memory_usage_history: Vec::new(),
            throughput_history: Vec::new(),
            gpu_utilization_history: Vec::new(),
            start_time: Instant::now(),
        }
    }

    pub fn record_operation(&mut self, op_name: &str, duration_ms: f64) {
        *self.operation_counts.entry(op_name.to_string()).or_insert(0) += 1;
        self.operation_times.entry(op_name.to_string()).or_insert_with(Vec::new).push(duration_ms);
    }

    pub fn record_throughput(&mut self, samples_per_sec: f64) {
        self.throughput_history.push(samples_per_sec);
    }

    pub fn record_memory_usage(&mut self, memory_mb: f64) {
        self.memory_usage_history.push(memory_mb);
    }

    pub fn record_gpu_utilization(&mut self, utilization_percent: f64) {
        self.gpu_utilization_history.push(utilization_percent);
    }

    pub fn get_average_time(&self, op_name: &str) -> Option<f64> {
        self.operation_times
            .get(op_name)
            .map(|times| { times.iter().sum::<f64>() / (times.len() as f64) })
    }

    pub fn get_total_runtime(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("\n📊 Performance Metrics Report\n");
        report.push_str(&"=".repeat(50));
        report.push_str("\n");

        // Runtime statistics
        let total_time = self.get_total_runtime();
        report.push_str(&format!("🕐 Total Runtime: {:.2}s\n", total_time.as_secs_f64()));

        // Operation statistics
        report.push_str("\n🔧 Operation Statistics:\n");
        for (op_name, count) in &self.operation_counts {
            if let Some(avg_time) = self.get_average_time(op_name) {
                report.push_str(
                    &format!("   • {}: {} operations, avg {:.2}ms\n", op_name, count, avg_time)
                );
            }
        }

        // Performance statistics
        if !self.throughput_history.is_empty() {
            let avg_throughput =
                self.throughput_history.iter().sum::<f64>() /
                (self.throughput_history.len() as f64);
            let max_throughput = self.throughput_history.iter().fold(0.0f64, |a, &b| a.max(b));
            report.push_str(
                &format!(
                    "\n🚀 Throughput: avg {:.1} samples/sec, peak {:.1} samples/sec\n",
                    avg_throughput,
                    max_throughput
                )
            );
        }

        if !self.memory_usage_history.is_empty() {
            let avg_memory =
                self.memory_usage_history.iter().sum::<f64>() /
                (self.memory_usage_history.len() as f64);
            let max_memory = self.memory_usage_history.iter().fold(0.0f64, |a, &b| a.max(b));
            report.push_str(
                &format!("💾 Memory Usage: avg {:.1}MB, peak {:.1}MB\n", avg_memory, max_memory)
            );
        }

        if !self.gpu_utilization_history.is_empty() {
            let avg_util =
                self.gpu_utilization_history.iter().sum::<f64>() /
                (self.gpu_utilization_history.len() as f64);
            let max_util = self.gpu_utilization_history.iter().fold(0.0f64, |a, &b| a.max(b));
            report.push_str(
                &format!("🖥️  GPU Utilization: avg {:.1}%, peak {:.1}%\n", avg_util, max_util)
            );
        }

        report
    }
}

/// Global metrics collection
pub static GLOBAL_METRICS: std::sync::LazyLock<Arc<Mutex<PerformanceMetrics>>> = std::sync::LazyLock::new(
    || Arc::new(Mutex::new(PerformanceMetrics::new()))
);

/// Convenience macros for metrics collection
#[macro_export]
macro_rules! record_operation {
    ($op_name:expr, $duration_ms:expr) => {
        if let Ok(mut metrics) = crate::metrics::GLOBAL_METRICS.lock() {
            metrics.record_operation($op_name, $duration_ms);
        }
    };
}

#[macro_export]
macro_rules! record_throughput {
    ($samples_per_sec:expr) => {
        if let Ok(mut metrics) = crate::metrics::GLOBAL_METRICS.lock() {
            metrics.record_throughput($samples_per_sec);
        }
    };
}

#[macro_export]
macro_rules! time_operation {
    ($op_name:expr, $code:block) => {
        {
        let start = std::time::Instant::now();
        let result = $code;
        let duration_ms = start.elapsed().as_millis() as f64;
        record_operation!($op_name, duration_ms);
        result
        }
    };
}
