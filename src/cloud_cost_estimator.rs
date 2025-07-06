// Cloud Cost Estimator for Real GPU Training
// Provides accurate pricing estimates for AWS, GCP, Azure

use std::collections::HashMap;
use serde::{ Deserialize, Serialize };
use crate::model_benchmarks::{ ModelInfo, PopularModels };

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudProvider {
    pub name: String,
    pub gpu_pricing: HashMap<String, GpuInstancePricing>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInstancePricing {
    pub instance_type: String,
    pub gpu_model: String,
    pub gpu_count: usize,
    pub hourly_cost_usd: f64,
    pub memory_gb: f32,
    pub compute_tflops: f32,
    pub spot_discount_percent: f64, // How much cheaper spot instances are
}

impl GpuInstancePricing {
    pub fn spot_cost(&self) -> f64 {
        self.hourly_cost_usd * (1.0 - self.spot_discount_percent / 100.0)
    }
}

/// Real cloud pricing data (as of 2024)
pub struct CloudPricing;

impl CloudPricing {
    pub fn aws() -> CloudProvider {
        let mut pricing = HashMap::new();

        // AWS EC2 instances (US-East-1 pricing)
        pricing.insert("p3.2xlarge".to_string(), GpuInstancePricing {
            instance_type: "p3.2xlarge".to_string(),
            gpu_model: "Tesla V100".to_string(),
            gpu_count: 1,
            hourly_cost_usd: 3.06,
            memory_gb: 16.0,
            compute_tflops: 15.7,
            spot_discount_percent: 70.0,
        });

        pricing.insert("p3.8xlarge".to_string(), GpuInstancePricing {
            instance_type: "p3.8xlarge".to_string(),
            gpu_model: "Tesla V100".to_string(),
            gpu_count: 4,
            hourly_cost_usd: 12.24,
            memory_gb: 64.0,
            compute_tflops: 62.8,
            spot_discount_percent: 70.0,
        });

        pricing.insert("p4d.24xlarge".to_string(), GpuInstancePricing {
            instance_type: "p4d.24xlarge".to_string(),
            gpu_model: "Tesla A100".to_string(),
            gpu_count: 8,
            hourly_cost_usd: 32.77,
            memory_gb: 320.0,
            compute_tflops: 312.0,
            spot_discount_percent: 60.0,
        });

        pricing.insert("g5.xlarge".to_string(), GpuInstancePricing {
            instance_type: "g5.xlarge".to_string(),
            gpu_model: "RTX A10G".to_string(),
            gpu_count: 1,
            hourly_cost_usd: 1.006,
            memory_gb: 24.0,
            compute_tflops: 31.2,
            spot_discount_percent: 50.0,
        });

        CloudProvider {
            name: "AWS".to_string(),
            gpu_pricing: pricing,
        }
    }

    pub fn gcp() -> CloudProvider {
        let mut pricing = HashMap::new();

        // Google Cloud GPU pricing
        pricing.insert("n1-standard-4-v100".to_string(), GpuInstancePricing {
            instance_type: "n1-standard-4-v100".to_string(),
            gpu_model: "Tesla V100".to_string(),
            gpu_count: 1,
            hourly_cost_usd: 2.48,
            memory_gb: 16.0,
            compute_tflops: 15.7,
            spot_discount_percent: 60.0,
        });

        pricing.insert("a2-highgpu-1g".to_string(), GpuInstancePricing {
            instance_type: "a2-highgpu-1g".to_string(),
            gpu_model: "Tesla A100".to_string(),
            gpu_count: 1,
            hourly_cost_usd: 3.673,
            memory_gb: 40.0,
            compute_tflops: 39.0,
            spot_discount_percent: 50.0,
        });

        pricing.insert("a2-highgpu-8g".to_string(), GpuInstancePricing {
            instance_type: "a2-highgpu-8g".to_string(),
            gpu_model: "Tesla A100".to_string(),
            gpu_count: 8,
            hourly_cost_usd: 29.39,
            memory_gb: 320.0,
            compute_tflops: 312.0,
            spot_discount_percent: 50.0,
        });

        CloudProvider {
            name: "GCP".to_string(),
            gpu_pricing: pricing,
        }
    }

    pub fn azure() -> CloudProvider {
        let mut pricing = HashMap::new();

        // Azure pricing
        pricing.insert("nc6s_v3".to_string(), GpuInstancePricing {
            instance_type: "nc6s_v3".to_string(),
            gpu_model: "Tesla V100".to_string(),
            gpu_count: 1,
            hourly_cost_usd: 3.168,
            memory_gb: 16.0,
            compute_tflops: 15.7,
            spot_discount_percent: 80.0,
        });

        pricing.insert("nd96asr_v4".to_string(), GpuInstancePricing {
            instance_type: "nd96asr_v4".to_string(),
            gpu_model: "Tesla A100".to_string(),
            gpu_count: 8,
            hourly_cost_usd: 27.2,
            memory_gb: 320.0,
            compute_tflops: 312.0,
            spot_discount_percent: 70.0,
        });

        CloudProvider {
            name: "Azure".to_string(),
            gpu_pricing: pricing,
        }
    }

    pub fn all_providers() -> Vec<CloudProvider> {
        vec![Self::aws(), Self::gcp(), Self::azure()]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingWorkload {
    pub model_info: ModelInfo,
    pub dataset_size: usize, // Number of samples
    pub batch_size: usize,
    pub epochs: usize,
    pub estimated_duration_hours: f64,
}

impl TrainingWorkload {
    pub fn new(
        model_info: ModelInfo,
        dataset_size: usize,
        batch_size: usize,
        epochs: usize
    ) -> Self {
        // Estimate training duration based on model complexity
        let total_samples = dataset_size * epochs;
        let batches_needed = (total_samples as f64) / (batch_size as f64);

        // Rough estimate: 1ms per GFLOP (very approximate)
        let time_per_batch_ms = model_info.estimated_flops_g;
        let estimated_duration_hours = (batches_needed * time_per_batch_ms) / (1000.0 * 3600.0);

        Self {
            model_info,
            dataset_size,
            batch_size,
            epochs,
            estimated_duration_hours,
        }
    }

    pub fn imagenet_resnet50() -> Self {
        Self::new(
            PopularModels::resnet50(),
            1_300_000, // ImageNet size
            128, // Batch size
            90 // Epochs
        )
    }

    pub fn bert_fine_tuning() -> Self {
        Self::new(
            PopularModels::bert_base(),
            100_000, // Fine-tuning dataset
            32, // Batch size
            3 // Epochs
        )
    }

    pub fn llama_7b_training() -> Self {
        Self::new(
            PopularModels::llama_7b(),
            10_000_000, // Large text dataset
            4, // Small batch due to memory
            1 // One epoch is expensive
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    pub provider: String,
    pub instance_type: String,
    pub workload_name: String,
    pub on_demand_cost_usd: f64,
    pub spot_cost_usd: f64,
    pub training_hours: f64,
    pub cost_per_sample: f64,
    pub cost_breakdown: CostBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub compute_cost: f64,
    pub storage_cost: f64,
    pub data_transfer_cost: f64,
    pub total_cost: f64,
}

pub struct CostEstimator;

impl CostEstimator {
    pub fn estimate_workload_cost(
        workload: &TrainingWorkload,
        instance: &GpuInstancePricing,
        provider: &str
    ) -> CostEstimate {
        // Calculate training time based on GPU performance
        let gpu_efficiency = (instance.compute_tflops as f64) / 35.0; // Normalize to RTX 4090
        let adjusted_hours = workload.estimated_duration_hours / gpu_efficiency;

        // On-demand vs spot pricing
        let on_demand_cost = adjusted_hours * instance.hourly_cost_usd;
        let spot_cost = adjusted_hours * instance.spot_cost();

        // Additional costs (storage, data transfer)
        let storage_cost = 50.0; // $50 for model storage per month
        let data_transfer_cost = 20.0; // $20 for data transfer

        let cost_breakdown = CostBreakdown {
            compute_cost: on_demand_cost,
            storage_cost,
            data_transfer_cost,
            total_cost: on_demand_cost + storage_cost + data_transfer_cost,
        };

        let total_samples = workload.dataset_size * workload.epochs;
        let cost_per_sample = cost_breakdown.total_cost / (total_samples as f64);

        CostEstimate {
            provider: provider.to_string(),
            instance_type: instance.instance_type.clone(),
            workload_name: workload.model_info.name.clone(),
            on_demand_cost_usd: on_demand_cost,
            spot_cost_usd: spot_cost,
            training_hours: adjusted_hours,
            cost_per_sample,
            cost_breakdown,
        }
    }

    pub fn compare_all_providers(workload: &TrainingWorkload) -> Vec<CostEstimate> {
        let mut estimates = Vec::new();

        for provider in CloudPricing::all_providers() {
            for (_, instance) in &provider.gpu_pricing {
                // Only recommend instances that can fit the model
                if (instance.memory_gb as f64) * 1024.0 > workload.model_info.memory_mb {
                    let estimate = Self::estimate_workload_cost(workload, instance, &provider.name);
                    estimates.push(estimate);
                }
            }
        }

        // Sort by cost (cheapest first)
        estimates.sort_by(|a, b| a.spot_cost_usd.partial_cmp(&b.spot_cost_usd).unwrap());
        estimates
    }

    pub fn print_cost_comparison(workload: &TrainingWorkload) {
        println!("\n💰 Cloud Cost Estimation");
        println!("{}", "=".repeat(80));
        println!("Workload: {} training", workload.model_info.name);
        println!(
            "Dataset: {} samples × {} epochs = {} total samples",
            workload.dataset_size,
            workload.epochs,
            workload.dataset_size * workload.epochs
        );
        println!("Estimated duration: {:.1} hours", workload.estimated_duration_hours);

        let estimates = Self::compare_all_providers(workload);

        println!("\n📊 Cost Comparison (sorted by spot price):");
        println!(
            "{:<15} {:<20} {:>12} {:>12} {:>10} {:>15}",
            "Provider",
            "Instance",
            "On-Demand",
            "Spot Price",
            "Hours",
            "$/Sample"
        );
        println!("{}", "-".repeat(95));

        for estimate in &estimates {
            println!(
                "{:<15} {:<20} {:>10}$ {:>10}$ {:>8.1}h {:>12.6}$",
                estimate.provider,
                estimate.instance_type,
                estimate.on_demand_cost_usd,
                estimate.spot_cost_usd,
                estimate.training_hours,
                estimate.cost_per_sample
            );
        }

        if let Some(cheapest) = estimates.first() {
            println!("\n🏆 Best Option: {} {} (Spot)", cheapest.provider, cheapest.instance_type);
            println!(
                "   Cost: ${:.2} ({:.1} hours)",
                cheapest.spot_cost_usd,
                cheapest.training_hours
            );
            println!(
                "   Savings vs most expensive: ${:.2}",
                estimates.last().unwrap().spot_cost_usd - cheapest.spot_cost_usd
            );
        }
    }
}

/// Real-world cost analysis scenarios
pub async fn run_cost_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💼 Real-World Training Cost Analysis");
    println!("{}", "=".repeat(60));

    let workloads = vec![
        TrainingWorkload::imagenet_resnet50(),
        TrainingWorkload::bert_fine_tuning(),
        TrainingWorkload::llama_7b_training()
    ];

    for workload in workloads {
        CostEstimator::print_cost_comparison(&workload);
        println!();
    }

    // ROI Analysis
    println!("\n📈 ROI Analysis");
    println!("{}", "=".repeat(40));
    println!("Training LLaMA-7B locally vs cloud:");

    let local_gpu_cost = 5000.0; // RTX 4090 cost
    let cloud_cost = CostEstimator::compare_all_providers(&TrainingWorkload::llama_7b_training())
        .first()
        .map(|est| est.spot_cost_usd)
        .unwrap_or(0.0);

    let break_even_runs = local_gpu_cost / cloud_cost;

    println!("   Local GPU: ${:.0}", local_gpu_cost);
    println!("   Cloud cost: ${:.2} per training run", cloud_cost);
    println!("   Break-even point: {:.0} training runs", break_even_runs);

    if break_even_runs > 10.0 {
        println!("   💡 Recommendation: Use cloud for experimentation");
    } else {
        println!("   💡 Recommendation: Consider buying local GPU");
    }

    Ok(())
}

// CLI Interface Functions

/// Alias for run_cost_analysis for CLI compatibility
pub async fn analyze_cloud_costs() -> Result<(), Box<dyn std::error::Error>> {
    run_cost_analysis().await
}

/// Estimate training costs for a specific model, hours, and provider
pub async fn estimate_training_costs(
    model: &crate::cli::PretrainedModel,
    hours: f64,
    provider: &crate::cli::CloudProvider
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💰 Training Cost Estimation");
    println!("{}", "=".repeat(50));

    // Create workload based on model type
    let workload = match model {
        crate::cli::PretrainedModel::ResNet50 => {
            let mut w = TrainingWorkload::imagenet_resnet50();
            w.estimated_duration_hours = hours;
            w
        }
        crate::cli::PretrainedModel::BertBase => {
            let mut w = TrainingWorkload::bert_fine_tuning();
            w.estimated_duration_hours = hours;
            w
        }
        crate::cli::PretrainedModel::Llama7b => {
            let mut w = TrainingWorkload::llama_7b_training();
            w.estimated_duration_hours = hours;
            w
        }
        _ => {
            // Default to ResNet50 for other models
            let mut w = TrainingWorkload::imagenet_resnet50();
            w.estimated_duration_hours = hours;
            w
        }
    };

    // Get pricing for specific provider
    let cloud_provider = match provider {
        crate::cli::CloudProvider::Aws => CloudPricing::aws(),
        crate::cli::CloudProvider::Gcp => CloudPricing::gcp(),
        crate::cli::CloudProvider::Azure => CloudPricing::azure(),
    };

    println!("Model: {:?}", model);
    println!("Duration: {:.1} hours", hours);
    println!("Provider: {:?}", provider);
    println!();

    // Calculate costs for all instances from this provider
    let mut estimates = Vec::new();
    for (_, instance) in &cloud_provider.gpu_pricing {
        if (instance.memory_gb as f64) * 1024.0 > workload.model_info.memory_mb {
            let estimate = CostEstimator::estimate_workload_cost(
                &workload,
                instance,
                &cloud_provider.name
            );
            estimates.push(estimate);
        }
    }

    // Sort by spot cost
    estimates.sort_by(|a, b| a.spot_cost_usd.partial_cmp(&b.spot_cost_usd).unwrap());

    if estimates.is_empty() {
        println!("❌ No suitable instances found for this model");
        return Ok(());
    }

    println!("📊 Available Instances:");
    println!(
        "{:<20} {:>12} {:>12} {:>10} {:>15}",
        "Instance Type",
        "On-Demand",
        "Spot Price",
        "Hours",
        "$/Sample"
    );
    println!("{}", "-".repeat(70));

    for estimate in &estimates {
        println!(
            "{:<20} {:>10}$ {:>10}$ {:>8.1}h {:>12.6}$",
            estimate.instance_type,
            estimate.on_demand_cost_usd,
            estimate.spot_cost_usd,
            estimate.training_hours,
            estimate.cost_per_sample
        );
    }

    if let Some(cheapest) = estimates.first() {
        println!("\n🏆 Recommended: {}", cheapest.instance_type);
        println!("   Spot cost: ${:.2}", cheapest.spot_cost_usd);
        println!(
            "   Potential savings: ${:.2} vs on-demand",
            cheapest.on_demand_cost_usd - cheapest.spot_cost_usd
        );
    }

    Ok(())
}
