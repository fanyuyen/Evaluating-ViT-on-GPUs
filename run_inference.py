import os
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from dataloader import get_cifar10_dataloader, get_imagenet100_dataloader, get_food101_dataloader
import time
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from pynvml import *
import pandas as pd

class GPUMonitor:
    def __init__(self, gpu_index=1):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(gpu_index)
        self.gpu_name = nvmlDeviceGetName(self.handle)
        print(f"Running on GPU (pynvml): {self.gpu_name}")
        
    def get_metrics(self):
        """Get comprehensive GPU metrics"""
        util = nvmlDeviceGetUtilizationRates(self.handle)
        mem_info = nvmlDeviceGetMemoryInfo(self.handle)
        temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
        power = nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to watts
        
        return {
            'gpu_util': util.gpu,
            'gpu_mem_used': mem_info.used / (1024 ** 2),  # MB
            'gpu_mem_total': mem_info.total / (1024 ** 2),  # MB
            'gpu_temp': temp,
            'gpu_power': power
        }

class InferenceRunner:
    def __init__(self, dataset_name, batch_size, subset_size=400):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_monitor = GPUMonitor()
        
        # Initialize model and processor
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(self.device)
        self.model.eval()
        
        # Setup dataloader
        self.dataloader = self._get_dataloader()
        
    def _get_dataloader(self):
        """Get appropriate dataloader based on dataset name"""
        if self.dataset_name == 'cifar10':
            return get_cifar10_dataloader(self.batch_size, train=False, subset_size=self.subset_size)
        elif self.dataset_name == 'imagenet100':
            return get_imagenet100_dataloader(self.batch_size, train=False, subset_size=self.subset_size)
        elif self.dataset_name == 'food101':
            return get_food101_dataloader(self.batch_size, train=False, subset_size=self.subset_size)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def run_inference(self):
        """Run inference and collect metrics"""
        all_preds = []
        all_labels = []
        gpu_metrics = []
        latencies = []

        torch.cuda.reset_peak_memory_stats(self.device)
        inference_start = time.perf_counter()

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc=f"Running Inference on {self.dataset_name} (batch_size={self.batch_size})"):
                images = batch["image"]
                labels = batch["label"]
                
                batch_preds = []
                for image in images:
                    img_start = time.perf_counter()
                    
                    if image.dim() == 2:
                        image = image.unsqueeze(0).repeat(3, 1, 1)
                    
                    image_pil = transforms.ToPILImage()(image)
                    inputs = processor(images=image_pil, return_tensors="pt", do_rescale=False).to(self.device)
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    pred = logits.argmax(dim=-1).cpu().numpy()
                    batch_preds.append(pred[0])
                    
                    img_end = time.perf_counter()
                    latencies.append(img_end - img_start)
                
                all_preds.extend(batch_preds)
                all_labels.extend(labels.numpy())
                gpu_metrics.append(self.gpu_monitor.get_metrics())

        inference_end = time.perf_counter()
        
        # Calculate metrics
        total_time = inference_end - inference_start
        num_images = len(self.dataloader.dataset)
        throughput = num_images / total_time
        peak_memory_MB = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        
        # Calculate average GPU metrics
        avg_metrics = {
            'gpu_util': np.mean([m['gpu_util'] for m in gpu_metrics]),
            'gpu_mem_used': np.mean([m['gpu_mem_used'] for m in gpu_metrics]),
            'gpu_temp': np.mean([m['gpu_temp'] for m in gpu_metrics]),
            'gpu_power': np.mean([m['gpu_power'] for m in gpu_metrics])
        }
        
        # Calculate latency statistics
        latency_stats = {
            'avg_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99)
        }
        
        return {
            'dataset': self.dataset_name,
            'batch_size': self.batch_size,
            'total_time': total_time,
            'throughput': throughput,
            'peak_memory_MB': peak_memory_MB,
            **avg_metrics,
            **latency_stats
        }

def create_output_dirs():
    """Create output directory structure"""
    dirs = {
        'data': 'outputs/data',
        'plots': 'outputs/plots',
        'stats': 'outputs/stats'
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def main():
    # Setup
    output_dirs = create_output_dirs()
    datasets = ['cifar10', 'imagenet100', 'food101']
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    results = []
    
    # Run inference for each dataset and batch size
    for dataset in datasets:
        dataset_results = []
        for batch_size in batch_sizes:
            print(f"\nTesting {dataset} with batch size: {batch_size}")
            runner = InferenceRunner(dataset, batch_size)
            result = runner.run_inference()
            dataset_results.append(result)
            
            # Print immediate results
            print(f"Results for {dataset} (batch_size={batch_size}):")
            print(f"Total inference time: {result['total_time']:.2f} seconds")
            print(f"Throughput: {result['throughput']:.2f} images/second")
            print(f"Peak VRAM usage: {result['peak_memory_MB']:.2f} MB")
            print(f"Average GPU Utilization: {result['gpu_util']:.2f}%")
            print(f"Average GPU Temperature: {result['gpu_temp']:.2f}°C")
            print(f"Average Latency: {result['avg_latency']*1000:.2f}ms")
        
        # Save dataset results
        df = pd.DataFrame(dataset_results)
        df.to_csv(os.path.join(output_dirs['data'], f'gpu_performance_results_{dataset}.csv'), index=False)
        results.extend(dataset_results)
    
    # Save combined results
    df_all = pd.DataFrame(results)
    df_all.to_csv(os.path.join(output_dirs['data'], 'gpu_performance_results_all.csv'), index=False)
    
    print("\nAll results have been saved in the outputs directory")

if __name__ == "__main__":
    main() 