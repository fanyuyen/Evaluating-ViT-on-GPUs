from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
from dataloader import get_cifar10_dataloader, get_imagenet100_dataloader, get_food101_dataloader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import torch
from tqdm import tqdm
import subprocess
from torchvision import transforms
import pandas as pd
import numpy as np

from pynvml import *
nvmlInit()
gpu_index_nvml = 1
handle = nvmlDeviceGetHandleByIndex(gpu_index_nvml)
gpu_name = nvmlDeviceGetName(handle)
print(f"Running on GPU (pynvml): {gpu_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on GPU (pytorch): ", torch.cuda.get_device_name(0))

def get_gpu_metrics(handle):
    """Get comprehensive GPU metrics"""
    util = nvmlDeviceGetUtilizationRates(handle)
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
    power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
    
    return {
        'gpu_util': util.gpu,
        'gpu_mem_used': mem_info.used / (1024 ** 2),  # MB
        'gpu_mem_total': mem_info.total / (1024 ** 2),  # MB
        'gpu_temp': temp,
        'gpu_power': power
    }

def run_inference(batch_size):
    # DataLoader
    dataloader = get_food101_dataloader(batch_size, train=False, subset_size=4000)

    # Model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to("cuda")
    model.eval()

    # Metric tracking
    all_preds = []
    all_labels = []
    gpu_metrics = []
    latencies = []

    torch.cuda.reset_peak_memory_stats(device)
    inference_start = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Running Inference (batch_size={batch_size})"):
            images = batch["image"]
            labels = batch["label"]
            
            # Process each image individually to ensure we get predictions for all images
            batch_preds = []
            for image in images:
                # Record start time for this image
                img_start = time.perf_counter()
                
                # Ensure image has 3 channels if it doesn't already
                if image.dim() == 2:
                    image = image.unsqueeze(0).repeat(3, 1, 1)
                
                # Convert tensor to PIL Image
                image_pil = transforms.ToPILImage()(image)
                
                # Process single image
                inputs = processor(images=image_pil, return_tensors="pt", do_rescale=False).to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                pred = logits.argmax(dim=-1).cpu().numpy()
                batch_preds.append(pred[0])
                
                # Record latency for this image
                img_end = time.perf_counter()
                latencies.append(img_end - img_start)
            
            all_preds.extend(batch_preds)
            all_labels.extend(labels.numpy())
            
            # Collect GPU metrics
            gpu_metrics.append(get_gpu_metrics(handle))

    inference_end = time.perf_counter()
    
    # Calculate metrics
    total_time = inference_end - inference_start
    num_images = len(dataloader.dataset)
    throughput = num_images / total_time
    peak_memory_MB = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
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
        'batch_size': batch_size,
        'total_time': total_time,
        'throughput': throughput,
        'peak_memory_MB': peak_memory_MB,
        **avg_metrics,
        **latency_stats
    }

# Test different batch sizes
batch_sizes = [1, 2, 4, 8, 16, 32, 64]  
results = []

for batch_size in batch_sizes:
    print(f"\nTesting batch size: {batch_size}")
    result = run_inference(batch_size)
    results.append(result)
    
    # Print results for current batch size
    print(f"Results for batch size {batch_size}:")
    print(f"Total inference time: {result['total_time']:.2f} seconds")
    print(f"Throughput: {result['throughput']:.2f} images/second")
    print(f"Peak VRAM usage: {result['peak_memory_MB']:.2f} MB")
    print(f"Average GPU Utilization: {result['gpu_util']:.2f}%")
    print(f"Average GPU Memory Usage: {result['gpu_mem_used']:.2f} MB")
    print(f"Average GPU Temperature: {result['gpu_temp']:.2f}°C")
    print(f"Average GPU Power Usage: {result['gpu_power']:.2f}W")
    print(f"Average Latency per Image: {result['avg_latency']*1000:.2f}ms")
    print(f"P95 Latency: {result['p95_latency']*1000:.2f}ms")
    print(f"P99 Latency: {result['p99_latency']*1000:.2f}ms")

# Create a DataFrame and save results to CSV
df = pd.DataFrame(results)
df.to_csv('gpu_performance_results_food101.csv', index=False)
print("\nResults saved to 'gpu_performance_results_food101.csv'")