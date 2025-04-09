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
import socket
import datetime
import nvtx


class GPUMonitor:
    def __init__(self, gpu_index=0):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(gpu_index)
        self.gpu_name = nvmlDeviceGetName(self.handle)
        self.gpu_memory = nvmlDeviceGetMemoryInfo(self.handle).total / (1024 ** 3)  # Convert to GB
        self.gpu_compute_capability = nvmlDeviceGetCudaComputeCapability(self.handle)
        self.gpu_uuid = nvmlDeviceGetUUID(self.handle)
        
        # Validate that PyTorch and NVML are using the same GPU
        torch_index = torch.cuda.current_device()
        torch_device = torch.cuda.get_device_properties(torch_index)
        torch_uuid = str(torch_device.uuid)
        nvml_uuid_normalized = self.gpu_uuid[4:] if self.gpu_uuid.startswith('GPU-') else self.gpu_uuid
        
        if torch_uuid != nvml_uuid_normalized:
            raise RuntimeError(
                f"  UUID mismatch!\n"
                f"  PyTorch device {torch_index} UUID: {torch_uuid}\n"
                f"  NVML device {gpu_index} UUID: {self.gpu_uuid}\n"
                f"Make sure you're comparing the same physical GPU."
            )
        else:
            print(f"UUID match confirmed: PyTorch GPU {torch_index} ↔ NVML GPU {gpu_index}")
        
        print(f"Running on GPU (pynvml): {self.gpu_name}")
        print(f"GPU Memory: {self.gpu_memory:.1f} GB")
        print(f"Compute Capability: {self.gpu_compute_capability[0]}.{self.gpu_compute_capability[1]}")
        print(f"GPU UUID: {self.gpu_uuid}")
        
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
    
    def get_gpu_info(self):
        """Get static GPU information"""
        return {
            'gpu_name': self.gpu_name,
            'gpu_memory_gb': self.gpu_memory,
            'gpu_compute_capability': f"{self.gpu_compute_capability[0]}.{self.gpu_compute_capability[1]}",
            'gpu_uuid': self.gpu_uuid
        }

class InferenceRunner:
    def __init__(self, dataset_name, batch_size, subset_size=400):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model and processor
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(self.device)
        self.model.eval()
        
        # Setup GPU monitor
        self.gpu_monitor = GPUMonitor() # For jin (4070) and tian/shui (2080)
        #self.gpu_monitor = GPUMonitor(gpu_index=1) # For huo (A100)
        
        # Setup dataloader with proper subset handling
        self.dataloader = self._get_dataloader()
        
    def _get_dataloader(self):
        """Get appropriate dataloader based on dataset name with proper subset handling"""
        try:
            if self.dataset_name == 'cifar10':
                return get_cifar10_dataloader(self.batch_size, train=False, subset_size=self.subset_size)
            elif self.dataset_name == 'imagenet100':
                # For ImageNet100, ensure subset_size doesn't exceed validation set size
                actual_subset = min(self.subset_size, 5000)  # ImageNet100 typically has 5000 validation images
                return get_imagenet100_dataloader(self.batch_size, train=False, subset_size=actual_subset)
            elif self.dataset_name == 'food101':
                # For Food101, ensure subset_size doesn't exceed validation set size
                actual_subset = min(self.subset_size, 25250)  # Food101 has 25250 validation images
                return get_food101_dataloader(self.batch_size, train=False, subset_size=actual_subset)
            else:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")
        except Exception as e:
            print(f"Error initializing {self.dataset_name} dataset: {str(e)}")
            print("Detailed error information:")
            import traceback
            traceback.print_exc()
            raise
    
    def run_inference(self):
        """Run inference and collect metrics"""
        all_preds = []
        all_labels = []
        gpu_metrics = []
        latencies = []

        # Reset memory stats before starting
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize()  # Ensure all operations are completed
        
        # Get initial NVML memory
        initial_nvml_mem = self.gpu_monitor.get_metrics()['gpu_mem_used']
        
        inference_start = time.perf_counter()

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc=f"Running Inference on {self.dataset_name} (batch_size={self.batch_size})"):
                # Handle both tuple/list and dict batch formats
                if isinstance(batch, (tuple, list)):
                    images, labels = batch
                else:
                    images = batch["image"]
                    labels = batch["label"]
                
                # Process entire batch at once
                batch_start = time.perf_counter()
                
                # Convert all images to PIL and process as batch
                images_pil = [transforms.ToPILImage()(img) for img in images]
                inputs = self.processor(images=images_pil, return_tensors="pt", do_rescale=False).to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = logits.argmax(dim=-1).cpu().numpy()
                
                batch_end = time.perf_counter()
                batch_latency = (batch_end - batch_start) / len(images)  # Average latency per image
                latencies.extend([batch_latency] * len(images))
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                gpu_metrics.append(self.gpu_monitor.get_metrics())

        inference_end = time.perf_counter()
        torch.cuda.synchronize()  # Ensure all operations are completed
        
        # Get final NVML memory
        final_nvml_mem = self.gpu_monitor.get_metrics()['gpu_mem_used']
        nvml_memory_increase = final_nvml_mem - initial_nvml_mem
        
        # Calculate metrics
        total_time = inference_end - inference_start
        num_images = len(self.dataloader.dataset)
        throughput = num_images / total_time
        
        # Get peak memory after all operations are complete
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
        
        # Get GPU information
        gpu_info = self.gpu_monitor.get_gpu_info()
        
        return {
            'dataset': self.dataset_name,
            'batch_size': self.batch_size,
            'total_time': total_time,
            'throughput': throughput,
            'peak_memory_MB': peak_memory_MB,
            'nvml_memory_increase_MB': nvml_memory_increase,  # Added NVML memory tracking
            **avg_metrics,
            **latency_stats,
            **gpu_info
        }

def create_output_dirs():
    """Create output directory structure"""
    # Get hostname and timestamp for unique directory
    hostname = socket.gethostname()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"outputs_{hostname}_{timestamp}"
    
    dirs = {
        'data': os.path.join(base_dir, 'data')
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def main():
    # Setup
    output_dirs = create_output_dirs()
    hostname = socket.gethostname()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure datasets and their maximum validation set sizes
    dataset_configs = {
        'cifar10': {'max_size': 10000, 'test_size': 10000},  # Full validation set
        'imagenet100': {'max_size': 5000, 'test_size': 5000},  # Full validation set
        'food101': {'max_size': 25250, 'test_size': 25250}  # Full validation set
    }
    
    batch_sizes = [1, 8, 16, 32, 64, 128]
    results = []
    
    # Run inference for each dataset and batch size
    for dataset_name, config in dataset_configs.items():
        dataset_results = []
        print(f"\n=== Starting tests for {dataset_name} dataset ===")
        print(f"Maximum validation set size: {config['max_size']}")
        print(f"Testing with subset size: {config['test_size']}")
        
        try:
            # Test first if we can create a dataloader with batch size 1
            print("Initializing test dataloader...")
            test_runner = InferenceRunner(dataset_name, batch_size=1, subset_size=2)
            test_batch = next(iter(test_runner.dataloader))
            print(f"Successfully initialized {dataset_name} dataset")
            del test_runner  # Clean up test runner
            
            for batch_size in batch_sizes:
                try:
                    print(f"\nTesting {dataset_name} with batch size: {batch_size}")
                    with nvtx.annotate(f"{dataset_name} | batch_size={batch_size}", color="blue"):
                        runner = InferenceRunner(dataset_name, batch_size, subset_size=config['test_size'])
                        result = runner.run_inference()
                    
                    # Add host and timestamp information
                    result.update({
                        'hostname': hostname,
                        'timestamp': timestamp
                    })
                    
                    dataset_results.append(result)
                    
                    # Print immediate results
                    print(f"Results for {dataset_name} (batch_size={batch_size}):")
                    print(f"Total inference time: {result['total_time']:.2f} seconds")
                    print(f"Throughput: {result['throughput']:.2f} images/second")
                    print(f"Peak VRAM usage: {result['peak_memory_MB']:.2f} MB")
                    print(f"Average GPU Utilization: {result['gpu_util']:.2f}%")
                    print(f"Average GPU Temperature: {result['gpu_temp']:.2f}°C")
                    print(f"Average Latency: {result['avg_latency']*1000:.2f}ms")
                    
                except Exception as e:
                    print(f"Error running inference for {dataset_name} with batch size {batch_size}:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    import traceback
                    print("Detailed traceback:")
                    traceback.print_exc()
                    print("Skipping this batch size and continuing with the next one...")
                    continue
            
            # Save dataset results if we have any
            if dataset_results:
                df = pd.DataFrame(dataset_results)
                output_file = os.path.join(output_dirs['data'], f'gpu_performance_results_{dataset_name}_{hostname}_{timestamp}.csv')
                df.to_csv(output_file, index=False)
                results.extend(dataset_results)
                print(f"\nResults for {dataset_name} saved to {output_file}")
            
        except Exception as e:
            print(f"Error initializing {dataset_name} dataset:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("Detailed traceback:")
            traceback.print_exc()
            print(f"Skipping {dataset_name} dataset entirely...")
            continue
    
    # Save combined results if we have any
    if results:
        df_all = pd.DataFrame(results)
        output_file = os.path.join(output_dirs['data'], f'gpu_performance_results_all_{hostname}_{timestamp}.csv')
        df_all.to_csv(output_file, index=False)
        print(f"\nAll available results have been saved to {output_file}")
    else:
        print("\nNo results were generated. Please check the errors above.")

if __name__ == "__main__":
    main()