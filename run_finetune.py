import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import json
from datetime import datetime
from dataloader import get_brain_tumor_dataloader
from transformers import ViTForImageClassification, ViTImageProcessor
from pynvml import *
import numpy as np
import pandas as pd
import socket
from torchvision import transforms

class GPUMonitor:
    def __init__(self, gpu_index=0, torch_index=None):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(gpu_index)
        self.gpu_name = nvmlDeviceGetName(self.handle)
        self.gpu_memory = nvmlDeviceGetMemoryInfo(self.handle).total / (1024 ** 3)  # Convert to GB
        self.gpu_compute_capability = nvmlDeviceGetCudaComputeCapability(self.handle)
        self.gpu_uuid = nvmlDeviceGetUUID(self.handle)
        
        # Get PyTorch device index
        if torch_index is None:
            torch_index = torch.cuda.current_device()
        else:
            # Set PyTorch to use the specified device
            torch.cuda.set_device(torch_index)
        
        # Validate that PyTorch and NVML are using the same GPU
        torch_device = torch.cuda.get_device_properties(torch_index)
        torch_uuid = str(torch_device.uuid)
        nvml_uuid_normalized = self.gpu_uuid[4:] if self.gpu_uuid.startswith('GPU-') else self.gpu_uuid
        
        if torch_uuid != nvml_uuid_normalized:
            raise RuntimeError(
                f"❌ UUID mismatch!\n"
                f"  PyTorch device {torch_index} UUID: {torch_uuid}\n"
                f"  NVML device {gpu_index} UUID: {self.gpu_uuid}\n"
                f"Make sure you're comparing the same physical GPU."
            )
        else:
            print(f"✅ UUID match confirmed: PyTorch GPU {torch_index} ↔ NVML GPU {gpu_index}")
        
        print(f"Running on GPU: {self.gpu_name}")
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

def train_model(model, processor, train_loader, val_loader, num_epochs=10, learning_rate=0.001, gpu_index=0, torch_index=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize GPU monitor with optional torch_index
    gpu_monitor = GPUMonitor(gpu_index=gpu_index, torch_index=torch_index)
    
    # Create output directory
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'outputs_finetune_{hostname}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Training metrics
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_metrics': [],  # Will store per-epoch GPU and timing metrics
        'total_time': None
    }
    
    # Start total time tracking
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start = time.time()
        epoch_gpu_metrics = []  # Store GPU metrics for this epoch
        epoch_batch_times = []  # Store batch times for this epoch
        
        for i, (inputs, labels) in enumerate(train_loader):
            batch_start = time.time()
            
            # Convert tensor to PIL images for ViT processor
            pil_images = []
            for img in inputs:
                pil_img = transforms.ToPILImage()(img)
                pil_images.append(pil_img)
            
            # Process images using ViT processor
            processed = processor(images=pil_images, return_tensors="pt")
            inputs = processed.pixel_values.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            
            # Record metrics every 10 batches
            if i % 10 == 0:
                batch_time = time.time() - batch_start
                epoch_batch_times.append(batch_time)
                epoch_gpu_metrics.append(gpu_monitor.get_metrics())
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Convert tensor to PIL images for ViT processor
                pil_images = []
                for img in inputs:
                    pil_img = transforms.ToPILImage()(img)
                    pil_images.append(pil_img)
                
                # Process images using ViT processor
                processed = processor(images=pil_images, return_tensors="pt")
                inputs = processed.pixel_values.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Calculate average GPU metrics for this epoch
        avg_gpu_metrics = {
            'gpu_util': np.mean([m['gpu_util'] for m in epoch_gpu_metrics]),
            'gpu_mem_used': np.mean([m['gpu_mem_used'] for m in epoch_gpu_metrics]),
            'gpu_mem_total': epoch_gpu_metrics[0]['gpu_mem_total'],  # This is constant
            'gpu_temp': np.mean([m['gpu_temp'] for m in epoch_gpu_metrics]),
            'gpu_power': np.mean([m['gpu_power'] for m in epoch_gpu_metrics])
        }
        
        # Record metrics
        metrics['train_loss'].append(epoch_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_acc'].append(epoch_acc)
        metrics['val_acc'].append(val_acc)
        
        # Record epoch-level metrics
        epoch_time = time.time() - epoch_start
        metrics['epoch_metrics'].append({
            'epoch': epoch + 1,
            'time': epoch_time,
            'avg_batch_time': np.mean(epoch_batch_times),
            'gpu_metrics': avg_gpu_metrics
        })
        
        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
              f'Time: {epoch_time:.2f}s, '
              f'GPU Util: {avg_gpu_metrics["gpu_util"]:.1f}%, '
              f'GPU Mem: {avg_gpu_metrics["gpu_mem_used"]/1024:.1f}GB')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'train_acc': epoch_acc,
            'val_acc': val_acc
        }
        torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Calculate and save total training time
    total_time = time.time() - total_start_time
    metrics['total_time'] = total_time
    print(f'\nTotal training time: {total_time/3600:.2f} hours ({total_time:.2f} seconds)')
    
    # Save final metrics
    with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    # Save a more readable summary CSV
    summary_data = []
    for epoch_metric in metrics['epoch_metrics']:
        summary_data.append({
            'Epoch': epoch_metric['epoch'],
            'Training Loss': metrics['train_loss'][epoch_metric['epoch']-1],
            'Validation Loss': metrics['val_loss'][epoch_metric['epoch']-1],
            'Training Accuracy (%)': metrics['train_acc'][epoch_metric['epoch']-1],
            'Validation Accuracy (%)': metrics['val_acc'][epoch_metric['epoch']-1],
            'Epoch Time (s)': epoch_metric['time'],
            'Average Batch Time (s)': epoch_metric['avg_batch_time'],
            'GPU Utilization (%)': epoch_metric['gpu_metrics']['gpu_util'],
            'GPU Memory Used (GB)': epoch_metric['gpu_metrics']['gpu_mem_used']/1024,
            'GPU Temperature (°C)': epoch_metric['gpu_metrics']['gpu_temp'],
            'GPU Power (W)': epoch_metric['gpu_metrics']['gpu_power']
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, 'epoch_summary.csv'), index=False)
    
    return model, metrics

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create dataloaders
    train_loader = get_brain_tumor_dataloader(batch_size=32, train=True)
    val_loader = get_brain_tumor_dataloader(batch_size=32, train=False)
    
    # Load ViT model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    # Modify the classification head for brain tumor classification
    num_classes = len(train_loader.dataset.classes)
    model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    
    # Train the model
    trained_model, metrics = train_model(
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        learning_rate=0.001,
        gpu_index=0,  # For jin (4070) and tian (2080)
        # gpu_index=1 # For huo (A100)
        # gpu_index=3, torch_index=3  # For other GPUs on jin
    ) 