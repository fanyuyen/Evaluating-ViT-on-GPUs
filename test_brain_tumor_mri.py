import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import get_brain_tumor_dataloader
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from pynvml import *
import pandas as pd
import socket
import time
from torchvision import transforms
import argparse

class GPUMonitor:
    def __init__(self, gpu_index=0):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(gpu_index)
        self.gpu_name = nvmlDeviceGetName(self.handle)
        self.gpu_memory = nvmlDeviceGetMemoryInfo(self.handle).total / (1024 ** 3)  # Convert to GB
        self.gpu_compute_capability = nvmlDeviceGetCudaComputeCapability(self.handle)
        self.gpu_uuid = nvmlDeviceGetUUID(self.handle)
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

def evaluate_model(model, processor, test_loader, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor()
    
    # Create output directory inside the model directory
    output_dir = os.path.join(model_dir, 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    all_preds = []
    all_labels = []
    gpu_metrics = []
    batch_times = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
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
            
            outputs = model(inputs).logits
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Record metrics every 10 batches
            if i % 10 == 0:
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                gpu_metrics.append(gpu_monitor.get_metrics())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'gpu_metrics': gpu_metrics,
        'batch_times': batch_times
    }
    
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    # Save GPU metrics to CSV
    df_metrics = pd.DataFrame(gpu_metrics)
    df_metrics['batch_time'] = batch_times
    df_metrics.to_csv(os.path.join(output_dir, 'gpu_performance_metrics.csv'), index=False)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - Brain Tumor MRI Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Print results
    print("\nBrain Tumor MRI Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nResults and plots saved to {output_dir}")
    
    return metrics

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test brain tumor MRI classification model')
    parser.add_argument('--model_dir', type=str, help='Directory containing the fine-tuned model checkpoints')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_epoch_10.pt', 
                      help='Checkpoint file to load (default: checkpoint_epoch_10.pt)')
    args = parser.parse_args()
    
    # Create test dataloader
    test_loader = get_brain_tumor_dataloader(batch_size=32, train=False)
    class_names = test_loader.dataset.classes
    num_classes = len(class_names)
    
    # Find the model directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        # If no directory specified, use the latest one
        output_dirs = [d for d in os.listdir('.') if d.startswith('outputs_finetune_')]
        if not output_dirs:
            print("No fine-tuned model found! Please specify a model directory with --model_dir")
            exit(1)
        model_dir = max(output_dirs, key=os.path.getctime)
        print(f"Using latest model directory: {model_dir}")
    
    checkpoint_path = os.path.join(model_dir, args.checkpoint)
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        exit(1)
    
    # Load model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    # Modify the classification head for brain tumor classification
    model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    metrics = evaluate_model(model, processor, test_loader, class_names) 