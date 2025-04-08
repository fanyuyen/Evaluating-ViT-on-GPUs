import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import argparse

def plot_metrics(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create output directory for plots
    output_dir = os.path.join(os.path.dirname(metrics_path), 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training and validation metrics
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Training Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Training Accuracy')
    plt.plot(metrics['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()
    
    # Plot GPU metrics
    gpu_metrics = metrics['gpu_metrics']
    if gpu_metrics:
        plt.figure(figsize=(15, 10))
        
        # Extract GPU metrics
        gpu_util = [m['gpu_util'] for m in gpu_metrics]
        gpu_mem_used = [m['gpu_mem_used'] for m in gpu_metrics]
        gpu_mem_total = gpu_metrics[0]['gpu_mem_total']  # Total memory is constant
        gpu_temp = [m['gpu_temp'] for m in gpu_metrics]
        gpu_power = [m['gpu_power'] for m in gpu_metrics]
        
        # Plot GPU Utilization
        plt.subplot(2, 2, 1)
        plt.plot(gpu_util)
        plt.xlabel('Batch (every 10)')
        plt.ylabel('GPU Utilization (%)')
        plt.title('GPU Utilization')
        
        # Plot GPU Memory Usage
        plt.subplot(2, 2, 2)
        plt.plot(gpu_mem_used)
        plt.axhline(y=gpu_mem_total, color='r', linestyle='--', label='Total Memory')
        plt.xlabel('Batch (every 10)')
        plt.ylabel('GPU Memory Usage (MB)')
        plt.title('GPU Memory Usage')
        plt.legend()
        
        # Plot GPU Temperature
        plt.subplot(2, 2, 3)
        plt.plot(gpu_temp)
        plt.xlabel('Batch (every 10)')
        plt.ylabel('Temperature (°C)')
        plt.title('GPU Temperature')
        
        # Plot GPU Power Usage
        plt.subplot(2, 2, 4)
        plt.plot(gpu_power)
        plt.xlabel('Batch (every 10)')
        plt.ylabel('Power Usage (W)')
        plt.title('GPU Power Usage')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gpu_metrics.png'))
        plt.close()
    
    # Plot batch times
    plt.figure(figsize=(8, 4))
    plt.plot(metrics['batch_times'])
    plt.xlabel('Batch (every 10)')
    plt.ylabel('Time (seconds)')
    plt.title('Batch Processing Time')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_times.png'))
    plt.close()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot fine-tuning performance metrics')
    parser.add_argument('--metrics_file', type=str, help='Path to the training metrics JSON file')
    args = parser.parse_args()
    
    # Find the metrics file
    if args.metrics_file:
        metrics_path = args.metrics_file
    else:
        # If no file specified, use the latest one
        output_dirs = [d for d in os.listdir('.') if d.startswith('outputs_finetune_')]
        if not output_dirs:
            print("No training output directories found! Please specify a metrics file with --metrics_file")
            exit(1)
        latest_dir = max(output_dirs, key=os.path.getctime)
        metrics_path = os.path.join(latest_dir, 'training_metrics.json')
        print(f"Using metrics from latest directory: {latest_dir}")
    
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        exit(1)
    
    plot_metrics(metrics_path)
    print(f"Plots saved in {os.path.join(os.path.dirname(metrics_path), 'plots')}") 