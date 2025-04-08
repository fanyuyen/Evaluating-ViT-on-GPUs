import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import argparse
import pandas as pd

def plot_metrics(metrics_path, output_dir=None):
    """Plot metrics for a single training run"""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create output directory for plots
    if output_dir is None:
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
    
    # Extract epoch-level GPU metrics
    epochs = [m['epoch'] for m in metrics['epoch_metrics']]
    gpu_utils = [m['gpu_metrics']['gpu_util'] for m in metrics['epoch_metrics']]
    gpu_mems = [m['gpu_metrics']['gpu_mem_used']/1024 for m in metrics['epoch_metrics']]  # Convert to GB
    gpu_temps = [m['gpu_metrics']['gpu_temp'] for m in metrics['epoch_metrics']]
    gpu_powers = [m['gpu_metrics']['gpu_power'] for m in metrics['epoch_metrics']]
    epoch_times = [m['time'] for m in metrics['epoch_metrics']]
    batch_times = [m['avg_batch_time'] for m in metrics['epoch_metrics']]
    
    # Plot GPU metrics
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, gpu_utils)
    plt.xlabel('Epoch')
    plt.ylabel('GPU Utilization (%)')
    plt.title('Average GPU Utilization per Epoch')
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, gpu_mems)
    plt.xlabel('Epoch')
    plt.ylabel('GPU Memory Usage (GB)')
    plt.title('Average GPU Memory Usage per Epoch')
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, gpu_temps)
    plt.xlabel('Epoch')
    plt.ylabel('Temperature (°C)')
    plt.title('Average GPU Temperature per Epoch')
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, gpu_powers)
    plt.xlabel('Epoch')
    plt.ylabel('Power Usage (W)')
    plt.title('Average GPU Power Usage per Epoch')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gpu_metrics.png'))
    plt.close()
    
    # Plot timing metrics
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, epoch_times)
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Epoch Processing Time')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, batch_times)
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Average Batch Processing Time')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timing_metrics.png'))
    plt.close()

def plot_comparative_analysis(metrics_files, output_dir=None):
    """Plot comparative analysis of multiple training runs"""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'comparative_analysis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # GPU name mapping (server name -> GPU name)
    gpu_mapping = {
        'huo': 'NVIDIA A100 40GB',
        'jin': 'NVIDIA RTX 4070 Ti Super',
        'tian': 'NVIDIA RTX 2080 Super',
        # Add more mappings as needed
    }
    
    # Load all metrics
    all_metrics = []
    for file in metrics_files:
        with open(file, 'r') as f:
            metrics = json.load(f)
            # Extract server name from the directory
            server_name = os.path.basename(os.path.dirname(file)).split('_')[2]
            # Use mapping or fallback to server name
            gpu_name = gpu_mapping.get(server_name, f"GPU on {server_name}")
            metrics['gpu_name'] = gpu_name
            all_metrics.append(metrics)
    
    # Plot comparative training metrics
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    for metrics in all_metrics:
        plt.plot(metrics['train_loss'], label=f"{metrics['gpu_name']} - Train")
        plt.plot(metrics['val_loss'], '--', label=f"{metrics['gpu_name']} - Val")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Comparative Training and Validation Loss')
    plt.legend()
    
    # Plot training accuracy
    plt.subplot(2, 2, 2)
    for metrics in all_metrics:
        plt.plot(metrics['train_acc'], label=f"{metrics['gpu_name']} - Train")
        plt.plot(metrics['val_acc'], '--', label=f"{metrics['gpu_name']} - Val")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparative Training and Validation Accuracy')
    plt.legend()
    
    # Plot GPU utilization
    plt.subplot(2, 2, 3)
    for metrics in all_metrics:
        gpu_utils = [m['gpu_metrics']['gpu_util'] for m in metrics['epoch_metrics']]
        epochs = [m['epoch'] for m in metrics['epoch_metrics']]
        plt.plot(epochs, gpu_utils, label=metrics['gpu_name'])
    plt.xlabel('Epoch')
    plt.ylabel('GPU Utilization (%)')
    plt.title('Comparative GPU Utilization')
    plt.legend()
    
    # Plot GPU memory usage
    plt.subplot(2, 2, 4)
    for metrics in all_metrics:
        gpu_mems = [m['gpu_metrics']['gpu_mem_used']/1024 for m in metrics['epoch_metrics']]
        epochs = [m['epoch'] for m in metrics['epoch_metrics']]
        plt.plot(epochs, gpu_mems, label=metrics['gpu_name'])
    plt.xlabel('Epoch')
    plt.ylabel('GPU Memory Usage (GB)')
    plt.title('Comparative GPU Memory Usage')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparative_metrics.png'))
    plt.close()
    
    # Plot timing comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for metrics in all_metrics:
        epoch_times = [m['time'] for m in metrics['epoch_metrics']]
        epochs = [m['epoch'] for m in metrics['epoch_metrics']]
        plt.plot(epochs, epoch_times, label=metrics['gpu_name'])
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Comparative Epoch Processing Time')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for metrics in all_metrics:
        batch_times = [m['avg_batch_time'] for m in metrics['epoch_metrics']]
        epochs = [m['epoch'] for m in metrics['epoch_metrics']]
        plt.plot(epochs, batch_times, label=metrics['gpu_name'])
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Comparative Average Batch Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparative_timing.png'))
    plt.close()
    
    # Create a summary CSV
    summary_data = []
    for metrics in all_metrics:
        final_metrics = metrics['epoch_metrics'][-1]  # Get the last epoch's metrics
        
        summary_data.append({
            'GPU': metrics['gpu_name'],
            'Final Train Loss': metrics['train_loss'][-1],
            'Final Val Loss': metrics['val_loss'][-1],
            'Final Train Acc (%)': metrics['train_acc'][-1],
            'Final Val Acc (%)': metrics['val_acc'][-1],
            'Avg Epoch Time (s)': np.mean([m['time'] for m in metrics['epoch_metrics']]),
            'Avg Batch Time (s)': np.mean([m['avg_batch_time'] for m in metrics['epoch_metrics']]),
            'Avg GPU Util (%)': np.mean([m['gpu_metrics']['gpu_util'] for m in metrics['epoch_metrics']]),
            'Avg GPU Mem (GB)': np.mean([m['gpu_metrics']['gpu_mem_used']/1024 for m in metrics['epoch_metrics']]),
            'Avg GPU Temp (°C)': np.mean([m['gpu_metrics']['gpu_temp'] for m in metrics['epoch_metrics']]),
            'Avg GPU Power (W)': np.mean([m['gpu_metrics']['gpu_power'] for m in metrics['epoch_metrics']]),
            'Total Time (hours)': metrics['total_time']/3600
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, 'comparative_summary.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot fine-tuning performance metrics')
    parser.add_argument('--metrics_file', type=str, help='Path to the training metrics JSON file')
    parser.add_argument('--metrics_files', type=str, nargs='+', help='Paths to multiple training metrics JSON files for comparative analysis')
    parser.add_argument('--output_dir', type=str, help='Directory to save the plots (default: auto-generated)')
    args = parser.parse_args()
    
    if args.metrics_files:
        # Run comparative analysis
        plot_comparative_analysis(args.metrics_files, args.output_dir)
        print(f"Comparative analysis saved in '{os.path.abspath(args.output_dir or 'comparative_analysis_[timestamp]')}' directory")
    else:
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
        
        plot_metrics(metrics_path, args.output_dir)
        print(f"Plots saved in {os.path.abspath(args.output_dir or os.path.join(os.path.dirname(metrics_path), 'plots'))}") 