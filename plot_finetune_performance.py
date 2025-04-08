import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import argparse
import pandas as pd

def plot_metrics(metrics_path, output_dir=None):
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

def plot_comparative_analysis(metrics_files, output_dir=None):
    """Plot comparative analysis of multiple training runs"""
    if output_dir is None:
        # Create a new directory with timestamp for comparative analysis
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
            server_name = os.path.basename(os.path.dirname(file)).split('_')[2]  # Get server name from directory
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
        gpu_util = [m['gpu_util'] for m in metrics['gpu_metrics']]
        plt.plot(gpu_util, label=metrics['gpu_name'])
    plt.xlabel('Batch (every 10)')
    plt.ylabel('GPU Utilization (%)')
    plt.title('Comparative GPU Utilization')
    plt.legend()
    
    # Plot GPU memory usage
    plt.subplot(2, 2, 4)
    for metrics in all_metrics:
        gpu_mem = [m['gpu_mem_used'] for m in metrics['gpu_metrics']]
        plt.plot(gpu_mem, label=metrics['gpu_name'])
    plt.xlabel('Batch (every 10)')
    plt.ylabel('GPU Memory Usage (MB)')
    plt.title('Comparative GPU Memory Usage')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparative_metrics.png'))
    plt.close()
    
    # Create a summary CSV
    summary_data = []
    for metrics in all_metrics:
        # Calculate average GPU metrics
        gpu_metrics = metrics['gpu_metrics']
        avg_gpu_util = np.mean([m['gpu_util'] for m in gpu_metrics])
        avg_gpu_mem = np.mean([m['gpu_mem_used'] for m in gpu_metrics])
        avg_gpu_temp = np.mean([m['gpu_temp'] for m in gpu_metrics])
        avg_gpu_power = np.mean([m['gpu_power'] for m in gpu_metrics])
        
        # Get final training metrics
        final_train_loss = metrics['train_loss'][-1]
        final_val_loss = metrics['val_loss'][-1]
        final_train_acc = metrics['train_acc'][-1]
        final_val_acc = metrics['val_acc'][-1]
        
        summary_data.append({
            'GPU': metrics['gpu_name'],
            'Final Train Loss': final_train_loss,
            'Final Val Loss': final_val_loss,
            'Final Train Acc (%)': final_train_acc,
            'Final Val Acc (%)': final_val_acc,
            'Avg GPU Util (%)': avg_gpu_util,
            'Avg GPU Mem (MB)': avg_gpu_mem,
            'Avg GPU Temp (°C)': avg_gpu_temp,
            'Avg GPU Power (W)': avg_gpu_power
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, 'comparative_summary.csv'), index=False)
    
    # Also save individual plots for each run in the same directory
    for metrics in all_metrics:
        run_dir = os.path.join(output_dir, metrics['gpu_name'].replace(' ', '_'))
        os.makedirs(run_dir, exist_ok=True)
        plot_metrics(metrics_files[all_metrics.index(metrics)], run_dir)

if __name__ == '__main__':
    # Parse command line arguments
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