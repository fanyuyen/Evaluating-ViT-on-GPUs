import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse

# Create directories for organizing outputs
def create_directories():
    """Create directories for organizing different types of outputs"""
    # Create a results directory for the analysis
    results_dir = 'analysis_results'
    
    dirs = {
        'plots': os.path.join(results_dir, 'plots'),
        'stats': os.path.join(results_dir, 'stats')
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

# Set style for better visualizations
plt.style.use('default')

def plot_dataset_performance(df, dataset_name, gpu_name, output_dirs):
    """Generate and save performance analysis plots for a single dataset"""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2)

    # 1. Throughput vs Batch Size
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['batch_size'], df['throughput'], marker='o', linewidth=2, color='blue')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (images/second)')
    ax1.set_title(f'Throughput vs Batch Size ({dataset_name})\nGPU: {gpu_name}')
    ax1.grid(True)

    # 2. Peak Memory Usage vs Batch Size
    ax2 = fig.add_subplot(gs[0, 1])
    peak_line = ax2.plot(df['batch_size'], df['peak_memory_MB'], marker='o', linewidth=2, color='red', label='Peak Memory')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Peak Memory Usage (MB)')
    ax2.set_title(f'Peak GPU Memory Usage vs Batch Size ({dataset_name})\nGPU: {gpu_name}')
    if not df['peak_memory_MB'].empty:
        max_peak = df['peak_memory_MB'].max()
        min_peak = df['peak_memory_MB'].min()
        ax2.annotate(f'Max: {max_peak:.1f} MB', 
                    xy=(df.loc[df['peak_memory_MB'].idxmax(), 'batch_size'], max_peak),
                    xytext=(10, 10), textcoords='offset points')
        ax2.annotate(f'Min: {min_peak:.1f} MB',
                    xy=(df.loc[df['peak_memory_MB'].idxmin(), 'batch_size'], min_peak),
                    xytext=(10, -15), textcoords='offset points')
    ax2.grid(True)

    # 3. Latency Metrics vs Batch Size
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['batch_size'], df['avg_latency']*1000, marker='o', linewidth=2, label='Average')
    if 'p95_latency' in df.columns and 'p99_latency' in df.columns:
        ax3.plot(df['batch_size'], df['p95_latency']*1000, marker='s', linewidth=2, label='P95')
        ax3.plot(df['batch_size'], df['p99_latency']*1000, marker='^', linewidth=2, label='P99')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title(f'Latency Metrics vs Batch Size ({dataset_name})\nGPU: {gpu_name}')
    ax3.legend()
    ax3.grid(True)

    # 4. GPU Utilization and Temperature vs Batch Size
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_twin = ax4.twinx()
    ax4.plot(df['batch_size'], df['gpu_util'], marker='o', linewidth=2, color='blue', label='GPU Utilization')
    if 'gpu_temp' in df.columns:
        ax4_twin.plot(df['batch_size'], df['gpu_temp'], marker='s', linewidth=2, color='red', label='Temperature')
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('GPU Utilization (%)', color='blue')
    ax4_twin.set_ylabel('Temperature (°C)', color='red')
    ax4.set_title(f'GPU Utilization and Temperature vs Batch Size ({dataset_name})\nGPU: {gpu_name}')
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2)
    ax4.grid(True)

    # 5. Power Usage vs Batch Size
    ax5 = fig.add_subplot(gs[2, :])
    if 'gpu_power' in df.columns:
        ax5.plot(df['batch_size'], df['gpu_power'], marker='o', linewidth=2, color='green')
        ax5.set_xlabel('Batch Size')
        ax5.set_ylabel('Power Usage (W)')
        ax5.set_title(f'GPU Power Usage vs Batch Size ({dataset_name})\nGPU: {gpu_name}')
        ax5.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dirs['plots'], f'performance_analysis_{dataset_name}_{gpu_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated plots for {dataset_name} on {gpu_name}")

def plot_comparative_analysis(df_all, output_dirs, hosts=None):
    """Generate and save comparative analysis plots across datasets"""
    datasets = df_all['dataset'].unique()
    gpus = df_all['gpu_name'].unique()
    
    # Print available GPU names for debugging
    print("\nAvailable GPUs in data:")
    for gpu in gpus:
        print(f"- {gpu}")
    
    # Filter by hosts if specified
    if hosts:
        df_all = df_all[df_all['hostname'].isin(hosts)]
    
    # Create comparative plots - first by dataset
    metrics = [
        ('throughput', 'Throughput (images/second)'),
        ('peak_memory_MB', 'Peak Memory Usage (MB)'),
        ('avg_latency', 'Average Latency (ms)'),
        ('gpu_util', 'GPU Utilization (%)'),
        ('gpu_power', 'Power Usage (W)')
    ]
    
    # Define colors and markers for GPUs
    gpu_colors = {
        'A100': 'blue',
        '4070': 'green',
        '2080': 'red'
    }
    
    markers = ['o', 's', '^', 'D', 'p']
    
    # Plot metrics with all GPUs in the same figure
    for metric, ylabel in metrics:
        if metric not in df_all.columns:
            continue
            
        plt.figure(figsize=(10, 5))
        
        for i, gpu in enumerate(gpus):
            for j, dataset in enumerate(datasets):
                df_subset = df_all[(df_all['dataset'] == dataset) & (df_all['gpu_name'] == gpu)]
                if not df_subset.empty:
                    # Find matching color based on GPU name
                    color = 'gray'  # Default color
                    for gpu_key, gpu_color in gpu_colors.items():
                        if gpu_key in gpu:
                            color = gpu_color
                            break
                    
                    print(f"Using color {color} for GPU: {gpu}")  # Debug print
                    
                    if metric == 'avg_latency':
                        # Convert to milliseconds for latency
                        plt.plot(df_subset['batch_size'], df_subset[metric]*1000,
                                marker=markers[j], linestyle='-', color=color,
                                linewidth=2, label=f'{gpu} - {dataset}')
                    else:
                        plt.plot(df_subset['batch_size'], df_subset[metric],
                                marker=markers[j], linestyle='-', color=color,
                                linewidth=2, label=f'{gpu} - {dataset}')
        
        plt.xlabel('Batch Size')
        plt.ylabel(ylabel)
        plt.title(f'Comparative {ylabel} Across Datasets and GPUs')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dirs['plots'], f'comparative_{metric}_combined.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

def generate_statistics(df_all, output_dirs, hosts=None):
    """Generate and save comparative statistics"""
    datasets = df_all['dataset'].unique()
    gpus = df_all['gpu_name'].unique()
    
    # Filter by hosts if specified
    if hosts:
        df_all = df_all[df_all['hostname'].isin(hosts)]
    
    stats_list = []
    
    for dataset in datasets:
        for gpu in gpus:
            df_subset = df_all[(df_all['dataset'] == dataset) & (df_all['gpu_name'] == gpu)]
            if not df_subset.empty:
                stats = {
                    'Dataset': dataset,
                    'GPU': gpu,
                    'Max Throughput': df_subset['throughput'].max(),
                    'Min Latency (ms)': df_subset['avg_latency'].min() * 1000,
                    'Avg GPU Util (%)': df_subset['gpu_util'].mean(),
                    'Max Memory (MB)': df_subset['peak_memory_MB'].max(),
                }
                if 'gpu_power' in df_subset.columns:
                    stats['Avg Power (W)'] = df_subset['gpu_power'].mean()
                
                stats_list.append(stats)
    
    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        # Remove any duplicate rows
        stats_df = stats_df.drop_duplicates()
        stats_path = os.path.join(output_dirs['stats'], 'comparative_statistics.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"\nStatistics saved to {stats_path}")
        return stats_df
    else:
        return pd.DataFrame()

def load_results(input_dirs):
    """Load results from multiple input directories"""
    all_results = []
    
    for input_dir in input_dirs:
        print(f"Processing results from {input_dir}")
        
        # Find dataset-specific files
        dataset_files = glob.glob(os.path.join(input_dir, 'data', 'gpu_performance_results_*_*.csv'))
        for file_path in dataset_files:
            if 'all' not in file_path:  # Skip the combined files
                try:
                    df = pd.read_csv(file_path)
                    # Add hostname if not present
                    if 'hostname' not in df.columns:
                        hostname = os.path.basename(input_dir).split('_')[1]
                        df['hostname'] = hostname
                    all_results.append(df)
                    print(f"Loaded {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
    
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        # Remove any duplicate rows
        df_all = df_all.drop_duplicates()
        return df_all
    else:
        print("No results found!")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='Generate performance analysis plots')
    parser.add_argument('--input_dirs', type=str, nargs='+', required=True, 
                      help='Input directories containing the results (specify three directories)')
    parser.add_argument('--hosts', type=str, nargs='+', default=['huo', 'jin', 'tian'], 
                      help='Hosts to analyze (default: huo jin tian)')
    args = parser.parse_args()
    
    if len(args.input_dirs) != 3:
        print("Error: Please specify exactly three input directories")
        return
    
    output_dirs = create_directories()
    
    # Load results from all specified input directories
    df_all = load_results(args.input_dirs)
    
    if df_all.empty:
        print("No results were loaded. Exiting.")
        return
    
    # Process each dataset and GPU combination
    datasets = df_all['dataset'].unique()
    gpus = df_all['gpu_name'].unique() if 'gpu_name' in df_all.columns else ['unknown_gpu']
    
    for dataset in datasets:
        for gpu in gpus:
            if 'gpu_name' in df_all.columns:
                df_subset = df_all[(df_all['dataset'] == dataset) & (df_all['gpu_name'] == gpu)]
            else:
                df_subset = df_all[df_all['dataset'] == dataset]
                
            if not df_subset.empty:
                plot_dataset_performance(df_subset, dataset, gpu, output_dirs)
    
    # Create comparative analysis
    if not df_all.empty:
        plot_comparative_analysis(df_all, output_dirs, hosts=args.hosts)
        stats_df = generate_statistics(df_all, output_dirs, hosts=args.hosts)
        
        if not stats_df.empty:
            print("\nComparative Statistics:")
            print(stats_df.to_string(index=False))
    
    print("\nAll plots and statistics have been saved in the analysis_results directory")

if __name__ == "__main__":
    main() 