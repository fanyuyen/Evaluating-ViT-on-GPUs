import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create directories for organizing outputs
def create_directories():
    """Create directories for organizing different types of outputs"""
    dirs = {
        'data': 'outputs/data',
        'plots': 'outputs/plots',
        'stats': 'outputs/stats'
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

# Set style for better visualizations
plt.style.use('default')
sns.set_theme()

def plot_dataset_performance(df, dataset_name, output_dirs):
    """Generate and save performance analysis plots for a single dataset"""
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)

    # 1. Throughput vs Batch Size
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['batch_size'], df['throughput'], marker='o', linewidth=2, color='blue')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (images/second)')
    ax1.set_title(f'Throughput vs Batch Size ({dataset_name})')
    ax1.grid(True)

    # 2. Peak Memory Usage vs Batch Size
    ax2 = fig.add_subplot(gs[0, 1])
    peak_line = ax2.plot(df['batch_size'], df['peak_memory_MB'], marker='o', linewidth=2, color='red', label='Peak Memory')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Peak Memory Usage (MB)')
    ax2.set_title(f'Peak GPU Memory Usage vs Batch Size ({dataset_name})')
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
    ax3.plot(df['batch_size'], df['p95_latency']*1000, marker='s', linewidth=2, label='P95')
    ax3.plot(df['batch_size'], df['p99_latency']*1000, marker='^', linewidth=2, label='P99')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title(f'Latency Metrics vs Batch Size ({dataset_name})')
    ax3.legend()
    ax3.grid(True)

    # 4. GPU Utilization and Temperature vs Batch Size
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_twin = ax4.twinx()
    ax4.plot(df['batch_size'], df['gpu_util'], marker='o', linewidth=2, color='blue', label='GPU Utilization')
    ax4_twin.plot(df['batch_size'], df['gpu_temp'], marker='s', linewidth=2, color='red', label='Temperature')
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('GPU Utilization (%)', color='blue')
    ax4_twin.set_ylabel('Temperature (°C)', color='red')
    ax4.set_title(f'GPU Utilization and Temperature vs Batch Size ({dataset_name})')
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2)
    ax4.grid(True)

    # 5. Power Usage vs Batch Size
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(df['batch_size'], df['gpu_power'], marker='o', linewidth=2, color='green')
    ax5.set_xlabel('Batch Size')
    ax5.set_ylabel('Power Usage (W)')
    ax5.set_title(f'GPU Power Usage vs Batch Size ({dataset_name})')
    ax5.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dirs['plots'], f'performance_analysis_{dataset_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparative_analysis(df_all, output_dirs):
    """Generate and save comparative analysis plots across datasets"""
    datasets = df_all['dataset'].unique()
    
    # Create comparative plots
    metrics = [
        ('throughput', 'Throughput (images/second)'),
        ('peak_memory_MB', 'Peak Memory Usage (MB)'),
        ('avg_latency', 'Average Latency (ms)'),
        ('gpu_util', 'GPU Utilization (%)'),
        ('gpu_power', 'Power Usage (W)')
    ]
    
    for metric, ylabel in metrics:
        plt.figure(figsize=(12, 8))
        for dataset in datasets:
            df_dataset = df_all[df_all['dataset'] == dataset]
            plt.plot(df_dataset['batch_size'], df_dataset[metric], 
                    marker='o', linewidth=2, label=dataset)
        
        plt.xlabel('Batch Size')
        plt.ylabel(ylabel)
        plt.title(f'Comparative {ylabel} Across Datasets')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(output_dirs['plots'], f'comparative_{metric}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

def generate_statistics(df_all, output_dirs):
    """Generate and save comparative statistics"""
    datasets = df_all['dataset'].unique()
    stats_list = []
    
    for dataset in datasets:
        df_dataset = df_all[df_all['dataset'] == dataset]
        stats = {
            'Dataset': dataset,
            'Max Throughput': df_dataset['throughput'].max(),
            'Min Latency (ms)': df_dataset['avg_latency'].min() * 1000,
            'Avg GPU Util (%)': df_dataset['gpu_util'].mean(),
            'Max Memory (MB)': df_dataset['peak_memory_MB'].max(),
            'Avg Power (W)': df_dataset['gpu_power'].mean()
        }
        stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(os.path.join(output_dirs['stats'], 'comparative_statistics.csv'), index=False)
    return stats_df

def main():
    output_dirs = create_directories()
    datasets = ['cifar10', 'imagenet100', 'food101']
    all_results = []
    
    # Process each dataset
    for dataset in datasets:
        input_file = os.path.join(output_dirs['data'], f'gpu_performance_results_{dataset}.csv')
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            plot_dataset_performance(df, dataset, output_dirs)
            all_results.append(df)
            print(f"Generated plots for {dataset}")
    
    # Combine all results and create comparative analysis
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        plot_comparative_analysis(df_all, output_dirs)
        stats_df = generate_statistics(df_all, output_dirs)
        
        print("\nComparative Statistics:")
        print(stats_df.to_string(index=False))
    
    print("\nAll plots and statistics have been saved in the outputs directory")

if __name__ == "__main__":
    main() 