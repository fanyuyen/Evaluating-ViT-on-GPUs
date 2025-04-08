# GPU Performance Benchmarking for Vision Transformers

This project provides a comprehensive benchmarking tool for evaluating GPU performance when running inference with Vision Transformer (ViT) models across different datasets and batch sizes.

## Features

- Supports multiple datasets:
  - CIFAR-10 (10,000 validation images)
  - ImageNet-100 (5,000 validation images)
  - Food101 (25,250 validation images)
- Comprehensive GPU metrics collection:
  - GPU utilization
  - Memory usage
  - Temperature
  - Power consumption
  - Latency statistics
- Automated performance testing across different batch sizes
- Detailed CSV output with all metrics
- NVTX profiling support for detailed performance analysis

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- datasets (Hugging Face)
- torchvision
- pynvml (for GPU monitoring)
- nvtx (for profiling)
- pandas
- tqdm

## Installation

1. Clone the repository:
```bash
git clone git@github.com:fanyuyen/Evaluating-ViT-on-GPUs.git
cd Evaluating-ViT-on-GPUs
```

2. Install the required packages:
```bash
pip install torch torchvision transformers datasets pynvml nvtx pandas tqdm
```

## Usage

### Running the Benchmark

To run the benchmark with full dataset sizes:

```bash
python run_inference.py
```

The script will:
1. Create output directories with timestamps
2. Run inference on each dataset with different batch sizes (1, 8, 16, 32, 64, 128)
3. Collect comprehensive GPU metrics
4. Save results in CSV format

### Running the Benchmark with Nsight system

```bash
nsys profile --trace=cuda,nvtx,cudnn,osrt --stats=true --output=<device_name> python run_inference.py
```
e.g.
```bash
nsys profile --trace=cuda,nvtx,cudnn,osrt --stats=true --output=huo python run_inference.py
```


### Analyzing Results with Plots

To generate performance analysis plots and statistics:

```bash
python plot_performance.py --input_dirs <output_dir1> <output_dir2> ...
```

For example, to compare results between three machines:
```bash
python plot_performance.py --input_dir outputs_huo_20250407_141715 outputs_jin_20250407_153906 outputs_tian_20250407_141558
```

The script will:
1. Load results from all specified directories
2. Generate individual performance plots for each GPU
3. Create comparative plots showing GPU performance across different machines
4. Generate statistics comparing all GPUs
5. Save outputs in the `analysis_results` directory:
   - Individual plots in `analysis_results/plots/`
   - Statistics in `analysis_results/stats/comparative_statistics.csv`

### Output Structure

The script creates a directory structure:
```
outputs_[hostname]_[timestamp]/
└──  data/           # CSV files with benchmark results
```

### Results Format

The CSV output includes:
- Dataset information
- Batch size
- Total inference time
- Throughput (images/second)
- Peak VRAM usage
- GPU utilization
- GPU temperature
- Latency statistics (average, p95, p99)
- GPU information (name, memory, compute capability)

## GPU Requirements

- NVIDIA GPU with CUDA support
- Sufficient VRAM for the model and batch sizes
- NVML (NVIDIA Management Library) support

## Notes

- The script uses the ViT-Base model (patch16-224) from Google
- Images are resized to 224x224 before processing
- Results are saved with hostname and timestamp for easy tracking
- The script includes error handling and will continue testing even if some configurations fail

## Future Improvements

- Add visualization of results
- Support for more models and datasets
- Additional performance metrics
- Automated report generation
- Support for distributed testing

# Fine tune on MRI dataset
## To start fine-tune
```bash
python run_finetune.py
```

## Testing and Performance Analysis

### Testing the Model
To test a fine-tuned model on the brain tumor MRI dataset:

```bash
# Test with default settings (uses latest model and epoch 10 checkpoint)
python test_brain_tumor_mri.py

# Test a specific model directory and checkpoint
python test_brain_tumor_mri.py --model_dir outputs_finetune_huo_20250407_215953 --checkpoint checkpoint_epoch_20.pt
```

The script will:
- Load the specified model checkpoint
- Run inference on the test dataset
- Calculate accuracy, precision, recall, and F1 score
- Generate a confusion matrix
- Save all metrics and plots to an output directory
- Monitor and record GPU performance metrics

Output files are saved in a `test_results` directory inside the model directory:
- `test_metrics.json`: Contains accuracy, precision, recall, F1 score, and GPU metrics
- `gpu_performance_metrics.csv`: Detailed GPU performance data during testing
- `confusion_matrix.png`: Visualization of the classification results

For example, if testing a model in `outputs_finetune_huo_20250407_215953`, the results will be saved in:
```
outputs_finetune_huo_20250407_215953/
├── checkpoint_epoch_20.pt
├── training_metrics.json
└── test_results/
    ├── test_metrics.json
    ├── gpu_performance_metrics.csv
    └── confusion_matrix.png
```

### Plotting Training Performance
To visualize the training performance metrics:

```bash
# Plot metrics from the latest training run
python plot_finetune_performance.py

# Plot metrics from a specific file
python plot_finetune_performance.py --metrics_file outputs_finetune_huo_20250407_215953/training_metrics.json
```

The script will generate plots showing:
- Training and validation loss over epochs
- Training and validation accuracy over epochs
- GPU utilization during training
- GPU memory usage
- GPU temperature
- GPU power consumption
- Batch processing times

All plots are saved in a 'plots' subdirectory of the metrics file location:
- `plots/training_metrics.png`: Training and validation metrics
- `plots/gpu_metrics.png`: GPU performance metrics
- `plots/batch_times.png`: Batch processing times
