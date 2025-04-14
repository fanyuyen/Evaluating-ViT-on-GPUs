# GPU Performance Benchmarking for Vision Transformers

This project is for evaluating GPU performance when running inference with Vision Transformer (ViT) models across different datasets and batch sizes.

## Features

- Supports multiple datasets:
  - CIFAR-10 (10,000 validation images)
  - ImageNet-100 (5,000 validation images)
  - Food101 (25,250 validation images)
- GPU metrics collection:
  - GPU utilization
  - Memory usage
  - Power consumption
- Performance testing across different batch sizes
- CSV output with all metrics
- NVTX profiling support

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
python plot_performance.py --input_dirs outputs_huo_20250407_141715 outputs_jin_20250407_153906 outputs_tian_20250407_141558
python plot_performance.py --input_dirs outputs_huo_20250408_152341/ outputs_jin_20250408_152556/ outputs_shui_20250408_152539/ --hosts huo jin shui
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
