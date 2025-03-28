from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
from dataloader import get_cifar10_dataloader, get_imagenet100_dataloader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import torch
from tqdm import tqdm
import subprocess
from torchvision import transforms

from pynvml import *
nvmlInit()
gpu_index_nvml = 1
handle = nvmlDeviceGetHandleByIndex(gpu_index_nvml)
gpu_name = nvmlDeviceGetName(handle)
print(f"Running on GPU (pynvml): {gpu_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on GPU (pytorch): ", torch.cuda.get_device_name(0))
batch_size = 4

# DataLoader
dataloader = get_imagenet100_dataloader(batch_size, train=False, subset_size=40)

# Model and processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to("cuda")
model.eval()

# Metric tracking
all_preds = []
all_labels = []

gpu_utilizations = []
gpu_memory_usages = []

start_time = time.time()
torch.cuda.reset_peak_memory_stats(device)
inference_start = time.perf_counter()

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Running Inference"):
        images = batch["image"]  # Get images from the batch dictionary
        labels = batch["label"]
                
        # Process each image individually to ensure we get predictions for all images
        batch_preds = []
        for image in images:
            # Ensure image has 3 channels if it doesn't already
            if image.dim() == 2:
                image = image.unsqueeze(0).repeat(3, 1, 1)  # Add channel dimension and repeat to RGB
            
            # Convert tensor to PIL Image
            image_pil = transforms.ToPILImage()(image)
            
            # Process single image
            inputs = processor(images=image_pil, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            pred = logits.argmax(dim=-1).cpu().numpy()
            batch_preds.append(pred[0])
        
        ##
        all_preds.extend(batch_preds)  # Add all predictions from this batch
        all_labels.extend(labels.numpy())
        ##
        # GPU Monitoring
        util = nvmlDeviceGetUtilizationRates(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        ##
        gpu_utilizations.append(util.gpu)  # in %
        gpu_memory_usages.append(mem_info.used / (1024 ** 2))  # in MB

inference_end = time.perf_counter()

# === RESULTS ===
total_time = inference_end - inference_start
num_images = len(dataloader.dataset)
throughput = num_images / total_time
peak_memory_MB = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

# GPU usage
avg_util = sum(gpu_utilizations) / len(gpu_utilizations)
avg_mem = sum(gpu_memory_usages) / len(gpu_memory_usages)

# # === METRICS ===
# acc = accuracy_score(all_labels, all_preds)
# prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
# rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
# f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

# # === PRINT RESULTS ===
print(f"\n--- Results on ImageNet100 ---")
print(f"Batch size: {batch_size}")
print(f"Total images: {num_images}")
print(f"Total inference time: {total_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} images/second")
print(f"Peak VRAM usage: {peak_memory_MB:.2f} MB\n")

print("\n--- GPU Utilization (via pynvml) ---")
print(f"Average GPU Utilization: {avg_util:.2f}%")
print(f"Average GPU Memory Usage: {avg_mem:.2f} MB")

# print(f"Accuracy:  {acc:.4f}")
# print(f"Precision: {prec:.4f}")
# print(f"Recall:    {rec:.4f}")
# print(f"F1 Score:  {f1:.4f}")