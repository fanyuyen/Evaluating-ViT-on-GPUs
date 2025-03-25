from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
from dataloader import get_cifar10_dataloader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import torch
from tqdm import tqdm
import subprocess

def log_nvidia_smi():
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using: ", device)
batch_size = 128

# DataLoader
dataloader = get_cifar10_dataloader(batch_size, train=False)

# Model and processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to("cuda")
model.eval()

# Metric tracking
all_preds = []
all_labels = []

start_time = time.time()
torch.cuda.reset_peak_memory_stats(device)
inference_start = time.perf_counter()

with torch.no_grad():
    for images, labels in tqdm(dataloader, desc="Running Inference"):
        inputs = processor(images=list(images), return_tensors="pt", do_rescale=False).to(device) # do_rescale=False, since images already scaled
        outputs = model(**inputs)
        logits = outputs.logits
        preds = logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

inference_end = time.perf_counter()

# === RESULTS ===
total_time = inference_end - inference_start
num_images = len(dataloader.dataset)
throughput = num_images / total_time
peak_memory_MB = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

# # === METRICS ===
# acc = accuracy_score(all_labels, all_preds)
# prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
# rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
# f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

# # === PRINT RESULTS ===
print(f"\n--- Benchmark Results on CIFAR-10 ---")
print(f"Batch size: {batch_size}")
print(f"Total images: {num_images}")
print(f"Total inference time: {total_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} images/second")
print(f"Peak VRAM usage: {peak_memory_MB:.2f} MB\n")
log_nvidia_smi()


# print(f"Accuracy:  {acc:.4f}")
# print(f"Precision: {prec:.4f}")
# print(f"Recall:    {rec:.4f}")
# print(f"F1 Score:  {f1:.4f}")