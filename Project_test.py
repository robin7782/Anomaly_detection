!pip install --quiet kagglehub
!pip install --quiet open_clip_torch
!pip install --quiet torch torchvision matplotlib scikit-learn
import kagglehub
import os

# Download dataset
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
normal_dir = os.path.join(path, "no")
tumor_dir = os.path.join(path, "yes")

print("✅ Dataset downloaded.")
print("Normal images:", len(os.listdir(normal_dir)))
print("Tumor images:", len(os.listdir(tumor_dir)))
import open_clip
import torch
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().flatten()
  normal_features = []
for f in os.listdir(normal_dir):
    if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    try:
        feat = encode_image(os.path.join(normal_dir, f))
        if not np.isnan(feat).any():
            normal_features.append(feat)
    except Exception as e:
        print("Skipped:", f, "Error:", e)

if not normal_features:
    raise ValueError("No normal features extracted.")

ref_vector = np.mean(normal_features, axis=0).reshape(1, -1)
print("✅ Reference vector created.")
test_features = []
labels = []

for cls_dir, label in [(normal_dir, 0), (tumor_dir, 1)]:
    for f in os.listdir(cls_dir):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        try:
            feat = encode_image(os.path.join(cls_dir, f))
            if not np.isnan(feat).any():
                test_features.append(feat)
                labels.append(label)
        except Exception as e:
            print("Skipped:", f, "Error:", e)

if not test_features:
    raise ValueError("No test features extracted.")

test_features = np.array(test_features).reshape(len(test_features), -1)
similarities = cosine_similarity(test_features, ref_vector).flatten()
distances = 1 - similarities
# Plot distances
plt.figure(figsize=(10, 5))
plt.hist([distances[i] for i in range(len(distances)) if labels[i] == 0], bins=30, alpha=0.5, label='Normal')
plt.hist([distances[i] for i in range(len(distances)) if labels[i] == 1], bins=30, alpha=0.5, label='Tumor')
plt.title("Cosine Distance to Normal Reference Vector")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
