import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import random 

import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

from transformers import AutoModelForImageClassification, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR  = "/mnt/extended-home/dzakaaufa/dataset/all_images_captioning" 
CSV_PATH   = "/mnt/extended-home/dzakaaufa/leakage/model100_65/data100.csv" 
OUTPUT_DIR = "/mnt/extended-home/dzakaaufa/leakage/model100_65"

MODEL_NAME = "facebook/dinov2-large"

IMG_SIZE   = 224 
BATCH_SIZE = 16  
ACCUM_ITER = 2   
EPOCHS     = 10
SEED       = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SET REPRODUCIBILITY SEED
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed disetel ke: {seed}")

# ============================================================
# FOCAL LOSS & DATASET
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

class BatikDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = df["image_path"].tolist()
        self.labels = df["LABEL_IDX"].tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_dir, img_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# ============================================================
# TRANSFORM & MODEL
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def build_dinov2_lora_model(num_classes, id2label, label2id):
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True 
    )
    lora_config = LoraConfig(
        r=32, lora_alpha=64, target_modules=["query", "key", "value", "dense"], 
        lora_dropout=0.1, bias="none", modules_to_save=["classifier"] 
    )
    model = get_peft_model(model, lora_config)
    return model

# ============================================================
# TRAIN & EVALUATE FUNCTIONS
# ============================================================
def train_model(model, train_loader, val_loader, class_weights):
    model = model.to(DEVICE)
    criterion = FocalLoss(weight=class_weights, gamma=1.0)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)
    
    total_steps = (len(train_loader) // ACCUM_ITER) * EPOCHS
    warmup_steps = int(0.1 * total_steps) 
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = GradScaler()
    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        total_loss = 0
        optimizer.zero_grad() 

        for i, (imgs, labels) in enumerate(tqdm(train_loader, desc="Training")):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs.logits, labels) / ACCUM_ITER 
            scaler.scale(loss).backward()

            if ((i + 1) % ACCUM_ITER == 0) or (i + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            total_loss += loss.item() * ACCUM_ITER 

        # Validation
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(imgs)
                _, pred = torch.max(outputs.logits, 1)
                preds.extend(pred.cpu().numpy())
                gts.extend(labels.cpu().numpy())

        acc = accuracy_score(gts, preds)
        print(f"Loss: {total_loss / len(train_loader):.4f} | Val Acc: {acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(os.path.join(OUTPUT_DIR, "best_dinov2_lora"))
            print(f"Saved best adapter: {best_acc:.4f}")

    return best_acc

def evaluate_classifier(model, eval_df, image_dir, transform, classes):
    print(f"\n--- Mengambil Matriks Kinerja dari Validation Set ({len(eval_df)} citra) ---")
    model.eval()
    eval_dataset = BatikDataset(eval_df, image_dir, transform)
    eval_loader  = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_preds, all_gts = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(eval_loader, desc="Evaluating"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(imgs)
            _, pred = torch.max(outputs.logits, 1)
            all_preds.extend(pred.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())

    acc = accuracy_score(all_gts, all_preds)
    print(f"Final Validation Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_gts, all_preds, target_names=classes, zero_division=0))
    return all_gts, all_preds

# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED) 

    print(f"[*] Membaca dataset dari: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH) 

    # 1. Pemetaan Label & Kelas
    classes = sorted(df["class"].unique().tolist())
    class_to_idx = {c:i for i,c in enumerate(classes)}
    idx_to_class = {i:c for c,i in class_to_idx.items()}
    df["LABEL_IDX"] = df["class"].map(class_to_idx)
    
    # 2. Distribusi Data Menggunakan Kolom 'new_split'
    train_df = df[df["new_split"] == "train"].reset_index(drop=True)
    val_df   = df[df["new_split"] == "val"].reset_index(drop=True)
    test_df  = df[df["new_split"] == "test"].reset_index(drop=True)
    unused_df = df[df["new_split"] == "unused"]

    print(f"\n--- RINGKASAN DATASET ---")
    print(f"[*] Data Train       : {len(train_df)}")
    print(f"[*] Data Validation  : {len(val_df)}")
    print(f"[*] Data Test (Hold) : {len(test_df)} (TIDAK digunakan untuk DINOv2)")
    print(f"[*] Data Unused      : {len(unused_df)} (Dibuang karena reduksi AI/Dataset B)\n")

    # 3. Hitung Class Weights berdasarkan Train Set saja
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["LABEL_IDX"]),
        y=train_df["LABEL_IDX"] 
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    # 4. Inisialisasi Dataset & DataLoader
    train_dataset = BatikDataset(train_df, IMAGE_DIR, train_transform)
    val_dataset   = BatikDataset(val_df, IMAGE_DIR, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 5. Build Model
    model = build_dinov2_lora_model(len(classes), idx_to_class, class_to_idx)

    # 6. Proses Pelatihan Utama
    best_acc = train_model(model, train_loader, val_loader, class_weights)
    
    # 7. Pengujian Akhir & Pembuatan Matriks pada Data Validasi (20% Pool)
    # Memastikan file yang diload kembali adalah best weight
    print("\nMemuat bobot terbaik untuk evaluasi akhir...")
    
    # 1. Muat base model-nya terlebih dahulu
    base_model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(classes), 
        id2label=idx_to_class, 
        label2id=class_to_idx, 
        ignore_mismatched_sizes=True 
    )
    
    # 2. Bungkus dengan PeftModel langsung dari folder penyimpanan (otomatis mendeteksi safetensors)
    model = PeftModel.from_pretrained(base_model, os.path.join(OUTPUT_DIR, "best_dinov2_lora"))
    model = model.to(DEVICE)

    # Jalankan evaluasi
    y_test_true, y_test_pred = evaluate_classifier(model, val_df, IMAGE_DIR, val_transform, classes)

    # Plot Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_test_true, y_test_pred, labels=list(range(len(classes))))

    fig_size = max(10, len(classes) * 0.8)
    plt.figure(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix (Validation Set)', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    print(f"Confusion Matrix berhasil disimpan ke: {cm_path}")
    print("\nEKSPERIMEN DINOV2 SELESAI.")

if __name__ == "__main__":
    main()