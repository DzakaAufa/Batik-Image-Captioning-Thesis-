import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # PERBAIKAN: Tambah confusion_matrix
import numpy as np

# PERBAIKAN: Tambah library visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda.amp import GradScaler, autocast

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR  = "/mnt/extended-home/dzakaaufa/dataset/all_images_captioning"
CSV_PATH   = "/mnt/extended-home/dzakaaufa/dataset/caption/train_data_caption.csv"
OUTPUT_DIR = "/mnt/extended-home/dzakaaufa/models/vit"

IMG_SIZE   = 384
BATCH_SIZE = 16   # VRAM Aman
ACCUM_ITER = 2   # Effective Batch Size = 32
EPOCHS     = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DATASET
# ============================================================
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

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="black")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# ============================================================
# TRANSFORM (NATURAL / RAMAH BATIK)
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((400, 400)),       # Resize sedikit lebih besar
    transforms.RandomCrop(384),          # Crop natural tanpa distorsi skala agresif
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ============================================================
# MODEL (ViT)
# ============================================================
def build_model(num_classes):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    
    model.image_size = IMG_SIZE 
    
    old_pos_embed = model.state_dict()['encoder.pos_embedding']
    
    class_token_embed = old_pos_embed[:, :1, :]
    patch_embeds = old_pos_embed[:, 1:, :]
    
    old_grid_size = int(np.sqrt(patch_embeds.shape[1])) 
    patch_embeds = patch_embeds.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
    
    new_grid_size = IMG_SIZE // 16 
    patch_embeds = F.interpolate(patch_embeds, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
    
    patch_embeds = patch_embeds.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, -1)
    new_pos_embed = torch.cat((class_token_embed, patch_embeds), dim=1)
    
    model.encoder.pos_embedding = nn.Parameter(new_pos_embed)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    return model

# ============================================================
# TRAIN FUNCTION
# ============================================================
def train_model(model, train_loader, val_loader, num_classes, class_weights):
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5) 
    
    scaler = GradScaler()
    best_acc = 0
    
    # PERBAIKAN: Menyimpan prediksi terbaik
    best_gts = []
    best_preds = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        if epoch == 5:
            print("🔥 Partial Unfreezing (2 Blok Terakhir & Layer Norm)")
            for name, param in model.named_parameters():
                if "heads" in name or "encoder.ln" in name or "encoder_layer_11" in name or "encoder_layer_10" in name:
                    param.requires_grad = True
            
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-5)

        # ================= TRAIN =================
        model.train()
        total_loss = 0
        optimizer.zero_grad() 

        for i, (imgs, labels) in enumerate(tqdm(train_loader)):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels) 
                loss = loss / ACCUM_ITER

            scaler.scale(loss).backward()

            if ((i + 1) % ACCUM_ITER == 0) or (i + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUM_ITER 

        scheduler.step()

        # ================= VALIDATION =================
        model.eval()
        preds, gts = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(imgs)
                _, pred = torch.max(outputs, 1)

                preds.extend(pred.cpu().numpy())
                gts.extend(labels.cpu().numpy())

        acc = accuracy_score(gts, preds)
        print(f"Loss: {total_loss:.4f} | Val Acc: {acc:.4f}")

        # PERBAIKAN: Simpan gts dan preds jika akurasi adalah yang tertinggi
        if acc > best_acc:
            best_acc = acc
            best_gts = gts.copy()
            best_preds = preds.copy()
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_vit.pth"))
            print(f"Saved best model: {best_acc:.4f}")

    return best_acc, best_gts, best_preds

def evaluate_model_with_tta(model, val_loader):
    model.eval()
    preds, gts = [] , []
    
    print("🚀 Running Evaluation with TTA (Orig + H-Flip + V-Flip)...")
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):
            imgs = imgs.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                out_orig = model(imgs) 
                
                imgs_h = torch.flip(imgs, dims=[3])
                out_h = model(imgs_h)
                
                imgs_v = torch.flip(imgs, dims=[2])
                out_v = model(imgs_v)
                
                prob_orig = F.softmax(out_orig, dim=1)
                prob_h    = F.softmax(out_h, dim=1)
                prob_v    = F.softmax(out_v, dim=1)
                
                mean_prob = (prob_orig + prob_h + prob_v) / 3
                _, pred = torch.max(mean_prob, 1)

            preds.extend(pred.cpu().numpy())
            gts.extend(labels.numpy()) 

    return gts, preds

# ============================================================
# MAIN
# ============================================================
def main():
    df = pd.read_csv(CSV_PATH)

    classes = sorted(df["class"].unique().tolist())
    class_to_idx = {c:i for i,c in enumerate(classes)}
    idx_to_class = {i:c for c,i in class_to_idx.items()}

    df["LABEL_IDX"] = df["class"].map(class_to_idx)

    class_counts = np.bincount(df["LABEL_IDX"])
    soft_weights = 1.0 / np.sqrt(class_counts)
    soft_weights = soft_weights / np.sum(soft_weights) * len(classes)
    
    class_weights = torch.tensor(soft_weights, dtype=torch.float).to(DEVICE)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["LABEL_IDX"], random_state=42)

    train_dataset = BatikDataset(train_df, IMAGE_DIR, train_transform)
    val_dataset   = BatikDataset(val_df, IMAGE_DIR, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Jumlah kelas: {len(classes)}")

    model = build_model(len(classes))

    best_acc, y_true, y_pred = train_model(
        model, train_loader, val_loader, len(classes), class_weights
    )

    print("\nFINAL RESULT (No TTA)")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    # Load model terbaik sebelum TTA
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_vit.pth")))

    # Jalankan Evaluasi Spesial (TTA)
    y_true_tta, y_pred_tta = evaluate_model_with_tta(model, val_loader)

    print("\nFINAL RESULT WITH TTA")
    print(f"TTA Accuracy: {accuracy_score(y_true_tta, y_pred_tta):.4f}")
    print(classification_report(y_true_tta, y_pred_tta, target_names=classes, zero_division=0))

    import json
    with open(os.path.join(OUTPUT_DIR, "class_mapping.json"), "w") as f:
        json.dump(idx_to_class, f, indent=4)

    # ============================================================
    # PERBAIKAN: PLOT CONFUSION MATRIX DARI HASIL TTA
    # ============================================================
    print("\nGenerating Confusion Matrix for TTA Results...")
    cm = confusion_matrix(y_true_tta, y_pred_tta)
    
    # Atur ukuran gambar berdasarkan total jumlah kelas secara dinamis
    fig_size = max(10, len(classes) * 0.8)
    plt.figure(figsize=(fig_size, fig_size * 0.8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - ViT (with TTA)', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Simpan hasil confusion matrix TTA ke direktori output
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_tta.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    print(f"Confusion Matrix TTA berhasil disimpan ke: {cm_path}")

if __name__ == "__main__":
    main()