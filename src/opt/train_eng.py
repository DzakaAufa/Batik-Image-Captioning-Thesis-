import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import re
from torchvision import transforms
import math
from sklearn.model_selection import train_test_split

class BatikDataset(Dataset):
    def __init__(self, caption_dir, image_dir, processor, is_train=True):
        self.image_dir = image_dir
        self.processor = processor
        self.is_train = is_train

        self.captions_df = caption_dir
        self.image_files = self.captions_df['Image Path'].tolist()
        self.captions = self.captions_df['CAPTION_EN'].tolist()

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224))
            ])

    def clean_text(self, text):
        if not isinstance(text, str): return ""
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        return text

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path_raw = self.image_files[idx]
        img_path = img_path_raw if os.path.isabs(img_path_raw) else os.path.join(self.image_dir, img_path_raw)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        caption = self.clean_text(str(self.captions[idx]))

        prompt = "Answer in English only. Describe this batik comprehensively, including the motif name, visual elements, colors, crafting technique, and its philosophical meaning: "

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )

        label_enc = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )

        labels = label_enc["input_ids"].squeeze(0).clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        result = {k: v.squeeze(0) for k, v in inputs.items()}
        result["labels"] = labels
        return result


def get_blip2_target_modules():
    return [
        # Q-Former
        "query", "key", "value", "dense",
        # OPT LM
        "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2",
        "lm_head",
        # [PERBAIKAN] Vision encoder layer terakhir ikut dilatih via LoRA
        "patch_embedding",
        "position_embedding",
    ]


def create_blip2_model(model_size="opt-2.7b"):
    model_name = f"Salesforce/blip2-{model_size}"
    processor = Blip2Processor.from_pretrained(model_name, use_fast=False)
    dtype = torch.float16
    print(f"Model Type: {model_size} | Precision: Float16")

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
    )
    return model, processor, dtype


def train_blip2_model(
    image_dir,
    caption_dir,
    output_dir,
    epochs=50,
    batch_size=2,
    lr=5e-5,
    model_size="opt-2.7b",
    gradient_accumulation_steps=8,
    use_amp=True
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, processor, dtype = create_blip2_model(model_size)

    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False

    # [PERBAIKAN] Unfreeze 2 layer terakhir vision encoder agar model benar-benar "melihat" gambar
    for name, param in model.named_parameters():
        if any(layer in name for layer in [
            "vision_model.encoder.layers.38",
            "vision_model.encoder.layers.37",
        ]):
            param.requires_grad = True
            print(f"Unfrozen vision layer: {name}")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=get_blip2_target_modules(),
    )

    model = get_peft_model(model, peft_config)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()

    full_df = pd.read_csv(caption_dir)

    # [PERBAIKAN] Verifikasi kolom CAPTION_ID bukan numeric ID
    sample = full_df['CAPTION_ID'].iloc[0]
    if str(sample).strip().isdigit():
        raise ValueError(
            f"CAPTION_ID tampaknya berisi angka numerik ('{sample}'), "
            "bukan teks caption. Periksa kolom CSV Anda."
        )

    train_df, temp_df = train_test_split(
        full_df, test_size=0.2, random_state=42,
        stratify=full_df['CLASS']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42,
        stratify=temp_df['CLASS']
    )

    print(f"Dataset Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_dataset = BatikDataset(train_df, image_dir, processor, is_train=True)
    val_dataset   = BatikDataset(val_df,   image_dir, processor, is_train=False)
    test_dataset  = BatikDataset(test_df,  image_dir, processor, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler() if use_amp else None

    effective_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_training_steps = effective_steps_per_epoch * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * num_training_steps),
        num_training_steps=num_training_steps
    )

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_amp:
                with autocast(dtype=dtype):
                    outputs = model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    # [PERBAIKAN] Gradient clipping mencegah OPT repetition collapse
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    # [PERBAIKAN] Gradient clipping untuk non-AMP path
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            loss_val = loss.item() * gradient_accumulation_steps
            train_loss += loss_val
            progress_bar.set_postfix({'loss': f"{loss_val:.4f}"})

        avg_train_loss = train_loss / len(train_dataloader)

        model.eval()
        val_loss = 0
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")

        with torch.no_grad():
            for batch in val_progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                if use_amp:
                    with autocast(dtype=dtype):
                        outputs = model(**batch)
                        v_loss = outputs.loss
                else:
                    outputs = model(**batch)
                    v_loss = outputs.loss

                val_loss += v_loss.item()
                val_progress_bar.set_postfix({'val_loss': f"{v_loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}")

        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("Training Selesai. Memulai Evaluasi Final pada Test Set (Unbiased)...")
    print("="*50)

    model.load_adapter(output_dir, adapter_name="default")
    model.eval()

    test_loss = 0
    test_progress_bar = tqdm(test_dataloader, desc="Final Testing")

    with torch.no_grad():
        for batch in test_progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = autocast(dtype=dtype) if dtype != torch.float32 else torch.enable_grad()

            with ctx:
                outputs = model(**batch)
                t_loss = outputs.loss

            test_loss += t_loss.item()
            test_progress_bar.set_postfix({'test_loss': f"{t_loss.item():.4f}"})

    avg_test_loss = test_loss / len(test_dataloader)
    print(f"\nFINAL UNBIASED TEST LOSS: {avg_test_loss:.4f}")

if __name__ == "__main__":
    model, processor = train_blip2_model(
        image_dir="/mnt/extended-home/dzakaaufa/dataset/image",
        caption_dir="/mnt/extended-home/dzakaaufa/dataset/caption/dataset_captioned.csv",
        output_dir="/mnt/extended-home/dzakaaufa/models/opt-2.7b-eng",
        epochs=20,
        batch_size=1,
        lr=5e-5,
        model_size="opt-2.7b",
        gradient_accumulation_steps=8,
        use_amp=True
    )