import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
        self.image_files = self.captions_df['image_path'].tolist()
        self.captions = self.captions_df['caption_id'].tolist()
        
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.RandomResizedCrop(size=(364, 364), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            ])
        else:
            self.transform = transforms.Compose([])

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
        except:
            image = Image.new('RGB', (364, 364), color='black')

        if self.transform:
            image = self.transform(image)

        caption = self.clean_text(str(self.captions[idx]))
        prompt = "Jawab dalam Bahasa Indonesia. Deskripsikan batik ini secara komprehensif, meliputi nama motif, elemen visual, warna, teknik pembuatan, dan makna filosofisnya: "

        inputs = self.processor(
            images=image, 
            text=prompt, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=256 
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        text_target = self.processor.tokenizer(
            caption, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=256
        )
        labels = text_target['input_ids'].squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels

        return inputs

def get_blip2_target_modules():
    return [
        # Target Q-Former
        "query", "key", "value", "dense", "q",

        # Target Language Model (Flan)
        "k", "v", "o", "wi_0", "wi_1", "wo",

        "lm_head"
        ]

def create_blip2_model(model_size="flan-t5-xl"):
    model_name = f"Salesforce/blip2-{model_size}"    
    processor = Blip2Processor.from_pretrained(model_name, use_fast=False)

    has_bf16 = torch.cuda.is_bf16_supported()
    
    if has_bf16:
        dtype = torch.bfloat16
        print(f"Model Type: {model_size} | Precision: BFloat16 (Optimal)")
    else:
        dtype = torch.float32
        print(f"Model Type: {model_size} | Precision: Float32 (Fallback aman untuk T5)")

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
    lr=1e-4,
    model_size="flan-t5-xl", 
    gradient_accumulation_steps=8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, processor, dtype = create_blip2_model(model_size)

    model.config.use_cache = False 
    for param in model.parameters():
        param.requires_grad = False

    model.gradient_checkpointing_enable() 
    model.enable_input_require_grads() 

    target_modules = get_blip2_target_modules()

    peft_config = LoraConfig(
        r=32, 
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
    )

    model = get_peft_model(model, peft_config)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
            
    model.print_trainable_parameters()

    full_df = pd.read_csv(caption_dir)
    train_df, temp_df = train_test_split(full_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Dataset Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_dataset = BatikDataset(train_df, image_dir, processor, is_train=True)
    val_dataset = BatikDataset(val_df, image_dir, processor, is_train=False)
    test_dataset = BatikDataset(test_df, image_dir, processor, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    use_scaler = (dtype == torch.float16)
    scaler = GradScaler() if use_scaler else None

    effective_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_training_steps = effective_steps_per_epoch * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.05 * num_training_steps),
        num_training_steps=num_training_steps
    )

    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            ctx = autocast(dtype=dtype) if dtype != torch.float32 else torch.enable_grad()
            
            with ctx:
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()

            loss_val = loss.item() * gradient_accumulation_steps
            train_loss += loss_val
            progress_bar.set_postfix({'train_loss': f"{loss_val:.4f}"})
        
        avg_train_loss = train_loss / len(train_dataloader)

        model.eval()
        val_loss = 0
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)

        with torch.no_grad():
            for batch in val_progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                ctx = autocast(dtype=dtype) if dtype != torch.float32 else torch.enable_grad()
                
                with ctx:
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
    
    return model, processor

if __name__ == "__main__":    
    model, processor = train_blip2_model(
        image_dir="/mnt/extended-home/dzakaaufa/dataset/batik",
        caption_dir="/mnt/extended-home/dzakaaufa/dataset/caption/final_dataset.csv",
        output_dir="/mnt/extended-home/dzakaaufa/models/flan-t5-xl-indo",
        model_size="flan-t5-xl",
        epochs=50,
        batch_size=1, 
        lr=1e-4,
        gradient_accumulation_steps=8,
    )