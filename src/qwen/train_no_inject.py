import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
if not hasattr(torch.nn.Module, "set_submodule"):
    def set_submodule(self, target: str, module: torch.nn.Module) -> None:
        atoms: list[str] = target.split(".")
        name = atoms.pop(-1)
        mod = self
        for item in atoms:
            mod = getattr(mod, item)
        setattr(mod, name, module)
    torch.nn.Module.set_submodule = set_submodule

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, get_linear_schedule_with_warmup, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
import re
from torchvision import transforms
import math
from torch.nn.utils.rnn import pad_sequence

import os
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PROMPT_VISUAL_EN = (
    "You are an expert batik annotator. "
    "Your task is to generate a precise, objective, and highly accurate visual description of the provided batik image.\n"
    "Instructions:\n"
    "1. Start the first sentence with exactly: "
    "'The batik fabric features [motif descriptor] motifs, characterized by [list 2–4 dominant motif colors] motifs.'\n"
    "   For [motif descriptor], use a concise visual description of the dominant shape (e.g., 'floral', 'diagonal geometric', 'figurative').\n"
    "   List only the 2–4 most visually dominant motif colors. Do NOT enumerate all visible colors.\n"
    "2. In the second sentence, identify the dominant motif elements (e.g., fern plants, fish, geometric shapes, floral patterns).\n"
    "3. In the third sentence, describe how the motifs are arranged across the fabric "
    "(e.g., non-geometric, diagonal, repeating grid, symmetrical). "
    "Include spacing, density, and orientation if clearly visible.\n"
    "4. If isen-isen (small dots, short lines, or fillers) are clearly visible, include them.\n"
    "5. Describe only what is directly visible in the image. "
    "Do NOT invent colors not clearly present. "
    "Do NOT include cultural meanings, historical context, or symbolic interpretations.\n"
    "6. Output strictly ONE coherent paragraph of 3–4 sentences, without any introductory or concluding remarks."
)

class BatikDataset(Dataset):
    def __init__(self, caption_dir, image_dir, processor, is_train=True):
        self.image_dir = image_dir
        self.processor = processor
        self.is_train = is_train

        self.captions_df = caption_dir
        self.image_files = self.captions_df['image_path'].tolist()
        
        self.captions = self.captions_df['caption_en'].tolist() 
        
        self.classes = self.captions_df['class'].tolist() 
        
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.Resize((384, 384)), 
            ])
        else:
            self.transform = transforms.Compose([transforms.Resize((384, 384))])

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
            image = Image.new('RGB', (384, 384), color='black')

        if self.transform:
            image = self.transform(image)

        caption = self.clean_text(str(self.captions[idx]))
        
        batik_class = str(self.classes[idx])
        prompt = PROMPT_VISUAL_EN.format(kelas=batik_class)

        messages_full = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": caption}
        ]
        
        messages_prompt = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]

        text_full = self.processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
        text_prompt = self.processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

        inputs_full = self.processor(text=[text_full], images=[image], return_tensors="pt", padding=False)
        inputs_prompt = self.processor(text=[text_prompt], images=[image], return_tensors="pt", padding=False)

        prompt_len = inputs_prompt["input_ids"].shape[1]
        labels = inputs_full["input_ids"].clone()
        labels[0, :prompt_len] = -100

        return {
            "input_ids": inputs_full["input_ids"][0],
            "attention_mask": inputs_full["attention_mask"][0],
            "mm_token_type_ids": inputs_full.get("mm_token_type_ids", [None])[0] if "mm_token_type_ids" in inputs_full else None,
            "pixel_values": inputs_full["pixel_values"], 
            "image_grid_thw": inputs_full["image_grid_thw"], 
            "labels": labels[0]
        }

def qwen_collate_fn(batch, pad_token_id):
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
    
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw
    }
    
    if batch[0]["mm_token_type_ids"] is not None:
        mm_token_type_ids = [item["mm_token_type_ids"] for item in batch]
        mm_token_type_ids = pad_sequence(mm_token_type_ids, batch_first=True, padding_value=0)
        result["mm_token_type_ids"] = mm_token_type_ids

    return result

def create_qwen2_model(model_size="Qwen/Qwen2.5-VL-7B-Instruct"):
    processor = AutoProcessor.from_pretrained(model_size, max_pixels=512*512)
    
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    has_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if has_bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_size, 
        quantization_config=bnb_config,
        device_map="auto"
    )

    return model, processor, dtype

def train_qwen_model(
    image_dir,
    caption_dir,
    output_dir,
    epochs=10,
    batch_size=1,
    lr=1e-4,
    model_size="Qwen/Qwen2.5-VL-7B-Instruct", 
    gradient_accumulation_steps=8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, processor, dtype = create_qwen2_model(model_size)
    model.config.use_cache = False 

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    full_df = pd.read_csv(caption_dir)
    
    train_df = full_df[full_df['split'] == 'train']
    val_df = full_df[full_df['split'] == 'val']
    test_df = full_df[full_df['split'] == 'test']
    
    print(f"Data Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    train_dataset = BatikDataset(train_df, image_dir, processor, is_train=True)
    val_dataset = BatikDataset(val_df, image_dir, processor, is_train=False)
    test_dataset = BatikDataset(test_df, image_dir, processor, is_train=False)

    pad_id = processor.tokenizer.pad_token_id
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: qwen_collate_fn(b, pad_id))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: qwen_collate_fn(b, pad_id))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: qwen_collate_fn(b, pad_id))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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

            with autocast(dtype=dtype):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            loss_val = loss.item() * gradient_accumulation_steps
            train_loss += loss_val
            progress_bar.set_postfix({'loss': loss_val})
        
        avg_train_loss = train_loss / len(train_dataloader)

        model.eval()
        val_loss = 0
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast(dtype=dtype):
                    outputs = model(**batch)
                    v_loss = outputs.loss
                    
                val_loss += v_loss.item()
                val_progress_bar.set_postfix({'val_loss': v_loss.item()})

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)  
            processor.save_pretrained(output_dir)
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}")

    print("\nTraining Selesai. Memulai Evaluasi Final pada Test Set (Unbiased)...")
    model.load_adapter(output_dir, adapter_name="default")
    model.eval()
    
    test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Final Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(dtype=dtype):
                outputs = model(**batch)
                t_loss = outputs.loss 
            test_loss += t_loss.item()

    avg_test_loss = test_loss / len(test_dataloader)
    print(f"\nFINAL UNBIASED TEST LOSS: {avg_test_loss:.4f}")
    return model, processor

if __name__ == "__main__":
    model, processor = train_qwen_model(
        image_dir="/mnt/extended-home/dzakaaufa/dataset/all_images_captioning",
        caption_dir="/mnt/extended-home/dzakaaufa/dataset/caption/data_mix_inject_split1.csv",
        output_dir="/mnt/extended-home/dzakaaufa/models/qwen2.5-vl-7b-eng_noinject3",
        epochs=10,
        batch_size=8,
        lr=1e-4, 
        model_size="Qwen/Qwen2.5-VL-7B-Instruct",
        gradient_accumulation_steps=4,
    )