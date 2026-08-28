import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

# ==========================================
# PROMPT DEFINITIONS
# ==========================================

PROMPT_VISUAL_EN = (
    "You are an expert batik annotator. "
    "Your task is to generate a precise, objective, and highly accurate visual description of the provided batik image.\n"
    "This batik is officially classified as the '{kelas}' motif.\n"
    "Instructions:\n"
    "1. You MUST start the very first sentence of the paragraph with exactly this format: "
    "'The batik fabric features the {kelas} motif, characterized by a [specify background color] background and [specify motif colors] motifs.'\n"
    "2. Use the provided motif class as contextual information and ensure all visual descriptions remain consistent with the motif.\n"
    "3. Describe the exact shapes of the dominant motifs (e.g., floral elements, animals, geometric forms, curved lines, or abstract ornaments).\n"
    "4. Describe the visible isen-isen elements (small dots, short lines, or fillers) when present.\n"
    "5. Describe the arrangement of motifs across the fabric, including repetition, spacing, orientation, symmetry, and layout structure.\n"
    "6. Describe only visual elements that are clearly observable in the image.\n"
    "7. DO NOT hallucinate. Do not invent cultural stories, symbolic meanings, historical interpretations, or unseen details.\n"
    "8. Output strictly ONE coherent paragraph without introductory remarks, concluding remarks, bullet points, or lists."
)

PROMPT_VISUAL_PHILOSOPHY_EN = (
    "You are an expert batik annotator and cultural historian. "
    "Your task is to generate a precise visual description of the provided batik image, followed by its canonical philosophical meaning.\n"
    "This batik is officially classified as the '{kelas}' motif.\n"
    "Instructions:\n"
    "1. You MUST start the very first sentence of the first paragraph with exactly this format: "
    "'The batik fabric features the {kelas} motif, characterized by a [specify background color] background and [specify motif colors] motifs.'\n"
    "2. Objectively describe the visual elements: shapes of dominant motifs, isen-isen elements, and the arrangement/layout structure.\n"
    "3. After the visual description, provide a second paragraph that clearly states the traditional philosophical meaning, values, or symbolic representation of the '{kelas}' motif.\n"
    "4. Output strictly TWO coherent paragraphs (one for visuals, one for philosophy) without introductory remarks, bullet points, or lists."
)

# ==========================================
# DATASET CLASS
# ==========================================

class BatikDataset(Dataset):
    def __init__(self, caption_dir, image_dir, processor, task="VISUAL", is_train=True):
        self.image_dir = image_dir
        self.processor = processor
        self.is_train = is_train
        self.task = task.upper()

        self.captions_df = caption_dir
        self.image_files = self.captions_df['image_path'].tolist()
        
        self.captions = self.captions_df['caption_en'].tolist() 
        self.classes = self.captions_df['class'].tolist() 
        
        # Load philosophy column if it exists and task requires it
        if 'philosophy_en' in self.captions_df.columns:
            self.philosophies = self.captions_df['philosophy_en'].tolist()
        else:
            self.philosophies = [""] * len(self.captions_df)
        
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
        # text = text.lower()  # REMOVED: To preserve your proper capitalization
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

        caption_en = self.clean_text(str(self.captions[idx]))
        batik_class = str(self.classes[idx])
        
        # ----------------------------------------------------
        # TASK LOGIC: Determine Prompt and Target Output
        # ----------------------------------------------------
        if self.task == "VISUAL":
            prompt = PROMPT_VISUAL_EN.format(kelas=batik_class)
            target_output = caption_en
            
        elif self.task == "VISUAL_PHILOSOPHY":
            prompt = PROMPT_VISUAL_PHILOSOPHY_EN.format(kelas=batik_class)
            philosophy_en = self.clean_text(str(self.philosophies[idx]))
            
            # Combine caption and philosophy with double newline
            if philosophy_en:
                target_output = f"{caption_en}\n\n{philosophy_en}"
            else:
                target_output = caption_en # Fallback if philosophy is empty
        else:
            raise ValueError(f"Unknown task: {self.task}")
        # ----------------------------------------------------

        messages_full = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": target_output}
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

# ==========================================
# COLLATION & MODEL CREATION
# ==========================================

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

# ==========================================
# TRAINING LOOP
# ==========================================

def train_qwen_model(
    image_dir,
    caption_dir,
    output_dir,
    task="VISUAL", # Added task argument
    epochs=10,
    batch_size=1,
    lr=1e-4,
    model_size="Qwen/Qwen2.5-VL-7B-Instruct", 
    gradient_accumulation_steps=8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Task: {task}")

    model, processor, dtype = create_qwen2_model(model_size)
    model.config.use_cache = False 

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    full_df = pd.read_csv(caption_dir)
    
    train_df = full_df[full_df['split'] == 'train']
    val_df = full_df[full_df['split'] == 'val']
    test_df = full_df[full_df['split'] == 'test']
    
    print(f"Data Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Pass task argument to Dataset
    train_dataset = BatikDataset(train_df, image_dir, processor, task=task, is_train=True)
    val_dataset = BatikDataset(val_df, image_dir, processor, task=task, is_train=False)
    test_dataset = BatikDataset(test_df, image_dir, processor, task=task, is_train=False)

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
    EXPERIMENT_TASK = "VISUAL_PHILOSOPHY" 
    
    model, processor = train_qwen_model(
        image_dir="/mnt/extended-home/dzakaaufa/dataset/all_images_captioning",
        caption_dir="/mnt/extended-home/dzakaaufa/experiment/dataset/final_dataset_split.csv",
        output_dir="/mnt/extended-home/dzakaaufa/experiment/models/qwen2_5-VL-philosophy",
        task=EXPERIMENT_TASK, 
        epochs=10,
        batch_size=8,
        lr=1e-4, 
        model_size="Qwen/Qwen2.5-VL-7B-Instruct",
        gradient_accumulation_steps=4,
    )