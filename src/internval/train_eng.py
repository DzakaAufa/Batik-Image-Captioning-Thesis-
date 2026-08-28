import os
import re
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE      = 448

# def build_internvl_transform(is_train=True):
#     base = []
#     if is_train:
#         base += [
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
#             transforms.RandomResizedCrop(
#                 size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1),
#                 interpolation=InterpolationMode.BICUBIC
#             ),
#         ]
#     else:
#         base += [transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC)]

#     base += [
#         transforms.ToTensor(),
#         transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     ]
#     return transforms.Compose(base)

def build_internvl_transform(is_train=True):
    base = []
    if is_train:
        base += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            # Resize standar untuk thumbnail, kita tidak pakai crop ekstrem
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(base)

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images

# class BatikDataset(Dataset):
#     def __init__(self, caption_df, image_dir, tokenizer, is_train=True, max_length=512):
#         self.image_dir  = image_dir
#         self.tokenizer  = tokenizer
#         self.is_train   = is_train
#         self.max_length = max_length

#         self.image_files = caption_df["Image Path"].tolist()
#         self.captions    = caption_df["CAPTION_ENG"].tolist()
#         self.transform   = build_internvl_transform(is_train)

#         # ID untuk token khusus
#         self.img_context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
#         self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
#         # InternVL2 (tile tunggal 448x448) secara absolut menggunakan 256 token vision
#         self.num_image_tokens = 256 

#     def clean_text(self, text):
#         return re.sub(r"\s+", " ", str(text).encode("ascii", "ignore").decode("ascii")).strip().lower()

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         # 1. Load & Transform Gambar
#         img_path_raw = self.image_files[idx]
#         img_path = img_path_raw if os.path.isabs(img_path_raw) else os.path.join(self.image_dir, img_path_raw)

#         try:
#             image = Image.open(img_path).convert("RGB")
#         except Exception:
#             image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="black")

#         pixel_values = self.transform(image).unsqueeze(0)  # [1, 3, 448, 448]
#         image_flags = torch.ones(1, 1, dtype=torch.long)   # [1, 1]

#         # 2. Bangun Text ID secara Langsung (Mencegah Bug Tokenizer)
#         PROMPT_EN = (
#             "Describe this batik fabric objectively based ONLY on exact visual evidence. "
#             "This fabric belongs to the '{kelas}' class.\n"
#             "Instructions:\n"
#             "1. State the background color and the specific colors of the motifs.\n"
#             "2. Describe the exact shapes of the main motifs (e.g., specific floral shapes, animals, geometric lines, or abstract curves).\n"
#             "3. Describe the layout/pattern arrangement (e.g., diagonal, scattered, repeating grids, symmetrical).\n"
#             "4. DO NOT hallucinate. Do not name non-colors as colors. Do not invent philosophical meanings or clothing uses.\n"
#             "5. Write strictly ONE coherent paragraph."
#         )
#         label = str(self.labels[idx])  # tambahkan kolom label ke dataset
#         prompt_text = PROMPT_EN.format(kelas=label)
#         # prompt_text = "Answer in English only. Describe this batik comprehensively, including the motif name, visual elements, colors, crafting technique, and its philosophical meaning:"
#         response_text = self.clean_text(self.captions[idx])

#         # Pemisahan teks agar tokenisasi tidak terganggu oleh token gambar
#         user_str = f"<|im_start|>user\n<img>"
#         mid_str  = f"</img>\n{PROMPT_EN}<|im_end|>\n<|im_start|>assistant\n"
#         resp_str = f"{response_text}<|im_end|>"

#         user_ids = self.tokenizer(user_str, add_special_tokens=False).input_ids
#         img_ids  = [self.img_context_id] * self.num_image_tokens
#         mid_ids  = self.tokenizer(mid_str, add_special_tokens=False).input_ids
#         resp_ids = self.tokenizer(resp_str, add_special_tokens=False).input_ids

#         # 3. Gabungkan dan buat Labels (Masking prompt dengan -100)
#         input_ids = user_ids + img_ids + mid_ids + resp_ids
#         labels = [-100] * len(user_ids + img_ids + mid_ids) + resp_ids

#         # 4. Truncation & Padding
#         input_ids = input_ids[:self.max_length]
#         labels    = labels[:self.max_length]

#         pad_len = self.max_length - len(input_ids)
#         attention_mask = [1] * len(input_ids) + [0] * pad_len
#         input_ids      = input_ids + [self.pad_id] * pad_len
#         labels         = labels + [-100] * pad_len

#         return {
#             "pixel_values": pixel_values,
#             "image_flags": image_flags,
#             "input_ids": torch.tensor(input_ids, dtype=torch.long),
#             "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
#             "labels": torch.tensor(labels, dtype=torch.long),
#         }

class BatikDataset(Dataset):
    def __init__(self, caption_df, image_dir, tokenizer, is_train=True, max_length=1536): # MAX LENGTH DINAIKKAN
        self.image_dir  = image_dir
        self.tokenizer  = tokenizer
        self.is_train   = is_train
        self.max_length = max_length

        self.image_files = caption_df["Image Path"].tolist()
        self.captions    = caption_df["CAPTION_ENG"].tolist()
        # Ambil kolom CLASS untuk mengisi variabel {kelas} di prompt
        self.classes     = caption_df["CLASS"].tolist() 
        self.transform   = build_internvl_transform(is_train)

        self.img_context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        # 256 token PER TILE
        self.tokens_per_tile = 256 

    def clean_text(self, text):
        text = str(text).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path_raw = self.image_files[idx]
        img_path = img_path_raw if os.path.isabs(img_path_raw) else os.path.join(self.image_dir, img_path_raw)
        kelas_batik = str(self.classes[idx])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="black")

        # 1. DYNAMIC TILING: Menghasilkan beberapa patch gambar
        images = dynamic_preprocess(image, max_num=6)
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values) # Shape: [num_tiles, 3, 448, 448]
        num_tiles = pixel_values.size(0)
        
        image_flags = torch.ones(num_tiles, 1, dtype=torch.long)

        # 2. Bangun Text ID
        PROMPT_EN = (
            "Describe this batik fabric objectively based ONLY on exact visual evidence. "
            f"This fabric belongs to the '{kelas_batik}' class.\n"
            "Instructions:\n"
            "1. State the background color and the specific colors of the motifs.\n"
            "2. Describe the exact shapes of the main motifs (e.g., specific floral shapes, animals, geometric lines, or abstract curves).\n"
            "3. Describe the layout/pattern arrangement (e.g., diagonal, scattered, repeating grids, symmetrical).\n"
            "4. DO NOT hallucinate. Do not name non-colors as colors. Do not invent philosophical meanings or clothing uses.\n"
            "5. Write strictly ONE coherent paragraph."
        )
        response_text = self.clean_text(self.captions[idx])

        # Ulangi <image> token sesuai jumlah tile
        image_tokens_str = "<image>\n" * num_tiles
        user_str = f"<|im_start|>user\n{image_tokens_str}{PROMPT_EN}<|im_end|>\n<|im_start|>assistant\n"
        resp_str = f"{response_text}<|im_end|>"

        # Tokenisasi
        # Catatan: Kita replace <image> secara manual dengan img_context_id x 256
        user_ids = self.tokenizer(user_str, add_special_tokens=False).input_ids
        
        # Cari posisi token <image> dan ganti dengan rentetan <IMG_CONTEXT>
        # (Pendekatan sederhana: jika Anda mendaftarkan token <image>, ganti id-nya)
        # InternVL biasanya mengubah tag <image> menjadi 256 img_context_id
        final_user_ids = []
        image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        if image_token_id is None or image_token_id == self.tokenizer.unk_token_id:
            # Fallback aman
            pass 
        
        # Karena kita membangun manual, lebih baik merakitnya:
        user_text_only = f"<|im_start|>user\n{PROMPT_EN}<|im_end|>\n<|im_start|>assistant\n"
        user_ids = self.tokenizer(user_text_only, add_special_tokens=False).input_ids
        img_ids  = [self.img_context_id] * (self.tokens_per_tile * num_tiles)
        resp_ids = self.tokenizer(resp_str, add_special_tokens=False).input_ids

        input_ids = user_ids[:2] + img_ids + user_ids[2:] + resp_ids # Menyisipkan gambar setelah <|im_start|>user\n
        
        # Masking label
        labels = [-100] * len(user_ids[:2] + img_ids + user_ids[2:]) + resp_ids

        # 4. Truncation & Padding
        input_ids = input_ids[:self.max_length]
        labels    = labels[:self.max_length]

        pad_len = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids      = input_ids + [self.pad_id] * pad_len
        labels         = labels + [-100] * pad_len

        return {
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def collate_fn(batch):
    return {
        "pixel_values": torch.cat([s["pixel_values"] for s in batch], dim=0),
        "image_flags": torch.cat([s["image_flags"] for s in batch], dim=0),
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
        "labels": torch.stack([s["labels"] for s in batch]),
    }

def train_internvl2_model(image_dir, caption_dir, output_dir, epochs=20, batch_size=1, lr=1e-4, gradient_accumulation_steps=8, max_length=1536):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "OpenGVLab/InternVL2-8B"

    # Penentuan Presisi Otomatis
    has_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if has_bf16 else torch.float16 # Float16 sebagai fallback, BUKAN float32
    print(f"Menggunakan Presisi: {dtype}")

    print(f"Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
    )

    img_context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    
    # Injeksi ke config
    model.config.img_context_token_id = img_context_id
    
    # Injeksi langsung ke atribut class model (karena modeling_internvl_chat.py mencarinya di sini)
    if hasattr(model, 'img_context_token_id'):
        model.img_context_token_id = img_context_id
    else:
        setattr(model, 'img_context_token_id', img_context_id)
        
    print(f"🔧 Patched img_context_token_id: {model.img_context_token_id}")

    # Persiapan LoRA
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["qkv", "proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # "qkv", "proj", 
    model = get_peft_model(model, peft_config)
    
    # [FIX] Pastikan PEFT Base Model juga memiliki atribut ini
    base_model = model.get_base_model()
    if not hasattr(base_model, 'img_context_token_id'):
        setattr(base_model, 'img_context_token_id', img_context_id)
        
    model.print_trainable_parameters()

    # Dataset & Dataloader
    full_df = pd.read_csv(caption_dir)
    train_df, temp_df = train_test_split(full_df, test_size=0.2, random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_dl = DataLoader(BatikDataset(train_df, image_dir, tokenizer, True, max_length), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_dl   = DataLoader(BatikDataset(val_df, image_dir, tokenizer, False, max_length), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    test_dl  = DataLoader(BatikDataset(test_df, image_dir, tokenizer, False, max_length), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    use_scaler = (dtype == torch.float16)
    scaler = GradScaler() if use_scaler else None

    total_steps = math.ceil(len(train_dl) / gradient_accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) if k != "pixel_values" else v.to(device, dtype=dtype) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dl):
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="[Val]", leave=False):
                batch = {k: v.to(device) if k != "pixel_values" else v.to(device, dtype=dtype) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=dtype):
                    val_loss += model(**batch).loss.item()

        avg_val_loss = val_loss / len(val_dl)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_dl):.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f" -> Checkpoint disimpan!")

        torch.cuda.empty_cache()

    # Final Test
    print("\nEvaluasi Final Test Set...")
    model.load_adapter(output_dir, adapter_name="default")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="[Test]"):
            batch = {k: v.to(device) if k != "pixel_values" else v.to(device, dtype=dtype) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=dtype):
                test_loss += model(**batch).loss.item()

    print(f"FINAL TEST LOSS: {test_loss / len(test_dl):.4f}")
    return model, tokenizer

if __name__ == "__main__":
    train_internvl2_model(
        image_dir="/mnt/extended-home/dzakaaufa/dataset/gambar",
        caption_dir="/mnt/extended-home/dzakaaufa/dataset/caption/dataset_captioned_internvl.csv",
        output_dir="/mnt/extended-home/dzakaaufa/models/internvl2-8b-eng"
    )