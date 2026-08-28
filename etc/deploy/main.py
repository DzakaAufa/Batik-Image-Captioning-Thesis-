import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms

# FastAPI Imports
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import gc # Garbage collector Python

# Transformers & Hugging Face Imports
from transformers import (
    AutoProcessor,
    AutoModelForImageClassification,
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig

# Wajib untuk sinkronisasi token gambar Qwen
from qwen_vl_utils import process_vision_info

# ============================================================
# 1. SETUP & CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sesuaikan path ini dengan server lab kamu
CLASSIFIER_PATH = "/mnt/extended-home/dzakaaufa/leakage/models150/dino"
VLM_ADAPTER_PATH = "/mnt/extended-home/dzakaaufa/models/qwen2.5-vl-7b-eng_inject2"
CLASS_MAPPING_PATH = os.path.join(CLASSIFIER_PATH, "class_mapping.json")

# ============================================================
# 2. PROMPT TEMPLATES
# ============================================================

PROMPT_HIGH_CONF = (
    "Your task is to generate a precise, objective, and highly accurate visual description of the provided batik image.\n"
    "This batik is classified as the '{kelas}' motif.\n"
    "Instructions:\n"
    "1. Start the first sentence with exactly: "
    "'The batik fabric features the {kelas} motif, characterized by [list 2–4 dominant motif colors] motifs.'\n"
    "   Example: 'The batik fabric features the Malang motif, characterized by dark green motifs.'\n"
    "   List only the 2–4 most visually dominant motif colors. Do NOT enumerate all visible colors.\n"
    "2. In the second sentence, identify the dominant motif elements (e.g. fern plants, fish, geometric shapes, floral patterns).\n"
    "3. In the third sentence, describe how the motifs are arranged across the fabric "
    "(e.g. non-geometric, diagonal, repeating grid, symmetrical). "
    "Include spacing, density, and orientation if clearly visible.\n"
    "4. If isen-isen (small dots, short lines, or fillers) are clearly visible, include them.\n"
    "5. Describe only what is directly visible in the image. "
    "Do NOT invent colors not clearly present. "
    "Do NOT include cultural meanings, historical context, or symbolic interpretations.\n"
    "6. Output strictly ONE coherent paragraph of 3–4 sentences, without any introductory or concluding remarks."
)

PROMPT_MID_CONF = (
    "Your task is to generate a precise, objective, and highly accurate visual description of the provided batik image.\n"
    "This batik is classified as the '{kelas}' motif.\n"
    "Instructions:\n"
    "1. Start the first sentence with exactly: "
    "'The batik fabric features the {kelas} motif, characterized by [list 2–4 dominant motif colors] motifs.'\n"
    "   Example: 'The batik fabric features the Malang motif, characterized by dark green motifs.'\n"
    "   List only the 2–4 most visually dominant motif colors. Do NOT enumerate all visible colors.\n"
    "2. In the second sentence, identify the dominant motif elements (e.g. fern plants, fish, geometric shapes, floral patterns).\n"
    "3. In the third sentence, describe how the motifs are arranged across the fabric "
    "(e.g. non-geometric, diagonal, repeating grid, symmetrical). "
    "Include spacing, density, and orientation if clearly visible.\n"
    "4. If isen-isen (small dots, short lines, or fillers) are clearly visible, include them.\n"
    "5. Describe only what is directly visible in the image. "
    "Do NOT invent colors not clearly present. "
    "Do NOT include cultural meanings, historical context, or symbolic interpretations.\n"
    "6. Output strictly ONE coherent paragraph of 3–4 sentences, without any introductory or concluding remarks."
)

PROMPT_LOW_CONF = (
    "Your task is to generate a precise, objective, and highly accurate visual description of the provided batik image.\n"
    "This batik is classified as the '{kelas}' motif.\n"
    "Instructions:\n"
    "1. Start the first sentence with exactly: "
    "'The batik fabric features the {kelas} motif, characterized by [list 2–4 dominant motif colors] motifs.'\n"
    "   Example: 'The batik fabric features the Malang motif, characterized by dark green motifs.'\n"
    "   List only the 2–4 most visually dominant motif colors. Do NOT enumerate all visible colors.\n"
    "2. In the second sentence, identify the dominant motif elements (e.g. fern plants, fish, geometric shapes, floral patterns).\n"
    "3. In the third sentence, describe how the motifs are arranged across the fabric "
    "(e.g. non-geometric, diagonal, repeating grid, symmetrical). "
    "Include spacing, density, and orientation if clearly visible.\n"
    "4. If isen-isen (small dots, short lines, or fillers) are clearly visible, include them.\n"
    "5. Describe only what is directly visible in the image. "
    "Do NOT invent colors not clearly present. "
    "Do NOT include cultural meanings, historical context, or symbolic interpretations.\n"
    "6. Output strictly ONE coherent paragraph of 3–4 sentences, without any introductory or concluding remarks."
)

# ============================================================
# 3. LOAD MAPPING & TRANSFORMS
# ============================================================

with open(CLASS_MAPPING_PATH, "r") as f:
    IDX_TO_CLASS = json.load(f)

dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# 4. UTILITIES & MODEL LOADERS
# ============================================================

def clean_adapter_config(adapter_path):
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    config.pop("eva_config", None)
    config.pop("auto_mapping", None)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

def load_classification_model(lora_path):
    config = PeftConfig.from_pretrained(lora_path)
    base_model = AutoModelForImageClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=len(IDX_TO_CLASS),
        ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.to(DEVICE)
    model.eval()
    print("-> DINOv2 Classifier berhasil dimuat!")
    return model

def load_vlm_model(adapter_path_en, base_model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    print("-> Memuat Processor Qwen2.5-VL...")
    processor = AutoProcessor.from_pretrained(base_model_name, max_pixels=512 * 512)

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    print(f"-> Memuat Base Model: {base_model_name} (4-bit)")
    has_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if has_bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("-> Memperbaiki Config Adapter...")
    clean_adapter_config(adapter_path_en)

    print(f"-> Memasang Adapter LoRA Qwen2.5-VL:\n   {adapter_path_en}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path_en,
        adapter_name="eng"
    )
    model.eval()
    print("-> Qwen2.5-VL beserta adapter berhasil dimuat!\n")
    return processor, model

# ============================================================
# 5. MULTIMODAL PIPELINE CLASS
# ============================================================

class BatikMultimodalPipeline:
    def __init__(self, classifier_path, vlm_adapter_path):
        self.classifier = load_classification_model(classifier_path)
        self.vlm_processor, self.vlm_model = load_vlm_model(vlm_adapter_path)

    def classify_motif(self, image):
        img_tensor = dinov2_transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)[0]
            confidence, class_idx = torch.max(probs, dim=0)
            predicted_class = IDX_TO_CLASS[str(class_idx.item())]
            
        return predicted_class, confidence.item()

    def generate_caption(self, image, motif_label, confidence):
        if confidence > 0.80:
            prompt_text = PROMPT_HIGH_CONF.format(kelas=motif_label)
        elif confidence < 0.50:
            prompt_text = PROMPT_LOW_CONF.format(kelas=motif_label)
        else:
            prompt_text = PROMPT_MID_CONF.format(kelas=motif_label)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        text_formatted = self.vlm_processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Integrasi Fix: Ekstrak informasi visual dengan qwen_vl_utils
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.vlm_processor(
            text=[text_formatted],
            images=image_inputs, # Gunakan input gambar yang sudah diekstrak
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True
            )

        # Potong token prompt agar tersisa jawabannya saja
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]

        caption = self.vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()

        # Post-Processing
        caption_lower = caption.lower()
        motif_raw = motif_label.lower()
        motif_space = motif_raw.replace("_", " ")

        if motif_raw in caption_lower or motif_space in caption_lower:
            caption = caption.replace("batik'", "batik '")
            return caption
        return caption

    # Diubah agar menerima object PIL Image langsung dari UploadFile web
    def process(self, raw_image: Image.Image, filename: str = "web_upload"):
        try:
            motif_name, conf = self.classify_motif(raw_image)
            caption = self.generate_caption(raw_image, motif_name, conf)

            return {
                "file": filename,
                "analysis": {
                    "motif_detected": motif_name,
                    "confidence_score": f"{conf:.2%}",
                    "confidence_value": conf
                },
                "visual_description": caption
            }
        finally:
            # PEMBERSIHAN MEMORI MUTLAK: Dijalankan setiap kali selesai proses
            # Mencegah GPU ngadat di upload gambar ke-3 atau ke-4
            del raw_image
            gc.collect()
            torch.cuda.empty_cache()

# ============================================================
# 6. FASTAPI SETUP & ENDPOINTS
# ============================================================

app = FastAPI(title="Batik Analysis API untuk Skripsi")

# Menambahkan CORS agar bisa dipanggil oleh frontend/web HTML
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Saat production ganti dengan domain web-mu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi pipeline secara global saat server start
print("\n" + "=" * 50)
print("Memulai Inisialisasi Model Backend...")
pipeline = BatikMultimodalPipeline(CLASSIFIER_PATH, VLM_ADAPTER_PATH)
print("Backend siap menerima request!")
print("=" * 50 + "\n")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Menampilkan file index.html ke user saat membuka IP server
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>File index.html tidak ditemukan di server. Pastikan posisinya satu folder dengan main.py</h3>"

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # 1. Baca gambar dari web
        request_object_content = await file.read()
        
        # 2. Konversi ke PIL Image RGB
        image = Image.open(io.BytesIO(request_object_content)).convert("RGB")
        
        # 3. Proses melalui Pipeline
        hasil = pipeline.process(image, filename=file.filename)
        
        return {
            "status": "success",
            "data": hasil
        }
        
    except Exception as e:
        print(f"[ERROR API]: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }