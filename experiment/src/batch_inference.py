import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import re
import random
import numpy as np

from PIL import Image
from torchvision import transforms
from transformers import (
    AutoProcessor,
    AutoModelForImageClassification,
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig

# ============================================================
# 1. SETUP & CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

CLASSIFIER_PATH = "/mnt/extended-home/dzakaaufa/models/dinov2/best_dinov2_lora"
CLASS_MAPPING_PATH = os.path.join(CLASSIFIER_PATH, "class_mapping.json")

CORE_PROMPT_PURE_PHILOSOPHY = (
    "You are an expert batik cultural historian. "
    "Your task is to provide the canonical philosophical meaning, traditional values, and symbolic representation of the '{kelas}' batik motif.\n"
    "Instructions:\n"
    "1. Discuss explicitly what the '{kelas}' motif symbolizes and its deeper cultural philosophy.\n"
    "2. DO NOT include any visual descriptions of the fabric (such as background colors, specific motif colors, shapes, layout, or design structures).\n"
    "3. Output strictly ONE coherent paragraph detailing only the philosophy and symbolism, without introductory remarks, concluding remarks, bullet points, or lists."
)

CORE_PROMPT_VISUAL_PHILOSOPHY = (
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

# ============================================================
# 2. LOAD CLASS MAPPING & TRANSFORMS
# ============================================================

with open(CLASS_MAPPING_PATH, "r") as f:
    IDX_TO_CLASS = json.load(f)

dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# 3. UTILITIES
# ============================================================

def normalize_filename(path):
    return os.path.basename(str(path)).strip().lower()

# ============================================================
# 4. LOAD MODELS
# ============================================================

def load_classification_model(lora_path):
    config = PeftConfig.from_pretrained(lora_path)
    base_model = AutoModelForImageClassification.from_pretrained(
        config.base_model_name_or_path, num_labels=len(IDX_TO_CLASS), ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.to(DEVICE).eval()
    print("-> DINOv2 Classifier berhasil dimuat!")
    return model

def load_vlm_model(adapter_path, base_model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    print("-> Memuat Processor Qwen2.5-VL...")
    processor = AutoProcessor.from_pretrained(base_model_name, max_pixels=512 * 512)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    has_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if has_bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForVision2Seq.from_pretrained(base_model_name, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(base_model, adapter_path, adapter_name="batik_adapter")
    model.eval()
    print("-> Qwen2.5-VL beserta adapter berhasil dimuat!\n")
    return processor, model

# ============================================================
# 5. MULTIMODAL PIPELINE
# ============================================================

class BatikMultimodalPipeline:
    def __init__(self, classifier_path, vlm_adapter_path):
        self.classifier = load_classification_model(classifier_path)
        self.vlm_processor, self.vlm_model = load_vlm_model(vlm_adapter_path)

    def classify_motif(self, image):
        img_tensor = dinov2_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            probs = F.softmax(outputs.logits, dim=1)[0]
            confidence, class_idx = torch.max(probs, dim=0)
            predicted_class = IDX_TO_CLASS[str(class_idx.item())]
        return predicted_class, confidence.item()

    def generate_caption(self, image, motif_label=None, mode="no_class", experiment_type="PURE_PHILOSOPHY"):
        prefix = f"This batik is confidently identified as the '{motif_label}' motif. " if mode == "hard" else (f"This batik fabric may contain characteristics of the '{motif_label}' motif. " if mode == "soft" else "")
        
        if experiment_type == "PURE_PHILOSOPHY":
            prompt_text = prefix + CORE_PROMPT_PURE_PHILOSOPHY.format(kelas=motif_label)
            max_tokens = 256
        else:
            prompt_text = prefix + CORE_PROMPT_VISUAL_PHILOSOPHY.format(kelas=motif_label)
            max_tokens = 512

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
        text_formatted = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.vlm_processor(text=[text_formatted], images=[image], padding=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_ids = self.vlm_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.1)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        caption = self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
        return caption

    def process(self, image_path, mode="no_class", experiment_type="PURE_PHILOSOPHY"):
        raw_image = Image.open(image_path).convert("RGB")
        motif_name, conf = self.classify_motif(raw_image)
        effective_mode = "hard" if mode == "auto" and conf >= 0.80 else ("soft" if mode == "auto" and conf >= 0.50 else ("no_class" if mode == "auto" else mode))
        caption = self.generate_caption(raw_image, motif_label=motif_name, mode=effective_mode, experiment_type=experiment_type)
        return {"motif_detected": motif_name, "confidence_score": conf, "generated_output": caption, "used_prompt_mode": effective_mode}

def get_ground_truth_mapping(gt_file_path, experiment_type):
    caption_mapping, class_mapping, core_mapping, test_images_only = {}, {}, {}, set()
    if gt_file_path and os.path.exists(gt_file_path):
        gt_df = pd.read_csv(gt_file_path)
        if 'split' in gt_df.columns:
            gt_df = gt_df[gt_df['split'] == 'test']
        for _, row in gt_df.iterrows():
            filename = normalize_filename(row['image_path'])
            class_mapping[filename] = str(row['class'])
            test_images_only.add(filename)
            
            # [BARU] Map philosophy_core ke nama file (Aman jika kolom tidak ada/bernilai NaN)
            core_mapping[filename] = row.get('philosophy_core', None)
            
            if experiment_type == "PURE_PHILOSOPHY":
                caption_mapping[filename] = str(row['philosophy_en'])
            else:
                caption_mapping[filename] = f"{str(row['caption_en'])}\n\n{str(row['philosophy_en'])}"
    return caption_mapping, class_mapping, core_mapping, test_images_only

# ============================================================
# 6. BATCH RUN GENERATION / PREDICTION ONLY
# ============================================================

if __name__ == "__main__":
    EXPERIMENT_TYPE = "VISUAL_PHILOSOPHY"  # Pilihan: "PURE_PHILOSOPHY" atau "VISUAL_PHILOSOPHY"
    INFERENCE_MODE = "hard"
    
    VLM_ADAPTER_PATH = "/mnt/extended-home/dzakaaufa/experiment/models/qwen2_5-VL-philosophy"
    IMAGE_FOLDER = "/mnt/extended-home/dzakaaufa/dataset/all_images_captioning"
    GROUND_TRUTH_FILE = "/mnt/extended-home/dzakaaufa/experiment/dataset/final_dataset_split1.csv"
    OUTPUT_CSV = f"/mnt/extended-home/dzakaaufa/experiment/hasil/{EXPERIMENT_TYPE.lower()}_{INFERENCE_MODE}_predict.csv"
    
    pipeline = BatikMultimodalPipeline(CLASSIFIER_PATH, VLM_ADAPTER_PATH)
    gt_mapping, gt_class_mapping, gt_core_mapping, test_images_whitelist = get_ground_truth_mapping(GROUND_TRUTH_FILE, EXPERIMENT_TYPE)
    
    image_paths = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) and normalize_filename(os.path.join(IMAGE_FOLDER, f)) in test_images_whitelist]
    print(f"\n[INFO] Disaring berdasarkan split 'test': {len(image_paths)} gambar siap diinferensi.")
    
    extracted_data = []
    
    for img_path in tqdm(image_paths, desc=f"Running Inference [{EXPERIMENT_TYPE}]"):
        try:
            filename = normalize_filename(img_path)
            reference_full = gt_mapping.get(filename, "MISSING_GROUND_TRUTH_REFERENCE")
            gt_class = gt_class_mapping.get(filename, "UNKNOWN")
            philosophy_core = gt_core_mapping.get(filename, None) # [BARU] Ambil core keyword
            
            # Jalankan Model Pipeline
            res = pipeline.process(img_path, mode=INFERENCE_MODE, experiment_type=EXPERIMENT_TYPE)
            pred_full = res["generated_output"]
            
            # Kumpulkan data prediksi mentah
            row_data = {
                "image_path": img_path,
                "ground_truth_class": gt_class,
                "predicted_class": res["motif_detected"],
                "classification_confidence": res["confidence_score"],
                "prediction": pred_full,
                "reference": reference_full,
                "philosophy_core": philosophy_core, # [BARU] Masuk ke dalam tabel keluaran
                "used_prompt_mode": res["used_prompt_mode"]
            }
            extracted_data.append(row_data)
            
        except Exception as e:
            print(f"\n[WARN] Gagal memproses berkas {img_path}. Error: {e}")
            continue

    # ============================================================
    # SIMPAN HASIL PREDIKSI MENTAH
    # ============================================================
    if extracted_data:
        df_results = pd.DataFrame(extracted_data)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        
        # Urutkan susunan kolom agar rapi saat dibuka di Excel/WPS
        urutan_kolom = [
            "image_path", "ground_truth_class", "predicted_class", 
            "classification_confidence", "prediction", "reference", 
            "philosophy_core", "used_prompt_mode"
        ]
        df_results = df_results[urutan_kolom]
        
        df_results.to_csv(OUTPUT_CSV, index=False)
        print(f"\n{'='*60}")
        print(f"[SUKSES] Hasil prediksi murni disimpan ke → {OUTPUT_CSV}")
        print(f"         Total data berhasil diprediksi: {len(df_results)} baris")
        print(f"{'='*60}")
    else:
        print("\n[ERROR] Tidak ada data gambar uji yang berhasil diproses.")