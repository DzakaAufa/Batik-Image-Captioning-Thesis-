import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import re

from PIL import Image
from torchvision import transforms
from transformers import (
    AutoProcessor,
    AutoModelForImageClassification,
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
import random
import numpy as np

# ── Patch torch < 2.4 ─────────────────────────────────────────────
if not hasattr(torch.nn.Module, "set_submodule"):
    def set_submodule(self, target, module):
        atoms = target.split(".")
        name = atoms.pop(-1)
        mod = self
        for item in atoms:
            mod = getattr(mod, item)
        setattr(mod, name, module)
    torch.nn.Module.set_submodule = set_submodule

# ============================================================
# 1. SETUP & CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

CLASSIFIER_PATH = "/mnt/extended-home/dzakaaufa/leakage/models150/dino"
CLASS_MAPPING_PATH = os.path.join(CLASSIFIER_PATH, "class_mapping.json")

PROMPT_HARD = (
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

PROMPT_NO_CLASS = (
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

with open(CLASS_MAPPING_PATH, "r") as f:
    IDX_TO_CLASS = json.load(f)

ALL_CLASS_NAMES = [v.lower().replace("_", " ").strip() for v in IDX_TO_CLASS.values()]
CLASS_NORM_TO_RAW = {
    v.lower().replace("_", " ").strip(): v
    for v in IDX_TO_CLASS.values()
}

dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# 2. UTILITIES
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

def normalize_filename(path):
    return os.path.basename(str(path)).strip().lower()

def normalize_text(text):
    text = str(text).lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def check_motif_mentioned(pred_caption, gt_class):
    """Cek apakah ground truth class disebutkan di dalam prediksi caption."""
    return normalize_text(gt_class) in normalize_text(pred_caption)

def extract_vlm_motif_from_caption(caption):
    normalized_caption = normalize_text(caption)
    matched = []
    for norm_name in ALL_CLASS_NAMES:
        if norm_name in normalized_caption:
            matched.append(norm_name)
    if not matched:
        return "UNKNOWN"
    best = max(matched, key=len)
    return CLASS_NORM_TO_RAW.get(best, best)

# ============================================================
# 3. LOAD MODELS
# ============================================================

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
    return model

def load_vlm_model(adapter_path=None, base_model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    print(f"-> Memuat Processor: {base_model_name}")
    processor = AutoProcessor.from_pretrained(base_model_name, max_pixels=512 * 512)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    has_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if has_bf16 else torch.float16

    print(f"-> Memuat Base Model (4-bit)...")
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

    if adapter_path is None or adapter_path.lower() == "none":
        print("-> MODE ZERO-SHOT AKTIF: Tidak menggunakan LoRA Adapter.")
        model = base_model
    else:
        print(f"-> Memasang LoRA Adapter dari:\n   {adapter_path}")
        clean_adapter_config(adapter_path)
        model = PeftModel.from_pretrained(base_model, adapter_path, adapter_name="eng")

    model.eval()
    return processor, model, dtype

# ============================================================
# 4. MULTIMODAL PIPELINE
# ============================================================

class BatikMultimodalPipeline:
    def __init__(self, classifier_path, vlm_adapter_path=None):
        self.classifier = load_classification_model(classifier_path)
        self.vlm_processor, self.vlm_model, self.dtype = load_vlm_model(vlm_adapter_path)

    @torch.inference_mode()
    def generate_caption(self, image, motif_label=None, mode="no_class"):
        prompt_text = PROMPT_HARD.format(kelas=motif_label) if mode == "hard" else PROMPT_NO_CLASS

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
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.vlm_processor(
            text=[text_formatted],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            output_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=1.1
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        caption = self.vlm_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        return re.sub(r"\s+", " ", caption).strip()

    def classify_motif(self, image):
        img_tensor = dinov2_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            probs = F.softmax(outputs.logits, dim=1)[0]
            confidence, class_idx = torch.max(probs, dim=0)
            predicted_class = IDX_TO_CLASS[str(class_idx.item())]
        return predicted_class, confidence.item()

    def process(self, image_path, mode="no_class"):
        raw_image = Image.open(image_path).convert("RGB")

        dinov2_class = None
        dinov2_conf = None

        if mode == "hard":
            dinov2_class, dinov2_conf = self.classify_motif(raw_image)
            motif_for_prompt = dinov2_class
        else:
            motif_for_prompt = None

        caption = self.generate_caption(raw_image, motif_label=motif_for_prompt, mode=mode)

        vlm_predicted_class = extract_vlm_motif_from_caption(caption)

        return {
            "visual_description": caption,
            "vlm_predicted_class": vlm_predicted_class,
            "used_prompt_mode": mode,

            "dinov2_class": dinov2_class,
            "dinov2_confidence": dinov2_conf,
        }

# ============================================================
# 5. TEST SPLIT FILTER & EXECUTION
# ============================================================

def get_ground_truth_mapping(gt_file_path):
    caption_mapping = {}
    class_mapping = {}
    test_images_only = set()

    print(f"-> Memuat ground truth master dari {gt_file_path}...")
    gt_df = pd.read_csv(gt_file_path)

    if 'new_split' in gt_df.columns:
        gt_df = gt_df[gt_df['new_split'] == 'test']
        print(f"-> Berhasil menyaring data. Ditemukan {len(gt_df)} baris split 'test'.")
    else:
        print("[WARN] Kolom 'split' tidak ditemukan! Memproses semua baris.")

    for _, row in gt_df.iterrows():
        filename = normalize_filename(row['image_path'])
        caption_mapping[filename] = str(row['caption_en'])
        class_mapping[filename] = str(row['class'])
        test_images_only.add(filename)

    return caption_mapping, class_mapping, test_images_only

if __name__ == "__main__":

    # -----------------------------------------------------------------
    # KONFIGURASI EKSPERIMEN
    # -----------------------------------------------------------------

    # "no_class" → No Injection (VLM bebas)
    # "hard"     → With Injection (nama kelas di-inject ke prompt via DINOv2)
    INFERENCE_MODE = "hard"

    # None  → Zero-Shot Baseline
    # <path> → Fine-tuned adapter
    VLM_ADAPTER_PATH = "/mnt/extended-home/dzakaaufa/leakage/model150_65/qwen"

    OUTPUT_CSV = "/mnt/extended-home/dzakaaufa/leakage/evaluation_results/tes150_65/experiment_3b_class_inject150.csv"

    # -----------------------------------------------------------------

    IMAGE_FOLDER = "/mnt/extended-home/dzakaaufa/dataset/all_images_captioning"
    GROUND_TRUTH_FILE = "/mnt/extended-home/dzakaaufa/leakage/model150_65/data150.csv"

    print(f"=== INFERENSI: Mode={INFERENCE_MODE} | Adapter={VLM_ADAPTER_PATH} ===")

    pipeline = BatikMultimodalPipeline(CLASSIFIER_PATH, VLM_ADAPTER_PATH)
    gt_mapping, gt_class_mapping, test_images_whitelist = get_ground_truth_mapping(GROUND_TRUTH_FILE)

    valid_extensions = (".png", ".jpg", ".jpeg", ".webp")
    image_paths = [
        os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(valid_extensions) and normalize_filename(f) in test_images_whitelist
    ]

    print(f"[INFO] Total {len(image_paths)} gambar siap diinferensi.")

    extracted_data = []

    for img_path in tqdm(image_paths, desc="Evaluasi Test Set"):
        try:
            filename = normalize_filename(img_path)
            gt_class = gt_class_mapping.get(filename, "UNKNOWN")

            res = pipeline.process(img_path, mode=INFERENCE_MODE)

            extracted_data.append({
                "image_path":             img_path,
                "reference":              gt_mapping.get(filename, ""),
                "prediction":             res["visual_description"],
                "ground_truth_class":     gt_class,
                "predicted_class":        res["vlm_predicted_class"],
                "dinov2_class":           res["dinov2_class"],
                "dinov2_confidence":      res["dinov2_confidence"],
                "motif_mentioned":        check_motif_mentioned(res["visual_description"], gt_class),
                "used_prompt_mode":       res["used_prompt_mode"],
                "classification_confidence": res["dinov2_confidence"],
            })

        except Exception as e:
            print(f"\n[WARN] Error pada {img_path}: {e}")

    if extracted_data:
        df_results = pd.DataFrame(extracted_data)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df_results.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[SUKSES] Disimpan ke → {OUTPUT_CSV}")
        print(f"\n[INFO] Ringkasan kolom CSV:")
        print(f"  predicted_class  : nama motif yang VLM sebutkan dalam caption")
        print(f"  dinov2_class     : nama motif menurut DINOv2 (None jika no_class)")
        print(f"  motif_mentioned  : apakah gt_class muncul di caption (True/False)")