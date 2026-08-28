import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import torch.nn.functional as F
import textwrap

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

CLASSIFIER_PATH = "/mnt/extended-home/dzakaaufa/leakage/models150/dino"
VLM_ADAPTER_PATH = "/mnt/extended-home/dzakaaufa/models/qwen2.5-vl-7b-eng_inject2"
CLASS_MAPPING_PATH = os.path.join(CLASSIFIER_PATH, "class_mapping.json")

# ============================================================
# 2. PROMPT DEFINITION (Single Prompt)
# ============================================================

PROMPT_VISUAL_EN = (
    "Your task is to generate a precise, objective, and highly accurate visual description of the provided batik image.\n"
    "This fabric belongs to the '{kelas}' class.\n"
    "Instructions:\n"
    "1. Start the first sentence by stating the background color and 2 to 4 dominant motif colors, weaving the '{kelas}' name in naturally.\n"
    "   Example: 'This batik fabric displays visual characteristics of the {kelas} motif, featuring a dark blue background with gold and white motifs.'\n"
    "2. In the second sentence, describe the exact shapes of the main motifs (e.g. specific floral shapes, animals, geometric lines, or abstract curves).\n"
    "3. In the third sentence, describe the pattern layout and arrangement (e.g. diagonal, scattered, repeating grids, symmetrical).\n"
    "4. If isen-isen (small dots or lines filling spaces) are clearly visible, include them in your description.\n"
    "5. Describe only what is directly visible. Do NOT invent colors not clearly present. Do NOT include cultural meanings or philosophical interpretations.\n"
    "6. Output strictly ONE coherent paragraph of 3 to 4 sentences, without any introductory or concluding remarks."
)

# ============================================================
# 3. LOAD CLASS MAPPING
# ============================================================

with open(CLASS_MAPPING_PATH, "r") as f:
    IDX_TO_CLASS = json.load(f)

# Transformasi Gambar untuk DINOv2
dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# 4. UTILITIES
# ============================================================

def clean_adapter_config(adapter_path):
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Bersihkan key usang yang berpotensi memicu error peft
    config.pop("eva_config", None)
    config.pop("auto_mapping", None)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

# ============================================================
# 5. LOAD MODELS
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
# 6. MULTIMODAL PIPELINE
# ============================================================

class BatikMultimodalPipeline:
    def __init__(self, classifier_path, vlm_adapter_path):
        # Tahap 1: DINOv2 + LoRA
        self.classifier = load_classification_model(classifier_path)
        # Tahap 2: Qwen2.5-VL + LoRA
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

    def generate_caption(self, image, motif_label):
        prompt_text = PROMPT_VISUAL_EN.format(kelas=motif_label)

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

        inputs = self.vlm_processor(
            text=[text_formatted],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]

        caption = self.vlm_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()

    
        caption_lower = caption.lower()
        motif_raw = motif_label.lower()
        motif_space = motif_raw.replace("_", " ")

        if motif_raw in caption_lower or motif_space in caption_lower:
            caption = caption.replace("batik'", "batik '")
            return caption
        return caption

    def process(self, image_path):
        raw_image = Image.open(image_path).convert("RGB")

        motif_name, conf = self.classify_motif(raw_image)
        caption = self.generate_caption(raw_image, motif_name)

        return {
            "file": image_path,
            "analysis": {
                "motif_detected": motif_name,
                "confidence_score": f"{conf:.2%}"
            },
            "visual_description": caption
        }

# ============================================================
# 7. RUN EXECUTION
# ============================================================

if __name__ == "__main__":
    pipeline = BatikMultimodalPipeline(CLASSIFIER_PATH, VLM_ADAPTER_PATH)

    test_img = "/mnt/extended-home/dzakaaufa/dataset/test/liong.png"
    
    if os.path.exists(test_img):
        hasil = pipeline.process(test_img)

        print("\n" + "=" * 60)
        print("HASIL ANALISIS MULTIMODAL BATIK (Qwen2.5-VL)")
        print("=" * 60)
        print(f"\nFile Gambar:")
        print(f"   {hasil['file']}")
        print(f"\nHasil Klasifikasi:")
        print(f"   Motif       : {hasil['analysis']['motif_detected']}")
        print(f"   Confidence  : {hasil['analysis']['confidence_score']}")
        print(f"\nDeskripsi Visual:")
        print("-" * 60)
        
        wrapped_caption = textwrap.fill(hasil["visual_description"], width=80)
        print(wrapped_caption)
        print("=" * 60)
    else:
        print(f"[ERROR] File gambar contoh tidak ditemukan di path: {test_img}")