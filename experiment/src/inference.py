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

CLASSIFIER_PATH = "/mnt/extended-home/dzakaaufa/models/dinov2/best_dinov2_lora"
VLM_ADAPTER_PATH = "/mnt/extended-home/dzakaaufa/experiment/models/qwen2_5-VL-philosophy"
CLASS_MAPPING_PATH = os.path.join(CLASSIFIER_PATH, "class_mapping.json")

# ============================================================
# 1. SETUP & CONFIG (Updated Prompts for VISUAL_PHILOSOPHY)
# ============================================================

# HIGH CONFIDENCE (> 80%) -> Paksa model menyebutkan kelas sesuai format training dan berikan filosofi.
PROMPT_HIGH_CONF = (
    "You are an expert batik annotator and cultural historian. "
    "This batik is confidently identified as the '{kelas}' motif. "
    "Your task is to generate a precise visual description followed by its canonical philosophical meaning.\n"
    "Instructions:\n"
    "1. You MUST start the very first sentence of the first paragraph with exactly this format: "
    "'The batik fabric features the {kelas} motif, characterized by a [specify background color] background and [specify motif colors] motifs.'\n"
    "2. Objectively describe the visual elements: exact shapes of dominant motifs, isen-isen elements, and the layout/pattern arrangement.\n"
    "3. After the visual description, provide a second paragraph that clearly states the traditional philosophical meaning, values, or symbolic representation of the '{kelas}' motif.\n"
    "4. DO NOT hallucinate visual elements. Describe only what is observable in the image.\n"
    "5. Output strictly TWO coherent paragraphs (one for visuals, one for philosophy) without introductory remarks, bullet points, or lists."
)

# MID CONFIDENCE (50% - 80%) -> Sebutkan kelas secara natural, deskripsi visual, dan berikan filosofi.
PROMPT_MID_CONF = (
    "You are an expert batik annotator and cultural historian. "
    "This fabric belongs to the '{kelas}' class based on visual evidence. "
    "Your task is to describe it objectively and provide its philosophical meaning.\n"
    "Instructions:\n"
    "1. Start the first paragraph by stating the background color and the specific colors of the motifs. You can weave the '{kelas}' name into the description naturally.\n"
    "2. Objectively describe the visual elements: exact shapes of dominant motifs, isen-isen elements, and the layout/pattern arrangement.\n"
    "3. Provide a second paragraph that clearly states the traditional philosophical meaning, values, or symbolic representation of the '{kelas}' motif.\n"
    "4. DO NOT hallucinate visual elements. Describe only what is observable.\n"
    "5. Output strictly TWO coherent paragraphs (one for visuals, one for philosophy) without introductory remarks, bullet points, or lists."
)

# LOW CONFIDENCE (< 50%) -> Aman untuk deskripsi visual murni, filosofi disebutkan sebagai kemiripan terdekat.
PROMPT_LOW_CONF = (
    "You are an expert batik annotator and cultural historian. "
    "Note that this fabric merely shares visual similarities with the '{kelas}' motif. "
    "Your task is to describe it visually and provide the typical philosophy associated with its closest match.\n"
    "Instructions:\n"
    "1. Start the first paragraph by stating the background color and the specific colors of the patterns.\n"
    "2. Objectively describe the visual elements: exact shapes of main motifs, isen-isen elements, and layout/pattern arrangement.\n"
    "3. Provide a second paragraph stating the traditional philosophical meaning of the '{kelas}' motif, noting it as the closest thematic match.\n"
    "4. DO NOT hallucinate visual elements. Describe only what is observable.\n"
    "5. Output strictly TWO coherent paragraphs (one for visuals, one for philosophy) without introductory remarks, bullet points, or lists."
)

# ============================================================
# 2. LOAD CLASS MAPPING
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
# 3. UTILITIES
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
# 4. LOAD MODELS
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
# 5. MULTIMODAL PIPELINE
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

    def generate_caption(self, image, motif_label, confidence):
        # 1. Masukkan label motif ke dalam template prompt
        if confidence > 0.80:
            prompt_text = PROMPT_HIGH_CONF.format(kelas=motif_label)
        elif confidence < 0.50:
            prompt_text = PROMPT_LOW_CONF.format(kelas=motif_label)
        else:
            prompt_text = PROMPT_MID_CONF.format(kelas=motif_label)

        # 2. Format messages khusus Qwen
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        # 3. Aplikasikan Chat Template
        text_formatted = self.vlm_processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 4. Kirim ke processor
        inputs = self.vlm_processor(
            text=[text_formatted],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        # max_new_tokens dinaikkan ke 1024 agar menampung 2 paragraf penuh tanpa terpotong
        with torch.no_grad():
            output_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=1024, 
                temperature=0.3,
                do_sample=True
            )

        # 5. Potong token prompt
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]

        caption = self.vlm_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()

        # ========================================================
        # STRATEGI POST-PROCESSING BILINGUAL (ANTI-LEAK & DUPLIKASI)
        # ========================================================
        caption_lower = caption.lower()
        motif_raw = motif_label.lower()
        motif_space = motif_raw.replace("_", " ")

        if motif_raw in caption_lower or motif_space in caption_lower:
            caption = caption.replace("batik'", "batik '")
            return caption
        return caption

    def process(self, image_path):
        raw_image = Image.open(image_path).convert("RGB")

        # STEP 1: Klasifikasi Motif Batik via DINOv2
        motif_name, conf = self.classify_motif(raw_image)

        # STEP 2: Deskripsi Detail Visual & Filosofi via Qwen2.5-VL 
        caption = self.generate_caption(raw_image, motif_name, conf)

        return {
            "file": image_path,
            "analysis": {
                "motif_detected": motif_name,
                "confidence_score": f"{conf:.2%}"
            },
            "generated_text": caption
        }

# ============================================================
# 6. RUN EXECUTION
# ============================================================

if __name__ == "__main__":
    pipeline = BatikMultimodalPipeline(CLASSIFIER_PATH, VLM_ADAPTER_PATH)

    test_img = "/mnt/extended-home/dzakaaufa/dataset/test/parang.jpg"
    
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
        print(f"\nDeskripsi Visual & Makna Filosofis:")
        print("-" * 60)
        
        # Memisahkan paragraf untuk kemudahan membaca di terminal
        paragraphs = hasil["generated_text"].split("\n\n")
        for p in paragraphs:
            wrapped_text = textwrap.fill(p.strip(), width=80)
            print(wrapped_text)
            print() # Jarak antar paragraf
            
        print("=" * 60)
    else:
        print(f"[ERROR] File gambar contoh tidak ditemukan di path: {test_img}")