import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
import torchvision.transforms as T

from transformers import AutoTokenizer, AutoModelForImageClassification, AutoModel
from peft import PeftModel, PeftConfig

# ============================================================
# 1. SETUP
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. PATH CONFIG
# ============================================================

CLASSIFIER_PATH = "/mnt/extended-home/dzakaaufa/models/dinov2/best_dinov2_lora"
VLM_ADAPTER_PATH_EN = "/mnt/extended-home/dzakaaufa/models/internvl2-8b-eng"
CLASS_MAPPING_PATH = os.path.join(CLASSIFIER_PATH, "class_mapping.json")

# ============================================================
# 3. LOAD CLASS MAPPING
# ============================================================

with open(CLASS_MAPPING_PATH, "r") as f:
    IDX_TO_CLASS = json.load(f)

# ============================================================
# 4. TRANSFORMS
# ============================================================

dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_internvl_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_image_internvl(image_path, input_size=448):
    image = Image.open(image_path).convert("RGB")
    transform = build_internvl_transform(input_size)
    return transform(image).unsqueeze(0)

# ============================================================
# 5. CLEAN ADAPTER CONFIG
# ============================================================

def clean_adapter_config(adapter_path):
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, "r") as f:
        config = json.load(f)
    config.pop("eva_config", None)
    config.pop("alora_invocation_tokens", None)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

# ============================================================
# 6. LOAD DINOv2 CLASSIFIER
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
    print("Classifier berhasil dimuat!\n")
    return model

# ============================================================
# 7. LOAD InternVL2 (English only)
# ============================================================

def load_vlm_model(adapter_path_en, base_model_name="OpenGVLab/InternVL2-8B"):
    print("1. Memuat Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        use_fast=False
    )

    print(f"2. Memuat Base Model: {base_model_name} (bfloat16)...")
    base_model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    print("3. Memasang PATCH img_context_token_id...")
    img_context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    base_model.config.img_context_token_id = img_context_id
    if not hasattr(base_model, "img_context_token_id"):
        setattr(base_model, "img_context_token_id", img_context_id)

    print("4. Memperbaiki Config Adapter...")
    clean_adapter_config(adapter_path_en)

    print(f"5. Memasang Adapter Bahasa Inggris dari: {adapter_path_en}...")
    model = PeftModel.from_pretrained(base_model, adapter_path_en)

    model.eval()
    print("InternVL2 beserta adapter Inggris berhasil dimuat!\n")
    return tokenizer, model

# ============================================================
# 8. MULTIMODAL PIPELINE
# ============================================================

class BatikMultimodalPipeline:

    PROMPT_EN = (
        "Describe this batik fabric objectively based ONLY on exact visual evidence. "
        "This fabric belongs to the '{motif}' class.\n"
        "Instructions:\n"
        "1. State the background color and the specific colors of the motifs.\n"
        "2. Describe the exact shapes of the main motifs (e.g., specific floral shapes, animals, geometric lines, or abstract curves).\n"
        "3. Describe the layout/pattern arrangement (e.g., diagonal, scattered, repeating grids, symmetrical).\n"
        "4. DO NOT hallucinate. Do not name non-colors as colors. Do not invent philosophical meanings.\n"
        "5. Write strictly ONE coherent paragraph."
    )

    def __init__(self, classifier_path, vlm_adapter_path_en):
        self.classifier = load_classification_model(classifier_path)
        self.tokenizer, self.vlm_model = load_vlm_model(vlm_adapter_path_en)

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def classify_motif(self, image):
        img_tensor = dinov2_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            probs = F.softmax(outputs.logits, dim=1)[0]
            confidence, class_idx = torch.max(probs, dim=0)
            predicted_class = IDX_TO_CLASS[str(class_idx.item())]
        return predicted_class, confidence.item()

    # ========================================================
    # CAPTIONING
    # ========================================================

    def generate_caption(self, image_path, motif_label):
        device = next(self.vlm_model.parameters()).device

        try:
            pixel_values = load_image_internvl(image_path).to(device).to(torch.bfloat16)
        except Exception as e:
            return f"Error memproses gambar: {e}"

        prompt = self.PROMPT_EN.format(motif=motif_label)
        question = f"<image>\n{prompt}"

        generation_config = dict(
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.2,
        )

        with torch.no_grad():
            response, _ = self.vlm_model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                history=None,
                return_history=True
            )

        return response.strip()

    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process(self, image_path):
        raw_image = Image.open(image_path).convert("RGB")

        print("Mengklasifikasi motif...")
        motif_name, conf = self.classify_motif(raw_image)
        print(f"Motif terdeteksi: {motif_name} ({conf:.2%})")

        print("Generating English caption...")
        caption_en = self.generate_caption(image_path, motif_name)

        return {
            "file": image_path,
            "analysis": {
                "motif_detected": motif_name,
                "confidence_score": f"{conf:.2%}"
            },
            "caption_en": caption_en,
        }

# ============================================================
# 9. RUN PIPELINE
# ============================================================

if __name__ == "__main__":

    pipeline = BatikMultimodalPipeline(
        classifier_path     = CLASSIFIER_PATH,
        vlm_adapter_path_en = VLM_ADAPTER_PATH_EN,
    )

    test_img = "/mnt/extended-home/dzakaaufa/dataset/test/1778644600450-iqdtbi.png"
    hasil = pipeline.process(test_img)

    import textwrap

    print("\n" + "=" * 60)
    print("HASIL ANALISIS MULTIMODAL BATIK")
    print("=" * 60)

    print(f"\nFile Gambar\n   {hasil['file']}")

    print(f"\nHasil Klasifikasi")
    print(f"   Motif      : {hasil['analysis']['motif_detected']}")
    print(f"   Confidence : {hasil['analysis']['confidence_score']}")

    print(f"\nVisual Description (English)")
    print("-" * 60)
    print(textwrap.fill(hasil["caption_en"], width=80))

    print("\n" + "=" * 60)