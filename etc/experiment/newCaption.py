import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

# ============================================================
# KONFIGURASI — sesuaikan bagian ini saja
# ============================================================
IMAGE_DIR    = "/mnt/extended-home/dzakaaufa/dataset/gambar"
METADATA_CSV = "/mnt/extended-home/dzakaaufa/dataset/caption/batik_dataset.csv"
OUTPUT_CSV   = "/mnt/extended-home/dzakaaufa/dataset/caption/dataset_captioned_internvl.csv"
LANG         = "english"   # "indonesia" atau "english"

# Pilih salah satu model:
# "blip2"      → BLIP-2 Flan-T5-XL, ~12GB VRAM, output English
# "internvl"   → InternVL2-8B, ~16GB VRAM, output Indonesia/English (TERBAIK)
# "llava"      → LLaVA-1.5-7B, ~14GB VRAM, output Indonesia/English
# "moondream"  → Moondream2, ~4GB VRAM, output English (paling ringan)
MODEL_CHOICE = "internvl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------------------ #
#  PROMPT per model
# ------------------------------------------------------------------ #
# ============================================================
# REVISI PROMPT: 100% Visual Grounded, Bebas Halusinasi
# ============================================================
PROMPT_ID = (
    "Deskripsikan kain batik ini secara objektif dan faktual berdasarkan apa yang benar-benar terlihat di gambar. "
    "Kain ini teridentifikasi sebagai kelompok '{kelas}'.\n"
    "Instruksi:\n"
    "1. Sebutkan warna dasar latar belakang kain dan warna-warna spesifik yang digunakan pada motif.\n"
    "2. Deskripsikan bentuk objek motif utama secara detail (misal: apakah menyerupai tumbuhan, hewan, garis geometris, atau bentuk abstrak).\n"
    "3. Jelaskan pola penyusunan motif tersebut di atas kain (misal: apakah menyebar merata, diagonal, berulang rapi, atau asimetris).\n"
    "4. JANGAN mengarang informasi yang tidak terlihat secara visual (jangan mengarang teknik aneh atau filosofi yang tidak pasti).\n"
    "5. Rangkai menjadi SATU paragraf berbahasa Indonesia yang baku, logis, dan mengalir natural."
)

PROMPT_EN = (
    "Describe this batik fabric objectively based ONLY on exact visual evidence. "
    "This fabric belongs to the '{kelas}' class.\n"
    "Instructions:\n"
    "1. State the background color and the specific colors of the motifs.\n"
    "2. Describe the exact shapes of the main motifs (e.g., specific floral shapes, animals, geometric lines, or abstract curves).\n"
    "3. Describe the layout/pattern arrangement (e.g., diagonal, scattered, repeating grids, symmetrical).\n"
    "4. DO NOT hallucinate. Do not name non-colors as colors. Do not invent philosophical meanings or clothing uses.\n"
    "5. Write strictly ONE coherent paragraph."
)

def get_prompt(nama: str, kelas: str) -> str:
    template = PROMPT_ID if LANG == "indonesia" else PROMPT_EN
    return template.format(nama=nama, kelas=kelas)

# Load Model #
def load_blip2():
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print("Loading BLIP-2 Flan-T5-XL...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    return processor, model

def load_internvl():
    from transformers import AutoTokenizer
    import torch

    print("Loading InternVL2-8B...")
    model_name = "OpenGVLab/InternVL2-8B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return tokenizer, model

def load_llava():
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    print("Loading LLaVA-1.5-7B...")
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
    )
    print(f"Menggunakan Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"Nama GPU: {torch.cuda.get_device_name(0)}")
    
    model.eval()
    return processor, model

def load_moondream():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading Moondream2...")
    model_id = "vikhyatk/moondream2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model

def generate_blip2(image: Image.Image, nama: str, kelas: str, processor, model) -> str:
    prompt = get_prompt(nama, kelas)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=300,
            num_beams=5,
            repetition_penalty=2.5,
            no_repeat_ngram_size=4,
            length_penalty=1.2,
        )
    return processor.decode(ids[0], skip_special_tokens=True).strip()

def build_transform():
    import torchvision.transforms as T
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

def generate_internvl(image: Image.Image, nama: str, kelas: str, tokenizer, model) -> str:
    # 1. Gunakan Dynamic Preprocess agar model bisa melihat resolusi tinggi
    images = dynamic_preprocess(image, max_num=6)
    
    # 2. Transform dan Stack menjadi Batch Tiling
    transform = build_transform()
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values).to(torch.bfloat16).to(DEVICE) # bfloat16 lebih stabil untuk InternVL
    num_patches = pixel_values.size(0)

    # 3. Format Prompt Khusus InternVL dengan jumlah <image> token sesuai tile
    prompt_text = get_prompt(nama, kelas)
    prompt = f"<image>\n" * num_patches + prompt_text
    
    # 4. Tambahkan Repetition Penalty untuk mencegah Looping/Halusinasi
    generation_config = dict(
        max_new_tokens=400, 
        do_sample=True,          # Beralih ke sampling agar bahasa lebih natural
        temperature=0.2,         # Suhu rendah agar tetap faktual dan tidak berhalusinasi
        top_p=0.9,               # Membatasi token aneh
        repetition_penalty=1.05  # Turunkan penalti agar tidak memicu kata seperti "pengeboran"
    )
    
    with torch.no_grad():
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)
            
    # Bersihkan output dari kemungkinan token nyasar
    return response.replace("<img>", "").replace("</img>", "").strip()

def generate_llava(image: Image.Image, nama: str, kelas: str, processor, model) -> str:
    prompt_text = f"[INST] <image>\n{get_prompt(nama, kelas)} [/INST]"
    inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=300,
            repetition_penalty=2.5,
            no_repeat_ngram_size=4,
        )
    full = processor.decode(ids[0], skip_special_tokens=True)
    if "[/INST]" in full:
        full = full.split("[/INST]")[-1]
    return full.strip()

def generate_moondream(image: Image.Image, nama: str, kelas: str, tokenizer, model) -> str:
    enc = model.encode_image(image)
    prompt = get_prompt(nama, kelas)
    answer = model.answer_question(enc, prompt, tokenizer)
    return answer.strip()

LOADERS = {
    "blip2"     : load_blip2,
    "internvl"  : load_internvl,
    "llava"     : load_llava,
    "moondream" : load_moondream,
}

GENERATORS = {
    "blip2"     : generate_blip2,
    "internvl"  : generate_internvl,
    "llava"     : generate_llava,
    "moondream" : generate_moondream,
}


# ------------------------------------------------------------------ #
#  MAIN
# ------------------------------------------------------------------ #
def main():
    df = pd.read_csv(METADATA_CSV)

    required_cols = {"Image Path", "Nama", "CLASS"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan di CSV: {missing}")

    print(f"Model    : {MODEL_CHOICE.upper()}")
    print(f"Bahasa   : {LANG}")
    print(f"Device   : {DEVICE}")
    print(f"Total    : {len(df)} gambar\n")

    # Load model sesuai pilihan
    load_fn     = LOADERS[MODEL_CHOICE]
    generate_fn = GENERATORS[MODEL_CHOICE]
    model_obj_a, model_obj_b = load_fn()

    # Checkpoint
    if os.path.exists(OUTPUT_CSV):
        done_df = pd.read_csv(OUTPUT_CSV)
        done_paths = set(done_df["Image Path"].tolist())
        df = df[~df["Image Path"].isin(done_paths)]
        print(f"Checkpoint ditemukan. Melanjutkan {len(df)} gambar yang belum selesai...\n")
    else:
        done_df = pd.DataFrame()

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Captioning [{MODEL_CHOICE}]"):
        img_path_raw = row["Image Path"]
        img_path = img_path_raw if os.path.isabs(img_path_raw) \
                   else os.path.join(IMAGE_DIR, img_path_raw)
        nama  = str(row["Nama"])
        kelas = str(row["CLASS"])

        if not os.path.exists(img_path):
            print(f"\n  [SKIP] File tidak ditemukan: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            caption = generate_fn(image, nama, kelas, model_obj_a, model_obj_b)
        except Exception as e:
            print(f"\n  [ERROR] {img_path}: {e}")
            caption = ""

        if caption:
            results.append({
                "Image Path" : img_path_raw,
                "Nama"       : nama,
                "CLASS"      : kelas,
                "CAPTION_ID" : caption,
            })

        # Auto-save setiap 10 gambar
        if len(results) % 10 == 0 and results:
            checkpoint_df = pd.concat([done_df, pd.DataFrame(results)], ignore_index=True)
            checkpoint_df.to_csv(OUTPUT_CSV, index=False)

    final_df = pd.concat([done_df, pd.DataFrame(results)], ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Selesai! Total caption: {len(final_df)}")
    print(f"📁 Disimpan di: {OUTPUT_CSV}")
    print("\n--- Preview 2 caption pertama ---")
    for i, row in final_df.head(2).iterrows():
        print(f"\n[{i+1}] {row['Nama']} ({row['CLASS']})")
        print(f"     {row['CAPTION_ID'][:300]}...")

if __name__ == "__main__":
    main()