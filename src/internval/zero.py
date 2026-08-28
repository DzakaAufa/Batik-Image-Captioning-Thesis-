import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# KONFIGURASI
# ============================================================
IMAGE_DIR    = "/mnt/extended-home/dzakaaufa/dataset/baru_captioning"
METADATA_CSV = "/mnt/extended-home/dzakaaufa/dataset/caption/batik_dataset_baru.csv"
OUTPUT_CSV   = "/mnt/extended-home/dzakaaufa/dataset/caption/internvl_baru.csv"
LANG         = "english"   # "indonesia" atau "english"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# PROMPT: 100% Visual Grounded
# ============================================================
PROMPT_ID = (
    "Deskripsikan kain batik ini secara objektif dan faktual berdasarkan apa yang benar-benar terlihat di gambar. "
    "Kain ini teridentifikasi sebagai kelompok '{kelas}'.\n"
    "Instruksi:\n"
    "1. Sebutkan warna dasar latar belakang kain dan warna-warna spesifik yang digunakan pada motif.\n"
    "2. Deskripsikan bentuk objek motif utama secara detail (misal: apakah menyerupai tumbuhan, hewan, garis geometris, atau bentuk abstrak).\n"
    "3. Jelaskan pola penyusunan motif tersebut di atas kain (misal: apakah menyebar merata, diagonal, berulang rapi, atau asimetris).\n"
    "4. JANGAN mengarang informasi yang tidak terlihat secara visual.\n"
    "5. Rangkai menjadi SATU paragraf berbahasa Indonesia yang baku, logis, dan mengalir natural."
)

PROMPT_EN = (
    "Describe this batik fabric objectively based ONLY on exact visual evidence. "
    "This fabric belongs to the '{kelas}' class.\n"
    "Instructions:\n"
    "1. State the background color and the specific colors of the motifs.\n"
    "2. Describe the exact shapes of the main motifs (e.g., specific floral shapes, animals, geometric lines, or abstract curves).\n"
    "3. Describe the layout/pattern arrangement (e.g., diagonal, scattered, repeating grids, symmetrical).\n"
    "4. DO NOT hallucinate. Do not name non-colors as colors.\n"
    "5. Write strictly ONE coherent paragraph."
)

def get_prompt(kelas: str) -> str:
    template = PROMPT_ID if LANG == "indonesia" else PROMPT_EN
    return template.format(kelas=kelas)

# ============================================================
# INTERNVL UTILITIES (Preprocessing & Transformation)
# ============================================================
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

# ============================================================
# MODEL LOADING & GENERATION
# ============================================================
def load_internvl():
    print("Loading InternVL3.5-8B...")
    # Menggunakan model InternVL 3.5 versi 8B
    model_name = "OpenGVLab/InternVL3_5-8B"

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
    ).to(DEVICE)
    
    model.eval()
    return tokenizer, model

def generate_internvl(image: Image.Image, kelas: str, tokenizer, model) -> str:
    images = dynamic_preprocess(image, max_num=6)
    transform = build_transform()
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values).to(torch.bfloat16).to(DEVICE)
    num_patches = pixel_values.size(0)

    prompt_text = get_prompt(kelas)
    prompt = f"<image>\n" * num_patches + prompt_text
    
    generation_config = dict(
        max_new_tokens=400, 
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.05
    )
    
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
            
    return response.replace("<img>", "").replace("</img>", "").strip()

# ============================================================
# MAIN PROCESS
# ============================================================
def main():
    df = pd.read_csv(METADATA_CSV)

    # Validasi kolom
    required_cols = {"Image Path", "Nama", "CLASS"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Kolom berikut tidak ditemukan di CSV: {required_cols - set(df.columns)}")

    print(f"Bahasa   : {LANG}")
    print(f"Device   : {DEVICE}")
    print(f"Total    : {len(df)} gambar\n")

    tokenizer, model = load_internvl()

    # Checkpoint logic
    if os.path.exists(OUTPUT_CSV):
        done_df = pd.read_csv(OUTPUT_CSV)
        done_paths = set(done_df["Image Path"].tolist())
        df = df[~df["Image Path"].isin(done_paths)]
        print(f"Checkpoint ditemukan. Melanjutkan {len(df)} gambar tersisa...\n")
    else:
        done_df = pd.DataFrame()

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Captioning InternVL 3.5"):
        img_path_raw = row["Image Path"]
        img_path = img_path_raw if os.path.isabs(img_path_raw) else os.path.join(IMAGE_DIR, img_path_raw)
        
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            caption = generate_internvl(image, str(row["CLASS"]), tokenizer, model)
            
            results.append({
                "Image Path" : img_path_raw,
                "Nama"       : row["Nama"],
                "CLASS"      : row["CLASS"],
                "CAPTION_EN" : caption,
            })
        except Exception as e:
            print(f"\n[ERROR] {img_path}: {e}")

        # Simpan bertahap setiap 10 gambar
        if len(results) % 10 == 0 and results:
            checkpoint_df = pd.concat([done_df, pd.DataFrame(results)], ignore_index=True)
            checkpoint_df.to_csv(OUTPUT_CSV, index=False)

    # Simpan hasil akhir
    final_df = pd.concat([done_df, pd.DataFrame(results)], ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Selesai! Total: {len(final_df)} disimpan di {OUTPUT_CSV}")

if __name__ == "__main__":
    main()