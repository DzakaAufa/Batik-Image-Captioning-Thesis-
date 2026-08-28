import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

# ── Patch torch < 2.4 ────────────────────────────────────────────────────────
if not hasattr(torch.nn.Module, "set_submodule"):
    def set_submodule(self, target, module):
        atoms = target.split(".")
        name = atoms.pop(-1)
        mod = self
        for item in atoms:
            mod = getattr(mod, item)
        setattr(mod, name, module)
    torch.nn.Module.set_submodule = set_submodule

PROMPT_VISUAL_EN = (
    "You are an expert batik annotator. "
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

def load_model(model_name: str, use_4bit: bool = True):
    """Muat Qwen2.5-VL dengan opsional kuantisasi 4-bit."""
    print(f"Memuat model: {model_name}  (4-bit={use_4bit})")

    processor = AutoProcessor.from_pretrained(model_name, max_pixels=512 * 512)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if has_bf16 else torch.float16

    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )

    model.eval()
    print("Model siap.")
    return model, processor, dtype

@torch.inference_mode()
def generate_caption(
    model,
    processor,
    dtype,
    image: Image.Image,
    batik_class: str,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> str:
    prompt_text = PROMPT_VISUAL_EN.format(kelas=batik_class)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=dtype):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    caption = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    return caption

def run_zero_shot(
    csv_path: str,
    image_dir: str,
    output_csv: str,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    use_4bit: bool = True,
    max_new_tokens: int = 256,
    resume: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = pd.read_csv(csv_path)
    required_cols = {"Image Path", "CLASS"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Kolom berikut tidak ada di CSV: {missing}")

    already_done: set = set()
    if resume and os.path.isfile(output_csv):
        done_df = pd.read_csv(output_csv)
        already_done = set(done_df["Image Path"].tolist())
        print(f"Resume: {len(already_done)} gambar sudah diproses sebelumnya.")

    model, processor, dtype = load_model(model_name, use_4bit=use_4bit)

    results = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Zero-Shot Captioning"):
        name_batik   = str(row["Nama"]) 
        img_path_raw = str(row["Image Path"])
        batik_class  = str(row["CLASS"])

        if img_path_raw in already_done:
            skipped += 1
            continue

        img_path = (
            img_path_raw
            if os.path.isabs(img_path_raw)
            else os.path.join(image_dir, img_path_raw)
        )

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\nGagal buka gambar: {img_path} — {e}")
            caption = "ERROR: image not found"
        else:
            try:
                caption = generate_caption(
                    model, processor, dtype, image,
                    batik_class=batik_class,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
            except Exception as e:
                print(f"\nError saat inferensi {img_path}: {e}")
                caption = f"ERROR: {e}"

        results.append({
            "Image Path":  img_path_raw,
            "name":        name_batik,
            "CLASS":       batik_class,
            "caption_en":  caption,
        })

        if len(results) % 10 == 0:
            _append_or_create(output_csv, results, already_done)
            results = []

    # Flush sisa
    if results:
        _append_or_create(output_csv, results, already_done)

    print(f"\nSelesai!")
    print(f"   Output : {output_csv}")
    print(f"   Di-skip: {skipped} (sudah ada)")


def _append_or_create(output_csv: str, new_rows: list, already_done: set):
    """Tambahkan baris baru ke CSV output; buat jika belum ada."""
    new_df = pd.DataFrame(new_rows)
    if os.path.isfile(output_csv):
        new_df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        new_df.to_csv(output_csv, index=False)
    already_done.update(new_df["Image Path"].tolist())

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot image captioning batik dengan Qwen2.5-VL"
    )
    parser.add_argument(
        "--csv",
        default="/mnt/extended-home/dzakaaufa/dataset/caption/path_data_B.csv",
        help="Path CSV input (kolom: Image Path, CLASS)",
    )
    parser.add_argument(
        "--image_dir",
        default="/mnt/extended-home/dzakaaufa/dataset/baru_captioning",
        help="Root direktori gambar",
    )
    parser.add_argument(
        "--output",
        default="/mnt/extended-home/dzakaaufa/dataset/caption/data_B_zero_inject1.csv",
        help="Path CSV output hasil caption",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Nama model Qwen (HuggingFace hub atau path lokal)",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Nonaktifkan kuantisasi 4-bit (butuh lebih banyak VRAM)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Jumlah maksimum token yang digenerate per caption",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Mulai dari awal meskipun output CSV sudah ada",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_zero_shot(
        csv_path=args.csv,
        image_dir=args.image_dir,
        output_csv=args.output,
        model_name=args.model,
        use_4bit=not args.no_4bit,
        max_new_tokens=args.max_tokens,
        resume=not args.no_resume,
    )