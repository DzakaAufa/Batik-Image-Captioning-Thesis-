import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import open_clip

def evaluate_zero_shot_clip(csv_path):
    print(f"Memuat data prediksi zero-shot dari: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Hanya butuh dua kolom ini!
    image_paths = df['Image Path'].tolist()
    predictions = df['CAPTION_EN'].tolist()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("-> Memuat Model CLIP (ViT-B-32 LAION)...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    clip_scores = []
    
    print("-> Menghitung Keselarasan Gambar dan Prediksi...")
    with torch.no_grad():
        for img_path, pred in tqdm(zip(image_paths, predictions), total=len(predictions)):
            if not os.path.exists(img_path):
                continue
                
            # 1. Preprocess Gambar Batik
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            
            # 2. Tokenisasi Caption Hasil Prediksi Model
            text = tokenizer([str(pred)[:250]]).to(device) 
            
            # 3. Ekstraksi Fitur & Hitung Kedekatan Semantik
            image_features, text_features, _ = model(image, text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).item()
            clip_scores.append(similarity)
            
    avg_score = np.mean(clip_scores) if clip_scores else 0.0
    
    print("\n" + "="*40)
    print(" HASIL EVALUASI ZERO-SHOT (REFERENCE-FREE)")
    print("="*40)
    print(f"Total Gambar diuji     : {len(clip_scores)}")
    print(f"Rata-rata CLIPScore    : {avg_score:.4f}")
    print("📌 Kesimpulan: Semakin mendekati 1.0, model semakin akurat")
    print("   mencocokkan visual kain dengan deskripsi teksnya.")
    print("="*40)

if __name__ == "__main__":
    CSV_ZERO_SHOT = "/mnt/extended-home/dzakaaufa/dataset/caption/zeroshot_qwen.csv" 
    evaluate_zero_shot_clip(CSV_ZERO_SHOT)