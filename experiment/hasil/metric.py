import os
import re
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pycocoevalcap.spice.spice import Spice

import evaluate
import open_clip
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel

# Setup Java Path untuk SPICE
java_path = "/mnt/extended-home/dzakaaufa/java_libs/amazon-corretto-8.482.08.1-linux-x64/bin/java"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['JAVA_HOME'] = os.path.dirname(os.path.dirname(java_path))
os.environ['PATH'] = os.path.dirname(java_path) + os.pathsep + os.environ['PATH']
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# UTILITIES & TEXT SPLITTER
# ============================================================

def normalize_text(text):
    text = str(text).lower().strip()
    text = text.replace("_", " ")          
    text = re.sub(r"[^\w\s]", " ", text)  
    text = re.sub(r"\s+", " ", text).strip()
    return text

def check_motif_mentioned(pred_caption, gt_class):
    return normalize_text(gt_class) in normalize_text(pred_caption)

def split_visual_philosophy(text, experiment_type):
    """
    Memisahkan komponen teks visual dan filosofi berdasarkan double newline.
    Sama dengan logika pada batch inference.
    """
    text = str(text).strip()
    if experiment_type == "PURE_PHILOSOPHY":
        return "", text
    
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) >= 2:
        visual_part = parts[0]
        philosophy_part = "\n\n".join(parts[1:])
        return visual_part, philosophy_part
    
    return text, text

def clean_for_spice(text):
    sentences = str(text).split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    shortened = ". ".join(sentences[:3])
    if shortened and not shortened.endswith('.'):
        shortened += "."
    return shortened

# ============================================================
# NEW ADVANCED METRICS FOR SKRIPSI
# ============================================================

def calculate_keyword_coverage(philosophy_cores, predictions):
    """
    [PERBAIKAN] Menghitung persentase keyword utama dari philosophy_core 
    yang berhasil termuat di dalam hasil prediksi (pred_philos).
    """
    scores = []
    for core, pred in zip(philosophy_cores, predictions):
        # Jika NaN atau kosong (seperti pada Dataset A), dilewati agar tidak merusak rata-rata
        if pd.isna(core) or str(core).strip() == "" or str(core).lower() == "nan":
            continue
            
        # Pecah keyword berdasarkan koma atau titik koma, lalu bersihkan whitespace
        keywords = [kw.strip().lower() for kw in re.split(r'[,;]+', str(core)) if kw.strip()]
        if not keywords:
            continue
            
        pred_lower = str(pred).lower()
        
        # Hitung berapa keyword yang muncul di dalam teks prediksi
        matched_count = sum(1 for kw in keywords if kw in pred_lower)
        scores.append(matched_count / len(keywords))
        
    # Return rata-rata coverage, jika semua data kosong (Dataset A semua) beri default 0.0
    return np.mean(scores) if scores else 0.0

def calculate_semantic_similarity(references, predictions, device):
    """Menghitung kedekatan makna/konteks menggunakan Sentence Embedding MiniLM."""
    print("    -> Menghitung Semantic Similarity (MiniLM)...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device).eval()
    
    similarities = []
    for ref, pred in zip(references, predictions):
        if not ref.strip() or not pred.strip():
            similarities.append(0.0)
            continue
        inputs = tokenizer([ref, pred], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            similarity = (embeddings[0] @ embeddings[1]).item()
            similarities.append(max(0.0, similarity))
    return np.mean(similarities)

def calculate_philosophy_consistency(predictions):
    """Mengukur konsistensi filosofi dengan mendeteksi visual leakage di paragraf filosofi."""
    visual_leakage_words = ['background', 'color', 'fabric', 'cloth', 'motif features', 'colored', 'dominant color', 'pixels']
    scores = []
    for pred in predictions:
        score = 1.0
        pred_lower = pred.lower()
        for word in visual_leakage_words:
            if word in pred_lower:
                score -= 0.2
        scores.append(max(0.0, score))
    return np.mean(scores)

# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def compute_batik_metrics_clustered(csv_path, experiment_type="VISUAL_PHILOSOPHY"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File prediksi tidak ditemukan di: {csv_path}")

    print(f"\n[INFO] Membaca data hasil prediksi: {csv_path}")
    df = pd.read_csv(csv_path)

    # [PERBAIKAN] Menyertakan 'philosophy_core' ke dalam list kolom wajib
    required_cols = ['image_path', 'reference', 'prediction', 'ground_truth_class', 'predicted_class', 'philosophy_core']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV wajib memiliki kolom: {required_cols}")

    image_paths = df['image_path'].tolist()
    references_full = df['reference'].tolist()
    predictions_full = df['prediction'].tolist()
    philosophy_cores = df['philosophy_core'].tolist() # Ambil array mentah core keywords

    # --- PROSES PEMISAHAN KLASTER TEKS ---
    ref_visuals, ref_philos = [], []
    pred_visuals, pred_philos = [], []

    for ref, pred in zip(references_full, predictions_full):
        rv, rp = split_visual_philosophy(ref, experiment_type)
        pv, pp = split_visual_philosophy(pred, experiment_type)
        ref_visuals.append(rv)
        ref_philos.append(rp)
        pred_visuals.append(pv)
        pred_philos.append(pp)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    bertscore_metric = evaluate.load("bertscore")

    print(f"\n[Mulai] Menghitung Metrik Evaluasi Klaster [{experiment_type}]...")
    print("-" * 60)

    # ==========================================================
    # A. KLASTER KELAS (Classification Accuracy & Mention Rate)
    # ==========================================================
    print(" -> [1/3] Memproses Bagian Kelas...")
    gt_classes = df['ground_truth_class'].astype(str).str.lower().str.strip()
    pred_classes = df['predicted_class'].astype(str).str.lower().str.strip()
    
    results['Classification_Accuracy'] = accuracy_score(gt_classes, pred_classes)
    
    mentioned_count = sum(check_motif_mentioned(pred, gt) for pred, gt in zip(predictions_full, df['ground_truth_class']))
    results['Motif_Mention_Rate'] = mentioned_count / len(predictions_full) if predictions_full else 0.0

    # ==========================================================
    # B. KLASTER FILOSOFI (Evaluasi pada Paragraf Filosofi)
    # ==========================================================
    print(" -> [2/3] Memproses Bagian Filosofi...")
    
    # Philosophy BERTScore
    bs_phil = bertscore_metric.compute(predictions=pred_philos, references=ref_philos, model_type="xlm-roberta-base", lang="multilingual")
    results['Philosophy_BERTScore'] = np.mean(bs_phil["f1"])
    
    # Philosophy Semantic Similarity
    results['Philosophy_Semantic_Similarity'] = calculate_semantic_similarity(ref_philos, pred_philos, device)
    
    # [PERBAIKAN] Menghitung Keyword Coverage berbasis tabel philosophy_core asli
    results['Philosophy_Keyword_Coverage'] = calculate_keyword_coverage(philosophy_cores, pred_philos)
    
    # Philosophy Consistency
    results['Philosophy_Consistency'] = calculate_philosophy_consistency(pred_philos)

    # ==========================================================
    # C. KLASTER VISUAL (Evaluasi pada Paragraf Visual)
    # ==========================================================
    if experiment_type == "VISUAL_PHILOSOPHY":
        print(" -> [3/3] Memproses Bagian Visual...")
        
        # Visual BERTScore
        bs_vis = bertscore_metric.compute(predictions=pred_visuals, references=ref_visuals, model_type="xlm-roberta-base", lang="multilingual")
        results['Visual_BERTScore'] = np.mean(bs_vis["f1"])
        
        # METEOR Score
        meteor_metric = evaluate.load("meteor")
        meteor_results = meteor_metric.compute(predictions=pred_visuals, references=ref_visuals)
        results['METEOR'] = meteor_results["meteor"]
        
        # SPICE Score
        try:
            gts_spice = {str(i): [clean_for_spice(rv)] for i, rv in enumerate(ref_visuals)}
            res_spice = {str(i): [clean_for_spice(pv)] for i, pv in enumerate(pred_visuals)}
            spice_scorer = Spice()
            spice_score, _ = spice_scorer.compute_score(gts_spice, res_spice)
            results['SPICE'] = spice_score
        except Exception as e:
            print(f"    [WARN] SPICE gagal: {e}")
            results['SPICE'] = 0.0
            
        # CLIPScore
        try:
            model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            model_clip = model_clip.to(device).eval()
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
            clip_scores = []
            with torch.no_grad():
                for img_path, pv in zip(image_paths, pred_visuals):
                    if not os.path.exists(img_path) or not pv.strip():
                        continue
                    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    tokens_raw = tokenizer([str(pv)])
                    text = tokens_raw.to(device)
                    
                    img_feat = model_clip.encode_image(image)
                    txt_feat = model_clip.encode_text(text)
                    img_feat /= img_feat.norm(dim=-1, keepdim=True)
                    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                    
                    clip_scores.append((img_feat @ txt_feat.T).item())
            results['CLIPScore'] = np.mean(clip_scores) if clip_scores else 0.0
        except Exception as e:
            print(f"    [WARN] CLIPScore gagal: {e}")
            results['CLIPScore'] = 0.0
            
    else:
        print(" -> [3/3] Mode PURE_PHILOSOPHY aktif. Metriks visual dilewati.")
        results['Visual_BERTScore'] = 0.0
        results['METEOR'] = 0.0
        results['SPICE'] = 0.0
        results['CLIPScore'] = 0.0

    # ==========================================================
    # GRAND METRICS AGGREGATION
    # ==========================================================
    
    # 1. Perhitungan Rata-Rata Kualitas Filosofi
    results['Philosophy_Quality_Avg'] = np.mean([
        results['Philosophy_BERTScore'],
        results['Philosophy_Semantic_Similarity'],
        results['Philosophy_Keyword_Coverage']
    ])
    
    # 2. Perhitungan Rata-Rata Kualitas Visual
    if experiment_type == "VISUAL_PHILOSOPHY":
        results['Visual_Quality_Avg'] = np.mean([
            results['CLIPScore'],
            results['SPICE'],
            results['Visual_BERTScore']
        ])
    else:
        results['Visual_Quality_Avg'] = 0.0

    # ==========================================================
    # PRINT TABULAR REPORT (SIAP SIDANG)
    # ==========================================================
    print("\n" + "=" * 60)
    print("             HASIL EVALUASI KLASTER SKRIPSI")
    print("=" * 60)
    print(f" Tipe Eksperimen   : {experiment_type}")
    print(f" Jumlah Sampel     : {len(predictions_full)}")
    print("-" * 60)
    print(f" [KLASTER KELAS]")
    print(f"  > Classification Accuracy            : {results['Classification_Accuracy']:.4f}")
    print(f"  > Motif Mention Rate                 : {results['Motif_Mention_Rate']:.4f}")
    print("-" * 60)
    print(f" [KLASTER VISUAL]")
    print(f"  > CLIPScore                          : {results['CLIPScore']:.4f}")
    print(f"  > SPICE                              : {results['SPICE']:.4f}")
    print(f"  > Visual BERTScore                   : {results['Visual_BERTScore']:.4f}")
    print(f"  > METEOR                             : {results['METEOR']:.4f}")
    print(f"  * VISUAL_QUALITY_AVG                 : {results['Visual_Quality_Avg']:.4f}")
    print("-" * 60)
    print(f" [KLASTER FILOSOFI]")
    print(f"  > Philosophy BERTScore               : {results['Philosophy_BERTScore']:.4f}")
    print(f"  > Philosophy Semantic Similarity     : {results['Philosophy_Semantic_Similarity']:.4f}")
    print(f"  > Philosophy Keyword Coverage        : {results['Philosophy_Keyword_Coverage']:.4f}")
    print(f"  > Philosophy Consistency             : {results['Philosophy_Consistency']:.4f}")
    print(f"  * PHILOSOPHY_QUALITY_AVG             : {results['Philosophy_Quality_Avg']:.4f}")
    print("=" * 60)

    # Simpan JSON rekapitulasi baru
    save_path = os.path.join(os.path.dirname(csv_path), f"metrics_clustered_{experiment_type.lower()}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Sukses menyimpan laporan klaster skripsi ke → {save_path}\n")

    return results

if __name__ == "__main__":
    # Path file .csv dari skrip predict sebelumnya
    PREDICTIONS_CSV = "/mnt/extended-home/dzakaaufa/experiment/hasil/visual_philosophy_hard_predict.csv"
    
    # Pilihan: "VISUAL_PHILOSOPHY" atau "PURE_PHILOSOPHY"
    compute_batik_metrics_clustered(PREDICTIONS_CSV, experiment_type="VISUAL_PHILOSOPHY")