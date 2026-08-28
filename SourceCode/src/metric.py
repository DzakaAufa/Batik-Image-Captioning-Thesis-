import os
import re
java_path = "/mnt/extended-home/dzakaaufa/java_libs/amazon-corretto-8.482.08.1-linux-x64/bin/java"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['JAVA_HOME'] = os.path.dirname(os.path.dirname(java_path))
os.environ['PATH'] = os.path.dirname(java_path) + os.pathsep + os.environ['PATH']

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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# UTILITIES — konsisten dengan inference_pipeline_fixed.py
# ============================================================

# FIX #3: normalize_text yang seragam dengan inference script
# Urutan penting: ganti tanda baca → spasi DULU, baru compress whitespace
# Sehingga "kawung.motif" → "kawung motif", bukan "kawungmotif"
def normalize_text(text):
    text = str(text).lower().strip()
    text = text.replace("_", " ")          # underscore → spasi
    text = re.sub(r"[^\w\s]", " ", text)  # tanda baca → spasi (bukan dihapus)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def check_motif_mentioned(pred_caption, gt_class):
    """Cek apakah ground truth class disebutkan di dalam prediksi caption."""
    return normalize_text(gt_class) in normalize_text(pred_caption)

# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_nltk_bleu(references, predictions):
    """Hitung BLEU-1 dan BLEU-4 secara korpus."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    tokenized_refs  = [[nltk.word_tokenize(str(ref).lower())]  for ref in references]
    tokenized_preds = [nltk.word_tokenize(str(pred).lower())   for pred in predictions]

    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(tokenized_refs, tokenized_preds,
                        weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(tokenized_refs, tokenized_preds,
                        weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    return bleu1, bleu4

# FIX #4: clean_for_spice tanpa hard-limit karakter
# Hanya batasi jumlah kalimat (3 kalimat pertama) agar tidak terlalu panjang
# untuk Java SPICE tanpa memotong caption di tengah kata/kalimat
def clean_for_spice(text):
    """
    Bersihkan teks untuk SPICE: ambil 3 kalimat pertama saja.
    Tidak ada hard-limit karakter — menghindari caption terpotong di tengah
    yang merusak scene graph relation dan attribute extraction.
    """
    sentences = str(text).split('.')
    # Filter kalimat kosong, lalu ambil 3 pertama
    sentences = [s.strip() for s in sentences if s.strip()]
    shortened = ". ".join(sentences[:3])
    if shortened and not shortened.endswith('.'):
        shortened += "."
    return shortened

# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def compute_batik_metrics(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File prediksi tidak ditemukan di: {csv_path}")

    print(f" Membaca data hasil prediksi: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = ['image_path', 'reference', 'prediction',
                     'ground_truth_class', 'predicted_class']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV wajib memiliki kolom: {required_cols}")

    image_paths = df['image_path'].tolist()
    references  = df['reference'].tolist()
    predictions = df['prediction'].tolist()

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    print("\n[Mulai] Menghitung Metrik Evaluasi...")
    print("-" * 50)

    # ==========================================================
    # 1. BLEU-1 & BLEU-4
    # ==========================================================
    print(" -> [1/8] Menghitung BLEU Score (NLTK)...")
    bleu1, bleu4 = calculate_nltk_bleu(references, predictions)
    results['BLEU-1'] = bleu1
    results['BLEU-4'] = bleu4

    # ==========================================================
    # 2. ROUGE-L
    # ==========================================================
    print(" -> [2/8] Menghitung ROUGE-L Score...")
    rouge_metric  = evaluate.load("rouge")
    rouge_results = rouge_metric.compute(predictions=predictions,
                                         references=references,
                                         rouge_types=["rougeL"])
    results['ROUGE-L'] = rouge_results["rougeL"]

    # ==========================================================
    # 3. METEOR
    # ==========================================================
    print(" -> [3/8] Menghitung METEOR Score...")
    try:
        meteor_metric  = evaluate.load("meteor")
        meteor_results = meteor_metric.compute(predictions=predictions,
                                               references=references)
        results['METEOR'] = meteor_results["meteor"]
    except Exception as e:
        print(f"    [WARN] METEOR gagal: {e}")
        results['METEOR'] = 0.0

    # ==========================================================
    # 4. SPICE — FIX #4 clean_for_spice tanpa [:200]
    # ==========================================================
    print(" -> [4/8] Menghitung SPICE Score...")
    try:
        gts = {str(i): [clean_for_spice(ref)]  for i, ref  in enumerate(references)}
        res = {str(i): [clean_for_spice(pred)] for i, pred in enumerate(predictions)}

        spice_scorer = Spice()
        spice_score, _ = spice_scorer.compute_score(gts, res)
        results['SPICE'] = spice_score
    except Exception as e:
        print(f"    [WARN] SPICE gagal: {e}")
        results['SPICE'] = 0.0

    # ==========================================================
    # 5. BERTScore
    # ==========================================================
    print(" -> [5/8] Menghitung BERTScore...")
    bertscore_metric = evaluate.load("bertscore")
    bs_results = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        model_type="xlm-roberta-base",
        lang="multilingual"
    )
    results['BERTScore'] = np.mean(bs_results["f1"])

    # ==========================================================
    # 6. CLIPScore
    # ==========================================================
    print(" -> [6/8] Menghitung CLIPScore...")
    try:
        model_clip, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        model_clip = model_clip.to(device).eval()
        tokenizer  = open_clip.get_tokenizer('ViT-B-32')

        clip_scores = []
        with torch.no_grad():
            for img_path, pred in tqdm(zip(image_paths, predictions),
                                       total=len(predictions), desc="CLIP"):
                if not os.path.exists(img_path):
                    continue
                image        = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                # Batasi ke 77 token (batas arsitektur CLIP) bukan karakter
                # Tokenisasi dulu, lalu decode kembali agar truncation rapi
                tokens_raw   = tokenizer([str(pred)])
                # open_clip tokenizer sudah handle max_length=77 secara internal
                text         = tokens_raw.to(device)

                img_feat  = model_clip.encode_image(image)
                txt_feat  = model_clip.encode_text(text)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

                similarity = (img_feat @ txt_feat.T).item()
                clip_scores.append(similarity)

        results['CLIPScore'] = np.mean(clip_scores) if clip_scores else 0.0
    except Exception as e:
        print(f"    [WARN] CLIPScore gagal: {e}")
        results['CLIPScore'] = 0.0

    # ==========================================================
    # 7. Classification Accuracy
    #    FIX #1/#2: predicted_class di CSV sudah dari VLM (bukan DINOv2)
    #    sehingga accuracy di sini adalah accuracy VLM, bukan DINOv2
    #
    #    Jika CSV dari inference lama masih ada kolom 'dinov2_class',
    #    script ini juga menghitung DINOv2 accuracy secara terpisah
    # ==========================================================
    print(" -> [7/8] Menghitung Classification Accuracy...")

    gt_classes   = df['ground_truth_class'].astype(str).str.lower().str.strip()
    pred_classes = df['predicted_class'].astype(str).str.lower().str.strip()
    vlm_acc      = accuracy_score(gt_classes, pred_classes)
    results['VLM_Classification_Accuracy'] = vlm_acc

    # Jika kolom dinov2_class tersedia (ada di CSV baru), hitung juga
    if 'dinov2_class' in df.columns:
        dinov2_classes = df['dinov2_class'].astype(str).str.lower().str.strip()
        # Hanya hitung baris yang dinov2_class tidak None/nan (mode "hard")
        valid_mask = df['dinov2_class'].notna() & (df['dinov2_class'] != 'None')
        if valid_mask.sum() > 0:
            dinov2_acc = accuracy_score(
                gt_classes[valid_mask], dinov2_classes[valid_mask]
            )
            results['DINOv2_Classification_Accuracy'] = dinov2_acc
            print(f"    DINOv2 Acc (mode 'hard' only, n={valid_mask.sum()}): {dinov2_acc:.4f}")
        else:
            results['DINOv2_Classification_Accuracy'] = None
            print("    DINOv2 Acc: tidak tersedia (semua mode='no_class')")

    # ==========================================================
    # 8. Motif Mention Rate — FIX #3 normalize_text konsisten
    # ==========================================================
    print(" -> [8/8] Menghitung Motif Mention Rate...")

    mentioned_count = sum(
        check_motif_mentioned(pred_cap, gt_cls)
        for pred_cap, gt_cls in zip(predictions, df['ground_truth_class'])
    )
    results['Motif_Mention_Rate'] = mentioned_count / len(predictions) if predictions else 0.0

    # ==========================================================
    # FIX #5: Formula agregat yang transparan dan konsisten
    #
    # Sebelumnya Overall Average hanya pakai 6 metrik tapi tabel
    # menampilkan 10 baris. Sekarang dibuat eksplisit dua formula:
    #
    #   Lexical_Relevance_Avg  : BLEU-1, BLEU-4, ROUGE-L, METEOR
    #                            (BLEU-1 ditambahkan agar tidak terbuang)
    #   Overall_Quality_Avg    : semua metrik utama kecuali Classification
    #                            dan Motif Mention Rate (karena keduanya
    #                            mengukur hal berbeda dari captioning quality)
    # ==========================================================

    # FIX #6: BLEU-1 dimasukkan ke Lexical Relevance (sebelumnya hilang)
    results['Lexical_Relevance_Avg'] = np.mean([
        results['BLEU-1'],
        results['BLEU-4'],
        results['ROUGE-L'],
        results['METEOR'],
    ])

    results['Overall_Quality_Avg'] = np.mean([
        results['BLEU-1'],
        results['BLEU-4'],
        results['ROUGE-L'],
        results['METEOR'],
        results['SPICE'],
        results['BERTScore'],
        results['CLIPScore'],
    ])

    # Cetak Hasil
    print("\n" + "=" * 55)
    print(" HASIL EVALUASI CAPTIONING BATIK")
    print("=" * 55)
    print(f"Jumlah Sampel Diuji          : {len(predictions)}")
    print("-" * 55)
    print(f"BLEU-1 (Precision)           : {results['BLEU-1']:.4f}")
    print(f"BLEU-4 (Grammar)             : {results['BLEU-4']:.4f}")
    print(f"ROUGE-L (Recall)             : {results['ROUGE-L']:.4f}")
    print(f"METEOR (Synonyms)            : {results['METEOR']:.4f}")
    print(f"SPICE (Graph/Sem)            : {results['SPICE']:.4f}")
    print(f"BERTScore (Semantic)         : {results['BERTScore']:.4f}")
    print(f"CLIPScore (Visual)           : {results['CLIPScore']:.4f}")
    print("-" * 55)
    print(f"VLM Classification Acc       : {results['VLM_Classification_Accuracy']:.4f}")
    if results.get('DINOv2_Classification_Accuracy') is not None:
        print(f"DINOv2 Classification Acc    : {results['DINOv2_Classification_Accuracy']:.4f}")
    print(f"Motif Mention Rate           : {results['Motif_Mention_Rate']:.4f}")
    print("-" * 55)
    print(f"Lexical Relevance Avg        : {results['Lexical_Relevance_Avg']:.4f}")
    print(f"Overall Quality Avg          : {results['Overall_Quality_Avg']:.4f}")
    print("=" * 55)

    # Simpan JSON
    save_path = os.path.join(os.path.dirname(csv_path), "metrics_summary.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Ringkasan sukses disimpan ke → {save_path}\n")

    return results

if __name__ == "__main__":
    PREDICTIONS_CSV = "/mnt/extended-home/dzakaaufa/leakage/evaluation_results/experiment_3b_class_inject.csv"
    compute_batik_metrics(PREDICTIONS_CSV)