import os
import pandas as pd
import numpy as np
import pickle
from django.shortcuts import render
from transformers import AutoTokenizer, AutoModel  # untuk model SPECTER
from sklearn.metrics.pairwise import cosine_similarity
# from xgboost import XGBClassifier  # (opsional, jika butuh XGB library)

# Load metadata once at module load (optional for efficiency)
METADATA_DF = pd.read_csv(os.path.join(os.path.dirname(__file__), 'papers_metadata.csv'))

# (Opsional) If using GPU for embeddings:
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load SPECTER model & tokenizer once to reuse (optional to avoid re-loading on each request)
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter').to(device)

def upload_and_predict(request):
    if request.method == 'POST':
        form = PaperUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # 1. Baca file upload dari request.FILES
            file1 = request.FILES['paper1']
            file2 = request.FILES['paper2']
            text1 = file1.read().decode('utf-8', errors='ignore')
            text2 = file2.read().decode('utf-8', errors='ignore')

            # 2. Ambil metadata untuk masing-masing paper
            # Asumsi: nama file (tanpa .txt) adalah ID atau judul yang bisa dicocokkan dengan metadata
            paper_id1 = os.path.splitext(file1.name)[0]
            paper_id2 = os.path.splitext(file2.name)[0]
            # Cari baris metadata yang sesuai (misal ada kolom 'paper_id' di CSV)
            meta1 = METADATA_DF[METADATA_DF['paper_id'] == paper_id1].iloc[0].to_dict()
            meta2 = METADATA_DF[METADATA_DF['paper_id'] == paper_id2].iloc[0].to_dict()

            # 3. Hitung embedding vektor dokumen menggunakan model SPECTER
            # Tokenisasi dan encoding
            inputs1 = tokenizer(text1, truncation=True, padding=True, max_length=512, return_tensors='pt').to(device)
            inputs2 = tokenizer(text2, truncation=True, padding=True, max_length=512, return_tensors='pt').to(device)
            with torch.no_grad():
                output1 = model(**inputs1)
                output2 = model(**inputs2)
            # Ambil representasi vektor [CLS] token (dimensi 768)
            vec1 = output1.last_hidden_state[:, 0, :].cpu().numpy()
            vec2 = output2.last_hidden_state[:, 0, :].cpu().numpy()

            # 4. Menghasilkan fitur perbandingan dari kedua paper
            # Contoh fitur: cosine similarity antar embedding, selisih tahun, overlap penulis
            text_sim = float(cosine_similarity(vec1, vec2)[0][0])
            year_diff = abs(int(meta1.get('year', 0)) - int(meta2.get('year', 0)))
            # Asumsikan kolom authors berisi string nama-nama penulis yang dipisah koma atau titik koma
            authors1 = set([a.strip() for a in str(meta1.get('authors', '')).split(',')])
            authors2 = set([a.strip() for a in str(meta2.get('authors', '')).split(',')])
            author_overlap = len(authors1.intersection(authors2))
            # Kumpulkan fitur dalam urutan yang sesuai dengan model
            features = np.array([[text_sim, year_diff, author_overlap]], dtype=float)

            # 5. Muat model XGBoost dan lakukan prediksi
            model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
            with open(model_path, 'rb') as f:
                xgb_model = pickle.load(f)  # memuat model terlatih
            pred_label = xgb_model.predict(features)[0]  # misal hasil 1 untuk "referensi", 0 untuk "tidak"
            pred_prob = None
            # Jika model disimpan sebagai classifier sklearn, bisa juga dapatkan probabilitas:
            if hasattr(xgb_model, "predict_proba"):
                pred_prob = xgb_model.predict_proba(features)[0][1]  # probabilitas kelas "referensi"

            # Interpretasi hasil prediksi ke bentuk Yes/No
            prediction = "Yes" if pred_label == 1 else "No"

            # 6. Siapkan konteks data untuk template hasil
            context = {
                'prediction': prediction,
                'text_similarity': f"{text_sim:.3f}",
                'year_diff': year_diff,
                'author_overlap': author_overlap,
                'probability': f"{pred_prob:.2f}" if pred_prob is not None else None,
            }
            return render(request, 'result.html', context)
    else:
        form = PaperUploadForm()
    return render(request, 'upload.html', {'form': form})
