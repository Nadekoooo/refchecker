# Layanan Prediksi Referensi Paper (GammaFest)

Aplikasi web berbasis Django untuk memprediksi apakah satu paper (“suspect”) mereferensi paper lain (“source”) dengan menggabungkan:

- **Embedding dokumen** (SPECTER)  
- **Embedding chunk** (MiniLM-L6-v2)  
- **Metadata kaya** (tahun terbit, jumlah sitasi, nama penulis, konsep, kesamaan judul, dll.)  
- **Statistik agregat pra-hitung** (rata-rata, median, max, min, std, var per paper & per referenced_paper + fitur “diff”)  
- **Klasifier XGBoost** (pre-trained; tidak perlu retrain di runtime)

---

## Fitur Utama

1. **Pra-proses offline**  
   - Hitung statistik agregat numeric untuk setiap `paper` dan setiap `referenced_paper` dari dataset train, simpan di `models/paper_stats.pkl` & `models/ref_stats.pkl`.  
   - Simpan model XGBoost terlatih (`models/xgb_model_trial3.pkl`).  

2. **Pipeline inference online**  
   1. Upload dua file teks (.txt).  
   2. Hitung fitur dinamis:  
      - Kemiripan embedding dokumen (SPECTER)  
      - Statistik embedding chunk (MiniLM)  
      - Fitur metadata (selisih tahun, jumlah sitasi, overlap penulis/konsep, kesamaan judul, deteksi teks sitasi, dll.)  
   3. Lookup statistik agregat dari file `.pkl` → hitung juga fitur “diff” (perbedaan antara nilai dinamis dan agregat).  
   4. Susun DataFrame persis sesuai daftar fitur yang diharapkan model, isi missing dengan 0.  
   5. Prediksi probabilitas & label referensi.

3. **Tidak perlu koneksi internet** setelah semua model & metadata diunduh.

---

## Panduan Instalasi & Menjalankan

### 1. Clone & Pasang Dependensi

```bash
git clone https://github.com/your-org/gammafest-reference.git
cd gammafest-reference
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install -r requirements.txt
