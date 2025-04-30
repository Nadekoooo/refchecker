# views.py

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sentence_transformers import SentenceTransformer
import re

# Load metadata from CSV file
METADATA_PATH = 'checker/papers_metadata.csv'
try:
    metadata_df = pd.read_csv(METADATA_PATH)
except Exception as e:
    metadata_df = pd.DataFrame()  # Use empty DataFrame if file not found
    print(f"Warning: Metadata file not found at {METADATA_PATH}: {e}")

# Load the pre-trained XGBoost model (trial3)
try:
    xgb_model = joblib.load('models/xgb_model_trial3.pkl')
except Exception as e:
    # Fallback: load as Booster if not a pickled sklearn model
    xgb_model = xgb.Booster()
    xgb_model.load_model('models/xgb_model_trial3.pkl')
    print("Loaded XGBoost Booster model.")

# Load the sentence transformer model for chunk embeddings
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    embedder = None
    print(f"Error loading SentenceTransformer model: {e}")

def compute_metadata_features(paper_id_1, paper_id_2):
    """
    Compute document-level metadata features between two papers:
      - year_diff: absolute difference in publication year
      - author_overlap: count of common authors
      - title_similarity: cosine similarity between titles (TF-IDF based)
    Falls back to 0 if any metadata is missing or not found.
    """
    features = {'year_diff': 0, 'author_overlap': 0, 'title_similarity': 0.0}
    if metadata_df is None or metadata_df.empty:
        return features
    # Lookup entries for both documents (by ID if available, else by title)
    row1 = metadata_df[metadata_df['id'] == paper_id_1] if 'id' in metadata_df.columns else pd.DataFrame()
    row2 = metadata_df[metadata_df['id'] == paper_id_2] if 'id' in metadata_df.columns else pd.DataFrame()
    if row1.empty or row2.empty:
        # Try matching by title (case-insensitive) if IDs did not yield a result
        if 'title' in metadata_df.columns:
            if row1.empty:
                row1 = metadata_df[metadata_df['title'].str.lower() == str(paper_id_1).lower()]
            if row2.empty:
                row2 = metadata_df[metadata_df['title'].str.lower() == str(paper_id_2).lower()]
    if row1.empty or row2.empty:
        # If either document not found in metadata, return default features
        return features
    meta1 = row1.iloc[0].to_dict()
    meta2 = row2.iloc[0].to_dict()
    # Calculate year_diff
    year1 = meta1.get('year') or meta1.get('Year') or meta1.get('publication_year')
    year2 = meta2.get('year') or meta2.get('Year') or meta2.get('publication_year')
    try:
        y1, y2 = int(year1), int(year2)
        features['year_diff'] = abs(y1 - y2)
    except Exception:
        features['year_diff'] = 0
    # Calculate author_overlap (common authors count, case-insensitive)
    authors1 = meta1.get('authors') or meta1.get('Authors') or meta1.get('author_names')
    authors2 = meta2.get('authors') or meta2.get('Authors') or meta2.get('author_names')
    if authors1 and authors2:
        # Split author strings by comma/semicolon or handle list of authors
        set1 = {a.strip().lower() for a in re.split(r';|,', authors1)} if isinstance(authors1, str) \
               else {str(a).strip().lower() for a in authors1}
        set2 = {a.strip().lower() for a in re.split(r';|,', authors2)} if isinstance(authors2, str) \
               else {str(a).strip().lower() for a in authors2}
        features['author_overlap'] = len(set1 & set2) if set1 and set2 else 0
    else:
        features['author_overlap'] = 0
    # Calculate title_similarity (TF-IDF cosine similarity between titles)
    title1 = str(meta1.get('title') or meta1.get('Title') or "")
    title2 = str(meta2.get('title') or meta2.get('Title') or "")
    if title1 and title2:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            tfidf_matrix = TfidfVectorizer().fit_transform([title1, title2])
            sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            features['title_similarity'] = float(sim)
        except Exception:
            # Fallback to Jaccard similarity if sklearn is unavailable
            words1 = set(title1.lower().split())
            words2 = set(title2.lower().split())
            features['title_similarity'] = float(len(words1 & words2) / len(words1 | words2)) if words1 and words2 else 0.0
    else:
        features['title_similarity'] = 0.0
    print(features)
    return features

def chunk_text(text, chunk_size=100, overlap=0):
    """
    Split text into chunks of `chunk_size` words with `overlap` words overlapping.
    Returns a list of text chunks.
    """
    words = text.split()
    if chunk_size <= 0:
        return []
    if overlap >= chunk_size:
        overlap = chunk_size - 1  # ensure overlap is smaller than chunk size
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        # Advance index by chunk_size minus overlap for next chunk
        i += chunk_size - overlap if overlap > 0 else chunk_size
    return chunks

def compute_chunk_similarity_features(text1, text2, chunk_size=20, overlap=10):
    """
    Compute chunk-level similarity features between two texts using sentence embeddings.
    Returns a dict with keys: max_chunk_sim_20_10, mean_chunk_sim_20_10, std_chunk_sim_20_10,
    frac_above80_20_10, avg_chunk_sim, max_chunk_sim, chunk_sim_variance.
    """
    # Initialize all features to 0.0
    features = {
        'max_chunk_sim_20_10': 0.0,
        'mean_chunk_sim_20_10': 0.0,
        'std_chunk_sim_20_10': 0.0,
        'frac_above80_20_10': 0.0,
        'avg_chunk_sim': 0.0,
        'max_chunk_sim': 0.0,
        'chunk_sim_variance': 0.0
    }
    # If embedding model is not available or either text is empty, return zeros
    if embedder is None or not text1 or not text2:
        print(text1)
        print(text2)
        print("jawa")
        return features
    # Split both texts into overlapping chunks
    chunks1 = chunk_text(text1, chunk_size=chunk_size, overlap=overlap)
    chunks2 = chunk_text(text2, chunk_size=chunk_size, overlap=overlap)
    if not chunks1 or not chunks2:
        return features  # one of the texts is too short or empty after splitting
    try:
        # Generate embeddings for each chunk in both texts
        emb1 = np.asarray(embedder.encode(chunks1))
        emb2 = np.asarray(embedder.encode(chunks2))
    except Exception as e:
        print(f"Embedding error: {e}")
        return features
    # Normalize embeddings for cosine similarity calculation
    emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    # Compute cosine similarity between every chunk of text1 and every chunk of text2
    sim_matrix = np.dot(emb1_norm, emb2_norm.T)  # shape: (len(chunks1), len(chunks2))
    # For each chunk in text1, find the highest similarity with any chunk in text2
    max_sims_1to2 = sim_matrix.max(axis=1)  # array of length len(chunks1)
    # For each chunk in text2, find the highest similarity with any chunk in text1
    max_sims_2to1 = sim_matrix.max(axis=0)  # array of length len(chunks2)
    # Compute directed chunk similarity stats (suspicious -> source)
    if max_sims_1to2.size > 0:
        features['max_chunk_sim_20_10'] = float(max_sims_1to2.max())
        features['mean_chunk_sim_20_10'] = float(max_sims_1to2.mean())
        features['std_chunk_sim_20_10'] = float(max_sims_1to2.std())
        features['frac_above80_20_10'] = float((max_sims_1to2 >= 0.8).mean())
    else:
        # No chunks means no similarity (features remain 0.0)
        pass
    # Combine both directions for overall statistics
    if max_sims_1to2.size and max_sims_2to1.size:
        all_max_sims = np.concatenate([max_sims_1to2, max_sims_2to1])
    else:
        # If one direction is empty (shouldn't happen if both texts have content), use the one available
        all_max_sims = max_sims_1to2 if max_sims_1to2.size else max_sims_2to1
    if all_max_sims.size > 0:
        features['avg_chunk_sim'] = float(all_max_sims.mean())
        features['max_chunk_sim'] = float(all_max_sims.max())
        features['chunk_sim_variance'] = float(all_max_sims.var())
    return features

def upload_and_predict(request):
    if request.method == 'POST':
        # Retrieve uploaded files and any provided IDs from the form
        suspicious_file = request.FILES.get('suspect_file')   # suspicious document
        source_file = request.FILES.get('source_file')       # source (original) document
        suspect_id = request.POST.get('suspect_id')
        source_id = request.POST.get('source_id')
        # Read file contents into text strings (assuming text files; adjust if PDFs need parsing)
        suspicious_text = suspicious_file.read().decode('utf-8', errors='ignore') if suspicious_file else ""
        source_text = source_file.read().decode('utf-8', errors='ignore') if source_file else ""
        # 1. Compute metadata features (using provided IDs or file names as identifiers)
        if suspect_id and source_id:
            meta_features = compute_metadata_features(suspect_id, source_id)
        else:
            # If IDs not provided, use filename (without extension) as fallback key for metadata lookup
            base_id1 = suspicious_file.name.rsplit('.', 1)[0] if suspicious_file else ""
            base_id2 = source_file.name.rsplit('.', 1)[0] if source_file else ""
            meta_features = compute_metadata_features(base_id1, base_id2)
        # 2. Compute chunk similarity features from document texts
        chunk_features = compute_chunk_similarity_features(suspicious_text, source_text, chunk_size=20, overlap=10)
        # Merge all features into a single dictionary for model input
        all_features = {**meta_features, **chunk_features}
        # Log the feature names for debugging (printed to server console)
        print("Features used for prediction:", list(all_features.keys()))
        # Prepare feature DataFrame for model prediction
        X = pd.DataFrame([all_features])
        try:
            # Use the loaded XGBoost model to predict
            if hasattr(xgb_model, 'predict_proba'):
                # If xgb_model is an XGBClassifier or similar (sklearn API)
                proba = xgb_model.predict_proba(X)  # probability for each class
                pred_class = xgb_model.predict(X)   # predicted class label
                pred_label = int(pred_class[0])
                # If binary classification, probability of class 1 is proba[0][1]
                prob_score = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
            else:
                # If xgb_model is a Booster object
                dmatrix = xgb.DMatrix(X, feature_names=X.columns)
                prob_score = float(xgb_model.predict(dmatrix)[0])
                pred_label = 1 if prob_score >= 0.5 else 0
        except Exception as e:
            print(all_features)
            print(f"Prediction error: {e}")
            pred_label = -1
            prob_score = None
        # Return the prediction result (as JSON response; could also render to template as needed)
        result = {"prediction": pred_label, "probability": prob_score}
        return JsonResponse(result)
    else:
        # If GET request, render a template with an upload form (ensure 'upload_form.html' exists)
        return render(request, 'upload.html')
