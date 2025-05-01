# checker/views.py

import os, re
import torch
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from django.shortcuts import render
from django.http import JsonResponse
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# 1) Load metadata
BASE_DIR    = os.path.dirname(__file__)
METADATA_FP = os.path.join(BASE_DIR, 'papers_metadata.csv')
try:
    metadata_df = pd.read_csv(METADATA_FP)
except Exception:
    metadata_df = pd.DataFrame()
    print(f"Warning: cannot load metadata at {METADATA_FP}")
    
    
# load precomputed group stats
with open(os.path.join(BASE_DIR,'..', 'models', 'paper_stats.pkl'), 'rb') as f:
    PAPER_STATS = pickle.load(f)
with open(os.path.join(BASE_DIR, '..','models', 'ref_stats.pkl'), 'rb') as f:
    REF_STATS = pickle.load(f)

# 2) Device & Specter for doc‐level embeddings
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
specter   = AutoModel.from_pretrained('allenai/specter').to(device)

# 3) Chunk‐level embedder
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    embedder = None
    print("Warning: failed to load chunk embedder")

# 4) XGBoost model
MODEL_FP = os.path.join(BASE_DIR, '..', 'models', 'xgb_model_trial3.pkl')
try:
    xgb_model = joblib.load(MODEL_FP)
except Exception:
    xgb_model = xgb.Booster()
    xgb_model.load_model(MODEL_FP)
    print("Loaded XGBoost Booster fallback")
    

if hasattr(xgb_model, "feature_names_in_"):
    MODEL_FEATURES = list(xgb_model.feature_names_in_)
elif hasattr(xgb_model, "feature_names"):
    MODEL_FEATURES = list(xgb_model.feature_names)
else:
    MODEL_FEATURES = []
    


def compute_doc_meta_features(paper_id1, paper_id2, text1, text2):
    """
    Compute doc‐level embedding similarity + all metadata features.
    """
    feats = {
        'text_similarity':        0.0,
        'year_diff':              0,
        'can_cite':               0,
        'same_year':              0,
        'cited_by_count_paper':   0,
        'cited_by_count_ref':     0,
        'cited_by_count_ratio':   0.0,
        'author_overlap':         0.0,
        'concept_overlap':        0.0,
        'same_type':              0,
        'title_similarity':       0.0,
        'contains_citation_text': 0,
        'paper':                  paper_id1,
        'referenced_paper':       paper_id2,
    }

    # -- doc‐level embedding similarity via Specter --
    try:
        inp1 = tokenizer(text1, return_tensors='pt',
                         truncation=True, padding='max_length', max_length=512).to(device)
        inp2 = tokenizer(text2, return_tensors='pt',
                         truncation=True, padding='max_length', max_length=512).to(device)
        with torch.no_grad():
            o1 = specter(**inp1).last_hidden_state[:,0,:].cpu().numpy()
            o2 = specter(**inp2).last_hidden_state[:,0,:].cpu().numpy()
        feats['text_similarity'] = float(cosine_similarity(o1, o2)[0][0])
    except Exception:
        pass

    # -- metadata lookups --
    print(metadata_df.head())
    r1 = metadata_df[metadata_df['paper_id'] == paper_id1]
    r2 = metadata_df[metadata_df['paper_id'] == paper_id2]
    if r1.empty or r2.empty:
        return feats
    m1, m2 = r1.iloc[0], r2.iloc[0]

    # year_diff, can_cite, same_year
    y1, y2 = int(m1.publication_year), int(m2.publication_year)
    feats['year_diff'] = abs(y1 - y2)
    feats['can_cite']  = 1 if (y1 - y2) > 0 else 0
    feats['same_year'] = 1 if y1 == y2 else 0

    # citation counts
    c1 = int(m1.cited_by_count or 0)
    c2 = int(m2.cited_by_count or 0)
    feats.update({
        'cited_by_count_paper': c1,
        'cited_by_count_ref':   c2,
        'cited_by_count_ratio': c2 / (c1 + 1),
    })

    # author_overlap
    a1 = set(str(m1.authors).split(';')) if pd.notna(m1.authors) else set()
    a2 = set(str(m2.authors).split(';')) if pd.notna(m2.authors) else set()
    if a1 and a2:
        feats['author_overlap'] = len(a1 & a2) / len(a1 | a2)

    # concept_overlap
    cset1 = set(str(m1.concepts).lower().split(';')) if pd.notna(m1.concepts) else set()
    cset2 = set(str(m2.concepts).lower().split(';')) if pd.notna(m2.concepts) else set()
    if cset1 and cset2:
        feats['concept_overlap'] = len(cset1 & cset2) / len(cset1 | cset2)

    # same_type
    feats['same_type'] = 1 if m1.type == m2.type else 0

    # title_similarity (Jaccard)
    t1, t2 = set(str(m1.title).lower().split()), set(str(m2.title).lower().split())
    if t1 and t2:
        feats['title_similarity'] = len(t1 & t2) / len(t1 | t2)

    # contains_citation_text
    title2 = str(m2.title).lower()
    if len(title2) > 5 and title2 in text1.lower():
        feats['contains_citation_text'] = 1

    return feats

def chunk_text(text, chunk_size=25, overlap=10):
    words = text.split()
    overlap = min(overlap, chunk_size-1)
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

def compute_chunk_similarity_features(text1, text2, chunk_size=25, overlap=10):
    feats = {
        'max_chunk_sim_25_10':  0.0,
        'mean_chunk_sim_25_10': 0.0,
        'std_chunk_sim_25_10':  0.0,
        'frac_above80_25_10':   0.0,
        'avg_chunk_sim':        0.0,
        'max_chunk_sim':        0.0,
        'chunk_sim_variance':   0.0,
    }
    if embedder is None or not text1 or not text2:
        return feats

    c1 = chunk_text(text1, chunk_size, overlap)
    c2 = chunk_text(text2, chunk_size, overlap)
    if not c1 or not c2:
        return feats

    emb1 = np.asarray(embedder.encode(c1))
    emb2 = np.asarray(embedder.encode(c2))
    n1   = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    n2   = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    simm = np.dot(n1, n2.T)

    max1 = simm.max(axis=1)
    if max1.size:
        feats['max_chunk_sim_25_10']  = float(max1.max())
        feats['mean_chunk_sim_25_10'] = float(max1.mean())
        feats['std_chunk_sim_25_10']  = float(max1.std())
        feats['frac_above80_25_10']   = float((max1 >= 0.8).mean())

    all_max = np.hstack((max1, simm.max(axis=0)))
    if all_max.size:
        feats['avg_chunk_sim']      = float(all_max.mean())
        feats['max_chunk_sim']      = float(all_max.max())
        feats['chunk_sim_variance'] = float(all_max.var())

    return feats

def upload_and_predict(request):
    if request.method == 'POST':
        # 1) Read the two uploaded files
        f1 = request.FILES.get('suspect_file')
        f2 = request.FILES.get('source_file')
        id1 = os.path.splitext(f1.name)[0] if f1 else ""
        id2 = os.path.splitext(f2.name)[0] if f2 else ""
        t1  = f1.read().decode('utf-8', errors='ignore') if f1 else ""
        t2  = f2.read().decode('utf-8', errors='ignore') if f2 else ""

        # 2) Compute your dynamic features
        meta_feats  = compute_doc_meta_features(id1, id2, t1, t2)
        chunk_feats = compute_chunk_similarity_features(t1, t2, 25, 10)

        # 3) Lookup precomputed group stats
        p_stats = PAPER_STATS.get(id1, {})
        r_stats = REF_STATS.get(id2, {})

        # 4) Merge all into a single dict
        all_feats = {**meta_feats, **chunk_feats, **p_stats, **r_stats}

        # 5) Build the diff‐features on the fly
        for base_feat in list(meta_feats.keys()) + list(chunk_feats.keys()):
            for stat in ("mean","median","max","min","std","var"):
                kp = f"{base_feat}_agg{stat}paper"
                if kp in p_stats:
                    all_feats[f"{base_feat}_diff{stat}paper"] = all_feats[base_feat] - p_stats[kp]
                kr = f"{base_feat}_agg{stat}referenced_paper"
                if kr in r_stats:
                    all_feats[f"{base_feat}_diff{stat}referenced_paper"] = all_feats[base_feat] - r_stats[kr]

        # 6) Assemble into DataFrame, reindex to model’s expected features
        X = pd.DataFrame([all_feats])
        if MODEL_FEATURES:
            X = X.reindex(columns=MODEL_FEATURES, fill_value=0)
        print(X)
        # 7) Run the prediction
        try:
            if hasattr(xgb_model, 'predict_proba'):
                proba = xgb_model.predict_proba(X)
                # ambil probabilitas kelas 1
                score = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
                # gunakan threshold yang lebih rendah, misal 0.1
                threshold = 0.0001
                print(score)
                pred = 1 if score >= threshold else 0
            else:
                dmat  = xgb.DMatrix(X, feature_names=X.columns)
                p0    = float(xgb_model.predict(dmat)[0])
                pred  = int(p0 >= 0.25)
                score = p0
        except Exception as e:
            print("Prediction error:", e)
            pred, score = -1, None

        # 8) Prepare context for result.html
        ctx = {
            'prediction': "Yes" if pred == 1 else "No",
            'probability': score,
            # core features you want to display:
            'text_similarity':        all_feats.get('text_similarity'),
            'year_diff':              all_feats.get('year_diff'),
            'can_cite':               all_feats.get('can_cite'),
            'same_year':              all_feats.get('same_year'),
            'cited_by_count_paper':   all_feats.get('cited_by_count_paper'),
            'cited_by_count_ref':     all_feats.get('cited_by_count_ref'),
            'cited_by_count_ratio':   all_feats.get('cited_by_count_ratio'),
            'author_overlap':         all_feats.get('author_overlap'),
            'concept_overlap':        all_feats.get('concept_overlap'),
            'same_type':              all_feats.get('same_type'),
            'title_similarity':       all_feats.get('title_similarity'),
            'contains_citation_text': all_feats.get('contains_citation_text'),
        }
        return render(request, 'result.html', ctx)

    # GET: show upload form
    return render(request, 'upload.html')
