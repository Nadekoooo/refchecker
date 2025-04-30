# checker/utils.py

import re, pickle, torch, numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1) Load models sekali di import time
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Document‐level embedder (SPECTER)
DOC_TOKENIZER = AutoTokenizer.from_pretrained('allenai/specter')
DOC_MODEL     = AutoModel.from_pretrained('allenai/specter').to(DEVICE)

# Chunk‐level embedder
CHUNK_ENCODER = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)

# Classifier
with open('models/xgb_model_trial3.pkl', 'rb') as f:
    XGB_MODEL = pickle.load(f)

def read_txt(file_obj):
    return file_obj.read().decode('utf-8', errors='ignore')

def chunk_text(text, size=15, stride=5):
    tokens = re.sub(r'\s+',' ', text).split()
    chunks = [' '.join(tokens[i:i+size])
              for i in range(0, max(1, len(tokens)-size+1), stride)]
    return chunks or [' '.join(tokens)]

def embed_document(text):
    toks = DOC_TOKENIZER([text], return_tensors='pt', truncation=True,
                         padding='max_length', max_length=512).to(DEVICE)
    with torch.no_grad():
        out = DOC_MODEL(**toks)
    return out.last_hidden_state[:,0,:].cpu().numpy()

def embed_chunks(chunks):
    return CHUNK_ENCODER.encode(chunks,
                                batch_size=64,
                                convert_to_numpy=True,
                                show_progress_bar=False)

def extract_features(txt_a, txt_b):
    # 1) Full‐doc embeddings & cosine
    emb_a = embed_document(txt_a)[0]
    emb_b = embed_document(txt_b)[0]
    text_sim = cosine_similarity([emb_a], [emb_b])[0,0]
    
    # 2) Chunk embeddings & stats
    ch_a = embed_chunks(chunk_text(txt_a))
    ch_b = embed_chunks(chunk_text(txt_b))
    if ch_a.size and ch_b.size:
        sims = cosine_similarity(ch_a, ch_b).flatten()
        max_sim = sims.max()
        mean_sim = sims.mean()
        frac_above = (sims>0.8).mean()
    else:
        max_sim = mean_sim = frac_above = 0.0
    
    return {
        'text_similarity': text_sim,
        'max_chunk_sim':   max_sim,
        'mean_chunk_sim':  mean_sim,
        'frac_above80':    frac_above
    }

def predict_reference(txt_a, txt_b):
    feats = extract_features(txt_a, txt_b)
    X = np.array([list(feats.values())])
    pred = XGB_MODEL.predict(X)[0]
    prob = XGB_MODEL.predict_proba(X)[0,1]
    return pred, prob
