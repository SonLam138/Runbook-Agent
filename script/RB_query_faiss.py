import json
import re
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.load_model import embedding_model as model
# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATA_FILE = DATA_DIR / "runbook_data.json"
FAISS_INDEX_FILE = DATA_DIR / "runbook_faiss.index"

#MODEL_PATH = Path(r"D:\bge-m3")   # sửa nếu cần

RESULT_THRESHOLD = 0.50

# =========================
# UTILS
# =========================
def normalize(text):
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# LOAD ONCE
# =========================
runbooks = load_json(DATA_FILE)
index = faiss.read_index(str(FAISS_INDEX_FILE))
#model = SentenceTransformer(str(MODEL_PATH), trust_remote_code=True)

# =========================
# INTERNAL
# =========================
def _encode_query(query: str):
    vec = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True
    )[0]
    return np.array([vec], dtype="float32")

# =========================
# PUBLIC SEARCH
# =========================
def faiss_search_topk(query, exclude_titles=None, topk=3, result_threshold=RESULT_THRESHOLD):
    """
    Low-level FAISS retrieval.
    Trả FULL runbook object + score.
    """
    if exclude_titles is None:
        exclude_titles = []

    exclude_titles_norm = {normalize(x) for x in exclude_titles}

    q_vec = _encode_query(query)

    search_k = max(topk * 3, 10)
    scores, indices = index.search(q_vec, search_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        rb = runbooks[idx]
        title_norm = normalize(rb.get("title", ""))

        if title_norm in exclude_titles_norm:
            continue

        if score < result_threshold:
            continue

        item = {
            "title": rb.get("title", ""),
            "service": rb.get("service", ""),
            "keyword": rb.get("keyword", ""),
            "description": rb.get("description", ""),
            "precheck": rb.get("precheck", []),
            "steps": rb.get("steps", []),
            "postcheck": rb.get("postcheck", []),
            "source_file": rb.get("source_file", ""),
            "score": float(score)
        }

        results.append(item)

        if len(results) >= topk:
            break

    return results