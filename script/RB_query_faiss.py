import json
import re
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIG
# =====================================================
"""
DATA_DIR = Path(r"D:\runbook-agent\data")

RUNBOOK_JSON = DATA_DIR / "runbook_data.json"
FAISS_INDEX_FILE = DATA_DIR / "runbook_faiss.index"

MODEL_PATH = Path(r"C:\Setup\RB_Build\bge-m3")
"""
# path at laptop ===============================
DATA_DIR = Path(r"D:\LocalAI_v3_refactorCode\data")

RUNBOOK_JSON = DATA_DIR / "runbook_data.json"
FAISS_INDEX_FILE = DATA_DIR / "runbook_faiss.index"

MODEL_PATH = Path(r"D:\bge-m3")
#=====================================================

# Chỉ dùng cho retrieval hợp lệ, KHÔNG dùng cho cache
RESULT_THRESHOLD = 0.50

# Search rộng hơn topk để còn loại exclude_titles
DEFAULT_SEARCH_MULTIPLIER = 4


# =====================================================
# GLOBAL SINGLETONS
# =====================================================
_model = None
_index = None
_runbooks = None


# =====================================================
# UTILS
# =====================================================
def normalize(text):
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# RESOURCE LOADING
# =====================================================
def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy model path: {MODEL_PATH}")
        _model = SentenceTransformer(str(MODEL_PATH), trust_remote_code=True)
    return _model


def get_index():
    global _index
    if _index is None:
        if not FAISS_INDEX_FILE.exists():
            raise FileNotFoundError(f"Không tìm thấy FAISS index: {FAISS_INDEX_FILE}")
        _index = faiss.read_index(str(FAISS_INDEX_FILE))
    return _index


def get_runbooks():
    global _runbooks
    if _runbooks is None:
        if not RUNBOOK_JSON.exists():
            raise FileNotFoundError(f"Không tìm thấy runbook json: {RUNBOOK_JSON}")
        _runbooks = load_json(RUNBOOK_JSON)
    return _runbooks


# =====================================================
# ENCODING
# =====================================================
def encode_query(query: str):
    model = get_model()

    vec = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True
    )[0]

    return np.array([vec], dtype="float32")


# =====================================================
# INTERNAL BUILD RESULT
# =====================================================
def build_runbook_result(rb: dict, score: float):
    """
    Chuẩn hóa object runbook trả ra từ retrieval.
    Agent layer sẽ dùng object này trực tiếp.
    """
    return {
        "title": rb.get("title", ""),
        "service": rb.get("service", ""),
        "keyword": rb.get("keyword", ""),
        "description": rb.get("description", ""),
        "intents": rb.get("intents", []),
        "precheck": rb.get("precheck", []),
        "steps": rb.get("steps", []),
        "postcheck": rb.get("postcheck", []),
        "source_file": rb.get("source_file", ""),
        "score": float(score)
    }


# =====================================================
# TOP-K SEARCH (FULL RUNBOOK OBJECT)
# =====================================================
def search_RB_topk(query, exclude_titles=None, topk=3, result_threshold=RESULT_THRESHOLD):
    """
    Search top-k runbook bằng FAISS.

    Args:
        query (str): user query hoặc semantic query
        exclude_titles (list[str]|None): các runbook title cần loại
        topk (int): số kết quả cần trả
        result_threshold (float): score tối thiểu để coi là hợp lệ

    Returns:
        list[dict]: danh sách full runbook object + score
    """
    if exclude_titles is None:
        exclude_titles = []

    exclude_norm = {normalize(t) for t in exclude_titles if t}

    index = get_index()
    runbooks = get_runbooks()

    q_vec = encode_query(query)

    # Search rộng hơn topk để còn loại exclude titles
    search_k = max(topk * DEFAULT_SEARCH_MULTIPLIER, 10)
    scores, indices = index.search(q_vec, search_k)

    results = []
    seen_titles = set()

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        rb = runbooks[idx]
        title_norm = normalize(rb.get("title", ""))

        # loại title đã thử
        if title_norm in exclude_norm:
            continue

        # tránh duplicate cùng title
        if title_norm in seen_titles:
            continue

        # threshold
        if float(score) < float(result_threshold):
            continue

        seen_titles.add(title_norm)
        results.append(build_runbook_result(rb, score))

        if len(results) >= topk:
            break

    return results


# =====================================================
# TOP-1 SEARCH
# =====================================================
def search_RB(query, result_threshold=RESULT_THRESHOLD):
    """
    Lấy 1 runbook tốt nhất.
    """
    results = search_RB_topk(
        query=query,
        exclude_titles=[],
        topk=1,
        result_threshold=result_threshold
    )

    if not results:
        return None

    return results[0]


# =====================================================
# CANDIDATE RETRIEVAL (METADATA ONLY)
# =====================================================
def retrieve_candidate_meta(query, exclude_titles=None, topk=3, result_threshold=RESULT_THRESHOLD):
    """
    Candidate retrieval cho decision layer của agent.
    Chỉ trả metadata để LLM decide, không trả full steps cho LLM.

    Returns:
        (full_candidates, meta_candidates)
    """
    full_candidates = search_RB_topk(
        query=query,
        exclude_titles=exclude_titles or [],
        topk=topk,
        result_threshold=result_threshold
    )

    meta_candidates = []
    for i, rb in enumerate(full_candidates, start=1):
        meta_candidates.append({
            "idx": i,
            "title": rb.get("title", ""),
            "service": rb.get("service", ""),
            "keyword": rb.get("keyword", ""),
            "description": rb.get("description", ""),
            "score": rb.get("score", 0.0)
        })

    return full_candidates, meta_candidates


# =====================================================
# DEBUG / LOCAL TEST
# =====================================================
if __name__ == "__main__":
    while True:
        q = input("\n🔎 Query: ").strip()
        if not q:
            break

        results = search_RB_topk(q, topk=3)

        if not results:
            print("❌ Không tìm thấy runbook phù hợp")
            continue

        print("\n===== TOP RESULTS =====")
        for i, rb in enumerate(results, start=1):
            print(f"\nTop {i}")
            print(f"Title   : {rb['title']}")
            print(f"Service : {rb['service']}")
            print(f"Score   : {rb['score']:.4f}")
            print(f"Keyword : {rb['keyword']}")