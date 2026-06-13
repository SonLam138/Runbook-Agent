import json
import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
KB_DIR = Path(r"D:\Runbook\_kb_v2")

DATA_FILE = KB_DIR / "runbook_data.json"
VECTOR_FILE = KB_DIR / "runbook_vectors.npy"
CACHE_FILE = KB_DIR / "query_cache.json"

MODEL_PATH = Path(r"D:\bge-m3")
RESULT_THRESHOLD = 0.5   # trả kết quả
CACHE_THRESHOLD  = 0.75   # lưu cache

# semantic cache threshold
# 0.95 ~ rất chặt
# 0.90 ~ khá nhạy
SEMANTIC_CACHE_THRESHOLD = 0.95

# =========================
# UTILS
# =========================
def normalize(text):
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def load_json_safe(path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================
# LOAD KB
# =========================
print("🔄 Loading KB...")

runbooks = load_json_safe(DATA_FILE, [])
vectors = np.load(VECTOR_FILE).astype("float32")
query_cache = load_json_safe(CACHE_FILE, [])

if not runbooks:
    raise RuntimeError("Không tìm thấy runbook_data.json hoặc file rỗng")
if len(vectors) != len(runbooks):
    raise RuntimeError("Số vector không khớp số runbook")

model = SentenceTransformer(str(MODEL_PATH), trust_remote_code=True)

print("✅ KB loaded")

# =========================
# CACHE HELPERS
# =========================
def exact_cache_hit(query_norm):
    """
    Exact cache:
    nếu query giống 100% sau normalize thì skip luôn embedding
    """
    for item in query_cache:
        if item["query_norm"] == query_norm:
            return item
    return None

def semantic_cache_hit(query_vector):
    """
    Semantic cache:
    so query vector mới với các query vector đã cache
    nếu similarity > threshold thì reuse result
    """
    best_item = None
    best_score = -1.0

    for item in query_cache:
        old_vec = np.array(item["vector"], dtype="float32")
        sim = float(np.dot(old_vec, query_vector))

        if sim > best_score:
            best_score = sim
            best_item = item

    if best_item is not None and best_score >= SEMANTIC_CACHE_THRESHOLD:
        return best_item, best_score

    return None, best_score

def add_or_update_cache(query, query_norm, query_vector, best_idx):
    """
    Nếu exact query đã có thì update.
    Nếu chưa có thì append.
    """
    for item in query_cache:
        if item["query_norm"] == query_norm:
            item["query"] = query
            item["vector"] = query_vector.tolist()
            item["best_idx"] = int(best_idx)
            item["best_title"] = runbooks[best_idx]["title"]
            return

    query_cache.append({
        "query": query,
        "query_norm": query_norm,
        "vector": query_vector.tolist(),
        "best_idx": int(best_idx),
        "best_title": runbooks[best_idx]["title"]
    })

# =========================
# SEARCH LOGIC
# =========================
def search_runbook(query):
    query_norm = normalize(query)

    # ---------------------------------
    # 1) EXACT CACHE -> KHÔNG EMBED
    # ---------------------------------
    cached = exact_cache_hit(query_norm)
    if cached:
        print("✅ EXACT CACHE HIT")
        rb = runbooks[cached["best_idx"]]
        return rb

    # ---------------------------------
    # 2) EMBED QUERY (cần cho semantic cache + normal search)
    # ---------------------------------
    print("🔄 Embedding query...")
    query_vector = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True
    )[0].astype("float32")

# Semantic search top-k runbooks theo kiến trúc V2.
def search_RB_topk(query, exclude_titles=None, topk=3, result_threshold=None):
    """
    Semantic search top-k runbooks theo kiến trúc V2.

    Args:
        query (str): câu hỏi user / semantic query
        exclude_titles (list[str]): danh sách title runbook cần loại trừ
        topk (int): số runbook tối đa trả về
        result_threshold (float|None): nếu set thì chỉ lấy runbook có score >= threshold

    Returns:
        list[dict]: danh sách runbook, mỗi item có thêm field "score"
    """
    if exclude_titles is None:
        exclude_titles = []

    exclude_titles_norm = {normalize(t) for t in exclude_titles if t}

    # Embed query
    q_vec = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True
    )[0]

    # Cosine similarity (vì vectors đã normalize)
    scores = np.dot(vectors, q_vec)

    # sort từ cao xuống thấp
    sorted_indices = np.argsort(scores)[::-1]

    results = []

    for idx in sorted_indices:
        rb = runbooks[idx]
        title_norm = normalize(rb["title"])
        score = float(scores[idx])

        # bỏ qua runbook đã thử
        if title_norm in exclude_titles_norm:
            continue

        # threshold nếu có
        if result_threshold is not None and score < result_threshold:
            continue

        item = {
            "title": rb["title"],
            "service": rb["service"],
            "score": score,
            "precheck": rb["precheck"],
            "steps": rb["steps"],
            "postcheck": rb["postcheck"],
            "source_file": rb.get("source_file", "")
        }

        results.append(item)

        if len(results) >= topk:
            break

    return results

    # ---------------------------------
    # 3) SEMANTIC CACHE
    # ---------------------------------
    semantic_item, semantic_score = semantic_cache_hit(query_vector)
    if semantic_item:
        print(f"✅ SEMANTIC CACHE HIT | sim={semantic_score:.4f}")
        rb = runbooks[semantic_item["best_idx"]]

        # thêm exact query mới vào cache luôn
        add_or_update_cache(query, query_norm, query_vector, semantic_item["best_idx"])
        return rb

    print("🔄 Searching KB vectors...")

    scores = np.dot(vectors, query_vector)

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    print(f"✅ best_score={best_score:.4f}")

    # ----- RESULT CONTROL -----
    if best_score < RESULT_THRESHOLD:
        print(f"⚠️ No confident match (score={best_score:.4f})")
        return None

    # ----- CACHE CONTROL -----
    if best_score >= CACHE_THRESHOLD:
        add_or_update_cache(query, query_norm, query_vector, best_idx)
        print(f"✅ Cached (score={best_score:.4f})")
    else:
        print(f"⚠️ Not cached (score={best_score:.4f})")

    # ----- RETURN RESULT -----
    rb = runbooks[best_idx]
    return rb
    

    

# =========================
# PRINT RESULT
# =========================
def print_runbook(rb):
    print("\n" + "=" * 70)
    print(f"📘 Tiêu đề : {rb['title']}")
    print(f"🖥 Dịch vụ : {rb['service']}")

    print("\n🔹 Điều kiện thực hiện")
    if rb.get("precheck"):
        for x in rb["precheck"]:
            print(f"- {x}")
    else:
        print("- (không có dữ liệu)")

    print("\n🔹 Các bước thực hiện")
    if rb.get("steps"):
        for i, s in enumerate(rb["steps"], 1):
            print(f"{i}. {s}")
    else:
        print("- (không có dữ liệu)")

    print("\n🔹 Kiểm tra sau thực hiện")
    if rb.get("postcheck"):
        for x in rb["postcheck"]:
            print(f"- {x}")
    else:
        print("- (không có dữ liệu)")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    try:
        while True:
            q = input("\n🔎 Query: ").strip()
            if not q:
                break

            rb = search_runbook(q)
            print_runbook(rb)

    finally:
        save_json(CACHE_FILE, query_cache)
        print("\n💾 Cache saved")