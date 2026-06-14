from script.RB_query_faiss import faiss_search_topk

RESULT_THRESHOLD = 0.50

def search_RB_topk(query, exclude_titles=None, topk=3, result_threshold=RESULT_THRESHOLD):
    """
    Wrapper cho FAISS top-k full runbook.
    """
    return faiss_search_topk(
        query=query,
        exclude_titles=exclude_titles or [],
        topk=topk,
        result_threshold=result_threshold
    )


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


def retrieve_candidates_meta(query, state, topk=3):
    """
    Candidate retrieval cho decision layer:
    - search bằng FAISS
    - nhưng chỉ return metadata cho LLM decide
    - vẫn giữ full_candidates để khi LLM chọn xong thì return luôn
    """
    exclude_titles = state.get("tried_runbooks", [])

    full_candidates = search_RB_topk(
        query=query,
        exclude_titles=exclude_titles,
        topk=topk
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