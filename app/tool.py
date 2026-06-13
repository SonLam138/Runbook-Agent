import sys
sys.path.append("../scripts")

from script.RB_query_v2_cache import search_runbook,search_RB_topk
RESULT_THRESHOLD = 0.5
"""def search_RB(query):
    results = search_RB_topk(
        query=query,
        exclude_titles=None,
        topk=1,
        result_threshold=RESULT_THRESHOLD
    )

    if not results:
        return None

    return results[0] """

def retrieve_candidate_meta(query, exclude_titles=None, topk=3):
    """
    Candidate retrieval: chỉ dùng metadata để decision.
    Không trả full runbook cho LLM.
    """
    exclude_titles = exclude_titles or []

    results = search_RB_topk(
        query=query,
        exclude_titles=exclude_titles,
        topk=topk
    )

    candidates = []
    for rb in results:
        candidates.append({
            "title": rb.get("title", ""),
            "service": rb.get("service", ""),
            "keyword": rb.get("keyword", ""),
            "description": rb.get("description", ""),
            "score": rb.get("score", 0.0)
        })

    return candidates


def get_runbook_by_title(query, exclude_titles=None, topk=3):
    """
    Lấy full runbook tương ứng title.
    Vì search_RB_topk hiện đã trả full object,
    ta reuse top-k rồi match title.
    """
    exclude_titles = exclude_titles or []

    results = search_RB_topk(
        query=query,
        exclude_titles=exclude_titles,
        topk=topk
    )
    return results


def get_runbook_by_selected_title(selected_title, query_context="", exclude_titles=None):
    """
    Lấy full runbook theo title đã được LLM chọn.
    Nếu title không khớp exact, fallback theo query_context.
    """
    exclude_titles = exclude_titles or []

    # Lấy một ít candidate full để match title
    results = search_RB_topk(
        query=query_context or selected_title,
        exclude_titles=exclude_titles,
        topk=5
    )

    selected_norm = selected_title.strip().lower()

    for rb in results:
        if rb.get("title", "").strip().lower() == selected_norm:
            return rb

    # fallback: match contains
    for rb in results:
        title_norm = rb.get("title", "").strip().lower()
        if selected_norm in title_norm or title_norm in selected_norm:
            return rb

    return None