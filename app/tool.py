from script.RB_query_faiss import (
    search_RB,
    search_RB_topk,
    retrieve_candidate_meta,
)

# =====================================================
# TOOL LAYER
# =====================================================

def tool_search_runbook(query):
    """
    Dùng khi agent đã quyết định search.
    Trả về 1 runbook tốt nhất.
    """
    return search_RB(query)


def tool_search_topk(query, exclude_titles=None, topk=3):
    """
    Dùng cho:
    - failure retry
    - debug search
    """
    return search_RB_topk(
        query=query,
        exclude_titles=exclude_titles or [],
        topk=topk
    )


def tool_retrieve_candidates(query, state):
    """
    Dùng cho decision layer:
    - chỉ lấy metadata candidates
    - KHÔNG phải execute search
    """
    exclude_titles = state.get("tried_runbooks", [])

    full_candidates, meta_candidates = retrieve_candidate_meta(
        query=query,
        exclude_titles=exclude_titles,
        topk=3
    )

    return full_candidates, meta_candidates