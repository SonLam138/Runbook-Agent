import json
import re

#from cachetools import cached
from script.RB_query_faiss import retrieve_topk_candidates
from app.llm import call_llm
from app.session_store import get_session, update_session, reset_session
from app.tool import tool_retrieve_candidates


# =====================================================
# CONFIG
# =====================================================
MAX_CLARIFY_TURNS = 3
MAX_RETRY_RUNBOOKS = 3
FEEDBACK_CACHE = {}
FAILURE_SIGNALS = [
    "vẫn bị lỗi",
    "vẫn lỗi",
    "không được",
    "vẫn không được",
    "làm rồi vẫn lỗi",
    "vẫn không login được",
    "vẫn không vào được",
    "chưa được",
    "không ổn",
    "vẫn fail",
    "chưa ổn",
]

NEGATIVE_HINTS = [
    "không",
    "chưa",
    "vẫn",
    "lỗi",
    "sai",
    "fail",
    "not",
    "error",
    "unable"
]

# =====================================================
# BASIC HELPERS
# =====================================================
"""
def save_feedback(query, runbook):
    FEEDBACK_CACHE[query.lower()] = runbook

def check_cache(query):
    q = query.lower()
    for k in FEEDBACK_CACHE:
        if k in q or q in k:
            return FEEDBACK_CACHE[k]
    return None
"""
def normalize_query(q):
    return q.strip().lower()


def save_feedback(query, runbook):
    FEEDBACK_CACHE[normalize_query(query)] = runbook


def check_cache(query):
    q = normalize_query(query)

    for k, v in FEEDBACK_CACHE.items():
        if k in q or q in k:
            return v

    return None


def normalize(text):
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_parse_json(text):
    """
    Parse JSON an toàn từ output LLM.
    """
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print("⚠️ JSON parse error:", e)
    return None


def ensure_state_keys(state: dict):
    """
    Đảm bảo state luôn có đủ key.
    """
    state.setdefault("mode", "idle")  # idle | clarifying
    state.setdefault("original_query", "")
    state.setdefault("clarify_turns", 0)
    state.setdefault("pending_slot", None)

    state.setdefault("slots", {
        "issue_type": None,
        "service": None,
        "error_message": None
    })

    state.setdefault("history", [])

    # semantic memory
    state.setdefault("semantic_query", "")

    # retry / failure handling
    state.setdefault("last_runbook", None)
    state.setdefault("tried_runbooks", [])
    state.setdefault("last_result_status", None)

    # semantic cache theo session
    state.setdefault("semantic_cache", [])

    # debug
    state.setdefault("last_action", None)


def append_history(state, role, text):
    state["history"].append({
        "role": role,
        "text": text
    })
    # giữ tối đa 8 lượt gần nhất cho gọn
    state["history"] = state["history"][-8:]


def reply(session_id, state, message):
    """
    Chuẩn hóa đường ra:
    - append assistant history
    - save session
    - return message
    """
    append_history(state, "assistant", message)
    update_session(session_id, state)
    return message


# =====================================================
# RUNBOOK OUTPUT (RAW, NO LLM REWRITE)
# =====================================================
def format_runbook(rb, prefix=None):
    """
    Trả runbook nguyên bản tiếng Việt.
    Không dùng LLM rewrite.
    """
    if not rb:
        return "❌ Không tìm thấy runbook phù hợp."

    precheck_list = rb.get("precheck", [])
    steps_list = rb.get("steps", [])
    postcheck_list = rb.get("postcheck", [])

    precheck = "\n- ".join(precheck_list) if precheck_list else "(không có)"
    steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps_list)) if steps_list else "(không có)"
    postcheck = "\n- ".join(postcheck_list) if postcheck_list else "(không có)"

    blocks = []
    if prefix:
        blocks.append(prefix)

    blocks.append(
        f"""
📘 Tiêu đề: {rb.get('title', '')}
🖥 Dịch vụ: {rb.get('service', '')}

🔹 Điều kiện thực hiện:
- {precheck}

🔹 Các bước thực hiện:
{steps}

🔹 Kiểm tra sau thực hiện:
- {postcheck}
""".strip()
    )

    return "\n\n".join(blocks)


# =====================================================
# SLOT FILLING / CLARIFY
# =====================================================
def fill_pending_slot(state, user_input):
    pending = state.get("pending_slot")
    if pending in state["slots"]:
        state["slots"][pending] = user_input.strip()


def generate_clarify_message_fallback(next_slot, user_input):
    if next_slot == "issue_type":
        return "Bạn có thể mô tả rõ hơn lỗi hoặc vấn đề bạn đang gặp không?"
    elif next_slot == "service":
        return "Bạn đang thao tác trên hệ thống hoặc dịch vụ nào?"
    elif next_slot == "error_message":
        return "Bạn có thấy mã lỗi hoặc thông báo lỗi cụ thể nào không?"
    else:
        return f"Bạn có thể mô tả rõ hơn về '{user_input}' được không?"


# =====================================================
# SEMANTIC MEMORY / QUERY REFORMULATION
# =====================================================
def build_semantic_query(state):
    """
    Chỉ dùng khi đang clarify hoặc cần refine query.
    Có cache bằng state["semantic_query"] để tránh gọi LLM thừa.
    """
    if state.get("semantic_query"):
        return state["semantic_query"]

    slots = state["slots"]

    history_text = "\n".join(
        f"{item['role']}: {item['text']}" for item in state["history"]
    )

    prompt = f"""
Bạn là IT agent.

Nhiệm vụ:
Từ lịch sử hội thoại và thông tin đã biết, hãy viết lại thành 1 câu query NGẮN GỌN, bằng tiếng Việt,
dùng để tìm runbook phù hợp nhất.

QUAN TRỌNG:
- Chỉ trả về 1 dòng text
- KHÔNG giải thích
- KHÔNG dịch sang tiếng Anh

Original query:
{state["original_query"]}

Known slots:
- issue_type: {slots.get("issue_type")}
- service: {slots.get("service")}
- error_message: {slots.get("error_message")}

History:
{history_text}
"""

    raw = call_llm(prompt)
    semantic_query = (raw or "").strip()

    if not semantic_query:
        parts = [state["original_query"]]
        for key in ["issue_type", "service", "error_message"]:
            val = slots.get(key)
            if val:
                parts.append(val)

        seen = set()
        out = []
        for p in parts:
            p_norm = normalize(p)
            if p_norm and p_norm not in seen:
                seen.add(p_norm)
                out.append(p.strip())

        semantic_query = " ".join(out)

    state["semantic_query"] = semantic_query
    return semantic_query


def remember_success(state, final_query, rb):
    """
    Ghi nhớ kết quả thành công trong session.
    """
    if not rb:
        return

    title = rb.get("title")
    if title and title not in state["tried_runbooks"]:
        state["tried_runbooks"].append(title)

    state["last_runbook"] = {
        "title": rb.get("title"),
        "service": rb.get("service"),
        "source_file": rb.get("source_file", "")
    }
    state["last_result_status"] = "returned"
    state["last_action"] = "search"

    cache_item = {
        "semantic_query": final_query,
        "runbook_title": rb.get("title"),
        "service": rb.get("service")
    }
    state["semantic_cache"].append(cache_item)
    state["semantic_cache"] = state["semantic_cache"][-5:]

    state["semantic_query"] = final_query


def get_latest_semantic_query(state):
    if state.get("semantic_query"):
        return state["semantic_query"]

    if state.get("semantic_cache"):
        return state["semantic_cache"][-1].get("semantic_query", "")

    return ""


# =====================================================
# FAILURE DETECTION (HYBRID)
# =====================================================
def rule_detect_failure(user_input: str):
    q = normalize(user_input)
    return any(sig in q for sig in FAILURE_SIGNALS)


def has_negative_hint(user_input: str):
    q = normalize(user_input)
    return any(h in q for h in NEGATIVE_HINTS)


def llm_detect_failure(user_input: str, state: dict):
    """
    Chỉ gọi LLM detect failure khi:
    - đã có runbook trước đó
    - action trước là search
    - mode đang idle
    - input có dấu hiệu tiêu cực
    """
    if not state.get("last_runbook"):
        return False

    if state.get("last_action") != "search":
        return False

    if state.get("mode") != "idle":
        return False

    if state.get("last_result_status") not in ["returned", "retry_returned"]:
        return False

    if not has_negative_hint(user_input):
        return False

    prompt = f"""
Bạn là AI agent.

Ngữ cảnh:
- Agent đã cung cấp runbook cho user trước đó.
- Chỉ coi là FAILURE nếu user THỂ HIỆN RÕ rằng giải pháp trước không hiệu quả.

Last runbook:
- title: {state.get("last_runbook", {}).get("title")}
- service: {state.get("last_runbook", {}).get("service")}

User nói:
"{user_input}"

Câu hỏi:
User có đang PHẢN HỒI THẤT BẠI về runbook trước đó không?

Chỉ trả:
YES hoặc NO

Lưu ý:
- Nếu user chỉ đang nêu một keyword / chủ đề mới / runbook mới
  → trả NO
- Chỉ trả YES khi có dấu hiệu rõ ràng như:
  "vẫn lỗi", "không được", "chưa được", "làm rồi vẫn fail"
"""

    try:
        raw = call_llm(prompt)
        result = (raw or "").strip().upper()
        print("🧠 RAW failure detection:", result)
        return "YES" in result
    except Exception as e:
        print("⚠️ LLM failure detection error:", e)
        return False


def is_failure_feedback(user_input: str, state: dict):
    """
    Hybrid:
    1. Rule match exact phrases
    2. Nếu không match rule:
       - chỉ gọi LLM khi đúng context failure
    """
    if rule_detect_failure(user_input):
        print("✅ FAILURE detected by RULE")
        return True

    if llm_detect_failure(user_input, state):
        print("✅ FAILURE detected by LLM")
        return True

    return False


# =====================================================
# CANDIDATE RETRIEVAL (METADATA ONLY FOR DECISION)
# =====================================================
def retrieve_candidates_meta(query, state):
    """
    Candidate retrieval:
    - dùng vector search top-k
    - nhưng decision layer chỉ nhìn metadata
    - không thực thi 'search runbook' theo nghĩa business
    """
    exclude_titles = state.get("tried_runbooks", [])
    full_candidates = retrieve_topk_candidates(
        query=query,
        exclude_titles=exclude_titles,
        topk=3
    )


    meta_candidates = []
    for i, rb in enumerate(full_candidates, start=1):
        meta_candidates.append({
            "idx": i,  # 1-based cho LLM dễ chọn
            "title": rb.get("title", ""),
            "service": rb.get("service", ""),
            "keyword": rb.get("keyword", ""),
            "description": rb.get("description", ""),
            "score": rb.get("score", 0.0)
        })

    return full_candidates, meta_candidates


# =====================================================
# LLM DECISION WITH CANDIDATES
# =====================================================
def decide_with_candidates(user_input, state, meta_candidates):
    """
    LLM không quyết định trong 'bóng tối'.
    Nó nhìn:
    - query hiện tại
    - state
    - candidate metadata
    rồi quyết định:
    - search
    - ask_more
    """
    if not meta_candidates:
        return {
            "action": "ask_more",
            "selected_index": None,
            "message": "Tôi chưa tìm thấy runbook gần phù hợp. Bạn có thể mô tả rõ hơn không?",
            "next_slot": "issue_type"
        }

    history_text = "\n".join(
        f"{item['role']}: {item['text']}" for item in state.get("history", [])
    )

    candidate_text = ""
    for c in meta_candidates:
        candidate_text += f"""
[{c['idx']}]
Title: {c['title']}
Service: {c['service']}
Keyword: {c['keyword']}
Description: {c['description']}
Score: {c['score']:.4f}
"""

    prompt = f"""
Bạn là IT agent nội bộ.

Ngữ cảnh hội thoại:
- original_query: {state.get("original_query", "")}
- mode: {state.get("mode")}
- pending_slot: {state.get("pending_slot")}
- issue_type: {state.get("slots", {}).get("issue_type")}
- service: {state.get("slots", {}).get("service")}
- error_message: {state.get("slots", {}).get("error_message")}

History:
{history_text}

User query mới nhất:
"{user_input}"

Candidate runbook metadata (semantic pre-retrieval):
{candidate_text}

NHIỆM VỤ:
1. Ưu tiên so sánh query hiện tại (và context nếu có) với Title + Service + Keyword của candidate theo NGỮ NGHĨA.
2. Nếu có 1 candidate phù hợp rõ ràng:
   - action = "search"
   - selected_index = số thứ tự candidate phù hợp nhất (1..N)
   - message = ""
3. Nếu query vẫn mơ hồ hoặc chưa đủ chắc để chọn candidate:
   - action = "ask_more"
   - selected_index = null
   - message = câu hỏi làm rõ bằng tiếng Việt
   - next_slot = issue_type | service | error_message

QUAN TRỌNG:
- Không dùng từ tiếng Anh nếu user đang hỏi tiếng Việt
- Không dịch query
- Chỉ chọn "search" khi thực sự thấy candidate phù hợp rõ
- Nếu còn nghi ngờ, hãy chọn "ask_more"

CHỈ TRẢ JSON:
{{
  "action": "search hoặc ask_more",
  "selected_index": số hoặc null,
  "message": "...",
  "next_slot": "issue_type | service | error_message | null"
}}
"""

    raw = call_llm(prompt)
    print("\n🧠 RAW candidate decision:", raw)

    parsed = safe_parse_json(raw)
    if parsed:
        print("✅ PARSED candidate decision:", parsed)
        return parsed

    return {
        "action": "ask_more",
        "selected_index": None,
        "message": "Bạn có thể mô tả rõ hơn vấn đề bạn đang gặp không?",
        "next_slot": "issue_type"
    }


# =====================================================
# FAILURE RETRY
# =====================================================
def handle_failure_retry(session_id, state, user_input):
    """
    Retry bằng cách:
    - reuse semantic query
    - retrieve candidates (FAISS)
    - loại RB đã thử
    """

    tried = state.get("tried_runbooks", [])

    # Guard
    if not state.get("semantic_query"):
        return None

    if len(tried) >= 3:
        return reply(
            session_id,
            state,
            "❌ Tôi đã thử một số hướng nhưng chưa tìm được runbook phù hợp. "
            "Bạn có thể mô tả chi tiết hơn không?"
        )

    semantic_query = state["semantic_query"]

    print(f"🔁 RETRY with query: {semantic_query}")
    print(f"🚫 exclude: {tried}")

    # ✅ dùng FAISS candidate retrieval
    full_candidates, _ = tool_retrieve_candidates(semantic_query, state)

    if not full_candidates:
        return reply(
            session_id,
            state,
            "❌ Tôi chưa tìm thấy runbook phù hợp. Bạn có thể mô tả rõ hơn không?"
        )

    rb = full_candidates[0]   # lấy candidate tiếp theo

    # ✅ cập nhật state
    remember_success(state, semantic_query, rb)
    state["last_result_status"] = "retry_returned"

    return reply(
        session_id,
        state,
        "⚠️ Thử hướng khác:\n\n" + format_runbook(rb)
    )


def handle_failure_if_needed(session_id, state, user_input):
    if not is_failure_feedback(user_input, state):
        return None

    print("⚠️ FAILURE FEEDBACK DETECTED")
    return handle_failure_retry(session_id, state, user_input)


# =====================================================
# CLARIFY EXECUTION
# =====================================================
def execute_clarify(session_id, state, user_input, decision):
    if state["mode"] == "clarifying":
        state["clarify_turns"] += 1

        if state["clarify_turns"] > MAX_CLARIFY_TURNS:
            reset_session(session_id)
            return "❌ Tôi vẫn chưa đủ thông tin để xác định runbook phù hợp. Bạn vui lòng nêu rõ hệ thống và lỗi cụ thể giúp tôi."

    msg = decision.get("message")
    next_slot = decision.get("next_slot")

    if not msg or not msg.strip():
        msg = generate_clarify_message_fallback(next_slot, user_input)

    state["mode"] = "clarifying"
    state["pending_slot"] = next_slot
    state["last_action"] = "ask_more"

    if not state.get("original_query"):
        state["original_query"] = user_input
        state["clarify_turns"] = 1

    return reply(session_id, state, msg)


# =====================================================
# SEARCH EXECUTION
# =====================================================
def execute_search_from_candidates(session_id, state, user_input, decision, full_candidates, effective_query):
    selected_index = decision.get("selected_index")

    if selected_index is None:
        msg = "❌ Tôi chưa xác định được runbook phù hợp. Bạn có thể mô tả rõ hơn không?"
        return reply(session_id, state, msg)

    try:
        selected_index = int(selected_index)
    except Exception:
        selected_index = None

    if not selected_index or selected_index < 1 or selected_index > len(full_candidates):
        msg = "❌ Tôi chưa chọn được runbook phù hợp từ danh sách ứng viên. Bạn có thể mô tả rõ hơn không?"
        return reply(session_id, state, msg)

    rb = full_candidates[selected_index - 1]

    remember_success(state, effective_query, rb)
    handle_success_note = effective_query  # for clarity, no extra logic needed
    _ = handle_success_note

    answer = format_runbook(rb)
    return reply(session_id, state, answer)


# =====================================================
# MAIN AGENT RUNTIME
# =====================================================
def run_agent(session_id, user_input):
    state = get_session(session_id)
    ensure_state_keys(state)

    append_history(state, "user", user_input)

    # ===== CACHE HIT =====
    cached = check_cache(user_input)
    if cached:
        print("⚡ CACHE HIT")
        return reply(session_id, state, format_runbook(cached))

    # 1) failure handling
    failure_response = handle_failure_if_needed(session_id, state, user_input)
    if failure_response:
        return failure_response

    # 2) nếu đang clarifying thì fill slot
    if state["mode"] == "clarifying":
        fill_pending_slot(state, user_input)

    # 3) build effective query
    if state["mode"] == "clarifying":
        effective_query = build_semantic_query(state)
    else:
        effective_query = user_input

    # ✅ FIX: không overwrite semantic_query bừa
    if state["mode"] == "clarifying" or not state.get("semantic_query"):
        state["semantic_query"] = effective_query

    # 4) candidate retrieval
    full_candidates, meta_candidates = tool_retrieve_candidates(
        effective_query,
        state
    )

    # 5) decision
    decision = decide_with_candidates(user_input, state, meta_candidates)
    action = decision.get("action")

    # 6) execute
    if action == "search":
        return execute_search_from_candidates(
            session_id=session_id,
            state=state,
            user_input=user_input,
            decision=decision,
            full_candidates=full_candidates,
            effective_query=effective_query
        )

    if action == "ask_more":
        return execute_clarify(
            session_id=session_id,
            state=state,
            user_input=user_input,
            decision=decision
        )

    return reply(session_id, state,
        "❌ Tôi chưa hiểu rõ yêu cầu. Bạn có thể mô tả cụ thể hơn không?")