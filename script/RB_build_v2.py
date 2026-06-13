import re
import json
from pathlib import Path
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer

# =======================
# CONFIG
# =======================
RUNBOOK_DIR = Path(r"D:\Runbook")
OUTPUT_DIR = RUNBOOK_DIR / "_kb_v2"

MODEL_PATH = Path(r"D:\bge-m3")

DATA_FILE = OUTPUT_DIR / "runbook_data.json"
VECTOR_FILE = OUTPUT_DIR / "runbook_vectors.npy"

# =======================
# UTILS
# =======================
def normalize(text):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# =======================
# INTENT GENERATION (AUTO - NO HARDCODE)
# =======================
def generate_intents(keyword: str, description: str = "", max_intents=5):
    candidates = []

    # ---- từ keyword ----
    if keyword:
        kws = [k.strip().lower() for k in re.split(r"[,;]+", keyword)]
        candidates.extend(kws)

    # ---- từ description ----
    if description:
        desc = description.lower()
        parts = re.split(r"[.;,\n]+", desc)

        for p in parts:
            p = normalize(p)
            if 2 <= len(p.split()) <= 8:
                candidates.append(p)

    # ---- remove duplicate + giữ thứ tự ----
    seen = set()
    intents = []
    for c in candidates:
        c = normalize(c)
        if not c:
            continue
        if c not in seen:
            seen.add(c)
            intents.append(c)

    # ---- limit max intents ----
    return intents[:max_intents]


# =======================
# DETECT SECTION
# =======================
def detect_section(text):
    t = text.lower()

    if "thông tin" in t:
        return "info"
    if "điều kiện" in t:
        return "precheck"
    if "các bước" in t:
        return "steps"
    if "kiểm tra" in t:
        return "postcheck"

    return None


# =======================
# PARSE DOCX
# =======================
def parse_docx(file):
    doc = Document(file)

    sections = {
        "info": [],
        "precheck": [],
        "steps": [],
        "postcheck": []
    }

    current_section = None

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        sec = detect_section(text)

        if sec:
            current_section = sec
            continue

        if current_section:
            sections[current_section].append(text)

    # ---- parse info ----
    info_map = {}

    for line in sections["info"]:
        if ":" in line:
            k, v = line.split(":", 1)
            info_map[k.strip().lower()] = normalize(v)

    title = info_map.get("tiêu đề", file.stem)
    service = info_map.get("dịch vụ", "")
    keyword = info_map.get("keyword", "")
    description = info_map.get("kịch bản sử dụng", "")

    # ---- steps parser ----
    def parse_steps(lines):
        steps = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            m = re.match(r"^\d+[\.\)]\s*(.*)", line)
            if m:
                steps.append(normalize(m.group(1)))
            else:
                steps.append(normalize(line))

        return steps

    # ---- generate intents per runbook ----
    intents = generate_intents(keyword, description)

    return {
        "title": normalize(title),
        "service": normalize(service),
        "keyword": normalize(keyword),
        "description": normalize(description),
        "intents": intents,   # ✅ lưu intent để debug
        "precheck": sections["precheck"],
        "steps": parse_steps(sections["steps"]),
        "postcheck": sections["postcheck"],
        "source_file": str(file)
    }


# =======================
# BUILD EMBEDDING TEXT
# =======================
def build_embedding_text(rb):
    intent_block = "\n".join(f"- {i}" for i in rb.get("intents", []))

    return f"""
Runbook: {rb['title']}
Service: {rb['service']}

Keyword:
{rb['keyword']}

Description:
{rb['description']}

User can ask:
{intent_block}
""".strip()


# =======================
# MAIN BUILD
# =======================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    runbooks = []

    print("🔄 Parsing DOCX...")

    for f in RUNBOOK_DIR.rglob("*.docx"):
        if "_kb" in str(f):
            continue

        try:
            rb = parse_docx(f)
            runbooks.append(rb)

            print(f"✅ {f.name}")
            print(f"   intents: {rb['intents']}")

        except Exception as e:
            print(f"❌ {f.name}: {e}")

    if not runbooks:
        print("❌ No runbook found")
        return

    # ---- save data ----
    with open(DATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(runbooks, fh, ensure_ascii=False, indent=2)

    print(f"✅ Saved {DATA_FILE}")

    # ---- embedding ----
    print("🔄 Loading model...")
    model = SentenceTransformer(str(MODEL_PATH), trust_remote_code=True)

    texts = [build_embedding_text(rb) for rb in runbooks]

    print("🔄 Encoding vectors...")
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=8
    )

    vectors = np.array(vectors, dtype="float32")

    np.save(VECTOR_FILE, vectors)

    print(f"✅ Saved vectors: {VECTOR_FILE}")
    print("\n🎉 BUILD V2 DONE")


if __name__ == "__main__":
    main()