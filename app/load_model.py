from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_PATH = Path(r"D:\bge-m3")

# ✅ LOAD ONCE
embedding_model = SentenceTransformer(str(MODEL_PATH), trust_remote_code=True)