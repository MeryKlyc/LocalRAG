"""
rag_core.py
SecureLocal RAG projesinin ortak/paylaşılan fonksiyonları ve ayarları.
ingest.py, embed_documents.py, rag_app.py ve search.py bu modülü kullanır.
Amaç: aynı kodu (cosine similarity, güven etiketi, güvenlik taraması) birden
fazla dosyada tekrar etmemek.
"""

import json
import math
import sqlite3
from contextlib import contextmanager

# --- Ortak ayarlar ---------------------------------------------------------

DB_PATH = "secure_local_rag.db"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "Phi-3.5-mini-instruct-generic-cpu:2"

TOP_K = 3

# UYARI: Bu eşik ilk tahmindir. Belgelerinizi ingest edip embed ettikten
# sonra bilinen 5-6 soruyla test edin, çıkan gerçek skor dağılımına göre
# bu sayıyı ayarlayın (README'de "eşik kalibrasyonu" bölümüne bakın).
MINIMUM_CONFIDENCE = 0.30

SUSPICIOUS_PATTERNS = [
    "önceki talimatları yok say",
    "ignore previous instructions",
    "system prompt",
    "gizli talimat",
    "yeni talimat ver",
    "disregard the above",
    "you are now",
]


# --- Veritabanı bağlantısı --------------------------------------------------

@contextmanager
def get_connection():
    """`with get_connection() as connection:` şeklinde kullanın.
    Fonksiyon içinde erken `return` edilse bile bağlantı otomatik kapanır
    (eski koddaki 'bağlantı hiç kapanmıyor' hatasını önler)."""
    connection = sqlite3.connect(DB_PATH)
    try:
        yield connection
    finally:
        connection.close()


def fetch_chunks_with_embeddings(connection):
    """chunks + documents tablolarını birleştirip sadece embedding'i
    hesaplanmış satırları döner. embed_documents.py henüz çalıştırılmadıysa
    (embedding_json = NULL) o satırları atlar ve kullanıcıyı uyarır;
    eskisi gibi TypeError ile çökmez."""
    rows = connection.execute(
        """
        SELECT chunks.content, chunks.embedding_json,
               documents.filename, documents.security_flags
        FROM chunks
        JOIN documents ON chunks.document_id = documents.id
        """
    ).fetchall()

    usable_rows = [row for row in rows if row[1] is not None]
    missing = len(rows) - len(usable_rows)
    if missing:
        print(
            f"[Uyarı] {missing} parçanın embedding'i yok. "
            "'python embed_documents.py' komutunu çalıştırmayı unutmayın."
        )
    return usable_rows


# --- Benzerlik hesaplama -----------------------------------------------------

def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def confidence_label(score):
    if score >= 0.55:
        return "Yüksek"
    if score >= 0.35:
        return "Orta"
    return "Düşük"


def rank_chunks(question_embedding, rows, top_k=TOP_K):
    """rows: fetch_chunks_with_embeddings() çıktısı.
    Döner: skora göre büyükten küçüğe sıralı, en fazla top_k sonuç."""
    results = []

    for content, embedding_json, filename, security_flags in rows:
        chunk_embedding = json.loads(embedding_json)
        score = cosine_similarity(question_embedding, chunk_embedding)
        results.append(
            {
                "content": content,
                "filename": filename,
                "security_flags": security_flags,
                "score": score,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


# --- Güvenlik: prompt injection taraması -------------------------------------

def find_security_flags(text):
    """Metinde (belge içeriği ya da kullanıcının kendi sorusu) şüpheli
    talimat kalıpları arar. .casefold() Türkçe büyük/küçük harf
    dönüşümünde .lower()'dan biraz daha güvenlidir."""
    lower_text = text.casefold()
    return [pattern for pattern in SUSPICIOUS_PATTERNS if pattern in lower_text]
