"""
ingest.py
data/ klasöründeki .txt belgelerini okur, parçalara (chunk) böler,
her belge için SHA-256 alır, şüpheli talimat (prompt injection) taraması
yapar ve sonucu SQLite'a kaydeder.

Çalıştırma sırası: 1) ingest.py  2) embed_documents.py  3) rag_app.py
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path

from rag_core import DB_PATH, find_security_flags, get_connection

DATA_DIR = Path("data")
MAX_CHARS_PER_CHUNK = 700
OVERLAP_CHARS = 120


def sha256_of_file(file_path):
    content = file_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def split_into_chunks(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sentence in sentences:
        if len(sentence) > MAX_CHARS_PER_CHUNK:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(sentence), MAX_CHARS_PER_CHUNK):
                chunks.append(sentence[i:i + MAX_CHARS_PER_CHUNK])
            continue

        if len(current) + len(sentence) + 1 <= MAX_CHARS_PER_CHUNK:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            overlap_text = current[-OVERLAP_CHARS:] if len(current) > OVERLAP_CHARS else current
            current = overlap_text + sentence + " "

    if current.strip():
        chunks.append(current.strip())

    return chunks


def create_tables(connection):
    connection.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            sha256 TEXT NOT NULL,
            security_flags TEXT,
            ingested_at TEXT NOT NULL
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding_json TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)


def ingest_documents():
    if not DATA_DIR.exists():
        print(f"[Hata] '{DATA_DIR}' klasörü bulunamadı. Önce belgelerinizi bu klasöre koyun.")
        return

    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if not txt_files:
        print(f"[Hata] '{DATA_DIR}' klasöründe .txt dosyası bulunamadı.")
        return

    with get_connection() as connection:
        create_tables(connection)
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM documents")

        document_count = 0
        chunk_count = 0

        for file_path in txt_files:
            text = file_path.read_text(encoding="utf-8")
            file_hash = sha256_of_file(file_path)
            flags = find_security_flags(text)
            flags_text = ", ".join(flags) if flags else "Yok"

            cursor = connection.execute(
                """
                INSERT INTO documents (filename, sha256, security_flags, ingested_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    file_path.name,
                    file_hash,
                    flags_text,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

            document_id = cursor.lastrowid
            chunks = split_into_chunks(text)

            for index, chunk in enumerate(chunks, start=1):
                connection.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content)
                    VALUES (?, ?, ?)
                    """,
                    (document_id, index, chunk),
                )
                chunk_count += 1

            document_count += 1
            print(
                f"✓ {file_path.name}: {len(chunks)} parça | "
                f"SHA-256: {file_hash[:12]}... | "
                f"Şüpheli talimat: {flags_text}"
            )

        connection.commit()

    print(f"\nTamamlandı: {document_count} belge, {chunk_count} parça SQLite'a kaydedildi.")
    print(f"Veritabanı dosyası: {DB_PATH}")
    print("Sıradaki adım: 'python embed_documents.py' çalıştırın.")


if __name__ == "__main__":
    ingest_documents()
