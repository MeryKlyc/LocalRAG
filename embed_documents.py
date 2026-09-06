"""
embed_documents.py
ingest.py tarafından SQLite'a yazılan her metin parçası (chunk) için
Foundry Local üzerinden embedding üretir ve veritabanına kaydeder.

Çalıştırma sırası: 1) ingest.py  2) embed_documents.py  3) rag_app.py
"""

import json
import sqlite3

from foundry_local_sdk import Configuration, FoundryLocalManager

from rag_core import EMBEDDING_MODEL, get_connection


def main():
    with get_connection() as connection:
        try:
            connection.execute("ALTER TABLE chunks ADD COLUMN embedding_json TEXT")
            connection.commit()
        except sqlite3.OperationalError:
            pass  # Sütun zaten varsa buraya düşer, sorun değil.

        rows = connection.execute(
            "SELECT id, content FROM chunks ORDER BY id"
        ).fetchall()

        if not rows:
            print("Belge parçası bulunamadı. Önce 'python ingest.py' çalıştırın.")
            return

        texts = [row[1] for row in rows]

        FoundryLocalManager.initialize(
            Configuration(app_name="secure_local_rag_embeddings")
        )
        manager = FoundryLocalManager.instance

        model = manager.catalog.get_model(EMBEDDING_MODEL)

        print("Embedding modeli indiriliyor...")
        model.download(
            lambda p: print(f"\rİndirme: %{p:.1f}", end="", flush=True)
        )
        print()

        model.load()
        try:
            client = model.get_embedding_client()

            print(f"{len(texts)} parça embedding'e dönüştürülüyor...")
            response = client.generate_embeddings(texts)

            for row, item in zip(rows, response.data):
                connection.execute(
                    "UPDATE chunks SET embedding_json = ? WHERE id = ?",
                    (json.dumps(item.embedding), row[0]),
                )

            connection.commit()

            print(
                f"Tamamlandı: {len(rows)} parça, "
                f"{len(response.data[0].embedding)} boyutlu embedding kaydedildi."
            )
            print("Sıradaki adım: 'python rag_app.py' çalıştırın.")
        finally:
            model.unload()


if __name__ == "__main__":
    main()
