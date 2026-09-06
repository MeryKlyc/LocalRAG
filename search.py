"""
search.py
Sadece kaynak arama / skor debug aracı. Cevap ÜRETMEZ, sohbet (chat) modelini
yüklemez — sadece embedding modeliyle en ilgili parçaları ve benzerlik
skorlarını gösterir. Skorların neden düşük/yüksek çıktığını hızlıca kontrol
etmek içindir. Gerçek cevap üretmek için rag_app.py kullanın.
"""

from foundry_local_sdk import Configuration, FoundryLocalManager

from rag_core import (
    EMBEDDING_MODEL,
    TOP_K,
    confidence_label,
    fetch_chunks_with_embeddings,
    get_connection,
    rank_chunks,
)


def main():
    question = input("Sorunuzu yazın: ").strip()

    if not question:
        print("Boş soru gönderilemez.")
        return

    with get_connection() as connection:
        rows = fetch_chunks_with_embeddings(connection)

    if not rows:
        print(
            "Veritabanında kullanılabilir parça yok. "
            "Sırasıyla 'python ingest.py' ve 'python embed_documents.py' çalıştırın."
        )
        return

    FoundryLocalManager.initialize(
        Configuration(app_name="secure_local_rag_search")
    )
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(EMBEDDING_MODEL)
    model.download(lambda p: None)
    model.load()
    try:
        client = model.get_embedding_client()
        question_embedding = client.generate_embedding(question).data[0].embedding
    finally:
        model.unload()

    top_results = rank_chunks(question_embedding, rows, top_k=TOP_K)

    print("\n--- En ilgili kaynaklar ---")
    for index, result in enumerate(top_results, start=1):
        print(f"\n{index}. Kaynak: {result['filename']}")
        print(
            f"   Benzerlik: %{result['score'] * 100:.1f} | "
            f"Güven: {confidence_label(result['score'])}"
        )
        print(f"   Metin: {result['content']}")


if __name__ == "__main__":
    main()
