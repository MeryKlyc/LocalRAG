LocalRAGAssistant

Yerelde (tamamen offline) çalışan, Microsoft Foundry Local kullanan bir RAG
(Retrieval-Augmented Generation) tabanlı soru-cevap asistanı. Kurum içi
siber güvenlik politika belgelerine dayanarak sorulara cevap verir; hiçbir
veri internete çıkmaz.

Özellikler

- Tamamen yerel/offline çalışır (Microsoft Foundry Local ile on-device LLM)
- RAG: sorular, yerel belge koleksiyonundan (SQLite + embedding) en ilgili
  parçalar bulunarak cevaplanır
- Her cevap, kaynağa ne kadar "emin" olunduğunu gösteren bir güven skoru
  (Yüksek / Orta / Düşük) ile birlikte sunulur
- Yeterli kaynak bulunamazsa model tahmin üretmez, kullanıcıyı gerçek bir
  uzmana yönlendirir
- Prompt injection savunması: hem kaynak belgeler hem kullanıcının sorusu
  şüpheli talimatlara karşı taranır; şüpheli kaynaklar cevaba dahil edilmez
- Her cevabın altında hangi belgeden geldiğini gösteren bir "kanıt makbuzu"
 Kurulum


pip install foundry-local-sdk
pip install streamlit   # (web arayüzü istersen, opsiyonel)


 Kullanım sırası

Bu sıra önemlidir, atlamayın:

1. python ingest.py — data/ klasöründeki .txt belgeleri parçalayıp
   SHA-256 alarak ve şüpheli talimat taraması yaparak SQLite'a kaydeder.
2. python embed_documents.py — her belge parçası için embedding üretir.
3. Sorgulama için ikisinden birini kullan:
   - python rag_app.py — terminal üzerinden tek soru sorar
   - streamlit run streamlit_app.py — tarayıcı üzerinden web arayüzü açar

Test


python test_sorular.py


Belgelerin her birine karşılık gelen 10 örnek soruyla sistemin tamamını
(arama + cevap üretimi) tek seferde otomatik test eder.

Proje yapısı

| Dosya | Görevi |
|---|---|
| rag_core.py | Ortak fonksiyonlar: benzerlik hesaplama, güven eşikleri, güvenlik taraması, veritabanı erişimi |
| ingest.py | Belgeleri okur, parçalara böler, SQLite'a kaydeder |
| embed_documents.py | Her parça için embedding üretir |
| rag_app.py | Terminal arayüzü |
| streamlit_app.py | Web arayüzü |
| test_sorular.py | Otomatik test scripti (10 soru) |
| data/ | Kaynak belgeler (.txt) |
| secure_local_rag.db | SQLite veritabanı (ingest sonrası otomatik oluşur) |

 Mimari

Kullanıcı sorusu → embedding modeli (qwen3-embedding-0.6b) ile vektöre
çevrilir → SQLite'taki belge parçalarının embedding'leriyle cosine
similarity hesaplanır → en ilgili parçalar (context) sohbet modeline
(Phi-3.5-mini) sistem promptuyla birlikte verilir → model, yalnızca verilen
kaynaklara dayanarak cevap üretir.

Bilinen sınırlamalar

- Kullanılan yerel dil modeli (Phi-3.5-mini) küçük ölçekli olduğu için,
  cevaplarda zaman zaman dil bilgisi hataları olabiliyor.
- Bu embedding modelinde benzerlik skorları doğası gereği %30-65 aralığında
  toplanıyor; güven eşikleri (rag_core.py içindeki MINIMUM_CONFIDENCE)
  gerçek test sorularıyla bu dağılıma göre kalibre edilmiştir.
- Küçük belge koleksiyonu (10 belge) nedeniyle kapsam dışı sorularda sistem
  doğru şekilde "yeterli kaynak yok" diyerek cevap vermeyi reddediyor.