from foundry_local_sdk import Configuration, FoundryLocalManager


def main():
    FoundryLocalManager.initialize(
        Configuration(app_name="secure_local_rag")
    )
    manager = FoundryLocalManager.instance

    # Genel takma ad yerine açıkça instruct + CPU varyantı seçiyoruz.
    model = manager.catalog.get_model("phi-3.5-mini")
    print("Model hazırlanıyor...")
    model.download(
        lambda progress: print(
            f"\rİndirme: %{progress:.1f}",
            end="",
            flush=True,
        )
    )
    print()

    model.load()
    client = model.get_chat_client()

    # Kısa ve daha kararlı test ayarları
    client.settings.temperature = 0.2
    client.settings.max_tokens = 80

    response = client.complete_chat(
        [
            {
                "role": "user",
                "content": "Explain RAG in one short sentence.",
            }
        ]
    )

    print("\nAsistan:")
    print(response.choices[0].message.content)

    model.unload()
    print("\nModel kapatıldı.")


if __name__ == "__main__":
    main()