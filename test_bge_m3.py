
import time

import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("Loading BGE-M3...")
    start = time.perf_counter()

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16},
    )

    print("Model loaded successfully!")
    print("Device:", model.device)
    print("Load time:", round(time.perf_counter() - start, 2), "seconds")

    sentences = [
        "Artificial intelligence can connect to external tools.",
        "يمكن للذكاء الاصطناعي الاتصال بالأدوات الخارجية.",
        "The weather is sunny today.",
    ]

    embeddings = model.encode(
        sentences,
        batch_size=3,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("Embedding shape:", embeddings.shape)

    similarities = embeddings @ embeddings.T

    print("English-Arabic similarity:", round(float(similarities[0, 1]), 4))
    print("English-Weather similarity:", round(float(similarities[0, 2]), 4))

    print(
        "GPU allocated memory:",
        round(torch.cuda.memory_allocated() / 1024**2, 2),
        "MiB",
    )

    print("BGE-M3 test completed successfully!")


if __name__ == "__main__":
    main()
