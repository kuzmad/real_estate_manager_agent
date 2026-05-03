from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import sys
sys.path.append(str(Path(__file__).parent.parent))
from settings import Settings

settings = Settings()

# =========================================================
# Пути
# =========================================================
CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =========================================================
# Шаг 1 — Загрузка документов
# =========================================================
def load_documents():
    documents = []
    for file_path in CONTRACTS_DIR.glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

        # Добавляем метаданные — имя файла пригодится при ответе
        for doc in docs:
            doc.metadata["source"] = file_path.name

        documents.extend(docs)
        print(f"Загружен: {file_path.name} ({len(docs[0].page_content)} символов)")

    return documents


# =========================================================
# Шаг 2 — Разбивка на чанки
# =========================================================
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "]  # сначала по абзацам, потом по строкам
    )
    chunks = splitter.split_documents(documents)
    print(f"Всего чанков: {len(chunks)}")
    return chunks


# =========================================================
# Шаг 3 — Создание векторной базы
# =========================================================
def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
        base_url=settings.proxy_base_url
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    print(f"Векторная база создана: {CHROMA_DIR}")
    print(f"Документов в базе: {vectorstore._collection.count()}")
    return vectorstore


# =========================================================
# Шаг 4 — Загрузка существующей базы (для агента)
# =========================================================
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url=settings.proxy_base_url
    )

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    return vectorstore



# =========================================================
# Запуск
# =========================================================
if __name__ == "__main__":
    print("=== Загрузка документов ===")
    documents = load_documents()

    print("\n=== Разбивка на чанки ===")
    chunks = split_documents(documents)

    # Показываем пример чанка — полезно для отладки
    print("\n=== Пример первого чанка ===")
    print(f"Источник: {chunks[0].metadata['source']}")
    print(f"Текст: {chunks[0].page_content}")

    print("\n=== Создание векторной базы ===")
    vectorstore = create_vectorstore(chunks)

    print("\n=== Тест поиска ===")
    results = vectorstore.similarity_search(
        "штраф за досрочное расторжение",
        k=2  # вернуть 2 ближайших чанка
    )
    for i, doc in enumerate(results):
        print(f"\nРезультат {i+1} [{doc.metadata['source']}]:")
        print(doc.page_content)


