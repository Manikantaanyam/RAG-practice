from loaders.textLoader import splitDocs, directoryLoader
from embeddings.embedding import EmbeddingManager
from vectorStore.vectorStore import VectorStore

def run():
    print("Loading the docs")
    docs = directoryLoader("./data/pdf")
    

    print('splitting docs')
    chunks = splitDocs(docs)

    print("Generating embeddings")
    embedder = EmbeddingManager()
    texts = [doc.page_content for doc in chunks]
    embeddings = embedder.generate_embeddings(texts)

    print("saving to vector store")
    vector_Store = VectorStore()
    vector_Store.add_documents(chunks, embeddings)

    print("Done")

run()