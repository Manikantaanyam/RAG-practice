from typing import List, Dict, Any
from embeddings.embedding import EmbeddingManager
from vectorStores.vectorStore  import VectorStore
class RAG_RETRIEVER:
    """Handles query based retreival from the vector store"""

    def __init__(self, vector_store:VectorStore, embedding_manager:EmbeddingManager):
        """
        Initialize the retiever
        """

        self.vector_store=vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query:str, top_k:int=5, score_threshold:float=0.0)-> List[Dict[str, Any]]:
        """Retrieve relevant documents for the query"""

        print(f"Retreiving docs for the query: {query}")
        print(f"Top k: {top_k}, score thereshold: {score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )

            retireved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retireved_docs.append(
                            {
                                'id':doc_id,
                                'content': document,
                                'metadata': metadata,
                                'similarity_score': similarity_score,
                                'distance':distance,
                                'rank': i+1
                            }
                        )

                        print(f"Retrieved docs {len(retireved_docs)} after filtering")
                        return retireved_docs

                    else:
                        print("No documents")

        except Exception as e:
            print(f"Error while retrieving {e}")
            raise


vector_store = VectorStore()
embedding_manager = EmbeddingManager()
rag_retreiver = RAG_RETRIEVER(vector_store,embedding_manager)