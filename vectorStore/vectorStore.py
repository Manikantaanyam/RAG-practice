import os
import chromadb
from typing import List
import numpy as np
import uuid

class VectorStore:
    def __init__(self, collection_name:str = "pdf_documents", persistent_directory:str="../store"):
        """
        Initialize the vector store
        
        :param self: Description
        :param collection_name: name of the chromadb collection
        :param persistent_directory: Directory to persistent the vector store
        """

        self.collection_name = collection_name
        self.persistent_directory = persistent_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """ intializing the chroma db and collection"""
        try:
            os.makedirs(self.persistent_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persistent_directory)

            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata={"description": "Pdf document embeddings for RAG"}
            )

            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Exisiting documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error intializing the vector store")
            raise

        
    def add_documents(self, documents:List[any], embeddings:np.ndarray):
        """
        Add documents and their embeddings to the vector store
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to the vector store...")
        ids = []
        metadatas = []
        documents_text=[]
        embeddings_list=[]
        for i, (doc,embedding) in enumerate(zip(documents, embeddings)):
            #generating unique id
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            # prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            #document content 
            documents_text.append(doc.page_content)
            #embedding 
            embeddings_list.append(embedding.tolist())
        #add to the collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents = documents_text
            )  
            print(f"successfully added {len(documents)} documents to vector store")
            print(f"Total docs in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding docs to the vector store: {e}")
            raise


