import numpy as np 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
from typing import List, Dict, Set , Tuple
import chromadb
from chromadb.config import Settings


class EmbeddingManager:
    """Handles document Embedding generation using SentenceTransformer"""

    def __init__(self, model_name:str="all-MiniLM-L6-V2"):
        """
        initialize the embedding manager
        
        Args:
            Hugging face model for embeddings
        """

        self.model_name = model_name
        self.model= None
        self._load_model()

     
    def _load_model(self):
        """Load the sentence Transformer model"""
        try:
            print(f"loading embedded model {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model {self.model_name}:{e}")
            raise

    def generate_embeddings(self, texts:List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""

        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)}")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings