import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import os


class VectorStore:
    def __init__(self):
        os.makedirs("./data/chroma", exist_ok=True)

        # Use the same embedding function as in EmbeddingProcessor
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        self.client = chromadb.PersistentClient(path="./data/chroma")
        self.collection = self.client.get_or_create_collection(
            name="stories",
            embedding_function=self.embedding_function
        )

    def search_character(self, character_name: str, limit: int = 5) -> Dict:
        try:
            results = self.collection.query(
                query_texts=[character_name],
                n_results=limit
            )

            if not results["documents"][0]:
                return {"chunks": [], "metadata": []}

            return {
                "chunks": results["documents"][0],
                "metadata": results["metadatas"][0]
            }
        except Exception as e:
            print(f"Search error: {str(e)}")
            return {"chunks": [], "metadata": []}