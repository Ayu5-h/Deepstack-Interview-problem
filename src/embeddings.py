import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict
import json
import numpy as np


class EmbeddingProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
            task_type="retrieval_query"  # Specify task type for consistency
        )

        # Create data directory if it doesn't exist
        os.makedirs("./data/chroma", exist_ok=True)

        # Use default embedding function from ChromaDB
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./data/chroma")

        # Create or get collection with the embedding function
        self.collection = self.client.get_or_create_collection(
            name="stories",
            embedding_function=self.embedding_function
        )

    def process_story(self, file_path: str) -> Dict:
        with open(file_path, 'r') as f:
            content = f.read()

        story_title = os.path.basename(file_path).replace('.txt', '')

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(content)

        # Process chunks in batches
        for i, chunk in enumerate(chunks):
            # Use ChromaDB's default embedding function
            self.collection.add(
                documents=[chunk],
                metadatas=[{"story_title": story_title, "chunk_id": i}],
                ids=[f"{story_title}_{i}"]
            )

        return {"story_title": story_title, "chunks": len(chunks)}
