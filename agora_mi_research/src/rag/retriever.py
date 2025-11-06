"""
Semantic retrieval module for RAG pipeline.

This module implements semantic retrieval using sentence transformers and
ChromaDB for efficient similarity search over document segments.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger
import pandas as pd


class SemanticRetriever:
    """
    Semantic retriever using sentence transformers and ChromaDB.
    
    Retrieves relevant document segments based on semantic similarity
    to input queries.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "agora_documents",
        persist_directory: Optional[str] = None,
        similarity_threshold: float = 0.7,
        device: str = "cuda"
    ):
        """
        Initialize semantic retriever.
        
        Args:
            embedding_model_name: Name of sentence transformer model
            collection_name: Name of ChromaDB collection
            persist_directory: Directory to persist vector store
            similarity_threshold: Minimum similarity score for retrieval
            device: Device for embedding model (cuda/cpu)
        """
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device=device
        )
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        if persist_directory:
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing ChromaDB with persistence: {persist_directory}")
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_path)
            )
        else:
            logger.info("Initializing ChromaDB in-memory")
            self.chroma_client = chromadb.Client()
        
        self.collection = None
        logger.info("SemanticRetriever initialized")
    
    def create_collection(
        self,
        overwrite: bool = False
    ) -> None:
        """
        Create or get ChromaDB collection.
        
        Args:
            overwrite: Whether to delete existing collection
        """
        if overwrite:
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception as e:
                logger.debug(f"No existing collection to delete: {e}")
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection ready: {self.collection_name}")
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed texts using sentence transformer.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for embedding
            show_progress: Whether to show progress bar
        
        Returns:
            Numpy array of embeddings [num_texts, embedding_dim]
        """
        logger.info(f"Embedding {len(texts)} texts")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        logger.info(f"Embedding complete: shape {embeddings.shape}")
        return embeddings
    
    def index_segments(
        self,
        segments_df: pd.DataFrame,
        text_column: str = 'Text',
        metadata_columns: Optional[List[str]] = None,
        batch_size: int = 1000
    ) -> None:
        """
        Index document segments into ChromaDB.
        
        Args:
            segments_df: DataFrame containing segments
            text_column: Column name containing text to embed
            metadata_columns: Optional list of metadata columns to store
            batch_size: Batch size for indexing
        """
        if self.collection is None:
            self.create_collection()
        
        logger.info(f"Indexing {len(segments_df)} segments")
        
        if text_column not in segments_df.columns:
            raise ValueError(f"Text column '{text_column}' not found")
        
        texts = segments_df[text_column].fillna("").tolist()
        ids = [f"seg_{i}" for i in range(len(segments_df))]
        
        if metadata_columns:
            metadatas = []
            for _, row in segments_df.iterrows():
                metadata = {}
                for col in metadata_columns:
                    if col in segments_df.columns:
                        value = row[col]
                        if pd.notna(value):
                            metadata[col] = str(value)
                metadatas.append(metadata)
        else:
            metadatas = None
        
        embeddings = self.embed_texts(texts, batch_size=32)
        
        for i in range(0, len(segments_df), batch_size):
            end_idx = min(i + batch_size, len(segments_df))
            
            batch_ids = ids[i:end_idx]
            batch_texts = texts[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_metadatas = metadatas[i:end_idx] if metadatas else None
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            
            logger.info(f"Indexed batch: {i}-{end_idx}/{len(segments_df)}")
        
        logger.info(f"Indexing complete: {len(segments_df)} segments")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant segments for query.
        
        Args:
            query: Query text
            top_k: Number of segments to retrieve
            filter_metadata: Optional metadata filters
        
        Returns:
            List of dictionaries containing retrieved segments and metadata
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection() first.")
        
        query_embedding = self.embed_texts([query], show_progress=False)[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata
        )
        
        retrieved = []
        for i in range(len(results['ids'][0])):
            segment_id = results['ids'][0][i]
            text = results['documents'][0][i]
            distance = results['distances'][0][i]
            similarity = 1 - distance
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            
            if similarity >= self.similarity_threshold or len(retrieved) < 1:
                retrieved.append({
                    'segment_id': segment_id,
                    'text': text,
                    'similarity': similarity,
                    'metadata': metadata
                })
        
        logger.info(
            f"Retrieved {len(retrieved)} segments for query "
            f"(top_k={top_k}, threshold={self.similarity_threshold})"
        )
        
        return retrieved
    
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[List[Dict]]:
        """
        Retrieve relevant segments for multiple queries.
        
        Args:
            queries: List of query texts
            top_k: Number of segments to retrieve per query
            filter_metadata: Optional metadata filters
        
        Returns:
            List of retrieval results for each query
        """
        results = []
        for i, query in enumerate(queries):
            retrieved = self.retrieve(
                query=query,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            results.append(retrieved)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(queries)} queries")
        
        logger.info(f"Batch retrieval complete: {len(queries)} queries")
        return results
    
    def get_collection_size(self) -> int:
        """
        Get number of segments in collection.
        
        Returns:
            Number of indexed segments
        """
        if self.collection is None:
            return 0
        return self.collection.count()
    
    def get_statistics(self) -> Dict:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary with retriever statistics
        """
        stats = {
            'embedding_model': self.embedding_model_name,
            'embedding_dim': self.embedding_dim,
            'collection_name': self.collection_name,
            'collection_size': self.get_collection_size(),
            'similarity_threshold': self.similarity_threshold
        }
        
        logger.info(f"Retriever statistics: {stats}")
        return stats


def main():
    """
    Example usage of SemanticRetriever.
    """
    sample_segments = pd.DataFrame({
        'Document ID': [1, 1, 2, 2, 3],
        'Segment position': [1, 2, 1, 2, 1],
        'Text': [
            "The EU AI Act establishes requirements for high-risk AI systems.",
            "Providers must implement data governance and management practices.",
            "The US Executive Order on AI promotes safe and trustworthy AI.",
            "Federal agencies must establish AI governance structures.",
            "The UK AI White Paper proposes a principles-based regulatory approach."
        ]
    })
    
    retriever = SemanticRetriever(
        persist_directory="../../data/chromadb",
        device="cpu"
    )
    
    retriever.create_collection(overwrite=True)
    
    retriever.index_segments(
        sample_segments,
        metadata_columns=['Document ID', 'Segment position']
    )
    
    query = "What are the requirements for AI governance?"
    results = retriever.retrieve(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} segments:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Similarity: {result['similarity']:.3f}")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Metadata: {result['metadata']}\n")
    
    stats = retriever.get_statistics()
    print("Retriever statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

