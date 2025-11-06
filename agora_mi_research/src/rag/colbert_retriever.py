"""
ColBERT-based retriever module for AGORA Q&A integration.

This module implements ColBERT retrieval for more precise token-level semantic matching,
as described in the AGORA Q&A system architecture.

Reference: https://github.com/rrittner1/agora
ColBERT Paper: Khattab & Zaharia (2020), SIGIR
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger
import pandas as pd

try:
    from colbert import Indexer, Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries, Collection
    COLBERT_AVAILABLE = True
except ImportError:
    COLBERT_AVAILABLE = False
    logger.warning(
        "ColBERT not available. Install with: pip install colbert-ai\n"
        "Falling back to SemanticRetriever for retrieval tasks."
    )


class ColBERTRetriever:
    """
    ColBERT-based semantic retriever with late interaction mechanism.
    
    ColBERT uses token-level contextualized embeddings for more precise matching
    compared to sentence-level embeddings. This is particularly effective for
    technical and legal policy documents where precision is critical.
    
    Attributes:
        checkpoint: ColBERT model checkpoint path
        index_name: Name of the ColBERT index
        index_path: Path to store index
        similarity_threshold: Minimum similarity score
    """
    
    def __init__(
        self,
        checkpoint: str = "colbert-ir/colbertv2.0",
        index_name: str = "agora_colbert_index",
        index_path: Optional[str] = None,
        similarity_threshold: float = 0.7,
        nbits: int = 2,
        doc_maxlen: int = 300,
        query_maxlen: int = 32,
        kmeans_niters: int = 4
    ):
        """
        Initialize ColBERT retriever.
        
        Args:
            checkpoint: HuggingFace checkpoint or local path
            index_name: Name for the index
            index_path: Directory to store index
            similarity_threshold: Minimum similarity for retrieval
            nbits: Compression bits (1, 2, or 4)
            doc_maxlen: Maximum document tokens
            query_maxlen: Maximum query tokens
            kmeans_niters: K-means iterations for compression
        """
        if not COLBERT_AVAILABLE:
            raise ImportError(
                "ColBERT library not installed. "
                "Install with: pip install colbert-ai"
            )
        
        self.checkpoint = checkpoint
        self.index_name = index_name
        self.similarity_threshold = similarity_threshold
        
        if index_path is None:
            index_path = str(Path("data/colbert_indexes").absolute())
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            query_maxlen=query_maxlen,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
            index_path=str(self.index_path)
        )
        
        self.indexer = None
        self.searcher = None
        self.collection = None
        
        logger.info(
            f"Initialized ColBERTRetriever: "
            f"checkpoint={checkpoint}, "
            f"index={index_name}, "
            f"nbits={nbits}"
        )
    
    def index_segments(
        self,
        segments_df: pd.DataFrame,
        text_column: str = 'Text',
        metadata_columns: Optional[List[str]] = None,
        overwrite: bool = False
    ) -> None:
        """
        Build ColBERT index from document segments.
        
        Args:
            segments_df: DataFrame containing segments
            text_column: Column name with text to index
            metadata_columns: Metadata columns to preserve
            overwrite: Whether to rebuild existing index
        """
        logger.info(f"Building ColBERT index for {len(segments_df)} segments")
        
        if text_column not in segments_df.columns:
            raise ValueError(f"Text column '{text_column}' not found")
        
        index_full_path = self.index_path / self.index_name
        
        if index_full_path.exists() and not overwrite:
            logger.info(f"Index already exists at {index_full_path}")
            logger.info("Loading existing index (use overwrite=True to rebuild)")
            self._load_index()
            return
        
        texts = segments_df[text_column].fillna("").tolist()
        self.collection = texts
        
        if metadata_columns:
            self.metadata = segments_df[metadata_columns].to_dict('records')
        else:
            self.metadata = [{'index': i} for i in range(len(texts))]
        
        logger.info("Starting ColBERT indexing (this may take several minutes)")
        
        with Run().context(RunConfig(nranks=1, experiment=self.index_name)):
            self.indexer = Indexer(
                checkpoint=self.checkpoint,
                config=self.config
            )
            
            self.indexer.index(
                name=self.index_name,
                collection=texts,
                overwrite=overwrite
            )
        
        logger.info(f"Index built successfully: {len(texts)} documents indexed")
        
        self._save_metadata()
        self._load_index()
    
    def _load_index(self) -> None:
        """Load existing ColBERT index."""
        logger.info(f"Loading ColBERT index: {self.index_name}")
        
        with Run().context(RunConfig(experiment=self.index_name)):
            self.searcher = Searcher(
                index=self.index_name,
                config=self.config
            )
        
        self._load_metadata()
        
        logger.info("Index loaded successfully")
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        if self.metadata:
            import json
            metadata_path = self.index_path / f"{self.index_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            logger.debug(f"Metadata saved to {metadata_path}")
    
    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        import json
        metadata_path = self.index_path / f"{self.index_name}_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.debug(f"Metadata loaded from {metadata_path}")
        else:
            self.metadata = []
            logger.warning("No metadata file found")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant segments for query using ColBERT.
        
        Args:
            query: Query text
            top_k: Number of segments to retrieve
            filter_metadata: Optional metadata filters (not implemented yet)
        
        Returns:
            List of dictionaries with retrieved segments and metadata
        """
        if self.searcher is None:
            raise ValueError(
                "Index not loaded. Call index_segments() or _load_index() first."
            )
        
        logger.debug(f"Searching for query: {query[:50]}...")
        
        results = self.searcher.search(query, k=top_k)
        
        retrieved = []
        
        for passage_id, passage_rank, passage_score in zip(*results):
            similarity = float(passage_score)
            
            if similarity < self.similarity_threshold and len(retrieved) >= 1:
                continue
            
            segment_data = {
                'segment_id': f"seg_{passage_id}",
                'text': self.collection[passage_id] if self.collection else "",
                'similarity': similarity,
                'rank': passage_rank,
                'metadata': self.metadata[passage_id] if passage_id < len(self.metadata) else {}
            }
            
            retrieved.append(segment_data)
        
        logger.info(
            f"Retrieved {len(retrieved)} segments "
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
            top_k: Number of segments per query
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
        return len(self.collection) if self.collection else 0
    
    def get_statistics(self) -> Dict:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary with retriever statistics
        """
        stats = {
            'retriever_type': 'ColBERT',
            'checkpoint': self.checkpoint,
            'index_name': self.index_name,
            'index_path': str(self.index_path),
            'collection_size': self.get_collection_size(),
            'similarity_threshold': self.similarity_threshold,
            'config': {
                'doc_maxlen': self.config.doc_maxlen,
                'query_maxlen': self.config.query_maxlen,
                'nbits': self.config.nbits
            }
        }
        
        logger.info(f"ColBERT retriever statistics: {stats}")
        return stats


def main():
    """
    Example usage of ColBERTRetriever.
    """
    if not COLBERT_AVAILABLE:
        print("ColBERT not available. Please install: pip install colbert-ai")
        return
    
    sample_segments = pd.DataFrame({
        'Document ID': [1, 1, 2, 2, 3],
        'Segment position': [1, 2, 1, 2, 1],
        'Text': [
            "The EU AI Act establishes comprehensive requirements for high-risk AI systems, including conformity assessment procedures.",
            "Providers must implement appropriate data governance and management practices to ensure data quality.",
            "The US Executive Order on AI promotes safe, secure, and trustworthy AI development across federal agencies.",
            "Federal agencies must establish AI governance structures and conduct risk assessments for AI systems.",
            "The UK AI White Paper proposes a principles-based regulatory approach focusing on safety and transparency."
        ]
    })
    
    print("Initializing ColBERT retriever...")
    retriever = ColBERTRetriever(
        checkpoint="colbert-ir/colbertv2.0",
        index_name="test_index",
        nbits=2
    )
    
    print(f"\nIndexing {len(sample_segments)} segments...")
    retriever.index_segments(
        sample_segments,
        metadata_columns=['Document ID', 'Segment position'],
        overwrite=True
    )
    
    query = "What are the requirements for AI governance?"
    print(f"\nQuerying: {query}")
    
    results = retriever.retrieve(query, top_k=3)
    
    print(f"\nRetrieved {len(results)} segments:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['similarity']:.3f}")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Metadata: {result['metadata']}\n")
    
    stats = retriever.get_statistics()
    print("Retriever statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

