"""
RAGatouille-based ColBERT retriever (AGORA Q&A implementation).

This module implements ColBERT retrieval using the RAGatouille library,
matching the exact implementation from the AGORA Q&A system.

Reference: https://github.com/rrittner1/agora
"""

import pickle
import math
import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from loguru import logger

try:
    from ragatouille import RAGPretrainedModel
    RAGATOUILLE_AVAILABLE = True
except ImportError:
    RAGATOUILLE_AVAILABLE = False
    logger.warning(
        "RAGatouille not available. Install with: pip install ragatouille\n"
        "This is the library used by AGORA Q&A system."
    )


class RAGatouilleRetriever:
    """
    ColBERT retriever using RAGatouille library.
    
    This implementation follows the exact approach from AGORA Q&A system:
    - Uses RAGPretrainedModel.from_pretrained()
    - Indexes both segments and documents
    - Stores full content map in pickle file
    - Compatible with existing RAGPipeline interface
    """
    
    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        index_name: str = "agora_index",
        index_path: Optional[str] = None,
        content_map_path: Optional[str] = None,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize RAGatouille retriever.
        
        Args:
            model_name: ColBERT model name
            index_name: Name for the index
            index_path: Path to store index (default: .ragatouille/colbert/indexes/)
            content_map_path: Path to store content map pickle
            similarity_threshold: Minimum similarity for retrieval
        """
        if not RAGATOUILLE_AVAILABLE:
            raise ImportError(
                "RAGatouille library not installed. "
                "Install with: pip install ragatouille"
            )
        
        self.model_name = model_name
        self.index_name = index_name
        self.similarity_threshold = similarity_threshold
        
        if index_path is None:
            index_path = f"./.ragatouille/colbert/indexes/{index_name}"
        self.index_path = Path(index_path)
        
        if content_map_path is None:
            content_map_path = "./chunk_content/map.pkl"
        self.content_map_path = Path(content_map_path)
        
        self.rag_model = None
        self.content_map = {}
        
        logger.info(
            f"Initialized RAGatouilleRetriever: "
            f"model={model_name}, index={index_name}"
        )
    
    def create_document_chunks(
        self,
        segments_df: pd.DataFrame,
        documents_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Create chunks from segments and documents (AGORA format).
        
        This follows the exact chunking strategy from AGORA Q&A:
        - Each segment becomes a chunk with full metadata
        - Each document becomes a summary chunk
        
        Args:
            segments_df: DataFrame with segments
            documents_df: DataFrame with documents
        
        Returns:
            List of chunk dictionaries
        """
        logger.info("Creating document chunks in AGORA format...")
        
        chunks = []
        
        for idx, row in segments_df.iterrows():
            source_doc_df = documents_df.loc[
                documents_df["AGORA ID"] == row["Document ID"]
            ]
            
            source_doc = source_doc_df.iloc[0] if not source_doc_df.empty else None
            
            chunk = {
                'id': f"segment_{row['Document ID']}_{row['Segment position']}",
                'type': 'segment',
                'document_id': row['Document ID'],
                'segment_position': row['Segment position'],
                'official_name': str(source_doc['Official name']) if source_doc is not None else "",
                "casual_name": str(source_doc['Casual name']) if source_doc is not None and pd.notna(source_doc['Casual name']) else "",
                'text': str(row['Text']),
                'summary': str(row['Summary']) if pd.notna(row['Summary']) else "",
                'tags': str(row['Tags']) if pd.notna(row['Tags']) else "",
                'metadata': {
                    'non_operative': row.get('Non-operative', False),
                    'not_ai_related': row.get('Not AI-related', False),
                    'segment_annotated': row.get('Segment annotated', False),
                    'segment_validated': row.get('Segment validated', False)
                }
            }
            chunks.append(chunk)
        
        for idx, row in documents_df.iterrows():
            chunk = {
                'id': f"document_{row['AGORA ID']}",
                'type': 'document',
                'agora_id': row['AGORA ID'],
                'official_name': str(row['Official name']),
                'casual_name': str(row['Casual name']) if pd.notna(row['Casual name']) else "",
                'text': f"{row['Official name']} - {row['Short summary']} - {row['Long summary']}",
                'short_summary': str(row['Short summary']) if pd.notna(row['Short summary']) else "",
                'long_summary': str(row['Long summary']) if pd.notna(row['Long summary']) else "",
                'authority': str(row['Authority']) if pd.notna(row['Authority']) else "",
                'link': str(row['Link to document']) if pd.notna(row['Link to document']) else "",
                'tags': str(row['Tags']) if pd.notna(row['Tags']) else "",
                'metadata': {
                    'collections': row.get('Collections', ''),
                    'most_recent_activity': row.get('Most recent activity', ''),
                    'most_recent_activity_date': row.get('Most recent activity date', ''),
                    'proposed_date': row.get('Proposed date', ''),
                    'primarily_government': row.get('Primarily applies to the government', False),
                    'primarily_private': row.get('Primarily applies to the private sector', False)
                }
            }
            chunks.append(chunk)
        
        logger.info(
            f"Created {len(chunks)} chunks: "
            f"{len(segments_df)} segments + {len(documents_df)} documents"
        )
        
        return chunks
    
    def get_metadata_from_chunk(self, chunk: Dict) -> Dict:
        """Extract metadata for indexing (AGORA format)."""
        out = {
            "id": chunk["id"],
            "type": chunk["type"]
        }
        
        relevant_keys = ["link", "agora_id", "document_id", "segment_position"]
        relevant_metadata = [
            'collections', 'most_recent_activity', 'most_recent_activity_date',
            'segment_annotated', 'segment_validated'
        ]
        
        for key in relevant_keys:
            if key in chunk and chunk[key] != "":
                out[key] = chunk[key]
        
        for key in relevant_metadata:
            if key in chunk["metadata"] and chunk["metadata"][key] != "":
                out[key] = chunk["metadata"][key]
        
        return out
    
    def get_relevant_data_from_chunk(self, chunk: Dict) -> str:
        """Format chunk for indexing (AGORA format)."""
        out = (
            f'id: {chunk["id"]},\n'
            f'official_name: {chunk["official_name"]},\n'
            f'text: {chunk["text"]}'
        )
        
        relevant_keys = ["casual_name", 'short_summary', "summary", 'authority', 'tags']
        relevant_metadata = [
            'non_operative', 'not_ai_related', 'proposed_date',
            'primarily_government', 'primarily_private'
        ]
        
        for key in relevant_keys:
            if key in chunk and chunk[key] != "":
                out += f',\n{key}: {chunk[key]}'
        
        for key in relevant_metadata:
            if key in chunk["metadata"] and chunk["metadata"][key] != "":
                out += f',\n{key}: {chunk["metadata"][key]}'
        
        return out
    
    def index_segments(
        self,
        segments_df: pd.DataFrame,
        documents_df: Optional[pd.DataFrame] = None,
        text_column: str = 'Text',
        metadata_columns: Optional[List[str]] = None,
        overwrite: bool = False
    ) -> None:
        """
        Build ColBERT index using RAGatouille.
        
        Args:
            segments_df: DataFrame with segments
            documents_df: Optional DataFrame with documents (for full AGORA format)
            text_column: Column name with text (unused in AGORA format)
            metadata_columns: Metadata columns (unused in AGORA format)
            overwrite: Whether to rebuild existing index
        """
        logger.info("Building ColBERT index with RAGatouille...")
        
        if documents_df is None:
            logger.warning(
                "Documents DataFrame not provided. "
                "Creating simple chunks without document metadata."
            )
            chunks = []
            for idx, row in segments_df.iterrows():
                chunks.append({
                    'id': f"segment_{idx}",
                    'type': 'segment',
                    'document_id': row.get('Document ID', idx),
                    'segment_position': row.get('Segment position', 0),
                    'official_name': '',
                    'casual_name': '',
                    'text': str(row[text_column]),
                    'summary': '',
                    'tags': '',
                    'metadata': {}
                })
        else:
            chunks = self.create_document_chunks(segments_df, documents_df)
        
        texts = [self.get_relevant_data_from_chunk(chunk) for chunk in chunks]
        ids = [chunk["id"] for chunk in chunks]
        metadatas = [self.get_metadata_from_chunk(chunk) for chunk in chunks]
        
        for i, m in enumerate(metadatas):
            clean = {}
            for k, v in m.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    clean[k] = None
                else:
                    clean[k] = v
            metadatas[i] = clean
        
        logger.info(f"Prepared {len(texts)} texts for indexing")
        
        self.rag_model = RAGPretrainedModel.from_pretrained(self.model_name)
        
        logger.info("Starting RAGatouille indexing...")
        self.rag_model.index(
            collection=texts,
            document_ids=ids,
            index_name=self.index_name,
            document_metadatas=metadatas,
            overwrite=overwrite
        )
        
        self.content_map = {chunks[i]['id']: texts[i] for i in range(len(chunks))}
        
        self.content_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.content_map_path, "wb") as f:
            pickle.dump(self.content_map, f)
        
        logger.info(
            f"Index built successfully: {len(texts)} documents indexed, "
            f"content map saved to {self.content_map_path}"
        )
    
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
            filter_metadata: Optional metadata filters (not implemented)
        
        Returns:
            List of dictionaries with retrieved segments and metadata
        """
        if self.rag_model is None:
            if self.index_path.exists():
                logger.info(f"Loading existing index from {self.index_path}")
                self.rag_model = RAGPretrainedModel.from_index(str(self.index_path))
                
                if self.content_map_path.exists():
                    with open(self.content_map_path, "rb") as f:
                        self.content_map = pickle.load(f)
            else:
                raise ValueError(
                    "Index not found. Call index_segments() first."
                )
        
        logger.debug(f"Searching for query: {query[:50]}...")
        
        results = self.rag_model.search(query, k=top_k)
        
        retrieved = []
        
        for r in results:
            score = r.get("score", 0.0)
            
            if score < self.similarity_threshold and len(retrieved) >= 1:
                continue
            
            doc_id = r.get("document_id", "")
            content = self.content_map.get(doc_id, r.get("content", ""))
            
            segment_data = {
                'segment_id': doc_id,
                'text': content,
                'similarity': score,
                'rank': r.get("rank", 0),
                'metadata': r.get("document_metadata", {})
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
        Get number of chunks in collection.
        
        Returns:
            Number of indexed chunks
        """
        return len(self.content_map) if self.content_map else 0
    
    def get_statistics(self) -> Dict:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary with retriever statistics
        """
        stats = {
            'retriever_type': 'RAGatouille_ColBERT',
            'model_name': self.model_name,
            'index_name': self.index_name,
            'index_path': str(self.index_path),
            'collection_size': self.get_collection_size(),
            'similarity_threshold': self.similarity_threshold
        }
        
        logger.info(f"RAGatouille retriever statistics: {stats}")
        return stats


def main():
    """
    Example usage of RAGatouilleRetriever.
    """
    if not RAGATOUILLE_AVAILABLE:
        print("RAGatouille not available. Please install: pip install ragatouille")
        return
    
    sample_segments = pd.DataFrame({
        'Document ID': [2425, 2425, 2426],
        'Segment position': [1, 2, 1],
        'Text': [
            "The Act appropriates $500 million for AI-driven system modernization.",
            "The Secretary of Commerce may use funds to replace legacy systems.",
            "Federal agencies must establish AI governance structures."
        ],
        'Summary': ['', '', ''],
        'Tags': ['Government support', 'Infrastructure', 'Governance'],
        'Non-operative': [False, False, False]
    })
    
    sample_documents = pd.DataFrame({
        'AGORA ID': [2425, 2426],
        'Official name': ['One Big Beautiful Bill Act 2025', 'AI Safety Act'],
        'Casual name': ['Beautiful Bill', 'Safety Act'],
        'Authority': ['United States Congress', 'United States Congress'],
        'Short summary': ['Modernizes federal IT with AI', 'Establishes AI safety standards'],
        'Long summary': ['Appropriates $500M for AI modernization...', 'Creates framework...'],
        'Tags': ['Government support', 'Safety'],
        'Link to document': ['https://congress.gov/...', 'https://congress.gov/...'],
        'Collections': ['Federal laws', 'Federal laws']
    })
    
    print("Initializing RAGatouille retriever...")
    retriever = RAGatouilleRetriever(
        model_name="colbert-ir/colbertv2.0",
        index_name="test_agora_index"
    )
    
    print(f"\nIndexing {len(sample_segments)} segments and {len(sample_documents)} documents...")
    retriever.index_segments(
        sample_segments,
        documents_df=sample_documents,
        overwrite=True
    )
    
    query = "What funding is allocated for AI modernization?"
    print(f"\nQuerying: {query}")
    
    results = retriever.retrieve(query, top_k=3)
    
    print(f"\nRetrieved {len(results)} chunks:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['similarity']:.3f}")
        print(f"   ID: {result['segment_id']}")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Metadata: {result['metadata']}\n")


if __name__ == "__main__":
    main()

