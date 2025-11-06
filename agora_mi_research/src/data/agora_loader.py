"""
AGORA-specific data loader matching the official AGORA Q&A data format.

This module handles the specific column names and structure used in the
official AGORA dataset from https://github.com/rrittner1/agora
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from loguru import logger


class AGORADataLoaderOfficial:
    """
    Loader for official AGORA dataset format.
    
    Handles the specific schema used in AGORA Q&A system:
    - Documents: AGORA ID, Official name, Casual name, etc.
    - Segments: Document ID, Segment position, Text, Summary, Tags
    """
    
    REQUIRED_DOCUMENT_COLUMNS = [
        'AGORA ID', 'Official name', 'Authority'
    ]
    
    REQUIRED_SEGMENT_COLUMNS = [
        'Document ID', 'Segment position', 'Text'
    ]
    
    def __init__(
        self,
        data_path: str = "data/raw/agora",
        documents_file: str = "documents.csv",
        segments_file: str = "segments.csv"
    ):
        """
        Initialize AGORA official data loader.
        
        Args:
            data_path: Path to AGORA data directory
            documents_file: Filename of documents CSV
            segments_file: Filename of segments CSV
        """
        self.data_path = Path(data_path)
        self.documents_path = self.data_path / documents_file
        self.segments_path = self.data_path / segments_file
        
        logger.info(f"Initialized AGORADataLoaderOfficial: {self.data_path}")
    
    def load_documents(self, validate: bool = True) -> pd.DataFrame:
        """
        Load AGORA documents from official format.
        
        Args:
            validate: Whether to validate schema
        
        Returns:
            DataFrame with documents
        """
        logger.info(f"Loading documents from {self.documents_path}")
        
        if not self.documents_path.exists():
            raise FileNotFoundError(f"Documents file not found: {self.documents_path}")
        
        documents_df = pd.read_csv(self.documents_path, encoding='utf-8')
        
        logger.info(f"Loaded {len(documents_df)} documents")
        
        if validate:
            missing = [
                col for col in self.REQUIRED_DOCUMENT_COLUMNS
                if col not in documents_df.columns
            ]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
        
        return documents_df
    
    def load_segments(self, validate: bool = True) -> pd.DataFrame:
        """
        Load AGORA segments from official format.
        
        Args:
            validate: Whether to validate schema
        
        Returns:
            DataFrame with segments
        """
        logger.info(f"Loading segments from {self.segments_path}")
        
        if not self.segments_path.exists():
            raise FileNotFoundError(f"Segments file not found: {self.segments_path}")
        
        segments_df = pd.read_csv(self.segments_path, encoding='utf-8')
        
        logger.info(f"Loaded {len(segments_df)} segments")
        
        if validate:
            missing = [
                col for col in self.REQUIRED_SEGMENT_COLUMNS
                if col not in segments_df.columns
            ]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
        
        return segments_df
    
    def load_all(self, validate: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both documents and segments.
        
        Args:
            validate: Whether to validate schemas
        
        Returns:
            Tuple of (documents_df, segments_df)
        """
        documents_df = self.load_documents(validate=validate)
        segments_df = self.load_segments(validate=validate)
        
        logger.info(
            f"Successfully loaded {len(documents_df)} documents "
            f"and {len(segments_df)} segments"
        )
        
        return documents_df, segments_df
    
    def get_statistics(
        self,
        documents_df: Optional[pd.DataFrame] = None,
        segments_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Get dataset statistics in AGORA format.
        
        Args:
            documents_df: Documents DataFrame
            segments_df: Segments DataFrame
        
        Returns:
            Dictionary with statistics
        """
        if documents_df is None or segments_df is None:
            documents_df, segments_df = self.load_all(validate=False)
        
        stats = {
            'num_documents': len(documents_df),
            'num_segments': len(segments_df),
            'authority_distribution': documents_df['Authority'].value_counts().to_dict(),
            'avg_segments_per_document': len(segments_df) / len(documents_df) if len(documents_df) > 0 else 0
        }
        
        if 'Collections' in documents_df.columns:
            collections = documents_df['Collections'].value_counts()
            stats['collection_distribution'] = collections.head(10).to_dict()
        
        if 'Tags' in documents_df.columns:
            all_tags = []
            for tags_str in documents_df['Tags'].dropna():
                if isinstance(tags_str, str):
                    all_tags.extend([tag.strip() for tag in tags_str.split(';')])
            tag_counts = pd.Series(all_tags).value_counts()
            stats['tag_distribution'] = tag_counts.head(20).to_dict()
        
        logger.info(f"AGORA dataset statistics: {stats}")
        
        return stats


def main():
    """
    Example usage of AGORADataLoaderOfficial.
    """
    loader = AGORADataLoaderOfficial(
        data_path="C:/Users/23012/Downloads/agora-master/agora-master/data/agora"
    )
    
    try:
        documents_df, segments_df = loader.load_all()
        print(f"\nLoaded {len(documents_df)} documents and {len(segments_df)} segments")
        
        print("\nDocument columns:")
        print(documents_df.columns.tolist())
        
        print("\nSegment columns:")
        print(segments_df.columns.tolist())
        
        stats = loader.get_statistics(documents_df, segments_df)
        print("\nDataset Statistics:")
        print(f"  Documents: {stats['num_documents']}")
        print(f"  Segments: {stats['num_segments']}")
        print(f"  Avg segments/doc: {stats['avg_segments_per_document']:.1f}")
        
        print("\nAuthority distribution:")
        for auth, count in list(stats['authority_distribution'].items())[:5]:
            print(f"  {auth}: {count}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure AGORA data is available at:")
        print("  C:/Users/23012/Downloads/agora-master/agora-master/data/agora/")


if __name__ == "__main__":
    main()

