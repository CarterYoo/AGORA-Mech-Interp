"""
Data loader module for AGORA dataset.

This module provides functionality to load AGORA documents and segments from CSV files
with proper validation and error handling.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from loguru import logger


class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


class AGORADataLoader:
    """
    Loader for AGORA AI governance dataset.
    
    The AGORA dataset contains AI governance and policy documents from multiple
    jurisdictions including EU, US, UK, Canada, and Australia.
    
    Attributes:
        data_path: Path to directory containing AGORA CSV files
        documents_file: Name of documents CSV file
        segments_file: Name of segments CSV file
    """
    
    REQUIRED_DOCUMENT_COLUMNS = [
        'Document ID', 'Document name', 'Authority', 'Tags'
    ]
    
    REQUIRED_SEGMENT_COLUMNS = [
        'Document ID', 'Segment position', 'Text'
    ]
    
    def __init__(
        self,
        data_path: str = "data/raw",
        documents_file: str = "documents.csv",
        segments_file: str = "segments.csv"
    ):
        """
        Initialize AGORA data loader.
        
        Args:
            data_path: Path to directory containing CSV files
            documents_file: Filename of documents CSV
            segments_file: Filename of segments CSV
        """
        self.data_path = Path(data_path)
        self.documents_path = self.data_path / documents_file
        self.segments_path = self.data_path / segments_file
        
        logger.info(f"Initialized AGORADataLoader with data path: {self.data_path}")
    
    def load_documents(
        self,
        validate: bool = True,
        encoding: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load AGORA documents from CSV file.
        
        Args:
            validate: Whether to validate schema after loading
            encoding: File encoding (None for auto-detection)
        
        Returns:
            DataFrame containing documents with validated schema
        
        Raises:
            DataLoadError: If file not found or validation fails
        """
        logger.info(f"Loading documents from {self.documents_path}")
        
        if not self.documents_path.exists():
            error_msg = f"Documents file not found: {self.documents_path}"
            logger.error(error_msg)
            raise DataLoadError(error_msg)
        
        encodings_to_try = [encoding] if encoding else ['utf-8', 'latin-1', 'cp1252']
        
        documents_df = None
        for enc in encodings_to_try:
            if enc is None:
                continue
            try:
                documents_df = pd.read_csv(self.documents_path, encoding=enc)
                logger.info(f"Successfully loaded documents with encoding: {enc}")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to load with encoding: {enc}")
                continue
            except Exception as e:
                error_msg = f"Error loading documents: {str(e)}"
                logger.error(error_msg)
                raise DataLoadError(error_msg) from e
        
        if documents_df is None:
            error_msg = "Failed to load documents with any encoding"
            logger.error(error_msg)
            raise DataLoadError(error_msg)
        
        logger.info(f"Loaded {len(documents_df)} documents")
        
        if validate:
            self._validate_documents_schema(documents_df)
        
        return documents_df
    
    def load_segments(
        self,
        validate: bool = True,
        encoding: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load AGORA segments from CSV file.
        
        Args:
            validate: Whether to validate schema after loading
            encoding: File encoding (None for auto-detection)
        
        Returns:
            DataFrame containing segments with validated schema
        
        Raises:
            DataLoadError: If file not found or validation fails
        """
        logger.info(f"Loading segments from {self.segments_path}")
        
        if not self.segments_path.exists():
            error_msg = f"Segments file not found: {self.segments_path}"
            logger.error(error_msg)
            raise DataLoadError(error_msg)
        
        encodings_to_try = [encoding] if encoding else ['utf-8', 'latin-1', 'cp1252']
        
        segments_df = None
        for enc in encodings_to_try:
            if enc is None:
                continue
            try:
                segments_df = pd.read_csv(self.segments_path, encoding=enc)
                logger.info(f"Successfully loaded segments with encoding: {enc}")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to load with encoding: {enc}")
                continue
            except Exception as e:
                error_msg = f"Error loading segments: {str(e)}"
                logger.error(error_msg)
                raise DataLoadError(error_msg) from e
        
        if segments_df is None:
            error_msg = "Failed to load segments with any encoding"
            logger.error(error_msg)
            raise DataLoadError(error_msg)
        
        logger.info(f"Loaded {len(segments_df)} segments")
        
        if validate:
            self._validate_segments_schema(segments_df)
        
        return segments_df
    
    def load_all(
        self,
        validate: bool = True,
        encoding: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both documents and segments.
        
        Args:
            validate: Whether to validate schemas
            encoding: File encoding (None for auto-detection)
        
        Returns:
            Tuple of (documents_df, segments_df)
        
        Raises:
            DataLoadError: If loading or validation fails
        """
        documents_df = self.load_documents(validate=validate, encoding=encoding)
        segments_df = self.load_segments(validate=validate, encoding=encoding)
        
        logger.info(
            f"Successfully loaded {len(documents_df)} documents "
            f"and {len(segments_df)} segments"
        )
        
        return documents_df, segments_df
    
    def _validate_documents_schema(self, documents_df: pd.DataFrame) -> None:
        """
        Validate documents DataFrame schema.
        
        Args:
            documents_df: DataFrame to validate
        
        Raises:
            DataLoadError: If validation fails
        """
        missing_columns = [
            col for col in self.REQUIRED_DOCUMENT_COLUMNS
            if col not in documents_df.columns
        ]
        
        if missing_columns:
            error_msg = (
                f"Documents DataFrame missing required columns: {missing_columns}. "
                f"Available columns: {list(documents_df.columns)}"
            )
            logger.error(error_msg)
            raise DataLoadError(error_msg)
        
        if documents_df.empty:
            logger.warning("Documents DataFrame is empty")
        
        null_counts = documents_df[self.REQUIRED_DOCUMENT_COLUMNS].isnull().sum()
        if null_counts.any():
            logger.warning(
                f"Found null values in required columns: "
                f"{null_counts[null_counts > 0].to_dict()}"
            )
        
        logger.info("Documents schema validation passed")
    
    def _validate_segments_schema(self, segments_df: pd.DataFrame) -> None:
        """
        Validate segments DataFrame schema.
        
        Args:
            segments_df: DataFrame to validate
        
        Raises:
            DataLoadError: If validation fails
        """
        missing_columns = [
            col for col in self.REQUIRED_SEGMENT_COLUMNS
            if col not in segments_df.columns
        ]
        
        if missing_columns:
            error_msg = (
                f"Segments DataFrame missing required columns: {missing_columns}. "
                f"Available columns: {list(segments_df.columns)}"
            )
            logger.error(error_msg)
            raise DataLoadError(error_msg)
        
        if segments_df.empty:
            logger.warning("Segments DataFrame is empty")
        
        null_counts = segments_df[self.REQUIRED_SEGMENT_COLUMNS].isnull().sum()
        if null_counts.any():
            logger.warning(
                f"Found null values in required columns: "
                f"{null_counts[null_counts > 0].to_dict()}"
            )
        
        logger.info("Segments schema validation passed")
    
    def get_statistics(
        self,
        documents_df: Optional[pd.DataFrame] = None,
        segments_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Get dataset statistics.
        
        Args:
            documents_df: Documents DataFrame (loads if None)
            segments_df: Segments DataFrame (loads if None)
        
        Returns:
            Dictionary with dataset statistics
        """
        if documents_df is None or segments_df is None:
            documents_df, segments_df = self.load_all(validate=False)
        
        stats = {
            'num_documents': len(documents_df),
            'num_segments': len(segments_df),
            'authority_distribution': documents_df['Authority'].value_counts().to_dict(),
            'avg_segments_per_document': len(segments_df) / len(documents_df) if len(documents_df) > 0 else 0
        }
        
        if 'Tags' in documents_df.columns:
            all_tags = []
            for tags_str in documents_df['Tags'].dropna():
                if isinstance(tags_str, str):
                    all_tags.extend([tag.strip() for tag in tags_str.split(',')])
            tag_counts = pd.Series(all_tags).value_counts()
            stats['tag_distribution'] = tag_counts.head(10).to_dict()
        
        logger.info(f"Dataset statistics: {stats}")
        
        return stats


def main():
    """
    Example usage of AGORADataLoader.
    """
    loader = AGORADataLoader(data_path="../../data/raw")
    
    try:
        documents_df, segments_df = loader.load_all()
        print(f"Loaded {len(documents_df)} documents and {len(segments_df)} segments")
        
        stats = loader.get_statistics(documents_df, segments_df)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except DataLoadError as e:
        print(f"Error loading data: {e}")


if __name__ == "__main__":
    main()

