"""
Data preprocessing module for AGORA dataset.

This module provides functionality to clean, normalize, and prepare text data
for downstream RAG and MI analysis.
"""

import re
import unicodedata
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger


class TextPreprocessor:
    """
    Preprocessor for AGORA document text.
    
    Performs text normalization, cleaning, and quality filtering while preserving
    semantic content necessary for policy analysis.
    """
    
    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 2000,
        normalize_unicode: bool = True,
        remove_control_chars: bool = True
    ):
        """
        Initialize text preprocessor.
        
        Args:
            min_length: Minimum token count for valid text
            max_length: Maximum token count for valid text
            normalize_unicode: Whether to normalize Unicode characters
            remove_control_chars: Whether to remove control characters
        """
        self.min_length = min_length
        self.max_length = max_length
        self.normalize_unicode = normalize_unicode
        self.remove_control_chars = remove_control_chars
        
        logger.info(
            f"Initialized TextPreprocessor: "
            f"min_length={min_length}, max_length={max_length}"
        )
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text while preserving semantic content.
        
        Args:
            text: Raw text to normalize
        
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            return ""
        
        if self.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        
        if self.remove_control_chars:
            text = ''.join(char for char in text if not unicodedata.category(char).startswith('C') or char in '\n\t')
        
        text = re.sub(r'[\r\n]+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        
        return text
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using simple whitespace tokenization.
        
        Args:
            text: Text to tokenize
        
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(text.split())
    
    def is_valid_length(self, text: str) -> bool:
        """
        Check if text meets length requirements.
        
        Args:
            text: Text to check
        
        Returns:
            True if text length is within acceptable range
        """
        token_count = self.count_tokens(text)
        return self.min_length <= token_count <= self.max_length
    
    def extract_text_statistics(self, text: str) -> Dict:
        """
        Extract statistical features from text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with text statistics
        """
        tokens = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'num_chars': len(text),
            'num_tokens': len(tokens),
            'num_sentences': len([s for s in sentences if s.strip()]),
            'avg_token_length': np.mean([len(token) for token in tokens]) if tokens else 0,
            'avg_sentence_length': len(tokens) / max(len(sentences), 1)
        }
    
    def preprocess_documents(
        self,
        documents_df: pd.DataFrame,
        text_column: str = 'Text',
        filter_invalid: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess documents DataFrame.
        
        Args:
            documents_df: DataFrame containing documents
            text_column: Name of column containing text
            filter_invalid: Whether to filter out invalid documents
        
        Returns:
            Preprocessed DataFrame with additional statistics columns
        """
        logger.info(f"Preprocessing {len(documents_df)} documents")
        
        processed_df = documents_df.copy()
        
        if text_column not in processed_df.columns:
            logger.warning(f"Text column '{text_column}' not found, skipping preprocessing")
            return processed_df
        
        processed_df['normalized_text'] = processed_df[text_column].apply(
            lambda x: self.normalize_text(x) if pd.notna(x) else ""
        )
        
        processed_df['num_tokens'] = processed_df['normalized_text'].apply(
            self.count_tokens
        )
        
        processed_df['is_valid_length'] = processed_df['normalized_text'].apply(
            self.is_valid_length
        )
        
        stats_list = processed_df['normalized_text'].apply(
            self.extract_text_statistics
        ).tolist()
        stats_df = pd.DataFrame(stats_list)
        processed_df = pd.concat([processed_df, stats_df], axis=1)
        
        initial_count = len(processed_df)
        if filter_invalid:
            processed_df = processed_df[processed_df['is_valid_length']].reset_index(drop=True)
            filtered_count = initial_count - len(processed_df)
            logger.info(
                f"Filtered {filtered_count} documents with invalid length. "
                f"Remaining: {len(processed_df)}"
            )
        
        logger.info(f"Preprocessing complete. Final count: {len(processed_df)}")
        
        return processed_df
    
    def preprocess_segments(
        self,
        segments_df: pd.DataFrame,
        text_column: str = 'Text',
        filter_invalid: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess segments DataFrame.
        
        Args:
            segments_df: DataFrame containing segments
            text_column: Name of column containing text
            filter_invalid: Whether to filter out invalid segments
        
        Returns:
            Preprocessed DataFrame with additional statistics columns
        """
        logger.info(f"Preprocessing {len(segments_df)} segments")
        
        processed_df = segments_df.copy()
        
        if text_column not in processed_df.columns:
            logger.warning(f"Text column '{text_column}' not found, skipping preprocessing")
            return processed_df
        
        processed_df['normalized_text'] = processed_df[text_column].apply(
            lambda x: self.normalize_text(x) if pd.notna(x) else ""
        )
        
        processed_df['num_tokens'] = processed_df['normalized_text'].apply(
            self.count_tokens
        )
        
        processed_df['is_valid_length'] = processed_df['normalized_text'].apply(
            self.is_valid_length
        )
        
        stats_list = processed_df['normalized_text'].apply(
            self.extract_text_statistics
        ).tolist()
        stats_df = pd.DataFrame(stats_list)
        processed_df = pd.concat([processed_df, stats_df], axis=1)
        
        initial_count = len(processed_df)
        if filter_invalid:
            processed_df = processed_df[processed_df['is_valid_length']].reset_index(drop=True)
            filtered_count = initial_count - len(processed_df)
            logger.info(
                f"Filtered {filtered_count} segments with invalid length. "
                f"Remaining: {len(processed_df)}"
            )
        
        logger.info(f"Preprocessing complete. Final count: {len(processed_df)}")
        
        return processed_df
    
    def segment_long_text(
        self,
        text: str,
        max_segment_length: int = 1000,
        overlap: int = 100
    ) -> List[str]:
        """
        Segment long text into smaller chunks with overlap.
        
        Args:
            text: Text to segment
            max_segment_length: Maximum tokens per segment
            overlap: Number of overlapping tokens between segments
        
        Returns:
            List of text segments
        """
        tokens = text.split()
        segments = []
        
        if len(tokens) <= max_segment_length:
            return [text]
        
        start = 0
        while start < len(tokens):
            end = min(start + max_segment_length, len(tokens))
            segment_tokens = tokens[start:end]
            segments.append(' '.join(segment_tokens))
            
            if end >= len(tokens):
                break
            
            start = end - overlap
        
        logger.info(
            f"Segmented text of {len(tokens)} tokens into "
            f"{len(segments)} segments"
        )
        
        return segments


def main():
    """
    Example usage of TextPreprocessor.
    """
    preprocessor = TextPreprocessor(
        min_length=50,
        max_length=2000
    )
    
    sample_text = """
    The European Union Artificial Intelligence Act aims to regulate AI systems
    based on their risk level. High-risk AI systems must comply with strict
    requirements for data governance, transparency, and human oversight.
    
    The Act establishes a comprehensive framework for trustworthy AI development.
    """
    
    normalized = preprocessor.normalize_text(sample_text)
    print("Normalized text:")
    print(normalized)
    print()
    
    stats = preprocessor.extract_text_statistics(normalized)
    print("Text statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    is_valid = preprocessor.is_valid_length(normalized)
    print(f"Valid length: {is_valid}")
    
    long_text = sample_text * 20
    segments = preprocessor.segment_long_text(long_text, max_segment_length=100, overlap=20)
    print(f"\nSegmented long text into {len(segments)} segments")


if __name__ == "__main__":
    main()

