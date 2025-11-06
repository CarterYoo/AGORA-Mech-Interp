"""
Stratified sampling module for AGORA dataset.

This module implements stratified sampling following RAGTruth methodology to ensure
balanced representation of documents across authorities, policy tags, and length distributions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from loguru import logger


class StratifiedSampler:
    """
    Stratified sampler for AGORA documents.
    
    Implements multi-dimensional stratified sampling to ensure:
    1. Proportional authority representation
    2. Balanced policy tag coverage
    3. Representative document length distribution
    """
    
    def __init__(
        self,
        random_seed: int = 42,
        target_size: int = 50
    ):
        """
        Initialize stratified sampler.
        
        Args:
            random_seed: Random seed for reproducibility
            target_size: Target number of documents to sample
        """
        self.random_seed = random_seed
        self.target_size = target_size
        np.random.seed(random_seed)
        
        logger.info(
            f"Initialized StratifiedSampler: "
            f"seed={random_seed}, target_size={target_size}"
        )
    
    def compute_authority_targets(
        self,
        documents_df: pd.DataFrame,
        authority_column: str = 'Authority',
        custom_distribution: Optional[Dict[str, float]] = None
    ) -> Dict[str, int]:
        """
        Compute target sample size for each authority.
        
        Args:
            documents_df: DataFrame containing documents
            authority_column: Column name containing authority information
            custom_distribution: Optional custom distribution (must sum to 1.0)
        
        Returns:
            Dictionary mapping authority to target count
        """
        if custom_distribution is None:
            authority_counts = documents_df[authority_column].value_counts()
            total = len(documents_df)
            distribution = (authority_counts / total).to_dict()
        else:
            distribution = custom_distribution
            if not np.isclose(sum(distribution.values()), 1.0):
                logger.warning(
                    f"Custom distribution sums to {sum(distribution.values())}, "
                    "normalizing to 1.0"
                )
                total_weight = sum(distribution.values())
                distribution = {k: v / total_weight for k, v in distribution.items()}
        
        targets = {}
        allocated = 0
        
        sorted_authorities = sorted(
            distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for auth, proportion in sorted_authorities[:-1]:
            target = int(np.round(proportion * self.target_size))
            targets[auth] = target
            allocated += target
        
        last_auth = sorted_authorities[-1][0]
        targets[last_auth] = self.target_size - allocated
        
        logger.info(f"Authority targets: {targets}")
        
        return targets
    
    def categorize_length(
        self,
        num_tokens: int,
        short_threshold: int = 1000,
        long_threshold: int = 5000
    ) -> str:
        """
        Categorize document by length.
        
        Args:
            num_tokens: Number of tokens in document
            short_threshold: Upper bound for short documents
            long_threshold: Lower bound for long documents
        
        Returns:
            Length category: 'short', 'medium', or 'long'
        """
        if num_tokens < short_threshold:
            return 'short'
        elif num_tokens < long_threshold:
            return 'medium'
        else:
            return 'long'
    
    def compute_length_distribution(
        self,
        documents_df: pd.DataFrame,
        token_column: str = 'num_tokens'
    ) -> pd.DataFrame:
        """
        Add length category to documents.
        
        Args:
            documents_df: DataFrame containing documents
            token_column: Column name containing token counts
        
        Returns:
            DataFrame with added 'length_category' column
        """
        documents_df = documents_df.copy()
        
        if token_column not in documents_df.columns:
            logger.warning(
                f"Token column '{token_column}' not found, "
                "computing from text if available"
            )
            if 'Text' in documents_df.columns:
                documents_df[token_column] = documents_df['Text'].apply(
                    lambda x: len(str(x).split()) if pd.notna(x) else 0
                )
            else:
                logger.error("Cannot compute length distribution without token counts")
                documents_df['length_category'] = 'unknown'
                return documents_df
        
        documents_df['length_category'] = documents_df[token_column].apply(
            self.categorize_length
        )
        
        length_dist = documents_df['length_category'].value_counts().to_dict()
        logger.info(f"Length distribution: {length_dist}")
        
        return documents_df
    
    def extract_tags(
        self,
        tag_string: str,
        delimiter: str = ','
    ) -> List[str]:
        """
        Extract individual tags from tag string.
        
        Args:
            tag_string: Comma-separated tag string
            delimiter: Delimiter character
        
        Returns:
            List of individual tags
        """
        if not isinstance(tag_string, str) or pd.isna(tag_string):
            return []
        
        tags = [tag.strip() for tag in tag_string.split(delimiter)]
        return [tag for tag in tags if tag]
    
    def compute_tag_coverage(
        self,
        documents_df: pd.DataFrame,
        tags_column: str = 'Tags',
        priority_tags: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Compute tag distribution in dataset.
        
        Args:
            documents_df: DataFrame containing documents
            tags_column: Column name containing tags
            priority_tags: Optional list of tags to prioritize
        
        Returns:
            Dictionary mapping tag to document count
        """
        if tags_column not in documents_df.columns:
            logger.warning(f"Tags column '{tags_column}' not found")
            return {}
        
        tag_counts = defaultdict(int)
        
        for tags_str in documents_df[tags_column].dropna():
            tags = self.extract_tags(tags_str)
            for tag in tags:
                tag_counts[tag] += 1
        
        if priority_tags:
            logger.info(f"Priority tags coverage:")
            for tag in priority_tags:
                count = tag_counts.get(tag, 0)
                logger.info(f"  {tag}: {count} documents")
        
        return dict(tag_counts)
    
    def sample_stratified(
        self,
        documents_df: pd.DataFrame,
        authority_column: str = 'Authority',
        tags_column: str = 'Tags',
        token_column: str = 'num_tokens',
        priority_tags: Optional[List[str]] = None,
        authority_distribution: Optional[Dict[str, float]] = None,
        ensure_tag_coverage: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform stratified sampling on documents.
        
        Args:
            documents_df: DataFrame containing documents
            authority_column: Column name for authority
            tags_column: Column name for tags
            token_column: Column name for token counts
            priority_tags: Optional list of tags to ensure coverage
            authority_distribution: Optional custom authority distribution
            ensure_tag_coverage: Whether to ensure minimum tag coverage
        
        Returns:
            Tuple of (sampled_df, sampling_statistics)
        """
        logger.info(
            f"Starting stratified sampling: "
            f"{len(documents_df)} documents -> {self.target_size} documents"
        )
        
        documents_df = self.compute_length_distribution(
            documents_df,
            token_column=token_column
        )
        
        authority_targets = self.compute_authority_targets(
            documents_df,
            authority_column=authority_column,
            custom_distribution=authority_distribution
        )
        
        sampled_docs = []
        sampling_stats = {
            'total_sampled': 0,
            'by_authority': {},
            'by_length': defaultdict(int),
            'by_tag': defaultdict(int),
            'priority_tag_coverage': {}
        }
        
        for authority, target_count in authority_targets.items():
            auth_docs = documents_df[
                documents_df[authority_column] == authority
            ]
            
            if len(auth_docs) == 0:
                logger.warning(f"No documents found for authority: {authority}")
                continue
            
            if len(auth_docs) < target_count:
                logger.warning(
                    f"Insufficient documents for {authority}: "
                    f"have {len(auth_docs)}, need {target_count}"
                )
                sampled = auth_docs
            else:
                length_stratified = []
                for length_cat in ['short', 'medium', 'long']:
                    length_docs = auth_docs[
                        auth_docs['length_category'] == length_cat
                    ]
                    if len(length_docs) > 0:
                        target_props = {'short': 0.3, 'medium': 0.5, 'long': 0.2}
                        n_sample = int(np.round(target_count * target_props[length_cat]))
                        n_sample = min(n_sample, len(length_docs))
                        
                        if n_sample > 0:
                            sample = length_docs.sample(
                                n=n_sample,
                                random_state=self.random_seed
                            )
                            length_stratified.append(sample)
                
                if length_stratified:
                    sampled = pd.concat(length_stratified, ignore_index=True)
                    
                    if len(sampled) < target_count:
                        remaining_docs = auth_docs[
                            ~auth_docs.index.isin(sampled.index)
                        ]
                        if len(remaining_docs) > 0:
                            n_additional = target_count - len(sampled)
                            additional = remaining_docs.sample(
                                n=min(n_additional, len(remaining_docs)),
                                random_state=self.random_seed
                            )
                            sampled = pd.concat([sampled, additional], ignore_index=True)
                else:
                    sampled = auth_docs.sample(
                        n=min(target_count, len(auth_docs)),
                        random_state=self.random_seed
                    )
            
            sampled_docs.append(sampled)
            
            sampling_stats['by_authority'][authority] = len(sampled)
            for length_cat in sampled['length_category'].value_counts().items():
                sampling_stats['by_length'][length_cat[0]] += length_cat[1]
        
        result_df = pd.concat(sampled_docs, ignore_index=True)
        sampling_stats['total_sampled'] = len(result_df)
        
        if tags_column in result_df.columns:
            for tags_str in result_df[tags_column].dropna():
                tags = self.extract_tags(tags_str)
                for tag in tags:
                    sampling_stats['by_tag'][tag] += 1
            
            if priority_tags:
                for tag in priority_tags:
                    sampling_stats['priority_tag_coverage'][tag] = \
                        sampling_stats['by_tag'].get(tag, 0)
        
        logger.info(f"Sampling complete: {len(result_df)} documents selected")
        logger.info(f"Authority distribution: {sampling_stats['by_authority']}")
        logger.info(f"Length distribution: {dict(sampling_stats['by_length'])}")
        
        if priority_tags:
            logger.info(
                f"Priority tag coverage: "
                f"{sampling_stats['priority_tag_coverage']}"
            )
        
        return result_df, sampling_stats
    
    def validate_sample(
        self,
        sampled_df: pd.DataFrame,
        original_df: pd.DataFrame,
        min_authority_count: int = 3,
        min_tag_coverage: float = 0.5
    ) -> bool:
        """
        Validate quality of stratified sample.
        
        Args:
            sampled_df: Sampled DataFrame
            original_df: Original DataFrame
            min_authority_count: Minimum documents per authority
            min_tag_coverage: Minimum proportion of unique tags covered
        
        Returns:
            True if sample passes validation
        """
        valid = True
        
        if len(sampled_df) != self.target_size:
            logger.warning(
                f"Sample size mismatch: "
                f"expected {self.target_size}, got {len(sampled_df)}"
            )
        
        auth_counts = sampled_df['Authority'].value_counts()
        insufficient = auth_counts[auth_counts < min_authority_count]
        if not insufficient.empty:
            logger.warning(
                f"Authorities with insufficient samples: "
                f"{insufficient.to_dict()}"
            )
            valid = False
        
        if 'Tags' in sampled_df.columns and 'Tags' in original_df.columns:
            original_tags = set()
            for tags_str in original_df['Tags'].dropna():
                original_tags.update(self.extract_tags(tags_str))
            
            sampled_tags = set()
            for tags_str in sampled_df['Tags'].dropna():
                sampled_tags.update(self.extract_tags(tags_str))
            
            coverage = len(sampled_tags) / len(original_tags) if original_tags else 0
            logger.info(f"Tag coverage: {coverage:.2%} ({len(sampled_tags)}/{len(original_tags)})")
            
            if coverage < min_tag_coverage:
                logger.warning(
                    f"Insufficient tag coverage: {coverage:.2%} < {min_tag_coverage:.2%}"
                )
                valid = False
        
        if valid:
            logger.info("Sample validation passed")
        else:
            logger.warning("Sample validation failed")
        
        return valid


def main():
    """
    Example usage of StratifiedSampler.
    """
    sample_data = {
        'Document ID': range(1, 101),
        'Document name': [f'Doc_{i}' for i in range(1, 101)],
        'Authority': np.random.choice(['EU', 'US', 'UK', 'Canada'], 100),
        'Tags': np.random.choice(
            ['Applications, Harms', 'Risk factors, Strategies', 'Accountability'],
            100
        ),
        'num_tokens': np.random.lognormal(7, 1, 100).astype(int)
    }
    documents_df = pd.DataFrame(sample_data)
    
    sampler = StratifiedSampler(random_seed=42, target_size=20)
    
    priority_tags = ['Applications', 'Harms', 'Risk factors', 'Strategies']
    
    sampled_df, stats = sampler.sample_stratified(
        documents_df,
        priority_tags=priority_tags
    )
    
    print(f"Sampled {len(sampled_df)} documents")
    print(f"\nSampling statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    is_valid = sampler.validate_sample(sampled_df, documents_df)
    print(f"\nValidation passed: {is_valid}")


if __name__ == "__main__":
    main()

