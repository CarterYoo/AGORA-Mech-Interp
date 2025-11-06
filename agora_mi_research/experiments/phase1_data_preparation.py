"""
Phase 1: Data Preparation and Filtering

This experiment implements stratified sampling and preprocessing of the AGORA dataset
following RAGTruth methodology.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import yaml
from loguru import logger

from src.data.loader import AGORADataLoader
from src.data.preprocessor import TextPreprocessor
from src.data.sampler import StratifiedSampler


logger.add(
    "logs/phase1_data_preparation_{time}.log",
    rotation="500 MB",
    retention="30 days",
    level="INFO"
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load experiment configuration."""
    config_file = project_root / config_path
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def main():
    """
    Execute Phase 1: Data Preparation and Filtering.
    
    Steps:
    1. Load AGORA dataset
    2. Preprocess documents and segments
    3. Perform stratified sampling
    4. Export filtered dataset
    """
    logger.info("="*60)
    logger.info("Phase 1: Data Preparation and Filtering")
    logger.info("="*60)
    
    config = load_config()
    
    data_config = config['data']
    random_seed = config['random_seed']
    
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Target documents: {data_config['num_documents']}")
    
    logger.info("\nStep 1: Loading AGORA dataset...")
    loader = AGORADataLoader(
        data_path=data_config['raw_path']
    )
    
    try:
        documents_df, segments_df = loader.load_all(validate=True)
        
        stats = loader.get_statistics(documents_df, segments_df)
        logger.info(f"Dataset loaded: {stats['num_documents']} documents, {stats['num_segments']} segments")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Please ensure AGORA dataset files are in data/raw/ directory")
        logger.error("Required files: documents.csv, segments.csv")
        return 1
    
    logger.info("\nStep 2: Preprocessing data...")
    preprocessor = TextPreprocessor(
        min_length=50,
        max_length=2000,
        normalize_unicode=True,
        remove_control_chars=True
    )
    
    documents_df = preprocessor.preprocess_documents(
        documents_df,
        text_column='Text',
        filter_invalid=True
    )
    
    segments_df = preprocessor.preprocess_segments(
        segments_df,
        text_column='Text',
        filter_invalid=True
    )
    
    logger.info(
        f"Preprocessing complete: "
        f"{len(documents_df)} documents, "
        f"{len(segments_df)} segments after filtering"
    )
    
    logger.info("\nStep 3: Stratified sampling...")
    sampler = StratifiedSampler(
        random_seed=random_seed,
        target_size=data_config['num_documents']
    )
    
    priority_tags = ['Applications', 'Harms', 'Risk factors', 'Strategies']
    
    sampled_docs, sampling_stats = sampler.sample_stratified(
        documents_df,
        authority_column='Authority',
        tags_column='Tags',
        token_column='num_tokens',
        priority_tags=priority_tags,
        ensure_tag_coverage=True
    )
    
    logger.info(f"Sampled {len(sampled_docs)} documents")
    logger.info(f"Authority distribution: {sampling_stats['by_authority']}")
    logger.info(f"Priority tag coverage: {sampling_stats['priority_tag_coverage']}")
    
    is_valid = sampler.validate_sample(
        sampled_docs,
        documents_df,
        min_authority_count=3,
        min_tag_coverage=0.5
    )
    
    if not is_valid:
        logger.warning("Sample validation failed, but proceeding anyway")
    
    sampled_doc_ids = set(sampled_docs['Document ID'].tolist())
    sampled_segments = segments_df[segments_df['Document ID'].isin(sampled_doc_ids)].copy()
    
    logger.info(f"Filtered to {len(sampled_segments)} segments from sampled documents")
    
    logger.info("\nStep 4: Exporting results...")
    output_dir = project_root / data_config['processed_path']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sampled_docs.to_csv(output_dir / 'filtered_documents.csv', index=False)
    sampled_segments.to_csv(output_dir / 'filtered_segments.csv', index=False)
    
    metadata = {
        'phase': 'phase1_data_preparation',
        'random_seed': random_seed,
        'target_documents': data_config['num_documents'],
        'actual_documents': len(sampled_docs),
        'total_segments': len(sampled_segments),
        'sampling_statistics': sampling_stats,
        'priority_tags': priority_tags,
        'validation_passed': is_valid
    }
    
    with open(output_dir / 'experiment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    logger.info("\n" + "="*60)
    logger.info("Phase 1 Complete")
    logger.info("="*60)
    logger.info(f"Filtered documents: {len(sampled_docs)}")
    logger.info(f"Filtered segments: {len(sampled_segments)}")
    logger.info(f"Next step: Phase 2 - RAG Response Generation")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

