"""
Phase 4: Automated Annotation with IBM Granite Guardian

This experiment uses IBM Granite Guardian to automatically annotate 100 RAG responses
for hallucination detection, creating an initial Gold Dataset for analysis.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import yaml
import numpy as np
from loguru import logger

from src.annotation.granite_guardian import GraniteGuardianAnnotator
from src.annotation.interface import AnnotationInterface


logger.add(
    "logs/phase4_automated_annotation_{time}.log",
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


def sample_responses_for_annotation(
    rag_results: list,
    mi_analyses: list,
    sample_size: int = 100,
    stratify_by_ecs: bool = True,
    random_seed: int = 42
) -> tuple:
    """
    Sample responses for annotation with optional stratification.
    
    Args:
        rag_results: List of RAG results
        mi_analyses: List of MI analyses
        sample_size: Number of responses to sample
        stratify_by_ecs: Whether to stratify by ECS values
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (sampled_rag_results, sampled_mi_analyses, sample_indices)
    """
    np.random.seed(random_seed)
    
    if not stratify_by_ecs or not mi_analyses:
        indices = np.random.choice(
            len(rag_results),
            size=min(sample_size, len(rag_results)),
            replace=False
        )
        
        sampled_rag = [rag_results[i] for i in indices]
        sampled_mi = [mi_analyses[i] for i in indices] if mi_analyses else None
        
        logger.info(f"Random sampling: {len(sampled_rag)} responses")
        
        return sampled_rag, sampled_mi, indices.tolist()
    
    # Stratified sampling by ECS
    ecs_values = [
        mi.get('ecs_analysis', {}).get('overall_ecs', 0.5)
        for mi in mi_analyses
    ]
    
    # Define strata
    low_ecs = [i for i, ecs in enumerate(ecs_values) if ecs < 0.3]
    medium_ecs = [i for i, ecs in enumerate(ecs_values) if 0.3 <= ecs < 0.5]
    high_ecs = [i for i, ecs in enumerate(ecs_values) if ecs >= 0.5]
    
    # Proportional sampling: 30% low, 40% medium, 30% high
    n_low = int(sample_size * 0.3)
    n_medium = int(sample_size * 0.4)
    n_high = sample_size - n_low - n_medium
    
    sampled_indices = []
    
    if len(low_ecs) > 0:
        sampled_indices.extend(
            np.random.choice(low_ecs, size=min(n_low, len(low_ecs)), replace=False)
        )
    
    if len(medium_ecs) > 0:
        sampled_indices.extend(
            np.random.choice(medium_ecs, size=min(n_medium, len(medium_ecs)), replace=False)
        )
    
    if len(high_ecs) > 0:
        sampled_indices.extend(
            np.random.choice(high_ecs, size=min(n_high, len(high_ecs)), replace=False)
        )
    
    # If we don't have enough, sample randomly from remaining
    if len(sampled_indices) < sample_size:
        remaining = list(set(range(len(rag_results))) - set(sampled_indices))
        additional = np.random.choice(
            remaining,
            size=min(sample_size - len(sampled_indices), len(remaining)),
            replace=False
        )
        sampled_indices.extend(additional)
    
    sampled_rag = [rag_results[i] for i in sampled_indices]
    sampled_mi = [mi_analyses[i] for i in sampled_indices]
    
    logger.info(
        f"Stratified sampling by ECS: {len(sampled_rag)} responses\n"
        f"  Low ECS (<0.3): {sum(1 for i in sampled_indices if ecs_values[i] < 0.3)}\n"
        f"  Medium ECS (0.3-0.5): {sum(1 for i in sampled_indices if 0.3 <= ecs_values[i] < 0.5)}\n"
        f"  High ECS (>=0.5): {sum(1 for i in sampled_indices if ecs_values[i] >= 0.5)}"
    )
    
    return sampled_rag, sampled_mi, sampled_indices


def main():
    """
    Execute Phase 4: Automated Annotation with Granite Guardian.
    
    Steps:
    1. Load RAG responses and MI analyses
    2. Sample 100 responses (stratified by ECS)
    3. Annotate using IBM Granite Guardian
    4. Generate statistics
    5. Export Gold Dataset for manual validation
    """
    logger.info("="*60)
    logger.info("Phase 4: Automated Annotation (Granite Guardian)")
    logger.info("="*60)
    
    config = load_config()
    annotation_config = config.get('annotation', {})
    
    logger.info("\nStep 1: Loading RAG responses and MI analyses...")
    
    # Load RAG results
    rag_file = project_root / "outputs/phase2_agora_rag/rag_responses_agora.json"
    
    if not rag_file.exists():
        rag_file = project_root / "outputs/phase2_rag/rag_responses.json"
    
    if not rag_file.exists():
        logger.error("No RAG results found. Please run Phase 2 first.")
        return 1
    
    with open(rag_file, 'r', encoding='utf-8') as f:
        rag_results = json.load(f)
    
    successful_rag = [r for r in rag_results if r.get('success', False)]
    logger.info(f"Loaded {len(successful_rag)} successful RAG responses")
    
    # Load MI analyses (optional)
    mi_file = project_root / "outputs/phase3_mi/mi_analysis_results.json"
    
    if mi_file.exists():
        with open(mi_file, 'r', encoding='utf-8') as f:
            mi_analyses = json.load(f)
        logger.info(f"Loaded {len(mi_analyses)} MI analyses")
    else:
        logger.warning("No MI analyses found. Sampling will be random (not ECS-stratified)")
        mi_analyses = None
    
    logger.info("\nStep 2: Sampling 100 responses for annotation...")
    
    sample_size = annotation_config.get('gold_dataset_size', 100)
    
    sampled_rag, sampled_mi, sample_indices = sample_responses_for_annotation(
        successful_rag,
        mi_analyses,
        sample_size=sample_size,
        stratify_by_ecs=(mi_analyses is not None),
        random_seed=config['random_seed']
    )
    
    logger.info(f"Sampled {len(sampled_rag)} responses")
    
    logger.info("\nStep 3: Initializing IBM Granite Guardian...")
    
    try:
        import platform
        import torch
        
        # Auto-detect device and vLLM availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        use_vllm = (platform.system() != "Windows") and torch.cuda.is_available()
        
        logger.info(f"Device: {device}")
        logger.info(f"Using vLLM: {use_vllm}")
        
        if device == 'cpu':
            logger.warning("CUDA not available. Using CPU (significantly slower)")
            logger.warning("100 samples may take 1-2 hours on CPU")
        
        annotator = GraniteGuardianAnnotator(
            model_variant='latest',  # Use granite-guardian-3.3-8b
            use_vllm=use_vllm,       # Auto-detect
            device=device,           # Auto-detect
            temperature=0.0          # Deterministic
        )
        
        logger.info("Granite Guardian loaded successfully (official implementation)")
        
        model_info = annotator.get_model_info()
        logger.info(f"Model: {model_info['model_name']}")
        logger.info(f"Parameters: {model_info['num_parameters']:,}")
        
    except Exception as e:
        logger.error(f"Failed to load Granite Guardian: {e}")
        logger.error(
            "\nGranite Guardian may not be publicly available yet.\n"
            "Options:\n"
            "  1. Check IBM model hub: https://huggingface.co/ibm-granite\n"
            "  2. Request access from IBM Research\n"
            "  3. Use manual annotation only (Label Studio)\n"
            "  4. Use alternative: Train DistilBERT on manual annotations"
        )
        return 1
    
    logger.info("\nStep 4: Performing automated annotation...")
    logger.warning(f"Annotating {len(sampled_rag)} responses. This may take 5-15 minutes.")
    
    automated_annotations = annotator.annotate_batch(sampled_rag)
    
    logger.info(f"Automated annotation complete: {len(automated_annotations)} responses")
    
    # Compute statistics
    hallucinated_count = sum(
        1 for ann in automated_annotations
        if ann['automated_annotation']['is_hallucination']
    )
    
    hallucination_rate = hallucinated_count / len(automated_annotations)
    
    logger.info(f"\nAutomated Annotation Statistics:")
    logger.info(f"  Total responses: {len(automated_annotations)}")
    logger.info(f"  Hallucinated: {hallucinated_count}")
    logger.info(f"  Hallucination rate: {hallucination_rate:.1%}")
    
    # Confidence distribution
    confidences = [
        ann['automated_annotation']['confidence']
        for ann in automated_annotations
    ]
    
    logger.info(f"  Mean confidence: {np.mean(confidences):.3f}")
    logger.info(f"  Std confidence: {np.std(confidences):.3f}")
    
    # Identify low-confidence cases for manual review
    low_confidence = [
        ann for ann in automated_annotations
        if ann['automated_annotation']['confidence'] < 0.7
    ]
    
    logger.info(f"  Low confidence (<0.7): {len(low_confidence)} require manual review")
    
    logger.info("\nStep 5: Exporting results...")
    
    output_dir = project_root / "data/annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save automated annotations
    with open(output_dir / 'automated_annotations_granite.json', 'w', encoding='utf-8') as f:
        json.dump(automated_annotations, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved to: {output_dir / 'automated_annotations_granite.json'}")
    
    # Save metadata
    metadata = {
        'phase': 'phase4_automated_annotation',
        'model': annotator.get_model_info(),
        'sample_size': len(automated_annotations),
        'sample_indices': sample_indices,
        'statistics': {
            'total': len(automated_annotations),
            'hallucinated': hallucinated_count,
            'hallucination_rate': hallucination_rate,
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'low_confidence_count': len(low_confidence)
        },
        'stratification': 'ecs_based' if mi_analyses else 'random'
    }
    
    with open(output_dir / 'automated_annotation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Export for Label Studio (manual validation)
    logger.info("\nStep 6: Preparing for manual validation...")
    
    interface = AnnotationInterface()
    
    # Export all sampled responses for Label Studio
    # (to manually validate Granite Guardian's annotations)
    interface.export_for_annotation(
        rag_results=sampled_rag,
        mi_analyses=sampled_mi,
        output_path=str(output_dir / 'tasks_for_manual_validation.json'),
        sample_size=None  # Export all sampled
    )
    
    logger.info(f"Exported tasks for manual validation")
    
    # Also export low-confidence subset for priority review
    low_conf_rag = [
        sampled_rag[i] for i, ann in enumerate(automated_annotations)
        if ann['automated_annotation']['confidence'] < 0.7
    ]
    
    low_conf_mi = [
        sampled_mi[i] for i, ann in enumerate(automated_annotations)
        if ann['automated_annotation']['confidence'] < 0.7
    ] if sampled_mi else None
    
    if low_conf_rag:
        interface.export_for_annotation(
            rag_results=low_conf_rag,
            mi_analyses=low_conf_mi,
            output_path=str(output_dir / 'tasks_priority_review.json'),
            sample_size=None
        )
        
        logger.info(
            f"Exported {len(low_conf_rag)} low-confidence cases "
            "for priority manual review"
        )
    
    logger.info("\n" + "="*60)
    logger.info("Phase 4 (Automated Annotation) Complete")
    logger.info("="*60)
    logger.info(f"\nResults:")
    logger.info(f"  Automated annotations: {len(automated_annotations)}")
    logger.info(f"  Hallucination rate: {hallucination_rate:.1%}")
    logger.info(f"  Low confidence cases: {len(low_confidence)}")
    logger.info(f"\nOutputs:")
    logger.info(f"  1. {output_dir / 'automated_annotations_granite.json'}")
    logger.info(f"  2. {output_dir / 'tasks_for_manual_validation.json'}")
    logger.info(f"  3. {output_dir / 'tasks_priority_review.json'}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review automated annotations")
    logger.info(f"  2. Manually validate low-confidence cases ({len(low_confidence)})")
    logger.info(f"  3. Optional: Validate full set for inter-rater agreement")
    logger.info(f"  4. Proceed to Phase 5: Statistical Analysis")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

