"""
Phase 4B: Validate Automated Annotations

This script validates Granite Guardian automated annotations against
manual annotations (if available) and computes agreement metrics.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from loguru import logger

from src.annotation.granite_guardian import GraniteGuardianAnnotator
from src.annotation.interface import AnnotationInterface
from src.annotation.validator import AnnotationValidator
from src.evaluation.metrics import EvaluationMetrics


logger.add(
    "logs/phase4_validate_annotations_{time}.log",
    rotation="500 MB",
    retention="30 days",
    level="INFO"
)


def main():
    """
    Validate automated annotations against manual annotations.
    
    Steps:
    1. Load automated annotations (Granite Guardian)
    2. Load manual annotations (Label Studio export)
    3. Compute agreement metrics
    4. Generate validation report
    """
    logger.info("="*60)
    logger.info("Phase 4B: Annotation Validation")
    logger.info("="*60)
    
    annotations_dir = project_root / "data/annotations"
    
    logger.info("\nStep 1: Loading automated annotations...")
    
    auto_file = annotations_dir / 'automated_annotations_granite.json'
    
    if not auto_file.exists():
        logger.error(f"Automated annotations not found: {auto_file}")
        logger.error("Please run phase4_automated_annotation.py first")
        return 1
    
    with open(auto_file, 'r', encoding='utf-8') as f:
        automated_annotations = json.load(f)
    
    logger.info(f"Loaded {len(automated_annotations)} automated annotations")
    
    logger.info("\nStep 2: Loading manual annotations...")
    
    manual_file = annotations_dir / 'annotations_export.json'
    
    if not manual_file.exists():
        logger.warning(f"Manual annotations not found: {manual_file}")
        logger.warning(
            "Manual annotation not yet complete. "
            "Validation will be skipped."
        )
        logger.info("\nTo complete manual annotation:")
        logger.info("  1. Start Label Studio: label-studio start")
        logger.info("  2. Import: data/annotations/tasks_for_manual_validation.json")
        logger.info("  3. Annotate responses")
        logger.info("  4. Export annotations")
        logger.info("  5. Save as: data/annotations/annotations_export.json")
        logger.info("  6. Re-run this script")
        
        return 0
    
    interface = AnnotationInterface()
    
    manual_annotations_raw = interface.load_annotations(str(manual_file))
    manual_annotations = interface.parse_all_annotations(manual_annotations_raw)
    
    logger.info(f"Loaded {len(manual_annotations)} manual annotations")
    
    logger.info("\nStep 3: Computing agreement metrics...")
    
    # Create annotation lookup by question_id
    auto_dict = {
        ann['question_id']: ann['automated_annotation']['is_hallucination']
        for ann in automated_annotations
    }
    
    manual_dict = {
        ann['question_id']: len(ann.get('hallucination_spans', [])) > 0
        for ann in manual_annotations
    }
    
    # Find overlap
    common_ids = sorted(set(auto_dict.keys()) & set(manual_dict.keys()))
    
    logger.info(f"Found {len(common_ids)} responses with both annotations")
    
    if len(common_ids) == 0:
        logger.error("No overlap between automated and manual annotations")
        return 1
    
    # Extract labels
    y_manual = [manual_dict[qid] for qid in common_ids]
    y_auto = [auto_dict[qid] for qid in common_ids]
    
    # Compute metrics
    metrics_calc = EvaluationMetrics()
    validator = AnnotationValidator()
    
    classification_metrics = metrics_calc.compute_classification_metrics(
        y_true=y_manual,
        y_pred=y_auto
    )
    
    cohens_kappa = validator.compute_cohens_kappa(y_manual, y_auto)
    
    f1_agreement = validator.compute_f1_agreement(y_manual, y_auto)
    
    logger.info("\n" + "="*60)
    logger.info("Validation Results: Granite Guardian vs Manual")
    logger.info("="*60)
    logger.info(f"\nSample size: {len(common_ids)} responses")
    logger.info(f"\nClassification Metrics:")
    logger.info(f"  Accuracy:  {classification_metrics['accuracy']:.3f}")
    logger.info(f"  Precision: {classification_metrics['precision']:.3f}")
    logger.info(f"  Recall:    {classification_metrics['recall']:.3f}")
    logger.info(f"  F1-Score:  {classification_metrics['f1_score']:.3f}")
    logger.info(f"\nAgreement Metrics:")
    logger.info(f"  Cohen's kappa: {cohens_kappa:.3f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  {classification_metrics['confusion_matrix']}")
    
    # Interpretation
    if classification_metrics['f1_score'] >= 0.8:
        interpretation = "strong agreement"
    elif classification_metrics['f1_score'] >= 0.6:
        interpretation = "moderate agreement"
    else:
        interpretation = "weak agreement"
    
    logger.info(f"\nInterpretation: {interpretation}")
    
    logger.info("\nStep 4: Analyzing disagreements...")
    
    disagreements = []
    
    for qid in common_ids:
        if y_manual[common_ids.index(qid)] != y_auto[common_ids.index(qid)]:
            # Find detailed info
            auto_ann = next(a for a in automated_annotations if a['question_id'] == qid)
            manual_ann = next(a for a in manual_annotations if a['question_id'] == qid)
            
            disagreements.append({
                'question_id': qid,
                'manual_label': y_manual[common_ids.index(qid)],
                'automated_label': y_auto[common_ids.index(qid)],
                'confidence': auto_ann['automated_annotation']['confidence'],
                'manual_spans': len(manual_ann.get('hallucination_spans', [])),
                'question': auto_ann['question'][:100]
            })
    
    logger.info(f"Found {len(disagreements)} disagreements ({len(disagreements)/len(common_ids):.1%})")
    
    if disagreements:
        logger.info("\nSample disagreements (first 3):")
        for i, dis in enumerate(disagreements[:3], 1):
            logger.info(f"\n  {i}. Q: {dis['question']}...")
            logger.info(f"     Manual: {'Hallucination' if dis['manual_label'] else 'Factual'}")
            logger.info(f"     Granite: {'Hallucination' if dis['automated_label'] else 'Factual'}")
            logger.info(f"     Confidence: {dis['confidence']:.3f}")
    
    logger.info("\nStep 5: Saving validation results...")
    
    validation_results = {
        'phase': 'phase4b_annotation_validation',
        'num_compared': len(common_ids),
        'classification_metrics': classification_metrics,
        'cohens_kappa': cohens_kappa,
        'f1_agreement': f1_agreement,
        'interpretation': interpretation,
        'num_disagreements': len(disagreements),
        'disagreement_rate': len(disagreements) / len(common_ids),
        'disagreements': disagreements
    }
    
    output_dir = project_root / "outputs/phase4_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    logger.info(f"Saved to: {output_dir / 'validation_results.json'}")
    
    logger.info("\n" + "="*60)
    logger.info("Validation Complete")
    logger.info("="*60)
    logger.info(f"\nKey Findings:")
    logger.info(f"  Agreement: {interpretation}")
    logger.info(f"  F1-Score: {classification_metrics['f1_score']:.3f}")
    logger.info(f"  Cohen's κ: {cohens_kappa:.3f}")
    logger.info(f"\nFor paper:")
    logger.info(
        f"  'IBM Granite Guardian achieved F1={classification_metrics['f1_score']:.2f} "
        f"(accuracy={classification_metrics['accuracy']:.2f}, "
        f"kappa={cohens_kappa:.2f}) "
        f"when validated against expert annotations.'"
    )
    logger.info(f"\nNext: Use these annotations in Phase 5 statistical analysis")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

