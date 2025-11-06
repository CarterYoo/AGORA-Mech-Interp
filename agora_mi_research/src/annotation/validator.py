"""
Annotation validation module.

This module provides tools to validate annotation quality, check consistency,
and compute inter-annotator agreement metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger
from collections import defaultdict


class AnnotationValidator:
    """
    Validator for hallucination annotations.
    
    Checks annotation quality, consistency, and computes
    inter-annotator agreement metrics.
    """
    
    def __init__(
        self,
        min_annotation_time: float = 30.0,
        max_annotation_time: float = 600.0,
        min_agreement_threshold: float = 0.7
    ):
        """
        Initialize annotation validator.
        
        Args:
            min_annotation_time: Minimum expected annotation time (seconds)
            max_annotation_time: Maximum expected annotation time (seconds)
            min_agreement_threshold: Minimum acceptable agreement (Cohen's kappa)
        """
        self.min_annotation_time = min_annotation_time
        self.max_annotation_time = max_annotation_time
        self.min_agreement_threshold = min_agreement_threshold
        
        logger.info(
            f"Initialized AnnotationValidator: "
            f"time_range=[{min_annotation_time}, {max_annotation_time}], "
            f"agreement_threshold={min_agreement_threshold}"
        )
    
    def validate_annotation_completeness(
        self,
        annotation: Dict
    ) -> Tuple[bool, List[str]]:
        """
        Check if annotation is complete.
        
        Args:
            annotation: Parsed annotation dictionary
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not annotation.get('question'):
            issues.append("Missing question")
        
        if not annotation.get('response'):
            issues.append("Missing response")
        
        if annotation.get('severity') is None:
            issues.append("Missing severity rating")
        
        if annotation.get('overall_quality') is None:
            issues.append("Missing quality rating")
        
        if 'hallucination_spans' not in annotation:
            issues.append("Missing hallucination spans")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.debug(f"Incomplete annotation: {', '.join(issues)}")
        
        return is_valid, issues
    
    def check_span_validity(
        self,
        spans: List[Dict],
        response_text: str
    ) -> Tuple[bool, List[str]]:
        """
        Check if annotation spans are valid.
        
        Args:
            spans: List of hallucination span dictionaries
            response_text: Response text being annotated
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        response_length = len(response_text)
        
        for i, span in enumerate(spans):
            start = span.get('start', 0)
            end = span.get('end', 0)
            
            if start < 0 or end < 0:
                issues.append(f"Span {i}: Negative position")
            
            if start >= end:
                issues.append(f"Span {i}: Start >= End")
            
            if end > response_length:
                issues.append(f"Span {i}: End beyond text length")
            
            if not span.get('labels'):
                issues.append(f"Span {i}: No labels assigned")
        
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                span1 = spans[i]
                span2 = spans[j]
                
                if self._spans_overlap(span1, span2):
                    issues.append(f"Spans {i} and {j}: Overlapping")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.debug(f"Invalid spans: {', '.join(issues)}")
        
        return is_valid, issues
    
    def _spans_overlap(
        self,
        span1: Dict,
        span2: Dict
    ) -> bool:
        """
        Check if two spans overlap.
        
        Args:
            span1: First span dictionary
            span2: Second span dictionary
        
        Returns:
            True if spans overlap
        """
        start1, end1 = span1.get('start', 0), span1.get('end', 0)
        start2, end2 = span2.get('start', 0), span2.get('end', 0)
        
        return not (end1 <= start2 or end2 <= start1)
    
    def compute_cohens_kappa(
        self,
        annotations1: List[bool],
        annotations2: List[bool]
    ) -> float:
        """
        Compute Cohen's kappa for inter-annotator agreement.
        
        Args:
            annotations1: Binary annotations from annotator 1
            annotations2: Binary annotations from annotator 2
        
        Returns:
            Cohen's kappa score [-1, 1]
        """
        if len(annotations1) != len(annotations2):
            logger.error("Annotation lists have different lengths")
            return 0.0
        
        n = len(annotations1)
        if n == 0:
            return 0.0
        
        observed_agreement = sum(
            1 for a1, a2 in zip(annotations1, annotations2) if a1 == a2
        ) / n
        
        p1_true = sum(annotations1) / n
        p1_false = 1 - p1_true
        p2_true = sum(annotations2) / n
        p2_false = 1 - p2_true
        
        expected_agreement = (p1_true * p2_true) + (p1_false * p2_false)
        
        if expected_agreement == 1.0:
            return 1.0
        
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        
        return float(kappa)
    
    def compute_f1_agreement(
        self,
        annotations1: List[bool],
        annotations2: List[bool]
    ) -> Dict:
        """
        Compute F1-based agreement metrics.
        
        Args:
            annotations1: Binary annotations from annotator 1 (ground truth)
            annotations2: Binary annotations from annotator 2 (predictions)
        
        Returns:
            Dictionary with precision, recall, and F1 score
        """
        if len(annotations1) != len(annotations2):
            logger.error("Annotation lists have different lengths")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        true_positives = sum(
            1 for a1, a2 in zip(annotations1, annotations2)
            if a1 and a2
        )
        false_positives = sum(
            1 for a1, a2 in zip(annotations1, annotations2)
            if not a1 and a2
        )
        false_negatives = sum(
            1 for a1, a2 in zip(annotations1, annotations2)
            if a1 and not a2
        )
        
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0 else 0.0
        )
        
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0 else 0.0
        )
        
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def align_annotations(
        self,
        annotations_list: List[List[Dict]]
    ) -> List[List[bool]]:
        """
        Align annotations from multiple annotators for agreement computation.
        
        Args:
            annotations_list: List of annotation lists (one per annotator)
        
        Returns:
            List of aligned binary annotation lists
        """
        if not annotations_list:
            return []
        
        question_ids = set()
        for annotations in annotations_list:
            for ann in annotations:
                question_ids.add(ann.get('question_id', ''))
        
        question_ids = sorted(list(question_ids))
        
        aligned = []
        
        for annotations in annotations_list:
            ann_dict = {
                ann.get('question_id', ''): len(ann.get('hallucination_spans', [])) > 0
                for ann in annotations
            }
            
            aligned_list = [ann_dict.get(qid, False) for qid in question_ids]
            aligned.append(aligned_list)
        
        logger.info(
            f"Aligned annotations for {len(question_ids)} questions "
            f"across {len(annotations_list)} annotators"
        )
        
        return aligned
    
    def compute_inter_annotator_agreement(
        self,
        annotations_list: List[List[Dict]]
    ) -> Dict:
        """
        Compute inter-annotator agreement metrics.
        
        Args:
            annotations_list: List of annotation lists (one per annotator)
        
        Returns:
            Dictionary with agreement metrics
        """
        if len(annotations_list) < 2:
            logger.error("Need at least 2 annotators for agreement computation")
            return {}
        
        aligned = self.align_annotations(annotations_list)
        
        num_annotators = len(aligned)
        
        pairwise_kappas = []
        pairwise_f1s = []
        
        for i in range(num_annotators):
            for j in range(i + 1, num_annotators):
                kappa = self.compute_cohens_kappa(aligned[i], aligned[j])
                pairwise_kappas.append(kappa)
                
                f1_metrics = self.compute_f1_agreement(aligned[i], aligned[j])
                pairwise_f1s.append(f1_metrics['f1'])
        
        agreement = {
            'num_annotators': num_annotators,
            'num_questions': len(aligned[0]) if aligned else 0,
            'mean_cohens_kappa': float(np.mean(pairwise_kappas)) if pairwise_kappas else 0.0,
            'std_cohens_kappa': float(np.std(pairwise_kappas)) if pairwise_kappas else 0.0,
            'mean_f1': float(np.mean(pairwise_f1s)) if pairwise_f1s else 0.0,
            'pairwise_kappas': pairwise_kappas,
            'pairwise_f1s': pairwise_f1s
        }
        
        logger.info(
            f"Inter-annotator agreement: "
            f"kappa={agreement['mean_cohens_kappa']:.3f}, "
            f"f1={agreement['mean_f1']:.3f}"
        )
        
        return agreement
    
    def identify_disagreements(
        self,
        annotations_list: List[List[Dict]]
    ) -> List[Dict]:
        """
        Identify questions with annotator disagreement.
        
        Args:
            annotations_list: List of annotation lists (one per annotator)
        
        Returns:
            List of disagreement cases
        """
        aligned = self.align_annotations(annotations_list)
        
        if not aligned:
            return []
        
        disagreements = []
        num_questions = len(aligned[0])
        
        for q_idx in range(num_questions):
            votes = [aligned[i][q_idx] for i in range(len(aligned))]
            
            if not all(v == votes[0] for v in votes):
                disagreements.append({
                    'question_index': q_idx,
                    'annotator_votes': votes,
                    'agreement_ratio': sum(votes) / len(votes)
                })
        
        logger.info(
            f"Found {len(disagreements)} disagreements "
            f"out of {num_questions} questions"
        )
        
        return disagreements
    
    def compute_annotator_statistics(
        self,
        annotations: List[Dict]
    ) -> Dict:
        """
        Compute statistics for individual annotator.
        
        Args:
            annotations: List of annotations from one annotator
        
        Returns:
            Dictionary with annotator statistics
        """
        hallucination_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        quality_counts = defaultdict(int)
        
        total_spans = 0
        
        for ann in annotations:
            for span in ann.get('hallucination_spans', []):
                for label in span.get('labels', []):
                    if label != "Correct":
                        hallucination_counts[label] += 1
                        total_spans += 1
            
            if ann.get('severity'):
                severity_counts[ann['severity']] += 1
            
            if ann.get('overall_quality'):
                quality_counts[ann['overall_quality']] += 1
        
        stats = {
            'total_annotations': len(annotations),
            'total_hallucination_spans': total_spans,
            'avg_spans_per_annotation': total_spans / len(annotations) if annotations else 0,
            'hallucination_type_counts': dict(hallucination_counts),
            'severity_counts': dict(severity_counts),
            'quality_counts': dict(quality_counts)
        }
        
        return stats


def main():
    """
    Example usage of AnnotationValidator.
    """
    validator = AnnotationValidator()
    
    sample_annotation = {
        'question_id': 'q_001',
        'question': 'What are the requirements?',
        'response': 'High-risk AI systems must undergo assessment.',
        'hallucination_spans': [
            {
                'start': 0,
                'end': 9,
                'text': 'High-risk',
                'labels': ['Correct']
            },
            {
                'start': 35,
                'end': 45,
                'text': 'assessment',
                'labels': ['Subtle Conflict']
            }
        ],
        'severity': 'Medium',
        'overall_quality': 'Good'
    }
    
    print("Validating annotation completeness...")
    is_complete, issues = validator.validate_annotation_completeness(sample_annotation)
    print(f"Complete: {is_complete}")
    if issues:
        print(f"Issues: {issues}")
    
    print("\nValidating annotation spans...")
    is_valid_spans, span_issues = validator.check_span_validity(
        sample_annotation['hallucination_spans'],
        sample_annotation['response']
    )
    print(f"Valid: {is_valid_spans}")
    if span_issues:
        print(f"Issues: {span_issues}")
    
    annotations1 = [True, True, False, True, False]
    annotations2 = [True, False, False, True, True]
    
    print(f"\nComputing Cohen's kappa...")
    kappa = validator.compute_cohens_kappa(annotations1, annotations2)
    print(f"Cohen's kappa: {kappa:.3f}")
    
    print(f"\nComputing F1 agreement...")
    f1_metrics = validator.compute_f1_agreement(annotations1, annotations2)
    print(f"Precision: {f1_metrics['precision']:.3f}")
    print(f"Recall: {f1_metrics['recall']:.3f}")
    print(f"F1: {f1_metrics['f1']:.3f}")


if __name__ == "__main__":
    main()

