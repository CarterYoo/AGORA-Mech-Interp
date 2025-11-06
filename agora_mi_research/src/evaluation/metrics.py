"""
Evaluation metrics module.

This module provides comprehensive evaluation metrics for hallucination detection
and RAG system performance assessment.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


class EvaluationMetrics:
    """
    Metrics calculator for hallucination detection evaluation.
    
    Computes classification metrics, correlation metrics, and ROC analysis
    for assessing MI-based hallucination detection performance.
    """
    
    def __init__(self):
        """Initialize evaluation metrics calculator."""
        logger.info("Initialized EvaluationMetrics")
    
    def compute_classification_metrics(
        self,
        y_true: List[bool],
        y_pred: List[bool],
        labels: Optional[List[str]] = None
    ) -> Dict:
        """
        Compute standard classification metrics.
        
        Args:
            y_true: True labels (binary: hallucinated or not)
            y_pred: Predicted labels
            labels: Optional label names
        
        Returns:
            Dictionary with classification metrics
        """
        if len(y_true) != len(y_pred):
            logger.error("Length mismatch between y_true and y_pred")
            return {}
        
        accuracy = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'confusion_matrix': cm.tolist(),
            'support': {
                'negative': int(support[0]) if len(support) > 0 else 0,
                'positive': int(support[1]) if len(support) > 1 else 0
            }
        }
        
        logger.info(
            f"Classification metrics: "
            f"accuracy={accuracy:.3f}, "
            f"precision={precision:.3f}, "
            f"recall={recall:.3f}, "
            f"f1={f1:.3f}"
        )
        
        return metrics
    
    def compute_roc_analysis(
        self,
        y_true: List[bool],
        y_scores: List[float]
    ) -> Dict:
        """
        Compute ROC curve and AUC.
        
        Args:
            y_true: True binary labels
            y_scores: Predicted scores/probabilities
        
        Returns:
            Dictionary with ROC analysis results
        """
        if len(y_true) != len(y_scores):
            logger.error("Length mismatch between y_true and y_scores")
            return {}
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = float(thresholds[optimal_idx])
        
        roc_analysis = {
            'auc': float(auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': float(tpr[optimal_idx]),
            'optimal_fpr': float(fpr[optimal_idx])
        }
        
        logger.info(
            f"ROC analysis: AUC={auc:.3f}, "
            f"optimal_threshold={optimal_threshold:.3f}"
        )
        
        return roc_analysis
    
    def compute_precision_recall_analysis(
        self,
        y_true: List[bool],
        y_scores: List[float]
    ) -> Dict:
        """
        Compute precision-recall curve and average precision.
        
        Args:
            y_true: True binary labels
            y_scores: Predicted scores/probabilities
        
        Returns:
            Dictionary with precision-recall analysis
        """
        if len(y_true) != len(y_scores):
            logger.error("Length mismatch between y_true and y_scores")
            return {}
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])
        
        pr_analysis = {
            'average_precision': float(avg_precision),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': np.append(thresholds, 0).tolist(),
            'optimal_threshold': float(thresholds[optimal_idx]) if len(thresholds) > optimal_idx else 0.0,
            'optimal_precision': float(precision[optimal_idx]),
            'optimal_recall': float(recall[optimal_idx]),
            'optimal_f1': float(f1_scores[optimal_idx])
        }
        
        logger.info(
            f"PR analysis: AP={avg_precision:.3f}, "
            f"optimal_f1={f1_scores[optimal_idx]:.3f}"
        )
        
        return pr_analysis
    
    def compute_correlation(
        self,
        x: List[float],
        y: List[float],
        method: str = 'pearson'
    ) -> Dict:
        """
        Compute correlation between two variables.
        
        Args:
            x: First variable
            y: Second variable
            method: Correlation method ('pearson' or 'spearman')
        
        Returns:
            Dictionary with correlation results
        """
        if len(x) != len(y):
            logger.error("Length mismatch between x and y")
            return {}
        
        x_arr = np.array(x)
        y_arr = np.array(y)
        
        if method == 'pearson':
            from scipy.stats import pearsonr
            corr, p_value = pearsonr(x_arr, y_arr)
        elif method == 'spearman':
            from scipy.stats import spearmanr
            corr, p_value = spearmanr(x_arr, y_arr)
        else:
            logger.error(f"Unknown correlation method: {method}")
            return {}
        
        correlation = {
            'method': method,
            'correlation': float(corr),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'n': len(x)
        }
        
        logger.info(
            f"{method.capitalize()} correlation: r={corr:.3f}, "
            f"p={p_value:.4f}"
        )
        
        return correlation
    
    def compute_point_biserial_correlation(
        self,
        continuous: List[float],
        binary: List[bool]
    ) -> Dict:
        """
        Compute point-biserial correlation (continuous vs binary).
        
        Args:
            continuous: Continuous variable
            binary: Binary variable
        
        Returns:
            Dictionary with correlation results
        """
        if len(continuous) != len(binary):
            logger.error("Length mismatch")
            return {}
        
        from scipy.stats import pointbiserialr
        
        corr, p_value = pointbiserialr(binary, continuous)
        
        correlation = {
            'method': 'point_biserial',
            'correlation': float(corr),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'n': len(continuous)
        }
        
        logger.info(
            f"Point-biserial correlation: r={corr:.3f}, "
            f"p={p_value:.4f}"
        )
        
        return correlation
    
    def evaluate_hallucination_predictor(
        self,
        y_true: List[bool],
        y_scores: List[float],
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Comprehensive evaluation of hallucination predictor.
        
        Args:
            y_true: True hallucination labels
            y_scores: Predicted scores (e.g., ECS, PKS)
            threshold: Optional classification threshold (computed if None)
        
        Returns:
            Comprehensive evaluation dictionary
        """
        roc_analysis = self.compute_roc_analysis(y_true, y_scores)
        
        if threshold is None:
            threshold = roc_analysis['optimal_threshold']
        
        y_pred = [score >= threshold for score in y_scores]
        
        classification_metrics = self.compute_classification_metrics(
            y_true, y_pred
        )
        
        pr_analysis = self.compute_precision_recall_analysis(y_true, y_scores)
        
        correlation = self.compute_point_biserial_correlation(y_scores, y_true)
        
        evaluation = {
            'classification_metrics': classification_metrics,
            'roc_analysis': roc_analysis,
            'pr_analysis': pr_analysis,
            'correlation': correlation,
            'threshold_used': threshold
        }
        
        logger.info(
            f"Hallucination predictor evaluation complete: "
            f"AUC={roc_analysis['auc']:.3f}, "
            f"F1={classification_metrics['f1_score']:.3f}"
        )
        
        return evaluation
    
    def compute_task_specific_metrics(
        self,
        y_true: List[bool],
        y_pred: List[bool],
        task_types: List[str]
    ) -> Dict:
        """
        Compute metrics broken down by task type.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            task_types: Task type for each sample
        
        Returns:
            Dictionary with task-specific metrics
        """
        if len(y_true) != len(y_pred) or len(y_true) != len(task_types):
            logger.error("Length mismatch in inputs")
            return {}
        
        unique_tasks = list(set(task_types))
        task_metrics = {}
        
        for task in unique_tasks:
            task_indices = [i for i, t in enumerate(task_types) if t == task]
            
            task_y_true = [y_true[i] for i in task_indices]
            task_y_pred = [y_pred[i] for i in task_indices]
            
            if len(task_y_true) > 0:
                task_metrics[task] = self.compute_classification_metrics(
                    task_y_true, task_y_pred
                )
        
        logger.info(f"Computed metrics for {len(task_metrics)} task types")
        
        return task_metrics
    
    def compute_hallucination_rate_by_category(
        self,
        annotations: List[Dict]
    ) -> Dict:
        """
        Compute hallucination rates by category.
        
        Args:
            annotations: List of parsed annotation dictionaries
        
        Returns:
            Dictionary with category-wise hallucination rates
        """
        category_counts = {
            'Evident Baseless Info': 0,
            'Subtle Conflict': 0,
            'Nuanced Misrepresentation': 0,
            'Reasoning Error': 0,
            'Total Responses': len(annotations),
            'Hallucinated Responses': 0
        }
        
        for ann in annotations:
            has_hallucination = False
            
            for span in ann.get('hallucination_spans', []):
                for label in span.get('labels', []):
                    if label in category_counts:
                        category_counts[label] += 1
                        has_hallucination = True
            
            if has_hallucination:
                category_counts['Hallucinated Responses'] += 1
        
        category_rates = {
            'counts': category_counts,
            'overall_rate': (
                category_counts['Hallucinated Responses'] /
                category_counts['Total Responses']
                if category_counts['Total Responses'] > 0 else 0.0
            ),
            'category_proportions': {
                cat: (
                    category_counts[cat] /
                    category_counts['Hallucinated Responses']
                    if category_counts['Hallucinated Responses'] > 0 else 0.0
                )
                for cat in [
                    'Evident Baseless Info',
                    'Subtle Conflict',
                    'Nuanced Misrepresentation',
                    'Reasoning Error'
                ]
            }
        }
        
        logger.info(
            f"Hallucination rates computed: "
            f"overall={category_rates['overall_rate']:.3f}"
        )
        
        return category_rates


def main():
    """
    Example usage of EvaluationMetrics.
    """
    metrics_calc = EvaluationMetrics()
    
    y_true = [True, True, False, True, False, False, True, False, True, False]
    y_pred = [True, False, False, True, False, True, True, False, False, False]
    y_scores = [0.9, 0.6, 0.3, 0.8, 0.2, 0.7, 0.95, 0.1, 0.55, 0.25]
    
    print("Computing classification metrics...")
    class_metrics = metrics_calc.compute_classification_metrics(y_true, y_pred)
    print(f"Accuracy: {class_metrics['accuracy']:.3f}")
    print(f"Precision: {class_metrics['precision']:.3f}")
    print(f"Recall: {class_metrics['recall']:.3f}")
    print(f"F1 Score: {class_metrics['f1_score']:.3f}")
    
    print("\nComputing ROC analysis...")
    roc_analysis = metrics_calc.compute_roc_analysis(y_true, y_scores)
    print(f"AUC: {roc_analysis['auc']:.3f}")
    print(f"Optimal threshold: {roc_analysis['optimal_threshold']:.3f}")
    
    print("\nComputing PR analysis...")
    pr_analysis = metrics_calc.compute_precision_recall_analysis(y_true, y_scores)
    print(f"Average Precision: {pr_analysis['average_precision']:.3f}")
    print(f"Optimal F1: {pr_analysis['optimal_f1']:.3f}")
    
    print("\nComputing correlation...")
    correlation = metrics_calc.compute_point_biserial_correlation(y_scores, y_true)
    print(f"Correlation: {correlation['correlation']:.3f}")
    print(f"P-value: {correlation['p_value']:.4f}")


if __name__ == "__main__":
    main()

