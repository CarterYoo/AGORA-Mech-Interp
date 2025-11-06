"""
Phase 3.3: AARF Intervention Evaluation

Compare baseline and AARF responses to evaluate hallucination mitigation effectiveness.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import yaml
import numpy as np
from typing import List, Dict
from loguru import logger

from src.annotation.granite_guardian import GraniteGuardianAnnotator
from src.evaluation.statistical_tests import StatisticalTester


logger.add(
    "logs/phase3_3_evaluate_aarf_{time}.log",
    rotation="500 MB",
    retention="30 days",
    level="INFO"
)


def load_aarf_results(results_path: str) -> List[Dict]:
    """
    Load AARF intervention results.
    
    Args:
        results_path: Path to AARF results JSON file
    
    Returns:
        List of result dictionaries
    """
    results_file = project_root / results_path
    
    if not results_file.exists():
        logger.error(f"AARF results file not found: {results_file}")
        return []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    successful = [r for r in results if r.get('success')]
    
    logger.info(
        f"Loaded {len(successful)}/{len(results)} successful results "
        f"from {results_file}"
    )
    
    return successful


def evaluate_with_granite_guardian(
    question: str,
    context: str,
    response: str,
    annotator: GraniteGuardianAnnotator
) -> Dict:
    """
    Evaluate response using Granite Guardian.
    
    Args:
        question: Question text
        context: Retrieved context
        response: Generated response
        annotator: GraniteGuardianAnnotator instance
    
    Returns:
        Evaluation dictionary
    """
    try:
        result = annotator.detect_hallucination(
            question=question,
            context=context,
            response=response
        )
        return result
    except Exception as e:
        logger.warning(f"Granite Guardian evaluation failed: {e}")
        return {
            'is_hallucination': None,
            'score': None,
            'error': str(e)
        }


def main():
    """
    Execute Phase 3.3 Evaluation: Compare baseline vs AARF responses.
    
    Steps:
    1. Load AARF intervention results
    2. Initialize Granite Guardian annotator
    3. For each response pair:
       a. Evaluate baseline response
       b. Evaluate AARF response
       c. Compare results
    4. Compute aggregate statistics
    5. Perform statistical tests
    6. Generate comparison report
    """
    logger.info("="*60)
    logger.info("Phase 3.3: AARF Intervention Evaluation")
    logger.info("="*60)
    
    config = load_config()
    annotation_config = config.get('annotation', {})
    
    logger.info("\nStep 1: Loading AARF intervention results...")
    
    aarf_results_file = "outputs/phase3_3_aarf/aarf_intervention_results.json"
    aarf_results = load_aarf_results(aarf_results_file)
    
    if not aarf_results:
        logger.error("No AARF results found. Run Phase 3.3 AARF intervention first.")
        return 1
    
    logger.info(f"Loaded {len(aarf_results)} AARF results")
    
    logger.info("\nStep 2: Initializing Granite Guardian annotator...")
    
    try:
        annotator = GraniteGuardianAnnotator(
            use_vllm=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info("Granite Guardian initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize Granite Guardian: {e}")
        logger.warning("Skipping Granite Guardian evaluation")
        annotator = None
    
    logger.info("\nStep 3: Evaluating responses...")
    logger.warning(
        f"Evaluating {len(aarf_results)} response pairs. "
        "This may take a while."
    )
    
    evaluations = []
    
    for i, result in enumerate(aarf_results):
        try:
            question = result.get('question', '')
            baseline_response = result.get('baseline_response', '')
            aarf_response = result.get('aarf_response', '')
            
            if not question or not baseline_response:
                logger.warning(f"Skipping result {i+1}: missing question or response")
                continue
            
            context_segments = result.get('retrieved_segments', [])
            context = '\n\n'.join([seg.get('text', '') for seg in context_segments])
            
            eval_entry = {
                'question_id': result.get('question_id', f'q_{i:04d}'),
                'question': question,
                'aarf_intervention_applied': result.get('aarf_intervention_applied', False)
            }
            
            if annotator:
                baseline_eval = evaluate_with_granite_guardian(
                    question=question,
                    context=context,
                    response=baseline_response,
                    annotator=annotator
                )
                eval_entry['baseline_evaluation'] = baseline_eval
                
                if aarf_response:
                    aarf_eval = evaluate_with_granite_guardian(
                        question=question,
                        context=context,
                        response=aarf_response,
                        annotator=annotator
                    )
                    eval_entry['aarf_evaluation'] = aarf_eval
            else:
                eval_entry['baseline_evaluation'] = None
                eval_entry['aarf_evaluation'] = None
            
            evaluations.append(eval_entry)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i+1}/{len(aarf_results)} responses")
        
        except Exception as e:
            logger.error(f"Error evaluating result {i+1}: {e}")
            evaluations.append({
                'question_id': result.get('question_id', f'q_{i:04d}'),
                'error': str(e)
            })
    
    logger.info("\nStep 4: Computing statistics...")
    
    if annotator:
        baseline_hallucinations = sum(
            1 for e in evaluations
            if e.get('baseline_evaluation', {}).get('is_hallucination', False)
        )
        
        aarf_hallucinations = sum(
            1 for e in evaluations
            if e.get('aarf_evaluation', {}).get('is_hallucination', False)
        )
        
        total_evaluated = len([e for e in evaluations if 'baseline_evaluation' in e])
        
        if total_evaluated > 0:
            baseline_rate = baseline_hallucinations / total_evaluated
            aarf_rate = aarf_hallucinations / total_evaluated
            improvement = (baseline_rate - aarf_rate) / baseline_rate if baseline_rate > 0 else 0.0
            
            stats = {
                'total_evaluated': total_evaluated,
                'baseline_hallucination_rate': float(baseline_rate),
                'aarf_hallucination_rate': float(aarf_rate),
                'absolute_improvement': float(baseline_rate - aarf_rate),
                'relative_improvement': float(improvement),
                'baseline_hallucinations': baseline_hallucinations,
                'aarf_hallucinations': aarf_hallucinations
            }
        else:
            stats = {'error': 'No valid evaluations'}
    else:
        stats = {'error': 'Granite Guardian not available'}
    
    logger.info("\nStep 5: Performing statistical tests...")
    
    if annotator and 'baseline_hallucination_rate' in stats:
        tester = StatisticalTester()
        
        baseline_labels = [
            e.get('baseline_evaluation', {}).get('is_hallucination', False)
            for e in evaluations
            if 'baseline_evaluation' in e
        ]
        
        aarf_labels = [
            e.get('aarf_evaluation', {}).get('is_hallucination', False)
            for e in evaluations
            if 'aarf_evaluation' in e
        ]
        
        if len(baseline_labels) == len(aarf_labels) and len(baseline_labels) > 0:
            chi2, p_value = tester.chi_square_test(
                baseline_labels, aarf_labels
            )
            
            effect_size = tester.cohens_d(
                baseline_labels, aarf_labels
            )
            
            stats['statistical_tests'] = {
                'chi_square': float(chi2),
                'p_value': float(p_value),
                'effect_size_cohens_d': float(effect_size),
                'significant': p_value < 0.05
            }
            
            logger.info(f"Statistical test results:")
            logger.info(f"  Chi-square: {chi2:.4f}")
            logger.info(f"  p-value: {p_value:.6f}")
            logger.info(f"  Effect size (Cohen's d): {effect_size:.4f}")
            logger.info(f"  Significant: {stats['statistical_tests']['significant']}")
    
    logger.info("\nStep 6: Exporting evaluation results...")
    
    output_dir = project_root / "outputs/phase3_3_aarf"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluations_file = output_dir / 'aarf_evaluations.json'
    with open(evaluations_file, 'w', encoding='utf-8') as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)
    
    stats_file = output_dir / 'aarf_evaluation_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"  Evaluations: {evaluations_file}")
    logger.info(f"  Statistics: {stats_file}")
    
    logger.info("\n" + "="*60)
    logger.info("Phase 3.3 Evaluation Complete")
    logger.info("="*60)
    
    if 'baseline_hallucination_rate' in stats:
        logger.info(f"Baseline hallucination rate: {stats['baseline_hallucination_rate']:.2%}")
        logger.info(f"AARF hallucination rate: {stats['aarf_hallucination_rate']:.2%}")
        logger.info(f"Improvement: {stats['relative_improvement']:.2%}")
    
    return 0


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load experiment configuration."""
    config_file = project_root / config_path
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


if __name__ == "__main__":
    import torch
    exit_code = main()
    sys.exit(exit_code)

