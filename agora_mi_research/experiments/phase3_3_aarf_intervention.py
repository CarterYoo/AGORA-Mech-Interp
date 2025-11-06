"""
Phase 3.3: AARF Intervention Evaluation

This experiment applies AARF (Add Attention Reduce FFN) intervention to
RAG responses and evaluates hallucination mitigation effectiveness.

Reference: ReDEeP framework (Sun et al., 2024)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import yaml
from typing import List, Dict
from loguru import logger

from src.rag.generator import ResponseGenerator
from src.rag.retriever import SemanticRetriever
from src.mi.ecs_calculator import ECSCalculator
from src.mi.pks_calculator import PKSCalculator
from src.mi.aarf_intervention import HallucinationScoreCalculator


logger.add(
    "logs/phase3_3_aarf_{time}.log",
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


def load_rag_responses(phase2_output_path: str) -> List[Dict]:
    """
    Load RAG responses from Phase 2 output.
    
    Args:
        phase2_output_path: Path to Phase 2 RAG responses JSON file
    
    Returns:
        List of RAG response dictionaries
    """
    response_file = project_root / phase2_output_path
    
    if not response_file.exists():
        logger.error(f"RAG responses file not found: {response_file}")
        return []
    
    with open(response_file, 'r', encoding='utf-8') as f:
        responses = json.load(f)
    
    successful = [r for r in responses if r.get('success', True) and 'response' in r]
    
    logger.info(
        f"Loaded {len(successful)}/{len(responses)} successful responses "
        f"from {response_file}"
    )
    
    return successful


def main():
    """
    Execute Phase 3.3: AARF Intervention Evaluation.
    
    Steps:
    1. Load RAG responses from Phase 2
    2. Initialize generators (baseline and AARF-enabled)
    3. For each response:
       a. Compute baseline ECS/PKS
       b. Calculate hallucination score
       c. Apply AARF if threshold exceeded
       d. Regenerate with AARF
       e. Compare responses
    4. Aggregate statistics
    5. Export results
    """
    logger.info("="*60)
    logger.info("Phase 3.3: AARF Intervention Evaluation")
    logger.info("="*60)
    
    config = load_config()
    model_config = config['model']
    aarf_config = config.get('aarf_intervention', {})
    
    logger.info("\nStep 1: Loading RAG responses from Phase 2...")
    
    phase2_responses_file = "outputs/phase2_rag/rag_responses.json"
    baseline_responses = load_rag_responses(phase2_responses_file)
    
    if not baseline_responses:
        logger.error("No RAG responses found. Run Phase 2 first.")
        return 1
    
    logger.info(f"Loaded {len(baseline_responses)} baseline responses")
    
    logger.info("\nStep 2: Initializing generators...")
    
    baseline_generator = ResponseGenerator(
        model_name=model_config['name'],
        load_4bit=model_config['quantization'] == '4bit',
        device_map="auto",
        max_length=model_config['max_length'],
        temperature=model_config['temperature'],
        top_p=model_config['top_p'],
        enable_aarf=False
    )
    
    aarf_generator = ResponseGenerator(
        model_name=model_config['name'],
        load_4bit=model_config['quantization'] == '4bit',
        device_map="auto",
        max_length=model_config['max_length'],
        temperature=model_config['temperature'],
        top_p=model_config['top_p'],
        enable_aarf=True,
        aarf_threshold=aarf_config.get('threshold', 0.6),
        aarf_attention_multiplier=aarf_config.get('attention_multiplier', 1.5),
        aarf_ffn_suppress=aarf_config.get('ffn_suppress', 0.7)
    )
    
    logger.info("Generators initialized")
    
    logger.info("\nStep 3: Initializing MI calculators...")
    
    ecs_calculator = ECSCalculator(
        high_ecs_threshold=config['mi_analysis']['ecs'].get('high_ecs_threshold', 0.3)
    )
    pks_calculator = PKSCalculator()
    hallucination_calculator = HallucinationScoreCalculator(
        alpha=aarf_config.get('alpha', 0.6),
        beta=aarf_config.get('beta', 0.4),
        threshold=aarf_config.get('threshold', 0.6)
    )
    
    logger.info("\nStep 4: Processing responses with AARF...")
    logger.warning(
        f"Processing {len(baseline_responses)} responses. "
        "This may take several hours."
    )
    
    results = []
    intervention_count = 0
    
    for i, baseline_response in enumerate(baseline_responses):
        try:
            question = baseline_response.get('question', '')
            retrieved_segments = baseline_response.get('retrieved_docs', [])
            
            if not question or not retrieved_segments:
                logger.warning(f"Skipping response {i+1}: missing question or segments")
                continue
            
            logger.info(f"\nProcessing response {i+1}/{len(baseline_responses)}")
            logger.debug(f"Question: {question[:100]}...")
            
            aarf_result = aarf_generator.generate_with_aarf(
                question=question,
                retrieved_segments=retrieved_segments,
                max_new_tokens=512,
                ecs_calculator=ecs_calculator,
                pks_calculator=pks_calculator
            )
            
            if aarf_result.get('aarf_intervention_applied'):
                intervention_count += 1
            
            result_entry = {
                'question_id': baseline_response.get('question_id', f'q_{i:04d}'),
                'question': question,
                'baseline_response': baseline_response.get('response', ''),
                'aarf_response': aarf_result.get('aarf_response', aarf_result.get('response', '')),
                'aarf_analysis': aarf_result.get('aarf_analysis', {}),
                'aarf_intervention_applied': aarf_result.get('aarf_intervention_applied', False),
                'aarf_stats': aarf_result.get('aarf_stats', {}),
                'success': True
            }
            
            results.append(result_entry)
            
            if (i + 1) % 10 == 0:
                logger.info(
                    f"Processed {i+1}/{len(baseline_responses)} responses, "
                    f"{intervention_count} interventions applied"
                )
        
        except Exception as e:
            logger.error(f"Error processing response {i+1}: {e}")
            results.append({
                'question_id': baseline_response.get('question_id', f'q_{i:04d}'),
                'success': False,
                'error': str(e)
            })
    
    logger.info(f"\nStep 5: Computing statistics...")
    
    successful_results = [r for r in results if r.get('success')]
    intervention_applied = sum(
        1 for r in successful_results 
        if r.get('aarf_intervention_applied', False)
    )
    
    if successful_results:
        avg_baseline_ecs = np.mean([
            r['aarf_analysis'].get('baseline_ecs', 0.0)
            for r in successful_results
            if 'aarf_analysis' in r
        ])
        
        avg_baseline_pks = np.mean([
            r['aarf_analysis'].get('baseline_pks', 0.0)
            for r in successful_results
            if 'aarf_analysis' in r
        ])
        
        avg_hallucination_score = np.mean([
            r['aarf_analysis'].get('hallucination_score', 0.0)
            for r in successful_results
            if 'aarf_analysis' in r
        ])
        
        intervention_rate = intervention_applied / len(successful_results)
        
        stats = {
            'total_responses': len(baseline_responses),
            'successful': len(successful_results),
            'intervention_applied': intervention_applied,
            'intervention_rate': float(intervention_rate),
            'avg_baseline_ecs': float(avg_baseline_ecs),
            'avg_baseline_pks': float(avg_baseline_pks),
            'avg_hallucination_score': float(avg_hallucination_score),
            'aarf_config': aarf_config
        }
    else:
        stats = {
            'total_responses': len(baseline_responses),
            'successful': 0,
            'error': 'No successful responses'
        }
    
    logger.info("\nStatistics:")
    logger.info(f"  Total responses: {stats['total_responses']}")
    logger.info(f"  Successful: {stats.get('successful', 0)}")
    logger.info(f"  Interventions applied: {stats.get('intervention_applied', 0)}")
    logger.info(f"  Intervention rate: {stats.get('intervention_rate', 0.0):.2%}")
    
    logger.info("\nStep 6: Exporting results...")
    
    output_dir = project_root / "outputs/phase3_3_aarf"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'aarf_intervention_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    stats_file = output_dir / 'aarf_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"  Results: {results_file}")
    logger.info(f"  Statistics: {stats_file}")
    
    logger.info("\n" + "="*60)
    logger.info("Phase 3.3 Complete")
    logger.info("="*60)
    logger.info(f"Total responses: {len(baseline_responses)}")
    logger.info(f"Successful: {stats.get('successful', 0)}")
    logger.info(f"Interventions applied: {stats.get('intervention_applied', 0)}")
    logger.info(f"\nNext step: Phase 3.3 Evaluation (compare baseline vs AARF)")
    
    return 0


if __name__ == "__main__":
    import numpy as np
    exit_code = main()
    sys.exit(exit_code)

