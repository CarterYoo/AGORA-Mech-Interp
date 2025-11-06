"""
Phase 3: Generate MI Visualizations

This experiment generates all mechanistic interpretability visualizations
from collected MI analysis data.

Visualizations include:
- Attention heatmaps (layer x head)
- ECS/PKS trajectories
- Hallucination score analysis
- Logit Lens analysis
- AARF intervention effects
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import yaml
import numpy as np
from typing import List, Dict, Optional
from loguru import logger

from src.mi.data_collector import MIDataCollector
from src.visualization.mi_visualizer import MIVisualizer
from src.mi.data_structures import MIAnalysisData

logger.add(
    "logs/phase3_visualize_mi_{time}.log",
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


def load_mi_analysis_data(
    data_collector: MIDataCollector,
    response_ids: Optional[List[str]] = None
) -> List[MIAnalysisData]:
    """
    Load MI analysis data from storage.
    
    Args:
        data_collector: MIDataCollector instance
        response_ids: Optional list of response IDs to load (None = all)
    
    Returns:
        List of MIAnalysisData objects
    """
    summary_files = list(data_collector.output_dir.glob("mi_analysis_*.json"))
    
    if response_ids:
        summary_files = [
            f for f in summary_files
            if any(rid in f.name for rid in response_ids)
        ]
    
    mi_data_list = []
    
    for summary_file in summary_files:
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            
            mi_data = MIAnalysisData(
                response_id=data_dict.get('response_id', 'unknown'),
                question_id=data_dict.get('question_id', ''),
                question=data_dict.get('question', ''),
                context_start=data_dict.get('context_start'),
                context_end=data_dict.get('context_end'),
                input_length=data_dict.get('input_length', 0),
                output_length=data_dict.get('output_length', 0),
                copying_heads=data_dict.get('copying_heads', []),
                intervention_applied=data_dict.get('intervention_applied', False),
                baseline_metrics=data_dict.get('baseline_metrics', {}),
                aarf_metrics=data_dict.get('aarf_metrics', {}),
                timestamp=data_dict.get('timestamp', ''),
                model_name=data_dict.get('model_name', ''),
                config=data_dict.get('config', {})
            )
            
            if 'ecs_trajectory' in data_dict and data_dict['ecs_trajectory']:
                ecs_traj_dict = data_dict['ecs_trajectory']
                ecs_trajectory = data_collector.load_ecs_trajectory(mi_data.response_id)
                if ecs_trajectory:
                    mi_data.ecs_trajectory = ecs_trajectory
            
            if 'pks_trajectory' in data_dict and data_dict['pks_trajectory']:
                pks_traj_dict = data_dict['pks_trajectory']
                from src.mi.data_structures import TrajectoryData
                pks_trajectory = TrajectoryData()
                if 'token_pks' in pks_traj_dict:
                    pks_trajectory.token_pks = np.array(pks_traj_dict['token_pks'])
                if 'pks_confidence' in pks_traj_dict:
                    pks_trajectory.pks_confidence = np.array(pks_traj_dict['pks_confidence'])
                if 'pks_consistency' in pks_traj_dict:
                    pks_trajectory.pks_consistency = np.array(pks_traj_dict['pks_consistency'])
                if 'pks_entropy' in pks_traj_dict:
                    pks_trajectory.pks_entropy = np.array(pks_traj_dict['pks_entropy'])
                mi_data.pks_trajectory = pks_trajectory
            
            if 'hallucination_trajectory' in data_dict and data_dict['hallucination_trajectory']:
                hall_traj_dict = data_dict['hallucination_trajectory']
                from src.mi.data_structures import TrajectoryData
                hall_trajectory = TrajectoryData()
                if 'hallucination_scores' in hall_traj_dict:
                    hall_trajectory.hallucination_scores = np.array(hall_traj_dict['hallucination_scores'])
                if 'ecs_component' in hall_traj_dict:
                    hall_trajectory.ecs_component = np.array(hall_traj_dict['ecs_component'])
                if 'pks_component' in hall_traj_dict:
                    hall_trajectory.pks_component = np.array(hall_traj_dict['pks_component'])
                mi_data.hallucination_trajectory = hall_trajectory
            
            mi_data_list.append(mi_data)
            
        except Exception as e:
            logger.error(f"Error loading {summary_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(mi_data_list)} MI analysis data files")
    return mi_data_list


def main():
    """
    Generate all MI visualizations.
    
    Steps:
    1. Load MI analysis data from storage
    2. Initialize visualizer
    3. Generate visualizations for each response
    4. Generate aggregate visualizations
    """
    logger.info("="*60)
    logger.info("Phase 3: Generate MI Visualizations")
    logger.info("="*60)
    
    config = load_config()
    output_config = config['output']
    
    logger.info("\nStep 1: Initializing data collector and visualizer...")
    
    mi_data_dir = project_root / output_config['results_path'] / "phase3_mi"
    figures_dir = project_root / output_config['results_path'] / "figures" / "mi_analysis"
    
    data_collector = MIDataCollector(mi_data_dir)
    visualizer = MIVisualizer(figures_dir, style="publication")
    
    logger.info("\nStep 2: Loading MI analysis data...")
    mi_data_list = load_mi_analysis_data(data_collector)
    
    if not mi_data_list:
        logger.warning("No MI analysis data found. Please run Phase 3.3 AARF intervention first.")
        return 1
    
    logger.info(f"\nStep 3: Generating visualizations for {len(mi_data_list)} responses...")
    
    for i, mi_data in enumerate(mi_data_list):
        logger.info(f"Generating visualizations for response {i+1}/{len(mi_data_list)}: {mi_data.response_id}")
        
        try:
            visualizer.generate_all_visualizations(mi_data)
        except Exception as e:
            logger.error(f"Error generating visualizations for {mi_data.response_id}: {e}")
            continue
    
    logger.info("\nStep 4: Generating aggregate visualizations...")
    
    if len(mi_data_list) > 1:
        baseline_rates = []
        aarf_rates = []
        
        for mi_data in mi_data_list:
            if mi_data.baseline_metrics:
                baseline_rates.append(mi_data.baseline_metrics.get('hallucination_rate', 0.0))
            if mi_data.aarf_metrics:
                aarf_rates.append(mi_data.aarf_metrics.get('hallucination_rate', 0.0))
        
        if baseline_rates and aarf_rates:
            visualizer.plot_intervention_effectiveness(baseline_rates, aarf_rates)
    
    logger.info("\n" + "="*60)
    logger.info("Phase 3 (MI Visualizations) Complete")
    logger.info("="*60)
    logger.info(f"Visualizations saved to: {figures_dir}")
    logger.info(f"Total responses visualized: {len(mi_data_list)}")


if __name__ == "__main__":
    main()

