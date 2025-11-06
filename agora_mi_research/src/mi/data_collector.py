"""
Data collector module for efficient storage of mechanistic interpretability data.

Provides efficient storage of large tensors (HDF5) and structured data organization
for MI analysis results.
"""

import h5py
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from loguru import logger
import torch

from .data_structures import MIAnalysisData, TrajectoryData, InterventionEvent


class MIDataCollector:
    """
    Efficient collector and storage for mechanistic interpretability data.
    
    Handles large tensor storage using HDF5 and structured metadata using JSON.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        use_hdf5: bool = True
    ):
        """
        Initialize data collector.
        
        Args:
            output_dir: Root directory for storing MI data
            use_hdf5: Whether to use HDF5 for large tensor storage
        """
        self.output_dir = Path(output_dir)
        self.use_hdf5 = use_hdf5
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.attention_dir = self.output_dir / "attention_data"
        self.ecs_dir = self.output_dir / "ecs_trajectories"
        self.pks_dir = self.output_dir / "pks_trajectories"
        self.hallucination_dir = self.output_dir / "hallucination_scores"
        self.logit_lens_dir = self.output_dir / "logit_lens"
        self.intervention_dir = self.output_dir / "intervention_data"
        
        for dir_path in [
            self.attention_dir,
            self.ecs_dir,
            self.pks_dir,
            self.hallucination_dir,
            self.logit_lens_dir,
            self.intervention_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MIDataCollector: output_dir={output_dir}")
    
    def save_attention_data(
        self,
        response_id: str,
        attention_tensors: List[torch.Tensor],
        attention_stats: Dict,
        save_full: bool = True
    ):
        """
        Save attention pattern data.
        
        Args:
            response_id: Unique identifier for response
            attention_tensors: List of attention tensors [layers, heads, seq_len, seq_len]
            attention_stats: Aggregated attention statistics
            save_full: Whether to save full attention tensors (requires HDF5)
        """
        stats_file = self.attention_dir / f"attention_stats_{response_id}.json"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(attention_stats, f, indent=2, ensure_ascii=False)
        
        if save_full and self.use_hdf5 and attention_tensors:
            h5_file = self.attention_dir / f"full_attention_{response_id}.h5"
            
            with h5py.File(h5_file, 'w') as f:
                for i, att_tensor in enumerate(attention_tensors):
                    if isinstance(att_tensor, torch.Tensor):
                        att_array = att_tensor.cpu().numpy()
                    else:
                        att_array = att_tensor
                    
                    f.create_dataset(
                        f"step_{i}",
                        data=att_array,
                        compression="gzip",
                        compression_opts=9
                    )
                f.attrs['num_steps'] = len(attention_tensors)
                f.attrs['shape'] = att_array.shape if attention_tensors else None
            
            logger.debug(f"Saved full attention tensors to {h5_file}")
        
        logger.debug(f"Saved attention stats for {response_id}")
    
    def save_ecs_trajectory(
        self,
        response_id: str,
        trajectory: TrajectoryData,
        stats: Dict
    ):
        """
        Save ECS trajectory data.
        
        Args:
            response_id: Unique identifier for response
            trajectory: TrajectoryData object with ECS information
            stats: ECS statistics dictionary
        """
        if trajectory.token_ecs is not None:
            token_file = self.ecs_dir / f"token_ecs_{response_id}.npy"
            np.save(token_file, trajectory.token_ecs)
        
        if trajectory.layer_ecs is not None:
            layer_file = self.ecs_dir / f"layer_ecs_{response_id}.npy"
            np.save(layer_file, trajectory.layer_ecs)
        
        if trajectory.head_ecs is not None:
            head_file = self.ecs_dir / f"head_ecs_{response_id}.npy"
            np.save(head_file, trajectory.head_ecs)
        
        stats_file = self.ecs_dir / f"ecs_stats_{response_id}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved ECS trajectory for {response_id}")
    
    def save_pks_trajectory(
        self,
        response_id: str,
        trajectory: TrajectoryData,
        stats: Dict
    ):
        """
        Save PKS trajectory data.
        
        Args:
            response_id: Unique identifier for response
            trajectory: TrajectoryData object with PKS information
            stats: PKS statistics dictionary
        """
        if trajectory.token_pks is not None:
            token_file = self.pks_dir / f"token_pks_{response_id}.npy"
            np.save(token_file, trajectory.token_pks)
        
        if trajectory.pks_confidence is not None:
            conf_file = self.pks_dir / f"pks_confidence_{response_id}.npy"
            np.save(conf_file, trajectory.pks_confidence)
        
        if trajectory.pks_consistency is not None:
            cons_file = self.pks_dir / f"pks_consistency_{response_id}.npy"
            np.save(cons_file, trajectory.pks_consistency)
        
        if trajectory.pks_entropy is not None:
            ent_file = self.pks_dir / f"pks_entropy_{response_id}.npy"
            np.save(ent_file, trajectory.pks_entropy)
        
        stats_file = self.pks_dir / f"pks_stats_{response_id}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved PKS trajectory for {response_id}")
    
    def save_hallucination_scores(
        self,
        response_id: str,
        trajectory: TrajectoryData,
        triggers: List[Dict]
    ):
        """
        Save hallucination score trajectory.
        
        Args:
            response_id: Unique identifier for response
            trajectory: TrajectoryData object with hallucination scores
            triggers: List of intervention trigger points
        """
        if trajectory.hallucination_scores is not None:
            scores_file = self.hallucination_dir / f"scores_{response_id}.npy"
            np.save(scores_file, trajectory.hallucination_scores)
        
        if trajectory.ecs_component is not None:
            ecs_comp_file = self.hallucination_dir / f"ecs_component_{response_id}.npy"
            np.save(ecs_comp_file, trajectory.ecs_component)
        
        if trajectory.pks_component is not None:
            pks_comp_file = self.hallucination_dir / f"pks_component_{response_id}.npy"
            np.save(pks_comp_file, trajectory.pks_component)
        
        triggers_file = self.hallucination_dir / f"triggers_{response_id}.json"
        with open(triggers_file, 'w', encoding='utf-8') as f:
            json.dump(triggers, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved hallucination scores for {response_id}")
    
    def save_logit_lens_data(
        self,
        response_id: str,
        predictions: List[Dict],
        convergence: List[int]
    ):
        """
        Save Logit Lens analysis data.
        
        Args:
            response_id: Unique identifier for response
            predictions: List of layer-wise predictions per token
            convergence: List of convergence layers per token
        """
        predictions_file = self.logit_lens_dir / f"predictions_{response_id}.json"
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        convergence_file = self.logit_lens_dir / f"convergence_{response_id}.json"
        with open(convergence_file, 'w', encoding='utf-8') as f:
            json.dump(convergence, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved Logit Lens data for {response_id}")
    
    def save_intervention_data(
        self,
        response_id: str,
        events: List[InterventionEvent],
        comparisons: Optional[np.ndarray] = None
    ):
        """
        Save AARF intervention data.
        
        Args:
            response_id: Unique identifier for response
            events: List of intervention events
            comparisons: Optional before/after comparison arrays
        """
        events_file = self.intervention_dir / f"events_{response_id}.json"
        events_dict = [e.to_dict() for e in events]
        
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(events_dict, f, indent=2, ensure_ascii=False)
        
        if comparisons is not None:
            comp_file = self.intervention_dir / f"comparisons_{response_id}.npy"
            np.save(comp_file, comparisons)
        
        logger.debug(f"Saved intervention data for {response_id}")
    
    def save_mi_analysis(
        self,
        mi_data: MIAnalysisData,
        save_tensors: bool = True
    ):
        """
        Save complete MI analysis data.
        
        Args:
            mi_data: MIAnalysisData object containing all MI information
            save_tensors: Whether to save large tensors (attention, hidden states)
        """
        response_id = mi_data.response_id or "unknown"
        
        if mi_data.ecs_trajectory:
            ecs_stats = {
                'overall_ecs': float(np.mean(mi_data.ecs_trajectory.token_ecs)) if mi_data.ecs_trajectory.token_ecs is not None else 0.0,
                'min_ecs': float(np.min(mi_data.ecs_trajectory.token_ecs)) if mi_data.ecs_trajectory.token_ecs is not None else 0.0,
                'max_ecs': float(np.max(mi_data.ecs_trajectory.token_ecs)) if mi_data.ecs_trajectory.token_ecs is not None else 0.0,
                'std_ecs': float(np.std(mi_data.ecs_trajectory.token_ecs)) if mi_data.ecs_trajectory.token_ecs is not None else 0.0
            }
            self.save_ecs_trajectory(response_id, mi_data.ecs_trajectory, ecs_stats)
        
        if mi_data.pks_trajectory:
            pks_stats = {
                'overall_pks': float(np.mean(mi_data.pks_trajectory.token_pks)) if mi_data.pks_trajectory.token_pks is not None else 0.0,
                'min_pks': float(np.min(mi_data.pks_trajectory.token_pks)) if mi_data.pks_trajectory.token_pks is not None else 0.0,
                'max_pks': float(np.max(mi_data.pks_trajectory.token_pks)) if mi_data.pks_trajectory.token_pks is not None else 0.0,
                'std_pks': float(np.std(mi_data.pks_trajectory.token_pks)) if mi_data.pks_trajectory.token_pks is not None else 0.0
            }
            self.save_pks_trajectory(response_id, mi_data.pks_trajectory, pks_stats)
        
        if mi_data.hallucination_trajectory:
            triggers = [
                {
                    'token_pos': i,
                    'score': float(score),
                    'threshold_exceeded': score >= 0.6
                }
                for i, score in enumerate(mi_data.hallucination_trajectory.hallucination_scores)
                if mi_data.hallucination_trajectory.hallucination_scores is not None
            ]
            self.save_hallucination_scores(response_id, mi_data.hallucination_trajectory, triggers)
        
        if mi_data.intervention_events:
            self.save_intervention_data(response_id, mi_data.intervention_events)
        
        if mi_data.logit_lens_predictions:
            self.save_logit_lens_data(
                response_id,
                mi_data.logit_lens_predictions,
                mi_data.convergence_layers
            )
        
        if save_tensors and mi_data.attention_tensors:
            attention_stats = {
                'num_steps': len(mi_data.attention_tensors),
                'context_start': mi_data.context_start,
                'context_end': mi_data.context_end,
                'input_length': mi_data.input_length,
                'output_length': mi_data.output_length
            }
            self.save_attention_data(
                response_id,
                mi_data.attention_tensors,
                {**attention_stats, **mi_data.attention_stats}
            )
        
        summary_file = self.output_dir / f"mi_analysis_{response_id}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(mi_data.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved complete MI analysis for {response_id}")
    
    def load_attention_data(self, response_id: str) -> Optional[Dict]:
        """Load attention data for a response."""
        stats_file = self.attention_dir / f"attention_stats_{response_id}.json"
        
        if not stats_file.exists():
            return None
        
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        attention_tensors = []
        h5_file = self.attention_dir / f"full_attention_{response_id}.h5"
        
        if h5_file.exists():
            with h5py.File(h5_file, 'r') as f:
                num_steps = f.attrs.get('num_steps', 0)
                for i in range(num_steps):
                    att_array = f[f"step_{i}"][:]
                    attention_tensors.append(torch.from_numpy(att_array))
        
        return {
            'stats': stats,
            'tensors': attention_tensors if attention_tensors else None
        }
    
    def load_ecs_trajectory(self, response_id: str) -> Optional[TrajectoryData]:
        """Load ECS trajectory data."""
        token_file = self.ecs_dir / f"token_ecs_{response_id}.npy"
        
        if not token_file.exists():
            return None
        
        trajectory = TrajectoryData()
        trajectory.token_ecs = np.load(token_file)
        
        layer_file = self.ecs_dir / f"layer_ecs_{response_id}.npy"
        if layer_file.exists():
            trajectory.layer_ecs = np.load(layer_file)
        
        head_file = self.ecs_dir / f"head_ecs_{response_id}.npy"
        if head_file.exists():
            trajectory.head_ecs = np.load(head_file)
        
        return trajectory

