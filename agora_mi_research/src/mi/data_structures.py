"""
Data structures for mechanistic interpretability analysis.

Provides structured containers for organizing MI data, intervention events,
and trajectory data for efficient storage and analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from datetime import datetime


@dataclass
class TrajectoryData:
    """
    Container for time-series trajectory data.
    
    Stores per-token metrics across generation steps.
    """
    token_ecs: Optional[np.ndarray] = None  # [num_tokens]
    layer_ecs: Optional[np.ndarray] = None  # [num_layers, num_tokens]
    head_ecs: Optional[np.ndarray] = None  # [num_layers, num_heads, num_tokens]
    token_pks: Optional[np.ndarray] = None  # [num_tokens]
    pks_confidence: Optional[np.ndarray] = None  # [num_tokens]
    pks_consistency: Optional[np.ndarray] = None  # [num_tokens]
    pks_entropy: Optional[np.ndarray] = None  # [num_tokens]
    hallucination_scores: Optional[np.ndarray] = None  # [num_tokens]
    ecs_component: Optional[np.ndarray] = None  # [num_tokens] for h_score
    pks_component: Optional[np.ndarray] = None  # [num_tokens] for h_score
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    result[key] = value.cpu().numpy().tolist()
                else:
                    result[key] = value
        return result


@dataclass
class InterventionEvent:
    """
    Container for AARF intervention event data.
    
    Tracks when and where interventions occur.
    """
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_step: int = 0
    token_position: int = 0
    intervention_type: str = ""  # "attention" or "ffn"
    layer_idx: Optional[int] = None
    head_idx: Optional[int] = None
    context_start: int = 0
    context_end: int = 0
    attention_multiplier: Optional[float] = None
    ffn_suppress: Optional[float] = None
    before_value: Optional[float] = None
    after_value: Optional[float] = None
    hallucination_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'generation_step': self.generation_step,
            'token_position': self.token_position,
            'intervention_type': self.intervention_type,
            'layer_idx': self.layer_idx,
            'head_idx': self.head_idx,
            'context_start': self.context_start,
            'context_end': self.context_end,
            'attention_multiplier': self.attention_multiplier,
            'ffn_suppress': self.ffn_suppress,
            'before_value': self.before_value,
            'after_value': self.after_value,
            'hallucination_score': self.hallucination_score
        }


@dataclass
class MIAnalysisData:
    """
    Comprehensive container for mechanistic interpretability analysis data.
    
    Organizes all MI metrics, attention patterns, and intervention data
    for a single response.
    """
    response_id: str = ""
    question_id: str = ""
    question: str = ""
    
    # Attention data
    attention_tensors: Optional[List[torch.Tensor]] = None  # List of [layers, heads, seq_len, seq_len]
    attention_stats: Dict = field(default_factory=dict)
    context_start: Optional[int] = None
    context_end: Optional[int] = None
    input_length: int = 0
    output_length: int = 0
    
    # Hidden states
    hidden_states: Optional[List[torch.Tensor]] = None  # List of [layers, seq_len, hidden_dim]
    hidden_state_stats: Dict = field(default_factory=dict)
    
    # Logits
    logits: Optional[List[torch.Tensor]] = None  # List of [vocab_size]
    
    # ECS data
    ecs_trajectory: Optional[TrajectoryData] = None
    copying_heads: List[Dict] = field(default_factory=list)
    
    # PKS data
    pks_trajectory: Optional[TrajectoryData] = None
    
    # Hallucination score
    hallucination_trajectory: Optional[TrajectoryData] = None
    
    # Logit Lens data
    logit_lens_predictions: List[Dict] = field(default_factory=list)
    convergence_layers: List[int] = field(default_factory=list)
    
    # AARF intervention
    intervention_events: List[InterventionEvent] = field(default_factory=list)
    intervention_applied: bool = False
    baseline_metrics: Dict = field(default_factory=dict)
    aarf_metrics: Dict = field(default_factory=dict)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_name: str = ""
    config: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'response_id': self.response_id,
            'question_id': self.question_id,
            'question': self.question,
            'attention_stats': self.attention_stats,
            'context_start': self.context_start,
            'context_end': self.context_end,
            'input_length': self.input_length,
            'output_length': self.output_length,
            'hidden_state_stats': self.hidden_state_stats,
            'copying_heads': self.copying_heads,
            'logit_lens_predictions': self.logit_lens_predictions,
            'convergence_layers': self.convergence_layers,
            'intervention_applied': self.intervention_applied,
            'baseline_metrics': self.baseline_metrics,
            'aarf_metrics': self.aarf_metrics,
            'timestamp': self.timestamp,
            'model_name': self.model_name,
            'config': self.config
        }
        
        if self.ecs_trajectory:
            result['ecs_trajectory'] = self.ecs_trajectory.to_dict()
        
        if self.pks_trajectory:
            result['pks_trajectory'] = self.pks_trajectory.to_dict()
        
        if self.hallucination_trajectory:
            result['hallucination_trajectory'] = self.hallucination_trajectory.to_dict()
        
        result['intervention_events'] = [e.to_dict() for e in self.intervention_events]
        
        return result
    
    def get_trajectory_summary(self) -> Dict:
        """Get summary statistics of trajectories."""
        summary = {}
        
        if self.ecs_trajectory and self.ecs_trajectory.token_ecs is not None:
            ecs = self.ecs_trajectory.token_ecs
            summary['ecs'] = {
                'mean': float(np.mean(ecs)),
                'std': float(np.std(ecs)),
                'min': float(np.min(ecs)),
                'max': float(np.max(ecs)),
                'median': float(np.median(ecs))
            }
        
        if self.pks_trajectory and self.pks_trajectory.token_pks is not None:
            pks = self.pks_trajectory.token_pks
            summary['pks'] = {
                'mean': float(np.mean(pks)),
                'std': float(np.std(pks)),
                'min': float(np.min(pks)),
                'max': float(np.max(pks)),
                'median': float(np.median(pks))
            }
        
        if self.hallucination_trajectory and self.hallucination_trajectory.hallucination_scores is not None:
            h_scores = self.hallucination_trajectory.hallucination_scores
            summary['hallucination_score'] = {
                'mean': float(np.mean(h_scores)),
                'std': float(np.std(h_scores)),
                'min': float(np.min(h_scores)),
                'max': float(np.max(h_scores)),
                'median': float(np.median(h_scores))
            }
        
        return summary

