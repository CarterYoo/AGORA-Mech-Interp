"""
Mechanistic Interpretability analysis modules.

This package contains modules for:
- Attention pattern extraction and analysis
- External Context Score (ECS) calculation
- Parametric Knowledge Score (PKS) computation
- Logit Lens implementation
- Layer-wise model behavior analysis
- AARF intervention for hallucination mitigation
"""

from .aarf_intervention import AARFIntervention, HallucinationScoreCalculator
from .ecs_calculator import ECSCalculator
from .pks_calculator import PKSCalculator
from .attention_analyzer import AttentionAnalyzer
from .logit_lens import LogitLens
from .data_structures import MIAnalysisData, TrajectoryData, InterventionEvent
from .data_collector import MIDataCollector

__all__ = [
    'AARFIntervention',
    'HallucinationScoreCalculator',
    'ECSCalculator',
    'PKSCalculator',
    'AttentionAnalyzer',
    'LogitLens',
    'MIAnalysisData',
    'TrajectoryData',
    'InterventionEvent',
    'MIDataCollector'
]

