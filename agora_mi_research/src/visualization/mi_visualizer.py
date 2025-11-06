"""
Mechanistic Interpretability visualization module.

Generates publication-quality figures for attention patterns, ECS/PKS trajectories,
hallucination analysis, and AARF intervention effects.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

from ..mi.data_structures import MIAnalysisData, TrajectoryData
from ..mi.data_collector import MIDataCollector


class MIVisualizer:
    """
    Visualizer for mechanistic interpretability analysis.
    
    Generates publication-quality figures for various MI metrics.
    """
    
    def __init__(
        self,
        output_dir: Path,
        style: str = "publication"
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for saving figures
            style: Figure style ("publication" or "presentation")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.attention_dir = self.output_dir / "attention"
        self.ecs_pks_dir = self.output_dir / "ecs_pks"
        self.hallucination_dir = self.output_dir / "hallucination"
        self.logit_lens_dir = self.output_dir / "logit_lens"
        self.intervention_dir = self.output_dir / "intervention"
        
        for dir_path in [
            self.attention_dir,
            self.ecs_pks_dir,
            self.hallucination_dir,
            self.logit_lens_dir,
            self.intervention_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self._configure_style(style)
        logger.info(f"Initialized MIVisualizer: output_dir={output_dir}, style={style}")
    
    def _configure_style(self, style: str):
        """Configure matplotlib style for publication or presentation."""
        if style == "publication":
            plt.rcParams.update({
                'font.size': 10,
                'font.family': 'serif',
                'font.serif': ['Times New Roman'],
                'text.usetex': False,
                'figure.figsize': (6, 4),
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'axes.labelsize': 10,
                'axes.titlesize': 11,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'lines.linewidth': 1.5,
                'lines.markersize': 6
            })
        else:
            plt.rcParams.update({
                'font.size': 12,
                'figure.figsize': (10, 6),
                'figure.dpi': 150,
                'savefig.dpi': 150
            })
    
    def plot_attention_heatmap_layer_head(
        self,
        response_id: str,
        attention_means: np.ndarray,
        copying_heads: Optional[List[Dict]] = None,
        title: Optional[str] = None
    ):
        """
        Plot layer x head attention heatmap to context tokens.
        
        Args:
            response_id: Response identifier
            attention_means: Array [num_layers, num_heads] with mean attention to context
            copying_heads: Optional list of copying head dictionaries
            title: Optional plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(
            attention_means,
            aspect='auto',
            cmap='YlOrRd',
            interpolation='nearest'
        )
        
        if copying_heads:
            for ch in copying_heads:
                layer_idx = ch.get('layer', -1)
                head_idx = ch.get('head', -1)
                if 0 <= layer_idx < attention_means.shape[0] and 0 <= head_idx < attention_means.shape[1]:
                    ax.add_patch(plt.Rectangle(
                        (head_idx - 0.5, layer_idx - 0.5),
                        1, 1,
                        fill=False,
                        edgecolor='blue',
                        linewidth=2
                    ))
        
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Layer Index')
        ax.set_title(title or f'Attention to Context: Layer × Head (Response {response_id})')
        
        plt.colorbar(im, ax=ax, label='Mean Attention to Context')
        plt.tight_layout()
        
        output_file = self.attention_dir / f"heatmap_layer_head_{response_id}.png"
        plt.savefig(output_file)
        plt.close()
        
        logger.debug(f"Saved attention heatmap to {output_file}")
    
    def plot_attention_trajectory(
        self,
        response_id: str,
        trajectory: TrajectoryData,
        layers: Optional[List[int]] = None,
        title: Optional[str] = None
    ):
        """
        Plot attention to context over generation steps.
        
        Args:
            response_id: Response identifier
            trajectory: TrajectoryData with ECS information
            layers: Optional list of layer indices to plot (None = all)
            title: Optional plot title
        """
        if trajectory.layer_ecs is None:
            logger.warning(f"No layer ECS data for {response_id}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        num_layers = trajectory.layer_ecs.shape[0]
        num_tokens = trajectory.layer_ecs.shape[1]
        
        if layers is None:
            layers = list(range(num_layers))
        
        for layer_idx in layers:
            if layer_idx < num_layers:
                ax.plot(
                    range(num_tokens),
                    trajectory.layer_ecs[layer_idx, :],
                    label=f'Layer {layer_idx}',
                    alpha=0.7,
                    linewidth=1.5
                )
        
        ax.set_xlabel('Generation Step (Token Position)')
        ax.set_ylabel('ECS (Attention to Context)')
        ax.set_title(title or f'ECS Trajectory: Attention to Context over Generation (Response {response_id})')
        ax.legend(ncol=2, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = self.attention_dir / f"trajectory_context_attention_{response_id}.png"
        plt.savefig(output_file)
        plt.close()
        
        logger.debug(f"Saved attention trajectory to {output_file}")
    
    def plot_ecs_trajectory(
        self,
        response_id: str,
        trajectory: TrajectoryData,
        hallucination_labels: Optional[np.ndarray] = None,
        title: Optional[str] = None
    ):
        """
        Plot ECS over generation steps with optional hallucination labels.
        
        Args:
            response_id: Response identifier
            trajectory: TrajectoryData with ECS information
            hallucination_labels: Optional array of hallucination labels per token
            title: Optional plot title
        """
        if trajectory.token_ecs is None:
            logger.warning(f"No token ECS data for {response_id}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        num_tokens = len(trajectory.token_ecs)
        x = range(num_tokens)
        
        if hallucination_labels is not None:
            factual_mask = ~hallucination_labels
            hallucinated_mask = hallucination_labels
            
            ax.scatter(
                np.array(x)[factual_mask],
                trajectory.token_ecs[factual_mask],
                c='blue',
                alpha=0.6,
                label='Factual',
                s=20
            )
            ax.scatter(
                np.array(x)[hallucinated_mask],
                trajectory.token_ecs[hallucinated_mask],
                c='red',
                alpha=0.6,
                label='Hallucinated',
                s=20
            )
        else:
            ax.plot(x, trajectory.token_ecs, label='ECS', linewidth=2, color='blue')
        
        ax.set_xlabel('Generation Step (Token Position)')
        ax.set_ylabel('ECS Value')
        ax.set_title(title or f'ECS Trajectory over Generation (Response {response_id})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = self.ecs_pks_dir / f"ecs_trajectory_{response_id}.png"
        plt.savefig(output_file)
        plt.close()
        
        logger.debug(f"Saved ECS trajectory to {output_file}")
    
    def plot_pks_components(
        self,
        response_id: str,
        trajectory: TrajectoryData,
        title: Optional[str] = None
    ):
        """
        Plot PKS component breakdown as stacked area plot.
        
        Args:
            response_id: Response identifier
            trajectory: TrajectoryData with PKS component information
            title: Optional plot title
        """
        if trajectory.token_pks is None:
            logger.warning(f"No token PKS data for {response_id}")
            return
        
        if trajectory.pks_confidence is None or trajectory.pks_consistency is None or trajectory.pks_entropy is None:
            logger.warning(f"Incomplete PKS component data for {response_id}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        num_tokens = len(trajectory.token_pks)
        x = range(num_tokens)
        
        ax.fill_between(
            x,
            0,
            trajectory.pks_confidence,
            label='Confidence',
            alpha=0.6,
            color='blue'
        )
        ax.fill_between(
            x,
            trajectory.pks_confidence,
            trajectory.pks_confidence + trajectory.pks_consistency,
            label='Consistency',
            alpha=0.6,
            color='green'
        )
        ax.fill_between(
            x,
            trajectory.pks_confidence + trajectory.pks_consistency,
            trajectory.token_pks,
            label='Low Entropy',
            alpha=0.6,
            color='orange'
        )
        
        ax.set_xlabel('Generation Step (Token Position)')
        ax.set_ylabel('PKS Component Value')
        ax.set_title(title or f'PKS Component Breakdown (Response {response_id})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = self.ecs_pks_dir / f"pks_components_{response_id}.png"
        plt.savefig(output_file)
        plt.close()
        
        logger.debug(f"Saved PKS components plot to {output_file}")
    
    def plot_ecs_pks_scatter(
        self,
        response_id: str,
        ecs_trajectory: TrajectoryData,
        pks_trajectory: TrajectoryData,
        hallucination_labels: Optional[np.ndarray] = None,
        title: Optional[str] = None
    ):
        """
        Plot ECS vs PKS scatter plot with color coding by generation step.
        
        Args:
            response_id: Response identifier
            ecs_trajectory: TrajectoryData with ECS
            pks_trajectory: TrajectoryData with PKS
            hallucination_labels: Optional hallucination labels
            title: Optional plot title
        """
        if ecs_trajectory.token_ecs is None or pks_trajectory.token_pks is None:
            logger.warning(f"Incomplete ECS/PKS data for {response_id}")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        num_tokens = len(ecs_trajectory.token_ecs)
        
        if hallucination_labels is not None:
            factual_mask = ~hallucination_labels
            hallucinated_mask = hallucination_labels
            
            ax.scatter(
                ecs_trajectory.token_ecs[factual_mask],
                pks_trajectory.token_pks[factual_mask],
                c=range(num_tokens)[:len(ecs_trajectory.token_ecs[factual_mask])],
                cmap='Blues',
                alpha=0.6,
                label='Factual',
                s=30,
                edgecolors='black',
                linewidths=0.5
            )
            ax.scatter(
                ecs_trajectory.token_ecs[hallucinated_mask],
                pks_trajectory.token_pks[hallucinated_mask],
                c=range(num_tokens)[:len(ecs_trajectory.token_ecs[hallucinated_mask])],
                cmap='Reds',
                alpha=0.6,
                label='Hallucinated',
                s=30,
                edgecolors='black',
                linewidths=0.5
            )
        else:
            scatter = ax.scatter(
                ecs_trajectory.token_ecs,
                pks_trajectory.token_pks,
                c=range(num_tokens),
                cmap='viridis',
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidths=0.5
            )
            plt.colorbar(scatter, ax=ax, label='Generation Step')
        
        ax.set_xlabel('ECS (External Context Score)')
        ax.set_ylabel('PKS (Parametric Knowledge Score)')
        ax.set_title(title or f'ECS vs PKS Scatter (Response {response_id})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = self.ecs_pks_dir / f"ecs_pks_scatter_{response_id}.png"
        plt.savefig(output_file)
        plt.close()
        
        logger.debug(f"Saved ECS/PKS scatter to {output_file}")
    
    def plot_hallucination_score_trajectory(
        self,
        response_id: str,
        trajectory: TrajectoryData,
        threshold: float = 0.6,
        title: Optional[str] = None
    ):
        """
        Plot hallucination score evolution with intervention trigger points.
        
        Args:
            response_id: Response identifier
            trajectory: TrajectoryData with hallucination scores
            threshold: Intervention threshold
            title: Optional plot title
        """
        if trajectory.hallucination_scores is None:
            logger.warning(f"No hallucination score data for {response_id}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        num_tokens = len(trajectory.hallucination_scores)
        x = range(num_tokens)
        
        ax.plot(x, trajectory.hallucination_scores, label='Hallucination Score', linewidth=2, color='red')
        ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        
        trigger_indices = np.where(trajectory.hallucination_scores >= threshold)[0]
        if len(trigger_indices) > 0:
            ax.scatter(
                trigger_indices,
                trajectory.hallucination_scores[trigger_indices],
                c='red',
                s=100,
                marker='x',
                label='Intervention Trigger',
                zorder=5
            )
        
        ax.set_xlabel('Generation Step (Token Position)')
        ax.set_ylabel('Hallucination Score')
        ax.set_title(title or f'Hallucination Score Trajectory (Response {response_id})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = self.hallucination_dir / f"score_trajectory_{response_id}.png"
        plt.savefig(output_file)
        plt.close()
        
        logger.debug(f"Saved hallucination score trajectory to {output_file}")
    
    def plot_intervention_effectiveness(
        self,
        baseline_rates: List[float],
        aarf_rates: List[float],
        title: Optional[str] = None
    ):
        """
        Plot intervention effectiveness comparison.
        
        Args:
            baseline_rates: List of baseline hallucination rates
            aarf_rates: List of AARF hallucination rates
            title: Optional plot title
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = np.arange(len(baseline_rates))
        width = 0.35
        
        ax.bar(x - width/2, baseline_rates, width, label='Baseline', alpha=0.8, color='red')
        ax.bar(x + width/2, aarf_rates, width, label='AARF', alpha=0.8, color='green')
        
        ax.set_xlabel('Response Group')
        ax.set_ylabel('Hallucination Rate')
        ax.set_title(title or 'AARF Intervention Effectiveness Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Group {i+1}' for i in range(len(baseline_rates))])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        output_file = self.intervention_dir / "effectiveness_comparison.png"
        plt.savefig(output_file)
        plt.close()
        
        logger.debug(f"Saved intervention effectiveness plot to {output_file}")
    
    def generate_all_visualizations(
        self,
        mi_data: MIAnalysisData,
        hallucination_labels: Optional[np.ndarray] = None
    ):
        """
        Generate all visualizations for a complete MI analysis.
        
        Args:
            mi_data: MIAnalysisData object
            hallucination_labels: Optional hallucination labels per token
        """
        response_id = mi_data.response_id or "unknown"
        
        logger.info(f"Generating visualizations for {response_id}...")
        
        if mi_data.ecs_trajectory:
            self.plot_ecs_trajectory(response_id, mi_data.ecs_trajectory, hallucination_labels)
        
        if mi_data.pks_trajectory:
            self.plot_pks_components(response_id, mi_data.pks_trajectory)
        
        if mi_data.ecs_trajectory and mi_data.pks_trajectory:
            self.plot_ecs_pks_scatter(
                response_id,
                mi_data.ecs_trajectory,
                mi_data.pks_trajectory,
                hallucination_labels
            )
        
        if mi_data.hallucination_trajectory:
            self.plot_hallucination_score_trajectory(response_id, mi_data.hallucination_trajectory)
        
        logger.info(f"Completed visualizations for {response_id}")

