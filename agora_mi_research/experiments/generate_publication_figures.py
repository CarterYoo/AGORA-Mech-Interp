"""
Generate publication-quality figures for paper.

This script creates all figures and tables needed for the LaTeX paper,
following academic publication standards.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.metrics import roc_curve, roc_auc_score

logger.add(
    "logs/generate_figures_{time}.log",
    rotation="500 MB",
    level="INFO"
)


# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,  # Set True if LaTeX installed
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


def load_results():
    """Load experimental results."""
    logger.info("Loading experimental results...")
    
    results = {}
    
    # Load RAG responses
    rag_file = project_root / "outputs/phase2_agora_rag/rag_responses_agora.json"
    if rag_file.exists():
        with open(rag_file, 'r') as f:
            results['rag_responses'] = json.load(f)
        logger.info(f"Loaded {len(results['rag_responses'])} RAG responses")
    
    # Load MI analyses (placeholder - will be generated in Phase 3)
    mi_file = project_root / "outputs/phase3_mi/mi_analysis_results.json"
    if mi_file.exists():
        with open(mi_file, 'r') as f:
            results['mi_analyses'] = json.load(f)
        logger.info(f"Loaded {len(results['mi_analyses'])} MI analyses")
    
    # Load annotations (placeholder - from manual annotation)
    ann_file = project_root / "data/annotations/parsed_annotations.json"
    if ann_file.exists():
        with open(ann_file, 'r') as f:
            results['annotations'] = json.load(f)
        logger.info(f"Loaded {len(results['annotations'])} annotations")
    
    return results


def figure1_ecs_vs_pks_scatter(results, output_dir):
    """
    Figure 1: Scatter plot of ECS vs PKS with hallucination labels.
    """
    logger.info("Generating Figure 1: ECS vs PKS scatter plot...")
    
    if 'mi_analyses' not in results or 'annotations' not in results:
        logger.warning("Insufficient data for Figure 1, creating placeholder")
        return
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Extract data
    ecs_vals = []
    pks_vals = []
    is_hallucinated = []
    
    for mi_analysis, annotation in zip(results['mi_analyses'], results['annotations']):
        ecs_vals.append(mi_analysis.get('overall_ecs', 0))
        pks_vals.append(mi_analysis.get('overall_pks', 0))
        has_hall = len(annotation.get('hallucination_spans', [])) > 0
        is_hallucinated.append(has_hall)
    
    # Plot
    colors = ['#d62728' if h else '#1f77b4' for h in is_hallucinated]
    markers = ['x' if h else 'o' for h in is_hallucinated]
    
    for ecs, pks, color, marker, hall in zip(ecs_vals, pks_vals, colors, markers, is_hallucinated):
        label = 'Hallucinated' if hall else 'Factual'
        ax.scatter(ecs, pks, c=color, marker=marker, s=50, alpha=0.6, 
                  label=label if (hall and 'Hallucinated' not in ax.get_legend_handles_labels()[1]) or 
                        (not hall and 'Factual' not in ax.get_legend_handles_labels()[1]) else '')
    
    ax.set_xlabel('External Context Score (ECS)')
    ax.set_ylabel('Parametric Knowledge Score (PKS)')
    ax.set_title('ECS vs PKS: Hallucinated vs Factual Responses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_ecs_pks_scatter.pdf')
    plt.savefig(output_dir / 'figure1_ecs_pks_scatter.png')
    plt.close()
    
    logger.info("Figure 1 saved")


def figure2_roc_curve(results, output_dir):
    """
    Figure 2: ROC curve for ECS as hallucination predictor.
    """
    logger.info("Generating Figure 2: ROC curve...")
    
    if 'mi_analyses' not in results or 'annotations' not in results:
        logger.warning("Insufficient data for Figure 2, creating placeholder")
        return
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    ecs_scores = [mi.get('overall_ecs', 0) for mi in results['mi_analyses']]
    y_true = [len(ann.get('hallucination_spans', [])) > 0 for ann in results['annotations']]
    
    # Invert ECS (lower ECS = higher hallucination probability)
    ecs_scores_inv = [1 - ecs for ecs in ecs_scores]
    
    fpr, tpr, thresholds = roc_curve(y_true, ecs_scores_inv)
    auc = roc_auc_score(y_true, ecs_scores_inv)
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ECS (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    # Mark optimal point
    youden = tpr - fpr
    optimal_idx = np.argmax(youden)
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
            label=f'Optimal (threshold = {1-thresholds[optimal_idx]:.3f})')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: ECS-based Hallucination Detection')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_roc_curve.pdf')
    plt.savefig(output_dir / 'figure2_roc_curve.png')
    plt.close()
    
    logger.info(f"Figure 2 saved (AUC = {auc:.3f})")


def figure3_layer_wise_ecs(results, output_dir):
    """
    Figure 3: Layer-wise ECS for hallucinated vs non-hallucinated.
    """
    logger.info("Generating Figure 3: Layer-wise ECS...")
    
    if 'mi_analyses' not in results or 'annotations' not in results:
        logger.warning("Insufficient data for Figure 3, creating placeholder")
        return
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Aggregate layer ECS by hallucination status
    layers = range(32)
    ecs_no_hall_by_layer = [[] for _ in layers]
    ecs_hall_by_layer = [[] for _ in layers]
    
    for mi, ann in zip(results['mi_analyses'], results['annotations']):
        is_hall = len(ann.get('hallucination_spans', [])) > 0
        layer_ecs = mi.get('layer_ecs', [])
        
        for layer_info in layer_ecs:
            layer_idx = layer_info['layer']
            ecs = layer_info['ecs']
            
            if is_hall:
                ecs_hall_by_layer[layer_idx].append(ecs)
            else:
                ecs_no_hall_by_layer[layer_idx].append(ecs)
    
    mean_ecs_no_hall = [np.mean(ecs_list) if ecs_list else 0 for ecs_list in ecs_no_hall_by_layer]
    mean_ecs_hall = [np.mean(ecs_list) if ecs_list else 0 for ecs_list in ecs_hall_by_layer]
    
    ax.plot(layers, mean_ecs_no_hall, 'b-', linewidth=2, label='Factual')
    ax.plot(layers, mean_ecs_hall, 'r-', linewidth=2, label='Hallucinated')
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Mean External Context Score')
    ax.set_title('Layer-wise ECS: Factual vs Hallucinated Responses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_layer_wise_ecs.pdf')
    plt.savefig(output_dir / 'figure3_layer_wise_ecs.png')
    plt.close()
    
    logger.info("Figure 3 saved")


def table1_configuration_comparison(results, output_dir):
    """
    Table 1: Performance comparison across configurations.
    """
    logger.info("Generating Table 1: Configuration comparison...")
    
    # Load comparison results if available
    comparison_dir = project_root / "outputs/comparison"
    
    configs = ['baseline', 'colbert_only', 'dpo_only', 'full_integration']
    table_data = []
    
    for config in configs:
        config_file = comparison_dir / f"{config}_results.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_results = json.load(f)
            
            # Compute statistics
            successful = [r for r in config_results if r.get('success', False)]
            avg_output_len = np.mean([r['token_counts']['output'] for r in successful]) if successful else 0
            
            table_data.append({
                'Configuration': config.replace('_', ' ').title(),
                'N': len(successful),
                'Avg Tokens': f"{avg_output_len:.1f}",
                'Success Rate': f"{len(successful)/len(config_results)*100:.1f}%"
            })
    
    df = pd.DataFrame(table_data)
    
    # Save as LaTeX table
    latex_table = df.to_latex(index=False, escape=False)
    
    with open(output_dir / 'table1_configuration_comparison.tex', 'w') as f:
        f.write(latex_table)
    
    # Also save as CSV
    df.to_csv(output_dir / 'table1_configuration_comparison.csv', index=False)
    
    logger.info("Table 1 saved")


def main():
    """
    Generate all publication figures and tables.
    """
    logger.info("="*60)
    logger.info("Generating Publication Figures")
    logger.info("="*60)
    
    figures_dir = project_root / "paper/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {figures_dir}")
    
    results = load_results()
    
    if not results:
        logger.warning(
            "No experimental results found. Please run experiments first:\n"
            "  1. python experiments/phase2_rag_generation_agora.py\n"
            "  2. python experiments/phase3_mi_analysis.py\n"
            "  3. Complete manual annotations\n"
        )
        logger.info("Creating placeholder figures...")
    
    # Generate figures
    figure1_ecs_vs_pks_scatter(results, figures_dir)
    figure2_roc_curve(results, figures_dir)
    figure3_layer_wise_ecs(results, figures_dir)
    
    # Generate tables
    table1_configuration_comparison(results, figures_dir)
    
    logger.info("\n" + "="*60)
    logger.info("Figure Generation Complete")
    logger.info("="*60)
    logger.info(f"Figures saved to: {figures_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Review generated figures")
    logger.info("2. Update main.tex with actual results (replace TBD)")
    logger.info("3. Compile LaTeX: cd paper && pdflatex main.tex")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

