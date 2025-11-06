"""
Comparative Experiment: Vanilla vs ColBERT vs DPO vs ColBERT+DPO

This experiment compares four configurations to evaluate the impact of
ColBERT retrieval and DPO fine-tuning on:
1. Retrieval quality
2. Generation quality  
3. Hallucination rates
4. Mechanistic interpretability patterns (ECS, PKS)

Based on AGORA Q&A system architecture: https://github.com/rrittner1/agora
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import yaml
import pandas as pd
from loguru import logger
from typing import Dict, List
import time

from src.rag.retriever import SemanticRetriever
from src.rag.colbert_retriever import ColBERTRetriever, COLBERT_AVAILABLE
from src.rag.generator import ResponseGenerator
from src.rag.dpo_generator import DPOResponseGenerator
from src.rag.pipeline import RAGPipeline

logger.add(
    "logs/compare_configurations_{time}.log",
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


class ConfigurationComparison:
    """
    Manager for comparing different RAG configurations.
    
    Configurations:
    1. Baseline: Sentence-Transformers + Vanilla Mistral
    2. ColBERT Only: ColBERT + Vanilla Mistral
    3. DPO Only: Sentence-Transformers + DPO Model
    4. Full Integration: ColBERT + DPO Model
    """
    
    def __init__(self, config: Dict):
        """
        Initialize configuration comparison.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.segments_df = None
        self.questions = []
        self.results = {
            'baseline': [],
            'colbert_only': [],
            'dpo_only': [],
            'full_integration': []
        }
        
        logger.info("Initialized ConfigurationComparison")
    
    def load_data(self) -> None:
        """Load segments and questions for comparison."""
        logger.info("Loading data...")
        
        processed_path = project_root / self.config['data']['processed_path']
        segments_file = processed_path / 'filtered_segments.csv'
        
        if not segments_file.exists():
            raise FileNotFoundError(
                f"Segments not found: {segments_file}\n"
                "Please run Phase 1 first."
            )
        
        self.segments_df = pd.read_csv(segments_file)
        logger.info(f"Loaded {len(self.segments_df)} segments")
        
        questions_file = project_root / "outputs/phase1_2/generated_questions.json"
        
        if questions_file.exists():
            with open(questions_file, 'r', encoding='utf-8') as f:
                self.questions = json.load(f)
            logger.info(f"Loaded {len(self.questions)} questions")
        else:
            logger.warning("Questions file not found, using sample questions")
            self.questions = [
                {
                    'question_id': f'q_sample_{i}',
                    'question': f'Sample question {i}',
                    'task_type': 'QA'
                }
                for i in range(10)
            ]
    
    def setup_baseline(self) -> RAGPipeline:
        """Setup baseline configuration: Sentence-Transformers + Vanilla."""
        logger.info("Setting up BASELINE configuration...")
        
        retriever = SemanticRetriever(
            embedding_model_name=self.config['embedding']['model_name'],
            collection_name="baseline_collection",
            persist_directory=str(project_root / "data/chromadb_baseline"),
            device="cuda"
        )
        
        retriever.create_collection(overwrite=False)
        
        if retriever.get_collection_size() == 0:
            retriever.index_segments(
                self.segments_df,
                metadata_columns=['Document ID', 'Segment position']
            )
        
        generator = ResponseGenerator(
            model_name=self.config['model']['name'],
            load_4bit=self.config['model']['quantization'] == '4bit',
            temperature=self.config['model']['temperature']
        )
        
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            top_k=self.config['rag']['retriever']['top_k']
        )
        
        logger.info("BASELINE setup complete")
        return pipeline
    
    def setup_colbert_only(self) -> RAGPipeline:
        """Setup ColBERT only configuration: ColBERT + Vanilla."""
        if not COLBERT_AVAILABLE:
            logger.warning("ColBERT not available, skipping ColBERT configurations")
            return None
        
        logger.info("Setting up COLBERT ONLY configuration...")
        
        colbert_config = self.config['rag']['retriever']['colbert']
        
        retriever = ColBERTRetriever(
            checkpoint=colbert_config['checkpoint'],
            index_name="colbert_only_index",
            index_path=str(project_root / colbert_config['index_path']),
            nbits=colbert_config['nbits'],
            doc_maxlen=colbert_config['doc_maxlen'],
            query_maxlen=colbert_config['query_maxlen']
        )
        
        try:
            retriever._load_index()
        except:
            logger.info("Building new ColBERT index...")
            retriever.index_segments(
                self.segments_df,
                metadata_columns=['Document ID', 'Segment position'],
                overwrite=False
            )
        
        generator = ResponseGenerator(
            model_name=self.config['model']['name'],
            load_4bit=self.config['model']['quantization'] == '4bit',
            temperature=self.config['model']['temperature']
        )
        
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            top_k=self.config['rag']['retriever']['top_k']
        )
        
        logger.info("COLBERT ONLY setup complete")
        return pipeline
    
    def setup_dpo_only(self) -> RAGPipeline:
        """Setup DPO only configuration: Sentence-Transformers + DPO."""
        logger.info("Setting up DPO ONLY configuration...")
        
        retriever = SemanticRetriever(
            embedding_model_name=self.config['embedding']['model_name'],
            collection_name="dpo_collection",
            persist_directory=str(project_root / "data/chromadb_dpo"),
            device="cuda"
        )
        
        retriever.create_collection(overwrite=False)
        
        if retriever.get_collection_size() == 0:
            retriever.index_segments(
                self.segments_df,
                metadata_columns=['Document ID', 'Segment position']
            )
        
        dpo_config = self.config['model']['dpo']
        
        generator = DPOResponseGenerator(
            model_name=dpo_config['base_model'],
            adapter_path=dpo_config['adapter_path'],
            beta=dpo_config['beta'],
            load_4bit=self.config['model']['quantization'] == '4bit',
            temperature=self.config['model']['temperature']
        )
        
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            top_k=self.config['rag']['retriever']['top_k']
        )
        
        logger.info("DPO ONLY setup complete")
        return pipeline
    
    def setup_full_integration(self) -> RAGPipeline:
        """Setup full integration: ColBERT + DPO."""
        if not COLBERT_AVAILABLE:
            logger.warning("ColBERT not available, skipping full integration")
            return None
        
        logger.info("Setting up FULL INTEGRATION configuration...")
        
        colbert_config = self.config['rag']['retriever']['colbert']
        
        retriever = ColBERTRetriever(
            checkpoint=colbert_config['checkpoint'],
            index_name="full_integration_index",
            index_path=str(project_root / colbert_config['index_path']),
            nbits=colbert_config['nbits'],
            doc_maxlen=colbert_config['doc_maxlen'],
            query_maxlen=colbert_config['query_maxlen']
        )
        
        try:
            retriever._load_index()
        except:
            logger.info("Building new ColBERT index...")
            retriever.index_segments(
                self.segments_df,
                metadata_columns=['Document ID', 'Segment position'],
                overwrite=False
            )
        
        dpo_config = self.config['model']['dpo']
        
        generator = DPOResponseGenerator(
            model_name=dpo_config['base_model'],
            adapter_path=dpo_config['adapter_path'],
            beta=dpo_config['beta'],
            load_4bit=self.config['model']['quantization'] == '4bit',
            temperature=self.config['model']['temperature']
        )
        
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            top_k=self.config['rag']['retriever']['top_k']
        )
        
        logger.info("FULL INTEGRATION setup complete")
        return pipeline
    
    def run_comparison(self, sample_size: int = 20) -> None:
        """
        Run comparison across all configurations.
        
        Args:
            sample_size: Number of questions to test
        """
        self.load_data()
        
        sample_questions = self.questions[:sample_size]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running comparison with {len(sample_questions)} questions")
        logger.info(f"{'='*60}\n")
        
        configs_to_test = {
            'baseline': self.setup_baseline,
            'colbert_only': self.setup_colbert_only,
            'dpo_only': self.setup_dpo_only,
            'full_integration': self.setup_full_integration
        }
        
        for config_name, setup_func in configs_to_test.items():
            logger.info(f"\n--- Testing {config_name.upper()} ---")
            
            start_time = time.time()
            
            try:
                pipeline = setup_func()
                
                if pipeline is None:
                    logger.warning(f"Skipping {config_name} (not available)")
                    continue
                
                results = pipeline.process_batch(
                    questions=[q['question'] for q in sample_questions],
                    question_ids=[q['question_id'] for q in sample_questions],
                    task_types=[q.get('task_type', 'Unknown') for q in sample_questions]
                )
                
                self.results[config_name] = results
                
                elapsed = time.time() - start_time
                stats = pipeline.get_statistics(results)
                
                logger.info(f"{config_name.upper()} Results:")
                logger.info(f"  Success rate: {stats['success_rate']*100:.1f}%")
                logger.info(f"  Avg time: {stats['timing']['avg_total']:.2f}s")
                logger.info(f"  Total time: {elapsed/60:.1f}min")
                
            except Exception as e:
                logger.error(f"Error in {config_name}: {e}")
                continue
        
        self.save_results()
    
    def save_results(self) -> None:
        """Save comparison results to disk."""
        output_dir = project_root / "outputs/comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for config_name, results in self.results.items():
            if results:
                output_file = output_dir / f"{config_name}_results.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved {config_name} results to {output_file}")
        
        summary = {
            'configurations_tested': list(self.results.keys()),
            'num_questions': len(self.questions),
            'results_saved': True
        }
        
        with open(output_dir / 'comparison_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nAll results saved to {output_dir}")


def main():
    """
    Execute configuration comparison experiment.
    """
    logger.info("="*60)
    logger.info("Configuration Comparison Experiment")
    logger.info("Comparing: Baseline, ColBERT, DPO, ColBERT+DPO")
    logger.info("="*60)
    
    config = load_config()
    
    comparison = ConfigurationComparison(config)
    
    logger.warning(
        "\nThis experiment will run 4 configurations and may take several hours.\n"
        "Each configuration needs to:\n"
        "  1. Build/load index\n"
        "  2. Load model\n"
        "  3. Generate responses\n"
        "  4. Collect MI data\n"
    )
    
    sample_size = 20
    logger.info(f"\nTesting with {sample_size} questions per configuration\n")
    
    comparison.run_comparison(sample_size=sample_size)
    
    logger.info("\n" + "="*60)
    logger.info("Comparison Experiment Complete")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Review results in outputs/comparison/")
    logger.info("2. Run MI analysis on each configuration")
    logger.info("3. Compare ECS, PKS, and hallucination rates")
    logger.info("4. Generate comparative visualizations")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

