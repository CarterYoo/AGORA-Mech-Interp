"""
Phase 2: RAG Response Generation (AGORA Q&A Integration)

This experiment generates RAG responses using RAGatouille ColBERT retriever
matching the AGORA Q&A system architecture.

Reference: https://github.com/rrittner1/agora
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import yaml
import pandas as pd
from loguru import logger

from src.data.agora_loader import AGORADataLoaderOfficial
from src.rag.ragatouille_retriever import RAGatouilleRetriever, RAGATOUILLE_AVAILABLE
from src.rag.generator import ResponseGenerator
from src.rag.pipeline import RAGPipeline


logger.add(
    "logs/phase2_rag_generation_agora_{time}.log",
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


def main():
    """
    Execute Phase 2: RAG Response Generation with AGORA Q&A architecture.
    
    Steps:
    1. Load AGORA dataset in official format
    2. Build RAGatouille ColBERT index
    3. Load questions
    4. Generate RAG responses with MI data
    5. Export results
    """
    logger.info("="*60)
    logger.info("Phase 2: RAG Generation (AGORA Q&A Integration)")
    logger.info("="*60)
    
    if not RAGATOUILLE_AVAILABLE:
        logger.error("RAGatouille not available. Install with: pip install ragatouille")
        return 1
    
    config = load_config()
    model_config = config['model']
    rag_config = config['rag']
    
    logger.info("\nStep 1: Loading AGORA dataset (official format)...")
    
    agora_data_path = "C:/Users/23012/Downloads/agora-master/agora-master/data/agora"
    
    loader = AGORADataLoaderOfficial(data_path=agora_data_path)
    
    try:
        documents_df, segments_df = loader.load_all(validate=True)
        stats = loader.get_statistics(documents_df, segments_df)
        
        logger.info(f"AGORA dataset loaded:")
        logger.info(f"  Documents: {stats['num_documents']}")
        logger.info(f"  Segments: {stats['num_segments']}")
        logger.info(f"  Authorities: {len(stats['authority_distribution'])}")
        
    except Exception as e:
        logger.error(f"Failed to load AGORA dataset: {e}")
        logger.error(
            "Please ensure AGORA dataset is available at:\n"
            "  C:/Users/23012/Downloads/agora-master/agora-master/data/agora/"
        )
        return 1
    
    logger.info("\nStep 2: Building RAGatouille ColBERT index...")
    logger.warning("This may take 10-30 minutes for full dataset")
    
    retriever = RAGatouilleRetriever(
        model_name="colbert-ir/colbertv2.0",
        index_name="agora_mi_research_index",
        index_path="./.ragatouille/colbert/indexes/agora_mi_research_index",
        similarity_threshold=rag_config['retriever']['similarity_threshold']
    )
    
    if not Path(retriever.index_path).exists():
        logger.info("Index not found, building new index...")
        retriever.index_segments(
            segments_df,
            documents_df=documents_df,
            overwrite=False
        )
    else:
        logger.info("Loading existing index...")
        retriever.retrieve("test", top_k=1)
    
    logger.info(f"Index ready: {retriever.get_collection_size()} chunks indexed")
    
    logger.info("\nStep 3: Loading questions...")
    
    questions_file = project_root / "outputs/phase1_2/generated_questions.json"
    
    if not questions_file.exists():
        logger.warning(
            f"Questions file not found: {questions_file}\n"
            "Using AGORA official questions..."
        )
        
        agora_questions_file = Path(agora_data_path) / "generated_questions_compl.json"
        if agora_questions_file.exists():
            with open(agora_questions_file, 'r', encoding='utf-8') as f:
                agora_questions = json.load(f)
            
            questions_data = [
                {
                    'question_id': f'agora_q_{i:04d}',
                    'question': q.get('question', q.get('text', '')),
                    'task_type': 'QA'
                }
                for i, q in enumerate(agora_questions[:50])
            ]
            logger.info(f"Loaded {len(questions_data)} questions from AGORA official set")
        else:
            logger.error("No questions available")
            return 1
    else:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        logger.info(f"Loaded {len(questions_data)} questions")
    
    logger.info("\nStep 4: Initializing generator...")
    
    generator = ResponseGenerator(
        model_name=model_config['name'],
        load_4bit=model_config['quantization'] == '4bit',
        device_map="auto",
        max_length=model_config['max_length'],
        temperature=model_config['temperature'],
        top_p=model_config['top_p']
    )
    
    logger.info("Model loaded successfully")
    
    logger.info("\nStep 5: Creating RAG pipeline with AGORA retriever...")
    
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        top_k=rag_config['retriever']['top_k'],
        max_new_tokens=512,
        collect_mi_data=True
    )
    
    logger.info("\nStep 6: Generating RAG responses...")
    logger.warning(
        f"Processing {len(questions_data)} questions. "
        f"This may take several hours."
    )
    
    results = pipeline.process_batch(
        questions=[q['question'] for q in questions_data],
        question_ids=[q.get('question_id', f'q_{i}') for i, q in enumerate(questions_data)],
        task_types=[q.get('task_type', 'QA') for q in questions_data],
        save_intermediate=True,
        output_path=str(project_root / "outputs/phase2_agora_intermediate.json")
    )
    
    logger.info(f"Generated {len(results)} responses")
    
    stats = pipeline.get_statistics(results)
    logger.info(f"Success rate: {stats['success_rate']*100:.1f}%")
    logger.info(f"Average generation time: {stats['timing']['avg_total']:.2f}s")
    
    logger.info("\nStep 7: Exporting results...")
    
    output_dir = project_root / "outputs/phase2_agora_rag"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline.save_results(
        results,
        str(output_dir / 'rag_responses_agora.json'),
        include_prompts=False
    )
    
    metadata = {
        'phase': 'phase2_rag_generation_agora',
        'retriever': 'RAGatouille_ColBERT',
        'model': model_config['name'],
        'num_questions': len(questions_data),
        'num_responses': len(results),
        'statistics': stats,
        'agora_integration': True,
        'colbert_model': 'colbert-ir/colbertv2.0'
    }
    
    with open(output_dir / 'rag_summary_statistics_agora.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    logger.info("\n" + "="*60)
    logger.info("Phase 2 (AGORA Integration) Complete")
    logger.info("="*60)
    logger.info(f"Total responses: {len(results)}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"\nUsing AGORA Q&A architecture:")
    logger.info(f"  Retriever: RAGatouille ColBERT")
    logger.info(f"  Generator: {model_config['name']}")
    logger.info(f"\nNext step: Phase 3 - MI Analysis")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

