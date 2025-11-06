"""
Phase 2: RAG Response Generation

This experiment generates RAG responses using Mistral-7B with full attention
and hidden state extraction for subsequent MI analysis.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import yaml
import pandas as pd
from loguru import logger

from src.rag.retriever import SemanticRetriever
from src.rag.generator import ResponseGenerator
from src.rag.pipeline import RAGPipeline


logger.add(
    "logs/phase2_rag_generation_{time}.log",
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
    Execute Phase 2: RAG Response Generation.
    
    Steps:
    1. Load filtered segments from Phase 1
    2. Build vector store with ChromaDB
    3. Load questions (from existing generated_questions.json)
    4. Generate RAG responses with MI data
    5. Export results
    """
    logger.info("="*60)
    logger.info("Phase 2: RAG Response Generation")
    logger.info("="*60)
    
    config = load_config()
    
    data_config = config['data']
    model_config = config['model']
    embedding_config = config['embedding']
    rag_config = config['rag']
    
    logger.info("\nStep 1: Loading filtered segments from Phase 1...")
    processed_path = project_root / data_config['processed_path']
    
    segments_file = processed_path / 'filtered_segments.csv'
    if not segments_file.exists():
        logger.error(f"Filtered segments not found: {segments_file}")
        logger.error("Please run Phase 1 first")
        return 1
    
    segments_df = pd.read_csv(segments_file)
    logger.info(f"Loaded {len(segments_df)} segments")
    
    logger.info("\nStep 2: Building vector store...")
    retriever = SemanticRetriever(
        embedding_model_name=embedding_config['model_name'],
        collection_name=rag_config['vector_store']['collection_name'],
        persist_directory=str(project_root / "data" / "chromadb"),
        similarity_threshold=rag_config['retriever']['similarity_threshold'],
        device="cuda"
    )
    
    retriever.create_collection(overwrite=True)
    
    retriever.index_segments(
        segments_df,
        text_column='Text',
        metadata_columns=['Document ID', 'Segment position']
    )
    
    logger.info(f"Indexed {retriever.get_collection_size()} segments")
    
    logger.info("\nStep 3: Loading questions...")
    
    questions_file = project_root / "outputs/phase1_2/generated_questions.json"
    
    if not questions_file.exists():
        logger.warning(f"Questions file not found: {questions_file}")
        logger.warning("Using placeholder questions for demonstration")
        
        questions_data = [
            {
                'question_id': f'q_{i:04d}',
                'question': f'Sample question {i}',
                'task_type': 'QA'
            }
            for i in range(10)
        ]
    else:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
    
    logger.info(f"Loaded {len(questions_data)} questions")
    
    logger.info("\nStep 4: Initializing RAG generator...")
    generator = ResponseGenerator(
        model_name=model_config['name'],
        load_4bit=model_config['quantization'] == '4bit',
        device_map="auto",
        max_length=model_config['max_length'],
        temperature=model_config['temperature'],
        top_p=model_config['top_p']
    )
    
    logger.info("Model loaded successfully")
    
    logger.info("\nStep 5: Creating RAG pipeline...")
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        top_k=rag_config['retriever']['top_k'],
        max_new_tokens=512,
        collect_mi_data=True
    )
    
    logger.info("\nStep 6: Generating RAG responses...")
    logger.warning("This may take a long time depending on number of questions and GPU speed")
    
    results = pipeline.process_batch(
        questions=[q['question'] for q in questions_data],
        question_ids=[q.get('question_id', f'q_{i}') for i, q in enumerate(questions_data)],
        task_types=[q.get('task_type', 'Unknown') for q in questions_data],
        save_intermediate=True,
        output_path=str(project_root / "outputs/phase2_rag_responses_intermediate.json")
    )
    
    logger.info(f"Generated {len(results)} responses")
    
    stats = pipeline.get_statistics(results)
    logger.info(f"Success rate: {stats['success_rate']*100:.1f}%")
    logger.info(f"Average generation time: {stats['timing']['avg_total']:.2f}s")
    
    logger.info("\nStep 7: Exporting results...")
    output_dir = project_root / "outputs/phase2_rag"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline.save_results(
        results,
        str(output_dir / 'rag_responses.json'),
        include_prompts=False
    )
    
    metadata = {
        'phase': 'phase2_rag_generation',
        'model': model_config['name'],
        'num_questions': len(questions_data),
        'num_responses': len(results),
        'statistics': stats,
        'rag_config': rag_config
    }
    
    with open(output_dir / 'rag_summary_statistics.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    logger.info("\n" + "="*60)
    logger.info("Phase 2 Complete")
    logger.info("="*60)
    logger.info(f"Total responses: {len(results)}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Next step: Phase 3 - MI Analysis")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

