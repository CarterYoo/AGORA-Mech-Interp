"""
Complete RAG pipeline module.

This module orchestrates the complete RAG workflow by integrating
semantic retrieval and response generation components.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger
import pandas as pd
from datetime import datetime

from .retriever import SemanticRetriever
from .generator import ResponseGenerator


class RAGPipeline:
    """
    Complete RAG pipeline integrating retrieval and generation.
    
    Orchestrates the full workflow from question to response with
    comprehensive metadata collection for analysis.
    """
    
    def __init__(
        self,
        retriever: Optional[SemanticRetriever] = None,
        generator: Optional[ResponseGenerator] = None,
        top_k: int = 5,
        max_new_tokens: int = 512,
        collect_mi_data: bool = True
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: SemanticRetriever instance (creates default if None)
            generator: ResponseGenerator instance (creates default if None)
            top_k: Number of segments to retrieve
            max_new_tokens: Maximum tokens to generate
            collect_mi_data: Whether to collect MI analysis data
        """
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.collect_mi_data = collect_mi_data
        
        if retriever is None:
            logger.info("Creating default SemanticRetriever")
            self.retriever = SemanticRetriever()
        else:
            self.retriever = retriever
        
        if generator is None:
            logger.info("Creating default ResponseGenerator")
            self.generator = ResponseGenerator()
        else:
            self.generator = generator
        
        logger.info(
            f"RAGPipeline initialized: top_k={top_k}, "
            f"max_new_tokens={max_new_tokens}, "
            f"collect_mi_data={collect_mi_data}"
        )
    
    def process_single(
        self,
        question: str,
        question_id: Optional[str] = None,
        task_type: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Process a single question through the RAG pipeline.
        
        Args:
            question: Question text
            question_id: Optional question identifier
            task_type: Optional task type (QA, Data2txt, Summarization)
            metadata: Optional additional metadata
        
        Returns:
            Dictionary containing complete RAG results
        """
        start_time = datetime.now()
        
        if question_id is None:
            question_id = f"q_{start_time.timestamp()}"
        
        logger.info(f"Processing question: {question_id}")
        
        try:
            retrieved_segments = self.retriever.retrieve(
                query=question,
                top_k=self.top_k
            )
            
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            if not retrieved_segments:
                logger.warning(f"No segments retrieved for question: {question_id}")
                return {
                    'question_id': question_id,
                    'question': question,
                    'task_type': task_type,
                    'success': False,
                    'error': 'No segments retrieved',
                    'retrieval_time': retrieval_time
                }
            
            generation_start = datetime.now()
            
            generation_result = self.generator.generate(
                question=question,
                retrieved_segments=retrieved_segments,
                max_new_tokens=self.max_new_tokens,
                return_attention=self.collect_mi_data,
                return_hidden_states=self.collect_mi_data
            )
            
            generation_time = (datetime.now() - generation_start).total_seconds()
            total_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'question_id': question_id,
                'question': question,
                'response': generation_result['response'],
                'task_type': task_type,
                'success': True,
                'retrieved_segments': [
                    {
                        'segment_id': seg['segment_id'],
                        'text': seg['text'],
                        'similarity': seg['similarity'],
                        'metadata': seg.get('metadata', {})
                    }
                    for seg in retrieved_segments
                ],
                'num_retrieved': len(retrieved_segments),
                'retrieval_scores': [seg['similarity'] for seg in retrieved_segments],
                'context_boundaries': {
                    'start': generation_result.get('context_start_pos'),
                    'end': generation_result.get('context_end_pos')
                },
                'token_counts': {
                    'input': generation_result['input_length'],
                    'output': generation_result['output_length'],
                    'context': generation_result['context_token_count']
                },
                'timing': {
                    'retrieval': retrieval_time,
                    'generation': generation_time,
                    'total': total_time
                },
                'mi_data_available': {
                    'attention': generation_result.get('has_attention', False),
                    'hidden_states': generation_result.get('has_hidden_states', False)
                },
                'timestamp': start_time.isoformat()
            }
            
            if metadata:
                result['metadata'] = metadata
            
            logger.info(
                f"Completed {question_id}: "
                f"{result['token_counts']['output']} tokens in {total_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            error_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error processing {question_id}: {str(e)}")
            
            return {
                'question_id': question_id,
                'question': question,
                'task_type': task_type,
                'success': False,
                'error': str(e),
                'timing': {'total': error_time},
                'timestamp': start_time.isoformat()
            }
    
    def process_batch(
        self,
        questions: List[str],
        question_ids: Optional[List[str]] = None,
        task_types: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
        save_intermediate: bool = False,
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Process multiple questions through the RAG pipeline.
        
        Args:
            questions: List of question texts
            question_ids: Optional list of question identifiers
            task_types: Optional list of task types
            metadatas: Optional list of metadata dictionaries
            save_intermediate: Whether to save after each question
            output_path: Path to save intermediate results
        
        Returns:
            List of result dictionaries
        """
        n_questions = len(questions)
        
        if question_ids is None:
            question_ids = [f"q_{i:04d}" for i in range(n_questions)]
        
        if task_types is None:
            task_types = [None] * n_questions
        
        if metadatas is None:
            metadatas = [None] * n_questions
        
        if len(question_ids) != n_questions:
            raise ValueError(f"question_ids length mismatch: {len(question_ids)} != {n_questions}")
        
        if len(task_types) != n_questions:
            raise ValueError(f"task_types length mismatch: {len(task_types)} != {n_questions}")
        
        if len(metadatas) != n_questions:
            raise ValueError(f"metadatas length mismatch: {len(metadatas)} != {n_questions}")
        
        logger.info(f"Processing batch: {n_questions} questions")
        
        results = []
        successful = 0
        
        for i, (question, qid, task_type, metadata) in enumerate(
            zip(questions, question_ids, task_types, metadatas), 1
        ):
            result = self.process_single(
                question=question,
                question_id=qid,
                task_type=task_type,
                metadata=metadata
            )
            
            results.append(result)
            
            if result['success']:
                successful += 1
            
            if save_intermediate and output_path:
                self._save_intermediate(results, output_path)
            
            if i % 10 == 0:
                logger.info(
                    f"Progress: {i}/{n_questions} "
                    f"({successful}/{i} successful, {successful/i*100:.1f}%)"
                )
        
        logger.info(
            f"Batch processing complete: {successful}/{n_questions} successful "
            f"({successful/n_questions*100:.1f}%)"
        )
        
        return results
    
    def process_from_dataframe(
        self,
        questions_df: pd.DataFrame,
        question_column: str = 'question',
        id_column: Optional[str] = 'question_id',
        task_type_column: Optional[str] = 'task_type',
        metadata_columns: Optional[List[str]] = None,
        save_intermediate: bool = False,
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Process questions from a pandas DataFrame.
        
        Args:
            questions_df: DataFrame containing questions
            question_column: Column name for questions
            id_column: Optional column name for question IDs
            task_type_column: Optional column name for task types
            metadata_columns: Optional list of columns to include as metadata
            save_intermediate: Whether to save after each question
            output_path: Path to save intermediate results
        
        Returns:
            List of result dictionaries
        """
        if question_column not in questions_df.columns:
            raise ValueError(f"Question column '{question_column}' not found")
        
        questions = questions_df[question_column].tolist()
        
        question_ids = None
        if id_column and id_column in questions_df.columns:
            question_ids = questions_df[id_column].tolist()
        
        task_types = None
        if task_type_column and task_type_column in questions_df.columns:
            task_types = questions_df[task_type_column].tolist()
        
        metadatas = None
        if metadata_columns:
            metadatas = []
            for _, row in questions_df.iterrows():
                metadata = {}
                for col in metadata_columns:
                    if col in questions_df.columns:
                        metadata[col] = row[col]
                metadatas.append(metadata)
        
        return self.process_batch(
            questions=questions,
            question_ids=question_ids,
            task_types=task_types,
            metadatas=metadatas,
            save_intermediate=save_intermediate,
            output_path=output_path
        )
    
    def _save_intermediate(
        self,
        results: List[Dict],
        output_path: str
    ) -> None:
        """
        Save intermediate results to JSON file.
        
        Args:
            results: List of result dictionaries
            output_path: Path to save results
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved intermediate results to {output_path}")
    
    def save_results(
        self,
        results: List[Dict],
        output_path: str,
        include_prompts: bool = False
    ) -> None:
        """
        Save pipeline results to JSON file.
        
        Args:
            results: List of result dictionaries
            output_path: Path to save results
            include_prompts: Whether to include full prompts
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not include_prompts:
            results_to_save = []
            for result in results:
                result_copy = result.copy()
                if 'prompt' in result_copy:
                    del result_copy['prompt']
                if 'full_text' in result_copy:
                    del result_copy['full_text']
                results_to_save.append(result_copy)
        else:
            results_to_save = results
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} results to {output_path}")
    
    def get_statistics(
        self,
        results: List[Dict]
    ) -> Dict:
        """
        Compute statistics for pipeline results.
        
        Args:
            results: List of result dictionaries
        
        Returns:
            Dictionary with statistics
        """
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', True)]
        
        stats = {
            'total_questions': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) if results else 0
        }
        
        if successful:
            retrieval_times = [r['timing']['retrieval'] for r in successful if 'timing' in r]
            generation_times = [r['timing']['generation'] for r in successful if 'timing' in r]
            total_times = [r['timing']['total'] for r in successful if 'timing' in r]
            
            output_lengths = [r['token_counts']['output'] for r in successful if 'token_counts' in r]
            
            stats.update({
                'timing': {
                    'avg_retrieval': sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
                    'avg_generation': sum(generation_times) / len(generation_times) if generation_times else 0,
                    'avg_total': sum(total_times) / len(total_times) if total_times else 0
                },
                'tokens': {
                    'avg_output_length': sum(output_lengths) / len(output_lengths) if output_lengths else 0,
                    'min_output_length': min(output_lengths) if output_lengths else 0,
                    'max_output_length': max(output_lengths) if output_lengths else 0
                }
            })
            
            if any('task_type' in r for r in successful):
                task_types = [r['task_type'] for r in successful if 'task_type' in r and r['task_type']]
                task_counts = {}
                for task_type in task_types:
                    task_counts[task_type] = task_counts.get(task_type, 0) + 1
                stats['task_distribution'] = task_counts
        
        logger.info(f"Pipeline statistics: {stats}")
        return stats


def main():
    """
    Example usage of RAGPipeline.
    """
    logger.info("Initializing RAG Pipeline...")
    
    retriever = SemanticRetriever(device="cpu")
    
    sample_segments = pd.DataFrame({
        'Document ID': [1, 1, 2, 2],
        'Segment position': [1, 2, 1, 2],
        'Text': [
            "The EU AI Act establishes requirements for high-risk AI systems.",
            "Providers must implement data governance practices.",
            "The US Executive Order promotes safe AI development.",
            "Federal agencies must establish AI governance structures."
        ]
    })
    
    retriever.create_collection(overwrite=True)
    retriever.index_segments(sample_segments, metadata_columns=['Document ID'])
    
    generator = ResponseGenerator(load_4bit=False, temperature=0.7)
    
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        top_k=2,
        max_new_tokens=100,
        collect_mi_data=True
    )
    
    questions = [
        "What are the requirements for AI systems?",
        "How should governance be implemented?"
    ]
    
    logger.info("Processing questions...")
    results = pipeline.process_batch(
        questions=questions,
        task_types=['QA', 'QA']
    )
    
    for result in results:
        print(f"\nQuestion: {result['question']}")
        if result['success']:
            print(f"Response: {result['response']}")
            print(f"Retrieved: {result['num_retrieved']} segments")
            print(f"Time: {result['timing']['total']:.2f}s")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    stats = pipeline.get_statistics(results)
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

