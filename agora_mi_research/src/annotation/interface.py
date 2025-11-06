"""
Annotation interface module for Label Studio integration.

This module provides tools to integrate with Label Studio for manual
hallucination annotation following RAGTruth methodology.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

try:
    from label_studio_sdk import Client, Project
    LABEL_STUDIO_AVAILABLE = True
except ImportError:
    LABEL_STUDIO_AVAILABLE = False
    logger.warning("Label Studio SDK not available. Install with: pip install label-studio-sdk")


HALLUCINATION_CATEGORIES = [
    "Evident Baseless Info",
    "Subtle Conflict",
    "Nuanced Misrepresentation",
    "Reasoning Error"
]

LABEL_STUDIO_CONFIG = """
<View>
  <Header value="Hallucination Annotation - AGORA MI Research"/>
  
  <Text name="question" value="$question"/>
  
  <Header value="Retrieved Context"/>
  <Text name="context" value="$context" style="white-space: pre-wrap;"/>
  
  <Header value="Generated Response (Annotate hallucination spans)"/>
  <Labels name="hallucination" toName="response">
    <Label value="Evident Baseless Info" background="#FF6B6B" description="Clear fabrications without source support"/>
    <Label value="Subtle Conflict" background="#FFA500" description="Minor inconsistencies with retrieved context"/>
    <Label value="Nuanced Misrepresentation" background="#FFD700" description="Subtle distortions of source information"/>
    <Label value="Reasoning Error" background="#9370DB" description="Incorrect logical inferences from context"/>
    <Label value="Correct" background="#90EE90" description="Accurate information from context"/>
  </Labels>
  <Text name="response" value="$response" granularity="word"/>
  
  <Header value="Additional Metadata"/>
  <Choices name="severity" toName="response" choice="single">
    <Choice value="High"/>
    <Choice value="Medium"/>
    <Choice value="Low"/>
    <Choice value="None"/>
  </Choices>
  
  <Choices name="overall_quality" toName="response" choice="single">
    <Choice value="Excellent"/>
    <Choice value="Good"/>
    <Choice value="Fair"/>
    <Choice value="Poor"/>
  </Choices>
  
  <TextArea name="notes" toName="response" placeholder="Additional notes or comments" rows="3"/>
  
  <Header value="MI Metrics (Reference Only)"/>
  <Text name="mi_metrics" value="$mi_metrics"/>
</View>
"""


class AnnotationInterface:
    """
    Interface for Label Studio annotation.
    
    Manages export of RAG responses to Label Studio format and
    import of annotations for analysis.
    """
    
    def __init__(
        self,
        label_studio_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize annotation interface.
        
        Args:
            label_studio_url: URL of Label Studio instance
            api_key: API key for Label Studio (optional)
        """
        self.label_studio_url = label_studio_url or "http://localhost:8080"
        self.api_key = api_key
        
        self.client = None
        if LABEL_STUDIO_AVAILABLE and api_key:
            try:
                self.client = Client(
                    url=self.label_studio_url,
                    api_key=api_key
                )
                logger.info(f"Connected to Label Studio at {self.label_studio_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Label Studio: {e}")
        
        logger.info("AnnotationInterface initialized")
    
    def format_response_for_annotation(
        self,
        rag_result: Dict,
        mi_analysis: Optional[Dict] = None
    ) -> Dict:
        """
        Format RAG response for Label Studio annotation.
        
        Args:
            rag_result: RAG pipeline result dictionary
            mi_analysis: Optional MI analysis results
        
        Returns:
            Dictionary in Label Studio task format
        """
        context_parts = []
        for i, segment in enumerate(rag_result.get('retrieved_segments', []), 1):
            context_parts.append(f"[{i}] {segment['text']}")
        
        context_text = "\n\n".join(context_parts)
        
        mi_metrics_text = "Not available"
        if mi_analysis:
            mi_metrics_text = (
                f"ECS: {mi_analysis.get('overall_ecs', 0):.3f} | "
                f"PKS: {mi_analysis.get('overall_pks', 0):.3f} | "
                f"Copying Heads: {mi_analysis.get('num_copying_heads', 0)}"
            )
        
        task = {
            'data': {
                'question': rag_result.get('question', ''),
                'context': context_text,
                'response': rag_result.get('response', ''),
                'mi_metrics': mi_metrics_text
            },
            'meta': {
                'question_id': rag_result.get('question_id', ''),
                'task_type': rag_result.get('task_type', ''),
                'num_segments': rag_result.get('num_retrieved', 0)
            }
        }
        
        if mi_analysis:
            task['meta']['ecs'] = mi_analysis.get('overall_ecs', 0)
            task['meta']['pks'] = mi_analysis.get('overall_pks', 0)
        
        return task
    
    def export_for_annotation(
        self,
        rag_results: List[Dict],
        mi_analyses: Optional[List[Dict]] = None,
        output_path: str = "annotations/tasks.json",
        sample_size: Optional[int] = None
    ) -> None:
        """
        Export RAG results to Label Studio JSON format.
        
        Args:
            rag_results: List of RAG pipeline results
            mi_analyses: Optional list of MI analyses
            output_path: Path to save annotation tasks
            sample_size: Optional number of responses to sample
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if sample_size and sample_size < len(rag_results):
            import random
            random.seed(42)
            indices = random.sample(range(len(rag_results)), sample_size)
            rag_results = [rag_results[i] for i in indices]
            if mi_analyses:
                mi_analyses = [mi_analyses[i] for i in indices]
        
        tasks = []
        for i, rag_result in enumerate(rag_results):
            mi_analysis = mi_analyses[i] if mi_analyses else None
            
            task = self.format_response_for_annotation(rag_result, mi_analysis)
            tasks.append(task)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(tasks)} tasks to {output_path}")
    
    def create_project(
        self,
        project_name: str = "AGORA Hallucination Annotation",
        description: str = "Annotate hallucinations in RAG responses for AI governance documents"
    ) -> Optional['Project']:
        """
        Create Label Studio project.
        
        Args:
            project_name: Name of the project
            description: Project description
        
        Returns:
            Project instance or None if client not available
        """
        if not self.client:
            logger.error("Label Studio client not available")
            return None
        
        try:
            project = self.client.start_project(
                title=project_name,
                label_config=LABEL_STUDIO_CONFIG,
                description=description
            )
            
            logger.info(f"Created project: {project_name} (ID: {project.id})")
            return project
            
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return None
    
    def import_tasks_to_project(
        self,
        project_id: int,
        tasks: List[Dict]
    ) -> bool:
        """
        Import tasks to Label Studio project.
        
        Args:
            project_id: Label Studio project ID
            tasks: List of task dictionaries
        
        Returns:
            True if successful
        """
        if not self.client:
            logger.error("Label Studio client not available")
            return False
        
        try:
            project = self.client.get_project(project_id)
            project.import_tasks(tasks)
            
            logger.info(f"Imported {len(tasks)} tasks to project {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import tasks: {e}")
            return False
    
    def load_annotations(
        self,
        annotations_path: str
    ) -> List[Dict]:
        """
        Load annotations from Label Studio export.
        
        Args:
            annotations_path: Path to annotations JSON file
        
        Returns:
            List of annotation dictionaries
        """
        annotations_file = Path(annotations_path)
        
        if not annotations_file.exists():
            logger.error(f"Annotations file not found: {annotations_path}")
            return []
        
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        logger.info(f"Loaded {len(annotations)} annotations from {annotations_path}")
        
        return annotations
    
    def parse_annotation(
        self,
        annotation: Dict
    ) -> Dict:
        """
        Parse Label Studio annotation to structured format.
        
        Args:
            annotation: Raw annotation dictionary from Label Studio
        
        Returns:
            Parsed annotation dictionary
        """
        parsed = {
            'question_id': annotation.get('data', {}).get('meta', {}).get('question_id', ''),
            'question': annotation.get('data', {}).get('question', ''),
            'response': annotation.get('data', {}).get('response', ''),
            'hallucination_spans': [],
            'severity': None,
            'overall_quality': None,
            'notes': '',
            'annotator_id': None,
            'annotation_time': None
        }
        
        if 'annotations' in annotation and annotation['annotations']:
            result = annotation['annotations'][0].get('result', [])
            
            for item in result:
                if item.get('from_name') == 'hallucination' and item.get('type') == 'labels':
                    span_info = item.get('value', {})
                    parsed['hallucination_spans'].append({
                        'start': span_info.get('start', 0),
                        'end': span_info.get('end', 0),
                        'text': span_info.get('text', ''),
                        'labels': span_info.get('labels', [])
                    })
                
                elif item.get('from_name') == 'severity':
                    parsed['severity'] = item.get('value', {}).get('choices', [None])[0]
                
                elif item.get('from_name') == 'overall_quality':
                    parsed['overall_quality'] = item.get('value', {}).get('choices', [None])[0]
                
                elif item.get('from_name') == 'notes':
                    parsed['notes'] = item.get('value', {}).get('text', [''])[0]
            
            if 'completed_by' in annotation['annotations'][0]:
                parsed['annotator_id'] = annotation['annotations'][0]['completed_by']
            
            if 'created_at' in annotation['annotations'][0]:
                parsed['annotation_time'] = annotation['annotations'][0]['created_at']
        
        return parsed
    
    def parse_all_annotations(
        self,
        annotations: List[Dict]
    ) -> List[Dict]:
        """
        Parse all annotations from Label Studio export.
        
        Args:
            annotations: List of raw annotations
        
        Returns:
            List of parsed annotations
        """
        parsed_annotations = []
        
        for annotation in annotations:
            try:
                parsed = self.parse_annotation(annotation)
                parsed_annotations.append(parsed)
            except Exception as e:
                logger.error(f"Failed to parse annotation: {e}")
                continue
        
        logger.info(f"Parsed {len(parsed_annotations)} annotations")
        
        return parsed_annotations
    
    def compute_annotation_statistics(
        self,
        annotations: List[Dict]
    ) -> Dict:
        """
        Compute statistics on annotations.
        
        Args:
            annotations: List of parsed annotations
        
        Returns:
            Dictionary with annotation statistics
        """
        total = len(annotations)
        
        hallucinated_count = sum(
            1 for ann in annotations
            if len(ann['hallucination_spans']) > 0
        )
        
        hallucination_types = {}
        for ann in annotations:
            for span in ann['hallucination_spans']:
                for label in span['labels']:
                    if label != "Correct":
                        hallucination_types[label] = hallucination_types.get(label, 0) + 1
        
        severity_dist = {}
        for ann in annotations:
            if ann['severity']:
                severity_dist[ann['severity']] = severity_dist.get(ann['severity'], 0) + 1
        
        quality_dist = {}
        for ann in annotations:
            if ann['overall_quality']:
                quality_dist[ann['overall_quality']] = quality_dist.get(ann['overall_quality'], 0) + 1
        
        stats = {
            'total_annotations': total,
            'hallucinated_responses': hallucinated_count,
            'hallucination_rate': hallucinated_count / total if total > 0 else 0,
            'hallucination_type_distribution': hallucination_types,
            'severity_distribution': severity_dist,
            'quality_distribution': quality_dist
        }
        
        logger.info(f"Annotation statistics: {stats}")
        
        return stats


def main():
    """
    Example usage of AnnotationInterface.
    """
    interface = AnnotationInterface()
    
    sample_rag_result = {
        'question_id': 'q_001',
        'question': 'What are the requirements for high-risk AI systems?',
        'response': 'High-risk AI systems must undergo conformity assessment and implement data governance practices.',
        'task_type': 'QA',
        'retrieved_segments': [
            {
                'text': 'The EU AI Act requires high-risk AI systems to undergo conformity assessment.',
                'similarity': 0.95
            }
        ],
        'num_retrieved': 1
    }
    
    sample_mi_analysis = {
        'overall_ecs': 0.456,
        'overall_pks': 0.623,
        'num_copying_heads': 5
    }
    
    task = interface.format_response_for_annotation(
        sample_rag_result,
        sample_mi_analysis
    )
    
    print("Formatted annotation task:")
    print(json.dumps(task, indent=2))
    
    print("\nLabel Studio configuration:")
    print(LABEL_STUDIO_CONFIG)


if __name__ == "__main__":
    main()

