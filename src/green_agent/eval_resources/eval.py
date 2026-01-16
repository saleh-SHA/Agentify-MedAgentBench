import sys
import logging
from pathlib import Path

if __package__:
    from .utils import *
else:
    # Allow running as a script by adding the eval_scripts directory to sys.path
    sys.path.append(str(Path(__file__).resolve().parent))
    from utils import *

from src.typings import TaskOutput, SampleStatus, List, Dict, Any
import src.green_agent.eval_resources.refsol as refsol

logger = logging.getLogger(__name__)


def eval(case_data, results, fhir_api_base):
    task_id = case_data['id'].split('_')[0]
    grader_func = getattr(refsol, task_id)
    try:
        if grader_func(case_data, results, fhir_api_base) is True:
            return True
        else: return False
    except Exception as e:
        print(e)
        return False


def calculate_overall_metrics(
    results: List[TaskOutput],
    task_data_list: List[Dict[str, Any]],
    fhir_api_base: str
) -> Dict[str, Any]:
    """Calculate overall evaluation metrics from a list of task results.
    
    Args:
        results: List of TaskOutput objects
        task_data_list: List of task data dictionaries (for evaluation)
        fhir_api_base: Base URL for FHIR API
        
    Returns:
        Dictionary containing overall metrics including:
        - total: Total number of results
        - validation: Status distribution and history length statistics
        - custom: Task-specific metrics (success rate, raw results)
    """
    total = len(results)
    
    # Calculate status distribution
    status_counts: Dict[str, int] = {}
    for result in results:
        status = result.status or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Convert to percentages
    status_distribution = {k: v / total for k, v in status_counts.items()}
    
    # Calculate history length statistics
    history_lengths = [
        len(result.history) if result.history else 0 
        for result in results
    ]
    
    validation = {
        **status_distribution,
        "average_history_length": sum(history_lengths) / total if total > 0 else 0,
        "max_history_length": max(history_lengths) if history_lengths else 0,
        "min_history_length": min(history_lengths) if history_lengths else 0,
    }
    
    # Calculate task-specific metrics (success rate)
    correct_count = 0
    evaluated_results = []
    
    for i, result in enumerate(results):
        if result.result is not None and i < len(task_data_list):
            task_data = task_data_list[i]
            try:
                is_correct = eval(task_data, result, fhir_api_base) is True
            except Exception as e:
                logger.error(f"Evaluation error for task {result.index}: {e}")
                is_correct = False
            
            if is_correct:
                correct_count += 1
                result.status = (result.status or "") + " Correct"
            else:
                result.status = (result.status or "") + " Incorrect"
        
        evaluated_results.append(result.model_dump())
    
    success_rate = correct_count / total if total > 0 else 0
    
    custom = {
        "success_rate": success_rate,
        "correct_count": correct_count,
        "total_evaluated": total,
        "raw_results": evaluated_results,
    }
    
    return {
        "total": total,
        "validation": validation,
        "custom": custom,
    }
