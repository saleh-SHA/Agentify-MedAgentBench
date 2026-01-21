"""MedAgentBench evaluator (green agent) - orchestrates and evaluates medical AI agents."""

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medagentbench_evaluator")

# Output directory for runs.jsonl logging
OUTPUT_DIR = os.environ.get("MEDAGENT_OUTPUT_DIR", "outputs/medagentbench")


# ============================================================================
# Request/Response Models
# ============================================================================

class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class TaskResult(BaseModel):
    task_id: str
    correct: bool
    status: str
    agent_answer: str | None
    expected_answer: list | None
    time_used: float
    rounds: int


class EvalResults(BaseModel):
    domain: str
    total_tasks: int
    correct_count: int
    pass_rate: float
    task_results: dict[str, TaskResult]
    time_used: float


# ============================================================================
# FHIR Utilities
# ============================================================================

def send_get_request(url, params=None, headers=None):
    """Send GET request to FHIR server."""
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        # Always return text - callers will json.loads() if needed
        return {
            "status_code": response.status_code,
            "data": response.text
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# File Logging Utilities
# ============================================================================

def get_output_dir(agent_name: str, domain: str) -> str:
    """Get the output directory path for a specific agent and domain.
    
    Args:
        agent_name: Name of the agent being evaluated
        domain: Evaluation domain (e.g., 'medagentbench')
        
    Returns:
        Path to the output directory
    """
    return os.path.join(OUTPUT_DIR, agent_name, domain)


def write_run_result(
    output_dir: str,
    task_id: str,
    task_result: "TaskResult",
    task_data: dict,
    history: list,
    error: Optional[str] = None,
) -> None:
    """Write a single task run result to runs.jsonl or error.jsonl.
    
    Args:
        output_dir: Directory to write the result to
        task_id: Task identifier
        task_result: The TaskResult object containing results
        task_data: Original task data
        history: Conversation history
        error: Error message if the task failed
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp_ms = int(time.time() * 1000)
    time_str = datetime.fromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    
    # Build the result record
    record = {
        "index": task_id,
        "error": error,
        "output": {
            "result": task_result.agent_answer,
            "status": task_result.status,
            "correct": task_result.correct,
            "expected": task_result.expected_answer,
            "rounds": task_result.rounds,
            "time_used": task_result.time_used,
            "history": history,
        } if task_result else None,
        "task_data": {
            "id": task_data.get("id"),
            "instruction": task_data.get("instruction"),
            "context": task_data.get("context"),
        },
        "time": {"timestamp": timestamp_ms, "str": time_str},
    }
    
    # Write to appropriate file
    if error:
        target_file = os.path.join(output_dir, "error.jsonl")
    else:
        target_file = os.path.join(output_dir, "runs.jsonl")
    
    with open(target_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"Result written to {target_file}")


def write_overall_results(
    output_dir: str,
    eval_results: "EvalResults",
) -> None:
    """Write overall evaluation results to overall.json.
    
    Args:
        output_dir: Directory to write the results to
        eval_results: The EvalResults object containing overall metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "overall.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_results.model_dump(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Overall results written to {output_file}")


# ============================================================================
# Task Evaluation Functions
# ============================================================================

class TaskOutput:
    """Wrapper for task results to provide history access."""
    def __init__(self, result: str, history: list, status: str = "completed", rounds: int = 1, fhir_ops: list = None):
        self.result = result
        self.history = [HistoryItem(h["role"], h["content"]) for h in history]
        self.status = status
        self.rounds = rounds
        self.fhir_ops = fhir_ops or []


class HistoryItem:
    """History item with role and content attributes."""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

def extract_posts(results: TaskOutput):
    """Extract POST requests from fhir_ops.
    
    Returns list of (url, payload) tuples for POST operations.
    """
    posts = []
    for op in results.fhir_ops:
        fhir_url = op.get("fhir_url", "")
        parameters = op.get("parameters", {})
        if fhir_url and parameters:
            posts.append((fhir_url, parameters))
    return posts


def check_has_post(results: TaskOutput):
    """Check if agent made any FHIR POST requests."""
    return len(results.fhir_ops) > 0


def calculate_age(dob):
    """Calculate age from date of birth."""
    today = datetime(2023, 11, 13)
    age = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
    return age


# Task evaluation functions
def eval_task1(case_data, results, fhir_api_base):
    if check_has_post(results):
        return False
    ref_sol = case_data['sol']
    try:
        return ref_sol == json.loads(results.result)
    except:
        return False


def eval_task2(case_data, results, fhir_api_base):
    if check_has_post(results):
        return False
    url = f"{fhir_api_base}Patient?identifier={case_data['eval_MRN']}&_format=json"
    get_res = json.loads(send_get_request(url)['data'])
    dob_str = get_res['entry'][0]['resource']['birthDate']
    parsed_date = datetime.strptime(dob_str, "%Y-%m-%d")
    ref_sol = [calculate_age(parsed_date)]
    try:
        return ref_sol == json.loads(results.result)
    except:
        return False


def normalize_fhir_url(url: str) -> str:
    """Extract just the path portion of a FHIR URL for comparison.
    
    This handles cases where MCP server uses localhost but evaluator expects remote URL.
    """
    parsed = urlparse(url.rstrip("/"))
    return parsed.path


def eval_task3(case_data, results, fhir_api_base):
    posts = extract_posts(results)
    if len(posts) != 1:
        return False
    url, payload = posts[0]
    
    # Compare only the path portion (e.g., /fhir/Observation) to handle localhost vs remote
    expected_path = normalize_fhir_url(f'{fhir_api_base.rstrip("/")}/Observation')
    actual_path = normalize_fhir_url(url)
    if actual_path != expected_path:
        return False
    try:
        assert payload['resourceType'] == 'Observation'
        assert len(payload['category']) == 1
        assert len(payload['category'][0]['coding']) == 1
        
        assert payload['category'][0]['coding'][0] == {'system': "http://hl7.org/fhir/observation-category", "code": "vital-signs", "display": "Vital Signs"}
        assert payload['code']== {'text': 'BP'}
        
        assert payload['effectiveDateTime'] == '2023-11-13T10:15:00+00:00'
        assert payload['status'] == 'final'
        assert payload['valueString'] == '118/77 mmHg'
        assert payload['subject'] == {'reference': f"Patient/{case_data['eval_MRN']}"}
    except:
        return False
    return True


def eval_task4(case_data, results, fhir_api_base):
    if check_has_post(results):
        return False
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=MG&_count=5000&_format=json"
    get_res = json.loads(send_get_request(url)['data'])
    cutoff = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
    last_meas, last_value = None, None
    for i in get_res.get('entry', []):
        effective_time = datetime.fromisoformat(i['resource']['effectiveDateTime'])
        value = i['resource']['valueQuantity']['value']
        if effective_time >= (cutoff - timedelta(hours=24)):
            if (last_meas is None) or (effective_time > last_meas):
                last_meas = effective_time
                last_value = value
    ref_sol = [last_value if last_value is not None else -1]
    try:
        return ref_sol == json.loads(results.result)
    except:
        return False


def eval_task5(case_data, results, fhir_api_base):
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=MG&_count=5000&_format=json"
    get_res = json.loads(send_get_request(url)['data'])
    cutoff = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
    last_meas, last_value = None, None
    for i in get_res.get('entry', []):
        effective_time = datetime.fromisoformat(i['resource']['effectiveDateTime'])
        value = i['resource']['valueQuantity']['value']
        if effective_time >= (cutoff - timedelta(hours=24)):
            if (last_meas is None) or (effective_time > last_meas):
                last_meas = effective_time
                last_value = value
    
    if last_value is None:
        if check_has_post(results):
            return False
        return True
    elif last_value > 1.9:
        if check_has_post(results):
            return False
    else:
        posts = extract_posts(results)
        if len(posts) != 1:
            return False
        url, payload = posts[0]
        expected_path = normalize_fhir_url(f'{fhir_api_base.rstrip("/")}/MedicationRequest')
        if normalize_fhir_url(url) != expected_path:
            return False
        try:
            assert payload['resourceType'] == 'MedicationRequest'
            assert payload['medicationCodeableConcept']['coding'][0]['system'] == "http://hl7.org/fhir/sid/ndc"
            assert payload['medicationCodeableConcept']['coding'][0]['code'] == "0338-1715-40"
            assert '2023-11-13T10:15' in payload['authoredOn']
            
            assert payload['dosageInstruction'][0]['route'] == 'IV'
            
            if last_value < 1:
                dose, rate = 4, 4
            elif last_value < 1.5:
                dose, rate = 2, 2
            else:
                dose, rate = 1, 1
            assert payload['dosageInstruction'][0]['doseAndRate'][0]['doseQuantity'] == {'value': dose, 'unit': 'g'}
            assert payload['dosageInstruction'][0]['doseAndRate'][0]['rateQuantity'] == {'value': rate, 'unit': 'h'}
            assert payload['status'] == 'active'
            assert payload['intent'] == 'order'
            assert payload['subject'] == {'reference': f"Patient/{case_data['eval_MRN']}"}
        except:
            return False
    
    ref_sol = [last_value if last_value is not None else -1]
    try:
        if (ref_sol == json.loads(results.result)) or ([] == json.loads(results.result)): #We only ask the model to check, so it's fine if model returns []
            return True
        return False
    except:
        return False


def eval_task6(case_data, results, fhir_api_base):
    if check_has_post(results):
        return False
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=GLU&_count=5000&_format=json"
    get_res = json.loads(send_get_request(url)['data'])
    cutoff = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
    glu_sum, glu_count = 0., 0.
    for i in get_res.get('entry', []):
        effective_time = datetime.fromisoformat(i['resource']['effectiveDateTime'])
        value = i['resource']['valueQuantity']['value']
        if effective_time >= (cutoff - timedelta(hours=24)):
            glu_sum += value
            glu_count += 1
    
    ref_sol = [glu_sum/glu_count if glu_count != 0 else -1]
    try:
        l = json.loads(results.result)
        return len(l) == 1 and abs(l[0] - ref_sol[0]) < 0.1
    except:
        return False


def eval_task7(case_data, results, fhir_api_base):
    if check_has_post(results):
        return False
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=GLU&_count=5000&_format=json"
    get_res = json.loads(send_get_request(url)['data'])
    last_meas, last_value = None, None
    for i in get_res.get('entry', []):
        effective_time = datetime.fromisoformat(i['resource']['effectiveDateTime'])
        value = i['resource']['valueQuantity']['value']
        if (last_meas is None) or (effective_time > last_meas):
            last_meas = effective_time
            last_value = value
    ref_sol = [last_value if last_value is not None else -1]
    print(case_data['id'], ref_sol, results.result, flush=True)
    try:
        if ref_sol == json.loads(results.result):
            return True
        return False
    except:
        return False


def eval_task8(case_data, results, fhir_api_base):
    posts = extract_posts(results)
    if len(posts) != 1:
        return False
    url, payload = posts[0]
    expected_path = normalize_fhir_url(f'{fhir_api_base.rstrip("/")}/ServiceRequest')
    if normalize_fhir_url(url) != expected_path:
        return False
    comment = "Situation: acute left knee injury, Background: radiology report indicates ACL tear. Assessment: ACL tear grade II. Recommendation: request for Orthopedic service to evaluate and provide management recommendations."
    try:
        assert payload['resourceType'] == 'ServiceRequest'
        assert payload['code']['coding'][0]['system'] == 'http://snomed.info/sct'
        assert payload['code']['coding'][0]['code'] == '306181000000106'
        assert payload['authoredOn'] == '2023-11-13T10:15:00+00:00'
        assert payload['status'] == 'active'
        assert payload['intent'] == 'order'
        assert payload['priority'] == 'stat'
        assert comment in payload['note']['text']
        assert payload['subject'] == {'reference': f"Patient/{case_data['eval_MRN']}"}
    except:
        return False
    return True


def eval_task9(case_data, results, fhir_api_base):
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=K&_count=5000&_format=json"
    get_res = json.loads(send_get_request(url)['data'])
    cutoff = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
    last_meas, last_value = None, None
    for i in get_res.get('entry', []):
        effective_time = datetime.fromisoformat(i['resource']['effectiveDateTime'])
        value = i['resource']['valueQuantity']['value']
        if (last_meas is None) or (effective_time > last_meas):
            last_meas = effective_time
            last_value = value
    
    if last_value is None or last_value >= 3.5:
        if check_has_post(results):
            return False
    else:
        posts = extract_posts(results)
        if len(posts) != 2:
            return False
        url, payload = posts[0]
        expected_path = normalize_fhir_url(f'{fhir_api_base.rstrip("/")}/MedicationRequest')
        if normalize_fhir_url(url) != expected_path:
            return False
        try:
            assert payload['resourceType'] == 'MedicationRequest'
            assert payload['medicationCodeableConcept']['coding'][0]['system'] == "http://hl7.org/fhir/sid/ndc"
            assert payload['medicationCodeableConcept']['coding'][0]['code'] == "40032-917-01"
            assert '2023-11-13T10:15' in payload['authoredOn']
            assert payload['dosageInstruction'][0]['route'].lower().strip() == 'oral'
            dose = (3.5 - last_value) / 0.1 * 10
            actual_dose = payload['dosageInstruction'][0]['doseAndRate'][0]['doseQuantity']['value']
            assert abs(actual_dose - dose) <= 0.1
            assert payload['dosageInstruction'][0]['doseAndRate'][0]['doseQuantity']['unit'] == 'mEq'
            assert payload['status'] == 'active'
            assert payload['intent'] == 'order'
            assert payload['subject'] == {'reference': f"Patient/{case_data['eval_MRN']}"}
        except:
            return False
        
        url, payload = posts[1]
        expected_path = normalize_fhir_url(f'{fhir_api_base.rstrip("/")}/ServiceRequest')
        if normalize_fhir_url(url) != expected_path:
            return False
        try:
            assert payload['resourceType'] == 'ServiceRequest'
            assert payload['code']['coding'][0]['system'] == 'http://loinc.org'
            assert payload['code']['coding'][0]['code'] == '2823-3'
            assert payload['authoredOn'] == '2023-11-13T10:15:00+00:00'
            assert payload['status'] == 'active'
            assert payload['intent'] == 'order'
            assert payload['priority'] == 'stat'
            assert payload['subject'] == {'reference': f"Patient/{case_data['eval_MRN']}"}
            assert '2023-11-14T08:' in payload.get('occurrenceDateTime', '')
        except:
            return False

    ref_sol = [last_value if last_value is not None else -1]
    print(case_data['id'], ref_sol, results.result, flush=True)
    try:
        if (ref_sol == json.loads(results.result)) or ([] == json.loads(results.result)): #We only ask the model to check, so it's fine if model returns []
            return True
        return False
    except:
        return False


def eval_task10(case_data, results, fhir_api_base):
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=A1C&_count=5000&_format=json"
    get_res = json.loads(send_get_request(url)['data'])
    cutoff = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
    last_meas, last_value, last_time = None, None, None
    for i in get_res.get('entry', []):
        effective_time = datetime.fromisoformat(i['resource']['effectiveDateTime'])
        value = i['resource']['valueQuantity']['value']
        if (last_meas is None) or (effective_time > last_meas):
            last_meas = effective_time
            last_time = i['resource']['effectiveDateTime']
            last_value = value
    
    if (last_value is None):
        ref_sol = [-1]
    else: 
        ref_sol = [last_value, last_time]
    
    if (last_value is None) or (last_meas < datetime.fromisoformat("2022-11-13T10:15:00+00:00")): #Order needed
        posts = extract_posts(results)
        if len(posts) != 1: #Should be one for A1C test
            return False
        url, payload = posts[0]
        expected_path = normalize_fhir_url(f'{fhir_api_base.rstrip("/")}/ServiceRequest')
        if normalize_fhir_url(url) != expected_path:
            return False
        try:
            assert (payload['resourceType'] == 'ServiceRequest')
            assert payload['code']['coding'][0]['system'] == 'http://loinc.org'
            assert payload['code']['coding'][0]['code'] == '4548-4'
            assert payload['authoredOn'] == '2023-11-13T10:15:00+00:00'
            assert payload['status'] == 'active'
            assert payload['intent'] == 'order'
            assert payload['priority'] == 'stat'
            assert payload['subject'] == {'reference': f"Patient/{case_data['eval_MRN']}"}
        except Exception as e:
            print(e, flush=True)
            return False
    else:#No order needed
        if check_has_post(results) is True:
            return False


    print(case_data['id'], ref_sol, results.result, flush=True)
    try:
        if (ref_sol == json.loads(results.result)) or ([] == json.loads(results.result)): #We only ask the model to check, so it's fine if model returns []
            return True
        return False
    except:
        return False


# Task evaluation dispatcher
TASK_EVALUATORS = {
    "task1": eval_task1,
    "task2": eval_task2,
    "task3": eval_task3,
    "task4": eval_task4,
    "task5": eval_task5,
    "task6": eval_task6,
    "task7": eval_task7,
    "task8": eval_task8,
    "task9": eval_task9,
    "task10": eval_task10,
}


def evaluate_task(case_data: dict, results: TaskOutput, fhir_api_base: str) -> bool:
    """Evaluate a task using the appropriate grading function."""
    task_id = case_data['id'].split('_')[0]
    evaluator = TASK_EVALUATORS.get(task_id)
    if evaluator:
        try:
            return evaluator(case_data, results, fhir_api_base) is True
        except Exception as e:
            logger.error(f"Evaluation error for {task_id}: {e}")
            return False
    logger.warning(f"No evaluator found for task type: {task_id}")
    return False


# ============================================================================
# MedAgentBench Prompt Template
# ============================================================================

TOOL_CALL_EXAMPLES = """- search_patients:
  {"identifier":"EXAMPLE_ID"}
- list_patient_problems:
  {"patient":"Patient/EXAMPLE","category":"problem-list-item"}
- list_lab_observations:
  {"patient":"Patient/EXAMPLE","code":"EXAMPLE_CODE","date":"2000-01-01"}
- list_vital_signs:
  {"patient":"Patient/EXAMPLE","category":"vital-signs","date":"2000-01-01"}
- record_vital_observation:
  {"resourceType":"Observation","category":[{"coding":[{"system":"http://hl7.org/fhir/observation-category","code":"vital-signs","display":"Vital Signs"}]}],"code":{"text":"EXAMPLE_FLOW_ID"},"effectiveDateTime":"2000-01-01T00:00:00+00:00","status":"final","valueString":"120/80 mmHg","subject":{"reference":"Patient/EXAMPLE"}}
- list_medication_requests:
  {"patient":"Patient/EXAMPLE","category":"inpatient","date":"2000-01-01"}
- create_medication_request:
  {"resourceType":"MedicationRequest","medicationCodeableConcept":{"coding":[{"system":"http://hl7.org/fhir/sid/ndc","code":"00000","display":"ExampleMed"}],"text":"ExampleMed 100mg"},"authoredOn":"2000-01-01","dosageInstruction":[{"route":{"text":"oral"},"doseAndRate":[{"doseQuantity":{"value":100,"unit":"mg"}}]}],"status":"active","intent":"order","subject":{"reference":"Patient/EXAMPLE"}}
- list_patient_procedures:
  {"patient":"Patient/EXAMPLE","date":"2000-01-01","code":"00000"}
- create_service_request:
  {"resourceType":"ServiceRequest","code":{"coding":[{"system":"http://loinc.org","code":"00000","display":"Example Test"}]},"authoredOn":"2000-01-01T00:00:00+00:00","status":"active","intent":"order","priority":"stat","subject":{"reference":"Patient/EXAMPLE"},"occurrenceDateTime":"2000-01-01T01:00:00+00:00","note":{"text":"Example note"}}"""

MEDAGENT_PROMPT = """You are an expert medical AI assistant that uses FHIR functions to assist medical professionals. You are given a question and a set of available FHIR tools. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

You have access to FHIR tools via the MCP server at: {mcp_server_url}

Instructions:
1. Use the provided FHIR tools to query patient data, retrieve medical records, create orders, and perform other medical record operations as needed.

2. Read the arguments carefully for each tool before deciding which are the suitable arguments to use. Some arguments are too general and some are more relevant to the question.

3. CRITICAL: You MUST provide ALL required arguments for each tool call. Follow the EXACT format shown in the tool descriptions. Use example URIs exactly as written in the tool descriptions (do not substitute alternatives). Do not rely on your own FHIR knowledge for parameter structures.

4. Make as many tool calls as needed to gather the information required to answer the question.

TOOL CALL EXAMPLES:
{tool_examples}

5. When you have gathered all necessary information and have the final answer(s), you MUST respond with ONLY the finish format (make sure the list is JSON loadable):
FINISH([answer1, answer2, ...])

IMPORTANT: Your final response MUST be in the FINISH format with no other text. The list inside FINISH() must be valid JSON.

Task Context: {context}

Question: {question}"""


# ============================================================================
# Agent Class
# ============================================================================

class Agent:
    required_roles: list[str] = ["agent"]
    required_config_keys: list[str] = ["domain"]

    def __init__(self):
        self.messenger = Messenger()
        self.tasks: list[dict] = []

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"

    def load_tasks(self, tasks_file: str) -> list[dict]:
        """Load tasks from JSON file."""
        if os.path.exists(tasks_file):
            with open(tasks_file, 'r') as f:
                return json.load(f)
        logger.warning(f"Tasks file not found: {tasks_file}")
        return []

    def sanitize_task_data(self, task_data: dict) -> dict:
        """Remove evaluation fields before sending to agent."""
        evaluation_fields = {'sol', 'eval_MRN'}
        return {k: v for k, v in task_data.items() if k not in evaluation_fields}

    def extract_tool_history(self, response: str) -> tuple[str, list[dict]]:
        """Extract tool call history from agent response."""
        tool_history = []
        clean_response = response
        
        pattern = r'<tool_history>\s*(.*?)\s*</tool_history>'
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            try:
                tool_history = json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse tool_history JSON")
            clean_response = re.sub(pattern, '', clean_response, flags=re.DOTALL).strip()
        
        return clean_response, tool_history

    def extract_fhir_operations(self, response: str) -> tuple[str, list[dict]]:
        """Extract FHIR operations metadata from agent response."""
        fhir_ops = []
        clean_response = response
        
        pattern = r'<fhir_operations>\s*(.*?)\s*</fhir_operations>'
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            try:
                fhir_ops = json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse FHIR operations metadata")
            clean_response = re.sub(pattern, '', response, flags=re.DOTALL).strip()
        
        return clean_response, fhir_ops

    def parse_finish_response(self, response: str) -> tuple[str | None, str]:
        """Parse FINISH format from response."""
        # Clean up response
        response = response.replace('```tool_code', '').replace('```', '').strip()
        
        # Try exact match first
        if response.startswith('FINISH(') and response.endswith(')'):
            return response[len('FINISH('):-1], response
        
        # Try finding FINISH in response
        if 'FINISH(' in response and ')' in response:
            start_idx = response.find('FINISH(')
            paren_count = 0
            for i, char in enumerate(response[start_idx:]):
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        finish_match = response[start_idx:start_idx + i + 1]
                        return finish_match[len('FINISH('):-1], finish_match
        
        return None, response

    async def run_single_task(
        self,
        agent_url: str,
        task_data: dict,
        mcp_server_url: str,
        fhir_api_base: str,
        max_rounds: int,
        updater: TaskUpdater,
        output_dir: Optional[str] = None,
    ) -> TaskResult:
        """Run a single task and return results."""
        start_time = time.time()
        task_id = task_data['id']
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Running task {task_id}..."),
        )
        
        # Build prompt
        safe_task_data = self.sanitize_task_data(task_data)
        task_prompt = MEDAGENT_PROMPT.format(
            mcp_server_url=mcp_server_url,
            context=safe_task_data.get('context', 'N/A'),
            question=safe_task_data['instruction'],
            tool_examples=TOOL_CALL_EXAMPLES,
        )
        
        history = [{"role": "user", "content": task_prompt}]
        
        try:
            # Send to agent
            response = await self.messenger.talk_to_agent(
                message=task_prompt,
                url=agent_url,
                new_conversation=True,
            )
            
            # Extract tool history and FHIR operations
            response_after_tools, tool_history = self.extract_tool_history(response)
            clean_response, fhir_ops = self.extract_fhir_operations(response_after_tools)
            
            # Add tool calls to history
            for tool_call in tool_history:
                tool_name = tool_call.get("tool_name", "unknown")
                arguments = tool_call.get("arguments", {})
                result = tool_call.get("result", {})
                is_error = tool_call.get("is_error", False)
                
                # Add tool call entry
                history.append({
                    "role": "tool_call",
                    "content": json.dumps({
                        "tool": tool_name,
                        "arguments": arguments
                    })
                })
                
                # Add tool result entry
                history.append({
                    "role": "tool_result",
                    "content": json.dumps({
                        "tool": tool_name,
                        "result": result,
                        "is_error": is_error
                    })
                })
            
            # Add final response
            history.append({"role": "agent", "content": clean_response})
            
            # Parse response
            answer, _ = self.parse_finish_response(clean_response)
            
            if answer is not None:
                # Create TaskOutput for evaluation (pass fhir_ops directly)
                task_output = TaskOutput(
                    result=answer,
                    history=history,
                    status="completed",
                    rounds=1,
                    fhir_ops=fhir_ops,
                )
                
                # Evaluate
                is_correct = evaluate_task(task_data, task_output, fhir_api_base)
                
                result = TaskResult(
                    task_id=task_id,
                    correct=is_correct,
                    status="completed",
                    agent_answer=answer,
                    expected_answer=task_data.get('sol'),
                    time_used=time.time() - start_time,
                    rounds=1,
                )
                
                # Write to runs.jsonl
                if output_dir:
                    write_run_result(output_dir, task_id, result, task_data, history)
                
                return result
            else:
                result = TaskResult(
                    task_id=task_id,
                    correct=False,
                    status="agent_invalid_action",
                    agent_answer=clean_response[:200] if clean_response else None,
                    expected_answer=task_data.get('sol'),
                    time_used=time.time() - start_time,
                    rounds=1,
                )
                
                # Write to runs.jsonl
                if output_dir:
                    write_run_result(output_dir, task_id, result, task_data, history)
                
                return result
                
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            result = TaskResult(
                task_id=task_id,
                correct=False,
                status="error",
                agent_answer=str(e),
                expected_answer=task_data.get('sol'),
                time_used=time.time() - start_time,
                rounds=0,
            )
            
            # Write to error.jsonl
            if output_dir:
                write_run_result(output_dir, task_id, result, task_data, history, error=str(e))
            
            return result

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Main entry point for the evaluator."""
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
            ok, validation_msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(validation_msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting MedAgentBench assessment.\n{request.model_dump_json()}"),
        )

        # Extract configuration
        agent_url = str(request.participants["agent"])
        domain = str(request.config.get("domain", "medagentbench"))
        num_tasks = request.config.get("num_tasks")
        task_ids = request.config.get("task_ids")
        max_rounds = int(request.config.get("max_rounds", 10))
        mcp_server_url = str(request.config.get("mcp_server_url", "http://localhost:8002"))
        fhir_api_base = str(request.config.get("fhir_api_base", "http://medagentbench.ddns.net:8080/fhir/"))
        
        # Get agent name from config or derive from URL
        agent_name = str(request.config.get("agent_name", "default_agent"))
        
        # Set up output directory for logging
        output_dir = get_output_dir(agent_name, domain)
        logger.info(f"Results will be written to {output_dir}")
        
        # Load tasks
        tasks_file = os.environ.get(
            "MEDAGENTBENCH_TASKS_FILE",
            "src/mcp/resources/tasks/tasks.json"
        )
        all_tasks = self.load_tasks(tasks_file)
        
        # Filter tasks
        if task_ids:
            tasks = [t for t in all_tasks if t['id'] in task_ids]
        elif num_tasks:
            tasks = all_tasks[:num_tasks]
        else:
            tasks = all_tasks

        logger.info(f"Running {len(tasks)} tasks for domain {domain}")
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Evaluating {len(tasks)} tasks in {domain} domain"),
        )

        start_time = time.time()
        task_results: dict[str, TaskResult] = {}
        correct_count = 0

        try:
            for task in tasks:
                result = await self.run_single_task(
                    agent_url=agent_url,
                    task_data=task,
                    mcp_server_url=mcp_server_url,
                    fhir_api_base=fhir_api_base,
                    max_rounds=max_rounds,
                    updater=updater,
                    output_dir=output_dir,
                )
                task_results[result.task_id] = result
                if result.correct:
                    correct_count += 1
                
                status_emoji = "✓" if result.correct else "✗"
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Task {result.task_id}: {status_emoji} ({result.time_used:.1f}s)"),
                )

            time_used = time.time() - start_time
            pass_rate = (correct_count / len(tasks) * 100) if tasks else 0

            eval_results = EvalResults(
                domain=domain,
                total_tasks=len(tasks),
                correct_count=correct_count,
                pass_rate=pass_rate,
                task_results=task_results,
                time_used=time_used,
            )

            # Build summary
            task_results_str = "\n".join(
                f"  {tid}: {'✓' if r.correct else '✗'} ({r.status})"
                for tid, r in task_results.items()
            )
            summary = f"""MedAgentBench Results
Domain: {domain}
Tasks: {len(tasks)}
Pass Rate: {pass_rate:.1f}% ({correct_count}/{len(tasks)})
Time: {time_used:.1f}s

Task Results:
{task_results_str}"""

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary)),
                    Part(root=DataPart(data=eval_results.model_dump())),
                ],
                name="Result",
            )
            
            # Write overall results to file
            write_overall_results(output_dir, eval_results)
        finally:
            self.messenger.reset()

