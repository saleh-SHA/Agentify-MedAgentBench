"""MedAgentBench evaluator (green agent) - orchestrates and evaluates medical AI agents."""

import asyncio
import json
import logging
import os
import queue
import re
import socket
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import requests
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import AgentResponse, Messenger

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medagentbench_evaluator")

# Output directory for runs.jsonl logging (required)
OUTPUT_DIR = "outputs/medagentbench"


# ============================================================================
# Failure Type Definitions
# ============================================================================

class FailureType(str, Enum):
    """Primary failure categories for task evaluation.
    
    These are mutually exclusive categories used for overall failure breakdown.
    Order matters - earlier types take precedence when determining primary failure.
    """
    # System/format failures (highest precedence)
    SYSTEM_ERROR = "system_error"
    INVALID_FINISH_FORMAT = "invalid_finish_format"
    INVALID_JSON_RESULT = "invalid_json_result"
    MAX_ROUNDS_REACHED = "max_rounds_reached"  # Agent hit iteration limit
    
    # Operation type failures
    READONLY_VIOLATION = "readonly_violation"
    WRONG_POST_COUNT = "wrong_post_count"
    WRONG_ENDPOINT = "wrong_endpoint"
    
    # Payload validation failures
    PAYLOAD_VALIDATION_ERROR = "payload_validation_error"
    
    # Answer comparison failures (lowest precedence)
    ANSWER_MISMATCH = "answer_mismatch"


class DetailedFailure(str, Enum):
    """Detailed failure reasons for task-level diagnostics.
    
    These provide fine-grained information about what went wrong.
    A task can have multiple detailed failures.
    """
    # Format failures
    INVALID_JSON = "invalid_json"
    NO_FINISH_FORMAT = "no_finish_format"
    MAX_ITERATIONS_EXCEEDED = "max_iterations_exceeded"
    
    # Operation failures
    MADE_POST_ON_READONLY = "made_post_on_readonly"
    WRONG_NUMBER_OF_POSTS = "wrong_number_of_posts"
    WRONG_FHIR_ENDPOINT = "wrong_fhir_endpoint"
    
    # Resource type failures
    WRONG_RESOURCE_TYPE = "wrong_resource_type"
    
    # Observation-specific failures
    WRONG_CATEGORY_COUNT = "wrong_category_count"
    WRONG_CODING_COUNT = "wrong_coding_count"
    WRONG_CATEGORY_SYSTEM = "wrong_category_system"
    WRONG_CATEGORY_CODE = "wrong_category_code"
    WRONG_CATEGORY_DISPLAY = "wrong_category_display"
    WRONG_CODE_TEXT = "wrong_code_text"
    WRONG_EFFECTIVE_DATETIME = "wrong_effective_datetime"
    WRONG_STATUS = "wrong_status"
    WRONG_VALUE_STRING = "wrong_value_string"
    WRONG_SUBJECT = "wrong_subject"
    
    # MedicationRequest-specific failures
    WRONG_MEDICATION_SYSTEM = "wrong_medication_system"
    WRONG_MEDICATION_CODE = "wrong_medication_code"
    WRONG_AUTHORED_ON = "wrong_authored_on"
    WRONG_ROUTE = "wrong_route"
    WRONG_DOSE_VALUE = "wrong_dose_value"
    WRONG_DOSE_UNIT = "wrong_dose_unit"
    WRONG_RATE_VALUE = "wrong_rate_value"
    WRONG_RATE_UNIT = "wrong_rate_unit"
    WRONG_INTENT = "wrong_intent"
    
    # ServiceRequest-specific failures
    WRONG_CODE_SYSTEM = "wrong_code_system"
    WRONG_CODE_CODE = "wrong_code_code"
    WRONG_PRIORITY = "wrong_priority"
    MISSING_NOTE = "missing_note"
    WRONG_NOTE_CONTENT = "wrong_note_content"
    WRONG_OCCURRENCE_DATETIME = "wrong_occurrence_datetime"
    
    # Answer comparison failures
    ANSWER_VALUE_MISMATCH = "answer_value_mismatch"
    ANSWER_LENGTH_MISMATCH = "answer_length_mismatch"
    
    # Task 11 - BP Trend specific failures
    WRONG_PATIENT_ID = "wrong_patient_id"
    MISSING_BP_READINGS = "missing_bp_readings"
    WRONG_BP_READINGS_COUNT = "wrong_bp_readings_count"
    WRONG_TREND = "wrong_trend"
    WRONG_HYPERTENSION_ALERT = "wrong_hypertension_alert"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_BP_READING_FORMAT = "invalid_bp_reading_format"


@dataclass
class EvalOutcome:
    """Result of evaluating a task with detailed failure information.
    
    Attributes:
        passed: Whether the task passed evaluation
        primary_failure: The main failure category (for overall breakdown)
        failure_details: List of specific failure reasons (for task-level diagnostics)
    """
    passed: bool
    primary_failure: str | None = None
    failure_details: list[str] = field(default_factory=list)
    
    @classmethod
    def success(cls) -> "EvalOutcome":
        """Create a successful evaluation outcome."""
        return cls(passed=True)
    
    @classmethod
    def failure(
        cls, 
        primary: FailureType | str, 
        details: list[DetailedFailure | str] | None = None
    ) -> "EvalOutcome":
        """Create a failed evaluation outcome.
        
        Args:
            primary: The primary failure category
            details: List of detailed failure reasons
        """
        primary_str = primary.value if isinstance(primary, FailureType) else primary
        details_list = []
        if details:
            for d in details:
                details_list.append(d.value if isinstance(d, DetailedFailure) else d)
        return cls(passed=False, primary_failure=primary_str, failure_details=details_list)


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
    tools_called: int = 0  # Total number of tool calls made during the task
    # Failure tracking fields (only populated when correct=False)
    primary_failure: str | None = None
    failure_details: list[str] | None = None


class EvalResults(BaseModel):
    domain: str
    total_tasks: int
    correct_count: int
    pass_rate: float
    failure_breakdown: dict[str, float] | None = None  # Percentage breakdown of failure types
    min_rounds: int | None = None  # Minimum rounds across all tasks
    max_rounds: int | None = None  # Maximum rounds across all tasks
    avg_rounds: float | None = None  # Average rounds across all tasks
    avg_tools_called: float | None = None  # Average tool calls per task
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
    
    # Build the output section
    output_section = None
    if task_result:
        output_section = {
            "result": task_result.agent_answer,
            "status": task_result.status,
            "correct": task_result.correct,
            "expected": task_result.expected_answer,
            "rounds": task_result.rounds,
            "tools_called": task_result.tools_called,
            "time_used": task_result.time_used,
            "history": history,
        }
        # Add failure information if task failed
        if not task_result.correct:
            output_section["primary_failure"] = task_result.primary_failure
            output_section["failure_details"] = task_result.failure_details
    
    # Build the result record
    record = {
        "index": task_id,
        "error": error,
        "output": output_section,
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
# Thread-Safe File Writer for Parallel Execution
# ============================================================================

class ThreadSafeFileWriter:
    """Thread-safe file writer using threading.Lock for true parallel execution."""

    def __init__(self):
        self._locks: dict[str, threading.Lock] = {}
        self._master_lock = threading.Lock()

    def _get_lock(self, filepath: str) -> threading.Lock:
        """Get or create a lock for a specific file path."""
        with self._master_lock:
            if filepath not in self._locks:
                self._locks[filepath] = threading.Lock()
            return self._locks[filepath]

    def write_run_result(
        self,
        output_dir: str,
        task_id: str,
        task_result: "TaskResult",
        task_data: dict,
        history: list,
        error: Optional[str] = None,
    ) -> None:
        """Thread-safe write of a single task run result to runs.jsonl or error.jsonl.

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
                "tools_called": task_result.tools_called,
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

        # Use threading lock for thread-safe file writes
        lock = self._get_lock(target_file)
        with lock:
            with open(target_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Result written to {target_file}")


# ============================================================================
# Worker Pool for Multi-Process Parallelism
# ============================================================================

class WorkerPool:
    """Manages multiple agent worker processes for true parallel execution.

    Similar to MedAgentBench's worker architecture, this spawns multiple agent
    processes on different ports to handle tasks in parallel.
    """

    BASE_PORT = 9019  # Starting port for agent workers
    AGENT_SCRIPT = "scenarios/medagentbench/agent/src/server.py"

    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers
        self.workers: list[dict] = []  # List of {process, port, url, available}
        self._lock = threading.Lock()
        self._available_workers = queue.Queue()

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available for use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False

    def _wait_for_service(self, host: str, port: int, timeout: int = 60) -> bool:
        """Wait for a service to become available on a specific port."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    s.connect((host, port))
                    return True
            except (socket.error, socket.timeout):
                time.sleep(0.5)
        return False

    def start_workers(self) -> list[str]:
        """Start all worker processes and return their URLs.

        Returns:
            List of worker URLs (e.g., ["http://localhost:9019", "http://localhost:9020"])
        """
        logger.info(f"Starting {self.num_workers} agent worker(s)...")

        worker_urls = []

        for i in range(self.num_workers):
            port = self.BASE_PORT + i

            # Check if port is already in use (maybe from previous run)
            if not self._is_port_available(port):
                logger.warning(f"Port {port} already in use, assuming worker is running")
                url = f"http://localhost:{port}"
                worker_urls.append(url)
                self.workers.append({
                    "process": None,  # Not managed by us
                    "port": port,
                    "url": url,
                    "managed": False,
                })
                self._available_workers.put(url)
                continue

            # Start new worker process
            try:
                process = subprocess.Popen(
                    [
                        "python", self.AGENT_SCRIPT,
                        "--host", "0.0.0.0",
                        "--port", str(port),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                url = f"http://localhost:{port}"
                self.workers.append({
                    "process": process,
                    "port": port,
                    "url": url,
                    "managed": True,
                })

                logger.info(f"Started worker {i+1}/{self.num_workers} on port {port} (PID: {process.pid})")

            except Exception as e:
                logger.error(f"Failed to start worker on port {port}: {e}")
                continue

        # Wait for all workers to be ready
        logger.info("Waiting for workers to be ready...")
        for worker in self.workers:
            port = worker["port"]
            if self._wait_for_service("localhost", port, timeout=30):
                logger.info(f"Worker on port {port} is ready")
                worker_urls.append(worker["url"])
                self._available_workers.put(worker["url"])
            else:
                logger.error(f"Worker on port {port} failed to start within timeout")
                if worker["managed"] and worker["process"]:
                    worker["process"].terminate()

        logger.info(f"Successfully started {len(worker_urls)} worker(s)")
        return worker_urls

    def get_worker(self, timeout: float = 300) -> str:
        """Get an available worker URL (blocks until one is available).

        Args:
            timeout: Maximum time to wait for a worker

        Returns:
            Worker URL

        Raises:
            queue.Empty: If no worker becomes available within timeout
        """
        return self._available_workers.get(timeout=timeout)

    def release_worker(self, url: str) -> None:
        """Release a worker back to the pool."""
        self._available_workers.put(url)

    def stop_workers(self) -> None:
        """Stop all managed worker processes."""
        logger.info("Stopping worker processes...")

        for worker in self.workers:
            if worker["managed"] and worker["process"]:
                try:
                    worker["process"].terminate()
                    worker["process"].wait(timeout=10)
                    logger.info(f"Stopped worker on port {worker['port']}")
                except subprocess.TimeoutExpired:
                    worker["process"].kill()
                    logger.warning(f"Force killed worker on port {worker['port']}")
                except Exception as e:
                    logger.error(f"Error stopping worker on port {worker['port']}: {e}")

        self.workers.clear()
        # Clear the queue
        while not self._available_workers.empty():
            try:
                self._available_workers.get_nowait()
            except queue.Empty:
                break

        logger.info("All workers stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start_workers()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_workers()
        return False


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
def eval_task1(case_data, results, fhir_api_base) -> EvalOutcome:
    """Task 1: Basic Query (Read-Only) - Find patient MRN."""
    # Check for read-only violation
    if check_has_post(results):
        return EvalOutcome.failure(
            FailureType.READONLY_VIOLATION,
            [DetailedFailure.MADE_POST_ON_READONLY]
        )
    
    ref_sol = case_data['sol']
    
    # Check JSON parsing
    try:
        agent_result = json.loads(results.result)
    except (json.JSONDecodeError, TypeError):
        return EvalOutcome.failure(
            FailureType.INVALID_JSON_RESULT,
            [DetailedFailure.INVALID_JSON]
        )
    
    # Check answer match
    if ref_sol == agent_result:
        return EvalOutcome.success()
    else:
        return EvalOutcome.failure(
            FailureType.ANSWER_MISMATCH,
            [DetailedFailure.ANSWER_VALUE_MISMATCH]
        )


def eval_task2(case_data, results, fhir_api_base) -> EvalOutcome:
    """Task 2: Patient Age Calculation (Read-Only)."""
    # Check for read-only violation
    if check_has_post(results):
        return EvalOutcome.failure(
            FailureType.READONLY_VIOLATION,
            [DetailedFailure.MADE_POST_ON_READONLY]
        )
    
    # Calculate reference solution
    url = f"{fhir_api_base}Patient?identifier={case_data['eval_MRN']}&_format=json"
    get_res = json.loads(send_get_request(url)['data'])
    dob_str = get_res['entry'][0]['resource']['birthDate']
    parsed_date = datetime.strptime(dob_str, "%Y-%m-%d")
    ref_sol = [calculate_age(parsed_date)]
    
    # Check JSON parsing
    try:
        agent_result = json.loads(results.result)
    except (json.JSONDecodeError, TypeError):
        return EvalOutcome.failure(
            FailureType.INVALID_JSON_RESULT,
            [DetailedFailure.INVALID_JSON]
        )
    
    # Check answer match
    if ref_sol == agent_result:
        return EvalOutcome.success()
    else:
        return EvalOutcome.failure(
            FailureType.ANSWER_MISMATCH,
            [DetailedFailure.ANSWER_VALUE_MISMATCH]
        )


def normalize_fhir_url(url: str) -> str:
    """Extract just the path portion of a FHIR URL for comparison.
    
    This handles cases where MCP server uses localhost but evaluator expects remote URL.
    """
    parsed = urlparse(url.rstrip("/"))
    return parsed.path


def eval_task3(case_data, results, fhir_api_base) -> EvalOutcome:
    """Task 3: Blood Pressure Observation POST."""
    posts = extract_posts(results)
    
    # Check POST count
    if len(posts) != 1:
        return EvalOutcome.failure(
            FailureType.WRONG_POST_COUNT,
            [DetailedFailure.WRONG_NUMBER_OF_POSTS]
        )
    
    url, payload = posts[0]
    # Check endpoint
    expected_path = normalize_fhir_url(f'{fhir_api_base.rstrip("/")}/Observation')
    actual_path = normalize_fhir_url(url)
    if actual_path != expected_path:
        return EvalOutcome.failure(
            FailureType.WRONG_ENDPOINT,
            [DetailedFailure.WRONG_FHIR_ENDPOINT]
        )
    
    # Validate payload - collect all failures
    failures = []
    
    # Resource type
    if payload.get('resourceType') != 'Observation':
        failures.append(DetailedFailure.WRONG_RESOURCE_TYPE)
    
    # Category validation
    category = payload.get('category', [])
    if len(category) != 1:
        failures.append(DetailedFailure.WRONG_CATEGORY_COUNT)
    elif len(category[0].get('coding', [])) != 1:
        failures.append(DetailedFailure.WRONG_CODING_COUNT)
    else:
        coding = category[0]['coding'][0]
        expected_coding = {
            'system': "http://hl7.org/fhir/observation-category",
            "code": "vital-signs",
            "display": "Vital Signs"
        }
        if coding.get('system') != expected_coding['system']:
            failures.append(DetailedFailure.WRONG_CATEGORY_SYSTEM)
        if coding.get('code') != expected_coding['code']:
            failures.append(DetailedFailure.WRONG_CATEGORY_CODE)
        if coding.get('display') != expected_coding['display']:
            failures.append(DetailedFailure.WRONG_CATEGORY_DISPLAY)
    
    # Code text
    if payload.get('code') != {'text': 'BP'}:
        failures.append(DetailedFailure.WRONG_CODE_TEXT)
    
    # Effective datetime
    if payload.get('effectiveDateTime') != '2023-11-13T10:15:00+00:00':
        failures.append(DetailedFailure.WRONG_EFFECTIVE_DATETIME)
    
    # Status
    if payload.get('status') != 'final':
        failures.append(DetailedFailure.WRONG_STATUS)
    
    # Value string
    if payload.get('valueString') != '118/77 mmHg':
        failures.append(DetailedFailure.WRONG_VALUE_STRING)
    
    # Subject reference
    expected_subject = {'reference': f"Patient/{case_data['eval_MRN']}"}
    if payload.get('subject') != expected_subject:
        failures.append(DetailedFailure.WRONG_SUBJECT)
    
    if failures:
        return EvalOutcome.failure(FailureType.PAYLOAD_VALIDATION_ERROR, failures)
    
    return EvalOutcome.success()


def eval_task4(case_data, results, fhir_api_base) -> EvalOutcome:
    """Task 4: Last Magnesium Value (Read-Only)."""
    # Check for read-only violation
    if check_has_post(results):
        return EvalOutcome.failure(
            FailureType.READONLY_VIOLATION,
            [DetailedFailure.MADE_POST_ON_READONLY]
        )
    
    # Calculate reference solution
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=MG&_count=200&_format=json"
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
    
    # Check JSON parsing
    try:
        agent_result = json.loads(results.result)
    except (json.JSONDecodeError, TypeError):
        return EvalOutcome.failure(
            FailureType.INVALID_JSON_RESULT,
            [DetailedFailure.INVALID_JSON]
        )
    
    # Check answer match
    if ref_sol == agent_result:
        return EvalOutcome.success()
    else:
        return EvalOutcome.failure(
            FailureType.ANSWER_MISMATCH,
            [DetailedFailure.ANSWER_VALUE_MISMATCH]
        )


def eval_task5(case_data, results, fhir_api_base) -> EvalOutcome:
    """Task 5: Magnesium Check with Conditional MedicationRequest."""
    # Get magnesium data
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=MG&_count=200&_format=json"
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
    
    # Case 1: No magnesium data - should NOT make POST
    if last_value is None:
        if check_has_post(results):
            return EvalOutcome.failure(
                FailureType.READONLY_VIOLATION,
                [DetailedFailure.MADE_POST_ON_READONLY]
            )
    # Case 2: Mg >= 1.9 - should NOT make POST (no medication needed)
    elif last_value > 1.9:
        if check_has_post(results):
            return EvalOutcome.failure(
                FailureType.READONLY_VIOLATION,
                [DetailedFailure.MADE_POST_ON_READONLY]
            )
    # Case 3: Mg < 1.9 - MUST make MedicationRequest POST
    else:
        posts = extract_posts(results)
        if len(posts) != 1:
            return EvalOutcome.failure(
                FailureType.WRONG_POST_COUNT,
                [DetailedFailure.WRONG_NUMBER_OF_POSTS]
            )
        
        url, payload = posts[0]
        expected_path = normalize_fhir_url(f'{fhir_api_base.rstrip("/")}/MedicationRequest')
        if normalize_fhir_url(url) != expected_path:
            return EvalOutcome.failure(
                FailureType.WRONG_ENDPOINT,
                [DetailedFailure.WRONG_FHIR_ENDPOINT]
            )
        
        # Validate payload
        failures = []
        
        if payload.get('resourceType') != 'MedicationRequest':
            failures.append(DetailedFailure.WRONG_RESOURCE_TYPE)
        
        # Medication coding
        med_coding = payload.get('medicationCodeableConcept', {}).get('coding', [{}])[0]
        if med_coding.get('system') != "http://hl7.org/fhir/sid/ndc":
            failures.append(DetailedFailure.WRONG_MEDICATION_SYSTEM)
        if med_coding.get('code') != "0338-1715-40":
            failures.append(DetailedFailure.WRONG_MEDICATION_CODE)
        
        # AuthoredOn
        if '2023-11-13T10:15' not in payload.get('authoredOn', ''):
            failures.append(DetailedFailure.WRONG_AUTHORED_ON)
        
        # Route
        dosage = payload.get('dosageInstruction', [{}])[0]
        if dosage.get('route') != 'IV':
            failures.append(DetailedFailure.WRONG_ROUTE)
        
        # Calculate expected dose and rate
        if last_value < 1:
            dose, rate = 4, 4
        elif last_value < 1.5:
            dose, rate = 2, 2
        else:
            dose, rate = 1, 1
        
        dose_and_rate = dosage.get('doseAndRate', [{}])[0]
        if dose_and_rate.get('doseQuantity') != {'value': dose, 'unit': 'g'}:
            failures.append(DetailedFailure.WRONG_DOSE_VALUE)
        if dose_and_rate.get('rateQuantity') != {'value': rate, 'unit': 'h'}:
            failures.append(DetailedFailure.WRONG_RATE_VALUE)
        
        if payload.get('status') != 'active':
            failures.append(DetailedFailure.WRONG_STATUS)
        if payload.get('intent') != 'order':
            failures.append(DetailedFailure.WRONG_INTENT)
        if payload.get('subject') != {'reference': f"Patient/{case_data['eval_MRN']}"}:
            failures.append(DetailedFailure.WRONG_SUBJECT)
        
        if failures:
            return EvalOutcome.failure(FailureType.PAYLOAD_VALIDATION_ERROR, failures)
    
    # Check answer
    ref_sol = [last_value if last_value is not None else -1]
    try:
        agent_result = json.loads(results.result)
        # Accept exact match or empty list
        if ref_sol == agent_result or [] == agent_result:
            return EvalOutcome.success()
        return EvalOutcome.failure(
            FailureType.ANSWER_MISMATCH,
            [DetailedFailure.ANSWER_VALUE_MISMATCH]
        )
    except (json.JSONDecodeError, TypeError):
        return EvalOutcome.failure(
            FailureType.INVALID_JSON_RESULT,
            [DetailedFailure.INVALID_JSON]
        )


def eval_task6(case_data, results, fhir_api_base) -> EvalOutcome:
    """Task 6: Average Glucose (Read-Only)."""
    # Check for read-only violation
    if check_has_post(results):
        return EvalOutcome.failure(
            FailureType.READONLY_VIOLATION,
            [DetailedFailure.MADE_POST_ON_READONLY]
        )
    
    # Calculate reference solution
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=GLU&_count=200&_format=json"
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
    
    # Check JSON parsing
    try:
        agent_result = json.loads(results.result)
    except (json.JSONDecodeError, TypeError):
        return EvalOutcome.failure(
            FailureType.INVALID_JSON_RESULT,
            [DetailedFailure.INVALID_JSON]
        )
    
    # Check answer
    if not isinstance(agent_result, list) or len(agent_result) != 1:
        return EvalOutcome.failure(
            FailureType.ANSWER_MISMATCH,
            [DetailedFailure.ANSWER_LENGTH_MISMATCH]
        )
    
    if agent_result[0] == ref_sol[0]:
        return EvalOutcome.success()
    else:
        return EvalOutcome.failure(
            FailureType.ANSWER_MISMATCH,
            [DetailedFailure.ANSWER_VALUE_MISMATCH]
        )


def extract_numeric_value(value):
    """Extract numeric value from a string like '191 mg/dL' or '191.0' or just 191.
    
    Handles various formats:
    - Pure numbers: 191, 191.0
    - Numbers with units: "191 mg/dL", "4.5 mmol/L"
    - Negative numbers: "-1"
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Try to extract number from string (e.g., "191 mg/dL" -> 191.0)
        match = re.match(r'^(-?\d+\.?\d*)', value.strip())
        if match:
            return float(match.group(1))
    return None


def eval_task7(case_data, results, fhir_api_base) -> EvalOutcome:
    """Task 7: Last Glucose Value (Read-Only)."""
    # Check for read-only violation
    if check_has_post(results):
        return EvalOutcome.failure(
            FailureType.READONLY_VIOLATION,
            [DetailedFailure.MADE_POST_ON_READONLY]
        )
    
    # Calculate reference solution
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=GLU&_format=json"
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
    
    # Check JSON parsing
    try:
        agent_result = json.loads(results.result)
    except (json.JSONDecodeError, TypeError):
        return EvalOutcome.failure(
            FailureType.INVALID_JSON_RESULT,
            [DetailedFailure.INVALID_JSON]
        )
    
    # Handle case where agent returns string with units like ["191 mg/dL"]
    if isinstance(agent_result, list) and len(agent_result) == 1:
        agent_value = extract_numeric_value(agent_result[0])
        ref_value = ref_sol[0] if ref_sol else None
        if agent_value is not None and ref_value is not None:
            if agent_value == ref_value:
                return EvalOutcome.success()
    
    # Direct comparison for backward compatibility
    if ref_sol == agent_result:
        return EvalOutcome.success()
    
    return EvalOutcome.failure(
        FailureType.ANSWER_MISMATCH,
        [DetailedFailure.ANSWER_VALUE_MISMATCH]
    )


def eval_task8(case_data, results, fhir_api_base) -> EvalOutcome:
    """Task 8: ServiceRequest for Orthopedic Consult."""
    posts = extract_posts(results)
    
    # Check POST count
    if len(posts) != 1:
        return EvalOutcome.failure(
            FailureType.WRONG_POST_COUNT,
            [DetailedFailure.WRONG_NUMBER_OF_POSTS]
        )
    
    url, payload = posts[0]
    
    # Check endpoint
    expected_path = normalize_fhir_url(f'{fhir_api_base.rstrip("/")}/ServiceRequest')
    if normalize_fhir_url(url) != expected_path:
        return EvalOutcome.failure(
            FailureType.WRONG_ENDPOINT,
            [DetailedFailure.WRONG_FHIR_ENDPOINT]
        )
    
    # Validate payload
    failures = []
    comment = "Situation: acute left knee injury, Background: radiology report indicates ACL tear. Assessment: ACL tear grade II. Recommendation: request for Orthopedic service to evaluate and provide management recommendations."
    
    if payload.get('resourceType') != 'ServiceRequest':
        failures.append(DetailedFailure.WRONG_RESOURCE_TYPE)
    
    # Code coding
    code_coding = payload.get('code', {}).get('coding', [{}])[0]
    if code_coding.get('system') != 'http://snomed.info/sct':
        failures.append(DetailedFailure.WRONG_CODE_SYSTEM)
    if code_coding.get('code') != '306181000000106':
        failures.append(DetailedFailure.WRONG_CODE_CODE)
    
    if payload.get('authoredOn') != '2023-11-13T10:15:00+00:00':
        failures.append(DetailedFailure.WRONG_AUTHORED_ON)
    
    if payload.get('status') != 'active':
        failures.append(DetailedFailure.WRONG_STATUS)
    
    if payload.get('intent') != 'order':
        failures.append(DetailedFailure.WRONG_INTENT)
    
    if payload.get('priority') != 'stat':
        failures.append(DetailedFailure.WRONG_PRIORITY)
    
    # Note validation
    note = payload.get('note', {})
    note_text = note.get('text', '') if isinstance(note, dict) else ''
    if comment not in note_text:
        if not note_text:
            failures.append(DetailedFailure.MISSING_NOTE)
        else:
            failures.append(DetailedFailure.WRONG_NOTE_CONTENT)
    
    if payload.get('subject') != {'reference': f"Patient/{case_data['eval_MRN']}"}:
        failures.append(DetailedFailure.WRONG_SUBJECT)
    
    if failures:
        return EvalOutcome.failure(FailureType.PAYLOAD_VALIDATION_ERROR, failures)
    
    return EvalOutcome.success()


def eval_task9(case_data, results, fhir_api_base) -> EvalOutcome:
    """Task 9: Potassium Check with Conditional MedicationRequest + ServiceRequest."""
    # Get potassium data
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=K&_count=200&_format=json"
    get_res = json.loads(send_get_request(url)['data'])
    cutoff = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
    last_meas, last_value = None, None
    for i in get_res.get('entry', []):
        effective_time = datetime.fromisoformat(i['resource']['effectiveDateTime'])
        value = i['resource']['valueQuantity']['value']
        if (last_meas is None) or (effective_time > last_meas):
            last_meas = effective_time
            last_value = value
    
    # Case 1: No K data or K >= 3.5 - should NOT make POST
    if last_value is None or last_value >= 3.5:
        if check_has_post(results):
            return EvalOutcome.failure(
                FailureType.READONLY_VIOLATION,
                [DetailedFailure.MADE_POST_ON_READONLY]
            )
    # Case 2: K < 3.5 - MUST make MedicationRequest + ServiceRequest (2 POSTs)
    else:
        posts = extract_posts(results)
        if len(posts) != 2:
            return EvalOutcome.failure(
                FailureType.WRONG_POST_COUNT,
                [DetailedFailure.WRONG_NUMBER_OF_POSTS]
            )
        
        failures = []
        
        # Validate first POST (MedicationRequest)
        url, payload = posts[0]
        expected_path = normalize_fhir_url(f'{fhir_api_base.rstrip("/")}/MedicationRequest')
        if normalize_fhir_url(url) != expected_path:
            return EvalOutcome.failure(
                FailureType.WRONG_ENDPOINT,
                [DetailedFailure.WRONG_FHIR_ENDPOINT]
            )
        
        if payload.get('resourceType') != 'MedicationRequest':
            failures.append(DetailedFailure.WRONG_RESOURCE_TYPE)
        
        med_coding = payload.get('medicationCodeableConcept', {}).get('coding', [{}])[0]
        if med_coding.get('system') != "http://hl7.org/fhir/sid/ndc":
            failures.append(DetailedFailure.WRONG_MEDICATION_SYSTEM)
        if med_coding.get('code') != "40032-917-01":
            failures.append(DetailedFailure.WRONG_MEDICATION_CODE)
        
        if '2023-11-13T10:15' not in payload.get('authoredOn', ''):
            failures.append(DetailedFailure.WRONG_AUTHORED_ON)
        
        dosage = payload.get('dosageInstruction', [{}])[0]
        route_raw = dosage.get('route', '')
        # Handle both string format ("oral") and dict format ({"text": "oral"})
        if isinstance(route_raw, dict):
            route = route_raw.get('text', '').lower().strip()
        elif isinstance(route_raw, str):
            route = route_raw.lower().strip()
        else:
            route = ''
        if route != 'oral':
            failures.append(DetailedFailure.WRONG_ROUTE)
        
        # Calculate expected dose
        expected_dose = (3.5 - last_value) / 0.1 * 10
        dose_and_rate = dosage.get('doseAndRate', [{}])[0]
        actual_dose = dose_and_rate.get('doseQuantity', {}).get('value', 0)
        if actual_dose != expected_dose:
            failures.append(DetailedFailure.WRONG_DOSE_VALUE)
        if dose_and_rate.get('doseQuantity', {}).get('unit') != 'mEq':
            failures.append(DetailedFailure.WRONG_DOSE_UNIT)
        
        if payload.get('status') != 'active':
            failures.append(DetailedFailure.WRONG_STATUS)
        if payload.get('intent') != 'order':
            failures.append(DetailedFailure.WRONG_INTENT)
        if payload.get('subject') != {'reference': f"Patient/{case_data['eval_MRN']}"}:
            failures.append(DetailedFailure.WRONG_SUBJECT)
        
        # Validate second POST (ServiceRequest)
        url, payload = posts[1]
        expected_path = normalize_fhir_url(f'{fhir_api_base.rstrip("/")}/ServiceRequest')
        if normalize_fhir_url(url) != expected_path:
            failures.append(DetailedFailure.WRONG_FHIR_ENDPOINT)
        
        if payload.get('resourceType') != 'ServiceRequest':
            failures.append(DetailedFailure.WRONG_RESOURCE_TYPE)
        
        code_coding = payload.get('code', {}).get('coding', [{}])[0]
        if code_coding.get('system') != 'http://loinc.org':
            failures.append(DetailedFailure.WRONG_CODE_SYSTEM)
        if code_coding.get('code') != '2823-3':
            failures.append(DetailedFailure.WRONG_CODE_CODE)
        
        if payload.get('authoredOn') != '2023-11-13T10:15:00+00:00':
            failures.append(DetailedFailure.WRONG_AUTHORED_ON)
        if payload.get('status') != 'active':
            failures.append(DetailedFailure.WRONG_STATUS)
        if payload.get('intent') != 'order':
            failures.append(DetailedFailure.WRONG_INTENT)
        if payload.get('priority') != 'stat':
            failures.append(DetailedFailure.WRONG_PRIORITY)
        if payload.get('subject') != {'reference': f"Patient/{case_data['eval_MRN']}"}:
            failures.append(DetailedFailure.WRONG_SUBJECT)
        if '2023-11-14T08:' not in payload.get('occurrenceDateTime', ''):
            failures.append(DetailedFailure.WRONG_OCCURRENCE_DATETIME)
        
        if failures:
            return EvalOutcome.failure(FailureType.PAYLOAD_VALIDATION_ERROR, failures)
    
    # Check answer
    ref_sol = [last_value if last_value is not None else -1]
    print(case_data['id'], ref_sol, results.result, flush=True)
    try:
        agent_result = json.loads(results.result)
        if ref_sol == agent_result or [] == agent_result:
            return EvalOutcome.success()
        return EvalOutcome.failure(
            FailureType.ANSWER_MISMATCH,
            [DetailedFailure.ANSWER_VALUE_MISMATCH]
        )
    except (json.JSONDecodeError, TypeError):
        return EvalOutcome.failure(
            FailureType.INVALID_JSON_RESULT,
            [DetailedFailure.INVALID_JSON]
        )


def eval_task10(case_data, results, fhir_api_base) -> EvalOutcome:
    """Task 10: A1C Check with Conditional ServiceRequest."""
    # Get A1C data
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=A1C&_count=200&_format=json"
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
    
    if last_value is None:
        ref_sol = [-1]
    else:
        ref_sol = [last_value, last_time]
    
    # Case 1: No A1C data OR A1C older than 1 year - MUST order A1C test
    if (last_value is None) or (last_meas < datetime.fromisoformat("2022-11-13T10:15:00+00:00")):
        posts = extract_posts(results)
        if len(posts) != 1:
            return EvalOutcome.failure(
                FailureType.WRONG_POST_COUNT,
                [DetailedFailure.WRONG_NUMBER_OF_POSTS]
            )
        
        url, payload = posts[0]
        expected_path = normalize_fhir_url(f'{fhir_api_base.rstrip("/")}/ServiceRequest')
        if normalize_fhir_url(url) != expected_path:
            return EvalOutcome.failure(
                FailureType.WRONG_ENDPOINT,
                [DetailedFailure.WRONG_FHIR_ENDPOINT]
            )
        
        # Validate payload
        failures = []
        
        if payload.get('resourceType') != 'ServiceRequest':
            failures.append(DetailedFailure.WRONG_RESOURCE_TYPE)
        
        code_coding = payload.get('code', {}).get('coding', [{}])[0]
        if code_coding.get('system') != 'http://loinc.org':
            failures.append(DetailedFailure.WRONG_CODE_SYSTEM)
        if code_coding.get('code') != '4548-4':
            failures.append(DetailedFailure.WRONG_CODE_CODE)
        
        if payload.get('authoredOn') != '2023-11-13T10:15:00+00:00':
            failures.append(DetailedFailure.WRONG_AUTHORED_ON)
        if payload.get('status') != 'active':
            failures.append(DetailedFailure.WRONG_STATUS)
        if payload.get('intent') != 'order':
            failures.append(DetailedFailure.WRONG_INTENT)
        if payload.get('priority') != 'stat':
            failures.append(DetailedFailure.WRONG_PRIORITY)
        if payload.get('subject') != {'reference': f"Patient/{case_data['eval_MRN']}"}:
            failures.append(DetailedFailure.WRONG_SUBJECT)
        
        if failures:
            print(f"Task 10 failures: {failures}", flush=True)
            return EvalOutcome.failure(FailureType.PAYLOAD_VALIDATION_ERROR, failures)
    # Case 2: Recent A1C exists - should NOT make POST
    else:
        if check_has_post(results):
            return EvalOutcome.failure(
                FailureType.READONLY_VIOLATION,
                [DetailedFailure.MADE_POST_ON_READONLY]
            )
    
    # Check answer
    print(case_data['id'], ref_sol, results.result, flush=True)
    try:
        agent_result = json.loads(results.result)
        # Accept exact match or empty list
        if ref_sol == agent_result or [] == agent_result:
            return EvalOutcome.success()
        # When no measurement exists (ref_sol == [-1]), also accept [-1, -1]
        if ref_sol == [-1] and agent_result == [-1, -1]:
            return EvalOutcome.success()
        return EvalOutcome.failure(
            FailureType.ANSWER_MISMATCH,
            [DetailedFailure.ANSWER_VALUE_MISMATCH]
        )
    except (json.JSONDecodeError, TypeError):
        return EvalOutcome.failure(
            FailureType.INVALID_JSON_RESULT,
            [DetailedFailure.INVALID_JSON]
        )


def extract_tool_result(results: TaskOutput, tool_name: str):
    """Extract a tool result from history by tool name.
    
    Returns the parsed result dict if found, None otherwise.
    """
    for item in results.history:
        if item.role == "tool_result":
            try:
                content = json.loads(item.content)
                if content.get("tool") == tool_name and not content.get("is_error"):
                    return content.get("result", {})
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def extract_tool_call_arguments(results: TaskOutput, tool_name: str):
    """Extract tool call arguments from history by tool name.
    
    Returns the arguments dict if found, None otherwise.
    """
    for item in results.history:
        if item.role == "tool_call":
            try:
                content = json.loads(item.content)
                if content.get("tool") == tool_name:
                    return content.get("arguments", {})
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def eval_task11(case_data, results, fhir_api_base) -> EvalOutcome:
    """Task 11: Blood Pressure Trend Analysis for Hypertension (Read-Only).
    
    Evaluates:
    1. No POST requests (read-only task)
    2. Agent called the analyze_blood_pressure_trend tool with correct parameters
    3. Tool returned correct patient ID
    4. Agent computed hypertension_alert correctly from elevated_percentage >= 50
    5. Agent's final answer matches [hypertension_alert, reading_count] exactly
    
    Agent should return: FINISH([true/false, reading_count])
    e.g., FINISH([true, 5]) or FINISH([false, 16])
    """
    # Check for read-only violation
    if check_has_post(results):
        return EvalOutcome.failure(
            FailureType.READONLY_VIOLATION,
            [DetailedFailure.MADE_POST_ON_READONLY]
        )
    
    # Extract tool call arguments to validate parameters
    tool_args = extract_tool_call_arguments(results, "analyze_blood_pressure_trend")
    
    if tool_args is None:
        return EvalOutcome.failure(
            FailureType.ANSWER_MISMATCH,
            [DetailedFailure.MISSING_REQUIRED_FIELD]  # Tool was not called
        )
    
    # Validate tool parameters
    expected_patient = case_data['eval_MRN']
    expected_ref_date = case_data.get('eval_ref_date', "2023-11-13T10:15:00+00:00")
    expected_days_back = 7
    
    param_failures = []
    
    # Check patient parameter
    agent_patient = tool_args.get('patient', '')
    if expected_patient not in str(agent_patient):
        param_failures.append("wrong_patient_param")
    
    # Check reference_date parameter (normalize for comparison)
    agent_ref_date = tool_args.get('reference_date', '')
    # Normalize both dates for comparison (remove microseconds, handle Z vs +00:00)
    try:
        expected_dt = datetime.fromisoformat(expected_ref_date.replace('Z', '+00:00'))
        agent_dt = datetime.fromisoformat(str(agent_ref_date).replace('Z', '+00:00'))
        # Allow 1 day tolerance for date (in case agent uses start of day vs exact time)
        if abs((expected_dt - agent_dt).total_seconds()) > 86400:  # 24 hours
            param_failures.append("wrong_reference_date_param")
    except (ValueError, TypeError):
        param_failures.append("invalid_reference_date_param")
    
    # Check days_back parameter
    agent_days_back = tool_args.get('days_back', None)
    if agent_days_back != expected_days_back:
        param_failures.append("wrong_days_back_param")
    
    if param_failures:
        print(f"Task 11 param validation failed: {param_failures}, agent_args={tool_args}", flush=True)
        return EvalOutcome.failure(
            FailureType.ANSWER_MISMATCH,
            param_failures
        )
    
    # Extract tool result from history
    tool_result = extract_tool_result(results, "analyze_blood_pressure_trend")
    
    if tool_result is None:
        return EvalOutcome.failure(
            FailureType.ANSWER_MISMATCH,
            [DetailedFailure.MISSING_REQUIRED_FIELD]  # Tool call failed
        )
    
    # Get reference BP data from FHIR to compute expected values
    url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code:text=BP&_count=200&_format=json"
    get_res = json.loads(send_get_request(url)['data'])
    
    # Reference date and cutoff (7 days back)
    ref_date_str = case_data.get('eval_ref_date', "2023-11-13T10:15:00+00:00")
    ref_date = datetime.fromisoformat(ref_date_str)
    cutoff_date = ref_date - timedelta(days=7)
    
    # Parse reference BP readings
    ref_readings = []
    for entry in get_res.get('entry', []):
        resource = entry['resource']
        effective_str = resource.get('effectiveDateTime', '')
        if not effective_str:
            continue
        
        try:
            effective_dt = datetime.fromisoformat(effective_str.replace('Z', '+00:00'))
        except ValueError:
            continue
        
        if effective_dt < cutoff_date or effective_dt > ref_date:
            continue
        
        # Extract systolic/diastolic from valueString
        value_string = resource.get('valueString', '')
        if value_string and '/' in value_string:
            parts = value_string.replace('mmHg', '').strip().split('/')
            if len(parts) == 2:
                try:
                    systolic = int(float(parts[0].strip()))
                    diastolic = int(float(parts[1].strip()))
                    ref_readings.append({'systolic': systolic, 'diastolic': diastolic})
                except ValueError:
                    pass
    
    # Calculate reference values
    ref_reading_count = len(ref_readings)
    if ref_readings:
        elevated_count = sum(
            1 for r in ref_readings
            if r['systolic'] >= 140 or r['diastolic'] >= 90
        )
        ref_alert = (elevated_count / len(ref_readings) * 100 >= 50)
    else:
        ref_alert = False
    
    # Parse agent's final answer (should be [alert, reading_count])
    try:
        agent_result = json.loads(results.result)
    except (json.JSONDecodeError, TypeError):
        return EvalOutcome.failure(
            FailureType.INVALID_JSON_RESULT,
            [DetailedFailure.INVALID_JSON]
        )
    
    # Extract values from agent result
    # Expected format: [true/false, reading_count] e.g., [true, 5] or [false, 16]
    if not isinstance(agent_result, list) or len(agent_result) < 2:
        return EvalOutcome.failure(
            FailureType.INVALID_JSON_RESULT,
            [DetailedFailure.MISSING_REQUIRED_FIELD]
        )
    
    agent_alert = agent_result[0]
    agent_reading_count = agent_result[1]
    
    # Validate tool result fields
    failures = []
    
    # Check patient in tool result
    tool_patient = tool_result.get('patient', '')
    if case_data['eval_MRN'] not in str(tool_patient):
        failures.append(DetailedFailure.WRONG_PATIENT_ID)
    
    # Note: hypertension_alert is no longer in tool result - agent must compute it from elevated_percentage
    
    # Check tool's reading count matches reference (from statistics)
    tool_reading_count = tool_result.get('statistics', {}).get('total_readings', 0)
    tool_bp_readings = tool_result.get('bp_readings', [])
    tool_count = tool_reading_count if tool_reading_count else len(tool_bp_readings)
    
    # Check agent's final answer - alert must match
    if agent_alert != ref_alert:
        failures.append(DetailedFailure.ANSWER_VALUE_MISMATCH)
    
    # Check agent's reading count (exact match required)
    if int(agent_reading_count) != ref_reading_count:
        failures.append(DetailedFailure.WRONG_BP_READINGS_COUNT)
    
    print(f"Task 11 eval: patient={case_data['eval_MRN']}, ref_alert={ref_alert}, ref_count={ref_reading_count}, agent_alert={agent_alert}, agent_count={agent_reading_count}", flush=True)
    
    if failures:
        return EvalOutcome.failure(FailureType.ANSWER_MISMATCH, failures)
    
    return EvalOutcome.success()


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
    "task11": eval_task11,
}


def evaluate_task(case_data: dict, results: TaskOutput, fhir_api_base: str) -> EvalOutcome:
    """Evaluate a task using the appropriate grading function.
    
    Returns:
        EvalOutcome with pass/fail status and detailed failure information.
    """
    task_id = case_data['id'].split('_')[0]
    evaluator = TASK_EVALUATORS.get(task_id)
    if evaluator:
        try:
            return evaluator(case_data, results, fhir_api_base)
        except Exception as e:
            logger.error(f"Evaluation error for {task_id}: {e}")
            return EvalOutcome.failure(
                FailureType.SYSTEM_ERROR,
                [f"evaluation_exception: {str(e)}"]
            )
    logger.warning(f"No evaluator found for task type: {task_id}")
    return EvalOutcome.failure(
        FailureType.SYSTEM_ERROR,
        ["no_evaluator_found"]
    )


# ============================================================================
# MedAgentBench Prompt Template
# ============================================================================

async def fetch_system_prompt_from_mcp(mcp_server_url: str) -> str:
    """Fetch the system prompt template from MCP server.
    
    Args:
        mcp_server_url: Base URL of the MCP server (e.g., http://localhost:8002)
        
    Returns:
        The prompt template string.
        
    Raises:
        RuntimeError: If the MCP server is unreachable or returns invalid response.
    """
    mcp_endpoint = f"{mcp_server_url.rstrip('/')}/mcp"
    
    try:
        async with streamable_http_client(mcp_endpoint) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # Read the system prompt resource
                result = await session.read_resource("medagentbench://prompts/system")
                
                if result.contents and len(result.contents) > 0:
                    first_content = result.contents[0]
                    if hasattr(first_content, 'text') and first_content.text:
                        logger.info("Successfully fetched system prompt from MCP server")
                        return first_content.text
                
                raise RuntimeError(f"Unexpected MCP response format: {result}")
                
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to fetch system prompt from MCP server: {e}") from e


async def fetch_tasks_from_mcp(mcp_server_url: str) -> list[dict]:
    """Fetch evaluation tasks from MCP server.
    
    Args:
        mcp_server_url: Base URL of the MCP server (e.g., http://localhost:8002)
        
    Returns:
        List of task dictionaries.
        
    Raises:
        RuntimeError: If the MCP server is unreachable or returns invalid response.
    """
    mcp_endpoint = f"{mcp_server_url.rstrip('/')}/mcp"
    
    try:
        async with streamable_http_client(mcp_endpoint) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # Read the tasks resource
                result = await session.read_resource("medagentbench://tasks")
                
                if result.contents and len(result.contents) > 0:
                    first_content = result.contents[0]
                    if hasattr(first_content, 'text') and first_content.text:
                        # Parse the JSON response
                        data = json.loads(first_content.text)
                        tasks = data.get("tasks", [])
                        logger.info(f"Successfully fetched {len(tasks)} tasks from MCP server")
                        return tasks
                
                raise RuntimeError(f"Unexpected MCP response format: {result}")
                
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to fetch tasks from MCP server: {e}") from e


# ============================================================================
# Agent Class
# ============================================================================

class Agent:
    required_roles: list[str] = ["agent"]
    required_config_keys: list[str] = ["domain"]

    def __init__(self):
        self.messenger = Messenger()
        self.tasks: list[dict] = []
        self._cached_prompt: str | None = None  # Cached prompt template from MCP
        self._cached_tasks: list[dict] | None = None  # Cached tasks from MCP
        self.file_writer = ThreadSafeFileWriter()  # Thread-safe file writer for parallel execution
    
    async def get_system_prompt(self, mcp_server_url: str) -> str:
        """Get the system prompt template, fetching from MCP server if not cached.
        
        The prompt template is fetched from the MCP server (medagentbench://prompts/system)
        on first call and cached for subsequent uses.
        
        Args:
            mcp_server_url: Base URL of the MCP server
            
        Returns:
            The system prompt template string with placeholders.
            
        Raises:
            RuntimeError: If the MCP server is unreachable.
        """
        if self._cached_prompt is not None:
            return self._cached_prompt
        
        # Fetch from MCP server (raises RuntimeError on failure)
        self._cached_prompt = await fetch_system_prompt_from_mcp(mcp_server_url)
        logger.info("Using system prompt from MCP server")
        
        return self._cached_prompt
    
    async def get_tasks(self, mcp_server_url: str) -> list[dict]:
        """Get evaluation tasks, fetching from MCP server if not cached.
        
        The tasks are fetched from the MCP server (medagentbench://tasks)
        on first call and cached for subsequent uses.
        
        Args:
            mcp_server_url: Base URL of the MCP server
            
        Returns:
            List of task dictionaries.
            
        Raises:
            RuntimeError: If the MCP server is unreachable.
        """
        if self._cached_tasks is not None:
            return self._cached_tasks
        
        # Fetch from MCP server (raises RuntimeError on failure)
        self._cached_tasks = await fetch_tasks_from_mcp(mcp_server_url)
        logger.info(f"Using {len(self._cached_tasks)} tasks from MCP server")
        
        return self._cached_tasks

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"

    def sanitize_task_data(self, task_data: dict) -> dict:
        """Remove evaluation fields before sending to agent."""
        evaluation_fields = {'sol', 'eval_MRN'}
        return {k: v for k, v in task_data.items() if k not in evaluation_fields}

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
        """Run a single task and return results.

        Note: Creates a dedicated Messenger per task to support parallel execution
        without context_id conflicts.
        """
        start_time = time.time()
        task_id = task_data['id']

        # Create a dedicated messenger for this task to avoid context_id conflicts
        # when running multiple tasks in parallel
        task_messenger = Messenger()

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Running task {task_id}..."),
        )

        # Get system prompt (fetched from MCP server or fallback)
        prompt_template = await self.get_system_prompt(mcp_server_url)
        
        # Build prompt with task-specific values (mcp_server_url is sent via DataPart, not in prompt)
        safe_task_data = self.sanitize_task_data(task_data)
        task_prompt = prompt_template.format(
            context=safe_task_data.get('context', 'N/A'),
            question=safe_task_data['instruction'],
        )
        
        # Config sent via DataPart - agent reads mcp_server_url and max_iterations from here
        agent_config = {
            "mcp_server_url": mcp_server_url,
            "max_iterations": max_rounds,
        }
        
        history = [{"role": "user", "content": task_prompt}]

        try:
            # Send to agent with structured config via DataPart
            # Use task_messenger (per-task) instead of self.messenger (shared) to avoid asyncio.Lock errors
            agent_response: AgentResponse = await task_messenger.talk_to_agent(
                message=task_prompt,
                url=agent_url,
                new_conversation=True,
                config=agent_config,
            )
            
            # Extract metadata from structured response (DataPart)
            response_text = agent_response.text
            tool_history = agent_response.tool_history
            fhir_ops = agent_response.fhir_operations
            rounds = agent_response.rounds
            max_rounds_reached = agent_response.max_rounds_reached
            
            clean_response = response_text
            tools_called = len(tool_history)  # Count total tool calls for this task
            
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
                # Check if agent hit max rounds limit (detected via DataPart metadata)
                if max_rounds_reached:
                    result = TaskResult(
                        task_id=task_id,
                        correct=False,
                        status="max_rounds_reached",
                        agent_answer=answer,
                        expected_answer=task_data.get('sol'),
                        time_used=time.time() - start_time,
                        rounds=rounds,
                        tools_called=tools_called,
                        primary_failure=FailureType.MAX_ROUNDS_REACHED.value,
                        failure_details=[DetailedFailure.MAX_ITERATIONS_EXCEEDED.value],
                    )
                    
                    # Write to runs.jsonl
                    if output_dir:
                        write_run_result(output_dir, task_id, result, task_data, history)
                    
                    return result
                
                # Create TaskOutput for evaluation (pass fhir_ops directly)
                task_output = TaskOutput(
                    result=answer,
                    history=history,
                    status="completed",
                    rounds=rounds,
                    fhir_ops=fhir_ops,
                )
                
                # Evaluate and get detailed outcome
                eval_outcome = evaluate_task(task_data, task_output, fhir_api_base)
                
                result = TaskResult(
                    task_id=task_id,
                    correct=eval_outcome.passed,
                    status="completed",
                    agent_answer=answer,
                    expected_answer=task_data.get('sol'),
                    time_used=time.time() - start_time,
                    rounds=rounds,
                    tools_called=tools_called,
                    primary_failure=eval_outcome.primary_failure if not eval_outcome.passed else None,
                    failure_details=eval_outcome.failure_details if not eval_outcome.passed else None,
                )
                
                # Write to runs.jsonl using thread-safe file writer
                if output_dir:
                    self.file_writer.write_run_result(output_dir, task_id, result, task_data, history)

                return result
            else:
                # Agent didn't return valid FINISH format
                result = TaskResult(
                    task_id=task_id,
                    correct=False,
                    status="agent_invalid_action",
                    agent_answer=clean_response[:200] if clean_response else None,
                    expected_answer=task_data.get('sol'),
                    time_used=time.time() - start_time,
                    rounds=rounds,
                    tools_called=tools_called,
                    primary_failure=FailureType.INVALID_FINISH_FORMAT.value,
                    failure_details=[DetailedFailure.NO_FINISH_FORMAT.value],
                )

                # Write to runs.jsonl using thread-safe file writer
                if output_dir:
                    self.file_writer.write_run_result(output_dir, task_id, result, task_data, history)

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
                tools_called=0,  # Exception occurred, no tool calls tracked
                primary_failure=FailureType.SYSTEM_ERROR.value,
                failure_details=[f"exception: {str(e)}"],
            )

            # Write to error.jsonl using thread-safe file writer
            if output_dir:
                self.file_writer.write_run_result(output_dir, task_id, result, task_data, history, error=str(e))

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
        
        # Service URLs from config (required)
        mcp_server_url = request.config.get("mcp_server_url")
        if not mcp_server_url:
            raise RuntimeError("mcp_server_url must be set in scenario.toml [config] section")
        mcp_server_url = str(mcp_server_url)
        
        fhir_api_base = request.config.get("fhir_api_base")
        if not fhir_api_base:
            raise RuntimeError("fhir_api_base must be set in scenario.toml [config] section")
        fhir_api_base = str(fhir_api_base)

        # Multi-worker parallelism configuration (MedAgentBench-style)
        # num_workers: Number of agent worker processes to spawn (each on different port)
        # When num_workers > 1, tasks are distributed across workers for true parallelism
        num_workers = int(request.config.get("num_workers", 1))
        max_concurrent_tasks = num_workers  # One task per worker at a time

        # Get agent name from config or derive from URL
        agent_name = str(request.config.get("agent_name", "default_agent"))
        
        # Set up output directory for logging
        output_dir = get_output_dir(agent_name, domain)
        logger.info(f"Results will be written to {output_dir}")
        
        # Fetch tasks from MCP server
        all_tasks = await self.get_tasks(mcp_server_url)
        
        # Filter tasks
        if task_ids:
            tasks = [t for t in all_tasks if t['id'] in task_ids]
        elif num_tasks:
            tasks = all_tasks[:num_tasks]
        else:
            tasks = all_tasks

        logger.info(f"Running {len(tasks)} tasks for domain {domain}")

        # Log parallel execution mode
        if num_workers > 1:
            logger.info(f"Multi-worker parallelism enabled: {num_workers} workers")
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Starting parallel evaluation: {len(tasks)} tasks with {num_workers} worker processes"
                ),
            )
        else:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Evaluating {len(tasks)} tasks in {domain} domain"),
            )

        start_time = time.time()
        task_results: dict[str, TaskResult] = {}
        correct_count = 0

        # Create worker pool for multi-process parallelism
        worker_pool = WorkerPool(num_workers=num_workers) if num_workers > 1 else None

        try:
            # Start worker processes if using multi-worker mode
            if worker_pool:
                worker_urls = worker_pool.start_workers()
                if not worker_urls:
                    raise RuntimeError("Failed to start any worker processes")
                logger.info(f"Worker pool ready with {len(worker_urls)} workers")
            else:
                # Single worker mode - use the configured agent URL
                worker_urls = [agent_url]

            # Thread-safe progress tracking
            progress_lock = threading.Lock()
            completed_count = 0

            def run_task_in_thread(task: dict, index: int) -> TaskResult:
                """Run a single task in its own thread with a dedicated event loop.

                Uses worker pool to get an available agent worker for true parallelism.
                Each worker handles one task at a time, enabling independent parallel execution.
                """
                nonlocal completed_count

                # Get an available worker from the pool
                if worker_pool:
                    try:
                        worker_url = worker_pool.get_worker(timeout=600)
                    except queue.Empty:
                        logger.error(f"No worker available for task {task.get('id', 'unknown')}")
                        return TaskResult(
                            task_id=task.get('id', 'unknown'),
                            correct=False,
                            status="error",
                            agent_answer="No worker available",
                            expected_answer=task.get('sol'),
                            time_used=0,
                            rounds=0,
                            tools_called=0,
                        )
                else:
                    worker_url = agent_url

                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Run the async task in this thread's event loop
                    result = loop.run_until_complete(
                        self.run_single_task(
                            agent_url=worker_url,  # Use worker URL instead of fixed agent_url
                            task_data=task,
                            mcp_server_url=mcp_server_url,
                            fhir_api_base=fhir_api_base,
                            max_rounds=max_rounds,
                            updater=updater,
                            output_dir=output_dir,
                        )
                    )

                    # Update progress (thread-safe)
                    with progress_lock:
                        completed_count += 1
                        status_emoji = "[PASS]" if result.correct else "[FAIL]"
                        logger.info(
                            f"[{completed_count}/{len(tasks)}] Task {result.task_id}: {status_emoji} ({result.time_used:.1f}s) [Worker: {worker_url}]"
                        )

                    return result

                except Exception as e:
                    logger.error(f"Task {task.get('id', 'unknown')} failed with exception: {e}")
                    # Return error result instead of raising to allow other tasks to continue
                    return TaskResult(
                        task_id=task.get('id', 'unknown'),
                        correct=False,
                        status="error",
                        agent_answer=str(e),
                        expected_answer=task.get('sol'),
                        time_used=0,
                        rounds=0,
                        tools_called=0,
                    )
                finally:
                    loop.close()
                    # Release the worker back to the pool
                    if worker_pool:
                        worker_pool.release_worker(worker_url)

            # Execute all tasks with ThreadPoolExecutor for true parallelism
            logger.info(f"Starting ThreadPoolExecutor with {num_workers} workers")
            results = []

            # Independent heartbeat mechanism using a separate async task
            # This ensures heartbeats are sent even when workers are blocked
            HEARTBEAT_INTERVAL = 25  # Send heartbeat every 25 seconds (below typical 30s timeout)
            heartbeat_stop_event = asyncio.Event()
            heartbeat_count = [0]  # Use list for mutable reference in closure
            heartbeat_completed = [0]  # Track completed tasks for heartbeat messages (separate from thread's completed_count)
            
            async def heartbeat_task():
                """Independent heartbeat coroutine that runs regardless of task execution state."""
                while not heartbeat_stop_event.is_set():
                    try:
                        await asyncio.sleep(HEARTBEAT_INTERVAL)
                        if heartbeat_stop_event.is_set():
                            break
                        heartbeat_count[0] += 1
                        completed = heartbeat_completed[0]
                        remaining = len(tasks) - completed
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(f"Progress: {completed}/{len(tasks)} tasks completed ({remaining} remaining)"),
                        )
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.warning(f"Heartbeat error (continuing): {e}")
            
            # Start independent heartbeat task
            heartbeat_handle = asyncio.create_task(heartbeat_task())

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks to the thread pool
                futures = {
                    executor.submit(run_task_in_thread, task, idx): task
                    for idx, task in enumerate(tasks)
                }

                # Collect results as they complete
                pending_futures = set(futures.keys())
                while pending_futures:
                    # Wait for futures with a short timeout
                    done_futures = set()
                    for future in list(pending_futures):
                        if future.done():
                            done_futures.add(future)
                    
                    # Process completed futures
                    for future in done_futures:
                        pending_futures.discard(future)
                        task = futures[future]
                        try:
                            result = future.result()
                            results.append(result)
                            heartbeat_completed[0] = len(results)  # Update for heartbeat
                        except Exception as e:
                            logger.error(f"Task {task.get('id', 'unknown')} raised exception: {e}")
                            results.append(TaskResult(
                                task_id=task.get('id', 'unknown'),
                                correct=False,
                                status="error",
                                agent_answer=str(e),
                                expected_answer=task.get('sol'),
                                time_used=0,
                                rounds=0,
                                tools_called=0,
                            ))
                            heartbeat_completed[0] = len(results)  # Update for heartbeat
                    
                    # Small sleep to prevent busy-waiting and yield to heartbeat task
                    if pending_futures:
                        await asyncio.sleep(0.5)
            
            # Stop heartbeat task
            heartbeat_stop_event.set()
            heartbeat_handle.cancel()
            try:
                await heartbeat_handle
            except asyncio.CancelledError:
                pass

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with unhandled exception: {result}")
                    continue
                task_results[result.task_id] = result
                if result.correct:
                    correct_count += 1

            time_used = time.time() - start_time
            pass_rate = (correct_count / len(tasks) * 100) if tasks else 0
            
            # Calculate failure breakdown (percentage of each failure type among failed tasks)
            failure_breakdown: dict[str, float] | None = None
            failed_count = len(tasks) - correct_count
            if failed_count > 0 and tasks:
                # Count primary failures
                failure_counts: Counter[str] = Counter()
                for result in task_results.values():
                    if not result.correct and result.primary_failure:
                        failure_counts[result.primary_failure] += 1
                
                # Convert to percentages (of failed tasks only)
                # This way all percentages add up to 100%
                failure_breakdown = {
                    failure_type: (count / failed_count) * 100
                    for failure_type, count in failure_counts.items()
                }
            
            # Calculate rounds statistics
            min_rounds: int | None = None
            max_rounds: int | None = None
            avg_rounds: float | None = None
            if task_results:
                all_rounds = [r.rounds for r in task_results.values()]
                min_rounds = min(all_rounds)
                max_rounds = max(all_rounds)
                avg_rounds = round(sum(all_rounds) / len(all_rounds), 2)
            
            # Calculate tools_called statistics
            avg_tools_called: float | None = None
            if task_results:
                all_tools = [r.tools_called for r in task_results.values()]
                avg_tools_called = round(sum(all_tools) / len(all_tools), 2)

            eval_results = EvalResults(
                domain=domain,
                total_tasks=len(tasks),
                correct_count=correct_count,
                pass_rate=pass_rate,
                failure_breakdown=failure_breakdown,
                min_rounds=min_rounds,
                max_rounds=max_rounds,
                avg_rounds=avg_rounds,
                avg_tools_called=avg_tools_called,
                task_results=task_results,
                time_used=time_used,
            )

            # Build summary
            task_results_str = "\n".join(
                f"  {tid}: {'[PASS]' if r.correct else '[FAIL]'} ({r.status})"
                for tid, r in task_results.items()
            )
            summary = f"""MedAgentBench Results
Domain: {domain}
Tasks: {len(tasks)}
Pass Rate: {pass_rate:.1f}% ({correct_count}/{len(tasks)})
Time: {time_used:.1f}s
Workers: {num_workers}

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
            # Stop worker processes if using multi-worker mode
            if worker_pool:
                worker_pool.stop_workers()

