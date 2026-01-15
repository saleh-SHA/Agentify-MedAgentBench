"""FastMCP server for MedAgentBench FHIR tools."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastmcp import FastMCP

# Configuration
FHIR_API_BASE = os.environ.get("MCP_FHIR_API_BASE", "https://medagentbench.ddns.net:8080/fhir/").rstrip("/")
TASKS_FILE = os.environ.get("MCP_TASKS_FILE", "src/mcp/resources/tasks/tasks.json")

# Create FastMCP server
mcp = FastMCP(
    name="MedAgentBench MCP Server",
    instructions="MCP server providing FHIR tools and MedAgentBench evaluation tasks for healthcare agent benchmarking.",
)


# Load tasks from JSON file
def _load_tasks() -> List[Dict[str, Any]]:
    """Load tasks from JSON file."""
    tasks_path = Path(TASKS_FILE)
    if not tasks_path.exists():
        print(f"Warning: Tasks file '{TASKS_FILE}' not found. No tasks will be available.")
        return []
    
    try:
        with open(tasks_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        print(f"Loaded {len(tasks)} tasks from {TASKS_FILE}")
        return tasks
    except Exception as e:
        print(f"Error loading tasks from '{TASKS_FILE}': {e}")
        return []


# Initialize tasks
_TASKS = _load_tasks()


# FHIR helper function
def _call_fhir(method: str, path: str, params: Optional[Dict] = None, body: Optional[Dict] = None) -> Dict[str, Any]:
    """Make a request to the FHIR server.
    
    For POST requests, returns additional 'fhir_post' field with details needed for evaluation:
    - fhir_url: Full URL of the FHIR endpoint
    - payload: The request body that was sent
    - accepted: Whether the request was successful
    """
    url = f"{FHIR_API_BASE}{path}"
    try:
        with httpx.Client(timeout=30.0) as client:
            if method == "GET":
                response = client.get(url, params=params)
            else:
                response = client.request(method, url, json=body)
            response.raise_for_status()
            
            result = {"url": url, "method": method, "status_code": response.status_code, "response": response.json()}
            
            # For POST requests, include details needed for evaluation
            if method == "POST" and body:
                result["fhir_post"] = {
                    "fhir_url": url,
                    "payload": body,
                    "accepted": response.status_code in (200, 201)
                }
            
            return result
    except httpx.HTTPError as e:
        error_detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail += f" - Response: {e.response.text}"
            except:
                pass
        return {"error": f"FHIR server error: {error_detail}"}


# Resources
@mcp.resource("medagentbench://tasks")
def get_medagentbench_tasks() -> str:
    """Complete list of MedAgentBench evaluation tasks with instructions and expected solutions."""
    return json.dumps({
        "name": "MedAgentBench Tasks",
        "description": f"Complete list of {len(_TASKS)} evaluation tasks.",
        "total_tasks": len(_TASKS),
        "tasks": _TASKS,
    }, indent=2)


# Tools
@mcp.tool()
def search_patients(
    identifier: Optional[str] = None,
    name: Optional[str] = None,
    family: Optional[str] = None,
    given: Optional[str] = None,
    birthdate: Optional[str] = None,
) -> Dict[str, Any]:
    """Search for Patient resources by demographics, identifiers, or contact information.
    
    Args:
        identifier: Unique identifier such as MRN.
        name: Any part of the patient's name.
        family: Family (last) name.
        given: Given (first/middle) name.
        birthdate: Birth date in YYYY-MM-DD format.
    """
    params = {}
    if identifier:
        params["identifier"] = identifier
    if name:
        params["name"] = name
    if family:
        params["family"] = family
    if given:
        params["given"] = given
    if birthdate:
        params["birthdate"] = birthdate
    return _call_fhir("GET", "/Patient", params=params)


@mcp.tool()
def list_patient_problems(
    patient: str,
    category: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve Condition resources from a patient's problem list.
    
    Args:
        patient: FHIR patient ID.
        category: Optional category filter (e.g., 'problem-list-item').
        status: Optional status (e.g., 'active', 'resolved').
    """
    params = {"patient": patient}
    if category:
        params["category"] = category
    if status:
        params["status"] = status
    return _call_fhir("GET", "/Condition", params=params)


@mcp.tool()
def list_lab_observations(
    patient: str,
    code: str,
    date: Optional[str] = None,
) -> Dict[str, Any]:
    """Return laboratory Observation resources for a patient.
    
    Args:
        patient: FHIR patient ID.
        code: Observation code (LOINC or local).
        date: Optional specimen collection date or range.
    """
    params = {"patient": patient, "code": code}
    if date:
        params["date"] = date
    return _call_fhir("GET", "/Observation", params=params)


@mcp.tool()
def list_vital_signs(
    patient: str,
    category: str,
    date: Optional[str] = None,
) -> Dict[str, Any]:
    """Return vital sign Observation resources for a patient.
    
    Args:
        patient: FHIR patient ID.
        category: Use 'vital-signs' to fetch vitals.
        date: Optional date or range when vitals were taken.
    """
    params = {"patient": patient, "category": category}
    if date:
        params["date"] = date
    return _call_fhir("GET", "/Observation", params=params)


@mcp.tool()
def record_vital_observation(
    resourceType: str,
    category: List[Dict[str, Any]],
    code: Dict[str, Any],
    effectiveDateTime: str,
    status: str,
    valueString: str,
    subject: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a vital sign Observation for a patient.
    
    Args:
        resourceType: Use "Observation".
        category: Observation categories with coding metadata.
        code: Flowsheet row / vital concept.
        effectiveDateTime: ISO timestamp of measurement.
        status: Use "final".
        valueString: Measurement value.
        subject: Reference to the patient resource.
    """
    body = {
        "resourceType": resourceType,
        "category": category,
        "code": code,
        "effectiveDateTime": effectiveDateTime,
        "status": status,
        "valueString": valueString,
        "subject": subject,
    }
    return _call_fhir("POST", "/Observation", body=body)


@mcp.tool()
def list_medication_requests(
    patient: str,
    category: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve MedicationRequest orders for a patient.
    
    Args:
        patient: FHIR patient ID.
        category: Optional: inpatient, outpatient, community, discharge.
        status: Optional status such as 'active'.
    """
    params = {"patient": patient}
    if category:
        params["category"] = category
    if status:
        params["status"] = status
    return _call_fhir("GET", "/MedicationRequest", params=params)


@mcp.tool()
def create_medication_request(
    resourceType: str,
    medicationCodeableConcept: Dict[str, Any],
    authoredOn: str,
    dosageInstruction: List[Dict[str, Any]],
    status: str,
    intent: str,
    subject: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a MedicationRequest order for a patient.
    
    Args:
        resourceType: Use "MedicationRequest".
        medicationCodeableConcept: Medication coding and display name.
        authoredOn: Date/time when prescription was authored.
        dosageInstruction: Dose, rate, and route instructions.
        status: Order status such as "active".
        intent: Order intent, typically "order".
        subject: Reference to the patient resource.
    """
    body = {
        "resourceType": resourceType,
        "medicationCodeableConcept": medicationCodeableConcept,
        "authoredOn": authoredOn,
        "dosageInstruction": dosageInstruction,
        "status": status,
        "intent": intent,
        "subject": subject,
    }
    return _call_fhir("POST", "/MedicationRequest", body=body)


@mcp.tool()
def list_patient_procedures(
    patient: str,
    date: str,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve completed Procedure resources for a patient.
    
    Args:
        patient: FHIR patient ID.
        date: Date or range when procedure occurred.
        code: Optional CPT or coded value.
    """
    params = {"patient": patient, "date": date}
    if code:
        params["code"] = code
    return _call_fhir("GET", "/Procedure", params=params)


@mcp.tool()
def create_service_request(
    resourceType: str,
    code: Dict[str, Any],
    authoredOn: str,
    status: str,
    intent: str,
    priority: str,
    subject: Dict[str, Any],
    occurrenceDateTime: Optional[str] = None,
    note: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a ServiceRequest order (labs, imaging, consults) for a patient.
    
    Args:
        resourceType: Use "ServiceRequest".
        code: Terminology coding describing the service.
        authoredOn: Timestamp when order was authored.
        status: Order status, typically "active".
        intent: Order intent, typically "order".
        priority: Priority such as 'stat'.
        subject: Reference to the patient.
        occurrenceDateTime: Optional desired time for service.
        note: Optional clinician instructions or comments.
    """
    body = {
        "resourceType": resourceType,
        "code": code,
        "authoredOn": authoredOn,
        "status": status,
        "intent": intent,
        "priority": priority,
        "subject": subject,
    }
    if occurrenceDateTime:
        body["occurrenceDateTime"] = occurrenceDateTime
    if note:
        body["note"] = note
    return _call_fhir("POST", "/ServiceRequest", body=body)


def main() -> None:
    """Entrypoint used by `python -m src.mcp.server`."""
    print(f"Starting MedAgentBench FastMCP Server")
    print(f"FHIR API base: {FHIR_API_BASE}")
    print(f"Tasks file: {TASKS_FILE}")
    print(f"Loaded {len(_TASKS)} tasks")
    # Run with SSE transport on port 8002 for HTTP access
    mcp.run(transport="sse", host="0.0.0.0", port=8002)


if __name__ == "__main__":
    main()
