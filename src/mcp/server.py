"""FastMCP server for MedAgentBench FHIR tools."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import httpx
from fastmcp import FastMCP
from pydantic import Field

# Configuration (defaults for local development)
FHIR_API_BASE = os.environ.get("MCP_FHIR_API_BASE", "http://localhost:8080/fhir/").rstrip("/")
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
            result = {"url": url, "method": method}
            if method == "GET":
                response = client.get(url, params=params)
                response.raise_for_status()
                result["status_code"] = response.status_code
                result["response"] = response.json()
            else:
                # We want to only log the request body (in order not to change the database state and reinitialize the server), so we don't use the client.request method
                # For POST requests, include details needed for evaluation
                result["status_code"] = 200
                result["response"] = "POST request accepted and executed successfully."
                result["fhir_post"] = {
                    "fhir_url": url,
                    "parameters": body,  # Use body, not params - POST tools pass data as body
                    "accepted": True
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
    identifier: Annotated[Optional[str], Field(description="The patient's identifier (e.g., MRN).")] = None,
    name: Annotated[Optional[str], Field(description="Any part of the patient's name. Ignored when family or given are used.")] = None,
    family: Annotated[Optional[str], Field(description="The patient's family (last) name.")] = None,
    given: Annotated[Optional[str], Field(description="The patient's given name. May include first and middle names.")] = None,
    birthdate: Annotated[Optional[str], Field(description="The patient's date of birth in YYYY-MM-DD format.")] = None,
) -> Dict[str, Any]:
    """Search for Patient resources by demographics, identifiers, or contact information."""
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
    patient: Annotated[str, Field(description="The patient's FHIR resource ID.")],
    category: Annotated[Optional[str], Field(description="Always 'problem-list-item' for this API.")] = None,
    status: Annotated[Optional[str], Field(description="Problem status filter (e.g., 'active', 'resolved').")] = None,
) -> Dict[str, Any]:
    """Retrieve Condition resources from a patient's problem list. Returns data from the patient's problem list across all encounters."""
    params = {"patient": patient}
    if category:
        params["category"] = category
    if status:
        params["status"] = status
    return _call_fhir("GET", "/Condition", params=params)


@mcp.tool()
def list_lab_observations(
    patient: Annotated[str, Field(description="The patient's FHIR resource ID.")],
    code: Annotated[str, Field(description="The observation identifier (base name or LOINC code).")],
    date: Annotated[Optional[str], Field(description="Date when the specimen was obtained.")] = None,
) -> Dict[str, Any]:
    """Return laboratory Observation resources for a patient. Returns component level data for lab results."""
    params = {"patient": patient, "code": code}
    if date:
        params["date"] = date
    return _call_fhir("GET", "/Observation", params=params)


@mcp.tool()
def list_vital_signs(
    patient: Annotated[str, Field(description="The patient's FHIR resource ID.")],
    category: Annotated[str, Field(description="Use 'vital-signs' to search for vitals observations.")],
    date: Annotated[Optional[str], Field(description="The date range for when the observation was taken.")] = None,
) -> Dict[str, Any]:
    """Return vital sign Observation resources for a patient. Retrieves vital sign data and other non-duplicable flowsheet data."""
    params = {"patient": patient, "category": category}
    if date:
        params["date"] = date
    return _call_fhir("GET", "/Observation", params=params)


@mcp.tool()
def record_vital_observation(
    resourceType: Annotated[str, Field(description="Use 'Observation' for vitals observations.")],
    category: Annotated[List[Dict[str, Any]], Field(description="Array of category objects. Example: [{'coding': [{'system': 'http://hl7.org/fhir/observation-category', 'code': 'vital-signs', 'display': 'Vital Signs'}]}]")],
    code: Annotated[Dict[str, Any], Field(description="Object with 'text' field for the flowsheet ID. Example: {'text': 'BP'} for blood pressure.")],
    effectiveDateTime: Annotated[str, Field(description="The date and time the observation was taken, in ISO format (e.g., '2023-11-13T10:15:00+00:00').")],
    status: Annotated[str, Field(description="The status of the observation. Only 'final' is supported.")],
    valueString: Annotated[str, Field(description="The measurement value as a string (e.g., '118/77 mmHg').")],
    subject: Annotated[Dict[str, Any], Field(description="Object with 'reference' field for patient. Example: {'reference': 'Patient/12345'}")],
) -> Dict[str, Any]:
    """Create a vital sign Observation for a patient. Files to non-duplicable flowsheet rows including vital signs."""
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
    patient: Annotated[str, Field(description="The patient's FHIR resource ID.")],
    category: Annotated[Optional[str], Field(description="Category filter: 'inpatient', 'outpatient', 'community', or 'discharge'. By default all categories are searched.")] = None,
    status: Annotated[Optional[str], Field(description="Status filter (e.g., 'active').")] = None,
) -> Dict[str, Any]:
    """Retrieve MedicationRequest orders for a patient. Returns inpatient medications, clinic-administered medications, patient-reported medications, and reconciled medications."""
    params = {"patient": patient}
    if category:
        params["category"] = category
    if status:
        params["status"] = status
    return _call_fhir("GET", "/MedicationRequest", params=params)


@mcp.tool()
def create_medication_request(
    resourceType: Annotated[str, Field(description="Use 'MedicationRequest' for medication requests.")],
    medicationCodeableConcept: Annotated[Dict[str, Any], Field(description="Object with 'coding' array and 'text'. Example: {'coding': [{'system': 'http://hl7.org/fhir/sid/ndc', 'code': '12345', 'display': 'Aspirin'}], 'text': 'Aspirin 100mg'}")],
    authoredOn: Annotated[str, Field(description="The date the prescription was written in ISO format.")],
    dosageInstruction: Annotated[List[Dict[str, Any]], Field(description="Array of dosage instructions. Example: [{'route': {'text': 'oral'}, 'doseAndRate': [{'doseQuantity': {'value': 100, 'unit': 'mg'}}]}]")],
    status: Annotated[str, Field(description="The status of the medication request. Use 'active'.")],
    intent: Annotated[str, Field(description="Use 'order'.")],
    subject: Annotated[Dict[str, Any], Field(description="Object with 'reference' field. Example: {'reference': 'Patient/12345'}")],
) -> Dict[str, Any]:
    """Create a MedicationRequest order for a patient."""
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
    patient: Annotated[str, Field(description="The patient's FHIR resource ID.")],
    date: Annotated[str, Field(description="Date or period when the procedure was performed, using FHIR date format.")],
    code: Annotated[Optional[str], Field(description="External CPT codes associated with the procedure.")] = None,
) -> Dict[str, Any]:
    """Retrieve completed Procedure resources for a patient. Returns surgeries, procedures, endoscopies, biopsies, counseling, and physiotherapy."""
    params = {"patient": patient, "date": date}
    if code:
        params["code"] = code
    return _call_fhir("GET", "/Procedure", params=params)


@mcp.tool()
def create_service_request(
    resourceType: Annotated[str, Field(description="Use 'ServiceRequest' for service requests.")],
    code: Annotated[Dict[str, Any], Field(description="Object with 'coding' array. Supports LOINC, SNOMED, CPT codes. Example: {'coding': [{'system': 'http://loinc.org', 'code': '12345', 'display': 'Lab Test'}]}")],
    authoredOn: Annotated[str, Field(description="The order instant in ISO format. Date and time when the order is signed.")],
    status: Annotated[str, Field(description="The status of the service request. Use 'active'.")],
    intent: Annotated[str, Field(description="Use 'order'.")],
    priority: Annotated[str, Field(description="Priority of the request. Use 'stat' for urgent.")],
    subject: Annotated[Dict[str, Any], Field(description="Object with 'reference' field. Example: {'reference': 'Patient/12345'}")],
    occurrenceDateTime: Annotated[Optional[str], Field(description="The date and time for the service request to be conducted, in ISO format.")] = None,
    note: Annotated[Optional[Dict[str, Any]], Field(description="Object with 'text' field for free text comments. Example: {'text': 'Rush order'}")] = None,
) -> Dict[str, Any]:
    """Create a ServiceRequest order (labs, imaging, consults) for a patient."""
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
