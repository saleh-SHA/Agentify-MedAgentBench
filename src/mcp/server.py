"""Simplified MCP server for MedAgentBench FHIR tools."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configuration
FHIR_API_BASE = os.environ.get("MCP_FHIR_API_BASE", "http://localhost:8080/fhir/").rstrip("/")
TASKS_FILE = os.environ.get("MCP_TASKS_FILE", "src/mcp/resources/tasks/tasks.json")

app = FastAPI(title="MedAgentBench MCP Server", version="0.1.0")
_started_at = time.monotonic()


# Models
class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float


class ToolDescriptor(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class ResourceHandle(BaseModel):
    resource_id: str
    name: str
    description: Optional[str] = None


class ToolInvocationRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolInvocationResponse(BaseModel):
    tool_name: str
    result: Dict[str, Any]


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

# Resources
RESOURCES = [
    ResourceHandle(
        resource_id="medagentbench_tasks",
        name="MedAgentBench Tasks",
        description=f"Complete list of {len(_TASKS)} evaluation tasks with instructions and expected solutions.",
    ),
]


# Helper functions
def _call_fhir(method: str, path: str, params: Optional[Dict] = None, body: Optional[Dict] = None) -> Dict[str, Any]:
    """Make a request to the FHIR server."""
    url = f"{FHIR_API_BASE}{path}"
    try:
        response = requests.request(
            method, url, params=params if method == "GET" else None, json=body if method != "GET" else None, timeout=30
        )
        response.raise_for_status()
        return {"url": url, "method": method, "status_code": response.status_code, "response": response.json()}
    except requests.RequestException as e:
        error_detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail += f" - Response: {e.response.text}"
            except:
                pass
        raise HTTPException(status_code=400, detail=f"FHIR server error: {error_detail}")


# Tool handlers
def _get_handler(path: str, required_params: List[str]):
    """Create a GET handler for a FHIR endpoint."""
    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        missing = [p for p in required_params if p not in args]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")
        params = {k: v for k, v in args.items() if v is not None}
        return _call_fhir("GET", path, params=params)
    return handler


def _post_handler(path: str, required_fields: List[str]):
    """Create a POST handler for a FHIR endpoint."""
    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        missing = [f for f in required_fields if f not in args]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")
        return _call_fhir("POST", path, body=args)
    return handler


# Tool definitions
TOOLS = [
    ToolDescriptor(
        name="search_patients",
        description="Search for Patient resources by demographics, identifiers, or contact information.",
        input_schema={
            "type": "object",
            "properties": {
                "identifier": {"type": "string", "description": "Unique identifier such as MRN."},
                "name": {"type": "string", "description": "Any part of the patient's name."},
                "family": {"type": "string", "description": "Family (last) name."},
                "given": {"type": "string", "description": "Given (first/middle) name."},
                "birthdate": {"type": "string", "description": "Birth date in YYYY-MM-DD format."},
            },
        },
    ),
    ToolDescriptor(
        name="list_patient_problems",
        description="Retrieve Condition resources from a patient's problem list.",
        input_schema={
            "type": "object",
            "properties": {
                "patient": {"type": "string", "description": "FHIR patient ID."},
                "category": {"type": "string", "description": "Optional category filter (e.g., 'problem-list-item')."},
                "status": {"type": "string", "description": "Optional status (e.g., 'active', 'resolved')."},
            },
            "required": ["patient"],
        },
    ),
    ToolDescriptor(
        name="list_lab_observations",
        description="Return laboratory Observation resources for a patient.",
        input_schema={
            "type": "object",
            "properties": {
                "patient": {"type": "string", "description": "FHIR patient ID."},
                "code": {"type": "string", "description": "Observation code (LOINC or local)."},
                "date": {"type": "string", "description": "Optional specimen collection date or range."},
            },
            "required": ["patient", "code"],
        },
    ),
    ToolDescriptor(
        name="list_vital_signs",
        description="Return vital sign Observation resources for a patient.",
        input_schema={
            "type": "object",
            "properties": {
                "patient": {"type": "string", "description": "FHIR patient ID."},
                "category": {"type": "string", "description": "Use 'vital-signs' to fetch vitals."},
                "date": {"type": "string", "description": "Optional date or range when vitals were taken."},
            },
            "required": ["patient", "category"],
        },
    ),
    ToolDescriptor(
        name="record_vital_observation",
        description="Create a vital sign Observation for a patient.",
        input_schema={
            "type": "object",
            "properties": {
                "resourceType": {"type": "string", "description": 'Use "Observation".'},
                "category": {"type": "array", "description": "Observation categories with coding metadata."},
                "code": {"type": "object", "description": "Flowsheet row / vital concept."},
                "effectiveDateTime": {"type": "string", "description": "ISO timestamp of measurement."},
                "status": {"type": "string", "description": 'Use "final".'},
                "valueString": {"type": "string", "description": "Measurement value."},
                "subject": {"type": "object", "description": "Reference to the patient resource."},
            },
            "required": ["resourceType", "category", "code", "effectiveDateTime", "status", "valueString", "subject"],
        },
    ),
    ToolDescriptor(
        name="list_medication_requests",
        description="Retrieve MedicationRequest orders for a patient.",
        input_schema={
            "type": "object",
            "properties": {
                "patient": {"type": "string", "description": "FHIR patient ID."},
                "category": {"type": "string", "description": "Optional: inpatient, outpatient, community, discharge."},
                "status": {"type": "string", "description": "Optional status such as 'active'."},
            },
            "required": ["patient"],
        },
    ),
    ToolDescriptor(
        name="create_medication_request",
        description="Create a MedicationRequest order for a patient.",
        input_schema={
            "type": "object",
            "properties": {
                "resourceType": {"type": "string", "description": 'Use "MedicationRequest".'},
                "medicationCodeableConcept": {"type": "object", "description": "Medication coding and display name."},
                "authoredOn": {"type": "string", "description": "Date/time when prescription was authored."},
                "dosageInstruction": {"type": "array", "description": "Dose, rate, and route instructions."},
                "status": {"type": "string", "description": 'Order status such as "active".'},
                "intent": {"type": "string", "description": 'Order intent, typically "order".'},
                "subject": {"type": "object", "description": "Reference to the patient resource."},
            },
            "required": ["resourceType", "medicationCodeableConcept", "authoredOn", "dosageInstruction", "status", "intent", "subject"],
        },
    ),
    ToolDescriptor(
        name="list_patient_procedures",
        description="Retrieve completed Procedure resources for a patient.",
        input_schema={
            "type": "object",
            "properties": {
                "patient": {"type": "string", "description": "FHIR patient ID."},
                "date": {"type": "string", "description": "Date or range when procedure occurred."},
                "code": {"type": "string", "description": "Optional CPT or coded value."},
            },
            "required": ["patient", "date"],
        },
    ),
    ToolDescriptor(
        name="create_service_request",
        description="Create a ServiceRequest order (labs, imaging, consults) for a patient.",
        input_schema={
            "type": "object",
            "properties": {
                "resourceType": {"type": "string", "description": 'Use "ServiceRequest".'},
                "code": {"type": "object", "description": "Terminology coding describing the service."},
                "authoredOn": {"type": "string", "description": "Timestamp when order was authored."},
                "status": {"type": "string", "description": 'Order status, typically "active".'},
                "intent": {"type": "string", "description": 'Order intent, typically "order".'},
                "priority": {"type": "string", "description": "Priority such as 'stat'."},
                "subject": {"type": "object", "description": "Reference to the patient."},
                "occurrenceDateTime": {"type": "string", "description": "Optional desired time for service."},
                "note": {"type": "object", "description": "Optional clinician instructions or comments."},
            },
            "required": ["resourceType", "code", "authoredOn", "status", "intent", "priority", "subject"],
        },
    ),
]

# Tool handler registry
_TOOL_HANDLERS = {
    "search_patients": _get_handler("/Patient", []),
    "list_patient_problems": _get_handler("/Condition", ["patient"]),
    "list_lab_observations": _get_handler("/Observation", ["patient", "code"]),
    "list_vital_signs": _get_handler("/Observation", ["patient", "category"]),
    "record_vital_observation": _post_handler("/Observation", ["resourceType", "category", "code", "effectiveDateTime", "status", "valueString", "subject"]),
    "list_medication_requests": _get_handler("/MedicationRequest", ["patient"]),
    "create_medication_request": _post_handler("/MedicationRequest", ["resourceType", "medicationCodeableConcept", "authoredOn", "dosageInstruction", "status", "intent", "subject"]),
    "list_patient_procedures": _get_handler("/Procedure", ["patient", "date"]),
    "create_service_request": _post_handler("/ServiceRequest", ["resourceType", "code", "authoredOn", "status", "intent", "priority", "subject"]),
}


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", uptime_seconds=time.monotonic() - _started_at)


@app.get("/resources", response_model=List[ResourceHandle])
async def list_resources() -> List[ResourceHandle]:
    return RESOURCES


@app.get("/resources/{resource_id}")
async def get_resource(resource_id: str) -> Dict[str, Any]:
    """Get full resource data."""
    
    try:
        # Tasks resources
        if resource_id == "medagentbench_tasks":
            return {
                "resource_id": resource_id,
                "name": "MedAgentBench Tasks",
                "description": f"Complete list of {len(_TASKS)} evaluation tasks.",
                "data": {
                    "total_tasks": len(_TASKS),
                    "tasks": _TASKS,
                },
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get resource '{resource_id}': {e}")


@app.get("/tools", response_model=List[ToolDescriptor])
async def list_tools() -> List[ToolDescriptor]:
    return TOOLS


@app.get("/favicon.ico")
async def favicon() -> Dict[str, Any]:
    """Return empty favicon response to avoid noisy 404s."""
    return {"detail": "No favicon available"}


@app.post("/tools/invoke", response_model=ToolInvocationResponse)
async def invoke_tool(request: ToolInvocationRequest) -> ToolInvocationResponse:
    handler = _TOOL_HANDLERS.get(request.tool_name)
    if not handler:
        raise HTTPException(status_code=404, detail=f"Tool '{request.tool_name}' not found")
    try:
        result = handler(request.arguments)
        return ToolInvocationResponse(tool_name=request.tool_name, result=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def create_app() -> FastAPI:
    """Factory function for creating the app."""
    return app


def main() -> None:
    """Entrypoint used by `python -m src.mcp.server`."""
    print(f"Starting MedAgentBench MCP Server on port 8002")
    print(f"FHIR API base: {FHIR_API_BASE}")
    print(f"Tasks file: {TASKS_FILE}")
    print(f"Loaded {len(_TASKS)} tasks")
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=False)


if __name__ == "__main__":
    main()
