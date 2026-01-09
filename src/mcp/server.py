"""Simplified MCP server for MedAgentBench FHIR tools."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configuration
FHIR_API_BASE = os.environ.get("MCP_FHIR_API_BASE", "http://localhost:8080/fhir/").rstrip("/")
app = FastAPI(title="MedAgentBench MCP Server", version="0.1.0")
def create_app() -> FastAPI:
    return app

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


# Resources
RESOURCES = [
    ResourceHandle(
        resource_id="medagentbench.overview",
        name="MedAgentBench Overview",
        description="High-level facts about the MedAgentBench benchmark.",
    ),
    ResourceHandle(
        resource_id="medagentbench.quickstart",
        name="Quickstart Instructions",
        description="Setup instructions for MedAgentBench.",
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
        # Log more details about the error
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
                "category": {
                    "type": "array",
                    "description": "Observation categories with coding metadata.",
                    "items": {"type": "object"}
                },
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
                "dosageInstruction": {
                    "type": "array",
                    "description": "Dose, rate, and route instructions.",
                    "items": {"type": "object"}
                },
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
    resource = next((r for r in RESOURCES if r.resource_id == resource_id), None)
    if not resource:
        raise HTTPException(status_code=404, detail=f"Resource '{resource_id}' not found")
    
    # Return resource data
    if resource_id == "medagentbench.overview":
        return {
            "resource_id": resource.resource_id,
            "name": resource.name,
            "description": resource.description,
            "data": {
                "paper": {"title": "MedAgentBench: A Virtual EHR Environment", "journal": "NEJM AI", "year": 2025},
                "docker_image": "jyxsu6/medagentbench:latest",
                "task_ports": list(range(5000, 5016)),
            },
        }
    elif resource_id == "medagentbench.quickstart":
        return {
            "resource_id": resource.resource_id,
            "name": resource.name,
            "description": resource.description,
            "data": "1. Create conda environment\n2. Install dependencies\n3. Pull Docker image\n4. Configure agent\n5. Run tasks",
        }
    return {"resource_id": resource.resource_id, "name": resource.name, "description": resource.description}


@app.get("/tools", response_model=List[ToolDescriptor])
async def list_tools() -> List[ToolDescriptor]:
    return TOOLS


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


def main() -> None:
    """Entrypoint used by `python -m src.mcp.server`."""
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=False)


if __name__ == "__main__":
    main()
