"""FastMCP server for MedAgentBench FHIR tools."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Literal

import httpx
from fastmcp import FastMCP
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic models for FHIR resource structures
# These provide explicit schemas that help LLMs understand the expected format
# ============================================================================

# -----------------------------------------------------------------------------
# Common/Shared Models
# -----------------------------------------------------------------------------

class SubjectReference(BaseModel):
    """Subject reference object pointing to a patient."""
    reference: str = Field(description="The patient FHIR ID for whom the resource is about (e.g., 'Patient/12345')")


class TextObject(BaseModel):
    """Generic object with text field."""
    text: str = Field(description="Free text value")


# -----------------------------------------------------------------------------
# Observation (Vitals) Models - POST {api_base}/Observation
# -----------------------------------------------------------------------------

class VitalsCategoryCoding(BaseModel):
    """Coding element for vital signs category."""
    system: str = Field(
        default="http://hl7.org/fhir/observation-category",
        description="Use 'http://hl7.org/fhir/observation-category'"
    )
    code: str = Field(
        default="vital-signs",
        description="Use 'vital-signs'"
    )
    display: str = Field(
        default="Vital Signs",
        description="Use 'Vital Signs'"
    )


class VitalsCategoryElement(BaseModel):
    """Category element for vital signs observation."""
    coding: List[VitalsCategoryCoding] = Field(
        description="Array of coding objects for the category"
    )


class VitalsCodeObject(BaseModel):
    """Code object for vital signs - specifies what is being measured."""
    text: str = Field(
        description="The flowsheet ID, encoded flowsheet ID, or LOINC codes to flowsheet mapping. What is being measured (e.g., 'BP', 'Temp', 'HR', 'SpO2')"
    )


# -----------------------------------------------------------------------------
# MedicationRequest Models - POST {api_base}/MedicationRequest
# -----------------------------------------------------------------------------

class MedicationCoding(BaseModel):
    """Coding for medication."""
    system: str = Field(
        default="http://hl7.org/fhir/sid/ndc",
        description="Coding system such as 'http://hl7.org/fhir/sid/ndc'"
    )
    code: str = Field(description="The actual medication code")
    display: str = Field(description="Display name of the medication")


class MedicationCodeableConcept(BaseModel):
    """Medication codeable concept with coding and text."""
    coding: List[MedicationCoding] = Field(description="Array of medication coding objects")
    text: str = Field(description="The order display name of the medication, otherwise the record name")


class DoseQuantity(BaseModel):
    """Dose quantity with value and unit."""
    value: float = Field(description="Numeric dose value")
    unit: str = Field(description="Unit for the dose such as 'g', 'mg', 'mL'")


class RateQuantity(BaseModel):
    """Rate quantity with value and unit (for IV medications)."""
    value: float = Field(description="Numeric rate value")
    unit: str = Field(description="Unit for the rate such as 'h' (per hour)")


class DoseAndRate(BaseModel):
    """Dose and rate specification. Include doseQuantity for discrete doses, rateQuantity for IV rates."""
    doseQuantity: Optional[DoseQuantity] = Field(default=None, description="The dose quantity (for discrete doses)")
    rateQuantity: Optional[RateQuantity] = Field(default=None, description="The rate quantity (for IV medications)")


class RouteText(BaseModel):
    """Route of administration."""
    text: str = Field(description="The medication route (e.g., 'oral', 'intravenous', 'subcutaneous', 'topical')")


class DosageInstruction(BaseModel):
    """Dosage instruction with route and dose/rate."""
    route: RouteText = Field(description="Route of administration")
    doseAndRate: List[DoseAndRate] = Field(description="Array of dose and rate specifications")


# -----------------------------------------------------------------------------
# ServiceRequest Models - POST {api_base}/ServiceRequest
# -----------------------------------------------------------------------------

class ServiceRequestCoding(BaseModel):
    """Coding for service request - supports LOINC, SNOMED, CPT, CBV, THL, or Kuntalitto codes."""
    system: str = Field(description="Coding system such as 'http://loinc.org', 'http://snomed.info/sct', or CPT")
    code: str = Field(description="The actual code")
    display: str = Field(description="Display name")


class ServiceRequestCode(BaseModel):
    """Code object for service request - the standard terminology codes mapped to the procedure."""
    coding: List[ServiceRequestCoding] = Field(
        description="Array of coding objects. Supports LOINC, SNOMED, CPT, CBV, THL, or Kuntalitto codes"
    )


class NoteObject(BaseModel):
    """Note object with text field for comments."""
    text: str = Field(description="Free text comment")

# Configuration (defaults for local development)
FHIR_API_BASE = os.environ.get("MCP_FHIR_API_BASE", "http://localhost:8080/fhir/").rstrip("/")
TASKS_FILE = os.environ.get("MCP_TASKS_FILE", "src/mcp/resources/tasks/tasks.json")
SYSTEM_PROMPT_FILE = os.environ.get("MCP_SYSTEM_PROMPT_FILE", "src/mcp/resources/prompts/system_prompt.txt")

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


def _load_system_prompt() -> str:
    """Load system prompt template from file."""
    prompt_path = Path(SYSTEM_PROMPT_FILE)
    if not prompt_path.exists():
        print(f"Warning: System prompt file '{SYSTEM_PROMPT_FILE}' not found.")
        return ""
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
        print(f"Loaded system prompt from {SYSTEM_PROMPT_FILE}")
        return prompt
    except Exception as e:
        print(f"Error loading system prompt from '{SYSTEM_PROMPT_FILE}': {e}")
        return ""


# Initialize tasks and prompt
_TASKS = _load_tasks()
_SYSTEM_PROMPT = _load_system_prompt()


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


@mcp.resource("medagentbench://prompts/system")
def get_system_prompt() -> str:
    """System prompt template for MedAgentBench evaluation.
    
    This is the master prompt template used to instruct medical AI agents.
    Contains placeholders for: {mcp_server_url}, {context}, {question}
    
    The evaluator fetches this template, fills in the placeholders, and sends
    the complete prompt to the agent being evaluated.
    """
    return _SYSTEM_PROMPT


# Tools
@mcp.tool()
def search_patients(
    identifier: Annotated[Optional[str], Field(description="The patient's identifier.")] = None,
    name: Annotated[Optional[str], Field(description="Any part of the patient's name. When discrete name parameters are used, such as family or given, this parameter is ignored.")] = None,
    family: Annotated[Optional[str], Field(description="The patient's family (last) name.")] = None,
    given: Annotated[Optional[str], Field(description="The patient's given name. May include first and middle names.")] = None,
    birthdate: Annotated[Optional[str], Field(description="The patient's date of birth in the format YYYY-MM-DD.")] = None,
    gender: Annotated[Optional[str], Field(description="The patient's legal sex. The legal-sex parameter is preferred.")] = None,
    legal_sex: Annotated[Optional[str], Field(description="The patient's legal sex. Takes precedence over the gender search parameter.")] = None,
    address: Annotated[Optional[str], Field(description="The patient's street address.")] = None,
    address_city: Annotated[Optional[str], Field(description="The city for patient's home address.")] = None,
    address_state: Annotated[Optional[str], Field(description="The state for the patient's home address.")] = None,
    address_postalcode: Annotated[Optional[str], Field(description="The postal code for patient's home address.")] = None,
    telecom: Annotated[Optional[str], Field(description="The patient's phone number or email.")] = None,
) -> Dict[str, Any]:
    """Patient.Search - Filter or search for patients based on demographics, identifiers, or contact information. Retrieves patient demographic information from a patient's chart for each matching patient record."""
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
    if gender:
        params["gender"] = gender
    if legal_sex:
        params["legal-sex"] = legal_sex
    if address:
        params["address"] = address
    if address_city:
        params["address-city"] = address_city
    if address_state:
        params["address-state"] = address_state
    if address_postalcode:
        params["address-postalcode"] = address_postalcode
    if telecom:
        params["telecom"] = telecom
    return _call_fhir("GET", "/Patient", params=params)


@mcp.tool()
def list_patient_problems(
    patient: Annotated[str, Field(description="Reference to a patient resource the condition is for.")],
    category: Annotated[Optional[str], Field(description="Always 'problem-list-item' for this API.")] = None,
) -> Dict[str, Any]:
    """Condition.Search (Problems) - Retrieve problems from a patient's chart. This includes any data found in the patient's problem list across all encounters. Note that this resource retrieves only data stored in problem list records. Medical history data documented outside of a patient's problem list isn't available unless retrieved using another method."""
    params = {"patient": patient}
    if category:
        params["category"] = category
    return _call_fhir("GET", "/Condition", params=params)


@mcp.tool()
def list_lab_observations(
    patient: Annotated[str, Field(description="Reference to a patient resource the observation is for.")],
    code: Annotated[str, Field(description="The observation identifier (base name).")],
    date: Annotated[Optional[str], Field(description="Date when the specimen was obtained.")] = None,
) -> Dict[str, Any]:
    """Observation.Search (Labs) - Return component level data for lab results."""
    params = {"patient": patient, "code": code}
    if date:
        params["date"] = date
    return _call_fhir("GET", "/Observation", params=params)


@mcp.tool()
def list_vital_signs(
    patient: Annotated[str, Field(description="Reference to a patient resource the observation is for.")],
    category: Annotated[str, Field(description="Use 'vital-signs' to search for vitals observations.")],
    date: Annotated[Optional[str], Field(description="The date range for when the observation was taken.")] = None,
) -> Dict[str, Any]:
    """Observation.Search (Vitals) - Retrieve vital sign data from a patient's chart, as well as any other non-duplicable data found in the patient's flowsheets across all encounters. This resource requires the use of encoded flowsheet IDs which are different for each organization and between production and non-production environments."""
    params = {"patient": patient, "category": category}
    if date:
        params["date"] = date
    return _call_fhir("GET", "/Observation", params=params)


@mcp.tool()
def record_vital_observation(
    resourceType: Annotated[str, Field(description="Use 'Observation' for vitals observations.")],
    category: Annotated[List[VitalsCategoryElement], Field(description="Array of category objects. Each must contain coding with system='http://hl7.org/fhir/observation-category', code='vital-signs', display='Vital Signs'.")],
    code: Annotated[VitalsCodeObject, Field(description="Code object specifying what is being measured.")],
    effectiveDateTime: Annotated[str, Field(description="The date and time the observation was taken, in ISO format (e.g., '2023-11-13T10:15:00+00:00').")],
    status: Annotated[str, Field(description="The status of the observation. Only 'final' is supported. We do not support filing data that isn't finalized.")],
    valueString: Annotated[str, Field(description="Measurement value as a string (e.g., '118/77' for BP, '98.6' for temp).")],
    subject: Annotated[SubjectReference, Field(description="The patient this observation is about.")],
) -> Dict[str, Any]:
    """Observation.Create (Vitals) - File vital signs to all non-duplicable flowsheet rows. This resource can file vital signs for all flowsheets."""
    body = {
        "resourceType": resourceType,
        "category": [cat.model_dump() for cat in category],
        "code": code.model_dump(),
        "effectiveDateTime": effectiveDateTime,
        "status": status,
        "valueString": valueString,
        "subject": subject.model_dump(),
    }
    return _call_fhir("POST", "/Observation", body=body)


@mcp.tool()
def list_medication_requests(
    patient: Annotated[str, Field(description="The FHIR patient ID.")],
    category: Annotated[Optional[str], Field(description="The category of medication orders to search for. By default all categories are searched. Supported: 'Inpatient', 'Outpatient' (clinic-administered CAMS), 'Community' (prescriptions), 'Discharge'.")] = None,
    date: Annotated[Optional[str], Field(description="The medication administration date. Corresponds to dosageInstruction.timing.repeat.boundsPeriod. Use caution when filtering by date as it may filter out important active medications.")] = None,
) -> Dict[str, Any]:
    """MedicationRequest.Search (Signed Medication Order) - Query for medication orders based on a patient and optionally status or category. Returns inpatient-ordered medications, clinic-administered medications (CAMS), patient-reported medications, and reconciled medications from Care Everywhere and other external sources. Patient-reported medications have reportedBoolean=True."""
    params = {"patient": patient}
    if category:
        params["category"] = category
    if date:
        params["date"] = date
    return _call_fhir("GET", "/MedicationRequest", params=params)


@mcp.tool()
def create_medication_request(
    resourceType: Annotated[str, Field(description="Use 'MedicationRequest' for medication requests.")],
    medicationCodeableConcept: Annotated[MedicationCodeableConcept, Field(description="Medication codeable concept with coding array and text.")],
    authoredOn: Annotated[str, Field(description="The date the prescription was written, in ISO format.")],
    dosageInstruction: Annotated[List[DosageInstruction], Field(description="Array of dosage instructions with route and doseAndRate (containing doseQuantity and/or rateQuantity).")],
    status: Annotated[str, Field(description="The status of the medication request. Use 'active'.")],
    intent: Annotated[str, Field(description="Use 'order'.")],
    subject: Annotated[SubjectReference, Field(description="The patient for whom the medication request is for.")],
) -> Dict[str, Any]:
    """MedicationRequest.Create - Create a medication order for a patient."""
    body = {
        "resourceType": resourceType,
        "medicationCodeableConcept": medicationCodeableConcept.model_dump(exclude_none=True),
        "authoredOn": authoredOn,
        "dosageInstruction": [di.model_dump(exclude_none=True) for di in dosageInstruction],
        "status": status,
        "intent": intent,
        "subject": subject.model_dump(),
    }
    return _call_fhir("POST", "/MedicationRequest", body=body)


@mcp.tool()
def list_patient_procedures(
    patient: Annotated[str, Field(description="Reference to a patient resource the procedure is for.")],
    date: Annotated[str, Field(description="Date or period that the procedure was performed, using the FHIR date parameter format.")],
    code: Annotated[Optional[str], Field(description="External CPT codes associated with the procedure.")] = None,
) -> Dict[str, Any]:
    """Procedure.Search (Orders) - Retrieve completed procedures for a patient. Returns surgeries and procedures performed, including endoscopies and biopsies, as well as less invasive actions like counseling and physiotherapy. Only completed procedures are returned. This resource is designed for high-level summarization around the occurrence of a procedure."""
    params = {"patient": patient, "date": date}
    if code:
        params["code"] = code
    return _call_fhir("GET", "/Procedure", params=params)


@mcp.tool()
def create_service_request(
    resourceType: Annotated[str, Field(description="Use 'ServiceRequest' for service requests.")],
    code: Annotated[ServiceRequestCode, Field(description="The standard terminology codes mapped to the procedure (LOINC, SNOMED, CPT, CBV, THL, or Kuntalitto codes).")],
    authoredOn: Annotated[str, Field(description="The order instant in ISO format. This is the date and time when the order is signed or signed and held.")],
    status: Annotated[str, Field(description="The status of the service request. Use 'active'.")],
    intent: Annotated[str, Field(description="Use 'order'.")],
    priority: Annotated[Literal["stat"], Field(description="Priority of the request. Must be 'stat'.")],
    subject: Annotated[SubjectReference, Field(description="The patient for whom the service request is for.")],
    occurrenceDateTime: Annotated[Optional[str], Field(description="The date and time for the service request to be conducted, in ISO format.")] = None,
    note: Annotated[Optional[NoteObject], Field(description="Free text comment for the request.")] = None,
) -> Dict[str, Any]:
    """ServiceRequest.Create - Create an order for labs, imaging, or consults for a patient."""
    body = {
        "resourceType": resourceType,
        "code": code.model_dump(),
        "authoredOn": authoredOn,
        "status": status,
        "intent": intent,
        "priority": priority,
        "subject": subject.model_dump(),
    }
    if occurrenceDateTime:
        body["occurrenceDateTime"] = occurrenceDateTime
    if note:
        body["note"] = note.model_dump()
    return _call_fhir("POST", "/ServiceRequest", body=body)


# -----------------------------------------------------------------------------
# Utility Tools - Date Comparison
# -----------------------------------------------------------------------------

@mcp.tool()
def check_date_within_period(
    date_to_check: Annotated[str, Field(description="The date to check, in ISO format (e.g., '2023-11-09T03:05:00+00:00').")],
    reference_date: Annotated[str, Field(description="The reference date to compare against, in ISO format (e.g., '2023-11-13T10:15:00+00:00').")],
    period_days: Annotated[int, Field(description="The number of days for the period. Use 365 for 1 year, 30 for 1 month, 7 for 1 week, etc.")],
) -> Dict[str, Any]:
    """Date Utility - Check if a date is within a specified period from a reference date. 
    
    Returns whether the date_to_check is within period_days BEFORE the reference_date.
    Use this to determine if a lab result is recent enough (within the required period) 
    or if a new order should be placed.
    
    Example: To check if an HbA1C from '2023-11-09' is within 1 year of '2023-11-13':
    - date_to_check: '2023-11-09T03:05:00+00:00'
    - reference_date: '2023-11-13T10:15:00+00:00' 
    - period_days: 365
    - Result: is_within_period=True (Nov 9 is within 365 days before Nov 13)
    
    DECISION GUIDE:
    - If is_within_period is TRUE: The date is RECENT - do NOT order a new test
    - If is_within_period is FALSE: The date is OLD - you should order a new test
    """
    try:
        # Parse dates - handle various ISO formats
        def parse_iso_date(date_str: str) -> datetime:
            # Remove 'Z' suffix and handle timezone
            date_str = date_str.replace('Z', '+00:00')
            # Try parsing with timezone
            try:
                return datetime.fromisoformat(date_str)
            except ValueError:
                # Try without timezone
                return datetime.fromisoformat(date_str.split('+')[0].split('-')[0])
        
        check_date = parse_iso_date(date_to_check)
        ref_date = parse_iso_date(reference_date)
        
        # Calculate cutoff date (reference_date minus period_days)
        cutoff_date = ref_date - timedelta(days=period_days)
        
        # Check if date_to_check is within the period (i.e., on or after the cutoff)
        is_within = check_date >= cutoff_date
        
        # Calculate days difference
        days_diff = (ref_date - check_date).days
        
        return {
            "date_to_check": date_to_check,
            "reference_date": reference_date,
            "period_days": period_days,
            "cutoff_date": cutoff_date.isoformat(),
            "is_within_period": is_within,
            "days_since_date": days_diff,
        }
    except Exception as e:
        return {
            "error": f"Failed to parse dates: {str(e)}",
            "date_to_check": date_to_check,
            "reference_date": reference_date,
            "period_days": period_days,
        }


def main() -> None:
    """Entrypoint used by `python -m src.mcp.server`.

    Supports two transport modes:
    - stdio: For MCP Inspector and CLI tools (default when --stdio flag is passed)
    - streamable-http: For network access (default, runs on port 8002)

    Usage:
        # For MCP Inspector (stdio transport):
        npx @modelcontextprotocol/inspector python -m src.mcp.server --stdio

        # For network access (streamable-http transport):
        python -m src.mcp.server
        python -m src.mcp.server --transport streamable-http --port 8002
    """
    parser = argparse.ArgumentParser(description="MedAgentBench MCP Server")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Use stdio transport (for MCP Inspector and CLI tools)"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=None,
        help="Transport type (default: streamable-http, or stdio if --stdio is passed)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to for streamable-http transport (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8002,
        help="Port to bind to for streamable-http transport (default: 8002)"
    )

    args = parser.parse_args()

    # Determine transport: --stdio flag takes precedence, then --transport, then default
    if args.stdio:
        transport = "stdio"
    elif args.transport:
        transport = args.transport
    else:
        transport = "streamable-http"

    # For stdio, suppress print statements to avoid corrupting the protocol stream
    if transport == "stdio":
        # Run with stdio transport for MCP Inspector / CLI
        mcp.run(transport="stdio")
    else:
        # Print startup info only for non-stdio transport
        print(f"Starting MedAgentBench FastMCP Server")
        print(f"FHIR API base: {FHIR_API_BASE}")
        print(f"Tasks file: {TASKS_FILE}")
        print(f"Loaded {len(_TASKS)} tasks")
        print(f"Transport: {transport}")
        print(f"Listening on {args.host}:{args.port}")
        # Run with streamable-http transport for network access
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
