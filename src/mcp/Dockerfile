# Dockerfile for MedAgentBench MCP Server
# Builds a Docker image for the FastMCP-based FHIR tools server

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the MCP server requirements
# We install specific packages needed by the MCP server
RUN pip install --no-cache-dir \
    fastmcp>=2.0.0 \
    httpx>=0.28.1 \
    pydantic>=2.11.9 \
    uvicorn>=0.38.0

# Copy the MCP server code and resources
COPY __init__.py /app/src/mcp/__init__.py
COPY server.py /app/src/mcp/server.py
COPY resources/ /app/src/mcp/resources/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MCP_FHIR_API_BASE=http://host.docker.internal:8080/fhir/
ENV MCP_TASKS_FILE=/app/src/mcp/resources/tasks/tasks.json
ENV MCP_SYSTEM_PROMPT_FILE=/app/src/mcp/resources/prompts/system_prompt.txt

# Expose the MCP server port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the MCP server
CMD ["python", "-m", "src.mcp.server", "--host", "0.0.0.0", "--port", "8002"]
