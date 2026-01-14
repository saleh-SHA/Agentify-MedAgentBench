"""CLI entry point for MedAgentBench evaluation."""

import typer
import asyncio
from typing import List, Optional

from src.green_agent.medagent import start_medagent_green
from src.white_agent.medagent import start_medagent_white
from src.medagent_launcher import launch_medagent_evaluation, launch_medagent_batch_evaluation

app = typer.Typer(help="MedAgentBench - Medical agent evaluation framework")


@app.command()
def green():
    """Start the green agent (assessment manager)."""
    start_medagent_green()


@app.command()
def white():
    """Start the white agent (agent being tested)."""
    start_medagent_white()


@app.command()
def launch(
    task_index: int = typer.Option(0, help="Index of the task to run from test_data_v2.json"),
    mcp_server: str = typer.Option("http://0.0.0.0:8002", help="MCP server URL"),
    max_rounds: int = typer.Option(8, help="Maximum number of interaction rounds")
):
    """Launch a single MedAgentBench evaluation."""
    asyncio.run(launch_medagent_evaluation(
        task_index=task_index,
        mcp_server_url=mcp_server,
        max_rounds=max_rounds
    ))


@app.command()
def batch(
    task_indices: Optional[str] = typer.Option(None, help="Comma-separated task indices (e.g., '0,1,2'). If not provided, runs all tasks."),
    mcp_server: str = typer.Option("http://0.0.0.0:8002", help="MCP server URL"),
    max_rounds: int = typer.Option(8, help="Maximum number of interaction rounds")
):
    """Launch batch MedAgentBench evaluations on multiple tasks."""
    # Parse task indices
    task_list = None
    if task_indices:
        try:
            task_list = [int(idx.strip()) for idx in task_indices.split(',')]
        except ValueError:
            print(f"Error: Invalid task indices format. Use comma-separated integers (e.g., '0,1,2')")
            raise typer.Exit(code=1)

    asyncio.run(launch_medagent_batch_evaluation(
        task_indices=task_list,
        mcp_server_url=mcp_server,
        max_rounds=max_rounds
    ))


if __name__ == "__main__":
    app()
