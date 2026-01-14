#!/bin/bash

# FHIR Launcher Script
# This script pulls, tags, and runs the MedAgentBench Docker container

set -e  # Exit on error

echo "Pulling MedAgentBench Docker image..."
docker pull jyxsu6/medagentbench:latest

echo "Tagging image as medagentbench..."
docker tag jyxsu6/medagentbench:latest medagentbench

echo "Starting MedAgentBench container on port 8080..."
docker run -p 8080:8080 medagentbench