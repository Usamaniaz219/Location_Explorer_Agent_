# Location Explorer Agent

An intelligent agent for analyzing underutilized properties and recommending optimal reuse strategies based on zoning regulations, building characteristics, and machine learning predictions.

## Features

- 🏢 Identifies underutilized commercial properties using predictive modeling
- 📊 Analyzes zoning regulations and permitted uses via Zoneomics API
- 🗺️ Generates interactive Folium maps with property recommendations
- 🤖 Provides AI-powered recommendations for:
  - Office space conversions
- 📈 Regression modeling for available square footage prediction
- 🐳 Docker container for easy deployment

## Prerequisites

- Docker Desktop ([Install Guide](https://docs.docker.com/get-docker/))
- Zoneomics API key (for zoning analysis)
- Python 3.12 (if running locally)

## Installation

### Using Docker (Recommended)

```bash
# 1. Build the Docker image
docker build -t location-explorer .

# 2. Run the container with volume mounts
docker run -v ./output:/app/output -v ./pretrained_models:/app/pretrained_models location-explorer

#Local Python Installation
# 1. Install Poetry
pip install poetry

# 2. Install dependencies
poetry install

# 3. Run the agent
poetry run python scripts/run_agent.py

#Set environment variables in .env file:
ZONING_API_KEY=852fba3cae77d1533c563d58d6ad03e32d5d1e7
DATA_PATH=data/properties.xlsx
MODEL_PATH=pretrained_models/classification_model.joblib
