## Overview
This repository contains tools and resources for benchmarking and evaluating knowledge graph-based systems. It includes datasets, scripts, and source code for building and querying knowledge graphs, as well as evaluating their performance in various domains.

## Repository Structure

### Datasets
The `datasets/` directory contains various datasets used for benchmarking. These datasets are organized by domain (e.g., `agriculture/`, `legal/`, `mix/`, `sclc/`) and include raw data, processed data, and evaluation results.

### Scripts
The `scripts/` directory contains Python scripts for evaluation and analysis:
- `constant.py`: Contains constants used across scripts.
- `eval_multihopqa.py`: Script for evaluating multi-hop question answering.
- `eval_ultradomain.py`: Script for evaluating ultra-domain-specific tasks.

### Source Code
The `src/` directory contains the main source code for the project. It is divided into two key components:

#### Graph Builder
Located in `src/graph-builder/`, this module contains the code to build knowledge graphs for eGOT. Key files include:
- `config.toml`: Configuration file for the graph builder.
- `Dockerfile`: Docker setup for the graph builder.
- `requirements.txt`: Python dependencies for the graph builder.

#### Knowledge Graph Retrieval
Located in `src/knowledge_graph_retrieval/`, this module is a FastAPI server that provides endpoints to query the system. When a query is made, the system retrieves relevant knowledge from the knowledge graph. Key files include:
- `app.py`: Main FastAPI application.
- `got.py`: Handles graph operations and retrieval.
- `QA_integration.py`: Integrates question answering with knowledge retrieval.
- `constants.py`: Contains constants for the server.

## Usage

### Building Knowledge Graphs
To build knowledge graphs for eGOT, navigate to the `src/graph-builder/` directory and follow the instructions in the `README.md` file located there.

### Running the FastAPI Server
To start the knowledge graph retrieval server:
1. Navigate to the `src/knowledge_graph_retrieval/` directory.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the server using the command:
   ```bash
   uvicorn app:app --reload
   ```

### Evaluating Results
Use the scripts in the `scripts/` directory to evaluate the performance of the system on various datasets. Refer to the comments in each script for usage instructions.