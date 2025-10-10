"""
Knowledge Graph Generator and Visualizer main module.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid


# # Add the parent directory to the Python path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.knowledge_graph_builder.config import load_config
from src.knowledge_graph_builder.llm import (
    call_llm,
    extract_json_from_text,
    call_ollama,
    call_openai,
)
from src.knowledge_graph_builder.visualization import (
    visualize_knowledge_graph,
    sample_data_visualization,
)
from src.knowledge_graph_builder.text_utils import chunk_text
from src.knowledge_graph_builder.entity_standardization import (
    standardize_entities,
    infer_relationships,
    limit_predicate_length,
)
from src.knowledge_graph_builder.prompts import (
    MAIN_SYSTEM_PROMPT,
    MAIN_USER_PROMPT,
    MAIN_LUPUS_USER_PROMPT,
    MAIN_UV_USER_PROMPT,
    MAIN_UV_SYSTEM_PROMPT,
    MAIN_PLASTICS_SYSTEM_PROMPT,
    MAIN_PLASTICS_USER_PROMPT,
    MAIN_SCLC_USER_PROMPT,
    MAIN_SCLC_SYSTEM_PROMPT,
    MAIN_MHOP_PROMPT
)
# === NEW IMPORTS ===
from src.knowledge_graph_builder.embedding import get_embedding, load_embedding_model
# ===================


def process_with_llm(config, input_text, debug=False):
    """
    Process input text with LLM to extract triples.

    Args:
        config: Configuration dictionary
        input_text: Text to analyze
        debug: If True, print detailed debug information

    Returns:
        List of extracted triples or None if processing failed
    """
    # Use prompts from the prompts module
    system_prompt = MAIN_SYSTEM_PROMPT
    user_prompt = MAIN_MHOP_PROMPT
    user_prompt += f"```\n{input_text}```\n"

    # LLM configuration
    model = config["llm"]["model"]
    api_key = config["llm"]["api_key"]
    max_tokens = config["llm"]["max_tokens"]
    temperature = config["llm"]["temperature"]
    base_url = config["llm"]["base_url"]

    # Process with LLM
    metadata = {}
    response = call_openai(
        model, user_prompt, system_prompt, max_tokens, temperature, base_url
    )

    # Print raw response only if debug mode is on
    if debug:
        print("Raw LLM response:")
        print(response)
        print("\n---\n")

    # Extract JSON from the response
    result = extract_json_from_text(response)

    if result:
        # Validate and filter triples to ensure they have all required fields
        valid_triples = []
        invalid_count = 0

        for item in result:
            if (
                isinstance(item, dict)
                and "subject" in item
                and "predicate" in item
                and "object" in item
            ):
                # Add metadata to valid items
                valid_triples.append(dict(item, **metadata))
            else:
                invalid_count += 1

        if invalid_count > 0:
            print(
                f"Warning: Filtered out {invalid_count} invalid triples missing required fields"
            )

        if not valid_triples:
            print("Error: No valid triples found in LLM response")
            return None

        # Apply predicate length limit to all valid triples
        for triple in valid_triples:
            triple["predicate"] = limit_predicate_length(triple["predicate"])

        # Print extracted JSON only if debug mode is on
        if debug:
            print("Extracted JSON:")
            print(json.dumps(valid_triples, indent=2))  # Pretty print the JSON

        return valid_triples
    else:
        # Always print error messages even if debug is off
        print(
            "\n\nERROR ### Could not extract valid JSON from response: ",
            response,
            "\n\n",
        )
        return None


def process_text_in_chunks(config, full_text, embedding_model, debug=False):
    """
    PHASE 1: Process a large text by breaking it into chunks, generating embeddings,
    and extracting raw triples.

    Args:
        config: Configuration dictionary
        full_text: The complete text to process
        debug: If True, print detailed debug information

    Returns:
        A tuple containing:
        - all_triples: A list of all extracted raw triples.
        - processed_chunks: A list of dictionaries, each containing chunk text, id, and embedding.
    """
    chunk_size = config.get("chunking", {}).get("chunk_size", 500)
    overlap = config.get("chunking", {}).get("overlap", 50)
    text_chunks = chunk_text(full_text, chunk_size, overlap)

    print("=" * 50)
    print("PHASE 1: CHUNK PROCESSING (TRIPLES & EMBEDDINGS)")
    print("=" * 50)
    print(f"Processing text in {len(text_chunks)} chunks (size: {chunk_size} words, overlap: {overlap} words)")

    all_triples = []
    processed_chunks = []

    for i, chunk in enumerate(text_chunks):
        chunk_id = str(uuid.uuid4())
        print(f"Processing chunk {i + 1}/{len(text_chunks)} ({len(chunk.split())} words)")

        # 1. Generate Embedding for the chunk
        embedding = get_embedding(chunk, embedding_model)
        if not embedding:
            print(f"Warning: Failed to generate embedding for chunk {chunk_id}. Skipping.")
            continue
        
        processed_chunks.append({
            "id": chunk_id,
            "text": chunk,
            "embedding": embedding
        })

        # 2. Process the chunk with LLM to get triples
        chunk_triples = process_with_llm(config, chunk, debug)

        if chunk_triples:
            for item in chunk_triples:
                item["chunk_id"] = chunk_id  # Link triple to its source chunk
            all_triples.extend(chunk_triples)
        else:
            print(f"Warning: Failed to extract triples from chunk {chunk_id}")

    print(f"\nExtracted a total of {len(all_triples)} raw triples from all chunks.")
    return all_triples, processed_chunks


def get_unique_entities(triples):
    """
    Get the set of unique entities from the triples.

    Args:
        triples: List of triple dictionaries

    Returns:
        Set of unique entity names
    """
    entities = set()
    for triple in triples:
        if not isinstance(triple, dict):
            continue
        if "subject" in triple:
            entities.add(triple["subject"])
        if "object" in triple:
            entities.add(triple["object"])
    return entities


def process_file(config, input_file, output_file, embedding_model, debug=False):
    """
    Process a single file: extract data, run standardization/inference,
    store in Neo4j in batches, and generate outputs.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            input_text = f.read()
        print("=" * 50)
        print(f"Processing file: {input_file}")
        print("=" * 50)
    except Exception as e:
        print(f"Error reading input file {input_file}: {e}")
        return None

    # PHASE 1: Get raw triples and chunk data
    all_triples, processed_chunks = process_text_in_chunks(config, input_text, embedding_model, debug)

    if not all_triples:
        print(f"Knowledge graph generation failed for file: {input_file}. No triples found.")
        return None

    # PHASE 2: Entity Standardization (on all triples from the file)
    if config.get("standardization", {}).get("enabled", False):
        print("\n" + "=" * 50)
        print("PHASE 2: ENTITY STANDARDIZATION")
        print("=" * 50)
        all_triples = standardize_entities(all_triples, config)

    print(all_triples)
    # PHASE 3: Relationship Inference (on all triples from the file)
    if config.get("inference", {}).get("enabled", False):
        print("\n" + "=" * 50)
        print("PHASE 3: RELATIONSHIP INFERENCE")
        print("=" * 50)
        all_triples = infer_relationships(all_triples, config)

    # Combine processed triples into their respective chunks for unified data storage
    final_output_data = []
    inferred_triples = []  # List to hold triples without a chunk_id
    triples_by_chunk = {chunk['id']: [] for chunk in processed_chunks}

    # First, iterate through all triples to sort them
    for triple in all_triples:
        chunk_id = triple.get('chunk_id')
        # If the triple has a valid chunk_id, assign it to the correct chunk
        if chunk_id in triples_by_chunk:
            triples_by_chunk[chunk_id].append(triple)
        # Otherwise, check if it's an inferred triple and collect it
        elif triple.get("inferred") == True:
            inferred_triples.append(triple)

    # Next, build the output data for chunks
    for chunk in processed_chunks:
        chunk['triples'] = triples_by_chunk.get(chunk['id'], [])
        final_output_data.append(chunk)

    if inferred_triples:
        final_output_data.append({
            "id": "inferred-triples-group",
            "source": "inferred",
            "triples": inferred_triples
        })


    # Save the hybrid data (chunks, embeddings, final triples) to JSON
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output_data, f, indent=2)
        print(f"\nSaved processed hybrid data to {output_file}")
    except Exception as e:
        print(f"Warning: Could not save data to {output_file}: {e}")
    
    # Return all triples for final visualization purposes
    return all_triples


def main_kg_builder(
    input_path,
    output_dir,
    config_path="config.toml",
    experiment_name=None,
    debug=False,
    no_standardize=False,
    no_inference=False,
):
    """Main entry point for the knowledge graph generator."""
    config = load_config(config_path)
    if not config:
        print(f"Failed to load configuration from {config_path}. Exiting.")
        return

    if not input_path:
        print("Error: input_path is required.")
        return

    # Override config with command-line flags
    if no_standardize:
        config.setdefault("standardization", {})["enabled"] = False
    if no_inference:
        config.setdefault("inference", {})["enabled"] = False

    # Folder processing setup
    input_folder = input_path
    output_folder = os.path.join(output_dir, f"{experiment_name}_json_outputs")
    os.makedirs(output_folder, exist_ok=True)
    input_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

    # Load embedding model
    embedding_model = load_embedding_model("all-MiniLM-L6-v2")

    def process_wrapper(index, file_name):
        print("=" * 50)
        print(f"Processing file {index + 1}/{len(input_files)}: {file_name}")
        print("=" * 50)

        input_file = os.path.join(input_folder, file_name)
        output_file_name = f"{os.path.splitext(file_name)[0]}.json"
        output_file_path = os.path.join(output_folder, output_file_name)

        if os.path.exists(output_file_path):
            print(f"Skipping {output_file_name} as it already exists.")
            return

        # Pass the driver to the processing function
        process_file(config, input_file, output_file_path, embedding_model, debug)

    max_workers = min(6, len(input_files))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_wrapper, index, file_name)
            for index, file_name in enumerate(input_files)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred during file processing: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Knowledge Graph Generator (Embeddings + Triples)")
    # ... (parser arguments are unchanged) ...
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input folder containing text files",
    )
    parser.add_argument("--output", "-o", required=True, help="Output directory path")
    parser.add_argument(
        "--config", "-c", default="config.toml", help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment_name",
        "-e",
        default="sclc_main_exp",
        help="Name of the experiment for output files",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug output"
    )
    parser.add_argument(
        "--no-standardize", action="store_true", help="Disable entity standardization"
    )
    parser.add_argument(
        "--no-inference", action="store_true", help="Disable relationship inference"
    )
    args = parser.parse_args()

    main_kg_builder(
        args.input,
        args.output,
        config_path=args.config,
        experiment_name=args.experiment_name,
        debug=args.debug,
        no_standardize=args.no_standardize,
        no_inference=args.no_inference,
    )