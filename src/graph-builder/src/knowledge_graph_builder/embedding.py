"""
Embedding generation module.
"""
import torch
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Loads and returns a SentenceTransformer model, using a GPU if available.

    This function should be called once to initialize the model.

    Args:
        model_name: The name of the model to load (e.g., 'all-MiniLM-L6-v2').

    Returns:
        An instance of the SentenceTransformer model, or None if an error occurs.
    """
    # Check if a CUDA-enabled GPU is available, otherwise fall back to CPU
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        print(f"Loading embedding model: {model_name}...")
        # Load the model and send it to the specified device
        model = SentenceTransformer(model_name, device=device)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None


def get_embedding(text: str, model: SentenceTransformer) -> list[float]:
    """
    Generates a vector embedding for the given text using a pre-loaded model.

    Args:
        text: The input string to embed.
        model: The pre-loaded SentenceTransformer model instance.

    Returns:
        A list of floats representing the vector embedding, or an empty list on error.
    """
    # Ensure a model object was successfully passed
    if not isinstance(model, SentenceTransformer):
        print("Error: A valid SentenceTransformer model must be provided.")
        return []

    try:
        # Pre-process the text
        text_to_embed = text.replace("\n", " ")
        if not text_to_embed:
            return []

        # Generate the embedding
        embedding_array = model.encode(text_to_embed)

        # Convert numpy array to a standard Python list
        return embedding_array.tolist()

    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []
