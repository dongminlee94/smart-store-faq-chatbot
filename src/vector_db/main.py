"""Vector DB main module."""

import os

from embedder import FAQEmbedder

# OpenAI API key for accessing embedding services
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_MAX_TOKENS = 8192


def main() -> None:
    """
    Main function to create a vector database from FAQ data.

    This function initializes the FAQEmbedder with the OpenAI API key and processes the FAQ data
    to generate embeddings using the specified model. The embeddings are saved to a file for
    future similarity searches and other applications.
    """
    # Initialize the FAQEmbedder with the provided API key
    embedder = FAQEmbedder(api_key=OPENAI_API_KEY)

    # Generate embeddings for the FAQ data using the specified model and token limit
    embedder.embed_faq(model=EMBEDDING_MODEL, max_tokens=EMBEDDING_MAX_TOKENS, verbose=True)


if __name__ == "__main__":
    main()
