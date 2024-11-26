"""API main module."""

import os

from fastapi import Body, FastAPI, HTTPException
from rag import FAQRAG

# Environment variables and model settings
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_MAX_TOKENS = 8192
COMPLETION_MODEL = "gpt-4o-mini"
COMPLETION_CONTEXT_WINDOW = 128000

# Initialize the DocsRAG object with the given settings
rag = FAQRAG(
    api_key=OPENAI_API_KEY,
    embedding_model=EMBEDDING_MODEL,
    embedding_max_tokens=EMBEDDING_MAX_TOKENS,
    completion_model=COMPLETION_MODEL,
    completion_context_window=COMPLETION_CONTEXT_WINDOW,
)

# Create a FastAPI instance
app = FastAPI()


@app.post("/chat")
async def chat(request: dict[str, str] = Body(...)) -> str:
    """
    Generate a chat response based on input content using the RAG system.

    :param request: A dictionary containing the input content for generating the response.
                    The dictionary must include a "content" key with a string value.
    :returns response: The generated chat response as a string.

    Example:
        To use this endpoint, send a POST request with a JSON body containing the "content" field:
        {
            "content": "What are the considerations for large Kubernetes clusters?"
        }
        The response will be a string generated based on the retrieved documents.
    """
    try:
        # Extract content from the request body
        content = request.get("content")

        # Check if the content is provided
        if not content:
            raise HTTPException(status_code=400, detail='"content" field is required.')

        # Check if the token limit is within acceptable range
        valid = rag.check_token_limit(content=content)

        # If the content exceeds the token limit, summarize the chat history to reduce its size.
        # This ensures that the input fits within the model's context window before generating a response.
        if not valid:
            rag.summarize_chat_history()

        # Perform a similarity search and generate a chat response
        search_df = rag.get_similarity_search(content=content)
        response = rag.create_chat_response(search_df=search_df, content=content)

        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/clear")
async def clear() -> None:
    """Clear the current chat history."""
    rag.clear_chat_history()


@app.get("/health")
async def health() -> dict[str, str]:
    """
    Simple health check endpoint.

    :returns: A dictionary indicating the health status of the API.
    """
    return {"status": "healthy"}
