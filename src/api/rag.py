"""Retrieval-Augmented Generation (RAG)."""

import os
import time

import faiss
import numpy as np
import pandas as pd
import tiktoken
import yaml
from client import OpenAIClient


class FAQRAG:
    """
    FAQ Retrieval-Augmented Generation (RAG).

    This class integrates OpenAI's GPT models with a FAISS vector database to provide RAG capabilities.
    """

    def __init__(
        self,
        api_key: str,
        embedding_model: str,
        embedding_max_tokens: int,
        completion_model: str,
        completion_context_window: int,
    ) -> None:
        """
        Initialize the FAQRAG instance.

        :param api_key: The API key for accessing OpenAI services.
        :param embedding_model: The model used for generating embeddings.
        :param embedding_max_tokens: The maximum tokens allowed for embedding input.
        :param completion_model: The model used for generating completions.
        :param completion_context_window: The context window size for the completion model.
        """
        # Initialize OpenAI client
        self._openai_client = OpenAIClient(api_key=api_key)

        # Store model configurations
        self._embedding_model = embedding_model
        self._embedding_max_tokens = embedding_max_tokens
        self._completion_model = completion_model
        self._completion_context_window = completion_context_window

        # Load vector database and Faiss index
        self._vector_db = self._get_vector_db()
        self._index = self._get_index()

        # Load prompts for response generation and summarization
        self._response_prompt = self._load_prompt(prompt_filename="response.yaml")
        self._summarizer_prompt = self._load_prompt(prompt_filename="summary.yaml")

        # Initialize chat history
        self.clear_chat_history()

    def _get_vector_db(self) -> pd.DataFrame:
        """
        Load the vector database from a Parquet file.

        :returns: A pandas DataFrame containing the vector database.
        :raises FileNotFoundError: If the Parquet file is not found.
        """
        filepath = os.path.join(os.path.dirname(__file__), "../..", "data/vector_db.parquet")

        # Wait for the file to exist, retrying every 10 seconds
        while not os.path.exists(filepath):
            print("The vector_db.parquet file is not found. Please wait while it is being created...")
            time.sleep(10)

        return pd.read_parquet(filepath)

    def _get_index(self) -> faiss.IndexIDMap:
        """
        Build a Faiss index from the vector database.

        :returns: A Faiss IndexIDMap object with the embeddings added.
        """
        # Convert embedding outputs to a Numpy array
        embedding_data = np.array(list(self._vector_db["embedding_output"]))

        # Initialize a Faiss index for inner product similarity
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_data.shape[1]))

        # Add embeddings and their corresponding IDs to the index
        index.add_with_ids(embedding_data, np.array(range(0, len(self._vector_db))))

        return index

    def _validate_prompt(self, prompt: dict[str, str]) -> None:
        """
        Validate the format of the prompt.

        :param prompt: The prompt data to validate.
        :raises ValueError: If the prompt format is invalid.
        """
        if not isinstance(prompt, list) or not all(
            isinstance(piece, str) and len(msg) == 2 for msg in prompt for piece in msg
        ):
            raise ValueError('The prompt must be a list of pairs like [["role", "content"]].')

    def _load_prompt(self, prompt_filename: str) -> dict[str, str]:
        """
        Load a YAML prompt file.

        :param prompt_filename: The filename of the prompt file to load.
        :returns: A list of role-content pairs from the prompt file.
        :raises FileNotFoundError: If the prompt file is not found.
        """
        # Construct the file path for the prompt
        prompt_filepath = os.path.join(os.path.dirname(__file__), "../../prompt", f"{prompt_filename}")

        # Check if the prompt file exists
        if not os.path.exists(prompt_filepath):
            raise FileNotFoundError(f"Prompt file '{prompt_filename}' not found at {prompt_filepath}.")

        # Load the prompt file
        with open(prompt_filepath, "r", encoding="utf-8") as fp:
            prompt = yaml.safe_load(fp)

        # Validate the format of the loaded prompt
        self._validate_prompt(prompt=prompt)

        return prompt

    def clear_chat_history(self) -> None:
        """
        Clear the chat history.

        This method resets the chat history to the initial system prompt.

        Example:
            >>> rag = FAQRAG(api_key="your_api_key", ...)
            >>> rag.clear_chat_history()
            >>> print(rag.chat)  # Should reset to the initial system prompt
        """
        self.chat = [{"role": msg[0], "content": msg[1]} for msg in self._response_prompt]

    def check_token_limit(self, content: str) -> bool:
        """
        Check if the content exceeds the token limit for embedding or completion.

        :param content: The content to check against token limits.
        :returns: True if the content is within the token limits; False otherwise.

        Example:
            >>> rag = FAQRAG(api_key="your_api_key", ...)
            >>> content = "This is a sample content for testing token limits."
            >>> within_limit = rag.check_token_limit(content=content)
            >>> print(within_limit)  # True if within limit, False otherwise
        """
        # Combine the chat history content and the new input
        contents = "".join([chat["content"] for chat in self.chat]) + content

        # Define the token limits for each model
        models_limits = [
            (self._embedding_model, self._embedding_max_tokens, content),
            (self._completion_model, self._completion_context_window, contents),
        ]

        # Check each model's token limit
        for model, limit, text in models_limits:
            encoder = tiktoken.encoding_for_model(model_name=model)

            if len(encoder.encode(text=text)) > limit:
                return False

        return True

    def _create_chat_summary(
        self,
        chat: list[dict[str, str]],
        response_format: dict[str, str] = {"type": "text"},
        temperature: int = 0,
    ) -> str:
        """
        Create a summary of the chat history.

        :param chat: The chat history to summarize.
        :param response_format: The response format for the summarization model.
        :param temperature: The temperature for the summarization model.
        :returns: The summarized chat history as a string.
        """
        # Format chat history into a structured string
        chat_history = ""
        count = 1
        for msg in chat:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                chat_history += f"USER-ASSISTANT PAIR {count}\n{role}\n{content}\n"
            elif role == "assistant":
                chat_history += f"{role}\n{content}\n\n"
                count += 1
            else:  # "system"
                continue

        # Prepare messages for summarization
        messages = [{"role": msg[0], "content": msg[1]} for msg in self._summarizer_prompt]
        messages.append({"role": "user", "content": f"--- CHAT HISTORY ---\n{chat_history}"})

        # Call OpenAI API to generate a summary
        chat_summary = self._openai_client.create_completion(
            messages=messages,
            model=self._completion_model,
            response_format=response_format,
            temperature=temperature,
        )

        return chat_summary

    def summarize_chat_history(
        self, response_format: dict[str, str] = {"type": "text"}, temperature: int = 0
    ) -> None:
        """
        Summarize the chat history.

        This method summarizes the chat history and recent exchanges, replacing them with concise summaries.

        :param response_format: The response format for the summarization model.
        :param temperature: The temperature for the summarization model.

        Example:
            >>> rag = FAQRAG(api_key="your_api_key", ...)
            >>> rag.chat = [
            ...     {"role": "user", "content": "What is Kubernetes?"},
            ...     {"role": "assistant", "content": "Kubernetes is a container orchestration platform."},
            ...     {"role": "user", "content": "How do I use it?"},
            ...     {"role": "assistant", "content": "You can use it with a YAML configuration file."}
            ... ]
            >>> rag.summarize_chat_history()
            >>> print(rag.chat)  # Chat history should be summarized
        """
        # Split the chat history into recent and old chats
        recent_chat = self.chat[-2:]
        self.chat = self.chat[:-2]

        # Generate summaries for old and recent chat histories
        old_chat_summary = self._create_chat_summary(
            chat=self.chat, response_format=response_format, temperature=temperature
        )
        recent_chat_summary = self._create_chat_summary(
            chat=recent_chat, response_format=response_format, temperature=temperature
        )

        # Replace chat history with the summarized content
        self.clear_chat_history()
        self.chat.append(
            {
                "role": "system",
                "content": f"--- PAST CHAT SUMMARY ---\n{old_chat_summary}\n{recent_chat_summary}",
            }
        )

    def get_similarity_search(self, content: str, top_k: int = 5) -> pd.DataFrame:
        """
        Perform a similarity search on the vector database.

        :param content: The input content to search for similar entries.
        :param top_k: The number of top results to retrieve.
        :returns: A pandas DataFrame containing the top-k similar entries.

        Example:
            >>> rag = FAQRAG(api_key="your_api_key", ...)
            >>> content = "How do I set up Kubernetes?"
            >>> results = rag.get_similarity_search(content=content, top_k=3)
            >>> print(results)
        """
        # Generate embedding for the input content
        embedding_output = self._openai_client.create_embedding(text=content, model=self._embedding_model)

        # Perform a search in the Faiss index to find the top-k closest embeddings
        search_result = self._index.search(np.array([embedding_output]), top_k)

        # Retrieve the corresponding rows from the vector database
        search_df = self._vector_db.iloc[list(search_result[1][0])].reset_index(drop=True)

        # Remove duplicate entries based on 'question' and 'answer' columns
        search_df.drop_duplicates(subset=["question", "answer"], keep="first", inplace=True)

        return search_df

    def create_chat_response(
        self,
        search_df: pd.DataFrame,
        content: str,
        response_format: dict[str, str] = {"type": "text"},
        temperature: int = 0,
    ) -> str:
        """
        Generate a chat response based on similarity search results.

        :param search_df: A DataFrame containing similar FAQ entries.
        :param content: The user query to respond to.
        :param response_format: The response format for the completion model.
        :param temperature: The temperature for the completion model.
        :returns: The generated chat response as a string.

        Example:
            >>> rag = FAQRAG(api_key="your_api_key", ...)
            >>> search_df = rag.get_similarity_search(content="How do I scale Kubernetes?", top_k=3)
            >>> response = rag.create_chat_response(search_df=search_df, content="How do I scale Kubernetes?")
            >>> print(response)
        """
        # Format the FAQ data into a structured string for the prompt
        faq = "--- RELATED INTERNAL FAQ ---\n"
        for row in search_df.itertuples():
            faq += f"제공된 FAQ 데이터 {row.Index + 1}\n"
            faq += f" - Question: {row.question}\n"
            faq += f" - Answer: {row.answer}\n\n"

        # Append the FAQ data and user query to the chat history
        self.chat.append({"role": "system", "content": faq})
        self.chat.append({"role": "user", "content": f"--- QUESTION ---\n{content}"})

        # Use the OpenAI API to generate a response based on the chat history
        response = self._openai_client.create_completion(
            messages=self.chat,
            model=self._completion_model,
            response_format=response_format,
            temperature=temperature,
        )

        # Append the generated response to the chat history
        self.chat.append({"role": "assistant", "content": response})

        return response
