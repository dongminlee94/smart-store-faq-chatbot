"""Embedder."""

import os

import pandas as pd
import tiktoken
from openai import OpenAI
from tqdm import tqdm


class FAQEmbedder:
    """
    FAQ Embedder.

    This class provides methods for reading FAQ data, processing it into embedding inputs,
    generating embeddings using OpenAI API, and saving the results to a vector database.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialize the FAQEmbedder.

        :param api_key: The API key for accessing OpenAI services.
        """
        self._openai_client = OpenAI(api_key=api_key)

        self._input_filename = os.path.join(os.path.dirname(__file__), "../..", "data/final_result.pkl")
        self._output_filename = os.path.join(os.path.dirname(__file__), "../..", "data/vector_db.parquet")

    def _read_faq_data(self) -> pd.DataFrame:
        """
        Read FAQ data from a pickle file and convert it into a DataFrame.

        :returns: A pandas DataFrame containing FAQ data with 'question' and 'answer' columns.
        """
        # Load the pickle file containing FAQ data
        data = pd.read_pickle(self._input_filename)

        # Convert the data into a DataFrame with 'question' and 'answer' columns
        return pd.DataFrame(list(data.items()), columns=["question", "answer"])

    @staticmethod
    def make_chunks(data: str | list[int], length: int) -> list[str]:
        """
        Split the data into smaller chunks of a specified length.

        :param data: The input data, which can be a string or a list of integers (tokens).
        :param length: The maximum length of each chunk.
        :returns: A list of chunks.

        Example:
            >>> data = [1, 2, 3, 4, 5, 6]
            >>> length = 3
            >>> FAQEmbedder.make_chunks(data, length)
            [[1, 2, 3], [4, 5, 6]]
        """
        return [data[i : i + length] for i in range(0, len(data), length)]

    def _get_embedding_input(self, df: pd.DataFrame, model: str, max_tokens: int) -> pd.DataFrame:
        """
        Prepare embedding input data by combining questions and answers.

        :param df: A DataFrame containing the FAQ data.
        :param model: The name of the OpenAI model used for encoding tokens.
        :param max_tokens: The maximum number of tokens allowed for each embedding input chunk.
        :returns: A DataFrame with additional columns for embedding input and empty embedding output.

        This method tokenizes the combined 'question' and 'answer' text and splits it into smaller chunks,
        ensuring that the token count in each chunk does not exceed max_tokens.
        """
        # Combine 'question' and 'answer' columns to create embedding input
        df["embedding_input"] = df["question"] + " " + df["answer"]

        # Initialize the tokenizer for the specified model
        encoder = tiktoken.encoding_for_model(model_name=model)

        # Iterate over rows and process each embedding input
        rows = []
        for row in df.itertuples():
            tokens = encoder.encode(text=row.embedding_input)

            # Split tokens into chunks and decode them back to text
            for chunk in self.make_chunks(data=tokens, length=max_tokens):
                text = encoder.decode(tokens=chunk)

                # Append the processed data to rows
                rows.append(
                    {
                        "question": row.question,
                        "answer": row.answer,
                        "embedding_input": text,
                        "embedding_output": None,
                    },
                )

        return pd.DataFrame(data=rows)

    def _get_embedding_output(
        self,
        df: pd.DataFrame,
        model: str,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Generate embeddings for the input data using the specified model.

        :param df: A DataFrame containing embedding inputs.
        :param model: The name of the OpenAI model used for generating embeddings.
        :param verbose: If True, displays a progress bar during the embedding process.
        :returns: A DataFrame with embeddings added to the 'embedding_output' column.

        This method uses OpenAI's API to generate embeddings for each row of the input data.
        """
        embedding_outputs = []
        for row in tqdm(df.itertuples()) if verbose else df.itertuples():
            # Generate embeddings for the input text
            embedding_output = (
                self._openai_client.embeddings.create(input=row.embedding_input, model=model)
                .data[0]
                .embedding
            )

            embedding_outputs.append(embedding_output)

        # Add the embeddings to the DataFrame
        df["embedding_output"] = embedding_outputs

        return df

    def embed_faq(
        self,
        model: str,
        max_tokens: int,
        verbose: bool = False,
    ) -> None:
        """
        Embed FAQ data and save the results as a vector database.

        This method reads FAQ data, generates embedding inputs and outputs, and saves the vector database.
        If the vector database file already exists, the process is skipped.

        :param model: The name of the OpenAI model used for generating embeddings.
        :param max_tokens: The maximum number of tokens for each embedding input chunk.
        :param verbose: If True, displays a progress bar while generating embeddings.

        Example:
            >>> embedder = FAQEmbedder(api_key="your_api_key")
            >>> embedder.embed_faq(model="text-embedding-3-large", max_tokens=8192, verbose=True)
        """
        # Check if the output file already exists
        if os.path.exists(self._output_filename):
            print(
                f"The embedding file '{self._output_filename}' already exists."
                " Skipping the embedding process."
            )
            return

        print("Reading FAQ data...")
        faq_df = self._read_faq_data()
        print(f"Successfully loaded {len(faq_df)} FAQ entries.\n")

        print("Generating embedding inputs...")
        embedded_df = self._get_embedding_input(df=faq_df, model=model, max_tokens=max_tokens)
        print(f"Generated embedding inputs for {len(embedded_df)} chunks.\n")

        print("Generating embedding outputs...")
        vector_db_df = self._get_embedding_output(df=embedded_df, model=model, verbose=verbose)
        print(f"Successfully generated embeddings for {len(vector_db_df)} entries.\n")

        print(f"Saving embeddings to '{self._output_filename}'...")
        vector_db_df.to_parquet(path=self._output_filename, index=False)
        print(f"Embedding process completed. Data saved to '{self._output_filename}'.")
