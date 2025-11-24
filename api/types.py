"""Custom types to replace adalflow types.

This module provides the core data types used throughout the application,
replacing the adalflow dependency with custom implementations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ModelType(Enum):
    """Enum representing different types of models."""
    UNDEFINED = "undefined"
    LLM = "llm"
    EMBEDDER = "embedder"
    EMBEDDING = "embedding"  # Alias for EMBEDDER
    IMAGE_GENERATION = "image_generation"


@dataclass
class Document:
    """
    A document with text content, metadata, and optional embedding vector.

    Attributes:
        text: The text content of the document
        meta_data: Dictionary containing metadata about the document
        vector: Optional embedding vector for the document
        id: Optional unique identifier for the document
    """
    text: str = ""
    meta_data: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None
    id: Optional[str] = None

    def __post_init__(self):
        if self.meta_data is None:
            self.meta_data = {}


@dataclass
class EmbeddingData:
    """
    Container for a single embedding result.

    Attributes:
        embedding: The embedding vector
        index: The index of the embedding in the batch
    """
    embedding: List[float]
    index: int = 0


@dataclass
class EmbedderOutput:
    """
    Output from an embedding model.

    Attributes:
        data: List of embedding data objects
        error: Optional error message if embedding failed
        raw_response: The raw response from the embedding API
        model: The model used for embedding
        usage: Optional token usage information
    """
    data: List[EmbeddingData] = field(default_factory=list)
    error: Optional[str] = None
    raw_response: Any = None
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


@dataclass
class CompletionUsage:
    """
    Token usage information for a completion.

    Attributes:
        completion_tokens: Number of tokens in the completion
        prompt_tokens: Number of tokens in the prompt
        total_tokens: Total number of tokens used
    """
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class TokenLogProb:
    """
    Log probability for a token.

    Attributes:
        token: The token string
        logprob: The log probability of the token
    """
    token: str
    logprob: float


@dataclass
class GeneratorOutput:
    """
    Output from a text generation model.

    Attributes:
        data: The generated data (can be parsed structured output)
        error: Optional error message if generation failed
        raw_response: The raw response string from the model
        usage: Optional token usage information
    """
    data: Any = None
    error: Optional[str] = None
    raw_response: Any = None
    usage: Optional[CompletionUsage] = None


@dataclass
class RetrieverOutput:
    """
    Output from a retriever.

    Attributes:
        doc_indices: List of document indices
        doc_scores: List of document scores
        documents: List of retrieved documents
        query: The query used for retrieval
    """
    doc_indices: List[int] = field(default_factory=list)
    doc_scores: List[float] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    query: Optional[str] = None


class DataClass:
    """Base class for data classes with output field support."""
    __output_fields__: List[str] = []


def parse_embedding_response(response: Any) -> EmbedderOutput:
    """
    Parse an embedding response into EmbedderOutput format.

    This function handles responses from OpenAI-compatible embedding APIs.

    Args:
        response: The raw embedding API response

    Returns:
        EmbedderOutput with parsed embedding data
    """
    try:
        embeddings = []
        if hasattr(response, 'data'):
            for item in response.data:
                embeddings.append(EmbeddingData(
                    embedding=item.embedding,
                    index=item.index
                ))

        usage = None
        if hasattr(response, 'usage'):
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'total_tokens': response.usage.total_tokens
            }

        return EmbedderOutput(
            data=embeddings,
            raw_response=response,
            model=getattr(response, 'model', None),
            usage=usage
        )
    except Exception as e:
        return EmbedderOutput(
            data=[],
            error=str(e),
            raw_response=response
        )
