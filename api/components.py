"""Components for data processing using cocoindex.

This module provides text splitting, embedding, and database functionality
using cocoindex's transformation functions where available.
"""

import os
import pickle
import logging
from copy import deepcopy
from typing import List, Dict, Any, Optional, Sequence, Callable
from dataclasses import dataclass, field

from api.types import Document, EmbedderOutput, EmbeddingData

logger = logging.getLogger(__name__)

# Try to import cocoindex for advanced text splitting
try:
    import cocoindex
    COCOINDEX_AVAILABLE = True
    logger.info("cocoindex is available for text splitting")
except ImportError:
    COCOINDEX_AVAILABLE = False
    logger.warning("cocoindex not available, using fallback text splitter")


def get_default_root_path() -> str:
    """
    Get the default root path for storing data.

    Returns:
        str: The default root path (~/.adalflow for compatibility)
    """
    return os.path.expanduser(os.path.join("~", ".adalflow"))


class TextSplitter:
    """
    Split text into smaller chunks using cocoindex's SplitRecursively when available.

    This class provides text splitting functionality that leverages cocoindex's
    intelligent recursive splitting algorithm which respects document structure
    (headings, paragraphs, sentences) for better chunk boundaries.

    Falls back to a simple implementation if cocoindex is not available.
    """

    def __init__(
        self,
        split_by: str = "word",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        language: str = "markdown",
    ):
        """
        Initialize the text splitter.

        Args:
            split_by: How to split the text ("word", "sentence", "paragraph")
                     - Used for fallback mode only
            chunk_size: Maximum size of each chunk (in characters for cocoindex)
            chunk_overlap: Number of characters to overlap between chunks
            language: Language/format hint for cocoindex ("markdown", "python", etc.)
        """
        self.split_by = split_by
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self._use_cocoindex = COCOINDEX_AVAILABLE

    def _split_with_cocoindex(self, text: str) -> List[str]:
        """
        Split text using cocoindex's SplitRecursively function.

        cocoindex.functions.SplitRecursively intelligently splits text by trying
        higher-level boundaries first (sections, paragraphs) before falling back
        to lower-level boundaries (sentences, words).
        """
        try:
            # Use cocoindex's SplitRecursively function
            split_fn = cocoindex.functions.SplitRecursively(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            # Apply the transformation
            chunks = split_fn(text, language=self.language)

            # Handle the result - cocoindex may return different formats
            if isinstance(chunks, list):
                return [str(chunk) for chunk in chunks]
            elif hasattr(chunks, '__iter__'):
                return [str(chunk) for chunk in chunks]
            else:
                return [str(chunks)]
        except Exception as e:
            logger.warning(f"cocoindex SplitRecursively failed: {e}, falling back to simple split")
            return self._split_fallback(text)

    def _split_fallback(self, text: str) -> List[str]:
        """Fallback text splitting when cocoindex is not available."""
        parts = self._split_text(text)
        chunks = []

        current_chunk = []
        current_size = 0

        for part in parts:
            part_size = len(part)

            if current_size + part_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self._join_chunks(current_chunk))

                # Keep overlap
                overlap_chars = 0
                overlap_parts = []
                for p in reversed(current_chunk):
                    if overlap_chars + len(p) <= self.chunk_overlap:
                        overlap_parts.insert(0, p)
                        overlap_chars += len(p)
                    else:
                        break
                current_chunk = overlap_parts
                current_size = overlap_chars

            current_chunk.append(part)
            current_size += part_size

        # Add the last chunk
        if current_chunk:
            chunks.append(self._join_chunks(current_chunk))

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text based on the split_by parameter."""
        if self.split_by == "word":
            return text.split()
        elif self.split_by == "sentence":
            import re
            return re.split(r'(?<=[.!?])\s+', text)
        elif self.split_by == "paragraph":
            return text.split("\n\n")
        else:
            return text.split(self.split_by)

    def _join_chunks(self, parts: List[str]) -> str:
        """Join parts back into text."""
        if self.split_by == "word":
            return " ".join(parts)
        elif self.split_by == "sentence":
            return " ".join(parts)
        elif self.split_by == "paragraph":
            return "\n\n".join(parts)
        else:
            return self.split_by.join(parts)

    def split(self, text: str) -> List[str]:
        """
        Split a single text into chunks.

        Uses cocoindex's SplitRecursively when available for intelligent
        splitting that respects document structure.

        Args:
            text: The text to split

        Returns:
            List of text chunks
        """
        if self._use_cocoindex:
            return self._split_with_cocoindex(text)
        else:
            return self._split_fallback(text)

    def __call__(self, documents: Sequence[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents: Sequence of Document objects

        Returns:
            List of Document objects with split text
        """
        result = []

        for doc in documents:
            # Detect language from file extension if available
            file_path = doc.meta_data.get("file_path", "")
            language = self._detect_language(file_path)

            # Temporarily set language for this document
            original_language = self.language
            self.language = language

            chunks = self.split(doc.text)

            # Restore original language
            self.language = original_language

            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    text=chunk,
                    meta_data={
                        **doc.meta_data,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "parent_id": doc.id or doc.meta_data.get("file_path", "unknown"),
                    },
                    id=f"{doc.id or doc.meta_data.get('file_path', 'doc')}_{i}" if doc.id or doc.meta_data.get('file_path') else None,
                )
                result.append(new_doc)

        return result

    def _detect_language(self, file_path: str) -> str:
        """Detect the language/format from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".md": "markdown",
            ".rst": "rst",
            ".txt": "text",
            ".html": "html",
            ".css": "css",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".xml": "xml",
            ".sql": "sql",
            ".sh": "bash",
            ".bash": "bash",
        }
        ext = os.path.splitext(file_path)[1].lower()
        return ext_map.get(ext, "text")


class CocoIndexEmbedder:
    """
    Create embeddings using cocoindex's SentenceTransformerEmbed.

    This uses cocoindex's built-in embedding function which leverages
    the sentence-transformers library for high-quality text embeddings.
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the cocoindex embedder.

        Args:
            model: The sentence-transformers model to use
        """
        self.model = model
        self._embed_fn = None

        if COCOINDEX_AVAILABLE:
            try:
                self._embed_fn = cocoindex.functions.SentenceTransformerEmbed(model=model)
                logger.info(f"Using cocoindex SentenceTransformerEmbed with model: {model}")
            except Exception as e:
                logger.warning(f"Failed to initialize cocoindex embedder: {e}")
                self._embed_fn = None

    def __call__(self, text: str) -> EmbedderOutput:
        """
        Create embeddings for the input text.

        Args:
            text: The text to embed

        Returns:
            EmbedderOutput with embedding data
        """
        if self._embed_fn is None:
            raise RuntimeError("cocoindex embedder not available")

        try:
            embedding = self._embed_fn(text)

            # Convert to list if necessary
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                embedding = list(embedding)

            return EmbedderOutput(
                data=[EmbeddingData(embedding=embedding, index=0)],
                raw_response=embedding,
            )
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return EmbedderOutput(
                data=[],
                error=str(e),
            )


class Embedder:
    """
    Create embeddings for text using a model client.

    This wraps a model client to provide embedding functionality.
    Can also use cocoindex's SentenceTransformerEmbed as a fallback.
    """

    def __init__(
        self,
        model_client: Any = None,
        model_kwargs: Dict[str, Any] = None,
        use_cocoindex: bool = False,
        cocoindex_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the embedder.

        Args:
            model_client: The model client to use for embeddings
            model_kwargs: Additional kwargs for the model
            use_cocoindex: Whether to use cocoindex's SentenceTransformerEmbed
            cocoindex_model: The model to use with cocoindex embedder
        """
        self.model_client = model_client
        self.model_kwargs = model_kwargs or {}
        self.use_cocoindex = use_cocoindex and COCOINDEX_AVAILABLE

        if self.use_cocoindex:
            self._cocoindex_embedder = CocoIndexEmbedder(model=cocoindex_model)
        else:
            self._cocoindex_embedder = None

    def __call__(self, input: str) -> EmbedderOutput:
        """
        Create embeddings for the input text.

        Args:
            input: The text to embed

        Returns:
            EmbedderOutput with embedding data
        """
        # Try cocoindex first if enabled
        if self.use_cocoindex and self._cocoindex_embedder is not None:
            try:
                return self._cocoindex_embedder(input)
            except Exception as e:
                logger.warning(f"cocoindex embedding failed: {e}, falling back to model client")

        # Use model client
        if self.model_client is None:
            raise ValueError("No model client or cocoindex embedder available")

        from api.types import ModelType

        api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
            input=input,
            model_kwargs=self.model_kwargs,
            model_type=ModelType.EMBEDDER
        )

        result = self.model_client.call(api_kwargs, model_type=ModelType.EMBEDDER)

        # Handle different return types
        if isinstance(result, EmbedderOutput):
            return result
        else:
            # Try to parse the embedding response
            return self.model_client.parse_embedding_response(result)


class ToEmbeddings:
    """
    Transform documents by adding embeddings.

    This processes documents in batches and adds embedding vectors.
    """

    def __init__(
        self,
        embedder: Embedder,
        batch_size: int = 100,
    ):
        """
        Initialize the ToEmbeddings transformer.

        Args:
            embedder: The embedder to use
            batch_size: Number of documents to process at once
        """
        self.embedder = embedder
        self.batch_size = batch_size

    def __call__(self, documents: Sequence[Document]) -> List[Document]:
        """
        Add embeddings to documents.

        Args:
            documents: Sequence of Document objects

        Returns:
            List of Document objects with embeddings
        """
        from tqdm import tqdm

        output = list(deepcopy(documents))
        logger.info(f"Creating embeddings for {len(output)} documents in batches of {self.batch_size}")

        successful_docs = []
        expected_embedding_size = None

        # Process in batches
        for i in tqdm(range(0, len(output), self.batch_size), desc="Creating embeddings"):
            batch = output[i:i + self.batch_size]
            texts = [doc.text for doc in batch]

            try:
                for j, text in enumerate(texts):
                    result = self.embedder(text)

                    if result.data and len(result.data) > 0:
                        embedding = result.data[0].embedding

                        # Validate embedding size
                        if expected_embedding_size is None:
                            expected_embedding_size = len(embedding)
                            logger.info(f"Expected embedding size: {expected_embedding_size}")
                        elif len(embedding) != expected_embedding_size:
                            file_path = batch[j].meta_data.get('file_path', f'doc_{i+j}')
                            logger.warning(f"Document '{file_path}' has inconsistent embedding size {len(embedding)}, skipping")
                            continue

                        batch[j].vector = embedding
                        successful_docs.append(batch[j])
                    else:
                        file_path = batch[j].meta_data.get('file_path', f'doc_{i+j}')
                        logger.warning(f"No embedding returned for document '{file_path}'")

            except Exception as e:
                logger.error(f"Error creating embeddings for batch {i}: {e}")
                # Try individual documents
                for j, text in enumerate(texts):
                    try:
                        result = self.embedder(text)
                        if result.data and len(result.data) > 0:
                            batch[j].vector = result.data[0].embedding
                            successful_docs.append(batch[j])
                    except Exception as e2:
                        file_path = batch[j].meta_data.get('file_path', f'doc_{i+j}')
                        logger.error(f"Error embedding document '{file_path}': {e2}")

        logger.info(f"Successfully embedded {len(successful_docs)}/{len(output)} documents")
        return successful_docs


class Sequential:
    """
    Chain multiple transformers together.

    This allows composing multiple transformation steps.
    """

    def __init__(self, *transformers):
        """
        Initialize with a sequence of transformers.

        Args:
            *transformers: Variable number of transformer objects
        """
        self.transformers = transformers

    def __call__(self, data: Any) -> Any:
        """
        Apply all transformers in sequence.

        Args:
            data: Input data

        Returns:
            Transformed data after all transformers
        """
        result = data
        for transformer in self.transformers:
            result = transformer(result)
        return result


class LocalDB:
    """
    Local database for storing and loading documents with embeddings.

    This provides simple persistence using pickle files.
    """

    def __init__(self):
        """Initialize the database."""
        self._data: List[Document] = []
        self._transformers: Dict[str, Any] = {}
        self._transformed_data: Dict[str, List[Document]] = {}

    def load(self, documents: List[Document]) -> None:
        """
        Load documents into the database.

        Args:
            documents: List of Document objects
        """
        self._data = list(documents)
        logger.info(f"Loaded {len(self._data)} documents into database")

    def register_transformer(self, transformer: Any, key: str) -> None:
        """
        Register a transformer with a key.

        Args:
            transformer: The transformer object
            key: Key to identify this transformer
        """
        self._transformers[key] = transformer
        logger.info(f"Registered transformer with key: {key}")

    def transform(self, key: str) -> None:
        """
        Apply a registered transformer to the data.

        Args:
            key: The key of the transformer to apply
        """
        if key not in self._transformers:
            raise ValueError(f"Transformer '{key}' not found")

        transformer = self._transformers[key]
        self._transformed_data[key] = transformer(self._data)
        logger.info(f"Transformed data with '{key}': {len(self._transformed_data[key])} documents")

    def get_transformed_data(self, key: str) -> List[Document]:
        """
        Get the transformed data for a key.

        Args:
            key: The transformer key

        Returns:
            List of transformed documents
        """
        return self._transformed_data.get(key, [])

    def save_state(self, filepath: str) -> None:
        """
        Save the database state to a file.

        Args:
            filepath: Path to save the state
        """
        state = {
            "data": self._data,
            "transformed_data": self._transformed_data,
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved database state to {filepath}")

    @classmethod
    def load_state(cls, filepath: str) -> "LocalDB":
        """
        Load the database state from a file.

        Args:
            filepath: Path to load the state from

        Returns:
            LocalDB instance with loaded state
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        db = cls()
        db._data = state.get("data", [])
        db._transformed_data = state.get("transformed_data", {})

        logger.info(f"Loaded database state from {filepath}")
        return db
