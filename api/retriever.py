"""FAISS-based retriever for document similarity search.

This module provides a FAISS retriever that replaces the adalflow FAISSRetriever.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Union

from api.types import Document, RetrieverOutput

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """
    FAISS-based retriever for semantic similarity search.

    This retriever uses FAISS to build an index of document embeddings
    and performs efficient similarity search for retrieval.
    """

    def __init__(
        self,
        top_k: int = 5,
        embedder: Optional[Any] = None,
        documents: Optional[List[Document]] = None,
        document_map_func: Optional[Callable[[Document], List[float]]] = None,
        metric: str = "inner_product",  # or "l2" for L2 distance
    ):
        """
        Initialize the FAISS retriever.

        Args:
            top_k: Number of documents to retrieve
            embedder: Embedder to use for query embedding
            documents: List of documents to index
            document_map_func: Function to extract embedding from document
            metric: Distance metric to use ("inner_product" or "l2")
        """
        self.top_k = top_k
        self.embedder = embedder
        self.documents = documents or []
        self.document_map_func = document_map_func or (lambda doc: doc.vector)
        self.metric = metric
        self.index = None
        self._embedding_dim = None

        # Build index if documents are provided
        if self.documents:
            self._build_index()

    def _build_index(self):
        """Build the FAISS index from documents."""
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")

        if not self.documents:
            logger.warning("No documents to index")
            return

        # Extract embeddings from documents
        embeddings = []
        for doc in self.documents:
            try:
                vector = self.document_map_func(doc)
                if vector is not None:
                    # Convert to numpy array if needed
                    if isinstance(vector, list):
                        vector = np.array(vector, dtype=np.float32)
                    elif hasattr(vector, 'numpy'):
                        vector = vector.numpy().astype(np.float32)
                    else:
                        vector = np.array(vector, dtype=np.float32)
                    embeddings.append(vector)
                else:
                    logger.warning(f"Document has no embedding vector")
            except Exception as e:
                logger.error(f"Error extracting embedding: {e}")

        if not embeddings:
            raise ValueError("No valid embeddings found in documents")

        # Stack embeddings into a matrix
        embeddings_matrix = np.vstack(embeddings).astype(np.float32)
        self._embedding_dim = embeddings_matrix.shape[1]

        logger.info(f"Building FAISS index with {len(embeddings)} documents, dim={self._embedding_dim}")

        # Create FAISS index based on metric
        if self.metric == "inner_product":
            # Normalize vectors for cosine similarity via inner product
            faiss.normalize_L2(embeddings_matrix)
            self.index = faiss.IndexFlatIP(self._embedding_dim)
        else:
            # L2 distance
            self.index = faiss.IndexFlatL2(self._embedding_dim)

        # Add embeddings to index
        self.index.add(embeddings_matrix)
        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors")

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string."""
        if self.embedder is None:
            raise ValueError("No embedder provided for query embedding")

        # Get embedding from embedder
        result = self.embedder(query)

        # Handle different return types
        if hasattr(result, 'data') and result.data:
            # EmbedderOutput format
            embedding = result.data[0].embedding
        elif isinstance(result, dict) and 'embedding' in result:
            embedding = result['embedding']
        elif isinstance(result, (list, np.ndarray)):
            embedding = result
        else:
            raise ValueError(f"Unexpected embedder output type: {type(result)}")

        # Convert to numpy array
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        else:
            embedding = np.array(embedding, dtype=np.float32)

        return embedding.reshape(1, -1)

    def __call__(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[RetrieverOutput]:
        """
        Retrieve documents similar to the query.

        Args:
            query: The query string
            top_k: Number of documents to retrieve (overrides default)

        Returns:
            List containing a single RetrieverOutput with results
        """
        if self.index is None:
            raise ValueError("Index not built. Provide documents or call _build_index()")

        k = top_k or self.top_k

        # Embed the query
        query_embedding = self._embed_query(query)

        # Normalize for cosine similarity if using inner product
        if self.metric == "inner_product":
            import faiss
            faiss.normalize_L2(query_embedding)

        # Search the index
        scores, indices = self.index.search(query_embedding, k)

        # Create output
        doc_indices = indices[0].tolist()
        doc_scores = scores[0].tolist()

        # Filter out invalid indices (-1 means no result)
        valid_results = [
            (idx, score) for idx, score in zip(doc_indices, doc_scores)
            if idx >= 0
        ]

        doc_indices = [idx for idx, _ in valid_results]
        doc_scores = [score for _, score in valid_results]

        # Get documents
        documents = [self.documents[idx] for idx in doc_indices]

        output = RetrieverOutput(
            doc_indices=doc_indices,
            doc_scores=doc_scores,
            documents=documents,
            query=query
        )

        return [output]

    def add_documents(self, documents: List[Document]):
        """
        Add documents to the index.

        Args:
            documents: List of documents to add
        """
        self.documents.extend(documents)
        self._build_index()
