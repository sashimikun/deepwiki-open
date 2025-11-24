"""Base model client class and implementations for Google and Ollama.

This module provides the base model client class and concrete implementations
for Google Generative AI and Ollama, replacing the adalflow dependency.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, AsyncGenerator

from api.types import ModelType, GeneratorOutput, EmbedderOutput, EmbeddingData

log = logging.getLogger(__name__)


class ModelClient(ABC):
    """
    Abstract base class for model clients.

    All model clients should inherit from this class and implement
    the required methods for their specific API.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the model client."""
        pass

    @abstractmethod
    def convert_inputs_to_api_kwargs(
        self,
        input: Any,
        model_kwargs: Dict = None,
        model_type: ModelType = ModelType.UNDEFINED
    ) -> Dict:
        """
        Convert inputs to API-specific kwargs.

        Args:
            input: The input to the model
            model_kwargs: Additional model parameters
            model_type: The type of model operation

        Returns:
            Dict of kwargs for the API call
        """
        pass

    @abstractmethod
    def call(
        self,
        api_kwargs: Dict = None,
        model_type: ModelType = ModelType.UNDEFINED
    ) -> Any:
        """
        Make a synchronous call to the model API.

        Args:
            api_kwargs: The kwargs for the API call
            model_type: The type of model operation

        Returns:
            The API response
        """
        pass

    @abstractmethod
    async def acall(
        self,
        api_kwargs: Dict = None,
        model_type: ModelType = ModelType.UNDEFINED
    ) -> Any:
        """
        Make an asynchronous call to the model API.

        Args:
            api_kwargs: The kwargs for the API call
            model_type: The type of model operation

        Returns:
            The API response
        """
        pass


class GoogleGenAIClient(ModelClient):
    """
    Client for Google Generative AI API.

    This client wraps the google-generativeai library and provides
    a consistent interface for both LLM and embedding operations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        env_api_key_name: str = "GOOGLE_API_KEY",
        *args,
        **kwargs
    ):
        """
        Initialize the Google GenAI client.

        Args:
            api_key: Optional API key, will use environment variable if not provided
            env_api_key_name: Name of the environment variable containing the API key
        """
        super().__init__(*args, **kwargs)
        self._api_key = api_key or os.environ.get(env_api_key_name)
        self._env_api_key_name = env_api_key_name

        # Configure the API
        try:
            import google.generativeai as genai
            if self._api_key:
                genai.configure(api_key=self._api_key)
            self._genai = genai
        except ImportError:
            log.error("google-generativeai package not installed")
            self._genai = None

    def convert_inputs_to_api_kwargs(
        self,
        input: Any,
        model_kwargs: Dict = None,
        model_type: ModelType = ModelType.UNDEFINED
    ) -> Dict:
        """Convert inputs to Google GenAI API kwargs."""
        model_kwargs = model_kwargs or {}
        api_kwargs = model_kwargs.copy()

        if model_type == ModelType.EMBEDDER or model_type == ModelType.EMBEDDING:
            api_kwargs["content"] = input
        elif model_type == ModelType.LLM:
            api_kwargs["prompt"] = input
        else:
            api_kwargs["input"] = input

        return api_kwargs

    def call(
        self,
        api_kwargs: Dict = None,
        model_type: ModelType = ModelType.UNDEFINED
    ) -> Any:
        """Make a synchronous call to Google GenAI API."""
        if not self._genai:
            raise RuntimeError("Google GenAI not initialized")

        api_kwargs = api_kwargs or {}

        if model_type == ModelType.EMBEDDER or model_type == ModelType.EMBEDDING:
            model = api_kwargs.get("model", "models/text-embedding-004")
            content = api_kwargs.get("content", "")
            result = self._genai.embed_content(model=model, content=content)

            # Convert to EmbedderOutput format
            embeddings = [EmbeddingData(embedding=result["embedding"], index=0)]
            return EmbedderOutput(data=embeddings, raw_response=result)

        elif model_type == ModelType.LLM:
            model_name = api_kwargs.get("model", "gemini-2.0-flash")
            prompt = api_kwargs.get("prompt", "")

            # Create generation config
            generation_config = {}
            for key in ["temperature", "top_p", "top_k", "max_output_tokens"]:
                if key in api_kwargs:
                    generation_config[key] = api_kwargs[key]

            model = self._genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config if generation_config else None
            )

            if api_kwargs.get("stream", False):
                return model.generate_content(prompt, stream=True)
            else:
                response = model.generate_content(prompt)
                return response.text

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    async def acall(
        self,
        api_kwargs: Dict = None,
        model_type: ModelType = ModelType.UNDEFINED
    ) -> Any:
        """Make an asynchronous call to Google GenAI API."""
        # Google GenAI doesn't have native async support, so we use sync
        return self.call(api_kwargs, model_type)


class OllamaClient(ModelClient):
    """
    Client for Ollama API.

    This client wraps the ollama library and provides
    a consistent interface for both LLM and embedding operations.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        env_host_name: str = "OLLAMA_HOST",
        *args,
        **kwargs
    ):
        """
        Initialize the Ollama client.

        Args:
            host: Optional host URL for Ollama server
            env_host_name: Name of the environment variable containing the host
        """
        super().__init__(*args, **kwargs)
        self._host = host or os.environ.get(env_host_name, "http://localhost:11434")

        try:
            import ollama
            self._ollama = ollama
            self._client = ollama.Client(host=self._host)
            self._async_client = ollama.AsyncClient(host=self._host)
        except ImportError:
            log.error("ollama package not installed")
            self._ollama = None
            self._client = None
            self._async_client = None

    def convert_inputs_to_api_kwargs(
        self,
        input: Any,
        model_kwargs: Dict = None,
        model_type: ModelType = ModelType.UNDEFINED
    ) -> Dict:
        """Convert inputs to Ollama API kwargs."""
        model_kwargs = model_kwargs or {}
        api_kwargs = model_kwargs.copy()

        if model_type == ModelType.EMBEDDER or model_type == ModelType.EMBEDDING:
            api_kwargs["prompt"] = input
        elif model_type == ModelType.LLM:
            api_kwargs["prompt"] = input
        else:
            api_kwargs["input"] = input

        return api_kwargs

    def call(
        self,
        api_kwargs: Dict = None,
        model_type: ModelType = ModelType.UNDEFINED
    ) -> Any:
        """Make a synchronous call to Ollama API."""
        if not self._client:
            raise RuntimeError("Ollama client not initialized")

        api_kwargs = api_kwargs or {}
        model = api_kwargs.get("model", "llama3.2")

        if model_type == ModelType.EMBEDDER or model_type == ModelType.EMBEDDING:
            prompt = api_kwargs.get("prompt", "")
            result = self._client.embeddings(model=model, prompt=prompt)

            # Convert to EmbedderOutput format
            embedding = result.get("embedding", [])
            embeddings = [EmbeddingData(embedding=embedding, index=0)]
            return EmbedderOutput(data=embeddings, raw_response=result)

        elif model_type == ModelType.LLM:
            prompt = api_kwargs.get("prompt", "")
            stream = api_kwargs.get("stream", False)

            # Extract options
            options = api_kwargs.get("options", {})

            if stream:
                return self._client.generate(
                    model=model,
                    prompt=prompt,
                    stream=True,
                    options=options
                )
            else:
                response = self._client.generate(
                    model=model,
                    prompt=prompt,
                    stream=False,
                    options=options
                )
                return response.get("response", "")

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    async def acall(
        self,
        api_kwargs: Dict = None,
        model_type: ModelType = ModelType.UNDEFINED
    ) -> Any:
        """Make an asynchronous call to Ollama API."""
        if not self._async_client:
            raise RuntimeError("Ollama async client not initialized")

        api_kwargs = api_kwargs or {}
        model = api_kwargs.get("model", "llama3.2")

        if model_type == ModelType.EMBEDDER or model_type == ModelType.EMBEDDING:
            prompt = api_kwargs.get("prompt", "")
            result = await self._async_client.embeddings(model=model, prompt=prompt)

            # Convert to EmbedderOutput format
            embedding = result.get("embedding", [])
            embeddings = [EmbeddingData(embedding=embedding, index=0)]
            return EmbedderOutput(data=embeddings, raw_response=result)

        elif model_type == ModelType.LLM:
            prompt = api_kwargs.get("prompt", "")
            stream = api_kwargs.get("stream", False)

            # Extract options
            options = api_kwargs.get("options", {})

            if stream:
                return await self._async_client.generate(
                    model=model,
                    prompt=prompt,
                    stream=True,
                    options=options
                )
            else:
                response = await self._async_client.generate(
                    model=model,
                    prompt=prompt,
                    stream=False,
                    options=options
                )
                return response.get("response", "")

        else:
            raise ValueError(f"Unsupported model type: {model_type}")
