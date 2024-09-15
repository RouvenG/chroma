import logging

import httpx

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)


class GooglePalmEmbeddingFunction(EmbeddingFunction[Documents]):
    """To use this EmbeddingFunction, you must have the google.generativeai Python package installed and have a PaLM API key."""

    def __init__(self, api_key: str, model_name: str = "models/embedding-gecko-001"):
        if not api_key:
            raise ValueError("Please provide a PaLM API key.")

        if not model_name:
            raise ValueError("Please provide the model name.")

        try:
            import google.generativeai as palm
        except ImportError:
            raise ValueError(
                "The Google Generative AI python package is not installed. Please install it with `pip install google-generativeai`"
            )

        palm.configure(api_key=api_key)
        self._palm = palm
        self._model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        return [
            self._palm.generate_embeddings(model=self._model_name, text=text)[
                "embedding"
            ]
            for text in input
        ]


class GoogleGenerativeAiEmbeddingFunction(EmbeddingFunction[Documents]):
    """To use this EmbeddingFunction, you must have the google.generativeai Python package installed and have a Google API key."""

    """Use RETRIEVAL_DOCUMENT for the task_type for embedding, and RETRIEVAL_QUERY for the task_type for retrieval."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/embedding-001",
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        if not api_key:
            raise ValueError("Please provide a Google API key.")

        if not model_name:
            raise ValueError("Please provide the model name.")

        try:
            import google.generativeai as genai
        except ImportError:
            raise ValueError(
                "The Google Generative AI python package is not installed. Please install it with `pip install google-generativeai`"
            )

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model_name
        self._task_type = task_type
        self._task_title = None
        if self._task_type == "RETRIEVAL_DOCUMENT":
            self._task_title = "Embedding of single string"

    def __call__(self, input: Documents) -> Embeddings:
        return [
            self._genai.embed_content(
                model=self._model_name,
                content=text,
                task_type=self._task_type,
                title=self._task_title,
            )["embedding"]
            for text in input
        ]


class GoogleVertexEmbeddingFunction(EmbeddingFunction[Documents]):
    # Follow API Quickstart for Google Vertex AI
    # https://cloud.google.com/vertex-ai/docs/generative-ai/start/quickstarts/api-quickstart
    # Information about the text embedding modules in Google Vertex AI
    # https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings
    def __init__(
        self,
        model_name: str = "textembedding-gecko@003",
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):

        try:
            from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
        except ImportError:
            raise ValueError(
                "The Google Generative AI python package is not installed. Please install it with `pip install google-cloud-aiplatform`"
            )

        self._model_name = model_name
        self._model = TextEmbeddingModel.from_pretrained(self._model_name)
        self._task_type = task_type
        self._task_title = None
        if self._task_type == "SEMANTIC_SIMILARITY":
            self._task_title = "Embedding of single string"

    def __call__(self, input: Documents) -> Embeddings:
        inputs = [TextEmbeddingInput(
                    text=text,
                    task_type=self._task_type,
                ) for text in input ]

        embeddings = self._model.get_embeddings(inputs)
        return [embedding.values for embedding in embeddings]
