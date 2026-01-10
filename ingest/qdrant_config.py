from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os
from dotenv import load_dotenv
load_dotenv()

CORPUS_VERSION = int(os.getenv("CORPUS_VERSION", "1"))

class QdrantConfig(BaseModel):
    mode: Literal["docker", "embedded", "memory"] = Field(
        default_factory=lambda: os.getenv("QDRANT_MODE", "docker")
    ) #type: ignore
    host: str = Field(
        default_factory=lambda: os.getenv("QDRANT_HOST", "localhost")
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333"))
    )
    grpc_port: int = Field(
        default_factory=lambda: int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    )
    storage_path: str = Field(
        default_factory=lambda: os.getenv("QDRANT_STORAGE_PATH", "./data/qdrant_storage")
    )
    collection_name: str = Field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "researcher_chunks")
    )
    vector_size: int = 768  # Gemini embedding dimension
    distance: Distance = Distance.COSINE

    class Config:
        use_enum_values = True

    def get_client(self) -> QdrantClient:
        if self.mode == "memory":
            return QdrantClient(":memory:")
        elif self.mode == "embedded":
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=self.storage_path)
        elif self.mode == "docker":
            return QdrantClient(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                timeout=30
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'docker', 'embedded', or 'memory'")

    def create_collection(self, client: QdrantClient) -> None:
        if not client.collection_exists(self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )
