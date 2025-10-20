from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


class QdrantTools:
    def __init__(self, agent, config: dict):
        self.agent = agent
        self.config = config
        self.project_name = config.get("project_name")
        self.agent_config = self.config.get(agent, {})
        self.agent_prompt = self.agent_config.get("prompt")
        self.agent_model_name = self.agent_config.get("model_name")
        self.agent_temperature = self.agent_config.get("temperature")
        self.agent_description = self.agent_config.get("description")
        self.agent_model_provider = self.agent_config.get("model_provider")
        self.agent_alternative_model = self.agent_config.get("alternative_model")
        self.agent_alternative_provider = self.agent_config.get("alternative_provider")
        self.project_root = config.get("project_root")
        self.agent_skills = self.agent_config.get("skills")
        self.design_docs = config.get("design_docs")
        self.source = config.get("source")
        self.project_directories = config.get("project_directories")
        self.include_extensions = config.get("include_extensions")
        self.max_file_bytes = config.get("max_file_bytes")
        self.exclude_directories = config.get("exclude_directories", [])
        self.recent_minutes = config.get("recent_minutes")
        self.payload: dict = {}
        self.embedding_model = config.get("qdrant_embedding")
        self.qdrant_client = QdrantClient(url="http://localhost:6333")
        self.embedding_size = config.get("embedding_size")
        self.collection_name = f"{self.project_name}_{agent}"

    def run_qdrant(self, payload):
        self.payload = payload
        method = getattr(self, self.agent)
        return method()

    def readme_writer(self):
        try:
            # Check if the collection exists; create if not.
            # This avoids the deprecated recreate_collection.
            if not self.qdrant_client.collection_exists(
                collection_name=self.collection_name
            ):
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_size, distance=Distance.COSINE
                    ),
                )

            # Embed the README content. Note: add returns a list of operation IDs.
            # Ensure self.payload is an iterable of str (e.g., [content] if single str).
            documents = (
                self.payload
                if isinstance(self.payload, (list, tuple))
                else [str(self.payload)]
            )
            vectors = self.qdrant_client.add(
                collection_name=self.collection_name,
                documents=documents,
                model=self.embedding_model,
            )
            if not vectors:
                raise ValueError("Embedding failed.")

        except Exception as errorStoring:
            print(f"Qdrant: An error occurred while storing the README: {errorStoring}")

        return self.payload
