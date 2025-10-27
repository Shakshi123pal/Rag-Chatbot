from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.schema import TextNode
from dotenv import load_dotenv
from ...setting import RAGSettings

load_dotenv()


class LocalVectorStore:
    def __init__(
        self,
        host: str = "host.docker.internal",
        setting: RAGSettings | None = None,
    ) -> None:
        self._setting = setting or RAGSettings()
        self._host = host

        self._vector_store = SimpleVectorStore()
        self._index = None

    # ✅ Add nodes to vector store
    def add_nodes(self, nodes: list[TextNode]):
        """Add embedded document nodes into the local vector store."""
        try:
            if not nodes:
                print("[WARN] No nodes provided to add_nodes().")
                return

            # Convert to valid document structure for VectorStoreIndex
            storage_ctx = StorageContext.from_defaults(vector_store=self._vector_store)

            if self._index is None:
                # Create new index from text nodes
                self._index = VectorStoreIndex(nodes, storage_context=storage_ctx)
                print(f"[INFO] Created new index with {len(nodes)} nodes ✅")
            else:
                # Insert nodes safely
                self._index.insert_nodes(nodes)
                print(f"[INFO] Inserted {len(nodes)} nodes into existing index ✅")

        except Exception as e:
            print(f"[ERROR] Failed to store nodes in vector DB: {e}")

    # ✅ Compatible retriever method (no parameters)
    def as_retriever(self):
        """Return retriever for querying the stored vectors."""
        if not self._index:
            print("[WARN] No index available for retrieval.")
            return None
        return self._index.as_retriever()

    # ✅ Optional reset method
    def reset(self):
        """Reset the local vector store."""
        self._vector_store = SimpleVectorStore()
        self._index = None
        print("[INFO] Vector store reset complete.")
