from .core import (
    LocalChatEngine,
    LocalDataIngestion,
    LocalRAGModel,
    LocalEmbedding,
    LocalVectorStore,
    get_system_prompt,
)
from llama_index.core import Settings
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.prompts import ChatMessage, MessageRole


class LocalRAGPipeline:
    def __init__(self, host: str = "host.docker.internal") -> None:
        self._host = host
        self._language = "eng"
        self._model_name = ""
        self._system_prompt = get_system_prompt("eng", is_rag_prompt=False)
        # Core RAG components
        self._engine = LocalChatEngine(host=host)
        self._default_model = LocalRAGModel.set(self._model_name, host=host)
        self._query_engine = None
        self._ingestion = LocalDataIngestion()
        self._vector_store = LocalVectorStore(host=host)
        # Initialize embedding + model in llama-index settings
        Settings.llm = LocalRAGModel.set(host=host)
        Settings.embed_model = LocalEmbedding.set(host=host)
    
    # ---------------------- BASIC SETTERS / GETTERS ----------------------
    def get_model_name(self):
        # Agar model name empty hai, to .env se le lo
        if not self._model_name:
            import os
            return os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_0")
        return self._model_name


    def set_model_name(self, model_name: str):
        self._model_name = model_name

    def get_language(self):
        return self._language

    def set_language(self, language: str):
        self._language = language

    def get_system_prompt(self):
        return self._system_prompt

    def set_system_prompt(self, system_prompt: str | None = None):
        # automatically detect if RAG prompt is needed
        self._system_prompt = system_prompt or get_system_prompt(
            language=self._language, 
            is_rag_prompt=self._ingestion.check_nodes_exist(),
        )
    # ---------------------- MODEL / EMBEDDING MANAGEMENT ----------------------
    def set_model(self):
        Settings.llm = LocalRAGModel.set(
            model_name=self._model_name,
            system_prompt=self._system_prompt,
            host=self._host,
        )
        self._default_model = Settings.llm
    
    def set_embed_model(self, model_name: str | None = None):
        Settings.embed_model = LocalEmbedding.set(model_name)


    def pull_model(self, model_name: str):
        return LocalRAGModel.pull(self._host, model_name)

    def pull_embed_model(self, model_name: str):
        return LocalEmbedding.pull(self._host, model_name)

    def check_exist(self, model_name: str) -> bool:
        return LocalRAGModel.check_model_exist(self._host, model_name)

    def check_exist_embed(self, model_name: str) -> bool:
        return LocalEmbedding.check_model_exist(self._host, model_name)
    
    # ---------------------- DOCUMENT & VECTOR STORE ----------------------

    def store_nodes(self, input_files: list[str] = None) -> None:
        """
        Converts uploaded files into embeddings and stores them in vector DB.
        Handles compatibility for different LocalVectorStore methods.
        """
        if not input_files:
            print("[INFO] No input files found for ingestion.")
            return

        print(f"[INFO] Ingesting {len(input_files)} documents...")

        # ✅ Ensure embedding model is set before ingestion
        self.set_embed_model()

        # Step 1: Extract + embed documents
        nodes = self._ingestion.store_nodes(input_files=input_files)

        # Step 2: Save embeddings to the vector store
        if nodes:
            try:
                if hasattr(self._vector_store, "add_nodes"):
                    self._vector_store.add_nodes(nodes)
                elif hasattr(self._vector_store, "insert_nodes"):
                    self._vector_store.insert_nodes(nodes)
                elif hasattr(self._vector_store, "store_nodes"):
                    self._vector_store.store_nodes(nodes)
                elif hasattr(self._vector_store, "upsert_nodes"):
                    self._vector_store.upsert_nodes(nodes)
                else:
                    raise AttributeError(
                        "LocalVectorStore has no valid method to save nodes (expected add_nodes/insert_nodes/store_nodes/upsert_nodes)"
                    )

                print(f"[INFO] Stored {len(nodes)} embedded nodes in vector DB ✅")

            except Exception as e:
                print(f"[ERROR] Failed to store nodes in vector DB: {e}")
        else:
            print("[WARN] No nodes created during ingestion.")
        
        try:
            self.set_chat_mode(
                system_prompt="Use only the uploaded documents to answer. If answer not found, say: 'Not in document'."
            )
            print("[INFO] ✅ RAG mode activated using uploaded documents")
        except Exception as e:
            print(f"[WARN] Could not activate RAG mode: {e}")


    def reset_documents(self):
        """Reset all documents and clear stored embeddings"""
        self._ingestion.reset()
        self._vector_store.reset()

    # ---------------------- CHAT ENGINE / QUERY ----------------------

    def set_engine(self):
        """Creates or refreshes query engine with stored document context"""
        ingested_nodes = self._ingestion.get_ingested_nodes()

        retriever = self._vector_store.as_retriever()

        # ✅ Pass ingested nodes explicitly to set_engine()
        self._query_engine = self._engine.set_engine(
            nodes=ingested_nodes,
            llm=self._default_model
        )

        # ✅ Attach retriever manually if your engine needs it later
        self._query_engine.retriever = retriever

        print(f"[INFO] Engine set with {len(ingested_nodes)} nodes.")


    def reset_engine(self):
        """Reset engine without documents"""
        self._query_engine = self._engine.set_engine(
            llm=self._default_model,
            nodes=[],
            language=self._language,
        )

    def clear_conversation(self):
        if self._query_engine:
            self._query_engine.reset()

    def reset_conversation(self):
        self.reset_engine()
        self.set_system_prompt(
            get_system_prompt(language=self._language, is_rag_prompt=False)
        )

    def set_chat_mode(self, system_prompt: str | None = None):
        """Reinitialize chat mode after document update"""
        self.set_language(self._language)
        self.set_system_prompt(system_prompt)
        self.set_model()
        self.set_engine()
    
    # ---------------------- CHAT HISTORY / QUERY HANDLER ----------------------
    def get_history(self, chatbot: list[list[str]]):
        """
        Convert chatbot history into Ollama-compatible format.
        Each item is a dict: {"role": "user"/"assistant", "content": "text"}
        """
        history = []
        for chat in chatbot:
            if chat[0]:
                history.append({"role": "user", "content": chat[0]})
                history.append({"role": "assistant", "content": chat[1]})
        return history

    def query(self, mode: str, message: str, chatbot: list[list[str]]):
        """
        Handle user query and return streaming response from Ollama.
        Keeps chat_mode and chatbot history for internal logic,
        but does NOT send 'history' param (since Ollama rejects it).
        """
        if self._query_engine is None:
            self.set_engine()

        # convert chat history (for logs or future)
        history = self.get_history(chatbot)

        print(f"[DEBUG] Mode: {mode}, Message: {message}")
        print(f"[DEBUG] Chat history length: {len(history)}")

        # guard: if message empty
        if not message or not message.strip():
            raise ValueError("Message is empty, cannot send to Ollama API")

        try:
            # send only message (Ollama doesn't accept history)
            response = self._query_engine.chat(message)

            if not isinstance(response, StreamingAgentChatResponse):
                raise TypeError("Expected StreamingAgentChatResponse from engine")
            return response

        except Exception as e:
            print("Error in Ollama query:", e)
            raise e
