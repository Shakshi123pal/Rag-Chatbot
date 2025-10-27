from llama_index.core.chat_engine import CondensePlusContextChatEngine, SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import BaseNode
from typing import List
from .retriever import LocalRetriever
from ...setting import RAGSettings


class LocalChatEngine:
    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal"
    ):
        super().__init__()
        self._setting = setting or RAGSettings()
        self._retriever = LocalRetriever(self._setting)
        self._host = host

    def set_engine(
        self,
        llm: LLM,
        nodes: List[BaseNode],
        language: str = "eng",
    ) -> CondensePlusContextChatEngine | SimpleChatEngine:

        # No documents → simple LLM chat
        if len(nodes) == 0:
            print("[INFO] Using SimpleChatEngine (no docs uploaded)")
            return SimpleChatEngine.from_defaults(
                llm=llm,
                memory=ChatMemoryBuffer(
                    token_limit=self._setting.ollama.chat_token_limit
                )
            )

        # Documents available → set up retriever
        retriever = self._retriever.get_retrievers(
            llm=llm,
            language=language,
            nodes=nodes
        )

        # Create RAG-based chat engine
        engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            memory=ChatMemoryBuffer(
                token_limit=self._setting.ollama.chat_token_limit
            )
        )

        # ------------- PATCH: Intercept context before sending to Ollama -------------
        orig_chat = engine.chat

        def safe_chat(query: str):
            try:
                # Clean input
                query = query.encode("utf-8", "ignore").decode("utf-8")

                # Get retrieved context manually
                results = retriever.retrieve(query)
                combined_context = "\n\n".join(
                    [getattr(r, "text", "")[:1500] for r in results[:2]]
                )

                prompt = (
                    f"Use the following context to answer the user's query.\n"
                    f"Context:\n{combined_context}\n\n"
                    f"Question: {query}\n\n"
                    f"Give a concise and clear answer based on the context only."
                )

                print(f"[DEBUG] Context length sent to Ollama: {len(prompt)} chars")
                return orig_chat(prompt)
            except Exception as e:
                print(f"[ERROR] RAG Query failed inside engine: {e}")
                return f"⚠️ RAG query failed: {str(e)}"

        engine.chat = safe_chat
        print("[INFO] Enhanced CondensePlusContextChatEngine initialized ✅")
        return engine