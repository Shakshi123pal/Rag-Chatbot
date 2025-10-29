import re
import fitz
import os
from dotenv import load_dotenv
from typing import Any, List
from tqdm import tqdm

# ✅ Compatible imports for llama-index 0.10.22
from llama_index.core import Settings, Document
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter

from ...setting import RAGSettings

# Optional loaders for fallback
try:
    from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredFileLoader
except ImportError:
    PyMuPDFLoader = None
    UnstructuredFileLoader = None

load_dotenv()


class LocalDataIngestion:
    def __init__(self, setting: RAGSettings | None = None) -> None:
        self._setting = setting or RAGSettings()
        self._node_store = {}
        self._ingested_file = []

    def _filter_text(self, text):
        pattern = r'[a-zA-Z0-9 \u00C0-\u01B0\u1EA0-\u1EF9`~!@#$%^&*()_\-+=\[\]{}|\\;:\'",.<>/?]+'
        matches = re.findall(pattern, text)
        filtered_text = ' '.join(matches)
        return re.sub(r'\s+', ' ', filtered_text.strip())

    def store_nodes(
        self,
        input_files: list[str],
        embed_nodes: bool = True,
        embed_model: Any | None = None
    ) -> List[BaseNode]:
        return_nodes = []
        self._ingested_file = []
        if not input_files:
            print("[INFO] No input files found for ingestion.")
            return return_nodes

        splitter = SentenceSplitter.from_defaults(
            chunk_size=self._setting.ingestion.chunk_size,
            chunk_overlap=self._setting.ingestion.chunk_overlap,
            paragraph_separator=self._setting.ingestion.paragraph_sep,
            secondary_chunking_regex=self._setting.ingestion.chunking_regex,
        )
        if embed_nodes:
            Settings.embed_model = embed_model or Settings.embed_model

        for input_file in tqdm(input_files, desc="Ingesting data"):
            file_name = os.path.basename(input_file)
            self._ingested_file.append(file_name)
            ext = os.path.splitext(file_name)[1].lower()
            all_text = ""

            try:
                # ✅ Read PDF
                if ext == ".pdf" and PyMuPDFLoader:
                    try:
                        loader = PyMuPDFLoader(input_file)
                        docs = loader.load()
                        all_text = " ".join([self._filter_text(d.page_content) for d in docs])
                    except Exception as e:
                        print(f"[WARN] PyMuPDFLoader failed: {e}")
                        if UnstructuredFileLoader:
                            loader = UnstructuredFileLoader(input_file)
                            docs = loader.load()
                            all_text = " ".join([self._filter_text(d.page_content) for d in docs])

                # ✅ Read TXT/DOCX or other
                elif UnstructuredFileLoader:
                    loader = UnstructuredFileLoader(input_file)
                    docs = loader.load()
                    all_text = " ".join([self._filter_text(d.page_content) for d in docs])
                else:
                    print(f"[WARN] No loader available for {file_name}")

            except Exception as e:
                print(f"[ERROR] Failed to read {file_name}: {e}")
                continue

            if not all_text.strip():
                print(f"[WARN] Empty or unreadable file: {file_name}")
                continue

            document = Document(text=all_text.strip(), metadata={"file_name": file_name})
            nodes = splitter([document], show_progress=True)
            if embed_nodes:
                nodes = Settings.embed_model(nodes, show_progress=True)

            self._node_store[file_name] = nodes
            return_nodes.extend(nodes)

        print(f"[INFO] Created {len(return_nodes)} nodes ✅")
        return return_nodes

    def reset(self):
        self._node_store.clear()
        self._ingested_file.clear()

    def check_nodes_exist(self):
        return len(self._node_store) > 0

    def get_all_nodes(self):
        return [node for nodes in self._node_store.values() for node in nodes]

    def get_ingested_nodes(self):
        return_nodes = []
        for file in self._ingested_file:
            if file not in self._node_store:
                alt_file = file.replace(".pdf", ".txt")
                if alt_file in self._node_store:
                    file = alt_file
                else:
                    print(f"[WARN] File '{file}' not found in node store, skipping.")
                    continue
            return_nodes.extend(self._node_store[file])
        return return_nodes
