import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ollama model settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_0")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Vector database path
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")

# Gradio port (optional)
PORT = int(os.getenv("PORT", 7860))
