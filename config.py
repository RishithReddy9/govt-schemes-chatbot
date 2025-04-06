import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    SCHEMES_FILE = "data/schemes.json"
    EMBEDDING_MODEL = "models/embedding-001"
    GENERATION_MODEL = "gemini-1.5-pro-latest"
    FAISS_INDEX_FILE = "faiss_index.bin"
