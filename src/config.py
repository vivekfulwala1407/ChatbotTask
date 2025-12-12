from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()  # only for local development

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    INDEX_DIR = Path(os.getenv("INDEX_DIR", "./index"))
    MASTER_OUTLET = Path(os.getenv("MASTER_OUTLET", "./DataPreprocessing/output/master_outlet_v2.csv"))
    MASTER_PRODUCT = Path(os.getenv("MASTER_PRODUCT", "./DataPreprocessing/output/master_product_v2.csv"))
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", 8000))