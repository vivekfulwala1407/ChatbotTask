"""
Entry point to run the API server.
"""
import uvicorn
import warnings
from src.config import Config

warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    uvicorn.run("src.api.api:app", host=Config.HOST, port=Config.PORT, reload=True)