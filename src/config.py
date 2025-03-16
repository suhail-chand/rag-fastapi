import secrets
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env")
    
    SECRET_KEY: str = secrets.token_urlsafe(32)

    APP_NAME: str = "Retrieval Augmented Generation"
    APP_DESCRIPTION: str = "A RAG system with FastAPI, processes PDF and DOCX files, storing their content " \
    "in a FAISS index for efficient similarity search and generating context-aware responses to user queries."
    APP_VERSION: str = "1.0.0"

    HF_TOKEN: str | None = None
    MISTRAL_API_KEY: str | None = None
    FAISS_STORE_PATH: str = './faiss_store'


settings = Settings()
