import secrets
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env")
    
    SECRET_KEY: str = secrets.token_urlsafe(32)

    APP_NAME: str = "Document Summarizer"
    APP_DESCRIPTION: str = "An application to summarize PDF documents using LLM."
    APP_VERSION: str = "1.0.0"

    HF_TOKEN: str | None = None
    MISTRAL_API_KEY: str | None = None


settings = Settings()
