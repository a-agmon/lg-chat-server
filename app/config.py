from pydantic import SecretStr
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o"
    log_level: str = "INFO"
    # Path to complaint KB
    kb_path: str = "data/complaints.json"

    class Config:
        env_file = ".env"

settings = Settings()

