from pydantic import BaseModel, SecretStr
import os


class Settings(BaseModel):
    openai_api_key: SecretStr | None = SecretStr(os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") is not None else None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # Path to complaint KB
    kb_path: str = os.getenv("KB_PATH", "data/complaints.json")


settings = Settings()

