from functools import cache
from pydantic_settings import (
    SettingsConfigDict,
    BaseSettings
)



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_file=".env",
        extra="ignore",
    )
    
    OPENROUTER_API_KEY: str



@cache
def get_settings() -> Settings:
    return Settings()
