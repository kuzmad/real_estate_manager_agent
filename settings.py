from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str

    proxy_base_url: str = "https://openai.api.proxyapi.ru/v1"
    default_model: str = "openai/gpt-5.4-nano"
    max_file_size_mb: int = 10
    max_history_messages: int = 4

    model_config = {"env_file": ".env"}