from pydantic_settings import BaseSettings
from pathlib import Path

ROOT_DIR = Path(__file__).parent
CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"
contracts = [file.stem for file in CONTRACTS_DIR.glob("*.txt")]
contracts = ', '.join(contracts)

class Settings(BaseSettings):
    openai_api_key: str

    proxy_base_url: str = "https://openai.api.proxyapi.ru/v1"
    default_model: str = "openai/gpt-5.4-nano"
    max_file_size_mb: int = 10
    max_history_messages: int = 10
    max_tenant_history: int = 6
    tenants: str = contracts

    model_config = {"env_file": str(ROOT_DIR / ".env")}
