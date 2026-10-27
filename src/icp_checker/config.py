"""All settings in one place. Secrets come from environment variables (or a local .env)."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROMPTS_DIR = Path(__file__).parent / "prompts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- secrets / endpoints (env only) ---
    azure_openai_endpoint: str = ""
    azure_openai_api_key: SecretStr = SecretStr("")
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_deployment: str = ""
    azure_openai_deployment_small: str = ""
    serper_api_key: SecretStr = SecretStr("")

    # --- storage ---
    data_dir: Path = Path("data")

    # --- caching ---
    serper_cache_ttl_days: int = 7
    llm_cache_enabled: bool = True
    company_cache_ttl_days: int = 30

    # --- limits ---
    max_concurrent_companies: int = 5
    max_serper_calls_per_company: int = 10  # spec default was 6; research alone can use ~5
    max_pages_per_company: int = 3
    page_char_budget: int = 12_000  # chars per scraped page sent to the LLM (~3k tokens)
    news_results: int = 5
    news_time_filter: str = "qdr:m6"  # Google "past 6 months"
    max_stakeholders: int = 3
    http_timeout_s: float = 45.0
    http_max_retries: int = 3

    # --- LLM ---
    llm_temperature: float | None = None  # None = model default (gpt-5.x only accepts the default)

    # --- vocab (teams may extend) ---
    functions: list[str] = [
        "talent_acquisition", "hr", "operations", "finance", "clinical",
        "it", "sales", "marketing", "executive", "other",
    ]

    @property
    def small_deployment(self) -> str:
        return self.azure_openai_deployment_small or self.azure_openai_deployment


@lru_cache
def get_settings() -> Settings:
    return Settings()
