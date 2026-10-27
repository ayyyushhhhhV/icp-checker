"""The single Azure OpenAI wrapper. Every model call goes through `LLM.structured`."""

import json
import logging
from typing import Any, TypeVar

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, ValidationError

from .config import PROMPTS_DIR, get_settings
from .store import cache_get, cache_key, cache_set, record_usage

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

SYSTEM = (
    "You are a careful B2B research assistant. You extract and normalize facts from the text you are given. "
    "Never invent facts. If something is not supported by the provided text, use null. Respond only with JSON "
    "matching the schema."
)

_env = Environment(loader=FileSystemLoader(PROMPTS_DIR), undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


def render(prompt_name: str, **vars: Any) -> str:
    return _env.get_template(f"{prompt_name}.md.j2").render(**vars)


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, client: AsyncAzureOpenAI | AsyncOpenAI | None = None):
        s = get_settings()
        self.settings = s
        self._client = client

    @property
    def client(self) -> AsyncAzureOpenAI | AsyncOpenAI:
        if self._client is None:
            s = self.settings
            if not (s.azure_openai_endpoint and s.azure_openai_api_key.get_secret_value() and s.azure_openai_deployment):
                raise LLMError("Azure OpenAI env vars are not set (AZURE_OPENAI_ENDPOINT/_API_KEY/_DEPLOYMENT)")
            key = s.azure_openai_api_key.get_secret_value()
            if s.azure_openai_endpoint.rstrip("/").endswith("/openai/v1"):
                # Azure's v1 API (e.g. https://<res>.services.ai.azure.com/openai/v1) takes the standard client
                self._client = AsyncOpenAI(base_url=s.azure_openai_endpoint, api_key=key, max_retries=3)
            else:
                self._client = AsyncAzureOpenAI(
                    azure_endpoint=s.azure_openai_endpoint, api_key=key,
                    api_version=s.azure_openai_api_version, max_retries=3,  # SDK backs off on 429/5xx
                )
        return self._client

    async def structured(self, prompt_name: str, output: type[T], small: bool = False, **vars: Any) -> T | None:
        """Render a template, call Azure with a strict JSON schema, validate with Pydantic.

        On validation failure, retries once with the error appended. Returns None if it still fails
        (callers then treat the fields as unknown instead of guessing).
        """
        s = self.settings
        deployment = s.small_deployment if small else s.azure_openai_deployment
        schema = to_strict_json_schema(output)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": render(prompt_name, **vars)}]

        key = cache_key(deployment, messages, schema)
        if s.llm_cache_enabled and (hit := cache_get("llm", key, None)) is not None:
            record_usage(llm_cache_hits=1)
            return output.model_validate(hit)

        for attempt in range(2):
            resp = await self.client.chat.completions.create(
                model=deployment,
                messages=messages,
                **({"temperature": s.llm_temperature} if s.llm_temperature is not None else {}),
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": output.__name__, "schema": schema, "strict": True},
                },
            )
            record_usage(
                llm_calls=1,
                prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            )
            content = resp.choices[0].message.content or ""
            try:
                result = output.model_validate_json(content)
            except (ValidationError, json.JSONDecodeError) as e:
                log.warning("llm validation failed prompt=%s attempt=%d: %s", prompt_name, attempt + 1, str(e)[:300])
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Your output failed validation:\n{e}\nReturn corrected JSON only."},
                ]
                continue
            if s.llm_cache_enabled:
                cache_set("llm", key, result.model_dump(mode="json"))
            return result
        log.warning("llm gave up prompt=%s; fields will be treated as unknown", prompt_name)
        return None
