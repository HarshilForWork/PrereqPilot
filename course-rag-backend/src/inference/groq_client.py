import asyncio

from groq import AsyncGroq
from groq import RateLimitError as GroqRateLimitError

from src.core.config import settings
from src.core.exceptions import LLMServiceError

_client: AsyncGroq | None = None


def get_groq_client() -> AsyncGroq:
	global _client
	if _client is None:
		_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
	return _client


async def call_llm(system_prompt: str, user_prompt: str, temperature: float) -> str:
	client = get_groq_client()
	try:
		response = await client.chat.completions.create(
			model=settings.LLM_MODEL,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			temperature=temperature,
			max_tokens=settings.LLM_MAX_TOKENS,
		)
		return response.choices[0].message.content or ""
	except GroqRateLimitError:
		await asyncio.sleep(2)
		try:
			response = await client.chat.completions.create(
				model=settings.LLM_MODEL,
				messages=[
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": user_prompt},
				],
				temperature=temperature,
				max_tokens=settings.LLM_MAX_TOKENS,
			)
			return response.choices[0].message.content or ""
		except GroqRateLimitError as exc:
			raise LLMServiceError("Groq rate limit exceeded after retry.") from exc
