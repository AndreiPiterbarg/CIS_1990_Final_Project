from openai import OpenAI

from git_explainer import config

Message = dict[str, str]

client = OpenAI(
    api_key=config.GROQ_API_KEY,
    base_url=config.GROQ_BASE_URL,
    max_retries=3,
    timeout=60.0,
)


def chat(
    user_content: str,
    *,
    system_prompt: str = "",
    history: list[Message] | None = None,
    model: str = config.GROQ_MODEL,
    max_tokens: int = config.GROQ_MAX_TOKENS,
    temperature: float = 0.3,
) -> str:
    """Send a message to Groq and return the assistant's reply."""
    messages: list[Message] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return (response.choices[0].message.content or "").strip()
