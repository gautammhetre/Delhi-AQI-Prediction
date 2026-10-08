from core.llm import LLMClient


def site(request):
    """Values every template needs."""
    client = LLMClient()
    return {"llm_enabled": client.is_configured, "llm_model": client.model if client.is_configured else None}
