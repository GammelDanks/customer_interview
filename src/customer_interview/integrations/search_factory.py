import os
from .search_provider import NoopSearchProvider, YouSearchProvider

def get_search_provider():
    enabled = os.getenv("YOU_SEARCH_ENABLED", "false").lower() == "true"
    if not enabled:
        return NoopSearchProvider()
    key = (os.getenv("YOU_API_KEY") or "").strip()
    if not key:
        return NoopSearchProvider()
    return YouSearchProvider(api_key=key)
