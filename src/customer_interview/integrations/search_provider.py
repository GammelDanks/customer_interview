import os, requests
from typing import List, Dict

class NoopSearchProvider:
    def search(self, query: str, k: int = 6, freshness: str = "year", news: bool = False) -> List[Dict]:
        return []

class YouSearchProvider:
    BASE = "https://api.ydc-index.io/v1/search"  # offizieller Search-API-Endpoint

    def __init__(self, api_key: str | None = None):
        self.api_key = (api_key or os.getenv("YOU_API_KEY") or "").strip()

    def search(self, query: str, k: int = 6, freshness: str = "year", news: bool = False) -> List[Dict]:
        if not self.api_key:
            return []
        try:
            k = int(os.getenv("YOU_MAX_RESULTS", str(k)))
        except Exception:
            k = 6

        params = {
            "query": query,
            "count": max(1, min(k, 20)),
            "freshness": freshness,       # day|week|month|year
            "safesearch": "moderate",
        }
        headers = {"X-API-Key": self.api_key, "Accept": "application/json"}

        try:
            r = requests.get(self.BASE, params=params, headers=headers, timeout=25)
            if r.status_code != 200:
                return []
            data = r.json() or {}
        except Exception:
            return []

        items: List[Dict] = []
        results = data.get("results") or {}
        for it in (results.get("web") or []):
            url = it.get("url")
            title = it.get("title") or url
            if url:
                items.append({"title": title, "url": url})
        for it in (results.get("news") or []):
            url = it.get("url")
            title = it.get("title") or url
            if url:
                items.append({"title": title, "url": url})

        # dedupe
        seen, out = set(), []
        for item in items:
            u = item["url"]
            if u and u not in seen:
                seen.add(u); out.append(item)
        return out
