import os, requests
from dotenv import load_dotenv

load_dotenv()  # lädt .env aus dem Projektroot

def main():
    api_key = (os.getenv("YOU_API_KEY") or "").strip()
    if not api_key:
        print("❌ No YOU_API_KEY set in .env")
        return

    url = "https://api.ydc-index.io/v1/search"  # offizielle Search-API
    params = {
        "query": "market overview of password managers, comparison reports and user reviews",
        "count": 5,
        "freshness": "year",
        "safesearch": "moderate",
        # optional: "country": "DE",
    }
    headers = {
        "X-API-Key": api_key,   # genau so laut Doku
        "Accept": "application/json",
    }

    print(f"🔎 GET {url} ...")
    r = requests.get(url, headers=headers, params=params, timeout=25)
    print("Status:", r.status_code)
    if r.status_code != 200:
        print("❌ Error:", r.text)
        return

    data = r.json()
    results = (data.get("results") or {}).get("web") or []
    print(f"✅ Got {len(results)} web results\n")
    for i, item in enumerate(results[:5], start=1):
        print(f"{i}. {item.get('title')}\n   {item.get('url')}\n")

if __name__ == "__main__":
    main()
