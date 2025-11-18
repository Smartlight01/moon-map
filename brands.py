# brands.py - Finnhub branding helper
from typing import Optional, List, Dict, Any
import httpx
from app_config import FINNHUB_API_KEY
def company_logo(ticker: str) -> Optional[str]:
    try:
        r = httpx.get("https://finnhub.io/api/v1/stock/profile2",
                      params={"symbol": ticker, "token": FINNHUB_API_KEY}, timeout=10)
        return (r.json() or {}).get("logo")
    except Exception:
        return None
def company_news(ticker: str, date_from: str, date_to: str) -> List[Dict[str, Any]]:
    try:
        r = httpx.get("https://finnhub.io/api/v1/company-news",
                      params={"symbol": ticker, "from": date_from, "to": date_to, "token": FINNHUB_API_KEY}, timeout=10)
        return r.json() or []
    except Exception:
        return []
