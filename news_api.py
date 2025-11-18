from __future__ import annotations
import os, requests, datetime as dt
try:
    import config
except Exception:
    class config: FINNHUB_TOKEN = ""
def fetch_company_news_cards(symbol: str, token: str|None=None):
    token = token or os.environ.get("FINNHUB_TOKEN","") or getattr(config,"FINNHUB_TOKEN","")
    if not token: return []
    try:
        today = dt.date.today()
        start = (today - dt.timedelta(days=7)).isoformat()
        resp = requests.get("https://finnhub.io/api/v1/company-news",
                            params={"symbol":symbol,"from":start,"to":today.isoformat(),"token":token},timeout=8)
        if not resp.ok: return []
        items = resp.json() or []
        out=[]
        for it in items[:15]:
            ts = it.get("datetime",0) or 0
            try: ts_str = dt.datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
            except: ts_str = ""
            out.append({"headline":it.get("headline",""),"url":it.get("url",""),
                        "source":it.get("source",""),"datetime":ts_str,"summary":it.get("summary","")})
        return out
    except Exception:
        return []
