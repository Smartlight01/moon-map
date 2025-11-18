# data_provider.py - optimized: caching, pooling, windowed chains
from typing import Dict, Any, List, Optional
import httpx
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from app_config import POLYGON_API_KEY, TRADIER_ACCESS_TOKEN, TRADIER_ENV
INDEX_MAP = {
    "SPX":  {"poly_spot": None, "tradier_spot": "$SPX.X", "yf": "^GSPC", "chain": "SPX"},
    "SPXW": {"poly_spot": None, "tradier_spot": "$SPX.X", "yf": "^GSPC", "chain": "SPXW"},
}
def _norm(sym):
    s = (sym or "").upper()
    d = INDEX_MAP.get(s)
    if d:
        return d
    return {"poly_spot": s, "tradier_spot": s, "yf": s, "chain": s}


# HTTP clients held across reruns
@st.cache_resource
def _poly_client() -> httpx.Client:
    return httpx.Client(
        base_url="https://api.polygon.io",
        headers={"Authorization": f"Bearer {POLYGON_API_KEY}"},
        http2=True, timeout=15.0,
    )

@st.cache_resource
def _trad_client() -> httpx.Client:
    base = "https://api.tradier.com" if TRADIER_ENV == "live" else "https://sandbox.tradier.com"
    return httpx.Client(
        base_url=base,
        headers={"Authorization": f"Bearer {TRADIER_ACCESS_TOKEN}", "Accept": "application/json"},
        http2=True, timeout=20.0,
    )

# Short TTL for spot
@st.cache_data(ttl=5, show_spinner=False)
def get_spot(symbol: str) -> Optional[float]:
    poly = _poly_client()
    try:
        r = poly.get(f"/v2/last/trade/{symbol}")
        if r.status_code == 200:
            res = r.json().get("results") or {}
            if "p" in res:
                return float(res["p"])
    except Exception:
        pass
    trad = _trad_client()
    try:
        r = trad.get("/v1/markets/quotes", params={"symbols": symbol})
        if r.status_code == 200:
            q = ((r.json() or {}).get("quotes") or {}).get("quote") or {}
            for k in ("last","bid","ask"):
                v = q.get(k)
                if v not in (None, "", 0):
                    return float(v)
    except Exception:
        pass
    return None

@st.cache_data(ttl=60, show_spinner=False)
def _list_expiries(symbol: str) -> List[str]:
    trad = _trad_client()
    try:
        r = trad.get("/v1/markets/options/expirations", params={"symbol": symbol, "includeAllRoots": "true"})
        if r.status_code == 200:
            exps = ((r.json() or {}).get("expirations") or {}).get("date") or []
            return [e for e in exps if e]
    except Exception:
        pass
    return []

def _fetch_tradier_chain_expiry(symbol: str, exp: str) -> List[Dict[str, Any]]:
    trad = _trad_client()
    out: List[Dict[str, Any]] = []
    try:
        r = trad.get("/v1/markets/options/chains", params={"symbol": symbol, "expiration": exp, "greeks": "true"})
        if r.status_code != 200:
            return out
        chains = (r.json().get("options") or {}).get("option", []) or []
        for c in chains:
            g = c.get("greeks") or {}
            out.append({
                "ticker": c.get("symbol"),
                "strike": c.get("strike"),
                "expiration": c.get("expiration_date") or exp,
                "cp": "C" if c.get("option_type") == "call" else "P",
                "iv": g.get("mid_iv"), "delta": g.get("delta"), "gamma": g.get("gamma"),
                "vega": g.get("vega"), "theta": g.get("theta"),
                "oi": c.get("open_interest"), "volume": c.get("volume"),
            })
    except Exception:
        pass
    return out

@st.cache_data(ttl=60, show_spinner=False)
def get_chain(symbol: str, *, max_expiries: int = 4, strike_window: int = 40) -> List[Dict[str, Any]]:
    """
    Return flattened rows for first `max_expiries`. Window to ±strike_window/2 around spot.
    Compatible with existing callers that pass only symbol.
    """
    spot = get_spot(symbol)
    exps = _list_expiries(symbol)[:max_expiries]
    if not exps:
        return []

    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(8, max(2, len(exps)))) as pool:
        for part in pool.map(lambda e: _fetch_tradier_chain_expiry(symbol, e), exps):
            if part:
                rows.extend(part)

    if not rows or spot is None:
        return rows

    # Window per expiry
    try:
        from collections import defaultdict
        import bisect
        S = float(spot)
        half = max(8, int(strike_window)//2)
        by_exp = defaultdict(list)
        for r in rows:
            by_exp[r["expiration"]].append(r)
        out: List[Dict[str, Any]] = []
        for exp, lst in by_exp.items():
            ks = sorted({float(r["strike"]) for r in lst if r.get("strike") is not None})
            if not ks:
                continue
            i0 = bisect.bisect_left(ks, S)
            lo = max(0, i0 - half); hi = min(len(ks), i0 + half + 1)
            keep = set(ks[lo:hi])
            out.extend([r for r in lst if r.get("strike") in keep])
        return out
    except Exception:
        return rows


def fetch_tradier_chain(symbol: str, expiration: str):
    s = (symbol or "").upper()
    if s in ("SPX","SPXW"):
        try:
            from spx_chain_db import db_fetch_chain_for_spx
            chain = db_fetch_chain_for_spx(expiration)
            if chain:
                print(f"[MoonMap] SPX chain via Databento -> {len(chain)} contracts @ {expiration}")
                return chain
        except Exception as e:
            print("[MoonMap] Databento chain error:", e)
    # Fallback: call original if it exists in globals (some baselines define it elsewhere)
    try:
        return _fetch_tradier_chain_orig(symbol, expiration)  # type: ignore
    except Exception:
        return []


def fetch_chain(symbol: str, expiration: str):
    # Route to fetch_tradier_chain so legacy callers pick up Databento for SPX
    return fetch_tradier_chain(symbol, expiration)
