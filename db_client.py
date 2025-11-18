from __future__ import annotations
from typing import List, Dict, Any
import os
import databento as db

API_KEY = os.getenv("DATABENTO_API_KEY", "db-TcSWrtu4gdJAem8Ym7nCMkghkhdB5")

def _client() -> db.Historical:
    return db.Historical(API_KEY)

def list_spx_expiries() -> List[str]:
    c = _client()
    inst = c.reference.instruments(
        dataset="OPRA.PILLAR",
        stype_in="parent",
        symbols=["SPX"],
    )
    exps = set()
    for rec in inst:
        exp = getattr(rec, "expiration_date", None) or getattr(rec, "expiration", None)
        if exp:
            exps.add(str(exp))
    return sorted(exps)

def list_spx_chain(expiration: str) -> List[Dict[str, Any]]:
    c = _client()
    inst = c.reference.instruments(
        dataset="OPRA.PILLAR",
        stype_in="parent",
        symbols=["SPX"],
    )
    out: List[Dict[str, Any]] = []
    for rec in inst:
        exp = str(getattr(rec, "expiration_date", None) or getattr(rec, "expiration", ""))
        if exp != expiration:
            continue
        sym = getattr(rec, "raw_symbol", None) or getattr(rec, "symbol", None)
        strike = getattr(rec, "strike_price", None) or getattr(rec, "strike", None)
        opt_type = getattr(rec, "option_type", None) or getattr(rec, "put_call", None)
        root = getattr(rec, "root", None) or getattr(rec, "underlying", None)
        if sym and strike is not None and opt_type:
            out.append({
                "raw_symbol": str(sym),
                "strike": float(strike),
                "type": ("c" if str(opt_type).lower().startswith("c") else "p"),
                "root": str(root) if root else "SPX",
                "expiration": exp,
            })
    return out
