from __future__ import annotations
from typing import List, Dict, Any
from db_client import list_spx_expiries, list_spx_chain

def db_list_expiries_for_spx() -> List[str]:
    return list_spx_expiries()

def db_fetch_chain_for_spx(expiration: str) -> List[Dict[str, Any]]:
    return list_spx_chain(expiration)
