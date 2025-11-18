README_PATCH.txt
This package preserves your baseline exactly and only adds:
- app_config.py
- data_provider.py
- brands.py
- .streamlit/secrets.toml

No other files were altered. Single-ticker flow unchanged.
To use the new providers in your existing code, import:
    from data_provider import get_spot, get_chain
Then call get_spot(symbol) and get_chain(symbol) where you fetch price/chain.
