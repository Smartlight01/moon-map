from __future__ import annotations
from data_provider import get_chain
import streamlit as st

import math, numpy as np, pandas as pd


# --- Injected: provider-backed chain fetch ---
import pandas as _pd
def _fetch_chain_provider(symbol: str) -> _pd.DataFrame:
    rows = []
    for c in get_chain(symbol):
        rows.append({
            "symbol": c.get("ticker"),
            "expiration": c.get("expiration"),
            "strike": float(c.get("strike")) if c.get("strike") is not None else None,
            "type": c.get("cp"),  # 'C' or 'P'
            "openInterest": c.get("oi"),
            "impliedVolatility": c.get("iv"),
            "bid": None,
            "ask": None,
            "last": None,
            "volume": c.get("volume"),
        })
    return _pd.DataFrame(rows)

import yfinance as yf
from scipy.stats import norm
from session_anchor import session_date, ensure_ny, NY_TZ, now_ny

def _to_float(x, default=0.0):
    try:
        xx = float(x)
        if not np.isfinite(xx):
            return default
        return xx
    except Exception:
        return default

def _to_int(x, default=0):
    try:
        if pd.isna(x): return default
        xi = int(float(x))
        return xi
    except Exception:
        return default

def bs_d1(S, K, r, q, sigma, T):
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan
    return (np.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))

def bs_vega(S, K, r, q, sigma, T):
    d1 = bs_d1(S, K, r, q, sigma, T)
    if np.isnan(d1): return 0.0
    return S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)

def bs_gamma(S, K, r, q, sigma, T):
    d1 = bs_d1(S, K, r, q, sigma, T)
    if np.isnan(d1) or S <= 0 or sigma <= 0 or T <= 0:
        return 0.0
    return np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def expiry_trading_cutoff(expiry_ts: pd.Timestamp) -> pd.Timestamp:
    ex = ensure_ny(expiry_ts).tz_convert(NY_TZ)
    return ex.normalize() + pd.Timedelta(hours=16)

def compute_intraday_T_years(expiry_ts: pd.Timestamp, now_ts: pd.Timestamp) -> float:
    cutoff = expiry_trading_cutoff(expiry_ts)
    dt = (cutoff - now_ts).total_seconds()
    if dt <= 0:
        return 1e-6
    return dt / (365.25*24*3600.0)



# --- r and q helpers (step 2) ---
def _safe_yf_last_close(tk):
    try:
        h = tk.history(period='5d')
        if not h.empty:
            return float(h['Close'].dropna().iloc[-1])
    except Exception:
        pass
    return float('nan')

@st.cache_data(ttl=300, show_spinner=False)
def fetch_yield_curve():
    """Return list of (years, rate_decimal). Uses ^IRX, ^FVX, ^TNX; fallback 3%."""
    curve = []
    try:
        for years, sym in [(0.25,'^IRX'), (5.0,'^FVX'), (10.0,'^TNX')]:
            try:
                ytk = yf.Ticker(sym)
                px = _safe_yf_last_close(ytk)
                if np.isfinite(px):
                    curve.append((float(years), float(px)/100.0))
            except Exception:
                continue
    except Exception:
        curve = []
    if not curve:
        curve = [(0.25, 0.03), (5.0, 0.03), (10.0, 0.03)]
    return sorted(curve)

def r_for_T_years(curve, T):
    if not curve:
        return 0.02
    T = max(1e-6, float(T))
    xs = [c[0] for c in curve]
    ys = [c[1] for c in curve]
    if T <= xs[0]:
        return ys[0]
    if T >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if T <= xs[i]:
            x0,x1 = xs[i-1], xs[i]
            y0,y1 = ys[i-1], ys[i]
            w = (T - x0)/(x1 - x0)
            return y0 + w*(y1 - y0)
    return ys[-1]

@st.cache_data(ttl=600, show_spinner=False)
def dividend_yield(symbol: str, spot: float) -> float:
    try:
        tk = yf.Ticker(symbol)
        info = {}
        try:
            info = tk.fast_info or {}
        except Exception:
            info = {}
        y = float(info.get('dividendYield', float('nan')))
        if np.isfinite(y) and y>0:
            return y
        spot_px = float(spot) if np.isfinite(spot) and spot>0 else _safe_yf_last_close(tk)
        if not np.isfinite(spot_px) or spot_px <= 0:
            return 0.0
        divs = pd.Series(dtype=float)
        try:
            divs = tk.dividends
        except Exception:
            divs = pd.Series(dtype=float)
        if divs is None or divs.empty:
            return 0.0
        recent = divs[divs.index >= (pd.Timestamp.now(tz=NY_TZ) - pd.Timedelta(days=370))]
        total = float(recent.sum()) if not recent.empty else float(divs.tail(4).sum())
        return max(0.0, total / spot_px)
    except Exception:
        return 0.0
# --- end r and q helpers ---

# --- IV/OI fallback (step 3) ---
def _fallback_iv_oi(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    valid = w['iv'].astype(float).between(0.01, 5.0)
    med = w.groupby('expiry')['iv'].transform('median')
    w['iv'] = pd.Series(np.where(valid, w['iv'], med), index=w.index).fillna(0.20)
    if 'volume' in w.columns:
        bad = (~w['oi'].fillna(0).astype(float).gt(0)) & (w['volume'].fillna(0).astype(float).gt(0))
        w.loc[bad, 'oi'] = w.loc[bad, 'volume']
        w['oi_proxied'] = bad.astype(int)
    else:
        w['oi_proxied'] = 0
    return w
# --- end IV/OI fallback ---
@st.cache_data(ttl=120, show_spinner=False)
@st.cache_data(ttl=120, show_spinner=False)
def fetch_chain(symbol: str):
    tk = yf.Ticker(symbol)
    spot_px = np.nan
    try:
        spot_px = _to_float(tk.fast_info.get("lastPrice"), default=np.nan)
    except Exception:
        pass
    if not np.isfinite(spot_px):
        try:
            hist = tk.history(period="2d")
            if not hist.empty:
                spot_px = float(hist["Close"].iloc[-1])
        except Exception:
            pass

    expiries = tk.options or []
    rows = []
    for ex in expiries:
        try:
            ch = tk.option_chain(ex)
        except Exception:
            continue
        for side, df in (('call', ch.calls), ('put', ch.puts)):
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                oi = _to_int(r.get('openInterest', 0), default=0)
                iv = _to_float(r.get('impliedVolatility', 0.0), default=0.0)
                strike = _to_float(r.get('strike'), default=np.nan)
                if not np.isfinite(strike):
                    continue
                rows.append((side, strike, pd.Timestamp(ex, tz=NY_TZ), oi, iv, _to_int(r.get('volume',0),0), _to_float(r.get('bid',0.0),0.0), _to_float(r.get('ask',0.0),0.0), _to_float(r.get('lastPrice',0.0),0.0)))
                df = pd.DataFrame(rows, columns=['type','strike','expiry','oi','iv','volume','bid','ask','last'])
    if df.empty:
        return spot_px, df
    df['S'] = spot_px
    df = _fallback_iv_oi(df)
    df['iv'] = df['iv'].clip(0.0001, 5.0)
    df['symbol'] = symbol
    return spot_px, df
    df['S'] = spot_px; df['r'] = 0.00; df['q'] = 0.00
    df['iv'] = df['iv'].clip(0.0001, 5.0)
    return spot_px, df

@st.cache_data(ttl=120, show_spinner=False)
def compute_net_tables(df: pd.DataFrame, vex_unit='per_vol_point', dvol_per_pct_spot=-10.0):
    if df.empty: return pd.DataFrame(), pd.DataFrame(), float('nan')
    ref = session_date()
    work = df.copy()
    work['expiry'] = work['expiry'].map(ensure_ny)
    now_ts = now_ny()
    work['T'] = work['expiry'].apply(lambda ex: compute_intraday_T_years(ex, now_ts))
    S = float(work['S'].iloc[0])
    _curve = fetch_yield_curve()
    _sym = str(work['symbol'].iloc[0]) if 'symbol' in work.columns else ''
    _q = float(dividend_yield(_sym, S))
    g = []; v = []
    for row in work.itertuples(index=False):
        T = max(float(row.T), 1e-6)
        ivv = max(float(row.iv), 1e-6)
        r = float(r_for_T_years(_curve, T))
        q = _q
        g.append(bs_gamma(S, row.strike, r, q, ivv, T))
        v.append(bs_vega (S, row.strike, r, q, ivv, T))
    work['gamma']=g; work['vega']=v
    sign = np.where(work['type'].str.lower().eq('call'), 1.0, -1.0)
    contract_mult = 100.0
    work['gex'] = sign * work['gamma'] * (S*S*0.01) * work['oi'] * contract_mult
    if vex_unit == 'per_vol_point':
        work['vex'] = sign * work['vega'] * 0.01 * work['oi'] * contract_mult
    else:
        dvol_dS = (dvol_per_pct_spot/100.0) / max(S,1e-6)
        work['vex'] = sign * work['vega'] * dvol_dS * work['oi'] * contract_mult
    gex = work.pivot_table(index='strike', columns='expiry', values='gex', aggfunc='sum').sort_index().fillna(0.0)
    vex = work.pivot_table(index='strike', columns='expiry', values='vex', aggfunc='sum').sort_index().fillna(0.0)
    return gex, vex, S

def _floor_strike(strikes, spot):
    arr = np.array(list(strikes), dtype=float)
    if len(arr)==0 or not np.isfinite(spot):
        return np.nan
    floor = math.floor(spot)
    diffs = arr - floor
    diffs[diffs>0] = -1e12
    idx = int(np.argmax(diffs))
    return float(arr[idx])

def _nearest_red_magnet(col: pd.Series, start, direction, min_ratio, max_steps):
    ser = col.dropna()
    if ser.empty or start not in ser.index: return None
    m = float(np.nanmax(np.abs(ser.values)) or 1.0)
    if m <= 0: return None
    try:
        i0 = list(ser.index).index(start)
    except ValueError:
        return None
    if direction=='up':
        rng = range(i0+1, min(i0+1+max_steps, len(ser)))
    else:
        rng = range(i0-1, max(i0-1-max_steps, -1), -1)
    for i in rng:
        v = float(ser.iloc[i]); r = (abs(v)/m) if m>0 else 0.0
        if v > 0 and r >= min_ratio:
            return float(ser.index[i])
    return None

def classify_symbol(sym: str, min_ratio=0.25, max_steps=8, vex_unit='per_vol_point',
                    dvol_per_pct_spot=-10.0):
    spot, df = fetch_chain(sym)
    if df.empty or not np.isfinite(spot): 
        return None

    coverage = df.groupby('expiry')['oi'].sum().sort_index()
    usable_exps = list(coverage[coverage>0].index)
    chosen_exp = usable_exps[0] if usable_exps else sorted(df['expiry'].unique())[0]

    gex, vex, S = compute_net_tables(df, vex_unit=vex_unit, dvol_per_pct_spot=dvol_per_pct_spot)
    if gex.empty or chosen_exp not in gex.columns:
        return None

    gcol = gex[chosen_exp].copy().sort_index()
    vcol = vex[chosen_exp].copy().reindex(gcol.index) if (not vex.empty and chosen_exp in vex.columns) else pd.Series(index=gcol.index, dtype=float)

    start = _floor_strike(gcol.index, S)
    if not np.isfinite(start):
        return None

    m_g = float(np.nanmax(np.abs(gcol.values)) or 1.0)
    m_v = float(np.nanmax(np.abs(vcol.values)) or 1.0)
    g_spot = float(gcol.get(start, 0.0))
    v_spot = float(vcol.get(start, 0.0))
    g_strength = 0.0 if (m_g<=0) else abs(g_spot)/m_g
    v_strength = 0.0 if (m_v<=0) else abs(v_spot)/m_v

    g_spot_is_blue = (g_spot < 0) and (g_strength >= min_ratio)
    v_colored = (v_strength >= min_ratio)
    v_sign = 'RED' if v_spot > 0 else ('BLUE' if v_spot < 0 else 'NEUTRAL')

    up_mag = _nearest_red_magnet(gcol, start, 'up',   min_ratio, max_steps)
    dn_mag = _nearest_red_magnet(gcol, start, 'down', min_ratio, max_steps)

    # Direction without IVR:
    direction = 'NEUTRAL'
    if v_colored:
        direction = 'BULLISH' if v_sign=='RED' else ('BEARISH' if v_sign=='BLUE' else 'NEUTRAL')

    aplus_calls = bool(g_spot_is_blue and v_colored and direction=='BULLISH' and up_mag is not None)
    aplus_puts  = bool(g_spot_is_blue and v_colored and direction=='BEARISH' and dn_mag is not None)

    return dict(
        symbol=sym, spot=round(float(S),2), expiry=str(pd.to_datetime(chosen_exp).date()),
        GEX_spot=float(g_spot), VEX_spot=float(v_spot),
        GEX_spot_is_blue=g_spot_is_blue, GEX_spot_strength=round(g_strength,3),
        VEX_spot_sign=v_sign, VEX_spot_strength=round(v_strength,3),
        Magnet_Up=up_mag, Magnet_Down=dn_mag,
        Direction=direction, Aplus_Calls=aplus_calls, Aplus_Puts=aplus_puts
    )

def get_chain_df(symbol: str):
    return _fetch_chain_provider(symbol)
