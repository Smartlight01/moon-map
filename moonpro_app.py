from __future__ import annotations
from data_provider import get_spot


from datetime import datetime, time
from pathlib import Path as _Path
import json as _json
try:
    from zoneinfo import ZoneInfo as _ZoneInfo
except Exception:
    from pytz import timezone as _ZoneInfo

_ET = _ZoneInfo("America/New_York")
_SNAP_DIR = _Path(".cache/snapshots")
_SNAP_DIR.mkdir(parents=True, exist_ok=True)

def _snap_path(symbol: str) -> _Path:
    safe = ''.join([c for c in symbol if c.isalnum() or c in ('_','-')])[:24]
    return _SNAP_DIR / f"{safe}.json"

def _now_et() -> datetime:
    return datetime.now(_ET)

def _premarket_freeze_active(now: datetime) -> bool:
    return time(0, 0) <= now.time() < time(9, 30)

def _save_snapshot(symbol: str, spot, raw_df):
    try:
        import pandas as _pd
        if raw_df is None or getattr(raw_df, 'empty', True):
            return
        payload = {
            'ts': _now_et().isoformat(),
            'spot': float(spot) if spot is not None else None,
            'columns': list(raw_df.columns),
            'records': raw_df.to_dict(orient='records'),
        }
        _snap_path(symbol).write_text(_json.dumps(payload))
    except Exception:
        pass

def _load_snapshot(symbol: str):
    p = _snap_path(symbol)
    if not p.exists():
        return None, None
    try:
        import pandas as _pd
        obj = _json.loads(p.read_text())
        df = _pd.DataFrame(obj.get('records') or [], columns=obj.get('columns') or None)
        return obj.get('spot'), df
    except Exception:
        return None, None
import os, re, requests, pandas as pd, numpy as np, yfinance as yf, streamlit as st, datetime as dt
from scanner import fetch_chain, compute_net_tables
from gex_vex_ui import css, combined
from news_api import fetch_company_news_cards
import config

@st.cache_data(ttl=300, show_spinner=False)
def _atr14(sym: str):
    try:
        t = yf.Ticker(sym)
        h = t.history(period="60d", interval="1d")
        if h is None or h.empty: 
            return None
        high = h["High"].astype(float).values
        low  = h["Low"].astype(float).values
        close= h["Close"].astype(float).values
        trs = []
        prev = close[0]
        for i in range(1, len(close)):
            tr = max(high[i]-low[i], abs(high[i]-prev), abs(low[i]-prev))
            trs.append(tr)
            prev = close[i]
        if len(trs) < 14:
            return None
        atr = pd.Series(trs).rolling(14).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else None
    except Exception:
        return None


def finnhub_logo(sym: str):
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/profile2",
            params={"symbol": sym, "token": config.FINNHUB_TOKEN},
            timeout=4
        )
        if r.ok:
            j = r.json() or {}
            url = j.get("logo")
            if isinstance(url, str) and url.startswith("http"):
                return url
        return None
    except Exception:
        return None



def _aplus_bias_from_maps(spot, gex_col: pd.Series, vex_col: pd.Series, gex_strength=None, vex_strength=None):
    """A+ rules with better robustness:
    - Strong magnet = local red max with share >= 12% of |GEX| in ±15-strike window around spot.
    - Proximity uses dollar distance vs ATR(14): prox = exp(-|mag-spot| / (0.6*ATR)).
    - VEX dead-zone: if |VEX at spot| < 20th pct of |VEX| in the window -> treat VEX as NEUTRAL.
    - Same-week column only. 1–8 strikes by index still enforced.
    """
    try:
        if gex_col is None or vex_col is None or gex_col.empty or vex_col.empty or spot is None:
            return "Neutral", 0.0, {}
        strikes = gex_col.index.astype(float)
        S = float(spot)
        k_spot = int(np.argmin(np.abs(strikes - S)))
        # window around spot
        lo = max(0, k_spot-15); hi = min(len(strikes), k_spot+16)
        gwin = gex_col.iloc[lo:hi].astype(float)
        vwin = vex_col.iloc[lo:hi].astype(float)
        absG = np.abs(gwin.values); totalG = float(np.nansum(absG)) or 1.0
        # VEX at spot with dead-zone
        v_spot_val = float(vex_col.iloc[k_spot])
        v_abs_p20 = np.nanpercentile(np.abs(vwin.values), 20) if len(vwin)>0 else 0.0
        if abs(v_spot_val) < (v_abs_p20 or 0.0):
            v_spot_sign = "NEUTRAL"
        else:
            v_spot_sign = "RED" if v_spot_val > 0 else "BLUE"
        # GEX at spot must be BLUE lane (negative)
        g_spot_val = float(gex_col.iloc[k_spot])
        g_spot_is_blue = (g_spot_val < 0)
        # local maxima detection for strong red magnet
        def strong_red_indices():
            idxs = []
            for i in range(lo+1, hi-1):
                v = float(gex_col.iloc[i])
                if v > 0:  # red
                    left = float(gex_col.iloc[i-1]); right = float(gex_col.iloc[i+1])
                    if v >= left and v >= right:
                        share = abs(v) / totalG
                        if share >= 0.12:  # >=12% share
                            idxs.append(i)
            return idxs
        reds = strong_red_indices()
        # closest strong red above/below within 1..8 strikes by index
        up_ix = next((i for i in range(k_spot+1, min(k_spot+9, len(strikes))) if i in reds), None)
        dn_ix = next((i for i in range(k_spot-1, max(-1, k_spot-9), -1) if i in reds), None)
        have_up = up_ix is not None; have_dn = dn_ix is not None
        calls_ok = (v_spot_sign == "RED") and g_spot_is_blue and have_up
        puts_ok  = (v_spot_sign == "BLUE") and g_spot_is_blue and have_dn
        # Confidence
        conf = 0.0
        if calls_ok or puts_ok:
            ix = up_ix if calls_ok else dn_ix
            dist_idx = abs(ix - k_spot)
            # ATR-based proximity
            atr = _atr14(symbol)  # uses global symbol
            if atr and atr > 0:
                dist_dollar = abs(float(strikes[ix]) - S)
                prox = float(np.exp(- dist_dollar / (0.6*atr)))
            else:
                # fallback to index distance
                prox = max(0.0, min(1.0, (9 - dist_idx)/8.0))
            # magnitude share
            mag_share = abs(float(gex_col.iloc[ix])) / totalG
            mag_share = float(max(0.0, min(1.0, mag_share*4.0)))  # rescale to 0..~1
            sG = float(max(0.5, min(1.0, (gex_strength if gex_strength is not None else 1.0))))
            sV = float(max(0.5, min(1.0, (vex_strength if vex_strength is not None else 1.0))))
            conf = max(0.0, min(1.0, 0.5*prox + 0.35*mag_share + 0.15*((sG+sV)/2.0)))
        label = "Bullish" if calls_ok else ("Bearish" if puts_ok else "Neutral")
        extras = dict(
            v_spot=v_spot_sign, g_spot_blue=g_spot_is_blue,
            red_up_strike=float(strikes[up_ix]) if have_up else None,
            red_dn_strike=float(strikes[dn_ix]) if have_dn else None,
        )
        return label, conf, extras
    except Exception:
        return "Neutral", 0.0, {}
        strikes = gex_col.index.astype(float)
        S = float(spot)
        # nearest strike index
        k_spot = int(np.argmin(np.abs(strikes - S)))
        # spot signs
        g_spot_val = float(gex_col.iloc[k_spot])
        v_spot_val = float(vex_col.iloc[k_spot])
        g_spot_is_blue = (g_spot_val < 0)   # BLUE lane when negative
        v_spot_sign = "RED" if v_spot_val > 0 else ("BLUE" if v_spot_val < 0 else "NEUTRAL")
        # strong threshold
        p80 = np.nanpercentile(np.abs(gex_col.values), 80)
        def strong_red(ix):
            if ix<0 or ix>=len(gex_col): return False
            val = float(gex_col.iloc[ix])
            return (val > 0) and (abs(val) >= p80 if np.isfinite(p80) and p80>0 else True)
        # search windows 1..8 strikes away
        up_ix = next((i for i in range(k_spot+1, min(k_spot+9, len(gex_col))) if strong_red(i)), None)
        dn_ix = next((i for i in range(k_spot-1, max(-1, k_spot-9), -1) if strong_red(i)), None)
        have_up = up_ix is not None
        have_dn = dn_ix is not None
        # Conditions
        calls_ok = (v_spot_sign == "RED") and g_spot_is_blue and have_up
        puts_ok  = (v_spot_sign == "BLUE") and g_spot_is_blue and have_dn
        # Confidence (0..1): proximity & magnitude & map strength
        conf = 0.0
        if calls_ok or puts_ok:
            ix = up_ix if calls_ok else dn_ix
            dist_idx = abs(ix - k_spot)
            prox = max(0.0, min(1.0, (9 - dist_idx)/8.0))  # 1 step -> 1.0 ; 8 steps -> 0.125
            mag = abs(float(gex_col.iloc[ix])) / (p80 if (p80 and p80>0) else (abs(float(gex_col.iloc[ix])) or 1.0))
            mag = max(0.0, min(1.0, mag))
            sG = float(max(0.5, min(1.0, (gex_strength if gex_strength is not None else 1.0))))
            sV = float(max(0.5, min(1.0, (vex_strength if vex_strength is not None else 1.0))))
            conf = max(0.0, min(1.0, 0.5*prox + 0.3*mag + 0.2*((sG+sV)/2.0)))
        label = "Bullish" if calls_ok else ("Bearish" if puts_ok else "Neutral")
        extras = dict(
            v_spot=v_spot_sign, g_spot_blue=g_spot_is_blue,
            red_up_strike=float(strikes[up_ix]) if have_up else None,
            red_dn_strike=float(strikes[dn_ix]) if have_dn else None,
            dist_idx=int(dist_idx) if (calls_ok or puts_ok) else None
        )
        return label, conf, extras
    except Exception:
        return "Neutral", 0.0, {}




def _gex_vex_bias(spot, near_gex, near_vex, gex_strength=None, vex_strength=None):
    """Return (label, score in [-1,1]) using GEX/VEX magnet direction & proximity.
    Changes:
      - Proximity window ~3%% of spot (min 0.5)
      - Strength floor 0.5
      - Neutral band tightened to 0.10
      - If magnet equals spot, give a modest weight (0.6) instead of near-zero
    """
    try:
        spot = get_spot(symbol)
    except Exception:
        spot = None
        if spot is None or not (spot == spot):  # NaN guard
            return "Neutral", 0.0
        thr = max(0.03 * float(spot), 0.5)  # ~3% of spot, at least 0.5
        def contrib(near, strength):
            if near is None or not (near == near):
                return 0.0
            d = float(near) - float(spot)
            dist = abs(d)
            if dist < 1e-9:
                w = 0.6  # exactly at spot: modest influence
            else:
                w = max(0.0, 1.0 - min(1.0, dist / thr))
            s_mod = float(max(0.5, min(1.0, (strength if strength is not None else 1.0))))
            # direction: above spot pulls up, below pulls down
            return (1.0 if d > 0 else -1.0) * w * s_mod
        score = contrib(near_gex, gex_strength) + contrib(near_vex, vex_strength)
        if abs(score) < 0.10:
            return "Neutral", score
        return ("Bullish" if score > 0 else "Bearish"), score
    except Exception:
        return "Neutral", 0.0


@st.cache_data(ttl=180, show_spinner=False)
def yf_info(sym: str):
    # Single source of truth for Yahoo Finance data; exact values, cached briefly
    return yf.Ticker(sym).get_info()


st.set_page_config(page_title="Moon Map", layout="wide", page_icon="🌙")
st.markdown('<style>div.block-container{padding-top:1.0rem;}</style>', unsafe_allow_html=True)
css()

# Header
c1, c2 = st.columns([1,5])
with c1:
    try: st.image("MOONPRO LOGO 1.png", width=112)
    except: st.write("MoonPro")
with c2:
    st.markdown('<h1 class="maintitle">Moon Map</h1>', unsafe_allow_html=True)

row0 = st.columns([1,6])
with row0[0]:
    symbol = st.text_input("Ticker", value="SPY").strip().upper()

def rt_quote(sym: str):
    try:
        r = requests.get("https://finnhub.io/api/v1/quote",
                         params={"symbol": sym, "token": config.FINNHUB_TOKEN}, timeout=6)
        j = r.json() if r.ok else {}
        spot = float(j.get("c") or "nan")
        prev = float(j.get("pc") or "nan")
        change = float(j.get("d") or (spot - prev if spot==spot and prev==prev else 0.0))
        pct = float(j.get("dp") or ((spot/prev - 1)*100 if spot==spot and prev==prev else 0.0))
        return spot, prev, change, pct
    except Exception:
        return None, None, None, None

@st.cache_data(ttl=180, show_spinner=False)
def todays_volume(sym: str):
    try:
        gi = yf_info(sym)
        v = gi.get('volume') or gi.get('regularMarketVolume')
        return int(v) if v else None
    except Exception:
        return None

@st.cache_data(ttl=180, show_spinner=False)
def avg_volume(sym: str):
    try:
        gi = yf_info(sym)
        v = gi.get('averageVolume')
        return int(v) if v else None
    except Exception:
        return None

@st.cache_data(ttl=180, show_spinner=False)
def one_year_target(sym: str):
    try:
        gi = yf_info(sym)
        v = gi.get("targetMeanPrice") or gi.get("targetMeanPriceRaw")
        return float(v) if v else None
    except Exception: return None

@st.cache_data(ttl=180, show_spinner=False)
def next_earnings_date(sym: str):
    # Pull exactly what Yahoo provides; no heuristics.
    try:
        gi = yf_info(sym)
        # Prefer explicit 'nextEarningsDate' (string like '2025-11-14') if present.
        ned = gi.get('nextEarningsDate')
        if isinstance(ned, str) and len(ned) >= 8:
            return ned
        # Otherwise, Yahoo often provides a start/end timestamp window.
        start_ts = gi.get('earningsTimestampStart')
        end_ts   = gi.get('earningsTimestampEnd')
        def fmt(ts):
            try:
                return dt.datetime.fromtimestamp(int(ts)).strftime('%Y-%m-%d') if ts else None
            except Exception:
                return None
        s = fmt(start_ts); e = fmt(end_ts)
        if s and e and s != e:
            return f"{s} – {e}"
        if s: return s
        if e: return e
        # Fallback to last earnings timestamp if provided.
        last_ts = gi.get('earningsTimestamp')
        if last_ts:
            try:
                return dt.datetime.fromtimestamp(int(last_ts)).strftime('%Y-%m-%d')
            except Exception:
                return None
        return None
    except Exception:
        return None
        return None
    except Exception:
        return None

@st.cache_data(ttl=180, show_spinner=False)
def company_info(sym: str):
    t = yf.Ticker(sym)
    info={}
    try:
        gi=yf_info(sym)
        info["name"]=gi.get("longName") or gi.get("shortName",""); info["logo"]=gi.get("logo_url"); info["ex"]=gi.get("exchange") or gi.get("fullExchangeName") or ""
        info["ex"]=gi.get("exchange") or gi.get("fullExchangeName","")
        info["ind"] = gi.get("industry", "")
        info["sector"] = gi.get("sector", "")
        info["beta"]=gi.get("beta") or gi.get("beta5YMonthly") or gi.get("beta3Year")
        info["shortFloatPct"]=gi.get("shortPercentOfFloat")
        if isinstance(info["shortFloatPct"],float): info["shortFloatPct"]*=100.0
        info["offHigh"]=None
        try:
            year_high = gi.get("fiftyTwoWeekHigh") or gi.get("yearHigh")
            year_low = gi.get("fiftyTwoWeekLow") or gi.get("yearLow")
            info["yearHigh"] = float(year_high) if year_high else None
            info["yearLow"] = float(year_low) if year_low else None
            info["marketCap"] = gi.get("marketCap") or gi.get("market_cap")
            hist=t.history(period="1d"); spot=None
            if not hist.empty: spot=float(hist["Close"].iloc[-1])
            if spot and year_high: info["offHigh"]=round(100*(1-spot/float(year_high)),2)
        except Exception: pass
    except Exception: pass
    return info

info = company_info(symbol)
rt_spot, rt_prev, chg, chg_pct = rt_quote(symbol)
avg_vol = avg_volume(symbol)
target_est = one_year_target(symbol)
earn = next_earnings_date(symbol)

st.markdown("""
<style>
.symbolcard .name{font-weight:700; font-size:1.05rem}
.symbolcard .line2{opacity:0.85; font-size:0.9rem}
.up{color:#2ecc71;font-weight:600}
.down{color:#e74c3c;font-weight:600}
.neut{color:#90a4ae;font-weight:600}
</style>
""", unsafe_allow_html=True)

ra = st.columns([3,2,2,2,2,2])
with ra[0]:
    st.markdown('<div class="card symbolcard">', unsafe_allow_html=True)
    nm = info.get("name") or symbol
    ex = info.get("ex","")
    logo = finnhub_logo(symbol) or info.get("logo")
    price = f'{rt_spot:,.2f}' if rt_spot else '—'
    
    # Safe sign-aware diff coloring
    diff = ''
    if (chg is not None) and (chg_pct is not None) and (rt_spot is not None):
        if chg > 0:
            diff = f' <span class="up">{chg:+.2f} ({chg_pct:+.2f}%)</span>'
        elif chg < 0:
            diff = f' <span class="down">{chg:+.2f} ({chg_pct:+.2f}%)</span>'
        else:
            diff = f' <span class="neut">{chg:+.2f} ({chg_pct:+.2f}%)</span>'
# diff set below
    logo_html = f'<img src="{logo}" style="height:22px;vertical-align:middle;margin-right:8px;border-radius:4px">' if logo else ''
    st.markdown(f'<div class="line1">{logo_html}<span class="name">{symbol} — {ex}</span></div><div class="line2">{nm} · {info.get("sector","")}</div><div class="symbolprice">{price}{diff}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with ra[1]:
    vnow = todays_volume(symbol)
    vtxt = f"{vnow:,.0f}" if vnow else "—"
    st.markdown(f'<div class="card"><div class="kicker">Volume</div><div class="big">{vtxt}</div></div>', unsafe_allow_html=True)
with ra[2]: st.markdown(f'<div class="card"><div class="kicker">Beta</div><div class="big">{info.get("beta"):.2f}</div></div>' if info.get("beta") is not None else '<div class="card"><div class="kicker">Beta</div><div class="big">—</div></div>', unsafe_allow_html=True)
with ra[3]: st.markdown(f'<div class="card"><div class="kicker">1y Target Est</div><div class="big">{target_est:.2f}</div></div>' if target_est is not None else '<div class="card"><div class="kicker">1y Target Est</div><div class="big">—</div></div>', unsafe_allow_html=True)
with ra[4]: st.markdown(f'<div class="card"><div class="kicker">Short Float %</div><div class="big">{info.get("shortFloatPct"):.2f}%</div></div>' if info.get("shortFloatPct") is not None else '<div class="card"><div class="kicker">Short Float %</div><div class="big">—</div></div>', unsafe_allow_html=True)
with ra[5]:
    etxt = earn if earn else '—'
    st.markdown(f'<div class="card"><div class="kicker">Earnings Date</div><div class="big">{etxt}</div></div>', unsafe_allow_html=True)

# Second header row aligned with first-row columns
st.markdown('<div class="rowtight">', unsafe_allow_html=True)
rb = st.columns([3,2,2,2,2,2])
with rb[1]:
    av = avg_volume(symbol)
    val = f"{av:,.0f}" if av else "—"
    st.markdown(f'<div class="card"><div class="kicker">Avg Volume</div><div class="big">{val}</div></div>', unsafe_allow_html=True)
with rb[2]:
    _mag_placeholder = st.empty()
    _mag_placeholder.markdown(
        '<div class="card"><div class="big">—</div><div class="big">—</div></div>',
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)
with rb[3]:
    yhi = info.get("yearHigh")
    ylo = info.get("yearLow")
    hi_txt = f"{yhi:,.2f}" if yhi else "—"
    lo_txt = f"{ylo:,.2f}" if ylo else "—"
    st.markdown(
        f'<div class="card"><div class="kicker">52W High / Low</div>'
        f'<div class="big"><span class="up">{hi_txt}</span> / <span class="down">{lo_txt}</span></div></div>',
        unsafe_allow_html=True
    )

with rb[4]:
    mcap = info.get("marketCap")
    mcap_txt = "—"
    if mcap:
        if mcap >= 1e12:
            mcap_txt = f"{mcap/1e12:.2f}T"
        elif mcap >= 1e9:
            mcap_txt = f"{mcap/1e9:.2f}B"
        elif mcap >= 1e6:
            mcap_txt = f"{mcap/1e6:.2f}M"
        else:
            mcap_txt = f"{mcap:,.0f}"
    st.markdown(
        f'<div class="card"><div class="kicker">Market Cap</div><div class="big">{mcap_txt}</div></div>',
        unsafe_allow_html=True
    )

with rb[5]:
    # Yahoo-only PE
    try:
        y = yf.Ticker(symbol).get_info()
    except Exception:
        y = {}
    pe = None
    try:
        v = y.get("trailingPE")
        if v and float(v) > 0:
            pe = float(v)
    except Exception:
        pass
    if pe is None:
        try:
            v = y.get("forwardPE")
            if v and float(v) > 0:
                pe = float(v)
        except Exception:
            pass
    petxt = f"{pe:.2f}" if isinstance(pe, (int,float)) and pe and pe > 0 else "—"
    st.markdown(
        f'<div class="card"><div class="kicker">P/E</div><div class="big">{petxt}</div></div>',
        unsafe_allow_html=True
    )

# Options chain -> GEX/VEX
MIN_RATIO=0.25; MAX_STEPS=8; ROWS=25; VEX_UNIT="per_vol_point"; ELASTICITY=-10.0
# Toggle to show full strike range
full_chain = st.checkbox('Show all strikes', value=False, help='Enable to view the entire chain range instead of the 25-row window.')
if full_chain:
    # Use an effectively unlimited window so the renderer takes the whole chain
    ROWS = 10**6
now = _now_et()
try:
    live_spot, live_raw = fetch_chain(symbol)
except Exception:
    live_spot, live_raw = None, pd.DataFrame()

if live_raw is not None and not getattr(live_raw, 'empty', True) and now.time() >= time(23, 0):
    _save_snapshot(symbol, live_spot, live_raw)

if _premarket_freeze_active(now):
    snap_spot, snap_raw = _load_snapshot(symbol)
    if snap_raw is not None and not getattr(snap_raw, 'empty', True):
        spot, raw = (snap_spot if snap_spot is not None else live_spot), snap_raw
    else:
        spot, raw = live_spot, live_raw
else:
    spot, raw = live_spot, live_raw
if raw is None or raw.empty:
    st.error("Could not fetch a usable option chain for this symbol."); st.stop()
gex, vex, S = compute_net_tables(raw, vex_unit=VEX_UNIT, dvol_per_pct_spot=ELASTICITY)

cov = raw.groupby('expiry')['oi'].sum().sort_index()
exps = list(cov[cov>0].index) or sorted(raw['expiry'].unique())
p1, p2, _ = st.columns([1,1,8])
if p1.button('◀ Prev', key='exp_prev') and st.session_state.get('exp_page', 0) > 0:
    st.session_state['exp_page'] -= 1
if p2.button('Next ▶', key='exp_next'):
    st.session_state['exp_page'] = st.session_state.get('exp_page', 0) + 1
page = int(st.session_state.get('exp_page', 0))
start = max(0, page*4)
end = start + 4
exps = exps[start:end]
if not exps:
    total = len(cov)
    last_page = max(0, (max(0, total-1)) // 4)
    st.session_state['exp_page'] = last_page
    start = last_page*4; end = start+4
    exps = list(cov[cov>0].index)[start:end] if len(cov)>0 else []
oi_tbl = raw.pivot_table(values='oi', index='strike', columns='expiry', aggfunc='sum')
exp_gex = {e: gex[e].copy().sort_index() for e in exps if e in gex.columns}

# Compute strongest positive and strongest negative strikes across the visible GEX expiries
pin_strike = None
moon_level = None
try:
    best_pos_val = 0.0
    best_neg_val = 0.0
    for _e, _ser in exp_gex.items():
        if _ser is None or _ser.empty:
            continue
        try:
            _k_pos = _ser.idxmax()
            _v_pos = float(_ser.loc[_k_pos])
            if _v_pos > best_pos_val:
                best_pos_val = _v_pos
                pin_strike = float(_k_pos)
        except Exception:
            pass
        try:
            _k_neg = _ser.idxmin()
            _v_neg = float(_ser.loc[_k_neg])
            if _v_neg < best_neg_val:
                best_neg_val = _v_neg
                moon_level = float(_k_neg)
        except Exception:
            pass
except Exception:
    pass

exp_vex = {e: vex[e].copy().sort_index() for e in exps if e in vex.columns}
oi_map = {e: (oi_tbl[e].copy().sort_index() if e in oi_tbl.columns else None) for e in exps}



lc, rc = st.columns(2, gap="large")
with lc:
    st.markdown("### NetGEX")
    html, near_gex, gex_strength = combined(exp_gex, oi_map, S, MIN_RATIO, MAX_STEPS, ROWS)
    st.markdown(html, unsafe_allow_html=True)

    # Compute Moon level (most negative GEX across all displayed expiries)
    moon_level = None
    try:
        min_val = 0.0
        for _e, _ser in exp_gex.items():
            try:
                if _ser is None or _ser.empty: continue
                _k = _ser.idxmin()
                _v = float(_ser.loc[_k])
                if _v < min_val:
                    min_val = _v
                    moon_level = float(_k)
            except Exception:
                pass
    except Exception:
        moon_level = None

    # Update Pin/Moon card
try:
    pin_txt  = f"${pin_strike:,.2f}" if pin_strike is not None else "—"
    moon_txt = f"${moon_level:,.2f}" if moon_level is not None else "—"
    _mag_placeholder.markdown(
        f'<div class="card"><div class="big">📌 {pin_txt}</div><div class="big">🌙 {moon_txt}</div></div>',
        unsafe_allow_html=True
    )
except Exception:
    pass

    # Update Magnet Strike box with the strike closest to spot from NetGEX
    try:
        if near_gex is not None:
            _mag_placeholder.markdown(
                f'<div class="card"><div class="big">📌 ${near_gex:,.2f}</div>'
                f'<div class="big">{moon_txt}</div></div>',
                unsafe_allow_html=True
            )
    except Exception:
        pass
with rc:
    st.markdown("### NetVEX")
    html2, near_vex, vex_strength = combined(exp_vex, oi_map, S, MIN_RATIO, MAX_STEPS, ROWS, mode='vex')
    st.markdown(html2, unsafe_allow_html=True)

    # 52W metrics displayed above. Bias box removed.

# News
with st.expander("Company News"):
    cards = fetch_company_news_cards(symbol, config.FINNHUB_TOKEN)
    if not cards: st.info("No recent headlines.")
    else:
        for c in cards:
            with st.container(border=True):
                st.markdown(f"### [{c['headline']}]({c['url']})")
                st.caption(f"{c['source']} • {c['datetime']}")
                if c.get('summary'): st.write(c['summary'])

# Optional cached wrapper to speed up repeated renders without changing compute logic
@st.cache_data(ttl=60, show_spinner=False)
def compute_net_tables_cached(key, df, vex_unit, dvol_per_pct_spot):
    from scanner import compute_net_tables
    return compute_net_tables(df, vex_unit=vex_unit, dvol_per_pct_spot=dvol_per_pct_spot)
