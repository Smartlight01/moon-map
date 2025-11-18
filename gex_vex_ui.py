from __future__ import annotations
import pandas as pd, numpy as np, streamlit as st
MIN_OI_PER_STRIKE = 500
from typing import Dict, Optional

def _format_compact(x: float) -> str:
    try:
        n = float(x)
    except Exception:
        return str(x)
    sign = '-' if n < 0 else ''
    n = abs(n)
    if n >= 1_000_000:
        v = n / 1_000_000.0
        s = 'M'
    elif n >= 1_000:
        v = n / 1_000.0
        s = 'K'
    else:
        return f"{sign}{int(round(n))}"
    txt = f"{v:.1f}"
    if txt.endswith('.0'):
        txt = txt[:-2]
    return f"{sign}{txt}{s}"


def _format_m(x: float) -> str:
    try:
        n = float(x)
    except Exception:
        return str(x)
    sign = '-' if n < 0 else ''
    n = abs(n)
    m = n/1_000_000.0
    # No decimals for clean grid. 28,400,000 -> 28M
    return f"{sign}{int(round(m))}M"


def _format_km(x: float) -> str:
    try:
        n = float(x)
    except Exception:
        return str(x)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1_000_000:
        return f"{sign}${n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{sign}${n/1_000:.1f}K"
    return f"{sign}${n:.0f}"

def _lerp(c1, c2, t: float):
    # c1, c2 are (r,g,b)
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(c1[i] + (c2[i]-c1[i]) * t)) for i in range(3))

def _rgba(c, a: float) -> str:
    a = max(0.0, min(1.0, float(a)))
    return f"rgba({c[0]},{c[1]},{c[2]},{a:.3f})"

def css():
    css_block = """
    <style>
      .meter{height:10px;background:rgba(255,255,255,0.08);border-radius:8px;overflow:hidden;margin-top:6px}
      .meterfill{height:100%;}
.meterfill.bullfill{background:linear-gradient(90deg,#a5d6a7,#2ecc71);} /* green */
.meterfill.bearfill{background:linear-gradient(90deg,#f5b7b1,#e74c3c);} /* red */
.meterfill.neutf{background:linear-gradient(90deg,#cfd8dc,#90a4ae);} /* gray */
      .metercap{font-size:.8rem;opacity:.8;margin-top:4px;text-align:right}

      .bull{color:#2ecc71;font-weight:700}
      .bear{color:#e74c3c;font-weight:700}
      .neut{color:#bdc3c7;font-weight:700}
      .maintitle{ text-align:center; font-size:2.6rem; font-weight:800; letter-spacing:.3px; margin:0.2rem 0 0.8rem; }
      .rowtight{margin-top:-16px}
      body, .stApp { background:#000 !important; }
      .combo { border:1px solid rgba(255,255,255,0.12); border-radius:10px; max-height: 75vh; overflow:auto }
      .combo table { /* rowheight */ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
      .combo th, .combo td { padding:6px 8px; border-bottom:1px solid rgba(255,255,255,0.08); text-align:right; color:#e8e8e8 }
      .combo thead th { position:sticky; top:0; background:#000; z-index:5 }
      .strikecol { text-align:center; width:6rem; border-right:1px solid rgba(255,255,255,0.08); color:#fff; }
      .spot-cell { border:3px solid #ffd400; background:#f1c40f44 !important; box-shadow:0 0 0 2px rgba(255,212,0,0.25), 0 0 12px rgba(255,212,0,0.35) inset; position:relative; }
      .mag-star { font-weight:800; color:#f39c12; margin-right:4px; }
      .card { border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:12px; margin-bottom:8px; background:#0a0a0a }
      .kicker { opacity:0.7; font-size:0.8rem; margin-bottom:4px; color:#cfcfcf }
      .big { font-size:1.2rem; font-weight:700; color:#fff }
      .symbolprice { font-size:1.6rem; font-weight:800 }
      .up{color:#2ecc71} .down{color:#e74c3c}
      .biasbar { height:8px; background:#333; border-radius:6px; overflow:hidden; margin-top:4px }
      .biasbar > div { height:8px; background:#2ecc71 }
    
      .strike-spot { background:#f1c40f44 !important; border:3px solid #ffd400; box-shadow:0 0 0 2px rgba(255,212,0,0.25), 0 0 12px rgba(255,212,0,0.35) inset; position:relative; }
    </style>
    """
    st.markdown(css_block, unsafe_allow_html=True)

def _fmt_money(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "$0.00"
    sgn = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1_000_000: s = f"{v/1_000_000:.1f}M"
    elif v >= 1_000: s = f"{v/1_000:.1f}K"
    else: s = f"{v:.2f}"
    return f"{sgn}${s}"

def _closest_idx(strikes, spot):
    return int(np.argmin([abs(k-spot) for k in strikes]))

def _window(series: pd.Series, spot: float, rows: int=25):
    ser = series.dropna().sort_index()
    if ser.empty: return ser, None, None, None
    strikes = list(ser.index)
    i0 = _closest_idx(strikes, spot)
    half = max(1, rows//2)
    lo = max(0, i0-half); hi = min(len(strikes), i0+half+1)
    return ser.iloc[lo:hi], strikes[i0], lo, hi


def combined(expiry_to_series, expiry_to_oi, spot, min_ratio, max_steps, rows=25, mode="gex"):
    import numpy as np, pandas as pd
    # Preserve expiry order; front expiry is first
    exps = list(expiry_to_series.keys())
    if not exps:
        return "<div>No data</div>", None, 0.0
    front = exps[0]

    # Window around spot based on front expiry
    def _closest_idx(vals, x):
        return int(np.argmin([abs(v - x) for v in vals]))
    base = expiry_to_series[front].dropna().sort_index()
    if base.empty:
        return "<div>No strikes</div>", None, 0.0
    all_strikes = list(base.index)
    i0 = _closest_idx(all_strikes, spot)
    half = max(1, rows//2)
    lo = max(0, i0 - half); hi = min(len(all_strikes), i0 + half + 1)
    strikes = all_strikes[lo:hi]
    spot_k = all_strikes[i0]

    # Align visible slices
    aligned, oialigned, maxabs = {}, {}, {}
    for e in exps:
        s = expiry_to_series[e].reindex(strikes).fillna(0.0)
        aligned[e] = s
        # avoid zero max for normalization
        maxabs[e] = float(np.nanmax(np.abs(s.values))) if len(s.values) else 1.0
        if maxabs[e] == 0.0: maxabs[e] = 1.0
        oi = expiry_to_oi.get(e) if expiry_to_oi else None
        oialigned[e] = oi.reindex(strikes).fillna(0.0) if oi is not None else None

    # Filter out strikes where every expiry value is zero
    try:
        if strikes:
            import pandas as _pd
            _df_chk = _pd.DataFrame({e: aligned[e].values for e in exps}, index=strikes).fillna(0.0)
            _keep = ~(_df_chk.abs().sum(axis=1) == 0.0)
            strikes = [k for k, keep in zip(list(_df_chk.index), list(_keep.values)) if keep]
            for e in exps:
                aligned[e] = aligned[e].reindex(strikes).fillna(0.0)
                if oialigned.get(e) is not None:
                    oialigned[e] = oialigned[e].reindex(strikes).fillna(0.0)
    except Exception:
        pass


    # Global scale so same dollar = same shade
    gmax = max([v for v in maxabs.values()] + [1.0])
    # Use global p95 cap to avoid extremes crushing the palette
    _all_vals = []
    try:
        import numpy as _np
        for _e in exps:
            _s = aligned[_e].values if _e in aligned else []
            _all_vals.extend([abs(float(_v)) for _v in list(_s)])
        p95cap = float(_np.nanpercentile(_all_vals, 95)) if _all_vals else gmax
    except Exception:
        p95cap = gmax
    cap = max(1.0, p95cap)

    # Compute strongest POSITIVE (red) across ALL expiries; if none, disable purple
    strongest_key = None  # (expiry, float(strike))
    try:
        best_val = -1.0
        for _e in exps:
            _s = aligned.get(_e)
            if _s is None or _s.empty:
                continue
            for _k, _v in _s.items():
                _fv = float(_v)
                if _fv > 0 and _fv > best_val:
                    best_val = _fv
                    strongest_key = (_e, float(_k))
        if best_val <= 0:
            strongest_key = None
    except Exception:
        strongest_key = None

    # Marker placement
    # Most negative strike (for dark blue highlight)
    mostneg_key = None
    try:
        best_neg = 0.0
        for _e, _s in aligned.items():
            if _e not in exps or _s is None:
                continue
            for _k, _v in _s.items():
                try:
                    _fv = float(_v)
                except Exception:
                    continue
                if _fv < best_neg:
                    best_neg = _fv
                    mostneg_key = (_e, float(_k))
        if best_neg >= 0.0:
            mostneg_key = None
    except Exception:
        mostneg_key = None

    mark_pos = None  # (expiry, strike)
    strength = 0.0
    gatekeeper = None  # (expiry, strike)
    if mode == "gex":
        # Magnet on strongest RED (positive) in FRONT expiry only
        fser = aligned[front]
        cand = [(k, float(v)) for k, v in fser.items() if float(v) > 0]
        if cand:
            k_best, v_best = max(cand, key=lambda kv: kv[1])
            mark_pos = (front, k_best)
            strength = abs(v_best) / (gmax or 1.0)
        # Moon on strongest negative across visible map
        moon_pos = mostneg_key if 'mostneg_key' in locals() else None
        
        # Gatekeeper: strongest opposite-sign node between spot and king on front expiry
        try:
            if mark_pos is not None:
                e_gate = mark_pos[0]
                k_king = float(mark_pos[1])
                fser = aligned.get(e_gate)
                if fser is not None and len(strikes) > 0:
                    if k_king >= spot_k:
                        path = [k for k in strikes if spot_k <= k <= k_king]
                    else:
                        path = [k for k in strikes if k_king <= k <= spot_k]
                    if len(path) > 2:
                        path = path[1:-1]
                    king_val = float(fser.loc[k_king]) if k_king in fser.index else 0.0
                    if king_val != 0.0:
                        opp = [(k, float(fser.loc[k])) for k in path if (float(fser.loc[k]) * king_val) < 0]
                        if opp:
                            k_gk, v_gk = max(opp, key=lambda kv: abs(kv[1]))
                            gatekeeper = (e_gate, float(k_gk))
        except Exception:
            gatekeeper = None
# OI weighting
        try:
            oi_series = oi_map.get(e_best) if isinstance(oi_map, dict) else None
            if oi_series is not None:
                oi_here = float(oi_series.get(k_best, 0.0) or 0.0)
                total_oi = float(oi_series.fillna(0).sum() or 1.0)
                nb = [k_best-1, k_best, k_best+1]
                oi_near = sum(float(oi_series.get(k, 0.0) or 0.0) for k in nb)
                oi_share = max(oi_near / total_oi, 1e-6)
                if oi_here < MIN_OI_PER_STRIKE:
                    strength = 0.0
                else:
                    strength = strength * (oi_share ** 0.5)
        except Exception:
            pass

        mark_char, mark_style = "📌", 'style="color:#c0392b"'
    else:
        # VEX map: 📌 strongest positive across visible map; 🌙 strongest negative across visible map
        best_pos = None  # (val, expiry, strike)
        best_neg = None  # (val, expiry, strike)
        for e in exps:
            s = aligned.get(e)
            if s is None or s.empty:
                continue
            try:
                k_pos = s.idxmax()
                v_pos = float(s.loc[k_pos])
                if v_pos > 0 and (best_pos is None or v_pos > best_pos[0]):
                    best_pos = (v_pos, e, float(k_pos))
            except Exception:
                pass
            try:
                k_neg = s.idxmin()
                v_neg = float(s.loc[k_neg])
                if v_neg < 0 and (best_neg is None or v_neg < best_neg[0]):
                    best_neg = (v_neg, e, float(k_neg))
            except Exception:
                pass
        if best_pos is not None:
            v_best, e_best, k_best = best_pos
            mark_pos = (e_best, k_best)
            strength = abs(v_best) / (gmax or 1.0)
        else:
            mark_pos = None
            strength = 0.0
        mark_char, mark_style = "📌", 'style="color:#c0392b"'
        moon_pos = (best_neg[1], best_neg[2]) if best_neg is not None else None

    # Build HTML
    head = '<div class="combo"><table><thead><tr><th class="strikecol">Strike</th>' + ''.join( (lambda d: f"<th>{d.date()} 🚨</th>" if (d.weekday()==4 and 15<=d.day<=21) else f"<th>{d.date()}</th>") (pd.to_datetime(e)) for e in exps) + '</tr></thead><tbody>'
    rows_html = [head]
    for k in strikes:
        strike_cls = 'strikecol strike-spot' if abs(k - spot_k) < 1e-9 else 'strikecol'
        row = [f'<tr><th class="{strike_cls}">{k:g}</th>']
        for e in exps:
            v = float(aligned[e].loc[k])
            frac = abs(v) / cap
            # No OI weighting in color
            # alpha precompute line removed to avoid confusion
                # Muted two-stop gradients with smooth darkening
            # t is normalized magnitude in [0,1]
            t = max(0.0, min(1.0, frac))
            # 4-tier discrete palette using global cap (p95) and same-dollar = same tier
            # Determine tier by t
            if t < 0.25:
                tier = 0
            elif t < 0.50:
                tier = 1
            elif t < 0.75:
                tier = 2
            else:
                tier = 3
            # Palettes
            pos_colors = [(253,232,215), (240,180,170), (214,105,110), (190,50,65)]
            neg_colors = [(230,244,234), (200,225,220), (120,160,190), (80,120,170)]
            rgb = (pos_colors if v > 0 else neg_colors)[tier]
            # Alpha by tier, easy on eyes
            alphas = [0.16, 0.32, 0.55, 0.78]
            alpha = alphas[tier]
            bg = _rgba(rgb, alpha)
            # Purple highlight for strongest strike in this expiry
            # Dark blue highlight for most negative strike in this expiry
            if 'mostneg_key' in locals() and mostneg_key is not None and mostneg_key == (e, float(k)):
                bg = _rgba((0, 32, 96), max(0.42, alpha))  # dark blue

            if strongest_key is not None and strongest_key == (e, float(k)):
                bg = _rgba((140, 82, 255), max(0.28, alpha))

            is_mark = (mark_pos is not None and mark_pos[0] == e and abs(mark_pos[1] - k) < 1e-9)
            badge = f'<span class="badge" {mark_style}>{mark_char}</span> ' if is_mark else ''
            gk_badge = '<span class="badge" style="color:#ff6b6b; filter: drop-shadow(0 0 4px rgba(255,0,0,0.8)); font-weight:900">🚧</span> ' if (gatekeeper is not None and gatekeeper == (e, float(k))) else ''
            moon_badge = '🌙 ' if (('moon_pos' in locals()) and (moon_pos is not None) and moon_pos[0] == e and abs(moon_pos[1] - k) < 1e-9) else ''
            row.append(f'<td style="background:{bg}">{badge}{gk_badge}{moon_badge}{_format_compact(v)}</td>')
        row.append('</tr>')
        rows_html.append(''.join(row))
    rows_html.append('</tbody></table></div>')
    # Return marker strike (for potential header use) and normalized strength
    return ''.join(rows_html), (mark_pos[1] if mark_pos else None), float(strength)

