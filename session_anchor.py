
from __future__ import annotations
import pandas as pd, pytz

NY_TZ = pytz.timezone("America/New_York")

class RefSession:
    def __init__(self, session_date: pd.Timestamp, ref_open: pd.Timestamp, ref_close: pd.Timestamp):
        self.session_date = session_date
        self.ref_open = ref_open
        self.ref_close = ref_close

def now_ny():
    return pd.Timestamp.now(tz=NY_TZ)

def session_date() -> pd.Timestamp:
    """Return today's date in New York, tz-aware 00:00."""
    n = now_ny()
    return pd.Timestamp(year=n.year, month=n.month, day=n.day, tz=NY_TZ)

def ensure_ny(ts) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    return ts

def reference_session(anchor_until_open: bool=True) -> RefSession:
    n = now_ny()
    today = session_date()
    open_ts  = today + pd.Timedelta(hours=9, minutes=30)
    close_ts = today + pd.Timedelta(hours=16, minutes=0)
    # If anchoring and it's before 9:30 NY, freeze on yesterday's session
    if anchor_until_open and n < open_ts:
        y = today - pd.Timedelta(days=1)
        open_ts  = y + pd.Timedelta(hours=9, minutes=30)
        close_ts = y + pd.Timedelta(hours=16, minutes=0)
        return RefSession(y, open_ts, close_ts)
    return RefSession(today, open_ts, close_ts)
