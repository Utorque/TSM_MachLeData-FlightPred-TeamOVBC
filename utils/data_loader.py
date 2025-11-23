import pandas as pd
import numpy as np
import re

def parse_convert_price(x):
    '''Parse price value to float and convert to CHF'''

    indian_ruppes_price = float(str(x).replace(",", "").strip())
    chf_price = indian_ruppes_price * 0.0091
    return chf_price

def parse_duration(duration_str):
    '''Parse duration to minutes'''

    if pd.isna(duration_str):
        return np.nan
    match = re.match(r"(?:(\d+)h)?\s*(?:(\d+)m)?", str(duration_str))
    if not match:
        return np.nan
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    return hours * 60 + minutes

# ---------- HELPER FUNCTIONS ----------
def parse_price(x):
    return float(str(x).replace(",", "").strip())

def hhmm_to_hour(series):
    t = pd.to_datetime(series, format="%H:%M", errors="coerce")
    return t.dt.hour

def parse_duration_minutes(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace(" ", "")
    m = re.match(r"(?:(\d+)h)?(?:(\d+)m)?", s)
    if not m:
        return np.nan
    h = int(m.group(1) or 0)
    m_ = int(m.group(2) or 0)
    return h * 60 + m_

def parse_stops(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if "non" in s:
        return 0
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan

def day_of_week_to_int(x):
    match x:
        case "Monday": return 0
        case "Tuesday": return 1
        case "Wednesday": return 2
        case "Thursday": return 3
        case "Friday": return 4
        case "Saturday": return 5
        case "Sunday": return 6
        case _: return np.nan

def load_data(upto_week: int, data_drift=False, concept_drift=False, path='data/Flights.csv'):
    '''Load data and simulate drift if necessary'''

    df = pd.read_csv(path)

    df["price"] = df["price"].apply(parse_convert_price)
    df['time_taken_minutes'] = df['time_taken'].apply(parse_duration).astype(float)
    df["dep_hour"] = hhmm_to_hour(df["dep_time"])
    df["arr_hour"] = hhmm_to_hour(df["arr_time"])
    df["stops_n"] = df["stop"].str[0].replace("n", "0").astype(int)
    df["dayofweek"] = df["dayofweek"].apply(day_of_week_to_int)

    df = df[df['week'] <= upto_week].copy()

    # Simulate data drift
    if data_drift:
        mask = df['week'] == upto_week
        df.loc[mask, 'price'] *= 1.5
        df.loc[mask, 'time_taken_minutes'] *= 1.5
        df.loc[mask, 'airline'] = np.where(
            np.random.rand(mask.sum()) < 0.7,
            'Emirates',
            df.loc[mask, 'airline']
        )

    # Simulate concept drift
    if concept_drift:
        mask = df["week"] == upto_week
        noise_scale = 5
        rng = np.random.default_rng(seed=42)

        # Shift distribution
        base = df.loc[mask, "price"].to_numpy(dtype=float)
        noise = rng.normal(loc=0.0, scale=noise_scale*base.std(), size=base.shape[0])
        df.loc[mask, "price"] = np.maximum(0.0, base + noise)

    # Training info log dictionnary
    training_dict = {
        "upto_week": upto_week,
        "data_drift": data_drift,
        "concept_drift": concept_drift,
        "num_samples": len(df),
        "num_features": df.shape[1]
    }

    return df, training_dict