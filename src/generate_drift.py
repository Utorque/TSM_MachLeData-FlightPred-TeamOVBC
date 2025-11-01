import pandas as pd
import numpy as np
import re

def parse_duration(duration_str):
    if pd.isna(duration_str):
        return np.nan
    
    match = re.match(r"(?:(\d+)h)?\s*(?:(\d+)m)?", str(duration_str))
    if not match:
        return np.nan

    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    return hours * 60 + minutes

def format_duration(minutes):
    if pd.isna(minutes):
        return np.nan
    minutes = int(round(minutes))
    h = minutes // 60
    m = minutes % 60
    return f"{h}h {m:02d}m"

data = pd.read_csv("data/Flights.csv")
data['price'] = data['price'].astype(str).str.replace(',', '.').pipe(pd.to_numeric, errors='coerce')
data_drifted = data.copy()

# Numerical drift : increase price
data_drifted.loc[data_drifted['week'] >= 8, 'price'] *= 1.3

# Categorical : overrepresentation of a company
mask = data_drifted['week'] >= 8
data_drifted.loc[mask, 'airline'] = np.where(
    np.random.rand(mask.sum()) < 0.7,
    'Emirates',
    data_drifted.loc[mask, 'airline']
)

# --- Temporal: increase in duration
data_drifted.loc[data_drifted['week'] >= 8, 'time_taken'] = (
    (data_drifted['time_taken'].apply(parse_duration) * 1.25).apply(format_duration)
)

data_drifted.to_csv("data/Flights_drifted.csv", index=False)