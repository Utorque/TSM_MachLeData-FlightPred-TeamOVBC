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

def load_data(upto_week: int, data_drift=False, concept_drift=False):
    '''Load data and simulate drift if necessary'''

    df = pd.read_csv('data/Flights.csv')

    df["price"] = df["price"].apply(parse_convert_price)
    df['time_taken_minutes'] = df['time_taken'].apply(parse_duration).astype(float)

    df = df[df['week'] <= upto_week].copy()

    # Simulate data drift
    if data_drift:
        mask = df['week'] == upto_week
        df.loc[mask, 'price'] *= 1.5
        df.loc[mask, 'time_taken_minutes'] *= 1.33
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