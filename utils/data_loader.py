import pandas as pd

def load_data(upto_week):
    df = pd.read_csv('data/Flights.csv')
    return df[df['week'] <= upto_week]
