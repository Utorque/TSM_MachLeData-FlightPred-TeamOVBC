import pandas as pd

df_business = pd.read_csv("data/business.csv")
df_economy = pd.read_csv("data/economy.csv")

df_business["Class"] = "Business"
df_economy["Class"] = "Economy"

df = pd.concat([df_business, df_economy], ignore_index=True)

df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
df['dayofweek'] = df['date'].dt.day_name()
df['week'] = df['date'].dt.isocalendar().week

df.to_csv("data/Flights.csv", index=False)

print("Flights.csv généré avec succès.")
