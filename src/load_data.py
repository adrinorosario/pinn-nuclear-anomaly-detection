import pandas as pd

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = df.sort_values("Timestamp")
    df = df.set_index("Timestamp")

    return df