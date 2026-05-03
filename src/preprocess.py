from sklearn.preprocessing import MinMaxScaler
def select_features(df):
    features = [
        "Main steam flow (t/h)",
        "Main steam temperature (boiler side) (℃)",
        "Main steam pressure (boiler side) (Mpa)",
        "Feedwater flow (t/h)",
        "Feedwater temperature (℃)"
    ]

    df_selected = df[features].copy()
    return df_selected

def scale_features(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    df_scaled = df.copy()
    df_scaled[:] = scaled_data

    return df_scaled, scaler

def create_sequences(data, window_size=20):
    sequences = []

    for i in range(len(data) - window_size):
        seq = data[i:i + window_size]
        sequences.append(seq)

    return sequences