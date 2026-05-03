from src.load_data import load_and_prepare_data
from src.preprocess import select_features, scale_features, create_sequences

df = load_and_prepare_data("data/loop_dataset.csv")
df_selected = select_features(df)

df_scaled, scaler = scale_features(df_selected)

sequences = create_sequences(df_scaled.values, window_size=20)

print("Number of sequences:", len(sequences))
print("Shape of one sequence:", sequences[0].shape)