import matplotlib.pyplot as plt

def plot_basic(df):
    plt.figure(figsize=(12, 6))

    plt.plot(df.index, df["Main steam flow (t/h)"], label="Steam Flow")
    plt.plot(df.index, df["Main steam temperature (boiler side) (℃)"], label="Steam Temp")
    plt.plot(df.index, df["Main steam pressure (boiler side) (Mpa)"], label="Steam Pressure")

    plt.legend()
    plt.title("Core Signals Over Time")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.tight_layout()
    plt.show()