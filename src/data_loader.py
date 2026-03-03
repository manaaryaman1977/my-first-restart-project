import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    X = df[["hours"]]
    y = df["score"]
    return X, y