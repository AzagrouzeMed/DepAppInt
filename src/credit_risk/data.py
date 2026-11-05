import pandas as pd


def load_data(path):
    """Charge un fichier CSV et retourne un DataFrame."""
    return pd.read_csv(path)
