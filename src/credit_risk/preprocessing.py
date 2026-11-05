def clean_data(data):
    """Supprime les lignes contenant des valeurs manquantes."""
    return data.dropna()
