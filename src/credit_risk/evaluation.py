def evaluate_model(model, features, target):
    """Retourne la proportion de prédictions correctes."""
    return model.score(features, target)
