from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ModelTrainer(ABC):
    """Définit la méthode commune à tous les entraîneurs."""

    @abstractmethod
    def train(self, features, target):
        """Entraîne et retourne un modèle."""


class RandomForestTrainer(ModelTrainer):
    """Entraîne un modèle Random Forest."""

    def train(self, features, target):
        model = RandomForestClassifier(random_state=42)
        model.fit(features, target)
        return model


class LogisticRegressionTrainer(ModelTrainer):
    """Entraîne une régression logistique avec standardisation."""

    def train(self, features, target):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=42, max_iter=1000),
        )
        model.fit(features, target)
        return model


def train_model(trainer: ModelTrainer, features, target):
    """Délègue l'entraînement à l'entraîneur reçu."""
    return trainer.train(features, target)
