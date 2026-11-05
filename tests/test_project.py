import unittest

import pandas as pd

from src.credit_risk.prediction import predict
from src.credit_risk.preprocessing import clean_data
from src.credit_risk.training import (
    LogisticRegressionTrainer,
    RandomForestTrainer,
    train_model,
)


class TestCreditProject(unittest.TestCase):
    def test_clean_data_removes_missing_rows(self):
        """Le nettoyage supprime les lignes incomplètes."""
        data = pd.DataFrame(
            {
                "age": [25, None, 40],
                "revenu_mensuel": [4500, 3000, 7000],
            }
        )

        result = clean_data(data)

        self.assertEqual(len(result), 2)
        self.assertFalse(result.isna().any().any())
        self.assertEqual(len(data), 3)

    def test_both_trainers_can_predict(self):
        """Les deux entraîneurs respectent le même contrat."""
        features = pd.DataFrame(
            {
                "age": [25, 42, 31, 50, 23, 38],
                "revenu_mensuel": [4500, 8000, 3000, 12000, 2500, 6500],
                "montant_credit": [20000, 50000, 70000, 40000, 60000, 30000],
            }
        )
        target = pd.Series([0, 0, 1, 0, 1, 0])

        new_client = pd.DataFrame(
            {
                "age": [30],
                "revenu_mensuel": [4000],
                "montant_credit": [60000],
            }
        )

        trainers = [
            RandomForestTrainer(),
            LogisticRegressionTrainer(),
        ]

        for trainer in trainers:
            with self.subTest(trainer=type(trainer).__name__):
                model = train_model(trainer, features, target)
                predictions = predict(model, new_client)

                self.assertEqual(len(predictions), 1)
                self.assertIn(predictions[0], [0, 1])


if __name__ == "__main__":
    unittest.main()
