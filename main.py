from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.credit_risk.data import load_data
from src.credit_risk.evaluation import evaluate_model
from src.credit_risk.prediction import predict
from src.credit_risk.preprocessing import clean_data
from src.credit_risk.training import LogisticRegressionTrainer, train_model


def main():
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "raw" / "credit.csv"

    data = load_data(data_path)
    cleaned_data = clean_data(data)

    features = cleaned_data.drop(columns=["default"])
    target = cleaned_data["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.3,
        random_state=42,
        stratify=target,
    )

    print("Clients pour l'entraînement :", len(X_train))
    print("Clients pour le test :", len(X_test))

    trainer = LogisticRegressionTrainer()
    model = train_model(trainer, X_train, y_train)
    score = evaluate_model(model, X_test, y_test)

    print("\nModèle entraîné :", type(model).__name__)
    print(f"Exactitude sur le test : {score:.2%}")

    new_client = pd.DataFrame(
        [
            {
                "age": 30,
                "revenu_mensuel": 4000,
                "montant_credit": 60000,
            }
        ],
        columns=features.columns,
    )

    predictions = predict(model, new_client)
    prediction = predictions[0]

    print("\nNouveau client :")
    print(new_client)

    print("\nClasse prédite :", prediction)

    if prediction == 1:
        print("Résultat : défaut de paiement prédit.")
    else:
        print("Résultat : aucun défaut de paiement prédit.")


if __name__ == "__main__":
    main()
