# Prédiction du risque de crédit

Projet pédagogique réalisé dans le cadre du module
Déploiement des Systèmes Intelligents.

## Objectif

Organiser un projet Python de Machine Learning et appliquer
les principes SRP, OCP et DIP.

## Données

Le fichier data/raw/credit.csv contient 10 clients fictifs.

Variables explicatives :
- age
- revenu_mensuel
- montant_credit

Cible : default
- 0 : aucun défaut de paiement
- 1 : défaut de paiement

## Organisation

- data/raw/ : données originales
- src/credit_risk/ : chargement, nettoyage, entraînement,
  évaluation et prédiction
- main.py : coordination des étapes
- notebooks/ : exploration
- models/ : modèles sauvegardés
- tests/ : tests automatiques du nettoyage et des deux entraîneurs
- docs/ : documentation complémentaire

## Installation sous Windows PowerShell

Depuis la racine du projet :

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

## Exécution

    .\.venv\Scripts\python.exe main.py

## Vérification de la qualité

    .\.venv\Scripts\python.exe -m ruff check .
    .\.venv\Scripts\python.exe -m black --check .

## Modèles disponibles

- Random Forest
- Régression logistique avec standardisation

Les entraîneurs implémentent le contrat ModelTrainer.

## Limites

Les données sont fictives et très petites.
Le score obtenu sur trois clients de test ne mesure pas
la fiabilité réelle du modèle pour une utilisation bancaire.

## Tests automatiques

Depuis la racine du projet :

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v