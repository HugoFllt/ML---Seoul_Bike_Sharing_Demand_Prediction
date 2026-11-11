# Projet DAML : prédiction de la demande de vélos à Séoul

Prédire le nombre de vélos loués à chaque heure à partir de la météo et du calendrier, avec le dataset *Seoul Bike Sharing Demand* (UCI).

## Structure du projet

```text
ml-project/
├── SeoulBikeData.csv                  # Dataset brut d'origine
├── decoupage.py                       # Preprocessing + découpage train/test + plis de cross-validation (commun à tous)
├── introduction.ipynb                 # Sujet, problématique, exploration et preprocessing, protocoles, algorithmes, référence
├── decoupage_donnees.ipynb            # Détail des protocoles daily et temporal
├── regression_lineaire.ipynb          # Régression linéaire, Ridge, Lasso, polynomiale
├── modeles_arbres.ipynb               # Arbre, forêt aléatoire, gradient boosting, XGBoost
├── tuning_xgboost.ipynb               # Réglage des hyperparamètres de XGBoost
├── reseau_neurones.ipynb              # Réseau de neurones (MLP)
├── comparaison_finale.ipynb           # Comparaison de tous les modèles sur les jeux de test
└── figures/                           # Figures sauvegardées par les notebooks
```

Ordre de lecture : `introduction` → `decoupage_donnees` → notebooks de modèles → `comparaison_finale`.

## Mode d’emploi

### Installation

Le projet s’exécute dans Jupyter Notebook ou JupyterLab.

Dans un terminal ou Anaconda Prompt :

```bash
python -m pip install jupyterlab numpy pandas matplotlib
python -m pip install seaborn scikit-learn xgboost
```

Utiliser scikit-learn 1.4 ou ultérieur. LightGBM est facultatif :
le notebook d’arbres prévoit une solution de remplacement.

### Exécution

Télécharger et décompresser le projet, puis ouvrir un terminal
dans le dossier contenant `SeoulBikeData.csv` et `decoupage.py`.

Lancer Jupyter :

```bash
python -m jupyterlab
```

Ouvrir le notebook souhaité, puis utiliser
**Restart Kernel and Run All Cells** pour exécuter les cellules
dans l’ordre.

Les résultats enregistrés peuvent aussi être consultés sans
relancer les calculs. Les notebooks de modèles s’exécutent
indépendamment et utilisent le module commun `decoupage.py`.

Ordre de lecture conseillé :

1. `introduction.ipynb`
2. `decoupage_donnees.ipynb`
3. Les notebooks de modèles
4. `comparaison_finale.ipynb`

### Modifier les réglages

- Régression : `alpha` et degré polynomial
- Arbres : `max_depth` et `n_estimators`
- XGBoost : distributions et grilles de recherche
- Réseau : `hidden_layer_sizes` et `alpha`

Choisir les réglages avec la validation croisée, puis évaluer
la configuration retenue sur le test.

Pour actualiser la comparaison finale, reporter les paramètres
retenus dans `creer_modeles()` de `comparaison_finale.ipynb`.

Toute modification du prétraitement ou des découpages dans
`decoupage.py` nécessite de recalculer les résultats concernés.
La recherche des 100 configurations de XGBoost peut être longue.

Pour conserver les versions des bibliothèques utilisées :

```bash
python -m pip freeze > versions.txt
```
