# Prédiction de la condition d'une valve

Ce projet est une application de machine learning permettant de déterminer la **condition d'une valve** à partir de deux **séries temporelles** : un **flux** et une **pression**, enregistrés lors de séances de tests de 60 secondes.

---

## Structure du projet


```text
/codra-test-technique
├── test-individuals/                       # Échantillons du jeu de données à utiliser pour tester l'application web
├── valve-condition-server/                 # Backend - contient le modèle de deep learning et l'API
├── valve-condition-web-app/                # Frontend Angular - permet d'interroger le modèle
├── machine_learning_experimentations.ipynb # Notebook d'expérimentations ML et justification du choix de modèle
└── README.md
```


## Configuration

### 1. Backend (`valve-condition-server/app/config/settings.py`)

À modifier :
- `ALLOWED_ORIGINS` : liste des origines autorisées (ex. : `'http://localhost:4200'`)
- `API_HOST` : adresse IP ou nom d'hôte du serveur (ex. : `'0.0.0.0'`)
- `API_PORT` : port utilisé (ex. : `8000`)

### 2. Frontend (`valve-condition-web-app/src/app/config.ts`)

À modifier :
- `apiUrl` : URL de l’API backend (ex. : `http://localhost:8000`)

---

## Lancer l'application

### 1. Démarrer le serveur

```bash
cd valve-condition-server
pip install -r requirements.txt
cd app
python main.py
```

### 2. Démarrer l’application web

```bash
cd valve-condition-web-app
npm install
ng serve
```

---

## Test de l'application

Dans l'application web:
- entrer un fichier `pressure` (depuis le dossier `test-individuals`)
- entrer un fichier `flow`. Le nom des deux fichiers doit se terminer par le même identifiant (ex. : `FS1_90.txt` et `PS2_90.txt`)
- cliquer sur `Check Valve Condition`. Une condition prédite s'affiche, correspondant à l'étiquette du fichier.

Pour réentraîner le modele:
- cliquer sur `Train Model`
- Le serveur affiche les résultats d'entraînement en temps réel dans la console.
- Le message "Done" s'affiche à la fin de l'entraînement.