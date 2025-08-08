import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DATABASE = "iot"
MONGO_COLLECTION = "sensor_data"
SEUIL_TEMPERATURE = 28.0  # seuil d'alerte, à adapter
DECALAGE = 3  # nombre de lignes à décaler (ex: 3x10min = 30min)

# --- CHARGEMENT DES DONNÉES ---
client = MongoClient(MONGO_URI)
db = client[MONGO_DATABASE]
collection = db[MONGO_COLLECTION]

data = pd.DataFrame(list(collection.find()))

if data.empty:
    print("Aucune donnée trouvée dans la base MongoDB.")
    exit(1)

# --- PRÉPARATION DES DONNÉES ---
# Conversion du timestamp
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values('timestamp')

# Création de la cible : température dans 30 minutes
# (Suppose 1 ligne = 10 min, sinon adapter DECALAGE)
data['temp_in_30min'] = data['temperature'].shift(-DECALAGE)
data = data.dropna(subset=['temp_in_30min'])

# Features : on enlève _id, timestamp, et la cible future
drop_cols = ['_id', 'timestamp', 'temp_in_30min']
X = data.drop(columns=[col for col in drop_cols if col in data.columns])
y = data['temp_in_30min']

# --- SÉPARATION TRAIN/TEST 70/30 (pas de shuffle) ---
n_total = len(X)
n_train = int(n_total * 0.7)

X_train = X.iloc[:n_train]
y_train = y.iloc[:n_train]
X_test = X.iloc[n_train:]
y_test = y.iloc[n_train:]

# --- ENTRAÎNEMENT DU MODÈLE ---
print("Entraînement du modèle Random Forest...")
model = RandomForestRegressor()
model.fit(X_train, y_train)
print("Entraînement terminé.")

# --- PRÉDICTION ET ÉVALUATION AVEC BARRE DE CHARGEMENT ---
print("Prédiction sur le test set...")
preds = []
for i in tqdm(range(len(X_test)), desc="Prédiction"):
    preds.append(model.predict([X_test.iloc[i]])[0])
preds = np.array(preds)

mse = mean_squared_error(y_test, preds)
print(f"MSE sur le test set : {mse:.2f}")

# --- TABLEAU DE COMPARAISON ---
comparaison = pd.DataFrame({
    'Température réelle': y_test.values,
    'Température prédite': preds,
    'Ecart (°C)': np.abs(y_test.values - preds)
})
print("\nTableau de comparaison (30% test set) :")
print(comparaison.to_string(index=False))

# Sauvegarde dans un fichier CSV
comparaison.to_csv("comparaison_predictions.csv", index=False)
print("\nTableau sauvegardé dans 'comparaison_predictions.csv'.")

# --- GRAPHIQUE DE COMPARAISON ---
plt.figure(figsize=(12,6))
plt.plot(comparaison['Température réelle'].values, label='Température réelle', marker='o')
plt.plot(comparaison['Température prédite'].values, label='Température prédite', marker='x')
plt.xlabel('Index (test set)')
plt.ylabel('Température (°C)')
plt.title('Comparaison Température réelle vs prédite')
plt.legend()
plt.tight_layout()
plt.savefig('comparaison_predictions.png')
plt.show()
print("Graphique sauvegardé dans 'comparaison_predictions.png'.")

# --- PRÉDICTION SUR LES DERNIÈRES DONNÉES ---
# On prend la dernière ligne connue pour prédire la température dans 30 min
last_row = X.iloc[[-1]]
pred_temp = model.predict(last_row)[0]
print(f"\nTempérature prévue dans 30 minutes : {pred_temp:.2f}°C")

if pred_temp > SEUIL_TEMPERATURE:
    print("⚠️  Alerte : Température élevée prévue dans 30 minutes !")
else:
    print("Température future dans la normale.")
