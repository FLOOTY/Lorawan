# -*- coding: utf-8 -*-
"""
Script de pipeline de données IoT :
- Reçoit les données de capteurs depuis The Things Network (TTN) via MQTT.
- Stocke les données brutes et le timestamp dans une base de données MongoDB locale.
- Transfère les données essentielles à ThingSpeak pour la visualisation en temps réel.

Objectif final : Collecter des données pour des applications de Machine Learning.
"""
import paho.mqtt.client as mqtt
import requests
import json
from pymongo import MongoClient
from datetime import datetime
import sys
import csv
import os

# --- CONFIGURATION ---
# Remplacez ces valeurs par les vôtres si nécessaire.

# Configuration MQTT pour The Things Network
MQTT_BROKER = "eu1.cloud.thethings.network"
MQTT_PORT = 1883
MQTT_USERNAME = "fiek-702@ttn"
MQTT_PASSWORD = "NNSXS.FOCVSH3WG7E6IRD57EN7EIP2ICZ6KA4L542SIOA.RL2IAE6WKA2HPMXCGFQVMNUCPO6PL2UCVYGRLPX3KMCPU7SFCJQA"
MQTT_TOPIC = "#"

# Configuration MongoDB (base de données locale)
MONGO_URI = "mongodb://mongo-db:27017/"
MONGO_DATABASE = "iot"
MONGO_COLLECTION = "sensor_data"

# Configuration ThingSpeak
THINGSPEAK_API_KEY = "OJC7LZ2EBZ9913VE"
THINGSPEAK_URL = "https://api.thingspeak.com/update"

CSV_FILE = "sensor_data.csv"
CSV_FIELDS = [
    "timestamp", "date", "heure", "air_quality", "humidity", "light_level", "pressure",
    "sound_level", "temperature"
]

# --- CONNEXION AUX SERVICES ---

# Connexion à la base de données MongoDB
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[MONGO_DATABASE]
    collection = db[MONGO_COLLECTION]
    print("Connexion à MongoDB réussie.")
except Exception as e:
    print(f"Erreur de connexion à MongoDB : {e}")
    sys.exit(1)

# --- FONCTIONS UTILITAIRES ---

def store_sensor_data(data_to_store):
    """
    Enregistre un document dans la collection MongoDB.
    Ajoute un timestamp au moment de l'enregistrement.
    """
    try:
        # Ajout d'un timestamp ISO 8601 pour une meilleure interopérabilité
        data_to_store['timestamp'] = datetime.utcnow().isoformat()
        result = collection.insert_one(data_to_store)
        print(f"Données insérées dans MongoDB avec l'ID : {result.inserted_id}")
    except Exception as e:
        print(f"Erreur lors de l'insertion dans MongoDB : {e}")

def store_sensor_data_csv(data_to_store):
    now = datetime.utcnow()
    data_to_store['timestamp'] = now.isoformat()
    data_to_store['date'] = now.strftime('%Y-%m-%d')
    data_to_store['heure'] = now.strftime('%H:%M:%S')
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: data_to_store.get(field, "") for field in CSV_FIELDS})
    print("Données insérées dans le CSV.")

def send_to_thingspeak(payload):
    """
    Envoie toutes les données du payload à ThingSpeak (jusqu'à 8 champs).
    """
    try:
        params = {'api_key': THINGSPEAK_API_KEY}
        # Mapping automatique des 8 premiers champs du payload
        for i, (key, value) in enumerate(payload.items()):
            if i >= 8:
                break
            params[f'field{i+1}'] = value
        print(f"Mapping ThingSpeak fields: {params}")
        if len(params) > 1:
            print(f"Envoi à ThingSpeak : {params}")
            r = requests.post(THINGSPEAK_URL, params=params)
            r.raise_for_status() # Lève une exception en cas d'erreur HTTP (4xx ou 5xx)
            print(f"Réponse de ThingSpeak : {r.status_code} - {r.text}")
        else:
            print("Aucune donnée pertinente à envoyer à ThingSpeak.")
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de l'envoi à ThingSpeak : {e}")

# --- LOGIQUE MQTT ---

def on_connect(client, userdata, flags, rc):
    """Callback pour la connexion au broker MQTT."""
    if rc == 0:
        print("Connecté au broker MQTT de TTN avec succès !")
        client.subscribe(MQTT_TOPIC)
        print(f"Abonné au topic : {MQTT_TOPIC}")
    else:
        print(f"Échec de la connexion MQTT, code de retour : {rc}")

def on_message(client, userdata, msg):
    """Callback pour la réception d'un message MQTT."""
    print(f"\nMessage reçu sur le topic : {msg.topic}")
    try:
        data = json.loads(msg.payload.decode())
        payload = data.get("uplink_message", {}).get("decoded_payload", {})

        if not payload:
            print("Message ignoré : ne contient pas de 'decoded_payload'.")
            return
            
        print(f"Payload décodé reçu : {payload}")
        
        # 1. Stocker les données dans MongoDB
        store_sensor_data(payload)
        # 2. Stocker les données dans le CSV
        store_sensor_data_csv(payload)
        # 3. Envoyer les données à ThingSpeak
        send_to_thingspeak(payload)

    except json.JSONDecodeError:
        print("Erreur : Le message reçu n'est pas un JSON valide.")
    except Exception as e:
        print(f"Une erreur inattendue est survenue dans on_message : {e}")


# --- POINT D'ENTRÉE DU SCRIPT ---
if __name__ == "__main__":
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        print("Connexion au broker MQTT...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        # loop_forever() est une boucle bloquante qui gère la reconnexion.
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nScript arrêté par l'utilisateur.")
        client.disconnect()
        mongo_client.close()
        sys.exit(0)
    except Exception as e:
        print(f"Erreur critique : Impossible de se connecter au broker MQTT. {e}")
        sys.exit(1) 