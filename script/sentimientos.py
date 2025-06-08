
import tensorflow
from tensorflow import keras
from keras import layers

from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pickle

#Carga del modelo
modelo = load_model('modelos/modelo_sentimientos.keras')

# Cargar el vectorizador de texto 
with open('JupiterNotebook/vectorizer.pkl', 'rb') as file:
    vectorizador = pickle.load(file)

reversing_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear'}

def predecir_sentimiento(texto):

    texto_vect = vectorizador.transform([texto]).toarray()

    prediccion = modelo.predict(texto_vect)
    sentimiento = np.argmax(prediccion)

    return reversing_mapping[sentimiento]

print("¡Bienvenido al detector de sentimientos!")
print("Escribe una frase en inglés para predecir el sentimiento. Escribe 'salir' para terminar.")

while True:
    entrada = input("Escribe una frase: ")
    if entrada.lower() == 'salir':
        print("¡Gracias por usar el detector de sentimientos!")
        break
    
    # Predecir el sentimiento
    resultado = predecir_sentimiento(entrada)
    print(f"El sentimiento detectado es: {resultado}")




