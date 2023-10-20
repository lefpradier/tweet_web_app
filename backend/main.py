import sys

sys.path.insert(1, "backend")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import uuid
import uvicorn
from fastapi import FastAPI
import numpy as np
import asyncio

import tensorflow as tf
import backend_functions as bf


# Chargement du modèle et allocation des tenseurs
interpreter = tf.lite.Interpreter(model_path="backend/model.tflite")
interpreter.allocate_tensors()
# Récupération des tenseurs d'entrée et de sortie
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test du modèle sur des données aléatoires
input_shape = input_details[0]["shape"]


app = FastAPI()


# PING
@app.get("/")
def read_root():
    """
    Racine de l'API
    """
    return {"message": "Welcome from the API"}


# predict
@app.post("/tweet")
async def predict_stm(tweet: str):
    """
    Prédiction du sentiment d'un tweet
    """
    input_data, cvocab, cnvocab, language, l_score = bf.preprocessing_transform(tweet)
    # todo : test non vocab words
    message = ""
    if cnvocab != 0:
        message += "Warning: %d words out of %d are unknown. " % (
            cnvocab,
            cnvocab + cvocab,
        )
    if language != "en":
        message += (
            "Warning : Tweet does not appear to be in English. However, language detection can be unreliable for tweets below 3 words. Detected language: %s with a probability of %.3f"
            % (language.upper(), l_score)
        )

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    pred = interpreter.get_tensor(output_details[0]["index"])
    score = float(pred[0][0])
    return {"score": score, "message": message}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
