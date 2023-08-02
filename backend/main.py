import sys

sys.path.insert(1, "backend")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import uuid
import uvicorn
from fastapi import FastAPI
import numpy as np
import asyncio

# from fastapi.exceptions import RequestValidationError
# from fastapi.responses import PlainTextResponse
import tensorflow as tf
import backend_functions as bf


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="backend/model.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test model on random input data.
input_shape = input_details[0]["shape"]


app = FastAPI()


# PING
@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


# predict
@app.post("/tweet")
async def predict_stm(tweet: str):
    input_data, cvocab, cnvocab = bf.preprocessing_transform(tweet)
    # todo : test non vocab words
    message = ""
    if cnvocab != 0:
        message = "Warning: %d words out of %d are unknown" % (
            cnvocab,
            cnvocab + cvocab,
        )
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    pred = interpreter.get_tensor(output_details[0]["index"])
    score = float(pred[0][0])
    return {"score": score, "message": message}


# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request, ex):
#     return PlainTextResponse(str(ex), status_code=400)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
