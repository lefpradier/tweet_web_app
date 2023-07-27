import uuid
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
import mlflow
import asyncio
import spacy_fastlang
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse


# load model and predict
loaded_model = mlflow.pyfunc.load_model("backend/model")
app = FastAPI()


# PING
@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


# predict
@app.post("/tweet")
async def predict_stm(tweet: str):
    pred = loaded_model.predict(tweet)
    score = float(pred[0][0])
    print(score)
    return {"score": score}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, ex):
    return PlainTextResponse(str(ex), status_code=400)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
