import requests
import cv2
import base64
import warnings

warnings.simplefilter("ignore")

from fastapi import FastAPI, Query

import trained_model
from configs.api_configs import get_image_from_url
from preprocessing import *

# ================================================================= #

app = FastAPI()

model = trained_model.load("Model/DeepOpacityNet.h5")

@app.post("/cataract-classification")
async def cataract_classification(image_url: str = Query(..., description=".")):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        mime_type = response.headers.get('Content-Type', 'image/png')
        image = get_image_from_url(response.content)
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve image from URL: {e}"}
    
    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_NEAREST)
    image = normalize_img(hist_equalize(image))
    image_4d = match_dims(image)

    pred = model.predict(image_4d, verbose=0).flatten().tolist()
    if pred[0] > pred[1]:
        label = "no cataract"
    else:
        label = "suspicious cataract"

    _, encoded_image = cv2.imencode(".png", image)
    base64_image = base64.b64encode(encoded_image).decode("utf-8")  

    results = {
        "image": f"data:{mime_type};base64,{base64_image}",
        "label": label,
        "confidence": f"{(pred[0] * 100):.2f}%" if pred[0] > pred[1] else f"{(pred[1] * 100):.2f}%",
    }

    return results