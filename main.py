from fastapi import FastAPI
from fastapi import File, UploadFile
from starlette.middleware.cors import CORSMiddleware
import os
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np

classes = ["dog", "cat"]
num_classes = len(classes)
image_size = 150

app = FastAPI()

# https://fastapi.tiangolo.com/tutorial/cors/
origins = [
    "https://pytorch-cpu.herokuapp.com",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get('/health')
def health():
    return {
        'message': 'ok'
    }

@app.post('/post')
def simple_post(param: str):
    return {
        'message': f'You posted `{param}`!'
    }

IMG_WIDTH, IMG_HEIGHT = 224, 224
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)

graph = tf.compat.v1.get_default_graph()

@app.post('/api/inference')
async def inference(file: UploadFile = File(...)):
    global graph
    with graph.as_default():
        model = load_model('./cats_dogs_model.h5')  # 学習済みモデルをロードする
        contents = await file.read()
        from io import BytesIO
        from PIL import Image
        im = Image.open(BytesIO(contents))
        im.save(file.filename)
        img = image.load_img(file.filename, target_size=TARGET_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        result = model.predict(x)
        print(result[0])
        if result[0] > 0.5:
            answer = "犬"
        else:
            answer = "猫"
        pred_answer = "これはもしや…" + answer + "では？"
        return {"result":"OK", "class_index":str(result[0]), "probality":str(pred_answer)}
