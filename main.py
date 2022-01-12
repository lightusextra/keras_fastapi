from fastapi import FastAPI
from fastapi import File, UploadFile
from starlette.middleware.cors import CORSMiddleware
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

#ラベル数
n_class = 6
#モデル名
model_keras = './cats_dogs_model.h5'

IMG_WIDTH, IMG_HEIGHT = 224, 224
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)

import tensorflow as tf

graph = tf.compat.v1.get_default_graph()

@app.post('/api/inference')
async def inference(file: UploadFile = File(...)):
    global graph
    with graph.as_default():
        contents = await file.read()
        from io import BytesIO
        from PIL import Image
        im = Image.open(BytesIO(contents))
        im.save(file.filename)
        from keras.preprocessing import image as preprocessing
        img = preprocessing.load_img(file.filename, target_size=TARGET_SIZE)
        img = preprocessing.img_to_array(img)
        import numpy as np
        x = np.expand_dims(img, axis=0)
        from os.path import join, dirname, realpath
        #model_path = os.path.join(os.getcwd(), "model", model_keras)
        #from pathlib import Path
        #path = Path(model_path)
        del im
        del contents
        del file
        from tensorflow import keras
        keras.backend.clear_session()
        import gc
        gc.collect()
        #from tensorflow.keras.models import load_model
        #model = load_model(model_keras, compile=False)
        import tensorflow as tf
        model = load_model(model_keras, compile=False)
        predict = model.predict(x)
        print(predict[0])
        for p in predict:
            class_index = p.argmax()
            probality = p.max()
            return {"result":"OK", "class_index":str(class_index), "probality":str(probality)}
