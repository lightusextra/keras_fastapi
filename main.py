from fastapi import FastAPI
from fastapi import File, UploadFile
from starlette.middleware.cors import CORSMiddleware

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
model_keras = "weight_2022-01-03-11_25_19.h5"

IMG_WIDTH, IMG_HEIGHT = 224, 224
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)

@app.post('/api/inference')
async def inference(file: UploadFile = File(...)):
    contents = await file.read()
    from io import BytesIO
    from PIL import Image
    im = Image.open(BytesIO(contents))
    im.save(file.filename)
    import os
    from keras.preprocessing import image as preprocessing
    img = preprocessing.load_img(file.filename, target_size=TARGET_SIZE)
    img = preprocessing.img_to_array(img)
    import numpy as np
    x = np.expand_dims(img, axis=0)
    from os.path import join, dirname, realpath
    #model_path = os.path.join(os.getcwd(), "model", model_keras)
    #from pathlib import Path
    #path = Path(model_path)
    from tensorflow.keras.models import load_model
    import os
    model_path = os.path.join(os.getcwd(), "model")
    path = os.path.join(model_path, model_keras)
    print(" model : ")
    print(os.path.exists(path))
    print(" model end ")
    model = load_model(path)
    predict = model.predict(x)
    for p in predict:
        class_index = p.argmax()
        probality = p.max()
        return {"result":"OK", "class_index":str(class_index), "probality":str(probality)}
