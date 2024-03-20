from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.utils
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from mnist import TinyCNN
import numpy as np
from PIL import Image
import io
import time
import base64
from concrete.ml.torch.compile import compile_torch_model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://fhe-playground.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

X, y = load_digits(return_X_y=True)
X = np.expand_dims(X.reshape((-1, 8, 8)), 1)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=True, random_state=42
)

model = TinyCNN(10)
model.load_state_dict(torch.load("mnist_state_dict.pt"))

q_model = compile_torch_model(model, x_train, rounding_threshold_bits=6, p_error=0.1)

@app.get("/img/")
async def get_img(id: int):
    X, y = img(id)
    X_img = img_to_base64(X[0])
    
    return { "id": id, "X": X[0].tolist(), "X_img": X_img,  "y": y.tolist() }

@app.post("/predict/")
async def predict(id: int):
    X, y = img(id)
    X = X.reshape(1, 1, 8, 8)
    start_time = time.time()

    q_model.fhe_circuit.keygen()
    keygen_time = time.time()
    X_q = q_model.quantize_input(X)
    X_q_enc = q_model.fhe_circuit.encrypt(X_q)
    enc_time = time.time()
    output_q_enc = q_model.fhe_circuit.run(X_q_enc)
    pred_time = time.time()
    output_q = q_model.fhe_circuit.decrypt(output_q_enc)
    dec_time = time.time()
    output = q_model.dequantize_output(output_q)
    output_proba = q_model.post_processing(output)
    output_class = output_proba.argmax(1)
    # output = q_model.forward(, fhe="execute").argmax(1)
    # output = model(input_tensor).argmax(1).detach().numpy()

    return {
        # "pub_key": q_model.fhe_circuit.keys,
        "X_q": X_q.tolist()[0][0],
        "X_q_img": img_to_base64(X_q[0][0]),
        "X_q_enc": base64.b64encode(X_q_enc.serialize()),
        "output_q_enc": base64.b64encode(output_q_enc.serialize()),
        "output": output_class.tolist()[0],
        "ans": y.tolist(),

        "keygen_time": keygen_time - start_time,
        "enc_time": enc_time - keygen_time,
        "pred_time": pred_time - enc_time,
        "dec_time": dec_time - pred_time,
    }

@app.post("/predict_raw/")
async def predict(id: int):
    X, y = img(id)
    input_tensor = torch.tensor(X, dtype=torch.float32).view(1,1,8,8)
    output = model(input_tensor).argmax(1).detach().numpy()

    return { "output": output.tolist()[0], "ans": y.tolist() }

def img(id: int):
    return (X[id], y[id])

def img_to_base64(img_array):
    img_array = np.round(img_array * (255 / 16)).astype(np.uint8)
    pil_image = Image.fromarray(img_array)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return encoded_image