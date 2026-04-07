from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI()
model = tf.keras.models.load_model("flower_model.h5")

class_names = ['daisy','dandelion','rose','sunflower','tulip']


def decode_prediction(prediction: np.ndarray):
    scores = np.squeeze(prediction)
    scores = np.atleast_1d(scores).astype(float)

    # Binary output case: model returns a single probability/logit.
    if scores.size == 1:
        prob = float(np.clip(scores[0], 0.0, 1.0))
        idx = 1 if prob >= 0.5 else 0
        label = class_names[idx]
        confidence = prob if idx == 1 else (1.0 - prob)
        return label, confidence

    # Multi-class output case.
    idx = int(np.argmax(scores))
    idx = min(idx, len(class_names) - 1)
    label = class_names[idx]
    confidence = float(scores[idx])
    return label, confidence

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Flower Classifier</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          background: #f4f7fb;
          margin: 0;
          padding: 40px 16px;
          color: #1f2937;
        }
        .card {
          max-width: 700px;
          margin: 0 auto;
          background: #ffffff;
          border-radius: 12px;
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
          padding: 24px;
        }
        h1 { margin-top: 0; }
        .row { margin: 14px 0; }
        button {
          border: 0;
          border-radius: 8px;
          background: #2563eb;
          color: white;
          padding: 10px 16px;
          cursor: pointer;
        }
        button:disabled { background: #9ca3af; cursor: not-allowed; }
        #preview {
          max-width: 100%;
          max-height: 260px;
          display: none;
          border-radius: 8px;
          margin-top: 12px;
        }
        #result {
          margin-top: 16px;
          padding: 12px;
          background: #f3f4f6;
          border-radius: 8px;
          font-weight: 600;
          min-height: 22px;
        }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>Flower Classifier</h1>
        <p>Upload a flower image to predict its class.</p>
        <div class="row">
          <input id="fileInput" type="file" accept="image/*" />
        </div>
        <div class="row">
          <button id="predictBtn">Predict</button>
        </div>
        <img id="preview" alt="Selected preview" />
        <div id="result">No prediction yet.</div>
      </div>
      <script>
        const fileInput = document.getElementById("fileInput");
        const predictBtn = document.getElementById("predictBtn");
        const result = document.getElementById("result");
        const preview = document.getElementById("preview");

        fileInput.addEventListener("change", () => {
          const file = fileInput.files[0];
          if (!file) {
            preview.style.display = "none";
            return;
          }
          preview.src = URL.createObjectURL(file);
          preview.style.display = "block";
        });

        predictBtn.addEventListener("click", async () => {
          const file = fileInput.files[0];
          if (!file) {
            result.textContent = "Please select an image first.";
            return;
          }

          predictBtn.disabled = true;
          result.textContent = "Predicting...";

          try {
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch("/predict", {
              method: "POST",
              body: formData
            });

            if (!response.ok) {
              throw new Error("Request failed");
            }

            const data = await response.json();
            result.textContent = `Prediction: ${data.prediction} (confidence: ${(data.confidence * 100).toFixed(2)}%)`;
          } catch (err) {
            result.textContent = "Prediction failed. Please try another image.";
          } finally {
            predictBtn.disabled = false;
          }
        });
      </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB").resize((180,180))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    result, confidence = decode_prediction(prediction)

    return {"prediction": result, "confidence": confidence}