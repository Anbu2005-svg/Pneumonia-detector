import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("pneumonia_cnn_model.h5")

# Prediction function
def predict(img: Image.Image):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    label = "PNEUMONIA 🦠" if pred > 0.5 else "NORMAL ✅"
    return label

# Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Pneumonia Detection from Chest X-ray",
    description="Upload a chest X-ray image and detect if pneumonia is present."
)

interface.launch()
