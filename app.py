import gradio as gr
import numpy as np
import cv2

from tensorflow.keras.models import load_model

model = load_model("diabetic_retinopathy_model.h5")

classes = {
    0:"No DR",
    1:"Mild DR",
    2:"Moderate DR",
    3:"Severe DR",
    4:"Proliferative DR"
}

def predict(image):

    image = cv2.resize(image,(224,224))

    image = image/255.0

    image = np.expand_dims(image,axis=0)

    pred = model.predict(image)

    class_id = np.argmax(pred)

    confidence = float(np.max(pred))

    return f"{classes[class_id]} | Confidence: {confidence*100:.2f}%"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs="text",
    title="Diabetic Retinopathy Detection"
)

demo.launch()