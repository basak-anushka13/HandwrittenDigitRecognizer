import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

model = load_model("model.h5")

def predict_digit(image_dict):
    image = image_dict["composite"]
    image = Image.fromarray(image).convert("L")
    image = ImageOps.invert(image.resize((28, 28)))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    predictions = model.predict(img_array)[0]
    return {str(i): float(predictions[i]) for i in range(10)}

demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.ImageEditor(type="numpy", image_mode="L", label="Draw a digit", width=280, height=280),
    outputs=gr.Label(num_top_classes=3),
    title="🧠 Handwritten Digit Recognizer",
    description="Draw a digit (0–9) and let the AI guess!"
)

if __name__ == "__main__":
    demo.launch()
