import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model.h5")

# Predict function
def predict_digit(image_dict):
    image = image_dict["composite"]
    image = Image.fromarray(image).convert("L")  # grayscale
    image = ImageOps.invert(image.resize((28, 28)))  # invert + resize
    img_array = np.array(image).astype("float32") / 255.0  # normalize
    img_array = img_array.reshape(1, 28, 28, 1)
    
    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[-3:][::-1]
    return {str(i): float(predictions[i]) for i in top_indices}

# Launch UI
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.ImageEditor(
        type="numpy",
        image_mode="L",
        label="🖌️Draw",
        width=280,
        height=280
    ),
    outputs=gr.Label(num_top_classes=3, label="🔍 Top 3 Predictions"),
    title="🧠 Handwritten Digit Recognizer",
    description="""
Draw a number(0-9) using your mouse or finger and let the AI guess it!
This app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize digits with over 98% accuracy.
""",
    theme=gr.themes.Glass(),  # You can also try Soft() or Default()
    allow_flagging="never",
    article="💡 *Tip: Draw clearly inside the box using your mouse or finger for best results.*"
)

if __name__ == "__main__":
    demo.launch()
