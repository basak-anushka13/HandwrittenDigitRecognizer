---
title: Handwritten Digit Recognizer
emoji: 🔢
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
---
# 🧠 Handwritten Digit Recognizer 🔢

This project uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits drawn by the user.

## 🛠️ Technologies Used
- TensorFlow/Keras
- Gradio (for the drawing UI)
- Python
- Hugging Face Spaces (for online deployment)

## 🚀 Live Demo
👉 Try the app here: https://huggingface.co/spaces/Basakanu/digit-recognizer

## 📷 How It Works

Users draw a digit (0–9) on the canvas. The app preprocesses the image, resizes, inverts colors, and feeds it into the trained CNN model.

## 🎯 Sample Predictions

| Drawn Digit | Top Prediction |
|-------------|----------------|
| 7           | 7 ✅           |
| 3           | 3 ✅           |
| 9 (badly)   | 4 ❌ (maybe try again) |

## 📁 Folder Structure
HandwrittenDigitRecognizer/
├── app.py
├── requirements.txt
├── model.h5
└── README.md

---

## 💡 To Run Locally

```bash
pip install -r requirements.txt
python app.py

Made with ❤️ by Anushka

