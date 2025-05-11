# app.py
import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
MODEL_PATH = 'models/yolov8l.pt'  #update with your path
model = YOLO(MODEL_PATH)

def detect_objects(image):
    """Runs YOLO detection on the input image and returns the processed image."""
    image = image.convert('RGB')  # Ensure 3 channels
    image = np.array(image)
    results = model(image)
    
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            label = f"Mobile Phone: {conf:.2f}"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image


# Streamlit UI
st.title("Mobile Phone Detection")
st.write("Upload an image to detect mobile phones.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect Mobile Phones"):
        result_image = detect_objects(image)
        st.image(result_image, caption="Detection Result", use_column_width=True)

st.write("Developed with using YOLOv8 and Streamlit")
